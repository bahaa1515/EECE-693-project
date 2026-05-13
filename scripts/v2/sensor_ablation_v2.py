"""Phase 6: Sensor ablation on the Phase-4 tabular winners.

For each per-(T, L, W, algo) winner in ``tune_tabular_v2_best.csv``, refit
the model on each sensor subset and record val/test metrics.  No HPO here
— we lock in the hyperparameters from Phase 4 and only vary the input
modalities so deltas are attributable to the sensors.

Sensor subsets (default):

* ``all``                                       – baseline.
* singletons   : smartwatch, smartinhaler, peakflow, environment.
* leave-one-out: all\\smartwatch, all\\smartinhaler, all\\peakflow, all\\environment.

We also report test metrics here for completeness, but the headline
"final" numbers should still come from Phase 8 with the winner-of-winners.

Outputs
-------
``outputs/v2/tables/sensor_ablation_v2.csv``
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.event_features import build_daily_feature_tables, build_patient_static_features
from src.event_labels import DEFAULT_THRESHOLDS, run_event_labeling
from src.event_v2 import OUTPUT_TABLES_V2
from src.event_v2.features_v2 import ALL_SENSOR_SOURCES, build_training_feature_table
from src.event_v2.modeling_v2 import build_model, select_feature_columns
from src.event_v2.samples_v2 import build_all_sample_indexes
from src.event_v2.split_v2 import (
    PatientSplit,
    compute_metrics,
    make_patient_three_way_split,
    split_feature_table,
)


DEFAULT_SUBSETS = [
    ("all", list(ALL_SENSOR_SOURCES)),
    ("smartwatch_only", ["smartwatch"]),
    ("smartinhaler_only", ["smartinhaler"]),
    ("peakflow_only", ["peakflow"]),
    ("environment_only", ["environment"]),
    ("no_smartwatch", [s for s in ALL_SENSOR_SOURCES if s != "smartwatch"]),
    ("no_smartinhaler", [s for s in ALL_SENSOR_SOURCES if s != "smartinhaler"]),
    ("no_peakflow", [s for s in ALL_SENSOR_SOURCES if s != "peakflow"]),
    ("no_environment", [s for s in ALL_SENSOR_SOURCES if s != "environment"]),
]


def _parse_hp(winner_row: pd.Series, algo: str) -> dict:
    """Extract algo-specific hp dict from a winner row."""
    hp = {}
    for k in winner_row.index:
        if not k.startswith("param_"):
            continue
        v = winner_row[k]
        if isinstance(v, float) and np.isnan(v):
            continue
        hp[k[len("param_") :]] = v
    # cast some commonly-cast numeric fields
    for k in ("n_estimators", "max_depth", "min_samples_leaf", "num_layers"):
        if k in hp and pd.notna(hp[k]) and not isinstance(hp[k], bool):
            try:
                hp[k] = int(hp[k])
            except (TypeError, ValueError):
                pass
    # scale_pos_weight may be "auto" string
    if algo == "xgb" and "scale_pos_weight" in hp:
        spw = hp["scale_pos_weight"]
        if isinstance(spw, str) and spw != "auto":
            try:
                hp["scale_pos_weight"] = float(spw)
            except ValueError:
                pass
    return hp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--best",
        default=str(OUTPUT_TABLES_V2 / "tune_tabular_v2_best.csv"),
        help="Phase-4 winners table.",
    )
    parser.add_argument(
        "--split-json",
        default=str(OUTPUT_TABLES_V2 / "tune_tabular_v2_split.json"),
        help="Patient split JSON written by Phase 4 (re-used here).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    best_path = Path(args.best)
    if not best_path.exists():
        raise SystemExit(
            f"Winner table not found: {best_path}\n"
            "Run scripts/v2/tune_tabular_v2.py first."
        )
    winners = pd.read_csv(best_path)
    if winners.empty:
        raise SystemExit("Winner table is empty.")

    split_path = Path(args.split_json)
    if split_path.exists():
        meta = json.loads(split_path.read_text())
        split = PatientSplit(
            train_users=tuple(meta["train_users"]),
            val_users=tuple(meta["val_users"]),
            test_users=tuple(meta["test_users"]),
            seed=int(meta["seed"]),
            val_frac=float(meta["val_frac"]),
            test_frac=float(meta["test_frac"]),
        )
        print(f"  reusing split from {split_path}")
    else:
        split = None
        print("  no split JSON found; will create from seed.")

    # Get unique (T, L, W) combos to build samples / features.
    unique_keys = sorted(
        {
            (int(r.threshold), int(r.input_length_days), int(r.washout_days))
            for r in winners.itertuples(index=False)
        }
    )
    thresholds = sorted({k[0] for k in unique_keys})
    lengths = sorted({k[1] for k in unique_keys})
    washouts = sorted({k[2] for k in unique_keys})

    print(f"  ablating {len(winners)} winners over {len(DEFAULT_SUBSETS)} subsets")
    print(f"  (T, L, W) keys: {unique_keys}")

    print("\n[1/3] Building labels + samples + daily tables...")
    label_artifacts = run_event_labeling(thresholds=thresholds)
    sample_indexes, _counts = build_all_sample_indexes(
        weekly_events=label_artifacts.weekly_events,
        probable_days_by_threshold=label_artifacts.probable_event_days,
        episodes_by_threshold=label_artifacts.event_episodes,
        input_lengths=lengths,
        washout_values=washouts,
    )
    sample_indexes = {k: v for k, v in sample_indexes.items() if k in unique_keys}
    daily_tables = build_daily_feature_tables()
    patient_features = build_patient_static_features()

    if split is None:
        any_idx = next(v for v in sample_indexes.values() if not v.empty)
        split = make_patient_three_way_split(
            groups=any_idx["user_key"], seed=args.seed
        )

    print("\n[2/3] Pre-building feature tables per (T, L, W, subset)...")
    # Cache feature tables per (T, L, W, subset_name).
    feature_cache: dict[tuple[int, int, int, str], pd.DataFrame] = {}
    for key, sample_index in sample_indexes.items():
        for subset_name, subset in DEFAULT_SUBSETS:
            feat = build_training_feature_table(
                sample_index=sample_index,
                daily_tables=daily_tables,
                patient_features=patient_features,
                sensor_sources=subset,
            )
            feature_cache[(*key, subset_name)] = feat

    print("\n[3/3] Refitting winners on each subset...")
    rows = []
    for winner in winners.itertuples(index=False):
        T = int(winner.threshold)
        L = int(winner.input_length_days)
        W = int(winner.washout_days)
        algo = str(winner.algo)
        hp = _parse_hp(pd.Series(winner._asdict()), algo)

        for subset_name, _subset in DEFAULT_SUBSETS:
            feat = feature_cache[(T, L, W, subset_name)]
            if feat.empty or feat["target"].nunique() < 2:
                continue
            train, val, test = split_feature_table(feat, split)
            if train.empty or val.empty or train["target"].nunique() < 2:
                continue
            feature_cols = select_feature_columns(feat)
            X_train, y_train = train[feature_cols], train["target"].astype(int)
            X_val, y_val = val[feature_cols], val["target"].astype(int)
            X_test, y_test = test[feature_cols], test["target"].astype(int)

            n_pos = int(y_train.sum())
            n_neg = int(len(y_train) - n_pos)
            try:
                model = build_model(
                    algo, hp, n_pos=n_pos, n_neg=n_neg, random_state=args.seed
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)
                    p_val = model.predict_proba(X_val)[:, 1]
                    p_test = model.predict_proba(X_test)[:, 1]
            except Exception as exc:
                print(
                    f"  !! refit failed T={T} L={L} W={W} {algo} subset={subset_name}: {exc!r}"
                )
                continue

            val_metrics = compute_metrics(y_val.to_numpy(), p_val)
            test_metrics = compute_metrics(y_test.to_numpy(), p_test)
            row = {
                "threshold": T,
                "input_length_days": L,
                "washout_days": W,
                "algo": algo,
                "subset": subset_name,
                "n_features": len(feature_cols),
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "train_positive": n_pos,
                "val_positive": int(y_val.sum()),
                "test_positive": int(y_test.sum()),
            }
            for k, v in val_metrics.items():
                row[f"val_{k}"] = v
            for k, v in test_metrics.items():
                row[f"test_{k}"] = v
            rows.append(row)

    if not rows:
        print("No ablation rows produced.")
        return 1
    df = pd.DataFrame(rows)
    out_path = OUTPUT_TABLES_V2 / "sensor_ablation_v2.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} ablation rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
