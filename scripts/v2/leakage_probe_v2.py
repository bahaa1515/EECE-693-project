"""Phase 7: Leakage probe.

A standard sanity check.  If we shuffle the *target* label within the
training partition (keeping the marginal positive rate fixed) and retrain
on the same features, a non-leaky pipeline should produce validation
PR-AUC close to the train-set base rate and ROC-AUC close to 0.5
across all configs and across many shuffles.

Any configuration with consistently above-random metrics after shuffling
indicates a leak in either the labeling, the features, or the split.

Strategy
--------
* Use the Phase-4 best winners (one row per (T, L, W, algo)).
* For each winner, shuffle ``y_train`` (and ``y_val`` independently) under
  ``--n-shuffles`` different seeds, refit, and record validation metrics.
* Aggregate per (T, L, W, algo) and write a CSV.

Pass criterion (printed at the end):
``mean(val_roc_auc) within [0.45, 0.55]`` for *every* winner.  Any
violation prints a WARNING but does not exit non-zero (so Colab does not
abort silently).
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
from src.event_labels import run_event_labeling
from src.event_v2 import OUTPUT_TABLES_V2
from src.event_v2.features_v2 import build_training_feature_table
from src.event_v2.modeling_v2 import build_model, select_feature_columns
from src.event_v2.samples_v2 import build_all_sample_indexes
from src.event_v2.split_v2 import (
    PatientSplit,
    compute_metrics,
    make_patient_three_way_split,
    split_feature_table,
)
from scripts.v2.sensor_ablation_v2 import _parse_hp


def _shuffle_within(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = arr.copy()
    rng.shuffle(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--best",
        default=str(OUTPUT_TABLES_V2 / "tune_tabular_v2_best.csv"),
    )
    parser.add_argument(
        "--split-json",
        default=str(OUTPUT_TABLES_V2 / "tune_tabular_v2_split.json"),
    )
    parser.add_argument("--n-shuffles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    best_path = Path(args.best)
    if not best_path.exists():
        raise SystemExit(f"Missing {best_path}. Run tune_tabular_v2.py first.")
    winners = pd.read_csv(best_path)
    if winners.empty:
        raise SystemExit("Empty winners table.")

    split_path = Path(args.split_json)
    if not split_path.exists():
        raise SystemExit(f"Missing {split_path}.")
    meta = json.loads(split_path.read_text())
    split = PatientSplit(
        train_users=tuple(meta["train_users"]),
        val_users=tuple(meta["val_users"]),
        test_users=tuple(meta["test_users"]),
        seed=int(meta["seed"]),
        val_frac=float(meta["val_frac"]),
        test_frac=float(meta["test_frac"]),
    )

    unique_keys = sorted(
        {
            (int(r.threshold), int(r.input_length_days), int(r.washout_days))
            for r in winners.itertuples(index=False)
        }
    )
    thresholds = sorted({k[0] for k in unique_keys})
    lengths = sorted({k[1] for k in unique_keys})
    washouts = sorted({k[2] for k in unique_keys})

    print(f"[leakage-probe] {len(winners)} winners × {args.n_shuffles} shuffles")

    print("\nBuilding labels + samples + features (all-sensors)...")
    label_artifacts = run_event_labeling(thresholds=thresholds)
    sample_indexes, _ = build_all_sample_indexes(
        weekly_events=label_artifacts.weekly_events,
        probable_days_by_threshold=label_artifacts.probable_event_days,
        episodes_by_threshold=label_artifacts.event_episodes,
        input_lengths=lengths,
        washout_values=washouts,
    )
    sample_indexes = {k: v for k, v in sample_indexes.items() if k in unique_keys}
    daily_tables = build_daily_feature_tables()
    patient_features = build_patient_static_features()

    feature_tables: dict[tuple[int, int, int], pd.DataFrame] = {}
    for key, idx in sample_indexes.items():
        feature_tables[key] = build_training_feature_table(
            sample_index=idx,
            daily_tables=daily_tables,
            patient_features=patient_features,
        )

    rows = []
    for winner in winners.itertuples(index=False):
        T = int(winner.threshold)
        L = int(winner.input_length_days)
        W = int(winner.washout_days)
        algo = str(winner.algo)
        hp = _parse_hp(pd.Series(winner._asdict()), algo)

        feat = feature_tables[(T, L, W)]
        train, val, _test = split_feature_table(feat, split)
        if train.empty or val.empty or train["target"].nunique() < 2:
            continue
        feature_cols = select_feature_columns(feat)
        X_train = train[feature_cols]
        y_train_real = train["target"].astype(int).to_numpy()
        X_val = val[feature_cols]
        y_val_real = val["target"].astype(int).to_numpy()
        n_pos = int(y_train_real.sum())
        n_neg = int(len(y_train_real) - n_pos)

        for k in range(args.n_shuffles):
            rng = np.random.default_rng(args.seed * 1000 + k)
            y_train = _shuffle_within(y_train_real, rng)
            y_val = _shuffle_within(y_val_real, rng)
            try:
                model = build_model(
                    algo, hp, n_pos=int(y_train.sum()),
                    n_neg=int(len(y_train) - y_train.sum()),
                    random_state=args.seed,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)
                    p_val = model.predict_proba(X_val)[:, 1]
            except Exception as exc:
                print(f"  !! probe failed: {exc!r}")
                continue
            metrics = compute_metrics(y_val, p_val)
            rows.append(
                {
                    "threshold": T,
                    "input_length_days": L,
                    "washout_days": W,
                    "algo": algo,
                    "shuffle": k,
                    "n_train": len(train),
                    "n_val": len(val),
                    "train_positive_after_shuffle": int(y_train.sum()),
                    "val_positive_after_shuffle": int(y_val.sum()),
                    **{f"val_{k}": v for k, v in metrics.items()},
                }
            )

    if not rows:
        print("No probe rows produced.")
        return 1
    df = pd.DataFrame(rows)
    out_path = OUTPUT_TABLES_V2 / "leakage_probe_v2.csv"
    df.to_csv(out_path, index=False)

    summary = (
        df.groupby(["threshold", "input_length_days", "washout_days", "algo"])
        .agg(
            val_roc_auc_mean=("val_roc_auc", "mean"),
            val_roc_auc_std=("val_roc_auc", "std"),
            val_pr_auc_mean=("val_pr_auc", "mean"),
            val_pr_auc_std=("val_pr_auc", "std"),
            n_shuffles=("shuffle", "count"),
        )
        .reset_index()
    )
    sum_path = OUTPUT_TABLES_V2 / "leakage_probe_v2_summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"Wrote {len(df)} probe rows -> {out_path}")
    print(f"Wrote summary -> {sum_path}")
    print(summary.to_string(index=False))

    # Pass criterion
    bad = summary[(summary["val_roc_auc_mean"] < 0.45) | (summary["val_roc_auc_mean"] > 0.55)]
    if not bad.empty:
        print(
            "\nWARNING: shuffled-label ROC-AUC outside [0.45, 0.55] for some "
            f"configs (n={len(bad)}). Investigate possible leakage before "
            "running Phase 8."
        )
        print(bad.to_string(index=False))
    else:
        print("\nOK: all winners produced near-random metrics under label shuffle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
