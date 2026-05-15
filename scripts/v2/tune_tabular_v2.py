"""Phase 4: Tabular HPO on (T, L, W) × {LR, RF, XGB} × hp grid.

For each (threshold T, input length L, washout W) and each algorithm, we:

* Train on the train split.
* Score on the validation split with PR-AUC + companion metrics.
* Record one row per (T, L, W, algo, hp_combo).

We do *not* touch the test split here.  Phase 8 will lock in the winners
from this script (selected via :func:`select_best_by_pr_auc`).

Outputs
-------
``outputs/v2/tables/tune_tabular_v2_trials.csv``   – full trial table.
``outputs/v2/tables/tune_tabular_v2_best.csv``     – winner per (T, L, W, algo).
``outputs/v2/tables/tune_tabular_v2_split.json``   – patient split used.

Usage
-----
::

    python -m scripts.v2.tune_tabular_v2 \
        --thresholds 2,3,4 --lengths 3,7,14 --washouts 0,7,14 \
        --seed 42
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.event_labels import DEFAULT_THRESHOLDS, run_event_labeling
from src.event_v2 import OUTPUT_TABLES_V2, DATA_PROCESSED_V2
from src.event_v2.features_v2 import (
    build_and_write_feature_tables_v2,
    feature_table_filename_v2,
)
from src.event_v2.modeling_v2 import build_model, select_feature_columns
from src.event_v2.samples_v2 import (
    build_all_sample_indexes,
    sample_index_filename,
    write_sample_indexes,
)
from src.event_v2.split_v2 import (
    METRIC_COLUMNS,
    add_selection_diagnostics,
    compute_metrics,
    make_patient_three_way_split,
    precision_at_recall_floor,
    prefix_metrics,
    recall_at_precision_floor,
    select_best_by_pr_auc,
    split_feature_table,
    tune_threshold_max_f1,
)


# --------------------------------------------------------------------------- #
# Hyperparameter grids
# --------------------------------------------------------------------------- #
LR_GRID: list[dict[str, Any]] = [
    {"C": C, "class_weight": cw}
    for C in (0.01, 0.1, 1.0, 10.0)
    for cw in (None, "balanced")
]
RF_GRID: list[dict[str, Any]] = [
    {
        "n_estimators": n,
        "max_depth": d,
        "min_samples_leaf": leaf,
        "class_weight": cw,
    }
    for n in (200, 500)
    for d in (None, 6, 10)
    for leaf in (1, 5)
    for cw in (None, "balanced")
]
XGB_GRID: list[dict[str, Any]] = [
    {
        "n_estimators": n,
        "max_depth": d,
        "learning_rate": lr,
        "subsample": sub,
        "scale_pos_weight": spw,
    }
    for n in (200, 500)
    for d in (3, 6)
    for lr in (0.05, 0.1)
    for sub in (0.7, 1.0)
    for spw in (1.0, "auto")
]


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _algo_grid(algo: str) -> list[dict[str, Any]]:
    return {"lr": LR_GRID, "rf": RF_GRID, "xgb": XGB_GRID}[algo]


def _run_trial(
    algo: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> dict[str, Any]:
    X_train = train[feature_cols]
    y_train = train["target"].astype(int)
    X_val = val[feature_cols]
    y_val = val["target"].astype(int)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)

    model = build_model(algo, params, n_pos=n_pos, n_neg=n_neg, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            train_score = model.predict_proba(X_train)[:, 1]
            y_score = model.predict_proba(X_val)[:, 1]
        else:
            train_score = model.decision_function(X_train)
            y_score = model.decision_function(X_val)

    train_metrics = compute_metrics(y_train.to_numpy(), train_score)
    val_metrics = compute_metrics(y_val.to_numpy(), y_score)
    tuned_threshold, val_tuned = tune_threshold_max_f1(y_val.to_numpy(), y_score)
    train_tuned = compute_metrics(
        y_train.to_numpy(),
        train_score,
        threshold=tuned_threshold,
    )
    val_recall_pf, val_recall_pf_threshold = recall_at_precision_floor(
        y_val.to_numpy(), y_score
    )
    val_precision_rf, val_precision_rf_threshold = precision_at_recall_floor(
        y_val.to_numpy(), y_score
    )

    row: dict[str, Any] = {
        **prefix_metrics(train_metrics, "train_"),
        **prefix_metrics(val_metrics, "val_"),
        "tuned_threshold": tuned_threshold,
        **prefix_metrics(train_tuned, "train_tuned_"),
        **prefix_metrics(val_tuned, "val_tuned_"),
        "val_recall_at_precision_floor": val_recall_pf,
        "val_recall_at_precision_floor_threshold": val_recall_pf_threshold,
        "val_precision_at_recall_floor": val_precision_rf,
        "val_precision_at_recall_floor_threshold": val_precision_rf_threshold,
    }
    return add_selection_diagnostics(row)


def _stringify_params(params: dict[str, Any]) -> str:
    """Stable string repr used as a key in the trial table."""
    items = sorted(params.items())
    return ";".join(f"{k}={v}" for k, v in items)


def _requested_keys(
    thresholds: Iterable[int],
    lengths: Iterable[int],
    washouts: Iterable[int],
) -> list[tuple[int, int, int]]:
    return [
        (int(threshold), int(length), int(washout))
        for threshold, length, washout in itertools.product(
            thresholds,
            lengths,
            washouts,
        )
    ]


def _load_existing_sample_indexes(
    keys: Iterable[tuple[int, int, int]],
) -> dict[tuple[int, int, int], pd.DataFrame]:
    """Load Phase-3 sample indexes without rewriting intermediate outputs."""
    sample_indexes: dict[tuple[int, int, int], pd.DataFrame] = {}
    missing: list[Path] = []
    for threshold, length, washout in keys:
        path = DATA_PROCESSED_V2 / sample_index_filename(threshold, length, washout)
        if not path.exists():
            missing.append(path)
            continue
        sample_indexes[(threshold, length, washout)] = pd.read_parquet(path)
    if missing:
        shown = "\n".join(f"  - {path}" for path in missing[:12])
        extra = "" if len(missing) <= 12 else f"\n  ... and {len(missing) - 12} more"
        raise SystemExit(
            "Missing Phase-3 sample-index parquet(s), and --reuse-features was "
            f"requested so they will not be rebuilt:\n{shown}{extra}"
        )
    return sample_indexes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS)
    )
    parser.add_argument("--lengths", default="3,7,14")
    parser.add_argument("--washouts", default="0,7,14")
    parser.add_argument(
        "--algos",
        default="lr,rf,xgb",
        help="Comma-separated subset of {lr, rf, xgb}.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument(
        "--sensor-tag",
        default="all",
        help="Sensor source tag passed to feature builder (currently 'all' supported).",
    )
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Skip rebuilding sample indexes & features if parquets already exist.",
    )
    args = parser.parse_args()

    thresholds = _parse_int_list(args.thresholds)
    lengths = _parse_int_list(args.lengths)
    washouts = _parse_int_list(args.washouts)
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    for algo in algos:
        if algo not in {"lr", "rf", "xgb"}:
            raise SystemExit(f"Unknown algo: {algo!r}")

    print(f"[v2-tune-tabular] outputs -> {OUTPUT_TABLES_V2}")
    print(f"  thresholds={thresholds}  lengths={lengths}  washouts={washouts}")
    print(f"  algos={algos}  seed={args.seed}  sensor_tag={args.sensor_tag}")

    # ------------------------------------------------------------------ #
    # 1. Labels (deterministic; safe to re-run).
    # ------------------------------------------------------------------ #
    if args.reuse_features:
        print("\n[1/4] Skipping event-label rebuild (--reuse-features).")
    else:
        print("\n[1/4] Building event labels...")
        label_artifacts = run_event_labeling(thresholds=thresholds)

    # ------------------------------------------------------------------ #
    # 2. Sample indexes (v2 contract-verified).
    # ------------------------------------------------------------------ #
    if args.reuse_features:
        print("\n[2/4] Loading existing v2 sample indexes (--reuse-features)...")
        sample_indexes = _load_existing_sample_indexes(
            _requested_keys(thresholds, lengths, washouts)
        )
    else:
        print("\n[2/4] Building v2 sample indexes...")
        sample_indexes, sample_counts = build_all_sample_indexes(
            weekly_events=label_artifacts.weekly_events,
            probable_days_by_threshold=label_artifacts.probable_event_days,
            episodes_by_threshold=label_artifacts.event_episodes,
            input_lengths=lengths,
            washout_values=washouts,
        )
        # Keep only requested thresholds:
        sample_indexes = {
            k: v for k, v in sample_indexes.items() if k[0] in thresholds
        }
        write_sample_indexes(sample_indexes, sample_counts)

    # ------------------------------------------------------------------ #
    # 3. Feature tables.
    # ------------------------------------------------------------------ #
    print("\n[3/4] Building v2 feature tables...")
    feature_paths: dict[tuple[int, int, int], Path] = {}
    if args.reuse_features:
        for key in sample_indexes:
            T, L, W = key
            path = DATA_PROCESSED_V2 / feature_table_filename_v2(
                T, L, W, sensor_tag=args.sensor_tag
            )
            if path.exists():
                feature_paths[key] = path
        missing = [k for k in sample_indexes if k not in feature_paths]
        if missing:
            shown = "\n".join(
                "  - "
                + str(
                    DATA_PROCESSED_V2
                    / feature_table_filename_v2(
                        T,
                        L,
                        W,
                        sensor_tag=args.sensor_tag,
                    )
                )
                for T, L, W in missing[:12]
            )
            extra = "" if len(missing) <= 12 else f"\n  ... and {len(missing) - 12} more"
            raise SystemExit(
                "Missing Phase-3 feature parquet(s), and --reuse-features was "
                f"requested so they will not be rebuilt:\n{shown}{extra}"
            )
    else:
        _, feature_paths = build_and_write_feature_tables_v2(
            sample_indexes=sample_indexes,
            sensor_sources=None if args.sensor_tag == "all" else args.sensor_tag.split("+"),
        )

    # ------------------------------------------------------------------ #
    # 4. HPO loop.
    # ------------------------------------------------------------------ #
    print("\n[4/4] Running HPO loop...")
    # Determine groups from one (any) feature table for split decision.
    first_key = sorted(feature_paths.keys())[0]
    first_table = pd.read_parquet(feature_paths[first_key])
    split = make_patient_three_way_split(
        groups=first_table["user_key"],
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print(
        f"  split (seed={args.seed}): "
        f"train_users={split.train_users} | "
        f"val_users={split.val_users} | "
        f"test_users={split.test_users}"
    )
    (OUTPUT_TABLES_V2 / "tune_tabular_v2_split.json").write_text(
        json.dumps(split.to_dict(), indent=2)
    )

    trial_rows: list[dict[str, Any]] = []
    for key in sorted(feature_paths):
        T, L, W = key
        feature_table = pd.read_parquet(feature_paths[key])
        if feature_table["target"].nunique() < 2:
            print(f"  skip T={T} L={L} W={W}: only one class.")
            continue
        train, val, test = split_feature_table(feature_table, split)
        if train.empty or val.empty or train["target"].nunique() < 2:
            print(f"  skip T={T} L={L} W={W}: empty/degenerate split.")
            continue
        feature_cols = select_feature_columns(feature_table)
        base_row = {
            "sensor_tag": args.sensor_tag,
            "threshold": T,
            "input_length_days": L,
            "washout_days": W,
            "n_features": len(feature_cols),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "train_users": train["user_key"].nunique(),
            "val_users": val["user_key"].nunique(),
            "test_users": test["user_key"].nunique(),
            "train_positive": int(train["target"].sum()),
            "val_positive": int(val["target"].sum()),
            "test_positive": int(test["target"].sum()),
            "seed": args.seed,
        }
        if base_row["val_positive"] == 0:
            print(
                f"  skip T={T} L={L} W={W}: validation has no positives "
                f"-> PR-AUC undefined for this split."
            )
            continue
        for algo in algos:
            grid = _algo_grid(algo)
            print(
                f"  T={T} L={L} W={W} algo={algo}: "
                f"{len(grid)} trials  (train={len(train)} val={len(val)})"
            )
            for params in grid:
                try:
                    metrics = _run_trial(
                        algo=algo,
                        params=params,
                        train=train,
                        val=val,
                        feature_cols=feature_cols,
                        seed=args.seed,
                    )
                except Exception as exc:  # pragma: no cover - HPO robustness
                    print(f"    !! trial failed ({algo} {params}): {exc!r}")
                    metrics = {f"val_{k}": float("nan") for k in METRIC_COLUMNS}
                row = {
                    **base_row,
                    "algo": algo,
                    "params": _stringify_params(params),
                    **{f"param_{k}": v for k, v in params.items()},
                    **metrics,
                }
                row = add_selection_diagnostics(row)
                trial_rows.append(row)

    if not trial_rows:
        print("No trials produced. Exiting.")
        return 1

    trials = pd.DataFrame(trial_rows)
    trials_path = OUTPUT_TABLES_V2 / "tune_tabular_v2_trials.csv"
    trials.to_csv(trials_path, index=False)
    print(f"\nWrote {len(trials)} trials -> {trials_path}")

    best = select_best_by_pr_auc(
        trials,
        score_col="val_pr_auc",
        group_cols=("sensor_tag", "threshold", "input_length_days", "washout_days", "algo"),
    )
    best_path = OUTPUT_TABLES_V2 / "tune_tabular_v2_best.csv"
    best.to_csv(best_path, index=False)
    print(f"Wrote {len(best)} winners -> {best_path}")

    overall = select_best_by_pr_auc(trials, score_col="val_pr_auc")
    if not overall.empty:
        row = overall.iloc[0]
        print(
            "\nBest overall (val): "
            f"algo={row['algo']}  T={row['threshold']}  L={row['input_length_days']}  "
            f"W={row['washout_days']}  PR-AUC={row['val_pr_auc']:.3f}  "
            f"ROC-AUC={row['val_roc_auc']:.3f}  F1={row['val_f1']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
