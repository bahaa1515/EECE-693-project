"""Phase 5: Deep-learning HPO + multi-seed for the v2 event-episode task.

We run a coarse HPO grid over GRU/LSTM/RNN/CNN architectures, for each
(T, L, W) in the requested grid, fitted with a single seed (default 42).
Then for the per-(T, L, W, arch) PR-AUC winner we re-fit across 5 seeds
to estimate stability.

Outputs
-------
``outputs/v2/tables/tune_dl_v2_trials.csv``        – single-seed HPO table.
``outputs/v2/tables/tune_dl_v2_best.csv``          – winner per (T, L, W, arch).
``outputs/v2/tables/tune_dl_v2_multiseed.csv``     – 5-seed re-fit of winners.
``outputs/v2/tables/tune_dl_v2_multiseed_summary.csv`` – mean/std per group.

Per the plan, L=28 is skipped by default to keep training time bounded.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.event_features import build_daily_feature_tables
from src.event_labels import DEFAULT_THRESHOLDS, run_event_labeling
from src.event_v2 import OUTPUT_TABLES_V2
from src.event_v2.deep_learning_v2 import (
    ARCH_FACTORIES,
    build_sequence_dataset,
    split_sequence_dataset,
    train_one_arch,
)
from src.event_v2.samples_v2 import build_all_sample_indexes, write_sample_indexes
from src.event_v2.split_v2 import (
    make_patient_three_way_split,
    select_best_by_pr_auc,
)

# ---------------------------------------------------------------- #
# Hyperparameter grids (coarse).
# ---------------------------------------------------------------- #
RECURRENT_GRID: list[dict[str, Any]] = [
    {"hidden_dim": h, "num_layers": n, "dropout": p}
    for h in (32, 64)
    for n in (1, 2)
    for p in (0.2, 0.4)
]
CNN_GRID: list[dict[str, Any]] = [{"dropout": p} for p in (0.2, 0.4)]


def _arch_grid(arch: str) -> list[dict[str, Any]]:
    return CNN_GRID if arch == "cnn" else RECURRENT_GRID


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _stringify(hp: dict[str, Any]) -> str:
    return ";".join(f"{k}={v}" for k, v in sorted(hp.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS)
    )
    parser.add_argument("--lengths", default="3,7,14",
                        help="Comma-separated input lengths; L=28 omitted by default.")
    parser.add_argument("--washouts", default="0,7,14")
    parser.add_argument(
        "--archs", default="gru,lstm,rnn,cnn",
        help="Comma-separated subset of {gru, lstm, rnn, cnn}.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument(
        "--skip-multi-seed", action="store_true",
        help="Run only the HPO sweep; don't re-fit winners across 5 seeds.",
    )
    args = parser.parse_args()

    thresholds = _parse_int_list(args.thresholds)
    lengths = _parse_int_list(args.lengths)
    washouts = _parse_int_list(args.washouts)
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    multi_seeds = _parse_int_list(args.multi_seeds)

    for arch in archs:
        if arch not in ARCH_FACTORIES:
            raise SystemExit(f"Unknown arch: {arch!r}")

    print(f"[v2-tune-dl] outputs -> {OUTPUT_TABLES_V2}")
    print(f"  thresholds={thresholds}  lengths={lengths}  washouts={washouts}")
    print(f"  archs={archs}  seed={args.seed}  epochs={args.epochs}")

    # ---- 1. Labels + samples + daily tables ------------------------ #
    print("\n[1/3] Building labels, samples, daily feature tables...")
    label_artifacts = run_event_labeling(thresholds=thresholds)
    sample_indexes, sample_counts = build_all_sample_indexes(
        weekly_events=label_artifacts.weekly_events,
        probable_days_by_threshold=label_artifacts.probable_event_days,
        episodes_by_threshold=label_artifacts.event_episodes,
        input_lengths=lengths,
        washout_values=washouts,
    )
    sample_indexes = {k: v for k, v in sample_indexes.items() if k[0] in thresholds}
    write_sample_indexes(sample_indexes, sample_counts)
    daily_tables = build_daily_feature_tables()

    # ---- 2. Determine split from any non-empty table --------------- #
    any_idx = next(v for v in sample_indexes.values() if not v.empty)
    split = make_patient_three_way_split(
        groups=any_idx["user_key"],
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    print(
        f"  split: train={split.train_users} val={split.val_users} "
        f"test={split.test_users}"
    )
    (OUTPUT_TABLES_V2 / "tune_dl_v2_split.json").write_text(
        json.dumps(split.to_dict(), indent=2)
    )

    # ---- 3. HPO sweep ---------------------------------------------- #
    print("\n[2/3] Running DL HPO sweep...")
    trial_rows: list[dict[str, Any]] = []
    for key in sorted(sample_indexes):
        T, L, W = key
        sample_index = sample_indexes[key]
        if sample_index.empty or sample_index["target"].nunique() < 2:
            print(f"  skip T={T} L={L} W={W}: empty/degenerate.")
            continue
        try:
            dataset = build_sequence_dataset(sample_index, daily_tables)
        except Exception as exc:  # pragma: no cover
            print(f"  skip T={T} L={L} W={W}: dataset build failed: {exc!r}")
            continue
        X_dict, y_dict = split_sequence_dataset(dataset, split)
        if (
            len(y_dict["train"]) == 0
            or len(y_dict["val"]) == 0
            or len(np.unique(y_dict["train"])) < 2
            or int(y_dict["val"].sum()) == 0
        ):
            print(f"  skip T={T} L={L} W={W}: invalid split (no positives in val).")
            continue
        base = {
            "threshold": T,
            "input_length_days": L,
            "washout_days": W,
            "n_train": int(len(y_dict["train"])),
            "n_val": int(len(y_dict["val"])),
            "n_test": int(len(y_dict["test"])),
            "train_positive": int(y_dict["train"].sum()),
            "val_positive": int(y_dict["val"].sum()),
            "test_positive": int(y_dict["test"].sum()),
            "n_channels": int(X_dict["train"].shape[2]),
            "seed": args.seed,
        }
        for arch in archs:
            grid = _arch_grid(arch)
            print(f"  T={T} L={L} W={W} arch={arch}: {len(grid)} trials")
            for hp in grid:
                try:
                    out = train_one_arch(
                        arch=arch,
                        X_dict=X_dict,
                        y_dict=y_dict,
                        hp=hp,
                        seed=args.seed,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        patience=args.patience,
                        learning_rate=args.learning_rate,
                    )
                except Exception as exc:  # pragma: no cover
                    print(f"    !! trial failed ({arch} {hp}): {exc!r}")
                    continue
                row = {
                    **base,
                    "arch": arch,
                    "hp": _stringify(hp),
                    **{f"hp_{k}": v for k, v in hp.items()},
                    "best_epoch": out["best_epoch"],
                    "best_val_loss": out["best_val_loss"],
                }
                for k, v in out.items():
                    if k in {"p_val", "p_test"}:
                        continue
                    if isinstance(v, np.ndarray):
                        continue
                    row[k] = v
                trial_rows.append(row)

    if not trial_rows:
        print("No DL trials produced. Exiting.")
        return 1
    trials = pd.DataFrame(trial_rows)
    trials_path = OUTPUT_TABLES_V2 / "tune_dl_v2_trials.csv"
    trials.to_csv(trials_path, index=False)
    print(f"\nWrote {len(trials)} DL trials -> {trials_path}")

    best = select_best_by_pr_auc(
        trials,
        score_col="val_pr_auc",
        group_cols=("threshold", "input_length_days", "washout_days", "arch"),
    )
    best_path = OUTPUT_TABLES_V2 / "tune_dl_v2_best.csv"
    best.to_csv(best_path, index=False)
    print(f"Wrote {len(best)} winners -> {best_path}")

    if args.skip_multi_seed:
        return 0

    # ---- 4. Multi-seed re-fit of overall winner -------------------- #
    print("\n[3/3] Multi-seed re-fit of the overall PR-AUC winner...")
    overall_winner = select_best_by_pr_auc(trials, score_col="val_pr_auc").iloc[0]
    T = int(overall_winner["threshold"])
    L = int(overall_winner["input_length_days"])
    W = int(overall_winner["washout_days"])
    arch = str(overall_winner["arch"])
    hp = {
        k[3:]: overall_winner[k]
        for k in overall_winner.index
        if k.startswith("hp_") and pd.notna(overall_winner[k])
    }
    # cast known numeric hps:
    for k in ("hidden_dim", "num_layers"):
        if k in hp:
            hp[k] = int(hp[k])
    print(f"  winner: T={T} L={L} W={W} arch={arch} hp={hp}")

    sample_index = sample_indexes[(T, L, W)]
    dataset = build_sequence_dataset(sample_index, daily_tables)
    X_dict, y_dict = split_sequence_dataset(dataset, split)

    ms_rows = []
    for s in multi_seeds:
        out = train_one_arch(
            arch=arch,
            X_dict=X_dict,
            y_dict=y_dict,
            hp=hp,
            seed=s,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            learning_rate=args.learning_rate,
        )
        row = {
            "threshold": T,
            "input_length_days": L,
            "washout_days": W,
            "arch": arch,
            "hp": _stringify(hp),
            "seed": s,
            "best_epoch": out["best_epoch"],
            "best_val_loss": out["best_val_loss"],
        }
        for k, v in out.items():
            if k in {"p_val", "p_test"}:
                continue
            if isinstance(v, np.ndarray):
                continue
            row[k] = v
        ms_rows.append(row)
    ms_df = pd.DataFrame(ms_rows)
    ms_path = OUTPUT_TABLES_V2 / "tune_dl_v2_multiseed.csv"
    ms_df.to_csv(ms_path, index=False)

    summary_cols = [
        c
        for c in ms_df.columns
        if c.startswith(("train_", "val_", "test_"))
        or c in {"train_val_pr_auc_gap", "val_test_pr_auc_gap"}
    ]
    summary = (
        ms_df.groupby(["threshold", "input_length_days", "washout_days", "arch"])[summary_cols]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary["n_seeds"] = len(multi_seeds)
    summary_path = OUTPUT_TABLES_V2 / "tune_dl_v2_multiseed_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {len(ms_df)} multi-seed rows -> {ms_path}")
    print(f"Wrote summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
