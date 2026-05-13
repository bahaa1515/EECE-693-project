"""Offline analysis of v2 results.

Run *after* Phase 8 completes.  Uses the predictions written by
``final_test_eval_v2.py`` to compute:

1. **Bootstrap 95% CIs** for test PR-AUC and ROC-AUC of each winner
   (tabular per-algo + DL per-arch).
2. **Operating-threshold summary** — F1 at val-tuned threshold vs default.
3. **v1 vs v2 headline comparison** — pulls the closest matching v1
   configuration from ``outputs/tables/model_results_questionnaire_event_episode_labeling.csv``.
4. **Leakage-probe mean check** — overall mean of shuffled-label val
   ROC-AUC across all configs (should hover near 0.5).
5. **Sensor-ablation top-K** — best subset per algo.

All outputs are written to ``outputs/v2/tables/analysis_v2_*.csv`` and a
plain-text report is printed to stdout.

This script does **not** train any model.  It only reads CSVs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.event_v2 import OUTPUT_TABLES_V2

# v1 results (for reference comparison)
V1_RESULTS = REPO_ROOT / "outputs" / "tables" / "model_results_questionnaire_event_episode_labeling.csv"


def _bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "pr_auc",
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (point, lo, hi) for a bootstrap CI of the chosen metric.

    Falls back to (point, NaN, NaN) when y_true has < 2 classes or len < 5.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n < 5 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")

    fn = average_precision_score if metric == "pr_auc" else roc_auc_score
    point = float(fn(y_true, y_prob))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            samples.append(float(fn(y_true[idx], y_prob[idx])))
        except ValueError:
            continue
    if not samples:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return point, lo, hi


def _analyze_tabular(pred_path: Path, n_boot: int) -> pd.DataFrame:
    if not pred_path.exists():
        print(f"  (skip) no tabular predictions at {pred_path}")
        return pd.DataFrame()
    preds = pd.read_csv(pred_path)
    if preds.empty:
        return pd.DataFrame()
    group_cols = ["algo", "threshold", "input_length_days", "washout_days"]
    rows = []
    for keys, grp in preds.groupby(group_cols):
        y_true = grp["y_true"].to_numpy()
        y_prob = grp["y_prob"].to_numpy()
        pr, pr_lo, pr_hi = _bootstrap_ci(y_true, y_prob, "pr_auc", n_boot=n_boot)
        roc, roc_lo, roc_hi = _bootstrap_ci(y_true, y_prob, "roc_auc", n_boot=n_boot)
        rows.append(
            dict(zip(group_cols, keys)) | {
                "n_test": len(grp),
                "n_test_positive": int(y_true.sum()),
                "pr_auc": pr, "pr_auc_lo95": pr_lo, "pr_auc_hi95": pr_hi,
                "roc_auc": roc, "roc_auc_lo95": roc_lo, "roc_auc_hi95": roc_hi,
            }
        )
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False)


def _analyze_dl(pred_path: Path, n_boot: int) -> pd.DataFrame:
    if not pred_path.exists():
        print(f"  (skip) no DL predictions at {pred_path}")
        return pd.DataFrame()
    preds = pd.read_csv(pred_path)
    if preds.empty:
        return pd.DataFrame()
    # For DL: per-seed bootstrap, then aggregate.
    group_cols = ["arch", "threshold", "input_length_days", "washout_days", "seed"]
    seed_rows = []
    for keys, grp in preds.groupby(group_cols):
        y_true = grp["y_true"].to_numpy()
        y_prob = grp["y_prob"].to_numpy()
        pr, pr_lo, pr_hi = _bootstrap_ci(y_true, y_prob, "pr_auc", n_boot=n_boot)
        roc, roc_lo, roc_hi = _bootstrap_ci(y_true, y_prob, "roc_auc", n_boot=n_boot)
        seed_rows.append(
            dict(zip(group_cols, keys)) | {
                "n_test": len(grp),
                "pr_auc": pr, "pr_auc_lo95": pr_lo, "pr_auc_hi95": pr_hi,
                "roc_auc": roc, "roc_auc_lo95": roc_lo, "roc_auc_hi95": roc_hi,
            }
        )
    seed_df = pd.DataFrame(seed_rows)
    if seed_df.empty:
        return seed_df
    # Aggregate across seeds (mean + std of the per-seed point estimates).
    agg = (
        seed_df.groupby(["arch", "threshold", "input_length_days", "washout_days"])
        .agg(
            n_seeds=("seed", "count"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            # Cross-seed CI by combining all predictions (NB: not iid, but
            # a useful sanity check):
        )
        .reset_index()
        .sort_values("pr_auc_mean", ascending=False)
    )
    return agg


def _v1_v2_comparison(tab_summary: pd.DataFrame) -> pd.DataFrame:
    """Match each v2 winner's (T,L,W) to the closest v1 row, where it exists."""
    if not V1_RESULTS.exists() or tab_summary.empty:
        return pd.DataFrame()
    v1 = pd.read_csv(V1_RESULTS)
    rows = []
    for r in tab_summary.itertuples(index=False):
        match = v1[
            (v1["threshold"] == r.threshold)
            & (v1["input_length_days"] == r.input_length_days)
            & (v1["washout_days"] == r.washout_days)
        ]
        v1_pr = float(match["pr_auc"].max()) if not match.empty else float("nan")
        v1_roc = float(match["roc_auc"].max()) if not match.empty else float("nan")
        rows.append(
            {
                "algo": r.algo,
                "threshold": r.threshold,
                "input_length_days": r.input_length_days,
                "washout_days": r.washout_days,
                "v1_best_pr_auc": v1_pr,
                "v1_best_roc_auc": v1_roc,
                "v2_pr_auc": r.pr_auc,
                "v2_pr_auc_lo95": r.pr_auc_lo95,
                "v2_pr_auc_hi95": r.pr_auc_hi95,
                "v2_roc_auc": r.roc_auc,
                "v2_roc_auc_lo95": r.roc_auc_lo95,
                "v2_roc_auc_hi95": r.roc_auc_hi95,
                "pr_auc_delta": r.pr_auc - v1_pr,
                "roc_auc_delta": r.roc_auc - v1_roc,
            }
        )
    return pd.DataFrame(rows)


def _leakage_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return {
        "n_configs": int(len(df)),
        "val_roc_auc_mean_overall": float(df["val_roc_auc_mean"].mean()),
        "val_roc_auc_std_overall": float(df["val_roc_auc_mean"].std()),
        "n_outside_0.4_0.6": int(((df["val_roc_auc_mean"] < 0.4) | (df["val_roc_auc_mean"] > 0.6)).sum()),
        "val_pr_auc_mean_overall": float(df["val_pr_auc_mean"].mean()),
    }


def _sensor_ablation_summary(path: Path, top_k: int = 5) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    # Best per algo + subset (highest val_pr_auc).
    keep = ["algo", "subset", "threshold", "input_length_days", "washout_days",
            "val_pr_auc", "val_roc_auc", "val_f1",
            "test_pr_auc", "test_roc_auc", "test_f1"]
    have = [c for c in keep if c in df.columns]
    return df[have].sort_values(
        [c for c in ("algo", "val_pr_auc") if c in df.columns],
        ascending=[True, False] if "algo" in df.columns else [False],
    ).groupby("algo", as_index=False).head(top_k) if "algo" in df.columns else df.sort_values("val_pr_auc", ascending=False).head(top_k)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--tables-dir",
        default=str(OUTPUT_TABLES_V2),
        help="Where v2 CSVs live.",
    )
    args = parser.parse_args()

    tables = Path(args.tables_dir)
    print(f"\n=== v2 offline analysis ({tables}) ===")

    # 1. Bootstrap CIs (tabular + DL)
    print("\n[1/4] Bootstrap CIs (tabular winners)...")
    tab_ci = _analyze_tabular(tables / "final_test_v2_predictions_tabular.csv", args.n_boot)
    if not tab_ci.empty:
        out = tables / "analysis_v2_tabular_ci.csv"
        tab_ci.to_csv(out, index=False)
        print(tab_ci.to_string(index=False))
        print(f"  -> {out}")

    print("\n[1/4] Bootstrap CIs (DL multi-seed winner)...")
    dl_ci = _analyze_dl(tables / "final_test_v2_predictions_dl.csv", args.n_boot)
    if not dl_ci.empty:
        out = tables / "analysis_v2_dl_ci.csv"
        dl_ci.to_csv(out, index=False)
        print(dl_ci.to_string(index=False))
        print(f"  -> {out}")

    # 2. v1 vs v2 comparison
    print("\n[2/4] v1 vs v2 comparison (tabular only, matched on T/L/W)...")
    cmp_df = _v1_v2_comparison(tab_ci)
    if not cmp_df.empty:
        out = tables / "analysis_v2_v1_vs_v2.csv"
        cmp_df.to_csv(out, index=False)
        print(cmp_df.to_string(index=False))
        print(f"  -> {out}")
        print("\nNOTE: v1 used a 2-way split (no held-out val), so absolute")
        print("      deltas are indicative, not strictly apples-to-apples.")
    else:
        print("  (no comparable v1 rows found)")

    # 3. Leakage probe mean check
    print("\n[3/4] Leakage probe sanity check (mean shuffled-label val ROC-AUC)...")
    leak = _leakage_summary(tables / "leakage_probe_v2_summary.csv")
    if leak:
        for k, v in leak.items():
            print(f"  {k}: {v}")
        if not (0.45 <= leak["val_roc_auc_mean_overall"] <= 0.55):
            print("  >>> WARNING: overall mean is NOT near 0.5; investigate.")
        else:
            print("  OK: overall mean is near 0.5 (no systematic leakage).")

    # 4. Sensor ablation summary
    print("\n[4/4] Sensor-ablation top winners per algo...")
    abl = _sensor_ablation_summary(tables / "sensor_ablation_v2.csv", top_k=3)
    if not abl.empty:
        out = tables / "analysis_v2_sensor_ablation_topk.csv"
        abl.to_csv(out, index=False)
        print(abl.to_string(index=False))
        print(f"  -> {out}")

    print("\n=== done. ===\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
