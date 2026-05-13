"""Aggregate all experiment results into one comparison table.

Reads every ``outputs/tables/exp_*_results.csv``, normalises the column
names, and produces:

- ``outputs/tables/experiment_comparison.csv`` — wide table sorted by AUC.
- ``outputs/tables/experiment_comparison.tex`` — LaTeX snippet for the report.
- Console: top-10 ranked experiments.

The locked baseline (LSTM_SW_inhaler_pef = 0.634 ± 0.062) is added as a row
so every experiment can be compared directly against it (delta column).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import OUTPUT_TABLES

BASELINE_NAME = "LSTM_SW_inhaler_pef (locked)"
BASELINE_AUC_MEAN = 0.634
BASELINE_AUC_STD = 0.062


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Compute mean/std for AUC and a couple of secondary metrics."""
    metrics = ["roc_auc", "pr_auc", "f1", "auc_ci_lo", "auc_ci_hi"]
    available = [c for c in metrics if c in df.columns]
    g = df.groupby(group_cols)[available].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    g = g.reset_index()
    g["n_seeds"] = df.groupby(group_cols).size().reset_index(drop=True)
    return g


def main() -> int:
    parts: list[pd.DataFrame] = []

    # GBM tabular
    p = OUTPUT_TABLES / "exp_gbm_results.csv"
    if p.exists():
        df = pd.read_csv(p)
        agg = _aggregate(df, ["model", "subset"])
        agg["experiment"] = "GBM_" + agg["model"] + "_" + agg["subset"]
        agg["family"] = "Tabular GBM/RF"
        parts.append(agg)

    # LightGBM original run
    p = OUTPUT_TABLES / "exp_lightgbm_results.csv"
    if p.exists():
        df = pd.read_csv(p)
        agg = _aggregate(df, ["model"])
        agg["experiment"] = agg["model"]
        agg["family"] = "LightGBM (early-stop)"
        parts.append(agg)

    # Focused A1 vs baseline run
    p = OUTPUT_TABLES / "exp_a1_results.csv"
    if p.exists():
        df = pd.read_csv(p)
        agg = _aggregate(df, ["variant"])
        agg["experiment"] = "LSTM_" + agg["variant"] + " (focused)"
        agg["family"] = "DL variants"
        parts.append(agg)

    # Stacking
    p = OUTPUT_TABLES / "exp_stacking_results.csv"
    if p.exists():
        df = pd.read_csv(p)
        agg = _aggregate(df, ["model"])
        agg["experiment"] = "Stack_" + agg["model"]
        agg["family"] = "Stacking"
        parts.append(agg)

    if not parts:
        print("No experiment results found yet.")
        return 1

    combined = pd.concat(parts, ignore_index=True)

    # Add locked baseline row
    baseline_row = pd.DataFrame([{
        "experiment": BASELINE_NAME,
        "family": "Baseline",
        "roc_auc_mean": BASELINE_AUC_MEAN,
        "roc_auc_std": BASELINE_AUC_STD,
        "n_seeds": 5,
    }])
    combined = pd.concat([combined, baseline_row], ignore_index=True)

    # Compute delta vs baseline
    combined["delta_auc"] = (combined["roc_auc_mean"] - BASELINE_AUC_MEAN).round(4)
    combined = combined.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)

    # Slim view for printing/saving
    display_cols = [
        "experiment", "family", "roc_auc_mean", "roc_auc_std",
        "delta_auc", "pr_auc_mean", "f1_mean", "n_seeds",
    ]
    display_cols = [c for c in display_cols if c in combined.columns]
    summary = combined[display_cols].copy()
    for c in ["roc_auc_mean", "roc_auc_std", "delta_auc", "pr_auc_mean", "f1_mean"]:
        if c in summary.columns:
            summary[c] = summary[c].round(4)

    out = OUTPUT_TABLES / "experiment_comparison.csv"
    summary.to_csv(out, index=False)
    print(f"\nFull table written to {out}")

    # LaTeX snippet for top-15
    top = summary.head(15).copy()
    top["AUC"] = top.apply(
        lambda r: f"{r['roc_auc_mean']:.3f} +/- {r.get('roc_auc_std', 0):.3f}",
        axis=1,
    )
    top["Delta"] = top["delta_auc"].apply(lambda v: f"{v:+.3f}")
    tex = top[["experiment", "family", "AUC", "Delta", "n_seeds"]].rename(columns={
        "experiment": "Experiment", "family": "Family",
        "n_seeds": "Seeds",
    })
    tex_path = OUTPUT_TABLES / "experiment_comparison.tex"
    with tex_path.open("w", encoding="utf-8") as fh:
        fh.write(tex.to_latex(index=False, escape=True, column_format="lllrr"))
    print(f"LaTeX snippet written to {tex_path}")

    print("\n=== Top 15 experiments (AUC mean ± std) ===")
    print(top[["experiment", "family", "AUC", "Delta", "n_seeds"]].to_string(index=False))
    print(f"\nBaseline: {BASELINE_NAME}  AUC = {BASELINE_AUC_MEAN:.3f} +/- {BASELINE_AUC_STD:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
