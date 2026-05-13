"""Experiment: Tabular GBM/RF baselines under the multi-seed protocol.

The val set is severely patient-shifted at n=18, which makes val-loss-based
early stopping unreliable.  This script:

1. Drops val-based early stopping in favour of a fixed shallow budget that
   matches the RF baseline philosophy (shallow trees, moderate ensemble).
2. Establishes a true *multi-seed* baseline for Random Forest using the same
   protocol as the deep-model experiments (5 patient-wise splits).
3. Compares Random Forest vs LightGBM vs XGBoost on identical splits and
   feature subsets, with patient-cluster bootstrap CIs.

Outputs:
    outputs/tables/exp_gbm_results.csv          per-seed metrics
    outputs/tables/exp_gbm_results_summary.csv  mean ± std aggregate
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

from src.config import OUTPUT_TABLES
from src.experiments import (
    ExperimentConfig,
    prepare_data_for_experiment,
    evaluate_with_ci,
)

SEEDS = [42, 43, 44, 45, 46]

# Cumulative feature subsets (eng = 29 Tier-1, mm = inhaler/peakflow/weekly/static)
FEATURE_SUBSETS: dict[str, list[str] | None] = {
    "ENG_only":         ["eng_"],
    "MM_inhaler_pef":   ["smartinhaler", "peakflow"],
    "MM_full":          ["smartinhaler", "peakflow", "weekly", "patient_"],
    "ENG_inhaler":      ["eng_", "smartinhaler"],
    "ENG_inhaler_pef":  ["eng_", "smartinhaler", "peakflow"],
    "ENG_MM_full":      None,  # everything = engineered + all multimodal
}


def make_models(seed: int):
    """Return list of (name, fitted-model-factory, requires_fit_call) tuples."""
    from sklearn.ensemble import RandomForestClassifier
    import lightgbm as lgb
    import xgboost as xgb

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.5,
        is_unbalance=True,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )
    xgbm = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.5,
        scale_pos_weight=1.0,  # set below
        random_state=seed,
        n_jobs=1,
        eval_metric="auc",
        verbosity=0,
    )
    return [("RandomForest", rf), ("LightGBM", lgbm), ("XGBoost", xgbm)]


def run() -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression

    rows: list[dict] = []

    for subset_name, prefixes in FEATURE_SUBSETS.items():
        for seed in SEEDS:
            cfg = ExperimentConfig(
                normalization="global",
                split_strategy="groupshuffle",
                label_name="canonical",
                include_engineered=True,
            )
            split, labeled = prepare_data_for_experiment(cfg, random_state=seed)

            if prefixes is None:
                col_idx = list(range(len(split.tab_feature_names)))
            else:
                col_idx = [
                    i for i, n in enumerate(split.tab_feature_names)
                    if any(n.startswith(p) for p in prefixes)
                ]
            if not col_idx:
                continue

            X_train = split.tab_train[:, col_idx]
            X_val = split.tab_val[:, col_idx]
            X_test = split.tab_test[:, col_idx]
            y_train, y_val, y_test = split.y_train, split.y_val, split.y_test
            test_groups = labeled.iloc[split.test_idx]["user_key"].to_numpy()

            pos = max(float((y_train == 1).sum()), 1.0)
            neg = max(float((y_train == 0).sum()), 1.0)
            scale_pos = max(neg / pos, 1.0)

            for name, model in make_models(seed):
                if name == "XGBoost":
                    model.set_params(scale_pos_weight=scale_pos)
                model.fit(X_train, y_train)
                p_val = model.predict_proba(X_val)[:, 1]
                p_test_raw = model.predict_proba(X_test)[:, 1]

                # Platt scaling on val
                cal = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
                cal.fit(p_val.reshape(-1, 1), y_val)
                p_test = cal.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

                metrics = evaluate_with_ci(
                    y_test, p_test, test_groups=test_groups, seed=seed,
                )
                metrics.update({
                    "model": name,
                    "subset": subset_name,
                    "seed": seed,
                    "n_features": len(col_idx),
                })
                rows.append(metrics)
                print(
                    f"[{name:12s} {subset_name:18s} seed={seed}] "
                    f"AUC={metrics['roc_auc']:.4f}  "
                    f"CI=[{metrics.get('auc_ci_lo', float('nan')):.3f}, "
                    f"{metrics.get('auc_ci_hi', float('nan')):.3f}]"
                )

    df = pd.DataFrame(rows)
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_TABLES / "exp_gbm_results.csv"
    df.to_csv(out, index=False)

    agg_cols = ["roc_auc", "pr_auc", "f1", "brier", "auc_ci_lo", "auc_ci_hi"]
    available = [c for c in agg_cols if c in df.columns]
    agg = (
        df.groupby(["model", "subset"])[available]
        .agg(["mean", "std"]).round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index().sort_values("roc_auc_mean", ascending=False)
    agg.to_csv(out.with_name("exp_gbm_results_summary.csv"), index=False)
    print("\n=== Aggregate (sorted by AUC mean) ===")
    print(agg.to_string(index=False))
    return agg


if __name__ == "__main__":
    run()
