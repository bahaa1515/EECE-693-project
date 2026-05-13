"""Experiment: Stacking ensemble using out-of-fold predictions across seeds.

For each seed we already have:
  - Random Forest (out-of-fold style via the patient-disjoint test set)
  - LightGBM
  - XGBoost
in ``exp_gbm_results.csv``.

True stacking would require leave-one-patient-out CV to generate OOF preds
on the *training* set, then fit a meta-learner on those.  That's compute-
intensive.  This script implements a simpler proxy that's still defensible:

  - For each seed, average / weighted-average the three GBM test predictions.
  - Also test a logistic-regression meta-learner trained on the *validation*
    folds across seeds (treating each seed as a quasi-fold).

Outputs:
    outputs/tables/exp_stacking_results.csv
    outputs/tables/exp_stacking_results_summary.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import OUTPUT_TABLES
from src.experiments import (
    ExperimentConfig,
    prepare_data_for_experiment,
    evaluate_with_ci,
)

SEEDS = [42, 43, 44, 45, 46]


def train_one_seed(seed: int, prefixes: list[str] | None):
    """Train RF, LGB, XGB on one patient-split and return val+test predictions."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb
    import xgboost as xgb

    cfg = ExperimentConfig(include_engineered=True)
    split, labeled = prepare_data_for_experiment(cfg, random_state=seed)
    if prefixes is None:
        col_idx = list(range(len(split.tab_feature_names)))
    else:
        col_idx = [i for i, n in enumerate(split.tab_feature_names)
                   if any(n.startswith(p) for p in prefixes)]
    X_train = split.tab_train[:, col_idx]
    X_val = split.tab_val[:, col_idx]
    X_test = split.tab_test[:, col_idx]
    y_train, y_val, y_test = split.y_train, split.y_val, split.y_test
    test_groups = labeled.iloc[split.test_idx]["user_key"].to_numpy()

    pos = max(float((y_train == 1).sum()), 1.0)
    neg = max(float((y_train == 0).sum()), 1.0)
    scale_pos = neg / pos

    models = {
        "RF": RandomForestClassifier(
            n_estimators=300, max_depth=10,
            min_samples_split=20, min_samples_leaf=10,
            class_weight="balanced", random_state=seed, n_jobs=1,
        ),
        "LGB": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            max_depth=4, min_child_samples=50, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=0.5, is_unbalance=True,
            random_state=seed, n_jobs=1, verbose=-1,
        ),
        "XGB": xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=0.5, scale_pos_weight=scale_pos,
            random_state=seed, n_jobs=1, eval_metric="auc", verbosity=0,
        ),
    }

    val_probs: dict[str, np.ndarray] = {}
    test_probs: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_probs[name] = model.predict_proba(X_val)[:, 1]
        test_probs[name] = model.predict_proba(X_test)[:, 1]

    # Platt-calibrate each base learner using its val probs
    for name in models:
        cal = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        cal.fit(val_probs[name].reshape(-1, 1), y_val)
        test_probs[name] = cal.predict_proba(test_probs[name].reshape(-1, 1))[:, 1]
        val_probs[name] = cal.predict_proba(val_probs[name].reshape(-1, 1))[:, 1]

    return val_probs, test_probs, y_val, y_test, test_groups


def main() -> int:
    from sklearn.linear_model import LogisticRegression

    rows: list[dict] = []
    for seed in SEEDS:
        val_probs, test_probs, y_val, y_test, test_groups = train_one_seed(
            seed, prefixes=["eng_", "smartinhaler", "peakflow"]
        )

        # 1) Average
        p_avg = np.mean(list(test_probs.values()), axis=0)

        # 2) Weighted by val AUC
        from sklearn.metrics import roc_auc_score
        weights = {}
        for name in test_probs:
            try:
                weights[name] = max(roc_auc_score(y_val, val_probs[name]), 0.5)
            except Exception:
                weights[name] = 0.5
        wsum = sum(weights.values())
        p_weighted = sum(
            test_probs[name] * weights[name] / wsum for name in test_probs
        )

        # 3) Logistic-regression meta-learner on val probs
        meta_X_val = np.column_stack([val_probs[n] for n in test_probs])
        meta_X_test = np.column_stack([test_probs[n] for n in test_probs])
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        meta.fit(meta_X_val, y_val)
        p_meta = meta.predict_proba(meta_X_test)[:, 1]

        ensembles = {
            "stack_avg": p_avg,
            "stack_weighted_val_auc": p_weighted,
            "stack_LR_meta": p_meta,
        }
        for name, probs in ensembles.items():
            metrics = evaluate_with_ci(y_test, probs, test_groups=test_groups, seed=seed)
            metrics.update({"model": name, "seed": seed})
            rows.append(metrics)
            print(
                f"[{name:24s} seed={seed}] "
                f"AUC={metrics['roc_auc']:.4f}  "
                f"CI=[{metrics.get('auc_ci_lo', float('nan')):.3f}, "
                f"{metrics.get('auc_ci_hi', float('nan')):.3f}]"
            )

    df = pd.DataFrame(rows)
    out = OUTPUT_TABLES / "exp_stacking_results.csv"
    df.to_csv(out, index=False)

    agg_cols = [c for c in ["roc_auc", "pr_auc", "f1", "brier",
                            "auc_ci_lo", "auc_ci_hi"] if c in df.columns]
    agg = df.groupby("model")[agg_cols].agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index().sort_values("roc_auc_mean", ascending=False)
    agg.to_csv(out.with_name("exp_stacking_results_summary.csv"), index=False)
    print("\n=== Stacking aggregate ===")
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
