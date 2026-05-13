"""Experiment: LightGBM multimodal baseline.

Rationale: at n_patients=18, the Tier-1 RF (0.633) already beats all the deep
sequence models (0.46-0.51); GBMs are *exactly* the right inductive bias for
small tabular regimes.  LightGBM with the full
[29 Tier-1 engineered + smart-inhaler + peak-flow + weekly + patient-static]
panel is the natural strongest tabular baseline.

Runs 5 seeds, patient-cluster bootstrap CIs, per-seed and aggregate output.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import OUTPUT_TABLES
from src.experiments import (
    ExperimentConfig,
    prepare_data_for_experiment,
    evaluate_with_ci,
)


def run_lightgbm_panel(
    feature_subsets: dict[str, list[str] | None],
    seeds: list[int],
    output_path: Path,
) -> pd.DataFrame:
    import lightgbm as lgb

    all_rows: list[dict] = []

    for subset_name, prefixes in feature_subsets.items():
        for seed in seeds:
            cfg = ExperimentConfig(
                normalization="global",
                split_strategy="groupshuffle",
                label_name="canonical",
                include_engineered=True,
            )
            split, labeled = prepare_data_for_experiment(cfg, random_state=seed)

            # Select tabular columns by prefix (None → all)
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

            # train+val concatenated for final fit (after using val for early stop)
            pos = float((y_train == 1).sum())
            neg = float((y_train == 0).sum())
            scale_pos = max(neg / max(pos, 1.0), 1.0)

            clf = lgb.LGBMClassifier(
                n_estimators=2000,
                learning_rate=0.02,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=20,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=scale_pos,
                random_state=seed,
                n_jobs=1,
                verbose=-1,
            )
            clf.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            p_val = clf.predict_proba(X_val)[:, 1]
            p_test_raw = clf.predict_proba(X_test)[:, 1]

            # Platt scaling on val for fair comparison vs deep models
            from sklearn.linear_model import LogisticRegression
            cal = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
            cal.fit(p_val.reshape(-1, 1), y_val)
            p_test = cal.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

            test_groups = labeled.iloc[split.test_idx]["user_key"].to_numpy()
            metrics = evaluate_with_ci(y_test, p_test, test_groups=test_groups, seed=seed)
            metrics.update({
                "model": f"LightGBM_{subset_name}",
                "subset": subset_name,
                "seed": seed,
                "n_features": len(col_idx),
                "n_estimators_used": int(clf.best_iteration_ or clf.n_estimators),
            })
            all_rows.append(metrics)
            print(
                f"[LGB {subset_name:25s} seed={seed}]  "
                f"AUC={metrics['roc_auc']:.4f}  "
                f"CI=[{metrics.get('auc_ci_lo', float('nan')):.3f}, "
                f"{metrics.get('auc_ci_hi', float('nan')):.3f}]  "
                f"n_feat={len(col_idx):3d}  "
                f"n_trees={metrics['n_estimators_used']}"
            )

    df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Aggregate
    agg_cols = ["roc_auc", "pr_auc", "f1", "brier", "auc_ci_lo", "auc_ci_hi"]
    available = [c for c in agg_cols if c in df.columns]
    agg = df.groupby("model")[available].agg(["mean", "std", "min", "max"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(output_path.with_name(output_path.stem + "_summary.csv"), index=False)
    print("\n=== Aggregate ===")
    print(agg.to_string(index=False))
    return agg


def main() -> int:
    seeds = [42, 43, 44, 45, 46]
    # Cumulative feature subsets, including the engineered Tier-1 block (prefix "eng_")
    feature_subsets: dict[str, list[str] | None] = {
        "ENG_only":        ["eng_"],
        "ENG_inhaler":     ["eng_", "smartinhaler"],
        "ENG_inhaler_pef": ["eng_", "smartinhaler", "peakflow"],
        "ENG_full":        None,  # all features
        "MM_inhaler_pef":  ["smartinhaler", "peakflow"],  # no engineered (apples-to-apples vs Tier-3 LSTM_SW_inhaler_pef)
        "MM_full":         ["smartinhaler", "peakflow", "weekly", "patient_"],  # everything except engineered
    }
    out = OUTPUT_TABLES / "exp_lightgbm_results.csv"
    run_lightgbm_panel(feature_subsets, seeds, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
