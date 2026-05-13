"""Targeted run: A1 per-patient normalization only (the most promising variant).

Single-seed signal showed A1 at 0.761 vs baseline 0.688 (Δ=+0.073).
This script runs A1 across all 5 seeds with the canonical LSTM_SW_inhaler_pef arch
to get a proper multi-seed mean±std comparable to the locked baseline 0.634±0.062.

Also runs the baseline architecture for the same seeds to control for split-lottery.
We accept the 0.634±0.062 baseline figure from the locked run but include
fresh baseline numbers here for direct apples-to-apples comparison.
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
from src.deep_learning import (
    _platt_scale, make_multimodal_loaders, make_multimodal_recurrent,
    predict_multimodal_model, set_seed, train_multimodal_model,
)
from src.experiments import (
    ExperimentConfig, evaluate_with_ci, prepare_data_for_experiment,
)

SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 10
PATIENCE = 3
BATCH_SIZE = 128  # 2x speedup vs 64

VARIANTS = [
    ("baseline", ExperimentConfig()),
    ("A1_per_patient_norm", ExperimentConfig(normalization="per_patient")),
]


def run_one(variant_name, cfg, seed):
    set_seed(seed)
    split, labeled = prepare_data_for_experiment(cfg, random_state=seed)
    col_idx = [
        i for i, n in enumerate(split.tab_feature_names)
        if n.startswith("smartinhaler") or n.startswith("peakflow")
    ]
    tl, vl, te = make_multimodal_loaders(split, col_indices=col_idx, batch_size=BATCH_SIZE)
    set_seed(seed)
    model = make_multimodal_recurrent(
        arch="lstm", tab_dim=len(col_idx),
        seq_input_dim=split.X_train.shape[-1],
    )
    model, _, meta = train_multimodal_model(
        model, tl, vl, split.y_train,
        epochs=EPOCHS, patience=PATIENCE, lr=5e-4, verbose=False,
    )
    p_val = predict_multimodal_model(model, vl)
    p_test = _platt_scale(split.y_val, p_val, predict_multimodal_model(model, te))
    test_groups = labeled.iloc[split.test_idx]["user_key"].to_numpy()
    metrics = evaluate_with_ci(split.y_test, p_test, test_groups=test_groups, seed=seed)
    metrics.update({
        "variant": variant_name, "seed": seed,
        "best_epoch": meta["best_epoch"],
        "tab_dim": len(col_idx),
    })
    return metrics


def main():
    rows = []
    for variant_name, cfg in VARIANTS:
        for seed in SEEDS:
            m = run_one(variant_name, cfg, seed)
            rows.append(m)
            print(
                f"[{variant_name:22s} seed={seed}] "
                f"AUC={m['roc_auc']:.4f}  "
                f"CI=[{m.get('auc_ci_lo', float('nan')):.3f}, "
                f"{m.get('auc_ci_hi', float('nan')):.3f}]  "
                f"ep={m['best_epoch']}",
                flush=True,
            )
            # Save after every seed
            df = pd.DataFrame(rows)
            df.to_csv(OUTPUT_TABLES / "exp_a1_results.csv", index=False)

    df = pd.DataFrame(rows)
    agg_cols = [c for c in ["roc_auc", "pr_auc", "f1", "brier",
                            "auc_ci_lo", "auc_ci_hi"] if c in df.columns]
    agg = df.groupby("variant")[agg_cols].agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(OUTPUT_TABLES / "exp_a1_results_summary.csv", index=False)
    print("\n=== A1 Aggregate ===")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
