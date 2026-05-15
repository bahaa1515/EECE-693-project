"""Phase 9 (diagnostic): Leave-one-patient-out CV + patient-level permutation test.

This script does **not** change the v2 pipeline, samples, features, or
3-way split.  It re-uses the existing v2 modules and only re-evaluates
two locked configurations under a different protocol to check whether
the Phase-8 headline numbers generalise across patients or were a
4-patient lottery.

Configurations validated (locked from Phase-4/Phase-6 winners):

  A) RF, sensors=all,            T=4, L=14, W=7
     hp: class_weight=balanced, n_estimators=200, max_depth=None,
         min_samples_leaf=1
  B) XGB, sensors=smartinhaler,  T=4, L=14, W=14
     hp: n_estimators=200, max_depth=3, learning_rate=0.1,
         subsample=1.0, scale_pos_weight=auto
     (hp inherited from the (4,14,14, all, xgb) Phase-4 winner, as in
     scripts/v2/sensor_ablation_v2.py)

Protocol
--------
LOPO CV: for each of the 18 unique patients p, train on the other 17,
predict on p, record per-patient ROC/PR + n_test + n_pos.  Pool all
predictions to compute a single global ROC/PR per config.

Patient-level permutation test: for each of K permutations, randomly
permute the mapping patient_id -> label-sequence (each patient's
features are kept intact; labels are borrowed positionally from a
randomly-paired patient).  This preserves within-patient temporal
structure but breaks any per-patient feature->label confound.  Re-run
LOPO under the permuted labels; record the pooled ROC.  The p-value is
P(pooled_ROC_null >= pooled_ROC_real).

Outputs (under outputs/v2/tables/)
----------------------------------
  lopo_v2_per_patient.csv         per-config x per-fold metrics
  lopo_v2_pooled.csv              per-config pooled metrics
  lopo_v2_permutation.csv         per-config x per-permutation pooled ROC
  lopo_v2_permutation_summary.csv per-config null summary + p-value
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score

from src.event_features import (
    build_daily_feature_tables,
    build_patient_static_features,
)
from src.event_labels import USER_COL, run_event_labeling
from src.event_v2 import OUTPUT_TABLES_V2
from src.event_v2.features_v2 import ALL_SENSOR_SOURCES, build_training_feature_table
from src.event_v2.modeling_v2 import build_model, select_feature_columns
from src.event_v2.samples_v2 import build_all_sample_indexes


# --------------------------------------------------------------------------
# Locked configurations
# --------------------------------------------------------------------------
CONFIGS = [
    {
        "name": "rf_all_T4_L14_W7",
        "algo": "rf",
        "sensors": list(ALL_SENSOR_SOURCES),
        "threshold": 4,
        "input_length_days": 14,
        "washout_days": 7,
        "hp": {
            "class_weight": "balanced",
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "n_jobs": -1,
        },
    },
    {
        "name": "xgb_smartinhaler_T4_L14_W14",
        "algo": "xgb",
        "sensors": ["smartinhaler"],
        "threshold": 4,
        "input_length_days": 14,
        "washout_days": 14,
        "hp": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "scale_pos_weight": "auto",
            "n_jobs": -1,
        },
    },
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _safe_roc(y, p) -> float:
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_pr(y, p) -> float:
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def _fit_predict_fold(
    cfg: dict,
    feat: pd.DataFrame,
    feature_cols: list[str],
    train_users: list,
    test_user,
    target_col: str = "target",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    train = feat[feat[USER_COL].isin(train_users)]
    test = feat[feat[USER_COL] == test_user]
    X_train, y_train = train[feature_cols], train[target_col].astype(int)
    X_test, y_test = test[feature_cols], test[target_col].astype(int)
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return y_test.to_numpy(), np.full(len(y_test), np.nan)
    model = build_model(cfg["algo"], cfg["hp"], n_pos=n_pos, n_neg=n_neg,
                        random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
        p_test = model.predict_proba(X_test)[:, 1]
    return y_test.to_numpy(), p_test


def _run_lopo(cfg: dict, feat: pd.DataFrame, target_col: str = "target",
              seed: int = 42, verbose: bool = True) -> pd.DataFrame:
    """Run LOPO CV for one (cfg, feat).  Returns per-fold rows."""
    users = sorted(feat[USER_COL].unique())
    feature_cols = select_feature_columns(feat)
    rows = []
    for held_out in users:
        train_users = [u for u in users if u != held_out]
        y_test, p_test = _fit_predict_fold(
            cfg, feat, feature_cols, train_users, held_out,
            target_col=target_col, seed=seed,
        )
        rows.append({
            "config": cfg["name"],
            "held_out_user": held_out,
            "n_test": int(len(y_test)),
            "n_test_pos": int(np.asarray(y_test).sum()),
            "roc_auc": _safe_roc(y_test, p_test),
            "pr_auc": _safe_pr(y_test, p_test),
            "_y": y_test,
            "_p": p_test,
        })
        if verbose:
            r = rows[-1]
            print(
                f"  [{cfg['name']}] user={held_out:>4} "
                f"n={r['n_test']:>4} pos={r['n_test_pos']:>2} "
                f"ROC={r['roc_auc']:.3f} PR={r['pr_auc']:.3f}"
            )
    return pd.DataFrame(rows)


def _pool_metrics(per_fold: pd.DataFrame) -> dict:
    """Concatenate all (y, p) across folds and compute global metrics."""
    y_all = np.concatenate([np.asarray(r) for r in per_fold["_y"]])
    p_all = np.concatenate([np.asarray(r) for r in per_fold["_p"]])
    mask = ~np.isnan(p_all)
    y_all, p_all = y_all[mask], p_all[mask]
    return {
        "pooled_roc_auc": _safe_roc(y_all, p_all),
        "pooled_pr_auc": _safe_pr(y_all, p_all),
        "pooled_n": int(len(y_all)),
        "pooled_n_pos": int(y_all.sum()),
        "per_patient_roc_mean": float(per_fold["roc_auc"].mean(skipna=True)),
        "per_patient_roc_std": float(per_fold["roc_auc"].std(skipna=True)),
        "per_patient_roc_min": float(per_fold["roc_auc"].min(skipna=True)),
        "per_patient_roc_max": float(per_fold["roc_auc"].max(skipna=True)),
        "n_folds_with_pos": int((per_fold["n_test_pos"] > 0).sum()),
        "n_folds_total": int(len(per_fold)),
    }


def _permute_labels_patient_block(
    feat: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.Series:
    """Permute patient -> label-block mapping.

    For each patient u with k samples, replace its label vector by the
    label vector of sigma(u) sampled positionally (with wrap-around if
    sizes differ).  Within-patient row order is preserved.
    """
    users = sorted(feat[USER_COL].unique())
    perm = list(users)
    rng.shuffle(perm)
    sigma = {u: perm[i] for i, u in enumerate(users)}
    # collect each user's label sequence in original within-patient order
    feat_sorted = feat.sort_values(
        [USER_COL, "sample_anchor_day"], kind="stable"
    )
    label_by_user = {
        u: feat_sorted.loc[feat_sorted[USER_COL] == u, "target"].to_numpy()
        for u in users
    }
    # build new label column, indexed back to feat's original index
    new_target = np.empty(len(feat_sorted), dtype=int)
    cursor = 0
    for u, group in feat_sorted.groupby(USER_COL, sort=False):
        donor = sigma[u]
        donor_labels = label_by_user[donor]
        k = len(group)
        if len(donor_labels) >= k:
            new_target[cursor:cursor + k] = donor_labels[:k]
        else:
            # wrap around (rare; only when donor has fewer samples)
            new_target[cursor:cursor + k] = np.resize(donor_labels, k)
        cursor += k
    out = pd.Series(new_target, index=feat_sorted.index, name="target_perm")
    return out.reindex(feat.index)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=50,
                        help="Number of patient-level label permutations.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(OUTPUT_TABLES_V2),
                        help="Destination for LOPO CSVs.")
    parser.add_argument("--skip-permutation", action="store_true",
                        help="Run LOPO only; skip the permutation null.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build labels, samples, daily tables, patient static -----------
    print("[1/4] Building labels + samples + daily tables ...")
    unique_keys = sorted({
        (c["threshold"], c["input_length_days"], c["washout_days"])
        for c in CONFIGS
    })
    thresholds = sorted({k[0] for k in unique_keys})
    lengths = sorted({k[1] for k in unique_keys})
    washouts = sorted({k[2] for k in unique_keys})

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

    # ---- Build feature tables per config -------------------------------
    print("[2/4] Building feature tables per config ...")
    feat_by_cfg: dict[str, pd.DataFrame] = {}
    for cfg in CONFIGS:
        key = (cfg["threshold"], cfg["input_length_days"], cfg["washout_days"])
        sample_index = sample_indexes[key]
        feat = build_training_feature_table(
            sample_index=sample_index,
            daily_tables=daily_tables,
            patient_features=patient_features,
            sensor_sources=cfg["sensors"],
        )
        n_users = feat[USER_COL].nunique()
        n_pos = int(feat["target"].sum())
        print(f"  {cfg['name']}: n={len(feat)} users={n_users} pos={n_pos}")
        feat_by_cfg[cfg["name"]] = feat.reset_index(drop=True)

    # ---- LOPO CV on the real labels ------------------------------------
    print("\n[3/4] Running LOPO CV on real labels ...")
    per_fold_rows = []
    pooled_rows = []
    real_pooled_roc = {}
    for cfg in CONFIGS:
        feat = feat_by_cfg[cfg["name"]]
        t0 = time.perf_counter()
        per_fold = _run_lopo(cfg, feat, seed=args.seed, verbose=True)
        elapsed = time.perf_counter() - t0
        pooled = _pool_metrics(per_fold)
        pooled["config"] = cfg["name"]
        pooled["elapsed_sec"] = round(elapsed, 1)
        pooled_rows.append(pooled)
        real_pooled_roc[cfg["name"]] = pooled["pooled_roc_auc"]
        # drop the raw arrays before persisting
        per_fold_rows.append(per_fold.drop(columns=["_y", "_p"]))
        print(
            f"  -> {cfg['name']}: pooled_ROC={pooled['pooled_roc_auc']:.4f} "
            f"pooled_PR={pooled['pooled_pr_auc']:.4f} "
            f"per-patient_ROC={pooled['per_patient_roc_mean']:.3f}"
            f"+/-{pooled['per_patient_roc_std']:.3f} "
            f"({elapsed:.1f}s)"
        )

    pd.concat(per_fold_rows).to_csv(out_dir / "lopo_v2_per_patient.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(out_dir / "lopo_v2_pooled.csv", index=False)
    print(f"\nWrote {out_dir / 'lopo_v2_per_patient.csv'}")
    print(f"Wrote {out_dir / 'lopo_v2_pooled.csv'}")

    if args.skip_permutation:
        print("\n--skip-permutation set; done.")
        return 0

    # ---- Patient-level permutation test --------------------------------
    print(f"\n[4/4] Patient-level permutation test (K={args.n_permutations}) ...")
    rng = np.random.default_rng(args.seed)
    perm_rows = []
    for cfg in CONFIGS:
        feat = feat_by_cfg[cfg["name"]]
        feature_cols = select_feature_columns(feat)
        users = sorted(feat[USER_COL].unique())
        print(f"  config={cfg['name']} (real pooled ROC = "
              f"{real_pooled_roc[cfg['name']]:.4f})")
        for k in range(args.n_permutations):
            t0 = time.perf_counter()
            permuted = _permute_labels_patient_block(feat, rng)
            feat_perm = feat.copy()
            feat_perm["target"] = permuted.astype(int).to_numpy()
            ys, ps = [], []
            for held_out in users:
                train_users = [u for u in users if u != held_out]
                y_t, p_t = _fit_predict_fold(
                    cfg, feat_perm, feature_cols, train_users, held_out,
                    target_col="target", seed=args.seed,
                )
                ys.append(y_t)
                ps.append(p_t)
            y_all = np.concatenate(ys)
            p_all = np.concatenate(ps)
            mask = ~np.isnan(p_all)
            roc = _safe_roc(y_all[mask], p_all[mask])
            pr = _safe_pr(y_all[mask], p_all[mask])
            dt = time.perf_counter() - t0
            perm_rows.append({
                "config": cfg["name"],
                "permutation": k,
                "pooled_roc_auc": roc,
                "pooled_pr_auc": pr,
                "elapsed_sec": round(dt, 2),
            })
            if (k + 1) % 5 == 0 or k == 0:
                print(f"    perm {k + 1:>3}/{args.n_permutations}: "
                      f"ROC={roc:.4f} PR={pr:.4f} ({dt:.1f}s)")

    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(out_dir / "lopo_v2_permutation.csv", index=False)
    print(f"\nWrote {out_dir / 'lopo_v2_permutation.csv'}")

    # Summary + p-values
    summary_rows = []
    for cfg in CONFIGS:
        sub = perm_df[perm_df["config"] == cfg["name"]]
        null_roc = sub["pooled_roc_auc"].dropna().to_numpy()
        real = real_pooled_roc[cfg["name"]]
        if len(null_roc) == 0 or np.isnan(real):
            p_value = float("nan")
        else:
            p_value = float((null_roc >= real).sum() + 1) / (len(null_roc) + 1)
        summary_rows.append({
            "config": cfg["name"],
            "real_pooled_roc": real,
            "null_n": len(null_roc),
            "null_mean": float(np.nanmean(null_roc)) if len(null_roc) else float("nan"),
            "null_std": float(np.nanstd(null_roc, ddof=1)) if len(null_roc) > 1 else float("nan"),
            "null_q05": float(np.nanquantile(null_roc, 0.05)) if len(null_roc) else float("nan"),
            "null_q50": float(np.nanquantile(null_roc, 0.50)) if len(null_roc) else float("nan"),
            "null_q95": float(np.nanquantile(null_roc, 0.95)) if len(null_roc) else float("nan"),
            "p_value": p_value,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "lopo_v2_permutation_summary.csv", index=False)
    print(f"Wrote {out_dir / 'lopo_v2_permutation_summary.csv'}\n")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
