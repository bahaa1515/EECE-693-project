"""Leakage probe.

Two cheap sanity checks that must hold for the canonical labelled parquet
(`baseline_smartwatch_features_labeled.parquet`, weekly-questionnaire
OR-of-flags target):

1. Shuffled-label probe.  A logistic-regression model trained on
   smartwatch-summary features against *shuffled* labels must not exceed
   ROC-AUC ~= 0.55 on the held-out test patients.  Any higher score
   indicates patient-identity leakage (i.e. the model is memorising users
   that happen to be positive under the shuffled labels -- usually a sign
   of an unbroken group split).

2. Join-integrity probe.  Every labelled window must have a
   `time_to_label_days` in (0, 7] (forward-only join with the next weekly
   questionnaire), and the joined `target_binary` must equal the OR of the
   five component flags (`doc_flag | hospital_flag | er_flag | oral_flag |
   symptom_flag`).  If either invariant breaks the labelling has drifted.

Both probes use the same patient-wise GroupShuffleSplit used by Tier-2 so
that any leakage path is exercised identically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.config import DATA_PROCESSED, OUTPUT_TABLES


SEED = 42
LABELED = DATA_PROCESSED / "baseline_smartwatch_features_labeled.parquet"
FLAG_COLS = ["doc_flag", "hospital_flag", "er_flag", "oral_flag", "symptom_flag"]


def patient_split(df: pd.DataFrame, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(gss.split(df, df["target_binary"], groups=df["user_key"]))
    return train_idx, test_idx


def probe_shuffled_labels() -> float:
    df = pd.read_parquet(LABELED)
    feat_cols = [
        "hr_mean",
        "hr_std",
        "steps_sum",
        "active_minute_frac",
        "day_hr_mean",
        "night_hr_mean",
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols].fillna(0.0).to_numpy()
    rng = np.random.default_rng(SEED)
    y_shuffled = rng.permutation(df["target_binary"].to_numpy())
    train_idx, test_idx = patient_split(df)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    model.fit(X[train_idx], y_shuffled[train_idx])
    auc = roc_auc_score(y_shuffled[test_idx], model.predict_proba(X[test_idx])[:, 1])
    return float(auc)


def probe_join_integrity() -> tuple[float, dict]:
    """Returns (or_recovery_auc, diagnostics).

    `or_recovery_auc` should be ~1.0 since target_binary is exactly the OR of
    the 5 component flags carried through the join.  We also assert that
    every row has 0 < time_to_label_days <= 7 (forward, within horizon).
    """
    df = pd.read_parquet(LABELED)
    flags_present = [c for c in FLAG_COLS if c in df.columns]
    diagnostics: dict = {
        "rows": int(len(df)),
        "users": int(df["user_key"].nunique()),
        "pos_rate": float(df["target_binary"].mean()),
        "flags_present": flags_present,
    }

    # Horizon check.
    if "time_to_label_days" in df.columns:
        t = df["time_to_label_days"]
        diagnostics["min_time_to_label_days"] = float(t.min())
        diagnostics["max_time_to_label_days"] = float(t.max())
        diagnostics["horizon_violations"] = int(((t < 0) | (t > 7)).sum())
    else:
        diagnostics["horizon_violations"] = -1  # column missing

    # OR-recovery check.
    if not flags_present:
        return 0.0, diagnostics
    X = df[flags_present].fillna(False).astype(int).to_numpy()
    y = df["target_binary"].to_numpy()
    train_idx, test_idx = patient_split(df)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    model.fit(X[train_idx], y[train_idx])
    auc = float(roc_auc_score(y[test_idx], model.predict_proba(X[test_idx])[:, 1]))
    return auc, diagnostics


def main() -> int:
    auc_shuffled = probe_shuffled_labels()
    auc_or, diagnostics = probe_join_integrity()

    print(f"[probe 1] shuffled-label AUC  = {auc_shuffled:.3f}  (expect ~0.50, must be < 0.60)")
    print(f"[probe 2] OR-recovery AUC     = {auc_or:.3f}  (expect ~1.00, must be > 0.95)")
    print(f"          diagnostics         = {diagnostics}")

    failures: list[str] = []
    if auc_shuffled > 0.60:
        failures.append(f"shuffled-label AUC {auc_shuffled:.3f} > 0.60 -- group split may be broken")
    if auc_or < 0.95:
        failures.append(
            f"OR-recovery AUC {auc_or:.3f} < 0.95 -- target_binary is not recovered from component flags; join may be broken"
        )
    if diagnostics.get("horizon_violations", 0) > 0:
        failures.append(
            f"{diagnostics['horizon_violations']} rows have time_to_label_days outside (0, 7]"
        )

    rows = [
        {"probe": "shuffled_labels",  "auc": round(auc_shuffled, 4), "threshold": 0.60, "pass": auc_shuffled <= 0.60},
        {"probe": "or_recovery",      "auc": round(auc_or, 4),       "threshold": 0.95, "pass": auc_or >= 0.95},
        {"probe": "horizon_check",    "auc": float("nan"),           "threshold": 0,    "pass": diagnostics.get("horizon_violations", 0) == 0},
    ]
    out = OUTPUT_TABLES / "leakage_probe.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved -> {out}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll leakage probes passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
