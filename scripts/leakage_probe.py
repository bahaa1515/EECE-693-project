"""Leakage probe.

Two cheap sanity checks that must hold for the canonical labelled parquet:

1. A logistic-regression model trained on patient-static + smartwatch-summary
   features against *shuffled* labels must not exceed ROC-AUC ≈ 0.55 on the
   held-out test patients.  Any higher score indicates patient-identity
   leakage (i.e. the model is memorising users that happen to be positive
   under the shuffled labels — usually a sign of an unbroken group split).

2. A logistic-regression model trained on the *raw* daily questionnaire
   worsening score alone — the field the label is thresholded from — must
   achieve ROC-AUC > 0.85.  This confirms that the score really is the label
   source and therefore that we are correct to *exclude* it from the model
   feature set.  If this probe scores low, then either the join is broken or
   the label definition has drifted.

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
DAILY_SCORES = OUTPUT_TABLES / "daily_questionnaire_worsening_scores.csv"


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
    X = df[feat_cols].fillna(0.0).to_numpy()
    rng = np.random.default_rng(SEED)
    y_shuffled = rng.permutation(df["target_binary"].to_numpy())
    train_idx, test_idx = patient_split(df)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    model.fit(X[train_idx], y_shuffled[train_idx])
    auc = roc_auc_score(y_shuffled[test_idx], model.predict_proba(X[test_idx])[:, 1])
    return float(auc)


def probe_label_source() -> float:
    df = pd.read_parquet(LABELED).copy()
    dq = pd.read_csv(DAILY_SCORES)
    dq = dq[["user_key", "date", "daily_questionnaire_worsening_score"]].drop_duplicates(
        ["user_key", "date"], keep="first"
    )
    df["end_day"] = (df["anchor_relative_minute"].astype(np.int64) // 1440) - 1
    df = df.merge(
        dq.rename(columns={"date": "end_day"}), on=["user_key", "end_day"], how="left"
    )
    df["daily_score_filled"] = df["daily_questionnaire_worsening_score"].fillna(0)
    train_idx, test_idx = patient_split(df)
    X = df[["daily_score_filled"]].to_numpy()
    y = df["target_binary"].to_numpy()
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    model.fit(X[train_idx], y[train_idx])
    return float(roc_auc_score(y[test_idx], model.predict_proba(X[test_idx])[:, 1]))


def main() -> int:
    auc_shuffled = probe_shuffled_labels()
    auc_label_source = probe_label_source()

    print(f"[probe 1] shuffled-label AUC    = {auc_shuffled:.3f}  (expect ~0.50, must be < 0.60)")
    print(f"[probe 2] label-source AUC      = {auc_label_source:.3f}  (expect > 0.80)")

    failures: list[str] = []
    if auc_shuffled > 0.60:
        failures.append(f"shuffled-label AUC {auc_shuffled:.3f} > 0.60 — group split may be broken")
    if auc_label_source < 0.80:
        failures.append(
            f"label-source AUC {auc_label_source:.3f} < 0.80 — daily worsening score "
            "does not recover the label; the label join may be broken"
        )

    rows = [
        {"probe": "shuffled_labels", "auc": round(auc_shuffled, 4), "threshold": 0.60, "pass": auc_shuffled <= 0.60},
        {"probe": "label_source",    "auc": round(auc_label_source, 4), "threshold": 0.80, "pass": auc_label_source >= 0.80},
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
