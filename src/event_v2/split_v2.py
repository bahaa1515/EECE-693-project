"""Patient-grouped train/val/test splits and metric helpers (v2).

* :func:`make_patient_three_way_split` returns disjoint train/val/test patient
  ID lists at fixed ratios (default 70/10/20), reproducible by seed.
* :func:`split_feature_table` applies a split to a feature table.
* :func:`compute_metrics` returns the 7-metric dict we report in v2.
* :func:`select_best_by_pr_auc` picks the winning config from an HPO trial
  table on the validation PR-AUC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.event_labels import USER_COL


METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
)


@dataclass(frozen=True)
class PatientSplit:
    train_users: tuple[int, ...]
    val_users: tuple[int, ...]
    test_users: tuple[int, ...]
    seed: int
    val_frac: float
    test_frac: float

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "val_frac": self.val_frac,
            "test_frac": self.test_frac,
            "train_users": list(self.train_users),
            "val_users": list(self.val_users),
            "test_users": list(self.test_users),
        }


def make_patient_three_way_split(
    groups: Sequence[int] | np.ndarray | pd.Series,
    val_frac: float = 0.15,
    test_frac: float = 0.20,
    seed: int = 42,
) -> PatientSplit:
    """Disjoint patient-level train/val/test split.

    Uses a deterministic permutation under ``seed`` and assigns the first
    ``ceil(n_users * test_frac)`` users to test, then ``ceil(n_users *
    val_frac)`` to val, and the remainder to train.  Guarantees at least
    one user in each split when ``n_users >= 3``.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1); got {val_frac}")
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1); got {test_frac}")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    unique_users = np.array(sorted({int(u) for u in groups}))
    n_users = len(unique_users)
    if n_users < 3:
        raise ValueError(f"Need at least 3 users for a 3-way split; got {n_users}")

    rng = np.random.default_rng(seed)
    permuted = unique_users[rng.permutation(n_users)]

    n_test = max(1, int(np.ceil(n_users * test_frac)))
    n_val = max(1, int(np.ceil(n_users * val_frac)))
    if n_test + n_val >= n_users:
        n_val = max(1, n_users - n_test - 1)

    test_users = tuple(int(u) for u in permuted[:n_test])
    val_users = tuple(int(u) for u in permuted[n_test : n_test + n_val])
    train_users = tuple(int(u) for u in permuted[n_test + n_val :])

    return PatientSplit(
        train_users=train_users,
        val_users=val_users,
        test_users=test_users,
        seed=seed,
        val_frac=val_frac,
        test_frac=test_frac,
    )


def split_feature_table(
    feature_table: pd.DataFrame,
    split: PatientSplit,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) sub-tables based on patient assignment."""
    if USER_COL not in feature_table.columns:
        raise KeyError(f"{USER_COL!r} not in feature_table columns")
    users = feature_table[USER_COL].astype(int)
    train = feature_table[users.isin(split.train_users)].copy()
    val = feature_table[users.isin(split.val_users)].copy()
    test = feature_table[users.isin(split.test_users)].copy()
    return train, val, test


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series | None,
    y_pred: np.ndarray | pd.Series | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return the 7 v2 headline metrics as a dict.

    Falls back to NaN for AUCs when only one class is present in y_true.
    """
    y_true = np.asarray(y_true).astype(int)
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        if y_pred is None:
            y_pred = (y_score >= threshold).astype(int)
    if y_pred is None:
        raise ValueError("Must provide y_score or y_pred")
    y_pred = np.asarray(y_pred).astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
        "brier": float("nan"),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        metrics["brier"] = float(brier_score_loss(y_true, y_score))
    return metrics


def select_best_by_pr_auc(
    trials: pd.DataFrame,
    score_col: str = "val_pr_auc",
    group_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Pick the best trial per group by validation PR-AUC.

    If ``group_cols`` is None, returns the single overall winner.
    Ties are broken by ``val_roc_auc`` then ``val_f1`` when those columns
    exist; otherwise by first occurrence.
    """
    if trials.empty:
        return trials.copy()
    if score_col not in trials.columns:
        raise KeyError(f"Missing score column {score_col!r}")

    sort_cols = [score_col]
    for fallback in ("val_roc_auc", "val_f1"):
        if fallback in trials.columns:
            sort_cols.append(fallback)
    ranked = trials.sort_values(sort_cols, ascending=False, kind="mergesort")

    if group_cols is None:
        return ranked.head(1).reset_index(drop=True)
    return (
        ranked.groupby(list(group_cols), as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )


__all__ = [
    "METRIC_COLUMNS",
    "PatientSplit",
    "make_patient_three_way_split",
    "split_feature_table",
    "compute_metrics",
    "select_best_by_pr_auc",
]
