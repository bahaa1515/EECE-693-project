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

DEFAULT_PRECISION_FLOOR = 0.20
DEFAULT_RECALL_FLOOR = 0.50
DEFAULT_OVERFIT_GAP = 0.20
DEFAULT_LOW_RECALL = 0.05


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


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """Return metrics with a prefix such as ``train_`` or ``val_``."""
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def tune_threshold_max_f1(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    default: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Pick a validation-only threshold that maximises F1.

    Returns ``(threshold, metrics_at_threshold)``.  AUC/Brier values remain
    probability metrics; precision/recall/F1/accuracy use the tuned threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float(default), compute_metrics(y_true, y_score, threshold=default)

    cuts = np.unique(np.concatenate([y_score, np.array([default])]))
    best_f1 = -1.0
    best_threshold = float(default)
    for threshold in cuts:
        pred = (y_score >= threshold).astype(int)
        if pred.sum() == 0:
            continue
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, compute_metrics(y_true, y_score, threshold=best_threshold)


def recall_at_precision_floor(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
) -> tuple[float, float]:
    """Return the best recall attainable while precision >= floor.

    The paired threshold is returned for reporting.  If no positive prediction
    can satisfy the floor, recall is 0 and threshold is NaN.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or y_true.sum() == 0:
        return float("nan"), float("nan")

    best_recall = -1.0
    best_threshold = float("nan")
    for threshold in np.unique(y_score):
        pred = (y_score >= threshold).astype(int)
        if pred.sum() == 0:
            continue
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        if precision >= precision_floor and recall > best_recall:
            best_recall = float(recall)
            best_threshold = float(threshold)
    if best_recall < 0:
        return 0.0, float("nan")
    return best_recall, best_threshold


def precision_at_recall_floor(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    recall_floor: float = DEFAULT_RECALL_FLOOR,
) -> tuple[float, float]:
    """Return the best precision attainable while recall >= floor."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or y_true.sum() == 0:
        return float("nan"), float("nan")

    best_precision = -1.0
    best_threshold = float("nan")
    for threshold in np.unique(y_score):
        pred = (y_score >= threshold).astype(int)
        if pred.sum() == 0:
            continue
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        if recall >= recall_floor and precision > best_precision:
            best_precision = float(precision)
            best_threshold = float(threshold)
    if best_precision < 0:
        return 0.0, float("nan")
    return best_precision, best_threshold


def add_selection_diagnostics(
    row: dict[str, object],
    *,
    precision_floor: float = DEFAULT_PRECISION_FLOOR,
    recall_floor: float = DEFAULT_RECALL_FLOOR,
    overfit_gap: float = DEFAULT_OVERFIT_GAP,
    low_recall: float = DEFAULT_LOW_RECALL,
) -> dict[str, object]:
    """Add gaps and overfitting flags to a metric row.

    Existing values are preserved.  Missing prerequisite values yield NaNs or
    False flags rather than raising, so older result tables remain readable.
    """
    out = dict(row)

    train_pr = pd.to_numeric(pd.Series([out.get("train_pr_auc")]), errors="coerce").iloc[0]
    val_pr = pd.to_numeric(pd.Series([out.get("val_pr_auc")]), errors="coerce").iloc[0]
    test_pr = pd.to_numeric(pd.Series([out.get("test_pr_auc")]), errors="coerce").iloc[0]
    val_recall = pd.to_numeric(pd.Series([out.get("val_recall")]), errors="coerce").iloc[0]
    val_f1 = pd.to_numeric(pd.Series([out.get("val_f1")]), errors="coerce").iloc[0]

    if pd.notna(train_pr) and pd.notna(val_pr):
        out["train_val_pr_auc_gap"] = float(train_pr - val_pr)
    else:
        out.setdefault("train_val_pr_auc_gap", float("nan"))
    if pd.notna(val_pr) and pd.notna(test_pr):
        out["val_test_pr_auc_gap"] = float(val_pr - test_pr)
    else:
        out.setdefault("val_test_pr_auc_gap", float("nan"))

    gap = pd.to_numeric(pd.Series([out.get("train_val_pr_auc_gap")]), errors="coerce").iloc[0]
    out["overfit_gap_flag"] = bool(pd.notna(gap) and gap > overfit_gap)
    out["low_val_event_detection_flag"] = bool(
        pd.notna(val_pr)
        and (
            (pd.notna(val_recall) and val_recall <= low_recall)
            or (pd.notna(val_f1) and val_f1 <= low_recall)
        )
    )
    out["precision_floor"] = precision_floor
    out["recall_floor"] = recall_floor
    return out


def select_best_by_pr_auc(
    trials: pd.DataFrame,
    score_col: str = "val_pr_auc",
    group_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Pick the best trial per group by validation PR-AUC.

    If ``group_cols`` is None, returns the single overall winner.
    Ties follow the project selection policy:
    validation recall at the fixed precision floor, lower train-validation
    PR-AUC gap, lower validation Brier, then simpler representation/model
    when complexity columns exist.
    """
    if trials.empty:
        return trials.copy()
    if score_col not in trials.columns:
        raise KeyError(f"Missing score column {score_col!r}")

    ranked = trials.copy()
    if "train_val_pr_auc_gap" not in ranked.columns:
        ranked = pd.DataFrame(
            add_selection_diagnostics(row)
            for row in ranked.to_dict(orient="records")
        )

    prefix = score_col[: -len("pr_auc")] if score_col.endswith("pr_auc") else "val_"
    candidate_cols: list[tuple[str, bool, float]] = [
        (score_col, False, float("-inf")),
        (f"{prefix}recall_at_precision_floor", False, float("-inf")),
        ("train_val_pr_auc_gap", True, float("inf")),
        (f"{prefix}brier", True, float("inf")),
        ("selection_complexity", True, float("inf")),
        ("representation_complexity", True, float("inf")),
        ("model_complexity", True, float("inf")),
        ("n_features", True, float("inf")),
        ("n_channels", True, float("inf")),
        ("val_roc_auc", False, float("-inf")),
        ("val_f1", False, float("-inf")),
    ]

    sort_cols: list[str] = []
    ascending: list[bool] = []
    for col, asc, fill in candidate_cols:
        if col in ranked.columns and col not in sort_cols:
            numeric = pd.to_numeric(ranked[col], errors="coerce")
            ranked[col] = numeric.fillna(fill)
            sort_cols.append(col)
            ascending.append(asc)
    ranked = ranked.sort_values(sort_cols, ascending=ascending, kind="mergesort")

    if group_cols is None:
        return ranked.head(1).reset_index(drop=True)
    return (
        ranked.groupby(list(group_cols), as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )


__all__ = [
    "METRIC_COLUMNS",
    "DEFAULT_PRECISION_FLOOR",
    "DEFAULT_RECALL_FLOOR",
    "DEFAULT_OVERFIT_GAP",
    "DEFAULT_LOW_RECALL",
    "PatientSplit",
    "make_patient_three_way_split",
    "split_feature_table",
    "compute_metrics",
    "prefix_metrics",
    "tune_threshold_max_f1",
    "recall_at_precision_floor",
    "precision_at_recall_floor",
    "add_selection_diagnostics",
    "select_best_by_pr_auc",
]
