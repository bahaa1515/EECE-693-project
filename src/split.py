from __future__ import annotations

from collections.abc import Callable

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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .features import FEATURE_COLUMNS


def get_xy_groups(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "target_binary",
    group_col: str = "user_key",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    feature_cols = feature_cols or FEATURE_COLUMNS
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return df[feature_cols].copy(), df[target_col].astype(int).copy(), df[group_col].copy()


def patient_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return (
        X.iloc[train_idx].copy(),
        X.iloc[test_idx].copy(),
        y.iloc[train_idx].copy(),
        y.iloc[test_idx].copy(),
        groups.iloc[train_idx].copy(),
        groups.iloc[test_idx].copy(),
    )


def classification_metrics(
    name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray | None = None,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is None:
        row.update({"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan})
    else:
        row.update(
            {
                "roc_auc": roc_auc_score(y_true, y_score),
                "pr_auc": average_precision_score(y_true, y_score),
                "brier": brier_score_loss(y_true, y_score),
            }
        )
    return row


def evaluate_sklearn_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float | str], np.ndarray, np.ndarray | None]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return classification_metrics(name, y_test, y_pred, y_score), y_pred, y_score


def grouped_cv(
    name: str,
    make_model: Callable[[pd.Series], object],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    rows = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups)):
        model = make_model(y.iloc[train_idx])
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = model.predict(X.iloc[test_idx])
        y_score = (
            model.predict_proba(X.iloc[test_idx])[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        row = classification_metrics(name, y.iloc[test_idx], y_pred, y_score)
        row["fold"] = fold
        rows.append(row)
    columns = ["model", "fold", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
    return pd.DataFrame(rows)[columns]
