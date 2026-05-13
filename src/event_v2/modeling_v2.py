"""Tabular modeling helpers for v2 (HPO-friendly).

Provides:
* :func:`select_feature_columns` – numeric, non-metadata columns.
* :func:`build_model` – factory returning a sklearn ``Pipeline`` for a
  given (algo, hyperparameter dict, n_pos, n_neg) triple.

The HPO grid itself lives in ``scripts/v2/tune_tabular_v2.py``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.event_labels import USER_COL


METADATA_COLUMNS = {
    USER_COL,
    "sample_anchor_day",
    "window_start_day",
    "window_end_day",
    "event_onset_day",
    "event_end_day",
    "target",
    "sample_type",
    "label_strategy",
    "threshold",
    "input_length_days",
    "washout_days",
}


def select_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    return [
        col
        for col in feature_table.columns
        if col not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(feature_table[col])
    ]


def build_model(
    algo: str,
    params: dict[str, Any],
    n_pos: int,
    n_neg: int,
    random_state: int = 42,
) -> Pipeline:
    """Construct a v2 model pipeline.

    ``algo`` is one of ``"lr"``, ``"rf"``, ``"xgb"``.
    ``params`` is the algorithm-specific hyperparameter dict.
    """
    if algo == "lr":
        clf = LogisticRegression(
            C=params.get("C", 1.0),
            class_weight=params.get("class_weight"),
            max_iter=params.get("max_iter", 2000),
            solver=params.get("solver", "lbfgs"),
            random_state=random_state,
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    if algo == "rf":
        clf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth"),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            class_weight=params.get("class_weight"),
            random_state=random_state,
            n_jobs=params.get("n_jobs", 1),
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", clf),
            ]
        )

    if algo == "xgb":
        from xgboost import XGBClassifier  # local import: optional dep

        spw_mode = params.get("scale_pos_weight", "auto")
        if spw_mode == "auto":
            scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
        else:
            scale_pos_weight = float(spw_mode)
        clf = XGBClassifier(
            n_estimators=params.get("n_estimators", 250),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=params.get("n_jobs", 1),
            tree_method=params.get("tree_method", "hist"),
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", clf),
            ]
        )

    raise ValueError(f"Unknown algo: {algo!r}")


__all__ = ["METADATA_COLUMNS", "select_feature_columns", "build_model"]
