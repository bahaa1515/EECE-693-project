from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import OUTPUT_TABLES
from .event_labels import USER_COL


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


def safe_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "brier": np.nan,
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
        metrics["brier"] = brier_score_loss(y_true, y_score)
    return metrics


def make_models(y_train: pd.Series, random_state: int = 42) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "Logistic Regression balanced": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest balanced": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        positives = int(y_train.sum())
        negatives = int(len(y_train) - positives)
        scale_pos_weight = negatives / positives if positives else 1.0
        models["XGBoost balanced"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    except ImportError:
        models["Gradient Boosting fallback"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    return models


def patient_split_indices(
    feature_table: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    groups = feature_table[USER_COL]
    y = feature_table["target"].astype(int)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(feature_table, y, groups=groups))
    return train_idx, test_idx


def train_evaluate_feature_table(
    feature_table: pd.DataFrame,
    threshold: int,
    input_length_days: int,
    washout_days: int,
    random_state: int = 42,
) -> pd.DataFrame:
    if feature_table.empty or feature_table["target"].nunique() < 2:
        return pd.DataFrame()

    feature_cols = select_feature_columns(feature_table)
    train_idx, test_idx = patient_split_indices(feature_table, random_state=random_state)
    train = feature_table.iloc[train_idx].copy()
    test = feature_table.iloc[test_idx].copy()
    X_train = train[feature_cols]
    y_train = train["target"].astype(int)
    X_test = test[feature_cols]
    y_test = test["target"].astype(int)

    rows = []
    for model_name, model in make_models(y_train, random_state=random_state).items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        row = safe_classification_metrics(y_test.to_numpy(), y_pred, y_score)
        row.update(
            {
                "model": model_name,
                "label_strategy": "questionnaire_event_episode_labeling",
                "threshold": threshold,
                "input_length_days": input_length_days,
                "washout_days": washout_days,
                "n_features": len(feature_cols),
                "n_train": len(train),
                "n_test": len(test),
                "train_users": train[USER_COL].nunique(),
                "test_users": test[USER_COL].nunique(),
                "train_positive": int(y_train.sum()),
                "test_positive": int(y_test.sum()),
            }
        )
        rows.append(row)

    columns = [
        "label_strategy",
        "threshold",
        "input_length_days",
        "washout_days",
        "model",
        "n_features",
        "n_train",
        "n_test",
        "train_users",
        "test_users",
        "train_positive",
        "test_positive",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier",
    ]
    return pd.DataFrame(rows)[columns]


def train_selected_feature_tables(
    feature_paths: dict[tuple[int, int, int], Path],
    selected_keys: list[tuple[int, int, int]] | None = None,
    output_dir: Path = OUTPUT_TABLES,
    random_state: int = 42,
) -> pd.DataFrame:
    selected_keys = selected_keys or sorted(feature_paths)
    result_frames = []
    for key in selected_keys:
        threshold, input_length_days, washout_days = key
        path = feature_paths[key]
        feature_table = pd.read_parquet(path)
        result = train_evaluate_feature_table(
            feature_table=feature_table,
            threshold=threshold,
            input_length_days=input_length_days,
            washout_days=washout_days,
            random_state=random_state,
        )
        if not result.empty:
            result_frames.append(result)

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        output_dir / "model_results_questionnaire_event_episode_labeling.csv",
        index=False,
    )
    return results
