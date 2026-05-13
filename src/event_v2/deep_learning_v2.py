"""Deep-learning v2: daily-level sequence models on event-episode samples.

Each sample is an ``L``-day window of multimodal daily-aggregated features
(``D`` channels).  We reuse the GRU/LSTM/RNN/CNN factories from
:mod:`src.deep_learning` but feed them ``L`` timesteps instead of 1440
minutes and disable subsampling.

This module purposefully does NOT touch :mod:`src.deep_learning`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.event_features import DATE_COL
from src.event_labels import USER_COL
from src.deep_learning import (
    _platt_scale,
    bootstrap_auc_ci,
    make_cnn,
    make_gru,
    make_lstm,
    make_rnn,
    set_seed,
    train_torch_model,
    predict_torch_model,
    _require_torch,
)
from . import OUTPUT_TABLES_V2
from .features_v2 import ALL_SENSOR_SOURCES, SOURCE_ZERO_IF_EMPTY
from .split_v2 import PatientSplit, compute_metrics


ARCH_FACTORIES = {
    "gru": make_gru,
    "lstm": make_lstm,
    "rnn": make_rnn,
    "cnn": make_cnn,
}


@dataclass
class SequenceDataset:
    X: np.ndarray  # (n_samples, L, D), float32
    y: np.ndarray  # (n_samples,), int
    users: np.ndarray  # (n_samples,), int
    targets: pd.Series
    feature_channels: list[str]
    input_length: int


def _daily_feature_columns(daily_table: pd.DataFrame) -> list[str]:
    return [
        col
        for col in daily_table.columns
        if col not in {USER_COL, DATE_COL}
        and pd.api.types.is_numeric_dtype(daily_table[col])
    ]


def build_sequence_dataset(
    sample_index: pd.DataFrame,
    daily_tables: dict[str, pd.DataFrame],
    sensor_sources: Iterable[str] | None = None,
) -> SequenceDataset:
    """Build an (N, L, D) array of daily features per sample.

    Days with no observation receive zero (if ``zero_if_empty=True`` for the
    source) or NaN otherwise; NaNs are then replaced with zero after
    standardisation (handled by the caller).
    """
    sources = tuple(s for s in ALL_SENSOR_SOURCES if (
        s in (sensor_sources or ALL_SENSOR_SOURCES)
    ))
    if not sources:
        raise ValueError("Need at least one sensor source")

    L_values = sample_index["input_length_days"].astype(int).unique()
    if len(L_values) != 1:
        raise ValueError(
            f"Mixed input_length_days in sample_index: {sorted(L_values)}"
        )
    L = int(L_values[0])

    # Gather channel lists per source (preserve order).
    channels_per_source: dict[str, list[str]] = {}
    for src in sources:
        if src not in daily_tables:
            raise KeyError(f"Missing daily table for source {src!r}")
        channels_per_source[src] = _daily_feature_columns(daily_tables[src])
    feature_channels: list[str] = []
    for src in sources:
        feature_channels.extend([f"{src}::{c}" for c in channels_per_source[src]])

    n = len(sample_index)
    D = len(feature_channels)
    X = np.full((n, L, D), np.nan, dtype=np.float32)

    # Pre-index daily tables by user for fast lookup.
    daily_by_user: dict[str, dict[int, pd.DataFrame]] = {}
    for src in sources:
        daily_by_user[src] = {
            int(u): grp.set_index(DATE_COL)
            for u, grp in daily_tables[src].groupby(USER_COL, observed=True)
        }

    for i, row in enumerate(sample_index.itertuples(index=False)):
        u = int(getattr(row, USER_COL))
        start = int(row.window_start_day)
        end = int(row.window_end_day)
        days = list(range(start, end + 1))
        col_offset = 0
        for src in sources:
            chans = channels_per_source[src]
            zero_if_empty = SOURCE_ZERO_IF_EMPTY[src]
            user_frame = daily_by_user[src].get(u)
            if user_frame is None:
                if zero_if_empty:
                    X[i, :, col_offset : col_offset + len(chans)] = 0.0
                col_offset += len(chans)
                continue
            sub = user_frame.reindex(days)[chans]
            arr = sub.to_numpy(dtype=np.float32)
            if zero_if_empty:
                arr = np.nan_to_num(arr, nan=0.0)
            X[i, :, col_offset : col_offset + len(chans)] = arr
            col_offset += len(chans)

    y = sample_index["target"].astype(int).to_numpy()
    users = sample_index[USER_COL].astype(int).to_numpy()
    return SequenceDataset(
        X=X,
        y=y,
        users=users,
        targets=sample_index["target"].astype(int),
        feature_channels=feature_channels,
        input_length=L,
    )


def _standardize_train(X: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    train = X[train_mask]
    mean = np.nanmean(train, axis=(0, 1)).astype(np.float32)
    std = np.nanstd(train, axis=(0, 1)).astype(np.float32)
    std[std == 0] = 1.0
    scaled = (X - mean) / std
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def split_sequence_dataset(
    dataset: SequenceDataset,
    split: PatientSplit,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return (X_dict, y_dict) with 'train', 'val', 'test' subsets."""
    train_mask = np.isin(dataset.users, split.train_users)
    val_mask = np.isin(dataset.users, split.val_users)
    test_mask = np.isin(dataset.users, split.test_users)
    X_scaled = _standardize_train(dataset.X, train_mask)
    return (
        {
            "train": X_scaled[train_mask],
            "val": X_scaled[val_mask],
            "test": X_scaled[test_mask],
        },
        {
            "train": dataset.y[train_mask],
            "val": dataset.y[val_mask],
            "test": dataset.y[test_mask],
        },
    )


def _make_loaders(X_dict, y_dict, batch_size: int):
    torch, _, DataLoader, TensorDataset = _require_torch()

    def _ds(name: str):
        return TensorDataset(
            torch.from_numpy(X_dict[name]),
            torch.from_numpy(y_dict[name].astype(np.float32)),
        )

    return (
        DataLoader(_ds("train"), batch_size=batch_size, shuffle=True),
        DataLoader(_ds("val"), batch_size=batch_size, shuffle=False),
        DataLoader(_ds("test"), batch_size=batch_size, shuffle=False),
    )


def _build_arch(arch: str, input_dim: int, hp: dict):
    factory = ARCH_FACTORIES[arch]
    common = dict(input_dim=input_dim, subsample=1)
    if arch == "cnn":
        common["dropout"] = hp.get("dropout", 0.3)
    else:
        common.update(
            hidden_dim=hp.get("hidden_dim", 64),
            num_layers=hp.get("num_layers", 2),
            dropout=hp.get("dropout", 0.3),
        )
    return factory(**common)


def train_one_arch(
    arch: str,
    X_dict: dict[str, np.ndarray],
    y_dict: dict[str, np.ndarray],
    hp: dict,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 32,
    patience: int = 5,
    learning_rate: float = 5e-4,
    apply_platt: bool = True,
) -> dict[str, float | str]:
    """Train one model and return a metrics dict (val + test)."""
    set_seed(seed)
    input_dim = X_dict["train"].shape[2]
    model = _build_arch(arch, input_dim=input_dim, hp=hp)
    train_loader, val_loader, test_loader = _make_loaders(X_dict, y_dict, batch_size=batch_size)

    model, _hist, meta = train_torch_model(
        model,
        train_loader,
        val_loader,
        y_dict["train"],
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
    )
    p_val = predict_torch_model(model, val_loader)
    p_test = predict_torch_model(model, test_loader)
    if apply_platt and len(np.unique(y_dict["val"])) == 2:
        p_test_cal = _platt_scale(y_dict["val"], p_val, p_test)
    else:
        p_test_cal = p_test

    val_metrics = compute_metrics(y_dict["val"], p_val)
    test_metrics = compute_metrics(y_dict["test"], p_test_cal)
    return {
        "arch": arch,
        "seed": seed,
        "best_epoch": meta["best_epoch"],
        "best_val_loss": meta["best_val_loss"],
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "p_val": p_val,
        "p_test": p_test_cal,
    }


__all__ = [
    "ARCH_FACTORIES",
    "SequenceDataset",
    "build_sequence_dataset",
    "split_sequence_dataset",
    "train_one_arch",
]
