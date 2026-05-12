from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from .config import (
    DATA_INTERIM,
    DATA_PROCESSED,
    OUTPUT_TABLES,
    AAMOS_SMARTINHALER_FILE,
    AAMOS_PEAKFLOW_FILE,
    AAMOS_PATIENT_INFO_FILE,
    AAMOS_WEEKLY_FILE,
)
from .features import FEATURE_COLUMNS, WINDOW_MINUTES
from .split import classification_metrics, get_xy_groups, patient_train_test_split


SEQUENCE_COLUMNS = ["hr", "activity_type", "intensity", "steps"]


@dataclass
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_users: list[int]
    val_users: list[int]
    test_users: list[int]


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs and request deterministic kernels where possible.

    Determinism notes
    -----------------
    - ``PYTHONHASHSEED`` is set so that pandas/sklearn hashing-based ops are
      stable.  This must be set *before* the Python process starts to take
      full effect, but exporting it from inside the process is a no-op for
      already-imported modules; we set it anyway as a safety net.
    - ``CUBLAS_WORKSPACE_CONFIG`` is required for deterministic CUDA matmuls
      under ``torch.use_deterministic_algorithms(True)``.
    - ``torch.use_deterministic_algorithms`` is called with ``warn_only=True``
      so a missing deterministic kernel downgrades to a warning instead of
      crashing on unsupported ops.
    """
    import os

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def build_sequence_array(
    labeled: pd.DataFrame,
    clean: pd.DataFrame,
    sequence_columns: list[str] | None = None,
    window_minutes: int = WINDOW_MINUTES,
) -> np.ndarray:
    """Build one 24-hour minute-level sequence per labeled feature row."""
    sequence_columns = sequence_columns or SEQUENCE_COLUMNS
    clean = clean.copy()
    clean["activity_type"] = pd.to_numeric(clean["activity_type"], errors="coerce")
    clean["intensity"] = pd.to_numeric(clean["intensity"], errors="coerce")
    clean["steps"] = pd.to_numeric(clean["steps"], errors="coerce")
    clean["hr"] = pd.to_numeric(clean["hr"], errors="coerce")

    user_frames = {
        user: group.sort_values("relative_minute").set_index("relative_minute")[
            sequence_columns
        ]
        for user, group in clean.groupby("user_key")
    }

    sequences = np.empty(
        (len(labeled), window_minutes, len(sequence_columns)), dtype=np.float32
    )
    for i, row in enumerate(labeled.itertuples(index=False)):
        user_frame = user_frames[int(row.user_key)]
        anchor = int(row.anchor_relative_minute)
        minutes = np.arange(anchor - window_minutes, anchor, dtype=np.int64)
        values = user_frame.reindex(minutes).to_numpy(dtype=np.float32)
        sequences[i] = values
    return sequences


def standardize_by_train(
    sequences: np.ndarray, train_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = sequences[train_idx]
    mean = np.nanmean(train, axis=(0, 1)).astype(np.float32)
    std = np.nanstd(train, axis=(0, 1)).astype(np.float32)
    std[std == 0] = 1.0
    scaled = (sequences - mean) / std
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return scaled, mean, std


def make_patient_splits(
    labeled: pd.DataFrame,
    test_size: float = 0.25,
    val_size: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = labeled["user_key"]
    y = labeled["target_binary"].astype(int)
    outer = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_val_idx, test_idx = next(outer.split(labeled, y, groups=groups))

    train_val = labeled.iloc[train_val_idx]
    inner = GroupShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state + 1
    )
    inner_train, inner_val = next(
        inner.split(
            train_val,
            train_val["target_binary"].astype(int),
            groups=train_val["user_key"],
        )
    )
    train_idx = train_val_idx[inner_train]
    val_idx = train_val_idx[inner_val]
    return train_idx, val_idx, test_idx


def prepare_sequence_data(
    labeled_path: Path = DATA_PROCESSED / "baseline_smartwatch_features_labeled.parquet",
    clean_path: Path = DATA_INTERIM / "smartwatch_cleaned.parquet",
    random_state: int = 42,
) -> SplitData:
    labeled = pd.read_parquet(labeled_path).reset_index(drop=True)
    clean = pd.read_parquet(clean_path)
    train_idx, val_idx, test_idx = make_patient_splits(labeled, random_state=random_state)
    sequences = build_sequence_array(labeled, clean)
    sequences, _, _ = standardize_by_train(sequences, train_idx)
    y = labeled["target_binary"].astype(int).to_numpy()

    return SplitData(
        X_train=sequences[train_idx],
        X_val=sequences[val_idx],
        X_test=sequences[test_idx],
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_users=sorted(labeled.iloc[train_idx]["user_key"].unique().astype(int).tolist()),
        val_users=sorted(labeled.iloc[val_idx]["user_key"].unique().astype(int).tolist()),
        test_users=sorted(labeled.iloc[test_idx]["user_key"].unique().astype(int).tolist()),
    )


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for Tier-2 models. Install it with `pip install torch`."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def make_loaders(split: SplitData, batch_size: int = 64):
    torch, _, DataLoader, TensorDataset = _require_torch()
    train_ds = TensorDataset(
        torch.from_numpy(split.X_train), torch.from_numpy(split.y_train.astype(np.float32))
    )
    val_ds = TensorDataset(
        torch.from_numpy(split.X_val), torch.from_numpy(split.y_val.astype(np.float32))
    )
    test_ds = TensorDataset(
        torch.from_numpy(split.X_test), torch.from_numpy(split.y_test.astype(np.float32))
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def make_gru(
    input_dim: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    subsample: int = 5,
):
    torch_mod, nn, _, _ = _require_torch()

    class GRUClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self._subsample = subsample
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, 1)
            self._init_weights()

        def _init_weights(self) -> None:
            """Orthogonal init for recurrent weights, Xavier for input weights.

            Default PyTorch GRU init (uniform [-1/sqrt(H), 1/sqrt(H)]) leaves a
            non-trivial fraction of seeds in a near-saturated regime that fails
            to escape the prior — this is the source of the train-time
            convergence collapse we observed across seeds.  Orthogonal recurrent
            weights stabilise BPTT signal flow and dramatically reduce seed
            sensitivity (see Saxe et al. 2014; Le et al. 2015).
            """
            for name, param in self.gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # Set forget-gate bias to 1 for the reset gate slice;
                    # PyTorch concatenates (r, z, n) for GRU bias layout.
                    H = param.shape[0] // 3
                    param.data[H : 2 * H].fill_(1.0)
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

        def forward(self, x):
            x = x[:, :: self._subsample, :]
            _, hidden = self.gru(x)
            return self.fc(hidden[-1]).squeeze(1)

    return GRUClassifier()


def make_cnn(input_dim: int = 4, dropout: float = 0.3, subsample: int = 5):
    _, nn, _, _ = _require_torch()

    class CNNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(input_dim, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(128, 1)

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.features(x)
            x = x.mean(dim=-1)
            x = self.dropout(x)
            return self.fc(x).squeeze(1)

    return CNNClassifier()


def train_torch_model(
    model,
    train_loader,
    val_loader,
    y_train: np.ndarray,
    epochs: int = 20,
    patience: int = 5,
    learning_rate: float = 5e-4,
    grad_clip: float | None = 1.0,
    device: str | None = None,
):
    torch, nn, _, _ = _require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pos = max(float((y_train == 1).sum()), 1.0)
    neg = max(float((y_train == 0).sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_losses.append(float(criterion(model(xb), yb).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), {"best_epoch": best_epoch, "best_val_loss": best_val}


def predict_torch_model(model, loader, device: str | None = None) -> np.ndarray:
    torch, _, _, _ = _require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    scores = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ROC-AUC.

    When ``groups`` is supplied, performs a *patient-cluster* bootstrap:
    resample whole patients (clusters) with replacement, then compute AUC
    over the concatenated windows of the sampled patients.  This is the
    statistically correct CI for a study with strong intra-patient
    correlation (95.8% window overlap in our case) and yields much wider
    — and honest — intervals than the naive per-window bootstrap.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    aucs: list[float] = []

    if groups is None:
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            aucs.append(float(roc_auc_score(y_true[idx], y_prob[idx])))
    else:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        index_by_group = {g: np.where(groups == g)[0] for g in unique_groups}
        n_groups = len(unique_groups)
        for _ in range(n_boot):
            sampled = rng.choice(unique_groups, size=n_groups, replace=True)
            idx = np.concatenate([index_by_group[g] for g in sampled])
            if len(np.unique(y_true[idx])) < 2:
                continue
            aucs.append(float(roc_auc_score(y_true[idx], y_prob[idx])))

    if not aucs:
        return float("nan"), float("nan")
    lo = float(np.percentile(aucs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(aucs, (1 + ci) / 2 * 100))
    return lo, hi


def train_evaluate_sequence_models(
    split: SplitData,
    epochs: int = 20,
    batch_size: int = 64,
    patience: int = 5,
    seed: int = 42,
    test_groups: np.ndarray | None = None,
    apply_platt: bool = True,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Train Tier-2 sequence models and evaluate with cluster-bootstrap CIs.

    Notes
    -----
    * A single ``seed`` is used for *both* models.  The previous version
      hard-coded different per-model seeds (8 for GRU, 43 for CNN) that were
      selected post-hoc — see issue #2 in the November audit.  The combination
      of orthogonal recurrent init + gradient clipping + lr=5e-4 in
      ``train_torch_model`` should remove the seed-sensitivity that motivated
      that workaround.
    * Platt scaling is fit on the validation probabilities and applied to the
      test probabilities before computing metrics, mirroring Tier-3.
    * If ``test_groups`` is supplied, the bootstrap CI is computed by
      resampling whole patients (cluster bootstrap).
    """
    train_loader, val_loader, test_loader = make_loaders(split, batch_size=batch_size)
    rows = []
    scores: dict[str, np.ndarray] = {}

    model_configs = [
        ("GRU (2-layer)", make_gru),
        ("CNN (3-layer)", make_cnn),
    ]

    for name, factory in model_configs:
        set_seed(seed)  # deterministic weight init per run
        model, history, meta = train_torch_model(
            factory(),
            train_loader,
            val_loader,
            split.y_train,
            epochs=epochs,
            patience=patience,
        )
        p_val = predict_torch_model(model, val_loader)
        p_test_raw = predict_torch_model(model, test_loader)
        if apply_platt:
            score = _platt_scale(split.y_val, p_val, p_test_raw)
        else:
            score = p_test_raw
        pred = (score >= 0.5).astype(int)
        row = classification_metrics(name, split.y_test, pred, score)
        row["best_epoch"] = meta["best_epoch"]
        row["best_val_loss"] = meta["best_val_loss"]
        row["seed"] = seed
        row["calibration"] = "platt" if apply_platt else "raw"
        ci_lo, ci_hi = bootstrap_auc_ci(split.y_test, score, groups=test_groups)
        row["auc_ci_lo"] = round(ci_lo, 3)
        row["auc_ci_hi"] = round(ci_hi, 3)
        rows.append(row)
        scores[name] = score
        history.to_csv(
            OUTPUT_TABLES / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_history.csv",
            index=False,
        )

    return pd.DataFrame(rows), scores


def train_rf_for_ensemble(
    labeled: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict[str, float | str], np.ndarray]:
    X, y, _ = get_xy_groups(labeled, FEATURE_COLUMNS)
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    score = model.predict_proba(X_test)[:, 1]
    pred = (score >= 0.5).astype(int)
    return classification_metrics("Random Forest", y_test, pred, score), score


def add_ensemble_rows(
    rows: pd.DataFrame,
    split: SplitData,
    rf_score: np.ndarray,
    gru_score: np.ndarray,
) -> pd.DataFrame:
    ensemble_avg = 0.5 * gru_score + 0.5 * rf_score
    ensemble_weighted = 0.6 * gru_score + 0.4 * rf_score
    extra = [
        classification_metrics(
            "Ensemble (avg)",
            split.y_test,
            (ensemble_avg >= 0.5).astype(int),
            ensemble_avg,
        ),
        classification_metrics(
            "Ensemble (60%GRU)",
            split.y_test,
            (ensemble_weighted >= 0.5).astype(int),
            ensemble_weighted,
        ),
    ]
    return pd.concat([rows, pd.DataFrame(extra)], ignore_index=True)


def compare_to_report(
    reproduced: pd.DataFrame,
    report_path: Path = OUTPUT_TABLES / "all_model_results.csv",
) -> pd.DataFrame:
    if not report_path.exists():
        return pd.DataFrame()
    reported = pd.read_csv(report_path)
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
    merged = reproduced.merge(reported, on="model", suffixes=("_new", "_reported"))
    for metric in metric_cols:
        merged[f"{metric}_diff"] = merged[f"{metric}_new"] - merged[f"{metric}_reported"]
    cols = ["model"]
    for metric in metric_cols:
        cols.extend([f"{metric}_new", f"{metric}_reported", f"{metric}_diff"])
    return merged[cols]


def run_deep_learning_experiment(
    epochs: int = 20,
    batch_size: int = 64,
    patience: int = 5,
    seed: int = 42,
    output_path: Path = OUTPUT_TABLES / "deep_learning_reproduced_results.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    labeled = pd.read_parquet(DATA_PROCESSED / "baseline_smartwatch_features_labeled.parquet")
    split = prepare_sequence_data(random_state=seed)
    test_groups = labeled.iloc[split.test_idx]["user_key"].to_numpy()
    sequence_results, scores = train_evaluate_sequence_models(
        split,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seed=seed,
        test_groups=test_groups,
    )
    rf_row, rf_score = train_rf_for_ensemble(labeled, split.train_idx, split.test_idx)
    rf_row["seed"] = seed
    results = pd.concat([sequence_results, pd.DataFrame([rf_row])], ignore_index=True)
    results = add_ensemble_rows(results, split, rf_score, scores["GRU (2-layer)"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    comparison = compare_to_report(results)
    if not comparison.empty:
        comparison.to_csv(OUTPUT_TABLES / "deep_learning_report_comparison.csv", index=False)
    return results, comparison


def run_deep_learning_multi_seed(
    seeds: list[int] | None = None,
    epochs: int = 20,
    batch_size: int = 64,
    patience: int = 5,
    output_path: Path = OUTPUT_TABLES / "tier2_multi_seed_results.csv",
) -> pd.DataFrame:
    """Run the Tier-2 sequence pipeline across multiple seeds and aggregate.

    For each ``seed`` in ``seeds`` (default 5 seeds: 42, 43, 44, 45, 46) we
    refit GRU + CNN + RandomForest + ensembles and record their metrics.  The
    aggregate output reports the mean, std, min, max and median of each
    metric per model, plus a patient-cluster bootstrap 95% CI computed from
    the *combined* (concatenated) test probabilities.

    This wholesale replaces the previous ``train_evaluate_sequence_models``
    convention of using cherry-picked seeds (8 for GRU, 43 for CNN); a model
    that only works for one seed out of thirteen is not a defensible result.
    """
    seeds = seeds or [42, 43, 44, 45, 46]
    all_rows: list[pd.DataFrame] = []
    for s in seeds:
        results, _ = run_deep_learning_experiment(
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            seed=s,
            output_path=OUTPUT_TABLES / f"tier2_seed{s}_results.csv",
        )
        results["seed"] = s
        all_rows.append(results)
    combined = pd.concat(all_rows, ignore_index=True)
    metric_cols = [
        c
        for c in combined.columns
        if c not in {"model", "seed", "best_epoch", "best_val_loss", "calibration"}
        and pd.api.types.is_numeric_dtype(combined[c])
    ]
    agg = (
        combined.groupby("model")[metric_cols]
        .agg(["mean", "std", "min", "median", "max"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg["n_seeds"] = len(seeds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    agg.to_csv(output_path.with_name(output_path.stem + "_summary.csv"), index=False)
    return agg


# ── PHASE 1: MULTIMODAL TABULAR FEATURE BUILDING ──────────────────────────


def _standardize_tabular(
    tabular: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalise tabular features using train-set statistics only.

    Returns the scaled array plus the train mean and std vectors so callers
    can apply the same transform to new data.
    """
    if tabular.shape[1] == 0:
        return tabular, np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    train = tabular[train_idx]
    mean = np.nanmean(train, axis=0).astype(np.float32)
    std = np.nanstd(train, axis=0).astype(np.float32)
    std[std == 0] = 1.0
    scaled = ((tabular - mean) / std).astype(np.float32)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return scaled, mean, std


def build_window_tabular_features(
    labeled: pd.DataFrame,
    smartinhaler_path: Path | None = None,
    peakflow_path: Path | None = None,
    daily_scores_path: Path | None = None,
    weekly_path: Path | None = None,
    patient_path: Path | None = None,
    window_minutes: int = WINDOW_MINUTES,
) -> tuple[np.ndarray, list[str]]:
    """Build per-window tabular features from multimodal sources.

    Each row corresponds to one labeled window identified by its
    ``anchor_relative_minute``.  The window covers the 24-hour period
    ``[anchor - window_minutes, anchor)``; the calendar day that contains
    the window's *first* minute is used as the daily join key.

    Missing modality days are filled with ``-1`` and an accompanying
    ``<modality>_missing`` indicator column (1 = no data for that window)
    is appended to each modality block.

    Returns
    -------
    tabular : np.ndarray of shape (N, K), dtype float32
    feature_names : list[str] of length K
    """
    # Lazy import: avoid pulling heavy dependencies at module-import time.
    from .event_features import (
        build_smartinhaler_daily_features,
        build_peakflow_daily_features,
        build_patient_static_features,
    )

    # Resolve default paths.
    if smartinhaler_path is None:
        smartinhaler_path = AAMOS_SMARTINHALER_FILE
    if peakflow_path is None:
        peakflow_path = AAMOS_PEAKFLOW_FILE
    if daily_scores_path is None:
        daily_scores_path = OUTPUT_TABLES / "daily_questionnaire_worsening_scores.csv"
    if weekly_path is None:
        weekly_path = AAMOS_WEEKLY_FILE
    if patient_path is None:
        patient_path = AAMOS_PATIENT_INFO_FILE

    N = len(labeled)
    anchors = labeled["anchor_relative_minute"].to_numpy(dtype=np.int64)
    user_keys = labeled["user_key"].to_numpy(dtype=np.int64)

    # Day that contains the window's first minute.
    start_days = ((anchors - window_minutes) // 1440).astype(np.int64)

    # Master index DataFrame — keeps all joins order-aligned with labeled.
    window_df = pd.DataFrame(
        {
            "window_idx": np.arange(N, dtype=np.int64),
            "user_key": user_keys,
            "start_day": start_days,
        }
    )

    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    # ── 1. Smart inhaler ──────────────────────────────────────────────────
    _si_path = Path(smartinhaler_path)
    if _si_path.exists():
        inhaler_daily = build_smartinhaler_daily_features(_si_path)
        inhaler_cols = [
            c for c in inhaler_daily.columns if c not in {"user_key", "date"}
        ]
        merged = window_df.merge(
            inhaler_daily.rename(columns={"date": "start_day"}),
            on=["user_key", "start_day"],
            how="left",
        ).sort_values("window_idx")
        block = merged[inhaler_cols].to_numpy(dtype=np.float32)
        # A row is missing when *all* inhaler columns are NaN.
        missing_flag = np.isnan(block).all(axis=1, keepdims=True).astype(np.float32)
        block = np.nan_to_num(block, nan=-1.0)
        feature_blocks.append(np.hstack([block, missing_flag]))
        feature_names.extend(inhaler_cols + ["smartinhaler_missing"])

    # ── 2. Peak flow ──────────────────────────────────────────────────────
    _pf_path = Path(peakflow_path)
    _pp_path = Path(patient_path)
    if _pf_path.exists() and _pp_path.exists():
        pf_daily = build_peakflow_daily_features(_pf_path, _pp_path)
        pf_cols = [c for c in pf_daily.columns if c not in {"user_key", "date"}]
        merged = window_df.merge(
            pf_daily.rename(columns={"date": "start_day"}),
            on=["user_key", "start_day"],
            how="left",
        ).sort_values("window_idx")
        block = merged[pf_cols].to_numpy(dtype=np.float32)
        missing_flag = np.isnan(block).all(axis=1, keepdims=True).astype(np.float32)
        block = np.nan_to_num(block, nan=-1.0)
        feature_blocks.append(np.hstack([block, missing_flag]))
        feature_names.extend(pf_cols + ["peakflow_missing"])

    # ── 3. Daily questionnaire worsening score ────────────────────────────
    # REMOVED (Phase 1.1 target-leakage fix): the per-day worsening score is
    # the *exact* quantity the canonical label is thresholded from
    # (event_labels.daily_questionnaire_worsening_score >= 3 → positive).
    # Including it as a feature is trivial leakage and was responsible for the
    # artefactual Tier-3 "Full" AUC ≈ 0.93.  The score is intentionally never
    # exposed as a model input.

    # ── 4. Weekly questionnaire – most recent score before window start ───
    _wq_path = Path(weekly_path)
    if _wq_path.exists():
        wq = pd.read_csv(_wq_path)
        wq["user_key"] = pd.to_numeric(wq["user_key"], errors="coerce")
        wq["date"] = pd.to_numeric(wq["date"], errors="coerce")
        wq = wq.dropna(subset=["user_key", "date"]).copy()
        wq["user_key"] = wq["user_key"].astype(np.int64)
        wq["date"] = wq["date"].astype(np.int64)
        symptom_cols = [
            "weekly_night_symp",
            "weekly_day_symp",
            "weekly_limit_activity",
            "weekly_short_breath",
            "weekly_wheeze",
            "weekly_relief_inhaler",
        ]
        avail_sym = [c for c in symptom_cols if c in wq.columns]
        for c in avail_sym:
            wq[c] = pd.to_numeric(wq[c], errors="coerce")
        wq = wq[["user_key", "date"] + avail_sym].drop_duplicates()

        # For each window, find the most recent weekly questionnaire row
        # whose date is *strictly before* the window's start day.  Using
        # strict-inequality removes a same-day look-ahead path: a weekly
        # questionnaire can be filed at any time within its calendar day, so
        # a row with date == start_day may be filled out *after* the window
        # begins.  Causal alignment requires we only see past questionnaires.
        merged = window_df.merge(wq, on="user_key", how="left")
        merged = merged[merged["date"] < merged["start_day"]].copy()
        latest = (
            merged.sort_values(["window_idx", "date"])
            .groupby("window_idx", as_index=False)
            .last()
        )
        result = (
            window_df[["window_idx"]]
            .merge(latest[["window_idx"] + avail_sym], on="window_idx", how="left")
            .sort_values("window_idx")
        )
        block = result[avail_sym].to_numpy(dtype=np.float32)
        missing_flag = np.isnan(block).all(axis=1, keepdims=True).astype(np.float32)
        block = np.nan_to_num(block, nan=-1.0)
        feature_blocks.append(np.hstack([block, missing_flag]))
        feature_names.extend(
            [f"weekly_{c}" for c in avail_sym] + ["weekly_score_missing"]
        )

    # ── 5. Patient static features ────────────────────────────────────────
    _pt_path = Path(patient_path)
    if _pt_path.exists():
        patient_static = build_patient_static_features(_pt_path)
        pat_cols = [c for c in patient_static.columns if c != "user_key"]
        merged = (
            window_df[["window_idx", "user_key"]]
            .merge(patient_static, on="user_key", how="left")
            .sort_values("window_idx")
        )
        block = merged[pat_cols].to_numpy(dtype=np.float32)
        block = np.nan_to_num(block, nan=-1.0)
        feature_blocks.append(block)
        feature_names.extend(pat_cols)

    if not feature_blocks:
        return np.empty((N, 0), dtype=np.float32), []

    return np.hstack(feature_blocks).astype(np.float32), feature_names


# ── PHASE 2: MULTIMODAL SPLIT DATA & DATA PREPARATION ────────────────────────


@dataclass
class MultimodalSplitData:
    """Extends SplitData with per-window tabular features from all modalities."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_users: list[int]
    val_users: list[int]
    test_users: list[int]
    # Tabular multimodal features (N_split, K), already z-score normalised.
    tab_train: np.ndarray
    tab_val: np.ndarray
    tab_test: np.ndarray
    tab_feature_names: list[str]


def prepare_multimodal_data(
    labeled_path: Path = DATA_PROCESSED / "baseline_smartwatch_features_labeled.parquet",
    clean_path: Path = DATA_INTERIM / "smartwatch_cleaned.parquet",
    smartinhaler_path: Path | None = None,
    peakflow_path: Path | None = None,
    daily_scores_path: Path | None = None,
    weekly_path: Path | None = None,
    patient_path: Path | None = None,
    random_state: int = 42,
) -> MultimodalSplitData:
    """Load, split, and standardise both sequence and tabular multimodal data.

    Uses the same patient-wise GroupShuffleSplit (seed=42 by default) as
    ``prepare_sequence_data`` so multimodal results are directly comparable
    to the Tier-1 and Tier-2 unimodal baselines.

    The labeled Parquet file is required for sequences (minute-level index);
    the CSV variant is accepted as a fallback so the function works even when
    the Parquet has not yet been generated.
    """
    # Accept both Parquet and CSV so callers don't need to generate the Parquet first.
    if labeled_path.exists():
        labeled = pd.read_parquet(labeled_path).reset_index(drop=True)
    else:
        csv_path = labeled_path.with_suffix(".csv")
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Neither {labeled_path} nor {csv_path} found. "
                "Run notebooks 03 and 04 first."
            )
        labeled = pd.read_csv(csv_path).reset_index(drop=True)

    # ── Sequence data ─────────────────────────────────────────────────────
    clean = pd.read_parquet(clean_path)
    train_idx, val_idx, test_idx = make_patient_splits(labeled, random_state=random_state)
    sequences = build_sequence_array(labeled, clean)
    sequences, _, _ = standardize_by_train(sequences, train_idx)
    y = labeled["target_binary"].astype(int).to_numpy()

    # ── Tabular multimodal features ───────────────────────────────────────
    tabular, tab_feature_names = build_window_tabular_features(
        labeled,
        smartinhaler_path=smartinhaler_path,
        peakflow_path=peakflow_path,
        daily_scores_path=daily_scores_path,
        weekly_path=weekly_path,
        patient_path=patient_path,
    )
    tabular, _, _ = _standardize_tabular(tabular, train_idx)

    return MultimodalSplitData(
        X_train=sequences[train_idx],
        X_val=sequences[val_idx],
        X_test=sequences[test_idx],
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_users=sorted(
            labeled.iloc[train_idx]["user_key"].unique().astype(int).tolist()
        ),
        val_users=sorted(
            labeled.iloc[val_idx]["user_key"].unique().astype(int).tolist()
        ),
        test_users=sorted(
            labeled.iloc[test_idx]["user_key"].unique().astype(int).tolist()
        ),
        tab_train=tabular[train_idx],
        tab_val=tabular[val_idx],
        tab_test=tabular[test_idx],
        tab_feature_names=tab_feature_names,
    )


# ── PHASE 3: MULTIMODAL GRU ARCHITECTURE + LOADERS + TRAINING ───────────────


def _select_tab_cols(
    feature_names: list[str],
    prefixes: list[str] | None,
) -> tuple[list[int], list[str]]:
    """Return ``(column_indices, column_names)`` for the given prefix filter.

    ``prefixes=None``  → all K columns (full multimodal, tab_dim=90).
    ``prefixes=[]``    → no columns (sequence-only, tab_dim=0).
    Otherwise          → columns whose name starts with any of the given prefixes.
    """
    if prefixes is None:
        return list(range(len(feature_names))), list(feature_names)
    if not prefixes:
        return [], []
    selected_idx, selected_names = [], []
    for i, name in enumerate(feature_names):
        if any(name.startswith(p) for p in prefixes):
            selected_idx.append(i)
            selected_names.append(name)
    return selected_idx, selected_names


def _platt_scale(
    y_val: np.ndarray,
    p_val: np.ndarray,
    p_test: np.ndarray,
) -> np.ndarray:
    """Apply Platt scaling (sigmoid calibration) fitted on the validation set.

    Corrects the systematic probability shift caused by the prevalence difference
    between the train set (22.5% positive) and test set (48.2% positive).  A
    ``LogisticRegression`` is fit on the 1-D validation probabilities and then
    applied to the test probabilities — equivalent to Platt scaling.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    calibrator.fit(p_val.reshape(-1, 1), y_val)
    return calibrator.predict_proba(p_test.reshape(-1, 1))[:, 1]


def make_multimodal_gru(
    seq_input_dim: int = 4,
    tab_dim: int = 0,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    subsample: int = 5,
):
    """Build an intermediate-fusion GRU for multimodal asthma exacerbation prediction.

    Architecture
    ────────────
    Sequence branch
        GRU(seq_input_dim, hidden_dim, num_layers, dropout) → h_T  (hidden_dim,)
    Tabular branch  (constructed only when tab_dim > 0)
        Linear(tab_dim → hidden_dim) → ReLU → Dropout
        Linear(hidden_dim → hidden_dim // 2) → ReLU → Dropout
        → tab_embed  (hidden_dim // 2,)
    Fusion FC
        tab_dim = 0 : Linear(hidden_dim, 1)              unimodal GRU
        tab_dim > 0 : Linear(hidden_dim + hidden_dim // 2, 1)   multimodal

    Parameters
    ----------
    seq_input_dim : channels in the minute-level sequence (HR, activity, intensity, steps).
    tab_dim       : number of tabular features; 0 → sequence-only (SW_only ablation).
    hidden_dim    : GRU hidden size (default 64, matches Tier-2 unimodal GRU).
    num_layers    : stacked GRU layers (default 2).
    dropout       : dropout rate applied inside GRU and tab MLP (default 0.3).
    subsample     : take every ``subsample``-th minute before feeding the GRU
                    (default 5 → 1440 → 288 steps, same as Tier-2 baseline).
    """
    torch, nn, _, _ = _require_torch()
    tab_hidden = hidden_dim // 2  # 32
    fc_in = hidden_dim + (tab_hidden if tab_dim > 0 else 0)

    class _MultimodalGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._subsample = subsample
            self.gru = nn.GRU(
                input_size=seq_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.tab_mlp: nn.Sequential | None
            if tab_dim > 0:
                self.tab_mlp = nn.Sequential(
                    nn.Linear(tab_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, tab_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            else:
                self.tab_mlp = None
            self.fc = nn.Linear(fc_in, 1)

        def forward(self, x_seq, x_tab=None):  # type: ignore[override]
            _, h = self.gru(x_seq[:, :: self._subsample, :])
            h_T = h[-1]  # (batch, hidden_dim)
            if self.tab_mlp is not None and x_tab is not None:
                fused = torch.cat([h_T, self.tab_mlp(x_tab)], dim=1)
            else:
                fused = h_T
            return self.fc(fused).squeeze(-1)

    return _MultimodalGRU()


def make_multimodal_loaders(
    split: MultimodalSplitData,
    col_indices: list[int],
    batch_size: int = 64,
):
    """Return ``(train_loader, val_loader, test_loader)`` yielding ``(seq, tab, label)`` triples.

    When ``col_indices`` is empty the tab tensors have shape ``(batch, 0)`` so
    the same loop can handle both unimodal and multimodal configs without branching.
    """
    torch, _, DataLoader, TensorDataset = _require_torch()

    def _make(seq: np.ndarray, tab_full: np.ndarray, y: np.ndarray, shuffle: bool):
        seq_t = torch.from_numpy(seq).float()
        if col_indices:
            tab_t = torch.from_numpy(tab_full[:, col_indices]).float()
        else:
            tab_t = torch.zeros(len(seq), 0, dtype=torch.float32)
        y_t = torch.from_numpy(y.astype(np.float32))
        ds = TensorDataset(seq_t, tab_t, y_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return (
        _make(split.X_train, split.tab_train, split.y_train, shuffle=True),
        _make(split.X_val, split.tab_val, split.y_val, shuffle=False),
        _make(split.X_test, split.tab_test, split.y_test, shuffle=False),
    )


def train_multimodal_model(
    model,
    train_loader,
    val_loader,
    y_train: np.ndarray,
    epochs: int = 20,
    patience: int = 5,
    lr: float = 1e-3,
    device: str | None = None,
    verbose: bool = True,
):
    """Train the multimodal GRU with BCEWithLogitsLoss and early stopping on val loss.

    Differences from ``train_torch_model``:
    - Handles ``(seq, tab, label)`` triples from ``make_multimodal_loaders``.
    - Uses ``Adam(lr, weight_decay=1e-4)`` for mild regularisation.
    - ``x_tab`` is passed as ``None`` when the tab tensor is empty (col_indices=[])
      so the model's forward path degrades gracefully to the unimodal GRU.
    """
    torch, nn, _, _ = _require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pos = max(float((y_train == 1).sum()), 1.0)
    neg = max(float((y_train == 0).sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_state: dict | None = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x_seq, x_tab, yb in train_loader:
            x_seq = x_seq.to(device)
            x_tab_d = x_tab.to(device) if x_tab.shape[1] > 0 else None
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_seq, x_tab_d), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_seq, x_tab, yb in val_loader:
                x_seq = x_seq.to(device)
                x_tab_d = x_tab.to(device) if x_tab.shape[1] > 0 else None
                yb = yb.to(device)
                val_losses.append(float(criterion(model(x_seq, x_tab_d), yb).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            improved = " ✓"
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"    early stop at epoch {epoch}", flush=True)
                break

        if verbose:
            print(
                f"    epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}{improved}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), {"best_epoch": best_epoch, "best_val_loss": best_val}


def predict_multimodal_model(
    model,
    loader,
    device: str | None = None,
) -> np.ndarray:
    """Return sigmoid probabilities ``(N,)`` from a multimodal DataLoader."""
    torch, _, _, _ = _require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    scores = []
    with torch.no_grad():
        for x_seq, x_tab, _ in loader:
            x_seq = x_seq.to(device)
            x_tab_d = x_tab.to(device) if x_tab.shape[1] > 0 else None
            logits = model(x_seq, x_tab_d)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


# ── PHASE 4: ABLATION EXPERIMENT RUNNER ──────────────────────────────────────

# Five ablation configurations ordered from least to most information.
# Each entry: (display_name, tab_prefixes)
#   tab_prefixes=[]   → SW-only; tab_dim=0, model degrades to unimodal GRU.
#   tab_prefixes=list → select columns whose names start with any listed prefix.
#   tab_prefixes=None → all K=90 columns (full multimodal).
#
# Patient-static features are included only in the "Full" config, matching the
# design in notes.txt (patient_static listed only for the Full block).
# Note (Phase 1 leakage fix): the previous "SW_inhaler_pef_dailyQ" config
# selected the `daily_*` worsening-score columns; those features have been
# removed (they were the label).  The remaining four configs form a clean
# information-ascending ablation: smartwatch-only → +inhaler → +peakflow →
# +weekly questionnaire symptoms + patient-static (Full).
MULTIMODAL_ABLATION_CONFIGS: list[tuple[str, list[str] | None]] = [
    ("SW_only",        []),
    ("SW_inhaler",     ["smartinhaler"]),
    ("SW_inhaler_pef", ["smartinhaler", "peakflow"]),
    ("Full",           None),
]


def run_multimodal_experiment(
    labeled_path: Path = DATA_PROCESSED / "baseline_smartwatch_features_labeled.parquet",
    clean_path: Path = DATA_INTERIM / "smartwatch_cleaned.parquet",
    epochs: int = 20,
    batch_size: int = 64,
    patience: int = 5,
    random_state: int = 42,
    output_path: Path = OUTPUT_TABLES / "multimodal_results.csv",
) -> pd.DataFrame:
    """Run all 5 multimodal ablation experiments and return consolidated results.

    For each config in ``MULTIMODAL_ABLATION_CONFIGS``:
      1. Select tabular columns by prefix using ``_select_tab_cols``.
      2. Build ``make_multimodal_gru`` + ``make_multimodal_loaders``.
      3. Train with ``train_multimodal_model`` (early stopping on val loss).
      4. Apply Platt scaling on validation probabilities before test evaluation,
         to correct for the train (22.5%) → test (48.2%) prevalence shift.
      5. Compute 7 metrics; append ``tab_dim``, ``best_epoch``, ``best_val_loss``.
      6. Save per-config training history to ``outputs/tables/mm_<name>_history.csv``.

    Notes from data audit
    ─────────────────────
    - ROC-AUC is the primary discrimination metric; the train/test prevalence
      gap under the canonical event-onset label is smaller than under the
      previous weekly-symptom label but Platt scaling is still applied.
    - Bootstrap CIs are computed by resampling whole test-set patients
      (cluster bootstrap), to honestly reflect the 95.8% intra-patient window
      overlap.
    - High modality-missing rates (smartinhaler 63%, peakflow varies) are
      handled by sentinel imputation (-1) + binary missing indicators baked
      into the tabular features.  The daily questionnaire worsening score is
      *excluded* from the feature set (it is the label source — see Phase 1).
    """
    set_seed(random_state)
    print("Preparing multimodal data (sequences + tabular)…")
    split = prepare_multimodal_data(
        labeled_path=labeled_path,
        clean_path=clean_path,
        random_state=random_state,
    )
    print(
        f"  train={split.X_train.shape[0]}  val={split.X_val.shape[0]}"
        f"  test={split.X_test.shape[0]}  K={split.tab_train.shape[1]}"
    )

    # Per-window patient identity for cluster bootstrap.
    labeled_for_groups = pd.read_parquet(labeled_path).reset_index(drop=True)
    test_groups = labeled_for_groups.iloc[split.test_idx]["user_key"].to_numpy()

    rows = []
    for config_name, prefixes in MULTIMODAL_ABLATION_CONFIGS:
        col_indices, col_names = _select_tab_cols(split.tab_feature_names, prefixes)
        tab_dim = len(col_indices)
        print(f"\n[Ablation] {config_name!r}  tab_dim={tab_dim}")

        set_seed(random_state)  # ensure identical weight init across configs
        model = make_multimodal_gru(tab_dim=tab_dim)
        train_loader, val_loader, test_loader = make_multimodal_loaders(
            split, col_indices=col_indices, batch_size=batch_size
        )
        model, history, meta = train_multimodal_model(
            model,
            train_loader,
            val_loader,
            split.y_train,
            epochs=epochs,
            patience=patience,
        )

        p_val = predict_multimodal_model(model, val_loader)
        p_test_raw = predict_multimodal_model(model, test_loader)
        p_test = _platt_scale(split.y_val, p_val, p_test_raw)

        pred = (p_test >= 0.5).astype(int)
        row = classification_metrics(config_name, split.y_test, pred, p_test)
        row["tab_dim"] = tab_dim
        row["best_epoch"] = meta["best_epoch"]
        row["best_val_loss"] = round(meta["best_val_loss"], 6)
        row["seed"] = random_state
        ci_lo, ci_hi = bootstrap_auc_ci(split.y_test, p_test, groups=test_groups)
        row["auc_ci_lo"] = round(ci_lo, 3)
        row["auc_ci_hi"] = round(ci_hi, 3)
        rows.append(row)

        safe_name = config_name.lower()
        history.to_csv(OUTPUT_TABLES / f"mm_{safe_name}_history.csv", index=False)
        print(
            f"  → ROC-AUC={row['roc_auc']:.3f}  PR-AUC={row['pr_auc']:.3f}"
            f"  F1={row['f1']:.3f}  best_epoch={meta['best_epoch']}"
        )

    results = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nSaved multimodal results → {output_path}")
    return results


def run_multimodal_multi_seed(
    seeds: list[int] | None = None,
    labeled_path: Path = DATA_PROCESSED / "baseline_smartwatch_features_labeled.parquet",
    clean_path: Path = DATA_INTERIM / "smartwatch_cleaned.parquet",
    epochs: int = 20,
    batch_size: int = 64,
    patience: int = 5,
    output_path: Path = OUTPUT_TABLES / "multimodal_multi_seed_results.csv",
) -> pd.DataFrame:
    """Run the Tier-3 multimodal ablation across multiple seeds and aggregate.

    For each seed we re-run all configurations in
    ``MULTIMODAL_ABLATION_CONFIGS`` with a fresh patient-grouped split and
    fresh model initialisation, then report mean / std / min / median / max
    of each metric per config.  This is the Tier-3 analogue of
    ``run_deep_learning_multi_seed`` and is the defensible way to compare
    modality blocks given the high seed sensitivity in an 18-patient cohort.
    """
    seeds = seeds or [42, 43, 44, 45, 46]
    all_rows: list[pd.DataFrame] = []
    for s in seeds:
        per_seed_path = OUTPUT_TABLES / f"multimodal_seed{s}_results.csv"
        df = run_multimodal_experiment(
            labeled_path=labeled_path,
            clean_path=clean_path,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            random_state=s,
            output_path=per_seed_path,
        )
        df["seed"] = s
        all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True)
    metric_cols = [
        c
        for c in combined.columns
        if c
        not in {"model", "seed", "tab_dim", "best_epoch", "best_val_loss", "calibration"}
        and pd.api.types.is_numeric_dtype(combined[c])
    ]
    agg = (
        combined.groupby("model")[metric_cols]
        .agg(["mean", "std", "min", "median", "max"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg["n_seeds"] = len(seeds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    agg.to_csv(output_path.with_name(output_path.stem + "_summary.csv"), index=False)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tier-2 GRU/CNN experiments.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    results, comparison = run_deep_learning_experiment(
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience
    )
    print("Reproduced results:")
    print(results.to_string(index=False))
    if not comparison.empty:
        print("\nComparison to outputs/tables/all_model_results.csv:")
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
