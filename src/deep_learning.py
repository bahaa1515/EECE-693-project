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

from .config import DATA_INTERIM, DATA_PROCESSED, OUTPUT_TABLES
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
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
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


def make_gru(input_dim: int = 4, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
    _, nn, _, _ = _require_torch()

    class GRUClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = x[:, ::5, :]
            _, hidden = self.gru(x)
            return self.fc(hidden[-1]).squeeze(1)

    return GRUClassifier()


def make_cnn(input_dim: int = 4, dropout: float = 0.3):
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
    learning_rate: float = 1e-3,
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


def train_evaluate_sequence_models(
    split: SplitData,
    epochs: int = 20,
    batch_size: int = 64,
    patience: int = 5,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    train_loader, val_loader, test_loader = make_loaders(split, batch_size=batch_size)
    rows = []
    scores: dict[str, np.ndarray] = {}

    for name, factory in [("GRU (2-layer)", make_gru), ("CNN (3-layer)", make_cnn)]:
        model, history, meta = train_torch_model(
            factory(),
            train_loader,
            val_loader,
            split.y_train,
            epochs=epochs,
            patience=patience,
        )
        score = predict_torch_model(model, test_loader)
        pred = (score >= 0.5).astype(int)
        row = classification_metrics(name, split.y_test, pred, score)
        row["best_epoch"] = meta["best_epoch"]
        row["best_val_loss"] = meta["best_val_loss"]
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
    output_path: Path = OUTPUT_TABLES / "deep_learning_reproduced_results.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(42)
    labeled = pd.read_parquet(DATA_PROCESSED / "baseline_smartwatch_features_labeled.parquet")
    split = prepare_sequence_data(random_state=42)
    sequence_results, scores = train_evaluate_sequence_models(
        split, epochs=epochs, batch_size=batch_size, patience=patience
    )
    rf_row, rf_score = train_rf_for_ensemble(labeled, split.train_idx, split.test_idx)
    results = pd.concat([sequence_results, pd.DataFrame([rf_row])], ignore_index=True)
    results = add_ensemble_rows(results, split, rf_score, scores["GRU (2-layer)"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    comparison = compare_to_report(results)
    if not comparison.empty:
        comparison.to_csv(OUTPUT_TABLES / "deep_learning_report_comparison.csv", index=False)
    return results, comparison


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
