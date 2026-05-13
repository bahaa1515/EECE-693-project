"""Experimental variants on the canonical Tier-2/Tier-3 pipeline.

This module preserves the existing ``src.deep_learning`` API and adds:
- Per-patient normalisation of the sequence input (``standardize_per_patient``).
- Stratified-group patient splits (``make_patient_splits_stratified``).
- Alternative label constructions (``derive_alternative_labels``).
- Helpers to build the per-window Tier-1 engineered feature matrix
  (``get_engineered_feature_matrix``) for use in hybrid Tier-3 / LightGBM /
  TabPFN / stacking experiments.

Every experiment exposed here is opt-in; the default ``run_full_pipeline.py``
behaviour is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_INTERIM, DATA_PROCESSED, OUTPUT_TABLES
from .deep_learning import (
    SEQUENCE_COLUMNS,
    SplitData,
    MultimodalSplitData,
    bootstrap_auc_ci,
    build_sequence_array,
    build_window_tabular_features,
    make_patient_splits,
    standardize_by_train,
    _standardize_tabular,
)
from .features import FEATURE_COLUMNS, WINDOW_MINUTES


# ── A1: Per-patient sequence normalisation ───────────────────────────────────


def standardize_per_patient(
    sequences: np.ndarray,
    user_keys: np.ndarray,
    train_idx: np.ndarray,
    floor_std: float = 1e-3,
) -> np.ndarray:
    """Per-user z-score: each user's windows normalised by *that user's*
    own training-window mean and std.

    Why this can help at small cohort sizes:
        - Resting-HR baselines differ by 20-30 bpm across patients in this
          cohort; the global z-score forces the model to spend capacity
          learning to ignore each patient's level.
        - Subtraction of the per-user mean removes between-patient nuisance
          variance, leaving only within-patient deviation — which is what
          actually relates to exacerbation events.

    Leakage safety:
        - Mean/std are computed using **only training-index windows of each
          user**, never test or validation windows.  This is identical in
          spirit to ``standardize_by_train`` (global) — the only change is
          per-user grouping.
        - For test users (no overlap with training users under
          patient-wise split), we fall back to the training-set *global*
          mean/std, so test users see the same fixed transform a deployed
          model would see when handed an unseen patient.
    """
    sequences = sequences.astype(np.float32, copy=True)
    user_keys = np.asarray(user_keys)

    train_users = np.unique(user_keys[train_idx])
    train_data = sequences[train_idx]
    global_mean = np.nanmean(train_data, axis=(0, 1)).astype(np.float32)
    global_std = np.nanstd(train_data, axis=(0, 1)).astype(np.float32)
    global_std[global_std < floor_std] = 1.0

    # Pre-compute per-user (mean, std) for users that appear in training.
    per_user_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for u in train_users:
        u_train_mask = (user_keys[train_idx] == u)
        u_seqs = train_data[u_train_mask]
        u_mean = np.nanmean(u_seqs, axis=(0, 1)).astype(np.float32)
        u_std = np.nanstd(u_seqs, axis=(0, 1)).astype(np.float32)
        u_std = np.where(u_std < floor_std, global_std, u_std).astype(np.float32)
        per_user_stats[int(u)] = (u_mean, u_std)

    out = np.empty_like(sequences)
    for i in range(len(sequences)):
        u = int(user_keys[i])
        mean, std = per_user_stats.get(u, (global_mean, global_std))
        out[i] = (sequences[i] - mean) / std

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ── A3: Stratified-group patient split ───────────────────────────────────────


def make_patient_splits_stratified(
    labeled: pd.DataFrame,
    test_size: float = 0.25,
    val_size: float = 0.20,
    random_state: int = 42,
    n_trials: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-stratified patient split balancing test-set positive rate.

    sklearn's ``StratifiedGroupKFold`` solves the binary case but
    ``GroupShuffleSplit`` is unaware of class balance.  This routine draws
    ``n_trials`` random group splits and keeps the one whose test-set
    positive rate is closest to the overall positive rate, then does the
    same for the inner val split.  Patient identities remain disjoint.
    """
    rng = np.random.default_rng(random_state)
    user_pos_rate = labeled.groupby("user_key")["target_binary"].mean()
    user_count = labeled.groupby("user_key").size()
    overall_pos = float(labeled["target_binary"].mean())
    users = np.array(user_pos_rate.index.tolist())
    n_users = len(users)
    n_test = max(2, int(round(test_size * n_users)))

    def _trial_split(pool: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.permutation(len(pool))
        test_users = pool[idx[:k]]
        rest_users = pool[idx[k:]]
        # Compute test-fold positive rate weighted by # windows.
        test_pos = (
            (user_pos_rate[test_users] * user_count[test_users]).sum()
            / max(user_count[test_users].sum(), 1)
        )
        return test_users, rest_users, abs(float(test_pos) - overall_pos)

    best = None
    for _ in range(n_trials):
        test_u, rest_u, gap = _trial_split(users, n_test)
        if best is None or gap < best[2]:
            best = (test_u, rest_u, gap)
    test_users, rest_users, _ = best  # type: ignore[misc]

    # Inner split: stratify val from rest.
    n_val = max(1, int(round(val_size * len(rest_users))))
    best_inner = None
    for _ in range(n_trials):
        val_u, train_u, gap = _trial_split(rest_users, n_val)
        if best_inner is None or gap < best_inner[2]:
            best_inner = (val_u, train_u, gap)
    val_users, train_users, _ = best_inner  # type: ignore[misc]

    user_arr = labeled["user_key"].to_numpy()
    train_idx = np.where(np.isin(user_arr, train_users))[0]
    val_idx = np.where(np.isin(user_arr, val_users))[0]
    test_idx = np.where(np.isin(user_arr, test_users))[0]
    return train_idx, val_idx, test_idx


# ── B7: Alternative-label sensitivity ────────────────────────────────────────


def derive_alternative_labels(labeled: pd.DataFrame) -> dict[str, pd.Series]:
    """Return a dict of alternative label series keyed by name.

    All alternatives are derived from columns already in the labelled parquet
    so no re-windowing is required:

    - ``canonical``    : OR of doc/hospital/er/oral/symptom (current label).
    - ``objective``    : OR of doc/hospital/er/oral (drops symptom_flag, the
                         soft, questionnaire-derived component).
    - ``hospitalisation``: OR of hospital/er (lowest-noise clinical event).
    - ``symptom_only`` : symptom_flag alone (control — should be much easier).
    """
    out: dict[str, pd.Series] = {}
    flags = ["doc_flag", "hospital_flag", "er_flag", "oral_flag", "symptom_flag"]
    out["canonical"] = labeled[flags].fillna(0).astype(int).max(axis=1)
    out["objective"] = labeled[["doc_flag", "hospital_flag", "er_flag", "oral_flag"]].fillna(0).astype(int).max(axis=1)
    out["hospitalisation"] = labeled[["hospital_flag", "er_flag"]].fillna(0).astype(int).max(axis=1)
    out["symptom_only"] = labeled["symptom_flag"].fillna(0).astype(int)
    return out


# ── Tier-1 engineered feature matrix helper ──────────────────────────────────


def get_engineered_feature_matrix(
    labeled: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return the per-window 29-feature Tier-1 engineered matrix.

    Useful for hybrid Tier-3 (engineered → tabular branch), LightGBM, TabPFN,
    and stacking experiments.  NaN values are zero-filled at the source; the
    z-score standardisation is left to the caller (so it can be applied with
    train-only stats via ``_standardize_tabular``).
    """
    feature_cols = feature_cols or list(FEATURE_COLUMNS)
    X = labeled[feature_cols].to_numpy(dtype=np.float32, copy=True)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feature_cols


# ── Shared multimodal-prep variant with knobs ────────────────────────────────


@dataclass
class ExperimentConfig:
    """Knobs for a single experimental run.

    Defaults reproduce the canonical pipeline exactly.
    """
    normalization: str = "global"          # "global" | "per_patient"
    split_strategy: str = "groupshuffle"   # "groupshuffle" | "stratified_group"
    label_name: str = "canonical"          # see derive_alternative_labels
    include_engineered: bool = False       # add 29-feature engineered block to tab
    drop_high_missing_users: bool = False  # drop users with HR_missing > 40%


def _apply_label_choice(labeled: pd.DataFrame, label_name: str) -> pd.DataFrame:
    if label_name == "canonical":
        return labeled
    alts = derive_alternative_labels(labeled)
    if label_name not in alts:
        raise KeyError(f"Unknown label_name={label_name!r}")
    labeled = labeled.copy()
    labeled["target_binary"] = alts[label_name].astype(int).to_numpy()
    return labeled


def _maybe_drop_high_miss_users(
    labeled: pd.DataFrame, threshold: float = 0.40
) -> pd.DataFrame:
    miss = labeled.groupby("user_key")["hr_missing_pct"].mean()
    keep = miss[miss <= threshold * 100].index.tolist()  # hr_missing_pct is 0-100
    if len(keep) < labeled["user_key"].nunique():
        labeled = labeled[labeled["user_key"].isin(keep)].reset_index(drop=True)
    return labeled


def prepare_data_for_experiment(
    cfg: ExperimentConfig,
    labeled_path: Path = DATA_PROCESSED / "baseline_smartwatch_features_labeled.parquet",
    clean_path: Path = DATA_INTERIM / "smartwatch_cleaned.parquet",
    random_state: int = 42,
    return_multimodal: bool = True,
):
    """Build a (multi-modal) split honouring an ``ExperimentConfig``."""
    labeled = pd.read_parquet(labeled_path).reset_index(drop=True)
    labeled = _apply_label_choice(labeled, cfg.label_name)
    if cfg.drop_high_missing_users:
        labeled = _maybe_drop_high_miss_users(labeled)

    # Split
    if cfg.split_strategy == "stratified_group":
        train_idx, val_idx, test_idx = make_patient_splits_stratified(
            labeled, random_state=random_state
        )
    else:
        train_idx, val_idx, test_idx = make_patient_splits(
            labeled, random_state=random_state
        )

    # Sequences
    clean = pd.read_parquet(clean_path)
    user_keys = labeled["user_key"].to_numpy()
    sequences = build_sequence_array(labeled, clean)
    if cfg.normalization == "per_patient":
        sequences = standardize_per_patient(sequences, user_keys, train_idx)
    else:
        sequences, _, _ = standardize_by_train(sequences, train_idx)
    y = labeled["target_binary"].astype(int).to_numpy()

    if not return_multimodal:
        return (
            SplitData(
                X_train=sequences[train_idx],
                X_val=sequences[val_idx],
                X_test=sequences[test_idx],
                y_train=y[train_idx],
                y_val=y[val_idx],
                y_test=y[test_idx],
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                train_users=sorted(np.unique(user_keys[train_idx]).astype(int).tolist()),
                val_users=sorted(np.unique(user_keys[val_idx]).astype(int).tolist()),
                test_users=sorted(np.unique(user_keys[test_idx]).astype(int).tolist()),
            ),
            labeled,
        )

    # Multi-modal tabular
    tabular, tab_feature_names = build_window_tabular_features(labeled)
    if cfg.include_engineered:
        eng_X, eng_cols = get_engineered_feature_matrix(labeled)
        if tabular.shape[1] == 0:
            tabular = eng_X
            tab_feature_names = [f"eng_{c}" for c in eng_cols]
        else:
            tabular = np.hstack([tabular, eng_X])
            tab_feature_names = list(tab_feature_names) + [f"eng_{c}" for c in eng_cols]
    if tabular.shape[1] > 0:
        tabular, _, _ = _standardize_tabular(tabular, train_idx)

    return (
        MultimodalSplitData(
            X_train=sequences[train_idx],
            X_val=sequences[val_idx],
            X_test=sequences[test_idx],
            y_train=y[train_idx],
            y_val=y[val_idx],
            y_test=y[test_idx],
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            train_users=sorted(np.unique(user_keys[train_idx]).astype(int).tolist()),
            val_users=sorted(np.unique(user_keys[val_idx]).astype(int).tolist()),
            test_users=sorted(np.unique(user_keys[test_idx]).astype(int).tolist()),
            tab_train=tabular[train_idx],
            tab_val=tabular[val_idx],
            tab_test=tabular[test_idx],
            tab_feature_names=list(tab_feature_names),
        ),
        labeled,
    )


# ── Common evaluation utility ────────────────────────────────────────────────


def evaluate_with_ci(
    y_true: np.ndarray,
    p_test: np.ndarray,
    test_groups: np.ndarray | None = None,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the 7-metric panel plus a patient-cluster bootstrap CI."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (p_test >= threshold).astype(int)
    row = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p_test)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, p_test)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_true, p_test)),
    }
    if test_groups is not None and len(np.unique(y_true)) > 1:
        lo, hi = bootstrap_auc_ci(y_true, p_test, groups=test_groups, seed=seed)
        row["auc_ci_lo"] = lo
        row["auc_ci_hi"] = hi
    return row
