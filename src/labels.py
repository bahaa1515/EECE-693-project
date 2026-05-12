from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW, OUTPUT_TABLES


USER_COL = "user_key"
WEEKLY_DAY_COL = "date"
SYMPTOM_COLUMNS = [
    "weekly_night_symp",
    "weekly_day_symp",
    "weekly_limit_activity",
    "weekly_short_breath",
    "weekly_wheeze",
    "weekly_relief_inhaler",
]
EVENT_COLUMNS = ["weekly_doc", "weekly_hospital", "weekly_er", "weekly_oral"]
WEEKLY_COLUMNS = SYMPTOM_COLUMNS + EVENT_COLUMNS
MAX_HORIZON_DAYS = 7


def load_weekly_questionnaire(
    weekly_path: Path = DATA_RAW / "anonym_aamos00_weeklyquestionnaire.csv",
) -> pd.DataFrame:
    return pd.read_csv(weekly_path)


def prepare_weekly_flags(weekly_df: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly_df[[USER_COL, WEEKLY_DAY_COL] + WEEKLY_COLUMNS].copy()
    weekly = weekly.dropna(subset=[USER_COL, WEEKLY_DAY_COL]).drop_duplicates().copy()

    numeric_cols = [
        "weekly_night_symp",
        "weekly_day_symp",
        "weekly_limit_activity",
        "weekly_short_breath",
        "weekly_wheeze",
        "weekly_relief_inhaler",
        "weekly_hospital",
        "weekly_er",
        "weekly_oral",
    ]
    for col in numeric_cols:
        weekly[col] = pd.to_numeric(weekly[col], errors="coerce")

    weekly["weekly_doc"] = weekly["weekly_doc"].fillna("0").astype(str).str.strip()
    weekly["doc_flag"] = weekly["weekly_doc"].ne("0")
    weekly["hospital_flag"] = weekly["weekly_hospital"].fillna(0).gt(0)
    weekly["er_flag"] = weekly["weekly_er"].fillna(0).gt(0)
    weekly["oral_flag"] = weekly["weekly_oral"].fillna(0).ge(3)

    symptom_domains = pd.DataFrame(
        {
            "day": weekly["weekly_day_symp"].fillna(0).ge(4),
            "night": weekly["weekly_night_symp"].fillna(0).ge(3),
            "breath": weekly["weekly_short_breath"].fillna(0).ge(4),
            "wheeze": weekly["weekly_wheeze"].fillna(0).ge(4),
            "relief": weekly["weekly_relief_inhaler"].fillna(0).ge(4),
            "limit": weekly["weekly_limit_activity"].fillna(0).ge(3),
        }
    )
    weekly["symptom_flag"] = symptom_domains.sum(axis=1).ge(3)
    weekly["target_binary"] = (
        weekly["doc_flag"]
        | weekly["hospital_flag"]
        | weekly["er_flag"]
        | weekly["oral_flag"]
        | weekly["symptom_flag"]
    ).astype(int)

    weekly["week_end_day"] = pd.to_numeric(weekly[WEEKLY_DAY_COL], errors="coerce")
    weekly["questionnaire_day"] = weekly["week_end_day"] + 1
    weekly["questionnaire_relative_minute"] = weekly["questionnaire_day"] * 1440
    return weekly


def label_feature_windows(
    features_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    max_horizon_days: int = MAX_HORIZON_DAYS,
) -> pd.DataFrame:
    features_sorted = features_df.sort_values("anchor_relative_minute").copy()
    weekly_sorted = weekly_df.sort_values("questionnaire_relative_minute").copy()
    merged_list = []

    for user in features_sorted[USER_COL].unique():
        features_user = features_sorted[features_sorted[USER_COL] == user]
        weekly_user = weekly_sorted[weekly_sorted[USER_COL] == user]
        if weekly_user.empty:
            continue
        merged = pd.merge_asof(
            left=features_user,
            right=weekly_user[
                [
                    USER_COL,
                    "questionnaire_relative_minute",
                    "target_binary",
                    "week_end_day",
                    "questionnaire_day",
                    "doc_flag",
                    "hospital_flag",
                    "er_flag",
                    "oral_flag",
                    "symptom_flag",
                ]
            ],
            left_on="anchor_relative_minute",
            right_on="questionnaire_relative_minute",
            by=USER_COL,
            direction="forward",
            allow_exact_matches=False,  # strict-greater-than (prevents same-minute look-ahead)
        )
        merged_list.append(merged)

    labeled = pd.concat(merged_list, ignore_index=True) if merged_list else pd.DataFrame()
    labeled["time_to_label_minutes"] = (
        labeled["questionnaire_relative_minute"] - labeled["anchor_relative_minute"]
    )
    labeled["time_to_label_days"] = labeled["time_to_label_minutes"] / 1440
    max_horizon_minutes = max_horizon_days * 1440
    return labeled[
        labeled["time_to_label_minutes"].between(0, max_horizon_minutes)
    ].copy()


def label_summary(labeled: pd.DataFrame) -> pd.DataFrame:
    summary = (
        labeled["target_binary"]
        .value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="count")
    )
    summary["pct"] = summary["count"] / summary["count"].sum() * 100
    return summary


def run_labeling(
    features_path: Path = DATA_PROCESSED / "baseline_smartwatch_features.parquet",
    weekly_path: Path = DATA_RAW / "anonym_aamos00_weeklyquestionnaire.csv",
    # NOTE: this legacy weekly-symptom labelling pipeline is kept ONLY for
    # ablation / leakage-probe comparison.  The canonical labelled parquet
    # consumed by Tier-1/2/3 (`baseline_smartwatch_features_labeled.parquet`)
    # is now produced by `src.event_labels.run_window_event_labeling`.  We
    # write to a separate filename here to avoid silently overwriting it.
    parquet_path: Path = DATA_PROCESSED / "baseline_smartwatch_features_labeled_weekly.parquet",
    csv_path: Path = DATA_PROCESSED / "baseline_smartwatch_features_labeled_weekly.csv",
    summary_path: Path = OUTPUT_TABLES / "label_distribution_summary_weekly.csv",
) -> pd.DataFrame:
    weekly = prepare_weekly_flags(load_weekly_questionnaire(weekly_path))
    labeled = label_feature_windows(pd.read_parquet(features_path), weekly)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(parquet_path, index=False)
    labeled.to_csv(csv_path, index=False)
    label_summary(labeled).to_csv(summary_path, index=False)
    return labeled


if __name__ == "__main__":
    data = run_labeling()
    print(f"Saved labeled feature table: {data.shape}")
