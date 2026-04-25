from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_INTERIM, DATA_PROCESSED, OUTPUT_TABLES


HISTORY_HOURS = 24
STRIDE_HOURS = 1
MIN_COVERAGE = 0.70
WINDOW_MINUTES = HISTORY_HOURS * 60
STRIDE_MINUTES = STRIDE_HOURS * 60
DAY_START = 6 * 60
DAY_END = 22 * 60

ACTIVITY_MAP = {
    1: "walk",
    16: "walk",
    33: "walk",
    3: "not_worn",
    6: "charging",
    80: "sedentary",
    89: "sedentary",
    90: "sedentary",
    91: "sedentary",
    92: "sedentary",
    96: "sedentary",
    66: "running",
    82: "running",
    98: "running",
    17: "activity_high",
    106: "sleep",
    112: "sleep",
    121: "sleep",
    122: "sleep",
    123: "sleep",
}

ACTIVITY_GROUPS = [
    "walk",
    "running",
    "activity_high",
    "sedentary",
    "sleep",
    "not_worn",
    "charging",
    "unknown",
]

FEATURE_COLUMNS = [
    "observed_minutes",
    "coverage",
    "n_rows_in_window",
    "hr_mean",
    "hr_std",
    "hr_min",
    "hr_max",
    "hr_median",
    "hr_missing_pct",
    "steps_sum",
    "steps_mean",
    "steps_max",
    "intensity_mean",
    "intensity_std",
    "intensity_max",
    "active_minute_frac",
    "activity_frac_walk",
    "activity_frac_running",
    "activity_frac_activity_high",
    "activity_frac_sedentary",
    "activity_frac_sleep",
    "activity_frac_not_worn",
    "activity_frac_charging",
    "activity_frac_unknown",
    "day_hr_mean",
    "night_hr_mean",
    "day_steps_sum",
    "night_steps_sum",
    "day_active_frac",
    "night_active_frac",
    "hr_slope",
    "steps_slope",
]


def add_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["activity_type"] = pd.to_numeric(out["activity_type"], errors="coerce")
    out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce")
    out["activity_group"] = out["activity_type"].map(ACTIVITY_MAP).fillna("unknown")
    out["is_active_minute"] = out["activity_group"].isin(
        ["walk", "running", "activity_high"]
    ).astype(int)
    out["is_day"] = (
        (out["minutes_from_midnight"] >= DAY_START)
        & (out["minutes_from_midnight"] < DAY_END)
    ).astype(int)
    out["is_night"] = 1 - out["is_day"]
    return out


def safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.all(x == x[0]):
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def extract_window_features(
    window: pd.DataFrame,
    user_key: int,
    anchor_minute: int,
    window_minutes: int = WINDOW_MINUTES,
) -> dict[str, float] | None:
    observed_minutes = window["relative_minute"].nunique()
    coverage = observed_minutes / window_minutes
    if coverage < MIN_COVERAGE:
        return None

    hr_valid = window["hr"].dropna()
    steps_valid = window["steps"].dropna()
    intensity_valid = window["intensity"].dropna()

    result = {
        "user_key": user_key,
        "anchor_relative_minute": anchor_minute,
        "window_start_minute": anchor_minute - window_minutes,
        "window_end_minute": anchor_minute,
        "observed_minutes": observed_minutes,
        "coverage": coverage,
        "n_rows_in_window": len(window),
        "hr_mean": hr_valid.mean() if len(hr_valid) else np.nan,
        "hr_std": hr_valid.std() if len(hr_valid) else np.nan,
        "hr_min": hr_valid.min() if len(hr_valid) else np.nan,
        "hr_max": hr_valid.max() if len(hr_valid) else np.nan,
        "hr_median": hr_valid.median() if len(hr_valid) else np.nan,
        "hr_missing_pct": 1 - (window["hr"].notna().sum() / window_minutes),
        "steps_sum": steps_valid.sum() if len(steps_valid) else np.nan,
        "steps_mean": steps_valid.mean() if len(steps_valid) else np.nan,
        "steps_max": steps_valid.max() if len(steps_valid) else np.nan,
        "intensity_mean": intensity_valid.mean() if len(intensity_valid) else np.nan,
        "intensity_std": intensity_valid.std() if len(intensity_valid) else np.nan,
        "intensity_max": intensity_valid.max() if len(intensity_valid) else np.nan,
        "active_minute_frac": window["is_active_minute"].mean(),
    }

    for group in ACTIVITY_GROUPS:
        result[f"activity_frac_{group}"] = (window["activity_group"] == group).mean()

    day_w = window[window["is_day"] == 1]
    night_w = window[window["is_night"] == 1]
    result["day_hr_mean"] = day_w["hr"].mean()
    result["night_hr_mean"] = night_w["hr"].mean()
    result["day_steps_sum"] = day_w["steps"].sum(min_count=1)
    result["night_steps_sum"] = night_w["steps"].sum(min_count=1)
    result["day_active_frac"] = day_w["is_active_minute"].mean() if len(day_w) else np.nan
    result["night_active_frac"] = (
        night_w["is_active_minute"].mean() if len(night_w) else np.nan
    )

    rel_x = window["relative_minute"] - window["relative_minute"].min()
    result["hr_slope"] = safe_slope(rel_x.values, window["hr"].values)
    result["steps_slope"] = safe_slope(rel_x.values, window["steps"].values)
    return result


def build_feature_table(cleaned: pd.DataFrame) -> pd.DataFrame:
    df = add_activity_features(cleaned)
    rows = []
    for user_key, user_df in df.groupby("user_key"):
        user_df = user_df.sort_values("relative_minute").copy()
        min_anchor = user_df["relative_minute"].min() + WINDOW_MINUTES
        max_anchor = user_df["relative_minute"].max()
        anchors = np.arange(min_anchor, max_anchor + 1, STRIDE_MINUTES)
        for anchor in anchors:
            window = user_df[
                (user_df["relative_minute"] >= anchor - WINDOW_MINUTES)
                & (user_df["relative_minute"] < anchor)
            ]
            if len(window) == 0:
                continue
            row = extract_window_features(window, int(user_key), int(anchor))
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    summary = features.describe().T
    summary["missing_pct"] = features.isna().mean() * 100
    return summary.sort_index()


def run_feature_engineering(
    cleaned_path: Path = DATA_INTERIM / "smartwatch_cleaned.parquet",
    parquet_path: Path = DATA_PROCESSED / "baseline_smartwatch_features.parquet",
    csv_path: Path = DATA_PROCESSED / "baseline_smartwatch_features.csv",
    summary_path: Path = OUTPUT_TABLES / "baseline_feature_summary.csv",
) -> pd.DataFrame:
    features = build_feature_table(pd.read_parquet(cleaned_path))
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(parquet_path, index=False)
    features.to_csv(csv_path, index=False)
    feature_summary(features).to_csv(summary_path)
    return features


if __name__ == "__main__":
    data = run_feature_engineering()
    print(f"Saved feature table: {data.shape}")
