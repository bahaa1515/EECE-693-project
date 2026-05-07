from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import clean_smartwatch_data, load_raw_smartwatch
from .config import (
    AAMOS_ENVIRONMENT_FILE,
    AAMOS_PATIENT_INFO_FILE,
    AAMOS_PEAKFLOW_FILE,
    AAMOS_SMARTINHALER_FILE,
    AAMOS_SMARTWATCH_FILES,
    DATA_PROCESSED,
    OUTPUT_TABLES,
)
from .event_labels import USER_COL
from .features import ACTIVITY_GROUPS, add_activity_features


DATE_COL = "date"
POLLEN_LEVELS = {"low": 1, "moderate": 2, "high": 3, "very high": 4}
STATIC_CATEGORICAL_COLUMNS = [
    "sex",
    "age_range",
    "bmi_range",
    "age_diagnosed_range",
    "smoker",
    "race",
    "severity",
    "region",
    "nation",
]
STATIC_NUMERIC_COLUMNS = [
    "max_pef_expected",
    "pef_best",
    "n_inhalers",
    "pack_years",
    "phase2_start",
    "daily_start_date",
    "weekly_start_date",
    "miband_start_date",
    "pef_start_date",
    "inhaler_start_date",
    "daily_end_date",
    "weekly_end_date",
    "miband_end_date",
    "pef_end_date",
    "inhaler_end_date",
    "phase2_end",
]


def _valid_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_numeric(out[DATE_COL], errors="coerce")
    out = out.dropna(subset=[USER_COL, DATE_COL]).copy()
    out[USER_COL] = out[USER_COL].astype(int)
    out[DATE_COL] = out[DATE_COL].astype(int)
    return out[out[DATE_COL].ge(0)].copy()


def build_smartwatch_daily_features(
    smartwatch_files: list[Path] | None = None,
) -> pd.DataFrame:
    raw = load_raw_smartwatch(smartwatch_files or AAMOS_SMARTWATCH_FILES)
    clean = clean_smartwatch_data(raw)
    df = add_activity_features(clean)

    group = df.groupby([USER_COL, DATE_COL], observed=True)
    daily = group.agg(
        smartwatch_rows=(USER_COL, "size"),
        smartwatch_observed_minutes=("relative_minute", "nunique"),
        smartwatch_hr_count=("hr", "count"),
        smartwatch_hr_mean=("hr", "mean"),
        smartwatch_hr_std=("hr", "std"),
        smartwatch_hr_min=("hr", "min"),
        smartwatch_hr_max=("hr", "max"),
        smartwatch_hr_median=("hr", "median"),
        smartwatch_steps_sum=("steps", "sum"),
        smartwatch_steps_mean=("steps", "mean"),
        smartwatch_steps_max=("steps", "max"),
        smartwatch_intensity_mean=("intensity", "mean"),
        smartwatch_intensity_std=("intensity", "std"),
        smartwatch_intensity_max=("intensity", "max"),
        smartwatch_active_minute_frac=("is_active_minute", "mean"),
    ).reset_index()
    daily["smartwatch_day_coverage"] = (
        daily["smartwatch_observed_minutes"] / 1440
    ).clip(upper=1)
    daily["smartwatch_hr_missing_pct"] = (
        1 - daily["smartwatch_hr_count"] / 1440
    ).clip(lower=0, upper=1)

    activity_counts = (
        df.groupby([USER_COL, DATE_COL, "activity_group"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for group_name in ACTIVITY_GROUPS:
        if group_name not in activity_counts.columns:
            activity_counts[group_name] = 0
    activity_counts = activity_counts[ACTIVITY_GROUPS]
    activity_frac = activity_counts.div(activity_counts.sum(axis=1), axis=0).fillna(0)
    activity_frac.columns = [
        f"smartwatch_activity_frac_{col}" for col in activity_frac.columns
    ]
    daily = daily.merge(activity_frac.reset_index(), on=[USER_COL, DATE_COL], how="left")

    day_part = (
        df[df["is_day"].eq(1)]
        .groupby([USER_COL, DATE_COL], observed=True)
        .agg(
            smartwatch_day_hr_mean=("hr", "mean"),
            smartwatch_day_steps_sum=("steps", "sum"),
            smartwatch_day_active_frac=("is_active_minute", "mean"),
        )
        .reset_index()
    )
    night_part = (
        df[df["is_night"].eq(1)]
        .groupby([USER_COL, DATE_COL], observed=True)
        .agg(
            smartwatch_night_hr_mean=("hr", "mean"),
            smartwatch_night_steps_sum=("steps", "sum"),
            smartwatch_night_active_frac=("is_active_minute", "mean"),
        )
        .reset_index()
    )
    daily = daily.merge(day_part, on=[USER_COL, DATE_COL], how="left")
    daily = daily.merge(night_part, on=[USER_COL, DATE_COL], how="left")
    return daily.sort_values([USER_COL, DATE_COL]).reset_index(drop=True)


def build_smartinhaler_daily_features(
    path: Path = AAMOS_SMARTINHALER_FILE,
) -> pd.DataFrame:
    df = _valid_date(pd.read_csv(path))
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df = df[df["hour"].between(0, 23)].copy()
    group = df.groupby([USER_COL, DATE_COL], observed=True)
    daily = group.agg(
        smartinhaler_total_actuations=("name", "size"),
        smartinhaler_unique_medications=("name", "nunique"),
        smartinhaler_night_early_actuations=("hour", lambda s: int(s.between(0, 5).sum())),
        smartinhaler_evening_actuations=("hour", lambda s: int(s.between(18, 23).sum())),
    ).reset_index()
    daily["smartinhaler_active_day"] = 1
    return daily.sort_values([USER_COL, DATE_COL]).reset_index(drop=True)


def build_peakflow_daily_features(
    peakflow_path: Path = AAMOS_PEAKFLOW_FILE,
    patient_path: Path = AAMOS_PATIENT_INFO_FILE,
) -> pd.DataFrame:
    df = _valid_date(pd.read_csv(peakflow_path))
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["pef_max"] = pd.to_numeric(df["pef_max"], errors="coerce")
    df = df[df["hour"].between(0, 23) & df["pef_max"].gt(0)].copy()

    patient = pd.read_csv(patient_path)[
        [USER_COL, "pef_best", "max_pef_expected"]
    ].copy()
    patient["pef_baseline"] = pd.to_numeric(
        patient["pef_best"], errors="coerce"
    ).fillna(pd.to_numeric(patient["max_pef_expected"], errors="coerce"))
    df = df.merge(patient[[USER_COL, "pef_baseline"]], on=USER_COL, how="left")
    df["peakflow_pef_pct_baseline"] = df["pef_max"] / df["pef_baseline"]
    df["morning"] = df["morning"].astype(bool)

    group = df.groupby([USER_COL, DATE_COL], observed=True)
    daily = group.agg(
        peakflow_measurement_count=("pef_max", "count"),
        peakflow_pef_mean=("pef_max", "mean"),
        peakflow_pef_std=("pef_max", "std"),
        peakflow_pef_min=("pef_max", "min"),
        peakflow_pef_max=("pef_max", "max"),
        peakflow_pct_baseline_mean=("peakflow_pef_pct_baseline", "mean"),
        peakflow_pct_baseline_min=("peakflow_pef_pct_baseline", "min"),
        peakflow_pct_baseline_max=("peakflow_pef_pct_baseline", "max"),
    ).reset_index()

    morning = (
        df[df["morning"]]
        .groupby([USER_COL, DATE_COL], observed=True)
        .agg(
            peakflow_morning_count=("pef_max", "count"),
            peakflow_morning_pef_mean=("pef_max", "mean"),
            peakflow_morning_pef_max=("pef_max", "max"),
        )
        .reset_index()
    )
    evening = (
        df[~df["morning"]]
        .groupby([USER_COL, DATE_COL], observed=True)
        .agg(
            peakflow_evening_count=("pef_max", "count"),
            peakflow_evening_pef_mean=("pef_max", "mean"),
            peakflow_evening_pef_max=("pef_max", "max"),
        )
        .reset_index()
    )
    daily = daily.merge(morning, on=[USER_COL, DATE_COL], how="left")
    daily = daily.merge(evening, on=[USER_COL, DATE_COL], how="left")
    return daily.sort_values([USER_COL, DATE_COL]).reset_index(drop=True)


def _encode_pollen(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().map(POLLEN_LEVELS)


def build_environment_daily_features(
    path: Path = AAMOS_ENVIRONMENT_FILE,
) -> pd.DataFrame:
    df = _valid_date(pd.read_csv(path))
    for col in ["grass_pollen", "tree_pollen", "weed_pollen"]:
        df[f"{col}_level"] = _encode_pollen(df[col])

    numeric_cols = [
        "temperature",
        "temperature_min",
        "temperature_max",
        "pressure",
        "humidity",
        "wind_speed",
        "wind_deg",
        "aqi",
        "co",
        "no",
        "no2",
        "o3",
        "so2",
        "pm2_5",
        "pm10",
        "nh3",
        "grass_pollen_level",
        "tree_pollen_level",
        "weed_pollen_level",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    daily = (
        df.groupby([USER_COL, DATE_COL], observed=True)[numeric_cols]
        .mean()
        .reset_index()
    )
    daily = daily.rename(columns={col: f"environment_{col}" for col in numeric_cols})
    for col in [
        "environment_grass_pollen_level",
        "environment_tree_pollen_level",
        "environment_weed_pollen_level",
    ]:
        daily[f"{col}_high"] = daily[col].ge(3).astype(int)
    return daily.sort_values([USER_COL, DATE_COL]).reset_index(drop=True)


def build_patient_static_features(
    path: Path = AAMOS_PATIENT_INFO_FILE,
) -> pd.DataFrame:
    patient = pd.read_csv(path).drop_duplicates(subset=[USER_COL]).copy()
    available_numeric = [col for col in STATIC_NUMERIC_COLUMNS if col in patient.columns]
    available_categorical = [
        col for col in STATIC_CATEGORICAL_COLUMNS if col in patient.columns
    ]
    for col in available_numeric:
        patient[col] = pd.to_numeric(patient[col], errors="coerce")
    numeric = patient[[USER_COL, *available_numeric]].copy()
    numeric = numeric.rename(
        columns={col: f"patient_{col}" for col in available_numeric}
    )
    categorical = pd.get_dummies(
        patient[[USER_COL, *available_categorical]],
        columns=available_categorical,
        dummy_na=True,
        prefix=[f"patient_{col}" for col in available_categorical],
        dtype=int,
    )
    return numeric.merge(categorical, on=USER_COL, how="left")


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in {USER_COL, DATE_COL}
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def aggregate_source_window(
    source_daily: pd.DataFrame,
    sample_index: pd.DataFrame,
    source_name: str,
    zero_if_empty: bool = False,
) -> pd.DataFrame:
    feature_cols = _numeric_feature_columns(source_daily)
    frames = {
        int(user): group.sort_values(DATE_COL)
        for user, group in source_daily.groupby(USER_COL, observed=True)
    }
    rows: list[dict[str, float]] = []

    for sample in sample_index.itertuples(index=False):
        user_key = int(sample.user_key)
        start = int(sample.window_start_day)
        end = int(sample.window_end_day)
        length = int(sample.input_length_days)
        user_frame = frames.get(user_key)
        if user_frame is None:
            window = pd.DataFrame(columns=source_daily.columns)
        else:
            window = user_frame[user_frame[DATE_COL].between(start, end)]

        row: dict[str, float] = {
            f"{source_name}_days_observed": float(window[DATE_COL].nunique()),
            f"{source_name}_day_coverage": float(window[DATE_COL].nunique() / length),
        }
        for col in feature_cols:
            values = pd.to_numeric(window[col], errors="coerce")
            if values.dropna().empty:
                fill_value = 0.0 if zero_if_empty else np.nan
                row[f"{col}_window_sum"] = fill_value
                row[f"{col}_window_mean"] = fill_value
                row[f"{col}_window_std"] = fill_value
                row[f"{col}_window_min"] = fill_value
                row[f"{col}_window_max"] = fill_value
                continue
            row[f"{col}_window_sum"] = float(values.sum())
            row[f"{col}_window_mean"] = float(values.mean())
            row[f"{col}_window_std"] = float(values.std(ddof=0))
            row[f"{col}_window_min"] = float(values.min())
            row[f"{col}_window_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def build_daily_feature_tables() -> dict[str, pd.DataFrame]:
    return {
        "smartwatch": build_smartwatch_daily_features(),
        "smartinhaler": build_smartinhaler_daily_features(),
        "peakflow": build_peakflow_daily_features(),
        "environment": build_environment_daily_features(),
    }


def write_daily_feature_tables(
    daily_tables: dict[str, pd.DataFrame],
    output_dir: Path = DATA_PROCESSED,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in daily_tables.items():
        table.to_parquet(output_dir / f"{name}_daily_features.parquet", index=False)
        table.to_csv(output_dir / f"{name}_daily_features.csv", index=False)


def build_training_feature_table(
    sample_index: pd.DataFrame,
    daily_tables: dict[str, pd.DataFrame],
    patient_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if sample_index.empty:
        return sample_index.copy()

    table = sample_index.reset_index(drop=True).copy()
    source_configs = {
        "smartwatch": False,
        "smartinhaler": True,
        "peakflow": False,
        "environment": False,
    }
    feature_blocks = []
    for source_name, zero_if_empty in source_configs.items():
        source_daily = daily_tables[source_name]
        feature_blocks.append(
            aggregate_source_window(
                source_daily=source_daily,
                sample_index=table,
                source_name=source_name,
                zero_if_empty=zero_if_empty,
            )
        )

    table = pd.concat([table, *feature_blocks], axis=1)
    patient_features = patient_features if patient_features is not None else build_patient_static_features()
    table = table.merge(patient_features, on=USER_COL, how="left")
    return table


def feature_table_filename(threshold: int, input_length_days: int, washout_days: int) -> str:
    return (
        "sensor_features_"
        f"L{input_length_days}_threshold{threshold}_washout{washout_days}.parquet"
    )


def build_and_write_feature_tables(
    sample_indexes: dict[tuple[int, int, int], pd.DataFrame],
    output_dir: Path = DATA_PROCESSED,
    table_dir: Path = OUTPUT_TABLES,
) -> tuple[dict[str, pd.DataFrame], dict[tuple[int, int, int], Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    daily_tables = build_daily_feature_tables()
    write_daily_feature_tables(daily_tables, output_dir)
    patient_features = build_patient_static_features()
    patient_features.to_parquet(output_dir / "patient_static_features.parquet", index=False)
    patient_features.to_csv(output_dir / "patient_static_features.csv", index=False)

    paths: dict[tuple[int, int, int], Path] = {}
    coverage_rows = []
    for key, sample_index in sample_indexes.items():
        threshold, input_length_days, washout_days = key
        feature_table = build_training_feature_table(
            sample_index=sample_index,
            daily_tables=daily_tables,
            patient_features=patient_features,
        )
        path = output_dir / feature_table_filename(
            threshold, input_length_days, washout_days
        )
        feature_table.to_parquet(path, index=False)
        paths[key] = path
        coverage_cols = [col for col in feature_table.columns if col.endswith("_day_coverage")]
        row = {
            "threshold": threshold,
            "input_length_days": input_length_days,
            "washout_days": washout_days,
            "rows": len(feature_table),
        }
        for col in coverage_cols:
            row[f"{col}_mean"] = feature_table[col].mean()
        coverage_rows.append(row)

    pd.DataFrame(coverage_rows).to_csv(
        table_dir / "sensor_feature_coverage_summary.csv", index=False
    )
    return daily_tables, paths
