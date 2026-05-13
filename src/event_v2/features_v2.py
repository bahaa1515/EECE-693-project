"""Feature builders for v2 event-episode samples.

Re-uses the daily-aggregation helpers from :mod:`src.event_features`
(no point duplicating ~500 lines of raw CSV plumbing), but exposes
v2-only functions that:

* Accept a ``sensor_sources`` parameter so we can fit on subsets
  (e.g. {"smartwatch"} only, or {"smartwatch", "smartinhaler"}).
* Write coverage / feature manifests under ``outputs/v2/tables``.
* Write feature parquets under ``data/processed/v2``.

The legacy module is **not** modified.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.event_features import (
    DATE_COL,
    aggregate_source_window,
    build_daily_feature_tables,
    build_environment_daily_features,
    build_patient_static_features,
    build_peakflow_daily_features,
    build_smartinhaler_daily_features,
    build_smartwatch_daily_features,
)
from src.event_labels import USER_COL
from . import DATA_PROCESSED_V2, OUTPUT_TABLES_V2

ALL_SENSOR_SOURCES = ("smartwatch", "smartinhaler", "peakflow", "environment")

# zero-imputation flags carried over from the legacy module so behaviour
# is identical when ``sensor_sources == ALL_SENSOR_SOURCES``.
SOURCE_ZERO_IF_EMPTY = {
    "smartwatch": False,
    "smartinhaler": True,
    "peakflow": False,
    "environment": False,
}


def _normalize_sources(sensor_sources: Iterable[str] | None) -> tuple[str, ...]:
    if sensor_sources is None:
        return ALL_SENSOR_SOURCES
    requested = tuple(sensor_sources)
    unknown = [s for s in requested if s not in ALL_SENSOR_SOURCES]
    if unknown:
        raise ValueError(f"Unknown sensor sources: {unknown!r}")
    # preserve declared order from ALL_SENSOR_SOURCES for stable column order
    return tuple(s for s in ALL_SENSOR_SOURCES if s in requested)


def build_training_feature_table(
    sample_index: pd.DataFrame,
    daily_tables: dict[str, pd.DataFrame],
    patient_features: pd.DataFrame | None = None,
    sensor_sources: Iterable[str] | None = None,
    include_patient_static: bool = True,
) -> pd.DataFrame:
    """Build a feature matrix from a sample index and pre-built daily tables.

    Parameters
    ----------
    sample_index : sample manifest from :func:`samples_v2.build_sample_index`.
    daily_tables : dict mapping source-name → daily feature DataFrame.
    patient_features : pre-built patient-static frame (built lazily if None
        and ``include_patient_static=True``).
    sensor_sources : iterable of source names to include
        (default: all four).
    include_patient_static : whether to merge patient-static features.
    """
    if sample_index.empty:
        return sample_index.copy()

    sources = _normalize_sources(sensor_sources)
    table = sample_index.reset_index(drop=True).copy()

    feature_blocks = []
    for source_name in sources:
        if source_name not in daily_tables:
            raise KeyError(f"Missing daily table for source: {source_name!r}")
        feature_blocks.append(
            aggregate_source_window(
                source_daily=daily_tables[source_name],
                sample_index=table,
                source_name=source_name,
                zero_if_empty=SOURCE_ZERO_IF_EMPTY[source_name],
            )
        )

    table = pd.concat([table, *feature_blocks], axis=1)

    if include_patient_static:
        patient = (
            patient_features
            if patient_features is not None
            else build_patient_static_features()
        )
        table = table.merge(patient, on=USER_COL, how="left")
    return table


def feature_table_filename_v2(
    threshold: int,
    input_length_days: int,
    washout_days: int,
    sensor_tag: str = "all",
) -> str:
    return (
        f"sensor_features_v2_T{threshold}_L{input_length_days}"
        f"_W{washout_days}_{sensor_tag}.parquet"
    )


def _sensor_tag(sensor_sources: Iterable[str] | None) -> str:
    sources = _normalize_sources(sensor_sources)
    if set(sources) == set(ALL_SENSOR_SOURCES):
        return "all"
    return "+".join(sources)


def build_and_write_feature_tables_v2(
    sample_indexes: dict[tuple[int, int, int], pd.DataFrame],
    daily_tables: dict[str, pd.DataFrame] | None = None,
    patient_features: pd.DataFrame | None = None,
    sensor_sources: Iterable[str] | None = None,
    output_dir: Path = DATA_PROCESSED_V2,
) -> tuple[dict[str, pd.DataFrame], dict[tuple[int, int, int], Path]]:
    """Build feature tables for many ``(T, L, W)`` triples; write parquets.

    Returns the daily tables (cached for re-use) and a mapping
    ``(T, L, W)`` → parquet path.
    """
    daily_tables = daily_tables if daily_tables is not None else build_daily_feature_tables()
    patient_features = (
        patient_features
        if patient_features is not None
        else build_patient_static_features()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = _sensor_tag(sensor_sources)

    paths: dict[tuple[int, int, int], Path] = {}
    coverage_rows: list[dict[str, float]] = []

    for key, sample_index in sample_indexes.items():
        threshold, input_length_days, washout_days = key
        feature_table = build_training_feature_table(
            sample_index=sample_index,
            daily_tables=daily_tables,
            patient_features=patient_features,
            sensor_sources=sensor_sources,
        )
        path = output_dir / feature_table_filename_v2(
            threshold, input_length_days, washout_days, sensor_tag=tag
        )
        feature_table.to_parquet(path, index=False)
        paths[key] = path

        coverage_cols = [c for c in feature_table.columns if c.endswith("_day_coverage")]
        row = {
            "sensor_tag": tag,
            "threshold": threshold,
            "input_length_days": input_length_days,
            "washout_days": washout_days,
            "rows": len(feature_table),
            "positive_rows": int(feature_table["target"].eq(1).sum()),
            "negative_rows": int(feature_table["target"].eq(0).sum()),
        }
        for col in coverage_cols:
            row[f"{col}_mean"] = float(feature_table[col].mean())
        coverage_rows.append(row)

    coverage_path = OUTPUT_TABLES_V2 / f"sensor_feature_coverage_v2_{tag}.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
    return daily_tables, paths


__all__ = [
    "ALL_SENSOR_SOURCES",
    "build_training_feature_table",
    "build_and_write_feature_tables_v2",
    "feature_table_filename_v2",
]
