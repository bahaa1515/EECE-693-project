from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_INTERIM, OUTPUT_TABLES, SMARTWATCH_FILES


SMARTWATCH_USECOLS = [
    "user_key",
    "date",
    "time",
    "activity_type",
    "intensity",
    "steps",
    "hr",
]

SMARTWATCH_DTYPES = {
    "user_key": "int32",
    "date": "int16",
    "activity_type": "float32",
    "intensity": "float32",
    "steps": "float32",
    "hr": "float32",
}


def load_raw_smartwatch(files: list[Path] | None = None) -> pd.DataFrame:
    """Load and concatenate the raw smartwatch CSV files used in the notebooks."""
    files = files or SMARTWATCH_FILES
    parts = [
        pd.read_csv(path, usecols=SMARTWATCH_USECOLS, dtype=SMARTWATCH_DTYPES)
        for path in files
    ]
    return pd.concat(parts, ignore_index=True)


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add minute-of-day and study-relative minute columns."""
    out = df.copy()
    parsed_time = pd.to_datetime(out["time"], format="%H:%M:%S", errors="coerce")
    out["time"] = parsed_time.dt.time
    out["minutes_from_midnight"] = parsed_time.dt.hour * 60 + parsed_time.dt.minute
    out = out.dropna(subset=["minutes_from_midnight"]).copy()
    out["date"] = out["date"].astype("int64")
    out["minutes_from_midnight"] = out["minutes_from_midnight"].astype("int64")
    out["relative_minute"] = out["date"] * 1440 + out["minutes_from_midnight"]
    return out


def clean_smartwatch_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean smartwatch data using the same rules as notebook 02."""
    df = raw.drop_duplicates().copy()
    df = df[df["date"] >= 0].copy()
    df.loc[df["steps"] < 0, "steps"] = np.nan
    df.loc[df["hr"] <= 0, "hr"] = np.nan
    df = add_time_columns(df)
    return df.sort_values(["user_key", "relative_minute"]).reset_index(drop=True)


def cleaning_log(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaning summary table saved by notebook 02."""
    return pd.DataFrame(
        {
            "metric": [
                "rows_after_cleaning",
                "unique_users",
                "missing_hr_pct",
                "missing_steps_pct",
            ],
            "value": [
                len(cleaned),
                cleaned["user_key"].nunique(),
                round(cleaned["hr"].isna().mean() * 100, 2),
                round(cleaned["steps"].isna().mean() * 100, 2),
            ],
        }
    )


def run_cleaning(
    output_path: Path = DATA_INTERIM / "smartwatch_cleaned.parquet",
    log_path: Path = OUTPUT_TABLES / "smartwatch_cleaning_log.csv",
) -> pd.DataFrame:
    """Run the full smartwatch cleaning stage and write notebook-compatible outputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = clean_smartwatch_data(load_raw_smartwatch())
    cleaned.to_parquet(output_path, index=False)
    cleaning_log(cleaned).to_csv(log_path, index=False)
    return cleaned


if __name__ == "__main__":
    data = run_cleaning()
    print(f"Saved cleaned smartwatch data: {data.shape}")
