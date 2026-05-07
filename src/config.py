"""Project paths and dataset configuration.

Environment overrides are useful for Colab/Google Drive runs:

- AAMOS_DATA_DIR: folder containing the full AAMOS CSV files.
- AAMOS_OUTPUT_DIR: folder where tables/figures/logs are written.
- AAMOS_INTERIM_DIR / AAMOS_PROCESSED_DIR: optional generated data folders.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_INTERIM = Path(os.environ.get("AAMOS_INTERIM_DIR", DATA_ROOT / "interim")).expanduser()
DATA_PROCESSED = Path(
    os.environ.get("AAMOS_PROCESSED_DIR", DATA_ROOT / "processed")
).expanduser()
DATA_METADATA = PROJECT_ROOT / "data" / "metadata"

OUTPUT_ROOT = Path(os.environ.get("AAMOS_OUTPUT_DIR", PROJECT_ROOT / "outputs")).expanduser()
OUTPUT_TABLES = OUTPUT_ROOT / "tables"
OUTPUT_FIGURES = OUTPUT_ROOT / "figures"
OUTPUT_LOGS = OUTPUT_ROOT / "logs"

DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_DIR = Path(
    os.environ.get(
        "AAMOS_DATA_DIR",
        DEFAULT_DATASET_DIR if DEFAULT_DATASET_DIR.exists() else DATA_RAW,
    )
).expanduser()

SMARTWATCH_FILES = [
    DATA_RAW / "anonym_aamos00_smartwatch1.csv",
    DATA_RAW / "anonym_aamos00_smartwatch2.csv",
    DATA_RAW / "anonym_aamos00_smartwatch3.csv",
]

PATIENT_INFO_FILE = DATA_RAW / "anonym_aamos00_patient_info.csv"
WEEKLY_FILE = DATA_RAW / "anonym_aamos00_weeklyquestionnaire.csv"

AAMOS_DAILY_FILE = DATASET_DIR / "anonym_aamos00_dailyquestionnaire.csv"
AAMOS_WEEKLY_FILE = DATASET_DIR / "anonym_aamos00_weeklyquestionnaire.csv"
AAMOS_PATIENT_INFO_FILE = DATASET_DIR / "anonym_aamos00_patient_info.csv"
AAMOS_SMARTINHALER_FILE = DATASET_DIR / "anonym_aamos00_smartinhaler.csv"
AAMOS_PEAKFLOW_FILE = DATASET_DIR / "anonym_aamos00_peakflow.csv"
AAMOS_ENVIRONMENT_FILE = DATASET_DIR / "anonym_aamos00_environment.csv"
AAMOS_SMARTWATCH_FILES = [
    DATASET_DIR / "anonym_aamos00_smartwatch1.csv",
    DATASET_DIR / "anonym_aamos00_smartwatch2.csv",
    DATASET_DIR / "anonym_aamos00_smartwatch3.csv",
]
