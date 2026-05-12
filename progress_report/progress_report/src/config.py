#keeps all important paths and settings in one place.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_METADATA = PROJECT_ROOT / "data" / "metadata"

OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_LOGS = PROJECT_ROOT / "outputs" / "logs"

SMARTWATCH_FILES = [
    DATA_RAW / "anonym_aamos00_smartwatch1.csv",
    DATA_RAW / "anonym_aamos00_smartwatch2.csv",
    DATA_RAW / "anonym_aamos00_smartwatch3.csv",
]

PATIENT_INFO_FILE = DATA_RAW / "anonym_aamos00_patient_info.csv"
WEEKLY_FILE = DATA_RAW / "anonym_aamos00_weeklyquestionnaire.csv"