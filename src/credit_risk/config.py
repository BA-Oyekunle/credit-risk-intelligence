from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Data directories
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

EXTERNAL_DATA_DIR = DATA_DIR / "external"


# Dataset paths
RAW_DATA_PATH = RAW_DATA_DIR / "loan_data.csv"

PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "loan_data.parquet"


# Ensure data directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)