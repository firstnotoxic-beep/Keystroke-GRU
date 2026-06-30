"""Central configuration for the Keystroke Dynamics pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# --- Password & feature dimensions ---
PASSWORD = ".tie5Roanl"
NUM_KEYS = 11
NUM_HOLD = 11
NUM_DD = 10
NUM_UD = 10
NUM_FEATURES = 31

# --- Column names ---
H_COLS = [f"H{i}" for i in range(1, NUM_HOLD + 1)]
DD_COLS = [f"DD{i}" for i in range(1, NUM_DD + 1)]
UD_COLS = [f"UD{i}" for i in range(1, NUM_UD + 1)]
FEAT_COLS = H_COLS + DD_COLS + UD_COLS
META_COLS = ["Subject_Name", "Label"]
REQUIRED_COLS = META_COLS + FEAT_COLS
OUTPUT_COLS = REQUIRED_COLS

# --- Paths (pathlib, cross-platform) ---
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "real_keystroke_data.csv"
NORMALIZED_DATA_PATH = PROJECT_ROOT / "normalized_keystroke_data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "keystroke_gru_model.h5"
DSL_DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "DSL-StrongPasswordData.csv"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

_INVALID_FILENAME_CHARS = r'\/:*?"<>|'


def normalize_subject_name(subject_name: str) -> str:
    """Normalize Subject_Name — lowercase + strip for Windows case-insensitivity."""
    return subject_name.strip().lower()


def subject_raw_path(subject_name: str) -> Path:
    """Return per-subject raw CSV path under RAW_DATA_DIR."""
    safe = normalize_subject_name(subject_name)
    for ch in _INVALID_FILENAME_CHARS:
        safe = safe.replace(ch, "_")
    return RAW_DATA_DIR / f"raw_data_{safe}.csv"

# --- Split ratios ---
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 7


def validate_csv_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and keep only rows with Label in {0, 1}."""
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {', '.join(missing)}")
        sys.exit(1)

    cleaned = df.copy()
    cleaned["Label"] = pd.to_numeric(cleaned["Label"], errors="coerce")
    invalid = ~cleaned["Label"].isin([0, 1])
    n_invalid = int(invalid.sum())
    if n_invalid:
        print(f"WARNING: Dropping {n_invalid} row(s) with Label not in {{0, 1}}")
        cleaned = cleaned.loc[~invalid]

    if cleaned.empty:
        print("ERROR: No valid rows remaining after label filtering.")
        sys.exit(1)

    return cleaned.reset_index(drop=True)
