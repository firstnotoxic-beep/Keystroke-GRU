"""Normalize keystroke features via intra-sequence ratio."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DD_COLS,
    H_COLS,
    NORMALIZED_DATA_PATH,
    NUM_FEATURES,
    OUTPUT_COLS,
    RAW_DATA_PATH,
    UD_COLS,
    validate_csv_schema,
)


def load_data(path: Path) -> pd.DataFrame:
    """Load and validate CSV keystroke data."""
    if not path.is_file():
        print(f"ERROR: Input file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    if df.empty:
        print(f"ERROR: Input file is empty (no data rows): {path}")
        sys.exit(1)
    return validate_csv_schema(df)


def normalize_group(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    """Normalize feature group by row-wise sum ratio."""
    sums = df[cols].sum(axis=1)
    n_zero = int((sums == 0).sum())
    if n_zero:
        print(f"WARNING: {n_zero} rows have zero-sum in {name} group — replaced with 0")
    return df[cols].div(sums.replace(0, np.nan), axis=0).fillna(0)


def normalize_feature_vector(features: np.ndarray | list[float]) -> np.ndarray:
    """Normalize one (31,) sample with the same intra-sequence ratio as CSV rows."""
    arr = np.asarray(features, dtype=np.float64).reshape(-1)
    if arr.size != NUM_FEATURES:
        raise ValueError(f"ต้องการ {NUM_FEATURES} ฟีเจอร์ แต่ได้ {arr.size}")

    groups = (arr[:11], arr[11:21], arr[21:31])
    parts = []
    for group in groups:
        total = float(group.sum())
        parts.append(np.zeros_like(group) if total == 0 else group / total)
    return np.concatenate(parts)


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Write normalized CSV without index."""
    df[OUTPUT_COLS].to_csv(path, index=False)


def main() -> None:
    """Run CLI normalization pipeline."""
    parser = argparse.ArgumentParser(description="Normalize keystroke CSV")
    parser.add_argument("--input", default=str(RAW_DATA_PATH))
    parser.add_argument("--output", default=str(NORMALIZED_DATA_PATH))
    args = parser.parse_args()
    df = load_data(Path(args.input))
    df[H_COLS] = normalize_group(df, H_COLS, "H")
    df[DD_COLS] = normalize_group(df, DD_COLS, "DD")
    df[UD_COLS] = normalize_group(df, UD_COLS, "UD")
    save_data(df, Path(args.output))


if __name__ == "__main__":
    main()
