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
