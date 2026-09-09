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


def group_divisor(values: np.ndarray, use_abs: bool = False) -> float:
    """Single source of truth for group normalization divisor."""
    arr = np.asarray(values, dtype=np.float64)
    if use_abs:
        return float(np.abs(arr).sum())
    return float(arr.sum())


def normalize_group_values(values: np.ndarray, use_abs: bool = False) -> np.ndarray:
    """Normalize one feature-group vector; raises ZeroDivisionError if divisor is 0."""
    arr = np.asarray(values, dtype=np.float64)
    divisor = group_divisor(arr, use_abs=use_abs)
    if divisor == 0:
        raise ZeroDivisionError("group divisor is zero")
    return arr / divisor


def normalize_group(
    df: pd.DataFrame, cols: list[str], name: str, use_abs: bool = False
) -> tuple[pd.DataFrame, pd.Index]:
    """Normalize feature group by row-wise sum (or abs-sum). Returns (normalized, keep_index)."""
    vals = df[cols].to_numpy(dtype=np.float64)
    if use_abs:
        divisors = np.abs(vals).sum(axis=1)
    else:
        divisors = vals.sum(axis=1)

    zero_mask = divisors == 0
    n_zero = int(zero_mask.sum())
    if n_zero:
        print(f"Dropped {n_zero} row(s) — zero divisor in {name} group")

    keep = ~zero_mask
    keep_index = df.index[keep]
    if not keep.any():
        return pd.DataFrame(columns=cols), keep_index

    normalized = vals[keep] / divisors[keep, None]
    return pd.DataFrame(normalized, index=keep_index, columns=cols), keep_index


def normalize_feature_vector(features: np.ndarray | list[float]) -> np.ndarray:
    """Normalize one (31,) sample with the same intra-sequence ratio as CSV rows."""
    arr = np.asarray(features, dtype=np.float64).reshape(-1)
    if arr.size != NUM_FEATURES:
        raise ValueError(f"ต้องการ {NUM_FEATURES} ฟีเจอร์ แต่ได้ {arr.size}")

    h = normalize_group_values(arr[:11], use_abs=False)
    dd = normalize_group_values(arr[11:21], use_abs=False)
    ud = normalize_group_values(arr[21:31], use_abs=True)
    return np.concatenate([h, dd, ud])


def drop_negative_dd(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows with any DD < 0 before normalization."""
    neg_mask = (df[DD_COLS] < 0).any(axis=1)
    n_drop = int(neg_mask.sum())
    if n_drop:
        print(f"Dropped {n_drop} row(s) — DD < 0")
    return df.loc[~neg_mask].copy(), n_drop


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
    n_input = len(df)

    # Order: drop DD<0 FIRST, then compute sums and normalize
    df, n_dd_neg = drop_negative_dd(df)

    h_norm, keep_h = normalize_group(df, H_COLS, "H", use_abs=False)
    df = df.loc[keep_h]
    dd_norm, keep_dd = normalize_group(df, DD_COLS, "DD", use_abs=False)
    df = df.loc[keep_dd]
    h_norm = h_norm.loc[keep_dd]
    ud_norm, keep_ud = normalize_group(df, UD_COLS, "UD", use_abs=True)
    df = df.loc[keep_ud]
    h_norm = h_norm.loc[keep_ud]
    dd_norm = dd_norm.loc[keep_ud]

    df = df.copy()
    df[H_COLS] = h_norm.to_numpy()
    df[DD_COLS] = dd_norm.to_numpy()
    df[UD_COLS] = ud_norm.to_numpy()
    df = df.reset_index(drop=True)

    n_output = len(df)
    n_dropped = n_input - n_output
    print(f"\n=== Normalization summary ===")
    print(f"Input rows : {n_input}")
    print(f"Dropped    : {n_dropped}")
    print(f"Output rows: {n_output}")
    print(
        f"H  min/max: {df[H_COLS].to_numpy().min():.6f} / {df[H_COLS].to_numpy().max():.6f}"
    )
    print(
        f"DD min/max: {df[DD_COLS].to_numpy().min():.6f} / {df[DD_COLS].to_numpy().max():.6f}"
    )
    print(
        f"UD min/max: {df[UD_COLS].to_numpy().min():.6f} / {df[UD_COLS].to_numpy().max():.6f}"
    )

    save_data(df, Path(args.output))


if __name__ == "__main__":
    main()
