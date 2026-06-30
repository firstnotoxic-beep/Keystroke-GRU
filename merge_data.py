"""Merge per-subject raw CSV files into a single dataset for preprocessing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from config import (
    OUTPUT_COLS,
    RAW_DATA_DIR,
    RAW_DATA_PATH,
    validate_csv_schema,
)


def merge_raw_files(raw_dir: Path, output_path: Path) -> pd.DataFrame:
    """Read, validate, and concatenate all raw_data_*.csv files in raw_dir."""
    csv_files = sorted(raw_dir.glob("raw_data_*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {raw_dir}")
        sys.exit(1)

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path)
        if df.empty:
            print(f"WARNING: Skipping empty file: {path.name}")
            continue
        validated = validate_csv_schema(df)
        frames.append(validated)
        print(f"  {path.name}: {len(validated)} row(s)")

    if not frames:
        print("ERROR: No valid data rows found in any raw file.")
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged[OUTPUT_COLS].to_csv(output_path, index=False)
    return merged


def main() -> None:
    """Run CLI merge pipeline."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Merge per-subject raw keystroke CSV files"
    )
    parser.add_argument(
        "--input-dir",
        default=str(RAW_DATA_DIR),
        help="Directory containing raw_data_*.csv files",
    )
    parser.add_argument(
        "--output",
        default=str(RAW_DATA_PATH),
        help="Merged output CSV path",
    )
    args = parser.parse_args()

    raw_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not raw_dir.is_dir():
        print(f"ERROR: Input directory not found: {raw_dir}")
        sys.exit(1)

    print(f"Merging CSV files from {raw_dir} ...")
    merged = merge_raw_files(raw_dir, output_path)

    n_rows = len(merged)
    n_subjects = merged["Subject_Name"].nunique()
    print(f"รวมข้อมูลทั้งหมด {n_rows} แถว จาก {n_subjects} Subjects")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
