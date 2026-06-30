"""Split normalized keystroke CSV into train/val/test .npy tensors."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    FEAT_COLS,
    MIN_SAMPLES_PER_CLASS,
    NORMALIZED_DATA_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    validate_csv_schema,
)


def load_normalized(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized CSV and return feature matrix and labels."""
    if not path.is_file():
        print(f"ERROR: Input file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    if df.empty:
        print(f"ERROR: Input file is empty (no data rows): {path}")
        sys.exit(1)
    df = validate_csv_schema(df)
    return df[FEAT_COLS].values.astype(np.float64), df["Label"].values.astype(np.int64)


def reshape_to_3d(features: np.ndarray) -> np.ndarray:
    """Reshape (N, 31) features into (N, 11, 3) timestep tensors."""
    h, dd, ud = features[:, :11], features[:, 11:21], features[:, 21:31]
    t0 = np.zeros((features.shape[0], 1, 3))
    t0[:, 0, 0] = h[:, 0]
    rest = np.stack([h[:, 1:], dd, ud], axis=2)
    return np.concatenate([t0, rest], axis=1)


def check_min_samples_per_class(y: np.ndarray) -> None:
    """Exit gracefully if any class has too few samples for a reliable split."""
    for label in np.unique(y):
        count = int((y == label).sum())
        if count < MIN_SAMPLES_PER_CLASS:
            label_name = "Owner" if label == 1 else "Impostor"
            print(
                f"ERROR: Class {label_name} (Label={label}) has only {count} sample(s). "
                f"Minimum required: {MIN_SAMPLES_PER_CLASS} per class."
            )
            print(
                "Collect more data with collect_data.py, then run preprocess.py "
                "before splitting again."
            )
            sys.exit(1)


def stratified_split(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into stratified train/val/test sets via sklearn."""
    check_min_samples_per_class(y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=VAL_RATIO + TEST_RATIO,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_splits(
    splits: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    out_dir: Path,
) -> None:
    """Save six split arrays as .npy files (including X_test.npy and y_test.npy)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    for name, arr in zip(names, splits):
        np.save(out_dir / f"{name}.npy", arr)


def main() -> None:
    """Run load, reshape, split, and save pipeline."""
    parser = argparse.ArgumentParser(description="Reshape and split keystroke data")
    parser.add_argument("--input", default=str(NORMALIZED_DATA_PATH))
    parser.add_argument("--output-dir", default=str(PROCESSED_DATA_PATH))
    args = parser.parse_args()
    feats, labels = load_normalized(Path(args.input))
    X = reshape_to_3d(feats)
    splits = stratified_split(X, labels)
    save_splits(splits, Path(args.output_dir))
    X_tr, y_tr, _, y_va, _, y_te = splits
    n = len(labels)
    print(f"Total samples: {n}")
    print(f"Train: {len(y_tr)} | Val: {len(y_va)} | Test: {len(y_te)}")
    print(f"X_train shape: {X_tr.shape}")
    print(
        f"Train labels — Owner(1): {int(y_tr.sum())}, "
        f"Impostor(0): {int(len(y_tr) - y_tr.sum())}"
    )


if __name__ == "__main__":
    main()
