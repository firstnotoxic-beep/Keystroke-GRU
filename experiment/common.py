"""Shared helpers: seeding, paths, data loading, splitting, normalization, reshaping."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    FEAT_COLS,
    NUM_KEYS,
    PROJECT_ROOT,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TEST_RATIO,
    VAL_RATIO,
    validate_csv_schema,
)

SEED = RANDOM_STATE  # 42
SUBJECT_COL = "Subject_Name"
LABEL_COL = "Label"

RESULTS_DIR = PROJECT_ROOT / "results"
SPLITS_DIR = RESULTS_DIR / "splits"
SANITY_DIR = RESULTS_DIR / "sanity_check"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models" / "experiment"

H_IDX = slice(0, 11)
DD_IDX = slice(11, 21)
UD_IDX = slice(21, 31)


def banner(title: str) -> None:
    """Print a visually distinct step header."""
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def set_global_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and TensorFlow (if importable) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:  # noqa: BLE001
            print(f"  (op determinism not enabled: {exc})")
    except ImportError:
        pass
    print(f"[seed] random / numpy / tensorflow seeded with {seed}")


def ensure_dirs() -> None:
    """Create all output directories."""
    for d in (RESULTS_DIR, SPLITS_DIR, SANITY_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_real_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the merged real dataset and validate schema via config.validate_csv_schema."""
    df = pd.read_csv(path)
    df = validate_csv_schema(df)
    df[FEAT_COLS] = df[FEAT_COLS].astype(np.float64)
    print(f"[data] Loaded {len(df)} real samples from {path}")
    print(f"[data] Subject column='{SUBJECT_COL}', label column='{LABEL_COL}'")
    counts = df.groupby([SUBJECT_COL, LABEL_COL]).size()
    for (subj, lab), n in counts.items():
        print(f"         {subj:<12} label={lab}  n={n}")
    owners = df.loc[df[LABEL_COL] == 1, SUBJECT_COL].unique()
    impostors = df.loc[df[LABEL_COL] == 0, SUBJECT_COL].unique()
    print(f"[data] {len(owners)} owner subject(s), {len(impostors)} impostor subject(s)")
    return df


def split_real(
    df: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split on subject+label (default seed 42) BEFORE any augmentation."""
    strata = df[SUBJECT_COL].astype(str) + "_" + df[LABEL_COL].astype(str)
    train, temp = train_test_split(
        df, test_size=VAL_RATIO + TEST_RATIO, stratify=strata, random_state=seed
    )
    temp_strata = temp[SUBJECT_COL].astype(str) + "_" + temp[LABEL_COL].astype(str)
    val, test = train_test_split(
        temp,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_strata,
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def print_split_counts(splits: dict[str, pd.DataFrame]) -> None:
    """Print a per-subject count table across splits."""
    table = pd.DataFrame(
        {name: d.groupby(SUBJECT_COL).size() for name, d in splits.items()}
    ).fillna(0).astype(int)
    table.loc["TOTAL"] = table.sum()
    print(table.to_string())


def intra_sequence_ratio_normalization(sequence: np.ndarray) -> np.ndarray:
    """Divide each feature by the sum of |values| of its type (H / DD / UD) in the same sequence.

    Accepts a single (31,) vector or an (N, 31) matrix. Using abs() in the divisor removes
    the dependency on the sign of UD (overlapping keys give negative UD) while preserving
    relative timing. Zero divisors are replaced by 1 (the group stays all-zero).
    For positive H/DD this equals preprocess.normalize_feature_vector.
    """
    arr = np.asarray(sequence, dtype=np.float64)
    single = arr.ndim == 1
    arr = np.atleast_2d(arr)
    out = np.empty_like(arr)
    for idx in (H_IDX, DD_IDX, UD_IDX):
        block = arr[:, idx]
        div = np.abs(block).sum(axis=1, keepdims=True)
        div[div == 0] = 1.0
        out[:, idx] = block / div
    return out[0] if single else out


def prepare_features(df: pd.DataFrame, normalize: bool) -> np.ndarray:
    """Return the (N, 31) feature matrix, optionally ratio-normalized."""
    X = df[FEAT_COLS].to_numpy(dtype=np.float64)
    return intra_sequence_ratio_normalization(X) if normalize else X


def reshape_sequences(X: np.ndarray) -> np.ndarray:
    """(N, 31) -> (N, 11, 3); timestep i = [H_i, DD_i, UD_i], DD_11/UD_11 padded with 0."""
    n = X.shape[0]
    out = np.zeros((n, NUM_KEYS, 3), dtype=np.float64)
    out[:, :, 0] = X[:, H_IDX]
    out[:, :10, 1] = X[:, DD_IDX]
    out[:, :10, 2] = X[:, UD_IDX]
    return out


def compare_with_existing_reshape(X: np.ndarray) -> bool:
    """Compare reshape_sequences with split_and_save.reshape_to_3d; print WARNING if different."""
    from split_and_save import reshape_to_3d

    ours = reshape_sequences(X[:5])
    theirs = reshape_to_3d(X[:5])
    if ours.shape == theirs.shape and np.allclose(ours, theirs):
        print("[reshape] Matches split_and_save.reshape_to_3d.")
        return False
    print(
        "WARNING [reshape]: padding differs from split_and_save.reshape_to_3d.\n"
        "  existing : timestep 0 = [H1, 0, 0], timestep i (1..10) = [H_{i+1}, DD_i, UD_i]\n"
        "             (padding at the START; DD_i/UD_i aligned with the key they lead INTO)\n"
        "  this run : timestep i (0..10) = [H_{i+1}, DD_{i+1}, UD_{i+1}], DD_11 = UD_11 = 0\n"
        "             (padding at the END; DD_i/UD_i aligned with the key they lead OUT OF)\n"
        "  Same information, different alignment -> models from the two pipelines are\n"
        "  NOT interchangeable. Use --reshape existing to reproduce the old layout."
    )
    return True


def check_normalization_matches_preprocess(X_raw: np.ndarray) -> None:
    """Cross-check our normalization against preprocess.normalize_feature_vector on real rows."""
    from preprocess import normalize_feature_vector

    ours = intra_sequence_ratio_normalization(X_raw)
    mismatches = 0
    for i in range(len(X_raw)):
        try:
            ref = normalize_feature_vector(X_raw[i])
        except ZeroDivisionError:
            continue
        if not np.allclose(ours[i], ref):
            mismatches += 1
    if mismatches:
        print(
            f"WARNING [norm]: {mismatches} row(s) differ from preprocess.normalize_feature_vector "
            "(preprocess divides H/DD by plain sum, this run by abs-sum)."
        )
    else:
        print("[norm] Identical to preprocess.normalize_feature_vector on all real rows.")
