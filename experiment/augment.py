"""Per-subject data augmentation applied ONLY to train_real (val/test are never augmented)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEAT_COLS, NUM_KEYS
from experiment.common import DD_IDX, H_IDX, LABEL_COL, SEED, SUBJECT_COL, UD_IDX

NOISE_KS = (0.05, 0.10, 0.15)
MIN_POSITIVE = 1e-4
WARP_KNOTS = 8
WARP_SIGMA = 0.15
MIXUP_LOW, MIXUP_HIGH = 0.3, 0.7


def gaussian_noise(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Noisy copies with std = k * per-feature std of this subject's real train samples.

    k cycles through NOISE_KS. H and DD are clipped to stay positive; UD stays signed.
    """
    std = X.std(axis=0, ddof=1)
    src = X[rng.integers(0, len(X), n)]
    ks = np.array([NOISE_KS[i % len(NOISE_KS)] for i in range(n)])[:, None]
    out = src + rng.normal(size=src.shape) * ks * std
    out[:, H_IDX] = np.clip(out[:, H_IDX], MIN_POSITIVE, None)
    out[:, DD_IDX] = np.clip(out[:, DD_IDX], MIN_POSITIVE, None)
    return out


def _events_from_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct key-down / key-up times (down_1 = 0) from H and DD of one sequence."""
    h, dd = x[H_IDX], x[DD_IDX]
    down = np.concatenate([[0.0], np.cumsum(dd)])
    up = down + h
    return down, up


def _features_from_events(down: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Re-derive H, DD, UD (UD_i = down_{i+1} - up_i, may be negative) from event times."""
    h = up - down
    dd = down[1:] - down[:-1]
    ud = down[1:] - up[:-1]
    return np.concatenate([h, dd, ud])


def _random_warp(total: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Monotonic time map on [0, total]: cumsum of exp(smoothed random walk), rescaled to total."""
    walk = np.cumsum(rng.normal(0.0, WARP_SIGMA, WARP_KNOTS))
    smooth = np.convolve(walk, np.ones(3) / 3.0, mode="same")
    increments = np.exp(smooth)
    warped = np.concatenate([[0.0], np.cumsum(increments)])
    warped = warped / warped[-1] * total
    grid = np.linspace(0.0, total, WARP_KNOTS + 1)
    return grid, warped


def time_warp(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Warp the time axis of reconstructed key events and re-derive H/DD/UD."""
    out = np.empty((n, X.shape[1]))
    for i in range(n):
        x = X[rng.integers(0, len(X))]
        down, up = _events_from_features(x)
        total = float(max(up.max(), down.max()))
        grid, warped = _random_warp(total, rng)
        out[i] = _features_from_events(np.interp(down, grid, warped), np.interp(up, grid, warped))
    out[:, H_IDX] = np.clip(out[:, H_IDX], MIN_POSITIVE, None)
    out[:, DD_IDX] = np.clip(out[:, DD_IDX], MIN_POSITIVE, None)
    return out


def mixup(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Same-subject mixup: x = lam * a + (1 - lam) * b, lam ~ U(0.3, 0.7).

    Substitute for sliding-window overlap: each sample is a fixed-length single password
    entry, not a continuous stream, so there is no window to slide over.
    """
    a = X[rng.integers(0, len(X), n)]
    b = X[rng.integers(0, len(X), n)]
    lam = rng.uniform(MIXUP_LOW, MIXUP_HIGH, size=(n, 1))
    return lam * a + (1.0 - lam) * b


def allocate_targets(train_real: pd.DataFrame, target_total: int) -> dict[str, int]:
    """Class-balanced per-subject targets: owner class and impostor class each get target/2,
    impostor half split evenly across impostor subjects. Never below the real count."""
    subj_label = train_real.groupby(SUBJECT_COL)[LABEL_COL].first()
    owners = subj_label[subj_label == 1].index.tolist()
    impostors = subj_label[subj_label == 0].index.tolist()
    half = target_total // 2
    real_counts = train_real.groupby(SUBJECT_COL).size()
    targets = {}
    for group in (owners, impostors):
        per = half // max(len(group), 1)
        for s in group:
            targets[s] = max(int(real_counts[s]), per)
    return targets


def augment_train(
    train_real: pd.DataFrame, target_total: int = 2000, seed: int = SEED
) -> pd.DataFrame:
    """Build train_augmented = real + synthetic (gaussian / timewarp / mixup), per subject."""
    rng = np.random.default_rng(seed)
    targets = allocate_targets(train_real, target_total)
    parts = []
    print(f"[augment] target ~{target_total} total training sequences (class-balanced)")
    print(f"  {'subject':<12}{'label':>6}{'real':>7}{'gauss':>7}{'warp':>7}{'mixup':>7}{'total':>7}")
    for subj, target in targets.items():
        sub = train_real[train_real[SUBJECT_COL] == subj]
        label = int(sub[LABEL_COL].iloc[0])
        X = sub[FEAT_COLS].to_numpy(dtype=np.float64)
        n_syn = max(target - len(X), 0)
        n_g, n_w = n_syn // 3, n_syn // 3
        n_m = n_syn - n_g - n_w
        blocks = [
            (X, "real", False),
            (gaussian_noise(X, n_g, rng), "gaussian", True),
            (time_warp(X, n_w, rng), "timewarp", True),
            (mixup(X, n_m, rng), "mixup", True),
        ]
        for arr, method, synth in blocks:
            if len(arr) == 0:
                continue
            block = pd.DataFrame(arr, columns=FEAT_COLS)
            block.insert(0, LABEL_COL, label)
            block.insert(0, SUBJECT_COL, subj)
            block["is_synthetic"] = synth
            block["aug_method"] = method
            parts.append(block)
        print(f"  {subj:<12}{label:>6}{len(X):>7}{n_g:>7}{n_w:>7}{n_m:>7}{len(X) + n_syn:>7}")
    aug = pd.concat(parts, ignore_index=True)
    aug = aug.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_neg_ud = int((aug[FEAT_COLS[UD_IDX]] < 0).to_numpy().sum())
    print(f"[augment] train_augmented: {len(aug)} rows "
          f"({int(aug['is_synthetic'].sum())} synthetic); negative UD values kept: {n_neg_ud}")
    assert aug[FEAT_COLS].shape[1] == 3 * NUM_KEYS - 2
    return aug
