"""Classical baselines: RBF-SVM and Manhattan (L1) distance to the owner template."""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiment.common import SEED
from experiment.metrics import compute_eer

SVM_C_GRID = (0.1, 1.0, 10.0, 100.0)
SVM_GAMMA_GRID = ("scale", 0.01, 0.1, 1.0)


def train_svm(X_train, y_train, X_val, y_val) -> tuple[SVC, StandardScaler, dict]:
    """Grid-search C x gamma, select by lowest val EER (tie-break: higher AUC)."""
    scaler = StandardScaler().fit(X_train)
    Xtr, Xva = scaler.transform(X_train), scaler.transform(X_val)
    best, best_key, grid_log = None, None, []
    for C, gamma in itertools.product(SVM_C_GRID, SVM_GAMMA_GRID):
        clf = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, y_train)
        m = compute_eer(y_val, clf.decision_function(Xva))
        grid_log.append({"C": C, "gamma": gamma, "val_eer": m["eer"], "val_auc": m["auc"]})
        print(f"  [svm] C={C:<6} gamma={str(gamma):<6} val EER={m['eer'] * 100:6.2f}%  "
              f"AUC={m['auc']:.4f}")
        key = (m["eer"], -m["auc"])
        if best_key is None or key < best_key:
            best, best_key = clf, key
    print(f"[svm] selected C={best.C}, gamma={best.gamma} "
          f"(val EER={best_key[0] * 100:.2f}%)")
    return best, scaler, {"grid": grid_log, "C": best.C, "gamma": best.gamma}


def svm_scores(clf: SVC, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Signed distance to the SVM hyperplane (higher = owner)."""
    return clf.decision_function(scaler.transform(X))


def manhattan_template(X_owner_train: np.ndarray) -> np.ndarray:
    """Owner reference vector = mean of the owner's real training samples (31-dim)."""
    return X_owner_train.mean(axis=0)


def manhattan_distance(template: np.ndarray, X: np.ndarray) -> np.ndarray:
    """L1 distance of each sample to the template (lower = more owner-like)."""
    return np.abs(X - template).sum(axis=1)
