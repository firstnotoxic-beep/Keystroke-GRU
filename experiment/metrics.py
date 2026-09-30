"""EER / FAR / FRR computation and ROC / DET comparison plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.metrics import auc, classification_report, roc_curve  # noqa: E402


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Full threshold sweep (higher score = owner). FAR = impostors accepted, FRR = owners rejected."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return {
        "eer": float((fpr[idx] + fnr[idx]) / 2),
        "far": float(fpr[idx]),
        "frr": float(fnr[idx]),
        "threshold": float(thr[idx]),
        "auc": float(auc(fpr, tpr)),
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
    }


def report(name: str, y_true: np.ndarray, scores: np.ndarray, threshold_note: str = "") -> dict:
    """Compute EER metrics, print them plus a classification report at the EER threshold."""
    m = compute_eer(y_true, scores)
    print(f"[{name}] EER={m['eer'] * 100:.2f}%  FAR={m['far'] * 100:.2f}%  "
          f"FRR={m['frr'] * 100:.2f}%  AUC={m['auc']:.4f}  "
          f"threshold={m['threshold']:.6g}{threshold_note}")
    y_pred = (scores >= m["threshold"]).astype(int)
    print(classification_report(y_true, y_pred, target_names=["Impostor", "Owner"], digits=4))
    return m


def plot_roc(results: dict[str, dict], path: Path) -> None:
    """Overlay ROC curves (FAR vs 1-FRR) of all models."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, m in results.items():
        ax.plot(m["fpr"], m["tpr"], lw=2,
                label=f"{name} (AUC={m['auc']:.3f}, EER={m['eer'] * 100:.1f}%)")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Acceptance Rate (1 - FRR)")
    ax.set_title("ROC comparison (real-only test set)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_det(results: dict[str, dict], path: Path) -> None:
    """Overlay DET curves (FAR vs FRR) on normal-deviate (probit) axes."""
    lo, hi = 1e-3, 1 - 1e-3
    ticks = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95])
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, m in results.items():
        x = norm.ppf(np.clip(m["fpr"], lo, hi))
        y = norm.ppf(np.clip(m["fnr"], lo, hi))
        ax.plot(x, y, lw=2, label=f"{name} (EER={m['eer'] * 100:.1f}%)")
    ax.plot(norm.ppf([lo, hi]), norm.ppf([lo, hi]), "k--", lw=1, alpha=0.5, label="EER line")
    ax.set_xticks(norm.ppf(ticks))
    ax.set_xticklabels([f"{t * 100:g}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks))
    ax.set_yticklabels([f"{t * 100:g}" for t in ticks])
    ax.set_xlim(norm.ppf(lo), norm.ppf(0.95))
    ax.set_ylim(norm.ppf(lo), norm.ppf(0.95))
    ax.set_xlabel("FAR (%)  [probit scale]")
    ax.set_ylabel("FRR (%)  [probit scale]")
    ax.set_title("DET comparison (real-only test set)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: dict, name: str, out_dir: Path) -> list[Path]:
    """Save loss and accuracy curves (train vs val) for a Keras history dict."""
    paths = []
    for key, label in (("loss", "Loss (BCE)"), ("accuracy", "Accuracy")):
        if key not in history:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history[key], label=f"train {key}")
        if f"val_{key}" in history:
            ax.plot(history[f"val_{key}"], label=f"val {key}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{name} - {label}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"{name}_{key}_curve.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths
