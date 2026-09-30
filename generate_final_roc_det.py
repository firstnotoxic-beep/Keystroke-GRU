"""Final ROC / DET figures for the paper: every model, normalization ON (dashed) vs OFF (solid).

Seed-42 split. GRU = converged multi-seed models (seed 42), LSTM = 300-epoch models from
run_baselines_norm_ablation.py, SVM / Manhattan are refit (deterministic).

Usage:
    python generate_final_roc_det.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from experiment.common import (  # noqa: E402
    LABEL_COL,
    MODELS_DIR,
    RESULTS_DIR,
    load_real_data,
    prepare_features,
    reshape_sequences,
    set_global_seed,
    split_real,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FIG_DIR = RESULTS_DIR.parent / "figures"
COLORS = {"GRU": "#1f77b4", "LSTM": "#9467bd", "SVM": "#d62728", "Manhattan": "#2ca02c"}


def collect_scores() -> tuple[np.ndarray, dict[tuple[str, bool], np.ndarray]]:
    """Return y_test and {(model, normalize): owner-scores} on the seed-42 real-only test set."""
    import tensorflow as tf

    from experiment.augment import augment_train
    from experiment.baselines import manhattan_distance, manhattan_template, svm_scores, train_svm

    set_global_seed()
    df = load_real_data()
    train_real, val_real, test_real = split_real(df)
    train_aug = augment_train(train_real)
    y_tr = train_aug[LABEL_COL].to_numpy(dtype=int)
    y_va = val_real[LABEL_COL].to_numpy(dtype=int)
    y_te = test_real[LABEL_COL].to_numpy(dtype=int)
    owner_train = train_real[train_real[LABEL_COL] == 1]

    rnn_paths = {
        ("GRU", True): MODELS_DIR / "multiseed" / "gru_with_correction_seed42.keras",
        ("GRU", False): MODELS_DIR / "multiseed" / "gru_without_correction_seed42.keras",
        ("LSTM", True): MODELS_DIR / "lstm_norm_on_ep300.keras",
        ("LSTM", False): MODELS_DIR / "lstm_norm_off_ep300.keras",
    }
    scores: dict[tuple[str, bool], np.ndarray] = {}
    for key, path in rnn_paths.items():
        model = tf.keras.models.load_model(path)
        X = reshape_sequences(prepare_features(test_real, key[1]))
        scores[key] = model.predict(X, verbose=0).ravel()
    for normalize in (True, False):
        clf, scaler, _ = train_svm(prepare_features(train_aug, normalize), y_tr,
                                   prepare_features(val_real, normalize), y_va)
        scores[("SVM", normalize)] = svm_scores(clf, scaler, prepare_features(test_real, normalize))
        template = manhattan_template(prepare_features(owner_train, normalize))
        scores[("Manhattan", normalize)] = -manhattan_distance(
            template, prepare_features(test_real, normalize))
    return y_te, scores


def plot_all(y_te: np.ndarray, scores: dict) -> list:
    """Draw ROC and DET overlays; return output paths and a metrics table."""
    from experiment.metrics import compute_eer

    metrics = {k: compute_eer(y_te, s) for k, s in scores.items()}
    order = [(m, n) for m in COLORS for n in (False, True)]

    def style(key):
        model, normalize = key
        return dict(color=COLORS[model], ls="--" if normalize else "-",
                    lw=1.6 if normalize else 2.2, alpha=0.85 if normalize else 1.0)

    def label(key, with_auc):
        m = metrics[key]
        tag = f"{key[0]} (norm {'ON' if key[1] else 'OFF'})"
        extra = f"AUC={m['auc']:.3f}, " if with_auc else ""
        return f"{tag}: {extra}EER={m['eer'] * 100:.2f}%"

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    roc_path, det_path = FIG_DIR / "figure5_roc_final.png", FIG_DIR / "figure6_det_final.png"

    fig, ax = plt.subplots(figsize=(7, 6.2))
    for k in order:
        ax.plot(metrics[k]["fpr"], metrics[k]["tpr"], label=label(k, True), **style(k))
    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Acceptance Rate (1 − FRR)")
    ax.set_title("ROC curves — real-only test set (n = 270)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    inset = ax.inset_axes([0.42, 0.38, 0.33, 0.33])
    for k in order:
        inset.plot(metrics[k]["fpr"], metrics[k]["tpr"], **style(k))
    inset.set_xlim(0, 0.2)
    inset.set_ylim(0.8, 1.0)
    inset.grid(alpha=0.3)
    inset.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)

    lo, hi = 1e-3, 1 - 1e-3
    ticks = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
    fig, ax = plt.subplots(figsize=(7, 6.2))
    for k in order:
        x = norm.ppf(np.clip(metrics[k]["fpr"], lo, hi))
        y = norm.ppf(np.clip(metrics[k]["fnr"], lo, hi))
        ax.plot(x, y, label=label(k, False), **style(k))
    ax.plot(norm.ppf([lo, hi]), norm.ppf([lo, hi]), "k:", lw=1, alpha=0.5, label="EER line")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks(norm.ppf(ticks))
        axis.set_ticklabels([f"{t * 100:g}" for t in ticks])
    ax.set_xlim(norm.ppf(0.004), norm.ppf(0.5))
    ax.set_ylim(norm.ppf(0.004), norm.ppf(0.5))
    ax.set_xlabel("FAR (%)  [normal-deviate scale]")
    ax.set_ylabel("FRR (%)  [normal-deviate scale]")
    ax.set_title("DET curves — real-only test set (n = 270)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(det_path, dpi=300)
    plt.close(fig)

    table = pd.DataFrame([{
        "model": k[0], "normalization": "ON" if k[1] else "OFF",
        "EER(%)": round(metrics[k]["eer"] * 100, 2), "AUC": round(metrics[k]["auc"], 4),
    } for k in order])
    return [roc_path, det_path], table


def main() -> None:
    """Compute scores, plot, and print the numbers shown in the legends."""
    y_te, scores = collect_scores()
    paths, table = plot_all(y_te, scores)
    print(table.to_string(index=False))
    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
