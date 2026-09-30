"""
Generate two presentation figures from the test-set FAR / FRR / EER scan.

Outputs:
  figures/figure3_far_frr_threshold.png
  figures/figure4_operating_points.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from config import MODEL_PATH, PROCESSED_DATA_PATH

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "figures"

STROKE = "#4A6FA5"
FAR_COLOR = "#C0392B"
FRR_COLOR = "#2C5F8A"
EER_COLOR = "#1A1A1A"
MUTED = "#5A6A7A"
GRID = "#E6E8EB"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": "#1a1a1a",
        "xtick.color": "#1a1a1a",
        "ytick.color": "#1a1a1a",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def scan_far_frr() -> dict:
    X_test = np.load(PROCESSED_DATA_PATH / "X_test.npy")
    y_test = np.load(PROCESSED_DATA_PATH / "y_test.npy")
    model = load_model(MODEL_PATH)
    y_pred_prob = model.predict(X_test, verbose=0).ravel()

    n_impostor = int((y_test == 0).sum())
    n_owner = int((y_test == 1).sum())
    thresholds = np.arange(0.0, 1.01, 0.01)
    far_list, frr_list, acc_list = [], [], []

    for t in thresholds:
        y_pred = (y_pred_prob >= t).astype(int)
        far = ((y_pred == 1) & (y_test == 0)).sum() / n_impostor
        frr = ((y_pred == 0) & (y_test == 1)).sum() / n_owner
        acc = (y_pred == y_test).mean()
        far_list.append(far)
        frr_list.append(frr)
        acc_list.append(acc)

    far_arr = np.array(far_list, dtype=float)
    frr_arr = np.array(frr_list, dtype=float)
    acc_arr = np.array(acc_list, dtype=float)
    eer_idx = int(np.argmin(np.abs(far_arr - frr_arr)))

    return {
        "thresholds": thresholds,
        "far": far_arr,
        "frr": frr_arr,
        "acc": acc_arr,
        "eer_idx": eer_idx,
        "eer_threshold": float(thresholds[eer_idx]),
        "eer": float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2),
        "far_at_eer": float(far_arr[eer_idx]),
        "frr_at_eer": float(frr_arr[eer_idx]),
        "acc_at_eer": float(acc_arr[eer_idx]),
        "idx_05": int(np.argmin(np.abs(thresholds - 0.50))),
    }


def _stamp_dpi(path: Path, dpi: int = 300) -> None:
    from PIL import Image

    with Image.open(path) as im:
        im.save(path, dpi=(dpi, dpi))


def figure_far_frr_curve(metrics: dict, path: Path) -> None:
    t = metrics["thresholds"]
    far = metrics["far"]
    frr = metrics["frr"]
    i = metrics["eer_idx"]
    i05 = metrics["idx_05"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ax.plot(t, far, color=FAR_COLOR, lw=2.0, label="FAR (impostor accepted)")
    ax.plot(t, frr, color=FRR_COLOR, lw=2.0, label="FRR (owner rejected)")
    ax.axvline(t[i05], color=MUTED, ls=":", lw=1.0, alpha=0.7)
    ax.axvline(t[i], color=EER_COLOR, ls="--", lw=1.1, alpha=0.85)
    ax.scatter(
        [t[i]],
        [metrics["eer"]],
        s=42,
        color=EER_COLOR,
        zorder=5,
        label=f"EER = {metrics['eer']:.2%} at threshold 0.62",
    )
    ax.scatter([t[i05]], [far[i05]], s=28, color=FAR_COLOR, zorder=4)
    ax.scatter([t[i05]], [frr[i05]], s=28, color=FRR_COLOR, zorder=4)
    ax.text(
        t[i05] - 0.02,
        0.52,
        "default 0.50",
        ha="right",
        va="bottom",
        fontsize=9,
        color=MUTED,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Decision threshold (P(owner))")
    ax.set_ylabel("Error rate")
    ax.set_title("FAR / FRR vs threshold (test set, n = 210)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    _stamp_dpi(path)


def figure_operating_points(metrics: dict, path: Path) -> None:
    i05 = metrics["idx_05"]
    labels = ["FAR", "FRR"]
    at_05 = [metrics["far"][i05], metrics["frr"][i05]]
    at_eer = [metrics["far_at_eer"], metrics["frr_at_eer"]]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)

    bars_a = ax.bar(
        x - width / 2,
        at_05,
        width,
        color="#D6DEE8",
        edgecolor=STROKE,
        label="threshold 0.50 (default)",
    )
    bars_b = ax.bar(
        x + width / 2,
        at_eer,
        width,
        color=STROKE,
        edgecolor=STROKE,
        label=f"threshold {metrics['eer_threshold']:.2f} (used in login)",
    )
    ax.axhline(
        metrics["eer"],
        color=EER_COLOR,
        ls="--",
        lw=1.2,
        label=f"EER = {metrics['eer']:.1%}",
    )

    def annotate(bars, values, color):
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color,
            )

    annotate(bars_a, at_05, MUTED)
    annotate(bars_b, at_eer, "#1a1a1a")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.42)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Error rate")
    acc05 = metrics["acc"][i05]
    acc_eer = metrics["acc_at_eer"]
    ax.set_title(
        "FAR / FRR at two operating points (test set)\n"
        f"Accuracy {acc05:.1%} at 0.50  |  {acc_eer:.1%} at EER"
    )
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    _stamp_dpi(path)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = scan_far_frr()
    p3 = FIGURES_DIR / "figure3_far_frr_threshold.png"
    p4 = FIGURES_DIR / "figure4_operating_points.png"
    figure_far_frr_curve(metrics, p3)
    figure_operating_points(metrics, p4)
    print(
        f"EER={metrics['eer']:.4f}  threshold={metrics['eer_threshold']:.2f}  "
        f"FAR={metrics['far_at_eer']:.4f}  FRR={metrics['frr_at_eer']:.4f}"
    )
    print(f"Wrote {p3}")
    print(f"Wrote {p4}")


if __name__ == "__main__":
    main()
