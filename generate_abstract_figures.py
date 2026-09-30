"""Two small figures sized for the abstract's side-by-side slots (7.8 x 5.0 cm each).

A: EER per model, normalization ON vs OFF (seed 42).
B: ROC (zoomed) of every model with raw features (normalization OFF).

Usage:
    python generate_abstract_figures.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiment.common import RESULTS_DIR  # noqa: E402
from generate_final_roc_det import COLORS, FIG_DIR, collect_scores  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SIZE_IN = (7.8 / 2.54, 5.0 / 2.54)
plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
                     "legend.fontsize": 5.8, "xtick.labelsize": 6, "ytick.labelsize": 6})


def eer_table() -> pd.DataFrame:
    """Seed-42 EER (%) for every model x normalization, from the saved result CSVs."""
    base = pd.read_csv(RESULTS_DIR / "baselines_norm_on_vs_off.csv")
    base["model"] = base["model_name"].replace({"Manhattan Distance": "Manhattan"})
    gru = pd.read_csv(RESULTS_DIR / "ablation_multiseed_results.csv").query("seed == 42")
    gru = pd.DataFrame({"model": "GRU",
                        "normalization": gru["condition"].map({"with_correction": "ON",
                                                               "without_correction": "OFF"}),
                        "EER(%)": gru["EER"]})
    return pd.concat([gru, base[["model", "normalization", "EER(%)"]]], ignore_index=True)


def figure_a(table: pd.DataFrame, path) -> None:
    """Grouped bars: EER with normalization ON vs OFF per model."""
    models = list(COLORS)
    on = [table.query("model == @m and normalization == 'ON'")["EER(%)"].item() for m in models]
    off = [table.query("model == @m and normalization == 'OFF'")["EER(%)"].item() for m in models]
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=SIZE_IN)
    b1 = ax.bar(x - w / 2, on, w, label="Ratio normalization ON", color="#c9d3e3",
                edgecolor="#4a6fa5", lw=0.6)
    b2 = ax.bar(x + w / 2, off, w, label="Raw features (OFF)", color="#4a6fa5")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", fontsize=5.3, padding=1)
    ax.set_xticks(x, models)
    ax.set_ylabel("EER (%)")
    ax.set_ylim(0, max(on + off) * 1.25)
    ax.set_title("EER by model (test set, n = 270)")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def figure_b(y_te: np.ndarray, scores: dict, path) -> None:
    """Zoomed ROC of each model using raw features."""
    from experiment.metrics import compute_eer

    fig, ax = plt.subplots(figsize=SIZE_IN)
    for model, color in COLORS.items():
        m = compute_eer(y_te, scores[(model, False)])
        ax.plot(m["fpr"], m["tpr"], color=color, lw=1.2,
                label=f"{model} (EER {m['eer'] * 100:.2f}%)")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0.8, 1.005)
    ax.set_xlabel("False Acceptance Rate")
    ax.set_ylabel("1 − False Rejection Rate")
    ax.set_title("ROC, raw features (zoomed)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=600)
    plt.close(fig)


def main() -> None:
    """Build both abstract figures."""
    table = eer_table()
    print(table.to_string(index=False))
    pa, pb = FIG_DIR / "abstract_fig_a_eer_bars.png", FIG_DIR / "abstract_fig_b_roc.png"
    figure_a(table, pa)
    y_te, scores = collect_scores()
    figure_b(y_te, scores, pb)
    print(f"Saved: {pa}\nSaved: {pb}")


if __name__ == "__main__":
    main()
