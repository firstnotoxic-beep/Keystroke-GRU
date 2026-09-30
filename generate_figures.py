"""
Generate academic flowchart PNGs (300 DPI) for ITJSAF/YSC reports.

Requires:
  1. pip install graphviz  (Python package)
  2. System Graphviz with `dot` on PATH
     (https://graphviz.org/download/ — restart the terminal after install)

Outputs:
  figures/figure1_pipeline.png
  figures/figure2_realtime.png

All Graphviz labels are English-only (Arial) to avoid missing-glyph boxes.
"""

from __future__ import annotations

from pathlib import Path

import graphviz

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "figures"

# Shared academic blue-gray palette
FILL_OVAL = "#E8EEF4"
FILL_BOX = "#F4F6F8"
FILL_DIAMOND = "#EEF2F6"
STROKE = "#4A6FA5"
STROKE_MUTED = "#5A6A7A"
FONT = "Arial"

# Must match login.py: config.REALTIME_THRESHOLD (Figure 2 shows it as τ).
LOGIN_THRESHOLD = 0.525
MAX_CONSECUTIVE_FAILS = 3


def _base_graph(name: str, caption: str) -> graphviz.Digraph:
    g = graphviz.Digraph(
        name=name,
        format="png",
        graph_attr={
            "rankdir": "TB",
            "dpi": "300",
            "bgcolor": "white",
            "fontname": FONT,
            "fontsize": "11",
            "labelloc": "b",
            "label": caption,
            "nodesep": "0.35",
            "ranksep": "0.45",
            "pad": "0.4",
            "splines": "ortho",
        },
        node_attr={
            "fontname": FONT,
            "fontsize": "10",
            "color": STROKE,
            "fontcolor": "#1a1a1a",
            "penwidth": "1.2",
        },
        edge_attr={
            "fontname": FONT,
            "fontsize": "9",
            "color": STROKE_MUTED,
            "arrowsize": "0.7",
            "penwidth": "1.1",
        },
    )
    return g


def _oval(g: graphviz.Digraph, node_id: str, label: str) -> None:
    g.node(
        node_id,
        label=label,
        shape="ellipse",
        style="filled",
        fillcolor=FILL_OVAL,
        color=STROKE,
    )


def _box(g: graphviz.Digraph, node_id: str, label: str) -> None:
    g.node(
        node_id,
        label=label,
        shape="box",
        style="filled,rounded",
        fillcolor=FILL_BOX,
        color=STROKE_MUTED,
    )


def _diamond(g: graphviz.Digraph, node_id: str, label: str) -> None:
    g.node(
        node_id,
        label=label,
        shape="diamond",
        style="filled",
        fillcolor=FILL_DIAMOND,
        color=STROKE,
        width="2.2",
        height="1.4",
    )


def build_figure1() -> graphviz.Digraph:
    """Experimental pipeline (left-to-right, sized for a full text-width figure)."""
    g = _base_graph("figure1_pipeline", "")
    g.graph_attr.update({"rankdir": "TB", "nodesep": "0.45", "ranksep": "0.35",
                         "splines": "spline", "pad": "0.15"})
    g.node_attr.update({"fontsize": "11", "margin": "0.08,0.05"})
    g.edge_attr.update({"fontsize": "9"})

    _box(g, "collect", "Data collection\n9 participants\n(1 owner, 8 impostors)\nQC at capture")
    _box(g, "data", "1,800 real samples\n31 features\n(H, DD, UD)")
    _box(g, "split", "Stratified split\n70 / 15 / 15")
    _box(g, "aug", "Augment train only\nGaussian · Time-warp · Mixup\n1,260 → 2,000\n+ sanity check (KS)")
    _box(g, "val", "Validation\n270 (real)")
    _box(g, "test", "Test\n270 (real)")
    _diamond(g, "norm", "Ratio\nnormalization\nON / OFF")
    g.node("norm", width="1.5", height="1.0")
    _box(g, "models", "Models\nGRU (64) · LSTM (64)\nSVM (RBF) · Manhattan")
    _box(g, "eval", "Evaluation\nEER · FAR · FRR\nROC / DET\nGRU: 5 seeds")

    for row in (("collect", "data", "split"), ("aug", "val", "test"), ("norm", "models", "eval")):
        with g.subgraph() as s:
            s.attr(rank="same")
            for n in row:
                s.node(n)
    g.edge("collect", "data")
    g.edge("data", "split")
    g.edge("split", "aug", label="train")
    g.edge("split", "val")
    g.edge("split", "test")
    g.edge("aug", "norm")
    g.edge("norm", "models")
    g.edge("val", "models", style="dashed", label="early stop /\ngrid search")
    g.edge("models", "eval")
    g.edge("test", "eval", style="dashed")
    return g


def build_figure2() -> graphviz.Digraph:
    """Real-time continuous authentication (login.py): producer-consumer + lock policy."""
    g = _base_graph("figure2_realtime", "")
    g.graph_attr.update({"rankdir": "TB", "splines": "spline", "nodesep": "0.4",
                         "ranksep": "0.35", "pad": "0.15"})
    g.node_attr.update({"fontsize": "11"})

    _oval(g, "start", "User types\npassword")
    _box(g, "t1", "Thread 1 (Producer)\npynput key events")
    _box(g, "queue", "FIFO\nQueue")
    _box(g, "t2", "Thread 2 (Consumer)\n31 features → (11, 3)\nGRU → P(owner)")
    _diamond(g, "dec", "P(owner) < τ ?")
    _box(g, "reset", "Accept\nfail count = 0")
    _box(g, "count", "fail count + 1")
    _diamond(g, "streak", f"fail count\n≥ {MAX_CONSECUTIVE_FAILS} ?")
    _box(g, "lock", "LockWorkStation()")

    for row in (("start", "t1", "queue", "t2"), ("lock", "streak", "count", "dec")):
        with g.subgraph() as s:
            s.attr(rank="same")
            for n in row:
                s.node(n)
    g.edge("start", "t1")
    g.edge("t1", "queue")
    g.edge("queue", "t2")
    g.edge("t2", "dec")
    g.edge("dec", "reset", label="No")
    g.edge("count", "dec", label="Yes", dir="back")
    g.edge("streak", "count", dir="back")
    g.edge("lock", "streak", label="Yes", dir="back")
    g.edge("streak", "t1", label="No", style="dashed", constraint="false")
    g.edge("reset", "t1", style="dashed", constraint="false")
    return g


def _stamp_dpi(path: Path, dpi: int = 300) -> None:
    """Embed DPI metadata so Word/LaTeX scale the figure correctly."""
    from PIL import Image

    with Image.open(path) as im:
        im.save(path, dpi=(dpi, dpi))


def render(graph: graphviz.Digraph, outfile: Path) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # render() expects path without extension when using format=; pass stem via filename.
    out = Path(
        graph.render(
            filename=outfile.stem,
            directory=str(FIGURES_DIR),
            format="png",
            cleanup=True,
        )
    )
    _stamp_dpi(out, dpi=300)
    return out


def main() -> None:
    f1 = render(build_figure1(), FIGURES_DIR / "figure1_pipeline.png")
    f2 = render(build_figure2(), FIGURES_DIR / "figure2_realtime.png")
    print(f"Wrote {f1}")
    print(f"Wrote {f2}")
    print(f"DPI=300 | LOGIN_THRESHOLD={LOGIN_THRESHOLD} (matches config/login.py)")


if __name__ == "__main__":
    main()
