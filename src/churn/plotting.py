"""One visual style for every figure in the project.

Palette: a validated categorical order (blue, orange, aqua, yellow, magenta,
green, violet, red) used in that fixed order and never cycled; a single-hue
blue ramp for magnitude; blue<->red with a grey midpoint for diverging values.
Marks are thin, gridlines are solid hairlines, and there is never a second
y-axis: two measures on different scales go in two panels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:   # scripts write files only; inside Jupyter keep the inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
DEEMPH = "#c3c2b7"          # grey for context series in emphasis charts

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ACCENT = CATEGORICAL[0]
SECOND = CATEGORICAL[1]

SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING_MID = "#f0efec"
DIV_NEG = "#e34948"  # red pole
DIV_POS = "#2a78d6"  # blue pole

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
DIV_CMAP = LinearSegmentedColormap.from_list("div_red_blue", [DIV_NEG, DIVERGING_MID, DIV_POS])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.prop_cycle": matplotlib.cycler(color=CATEGORICAL),
        "figure.dpi": 110,
        "savefig.dpi": 150,
    })


def subtitle(ax, text: str) -> None:
    """Small secondary line under the title (what the chart shows / units)."""
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom", ha="left")


def save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def hbar(ax, labels, values, color: str = ACCENT, highlight: set | None = None, fmt: str = "{:,.0f}",
         label_values: bool = True) -> None:
    """Horizontal bars, largest at top. Optional highlight set -> others greyed (emphasis)."""
    labels, values = list(labels), list(values)
    y = range(len(labels))[::-1]
    colors = [color if (highlight is None or l in highlight) else DEEMPH for l in labels]
    ax.barh(list(y), values, color=colors, height=0.62)
    ax.set_yticks(list(y), labels)
    ax.grid(axis="y", visible=False)
    if label_values:
        span = max(abs(v) for v in values) if values else 1
        for yi, v in zip(y, values):
            ax.text(v + (0.01 * span if v >= 0 else -0.01 * span), yi, fmt.format(v), va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5, color=INK_2)


apply_style()
