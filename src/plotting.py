"""Shared matplotlib style: one fixed colour per model (colour follows the model, never
its rank), thin marks, recessive grid, single y-axis per chart."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config as C

# categorical slots in fixed order (validated default palette of the dataviz guide)
COLORS = {"B0": "#2a78d6", "B1": "#eb6834", "M1": "#1baf7a", "TSO": "#eda100",
          "actual": "#2a78d6", "lagged": "#eb6834"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def setup() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 2,
        "legend.frameon": False, "font.size": 9,
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(C.FIG_DIR / f"{name}.png")
