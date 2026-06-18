from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "story": "#1f77b4",
    "code": "#2ca02c",
    "dialogue": "#9467bd",
    "math": "#d62728",
    "news": "#ff7f0e",
    "encyclopedia": "#17becf",
    "mixed": "#111111",
}

SOURCE_ORDER = ["story", "code", "dialogue", "math", "news", "encyclopedia"]
REFINED_SOURCES = ["story", "code", "encyclopedia"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
