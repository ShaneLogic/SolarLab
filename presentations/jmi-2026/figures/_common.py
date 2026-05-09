"""Matplotlib rcParams aligned with deck theme (spec § 7.6)."""
from pathlib import Path
import matplotlib as mpl

INK = "#1c2533"
BODY = "#475569"
HAIRLINE = "#cbd5e1"
ACCENT = "#c0392b"
SECONDARY = "#1a73d6"  # used only when a 2nd data series needs to disambiguate
FIG_OUT = Path(__file__).parent / "output"
FIG_DATA = Path(__file__).parent / "data"

def configure():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.8,
        "xtick.color": BODY,
        "ytick.color": BODY,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": 220,
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    FIG_OUT.mkdir(exist_ok=True)
