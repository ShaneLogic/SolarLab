"""s17 capability table: SolarLab vs SCAPS-1D."""
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, ACCENT, INK, BODY

ROWS = [
    ("Spatial dimensionality", "1D + 2D",            "1D only"),
    ("Mobile-ion transport",   "drift-diffusion",    "steady-state hack"),
    ("Optics — TMM coherent",  "multi-layer",        "Beer–Lambert only"),
    ("Hysteresis / scan rate", "time-resolved",      "—"),
    ("Selective Robin contacts", r"$S_n$, $S_p$",     "limited"),
    ("Tandem / multi-junction","series matched",     "manual"),
    ("Open-source / scriptable","Python + REST",     "GUI only"),
    ("Solver",                 "Radau + BDF",        "Gummel iter."),
]

def main():
    configure()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.axis("off")
    table = ax.table(
        cellText=[[r[1], r[2]] for r in ROWS],
        rowLabels=[r[0] for r in ROWS],
        colLabels=["SolarLab (ours)", "SCAPS-1D"],
        cellLoc="center", rowLoc="left", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(BODY)
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(color="white", weight="bold")
            cell.set_facecolor(INK)
        if c == 0 and r > 0:
            cell.set_text_props(color=ACCENT, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "s17_capability.png")
    print(f"wrote {FIG_OUT / 's17_capability.png'}")

if __name__ == "__main__":
    main()
