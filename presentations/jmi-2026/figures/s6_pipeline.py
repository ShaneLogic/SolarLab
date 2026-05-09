"""s6 pipeline schematic."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figures._common import configure, FIG_OUT, ACCENT, INK

STEPS = ["Materials DB", "Feature\nextraction", "ML\nsurrogate",
         "DFT\nvalidation", "Device sim", "Experiment"]

def main():
    configure()
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    for i, label in enumerate(STEPS):
        x = 0.5 + i * 1.95
        is_last = i == len(STEPS) - 1
        box = FancyBboxPatch((x, 1.0), 1.5, 1.7,
                             boxstyle="round,pad=0.03,rounding_size=0.16",
                             linewidth=1.0,
                             edgecolor=ACCENT if is_last else INK,
                             facecolor="#fde9e6" if is_last else "white")
        ax.add_patch(box)
        ax.text(x + 0.75, 1.85, label, ha="center", va="center",
                fontsize=11, color=INK, fontweight="bold")
        if i < len(STEPS) - 1:
            arrow = FancyArrowPatch((x + 1.55, 1.85), (x + 1.92, 1.85),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=INK, linewidth=1.2)
            ax.add_patch(arrow)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "s6_pipeline.png")
    print(f"wrote {FIG_OUT / 's6_pipeline.png'}")

if __name__ == "__main__":
    main()
