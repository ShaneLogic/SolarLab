"""s4 talk roadmap — 6-step funnel."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from figures._common import configure, FIG_OUT, ACCENT, INK, HAIRLINE

LABELS = ["1 · Problem", "2 · Strategy", "3 · ML + features",
          "4 · Candidates + DFT", "5 · Bridge: device sim", "6 · Payoff + collab"]

def main():
    configure()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    n = len(LABELS)
    for i, label in enumerate(LABELS):
        y = 9 - i * 1.4
        w = 8 - i * 0.8
        x0 = (10 - w) / 2
        box = FancyBboxPatch((x0, y - 0.5), w, 1.0,
                             boxstyle="round,pad=0.02,rounding_size=0.18",
                             linewidth=1.0,
                             edgecolor=INK if i < n - 1 else ACCENT,
                             facecolor="white" if i < n - 1 else "#fde9e6")
        ax.add_patch(box)
        ax.text(5, y, label, ha="center", va="center",
                fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "s4_funnel.png")
    print(f"wrote {FIG_OUT / 's4_funnel.png'}")

if __name__ == "__main__":
    main()
