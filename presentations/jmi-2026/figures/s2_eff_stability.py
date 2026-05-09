"""s2 efficiency vs stability landscape."""
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, ACCENT, INK, SECONDARY

# Hand-curated literature points. Replace with sourced data when available.
PEROVSKITE = [(120, 25.7), (180, 24.2), (200, 25.1), (90, 23.0), (240, 25.8)]
CIGS_SI    = [(50_000, 23.4), (40_000, 22.8), (60_000, 22.1), (35_000, 21.5),
              (80_000, 22.6), (45_000, 21.0)]

def main():
    configure()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xscale("log")
    px, py = zip(*PEROVSKITE)
    cx, cy = zip(*CIGS_SI)
    ax.scatter(px, py, s=64, color=ACCENT, alpha=0.85,
               label="Perovskite", edgecolor="white", linewidth=0.6)
    ax.scatter(cx, cy, s=64, color=INK, alpha=0.85,
               label="CIGS / Si", edgecolor="white", linewidth=0.6)
    ax.add_patch(plt.Rectangle((30_000, 24), 80_000, 4, fill=False,
                                edgecolor=SECONDARY, linewidth=2,
                                linestyle=(0, (6, 4))))
    ax.text(60_000, 28.4, "target", color=SECONDARY,
            ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Operational lifetime T₈₀ (h)")
    ax.set_ylabel("PCE (%)")
    ax.set_xlim(50, 200_000)
    ax.set_ylim(15, 30)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "s2_eff_stability.png")
    print(f"wrote {FIG_OUT / 's2_eff_stability.png'}")

if __name__ == "__main__":
    main()
