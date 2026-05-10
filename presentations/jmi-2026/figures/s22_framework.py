"""s22 — SolarLab framework block diagram (frontend / backend / solver)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figures._common import configure, FIG_OUT, INK, BODY, ACCENT, HAIRLINE


def _block(ax, x, y, w, h, *, title, body, fc, ec, title_color):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.01,rounding_size=0.04",
                         linewidth=0.9, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.20, title,
            ha="center", va="top",
            color=title_color, fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.15, body,
            ha="center", va="center",
            color=INK, fontsize=8.8)


def _arrow(ax, x0, y0, x1, y1, *, two_way=False):
    style = "<|-|>" if two_way else "-|>"
    arrow = FancyArrowPatch((x0, y0), (x1, y1),
                            arrowstyle=style, mutation_scale=14,
                            color=INK, lw=1.1)
    ax.add_patch(arrow)


def main():
    configure()
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Three primary blocks — frontend / backend / solver.
    _block(ax, 0.2, 3.5, 3.4, 1.9,
           title="Frontend",
           body="TypeScript / Vite\nLayer Builder UI\nlive J–V plots",
           fc="#ecf3fb", ec="#1a73d6", title_color="#0f3a78")

    _block(ax, 4.3, 3.5, 3.4, 1.9,
           title="Backend",
           body="FastAPI · SSE\nstreaming JSON\nREST job control",
           fc="#fff7ec", ec="#c87f08", title_color="#7a4a00")

    _block(ax, 8.4, 3.5, 3.4, 1.9,
           title="Solver core",
           body="Radau / BDF\nbisection-in-time\nNumPy + SciPy",
           fc="#fdecea", ec=ACCENT, title_color=ACCENT)

    # Solver feeds four physics modules along the bottom.
    physics = [
        ("Drift-diffusion\n+ Poisson", "#3a1c4a"),
        ("TMM optics\nAM1.5G", "#0f3a78"),
        ("Mobile-ion\ntransport", "#7a4a00"),
        ("Robin contacts\n($S_n$, $S_p$)", ACCENT),
    ]
    pw, gap = 2.6, 0.30
    total = len(physics) * pw + (len(physics) - 1) * gap
    x0 = (12 - total) / 2
    for i, (lab, c) in enumerate(physics):
        x = x0 + i * (pw + gap)
        _block(ax, x, 0.5, pw, 1.4,
               title="", body=lab,
               fc="white", ec=c, title_color=c)
        # Arrow from solver block down into each physics module.
        _arrow(ax, 10.1, 3.45, x + pw / 2, 1.95)

    # Frontend ⇄ Backend ⇄ Solver horizontal arrows.
    _arrow(ax, 3.65, 4.45, 4.25, 4.45, two_way=True)
    _arrow(ax, 7.75, 4.45, 8.35, 4.45, two_way=True)

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s22_framework.png")
    print(f"wrote {FIG_OUT / 's22_framework.png'}")


if __name__ == "__main__":
    main()
