"""s24 — TMM coherent optics: |E(x)|^2 standing-wave + generation profile."""
import numpy as np
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, INK, BODY, ACCENT, SECONDARY, HAIRLINE


def main():
    configure()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    # Toy stack — HTL [0, 0.2] / absorber [0.2, 0.8] / ETL [0.8, 1.0] (normalized).
    x = np.linspace(0, 1.0, 800)

    # |E(x)|^2 — sinusoidal standing wave inside absorber, decays into ETL.
    E2 = np.where(
        x < 0.2,
        1.0 + 0.10 * np.cos(2 * np.pi * 6 * x),
        np.where(
            x < 0.8,
            (0.85 + 0.30 * np.cos(2 * np.pi * 4 * (x - 0.2))) *
                np.exp(-1.5 * (x - 0.2)),
            (0.55 - 0.20 * (x - 0.8)) * np.exp(-1.5 * (0.8 - 0.2)),
        ),
    )
    ax.plot(x, E2, color=SECONDARY, lw=1.6,
            label=r"$|E(x,\lambda)|^{2}$")
    ax.fill_between(x, 0, E2, color=SECONDARY, alpha=0.10, lw=0)

    # Optical generation profile G(x) ~ |E|^2 alpha(x) — only in absorber.
    alpha = np.where((x >= 0.2) & (x <= 0.8), 1.5, 0.0)
    G = E2 * alpha
    ax.plot(x, G, color=ACCENT, lw=1.8,
            label=r"$G(x,\lambda) \propto |E|^{2}\,\alpha$")
    ax.fill_between(x, 0, G, color=ACCENT, alpha=0.18, lw=0)

    # Layer separators.
    for xb in (0.2, 0.8):
        ax.axvline(xb, color=HAIRLINE, lw=0.6, ls="--")
    ax.text(0.10, 1.55, "HTL", ha="center", color=BODY, fontsize=10)
    ax.text(0.50, 1.55, "absorber",  ha="center", color=BODY, fontsize=10)
    ax.text(0.90, 1.55, "ETL", ha="center", color=BODY, fontsize=10)

    # Incoming-illumination arrow on the left.
    ax.annotate("",
                xy=(0.02, 1.30), xytext=(-0.08, 1.30),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4),
                annotation_clip=False)
    ax.text(-0.10, 1.45, "AM1.5G", color=INK, fontsize=9, ha="left")

    # Layer-matrix annotation.
    ax.text(0.50, 0.05,
            r"$M = L_{\mathrm{HTL}}\,L_{\mathrm{abs}}\,L_{\mathrm{ETL}}$"
            r"$\quad\delta_j = \frac{2\pi}{\lambda}\tilde n_j d_j\cos\theta_j$",
            ha="center", va="bottom",
            color=INK, fontsize=10)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.85)
    ax.set_xlabel("position x / total stack thickness")
    ax.set_ylabel("relative amplitude")
    ax.set_xticks([])
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s24_tmm_interference.png")
    print(f"wrote {FIG_OUT / 's24_tmm_interference.png'}")


if __name__ == "__main__":
    main()
