"""s23 — band diagram with carrier-flux arrows for the drift-diffusion equations."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from figures._common import configure, FIG_OUT, INK, BODY, ACCENT, HAIRLINE


def main():
    configure()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    # Three regions along x: HTL [0, 1] / absorber [1, 4] / ETL [4, 5].
    x = np.linspace(0, 5, 500)

    # Conduction-band edge — small barrier at HTL, drop into absorber, level at ETL.
    Ec = np.piecewise(
        x,
        [x < 1, (x >= 1) & (x < 4), x >= 4],
        [lambda x: 1.7 - 0.1 * x,
         lambda x: 1.6 - 0.10 * (x - 1),
         lambda x: 1.30 - 0.05 * (x - 4)],
    )
    # Valence-band edge — mirrored offset to give a roughly 1.55 eV gap mid-stack.
    Ev = np.piecewise(
        x,
        [x < 1, (x >= 1) & (x < 4), x >= 4],
        [lambda x: 0.30 - 0.05 * x,
         lambda x: 0.05 - 0.02 * (x - 1),
         lambda x: -0.40 - 0.02 * (x - 4)],
    )

    ax.plot(x, Ec, color=INK, lw=1.6, label=r"$E_c$")
    ax.plot(x, Ev, color=INK, lw=1.6, label=r"$E_v$")
    ax.fill_between(x, Ev, Ec, color="#f3eaff", alpha=0.35, lw=0)

    # Layer separators + labels along the top.
    for xb in (1.0, 4.0):
        ax.axvline(xb, color=HAIRLINE, lw=0.6, ls="--")
    ax.text(0.5, 1.85, "HTL", ha="center", color=BODY, fontsize=10)
    ax.text(2.5, 1.85, "absorber",  ha="center", color=BODY, fontsize=10)
    ax.text(4.5, 1.85, "ETL", ha="center", color=BODY, fontsize=10)

    # Generation arrow — photons absorbed in absorber generating an e⁻/h⁺ pair.
    ax.annotate("",
                xy=(2.4, 0.10), xytext=(2.4, 1.55),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.4))
    ax.text(2.55, 0.85, "G", color=ACCENT, fontsize=12, weight="bold")

    # Electron flux arrow — toward the ETL (right).
    arr_n = FancyArrowPatch((1.3, 1.45), (4.0, 1.30),
                            arrowstyle="-|>", mutation_scale=14,
                            color="#1a73d6", lw=1.6)
    ax.add_patch(arr_n)
    ax.text(2.7, 1.55, r"$J_n$ — electrons", color="#1a73d6", fontsize=10)

    # Hole flux arrow — toward the HTL (left).
    arr_p = FancyArrowPatch((3.7, -0.35), (1.0, -0.20),
                            arrowstyle="-|>", mutation_scale=14,
                            color=ACCENT, lw=1.6)
    ax.add_patch(arr_p)
    ax.text(2.4, -0.50, r"$J_p$ — holes", color=ACCENT, fontsize=10)

    # Recombination arrow inside the absorber.
    ax.annotate("R", xy=(3.4, 0.6),
                color=BODY, fontsize=11, style="italic")
    ax.annotate("",
                xy=(3.5, 0.10), xytext=(3.5, 1.45),
                arrowprops=dict(arrowstyle="<->", color=BODY,
                                lw=1.0, ls=":"))

    ax.set_xlim(0, 5)
    ax.set_ylim(-0.7, 2.0)
    ax.set_xlabel("position x (a.u.)")
    ax.set_ylabel("energy (eV)")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s23_band_diagram.png")
    print(f"wrote {FIG_OUT / 's23_band_diagram.png'}")


if __name__ == "__main__":
    main()
