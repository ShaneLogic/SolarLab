"""s11 — composite PV-score factor curves f_gap, m, eps, E_b."""
import numpy as np
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, INK, BODY, ACCENT, SECONDARY


def _f_gap(eg):
    """Triangular SQ-like window peaked at 1.34 eV, vanishing at 0.5 / 2.5."""
    out = np.zeros_like(eg)
    rising = (eg >= 0.5) & (eg <= 1.34)
    falling = (eg > 1.34) & (eg <= 2.5)
    out[rising] = (eg[rising] - 0.5) / (1.34 - 0.5)
    out[falling] = (2.5 - eg[falling]) / (2.5 - 1.34)
    return out


def main():
    configure()
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.0))

    eg = np.linspace(0.3, 2.7, 400)
    m = np.linspace(0.0, 3.0, 400)
    eps = np.linspace(0.0, 30.0, 400)
    eb = np.linspace(0.0, 1.0, 400)

    panels = [
        (axes[0, 0], eg, _f_gap(eg), r"$f_{gap}(E_g)$",
         r"$E_g$ (eV)", ACCENT, 1.34, "1.34 eV peak"),
        (axes[0, 1], m, 1.0 / (1.0 + m), r"$1/(1+m_{avg})$",
         r"$m_{avg}$ ($m_0$)", SECONDARY, 0.5, ""),
        (axes[1, 0], eps, np.tanh(eps / 10.0), r"$\tanh(\varepsilon/10)$",
         r"$\varepsilon$ (—)", "#1aa37a", 10.0, "ε ≈ 10 saturation"),
        (axes[1, 1], eb, 1.0 / (1.0 + eb / 0.1), r"$1/(1 + E_b/0.1)$",
         r"$E_b$ (eV)", "#c87f08", 0.1, "Wannier–Mott < 0.1 eV"),
    ]

    for ax, x, y, title, xlab, color, vline, note in panels:
        ax.plot(x, y, color=color, lw=1.8)
        ax.fill_between(x, 0, y, color=color, alpha=0.15, lw=0)
        if vline is not None:
            ax.axvline(vline, color=BODY, lw=0.6, ls=":")
            if note:
                ax.text(vline, 0.05, "  " + note,
                        ha="left", va="bottom",
                        color=BODY, fontsize=7.5, style="italic")
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=2)
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(axis="both", labelsize=8)

    fig.suptitle(
        r"$y = f_{gap}\,\cdot\,\frac{1}{1+m_{avg}}\,\cdot\,"
        r"\tanh(\varepsilon/10)\,\cdot\,\frac{1}{1+E_b/0.1}$",
        fontsize=10.5, color=INK, y=1.00)

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s11_pv_score_factors.png")
    print(f"wrote {FIG_OUT / 's11_pv_score_factors.png'}")


if __name__ == "__main__":
    main()
