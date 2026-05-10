"""s19 predicted J-V on candidate (placeholder) with reference-stack inset."""
import json
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, FIG_DATA, ACCENT, BODY
from figures.stack_diagram import draw_mapbi3_stack


def main():
    configure()
    data = json.loads((FIG_DATA / "r6_candidate_jv.json").read_text())
    fig = plt.figure(figsize=(11.0, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.30)
    ax = fig.add_subplot(gs[0])
    ax_stack = fig.add_subplot(gs[1])

    # Standard PV sign convention — photocurrent plotted as negative so the
    # operating curve occupies the fourth quadrant. Truncate past V_oc so the
    # forward-bias diode tail does not dominate the y-axis.
    voc = data["metrics"]["voc_V"]
    jsc = data["metrics"]["jsc_A_per_m2"] / 10.0            # mA/cm²
    pce = data["metrics"]["pce_pct"]
    v_max = 1.05 * voc

    v_trim, j_mA = zip(*[(v, -j / 10.0) for v, j in zip(data["v"], data["j"])
                         if v <= v_max])

    ax.plot(v_trim, j_mA, color=ACCENT, lw=2.2,
            label=rf"Top candidate · $V_{{oc}}$ = {voc:.3f} V")
    ax.axhline(0, color="#999", lw=0.6)
    ax.axvline(voc, color=ACCENT, lw=0.6, ls=":", alpha=0.6)

    ax.set_xlim(0.0, v_max)
    ax.set_ylim(-1.20 * jsc, 0.10 * jsc)

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel(r"J (mA/cm$^{2}$)")
    ax.text(0.04, 0.94, f"Predicted PCE = {pce:.1f} %",
            transform=ax.transAxes, fontsize=12, color=BODY,
            fontweight="bold")
    if data.get("placeholder"):
        ax.text(0.04, 0.86, "(placeholder candidate — see notes)",
                transform=ax.transAxes, fontsize=9, color="#999",
                style="italic")
    ax.legend(frameon=False, loc="lower left")

    # Same reference stack as the 2D fidelity run (no GB highlight here).
    draw_mapbi3_stack(ax_stack, show_gb=False,
                      title="Stack (placeholder candidate)")

    fig.tight_layout()
    fig.savefig(FIG_OUT / "s19_candidate_jv.png")
    print(f"wrote {FIG_OUT / 's19_candidate_jv.png'}")


if __name__ == "__main__":
    main()
