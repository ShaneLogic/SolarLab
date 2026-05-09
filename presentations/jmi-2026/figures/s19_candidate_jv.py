"""s19 predicted J-V on candidate (placeholder until R3/R4 close)."""
import json
import matplotlib.pyplot as plt
from figures._common import configure, FIG_OUT, FIG_DATA, ACCENT, BODY

def main():
    configure()
    data = json.loads((FIG_DATA / "r6_candidate_jv.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(data["v"], [j / 10 for j in data["j"]],
            color=ACCENT, lw=2.0, label="Top candidate")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel(r"J (mA/cm$^{2}$)")
    pce = data["metrics"]["pce_pct"]
    ax.text(0.05, 0.92, f"Predicted PCE = {pce:.1f} %",
            transform=ax.transAxes, fontsize=12, color=BODY,
            fontweight="bold")
    if data.get("placeholder"):
        ax.text(0.05, 0.84, "(placeholder candidate — see notes)",
                transform=ax.transAxes, fontsize=9, color="#999",
                style="italic")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "s19_candidate_jv.png")
    print(f"wrote {FIG_OUT / 's19_candidate_jv.png'}")

if __name__ == "__main__":
    main()
