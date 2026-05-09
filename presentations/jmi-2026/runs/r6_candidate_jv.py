"""R6: candidate J-V via SolarLab device simulator (spec § 9, R6).

Until real DFT-screened candidate parameters arrive (gates R3 + R4 in the
risk register), this script produces a *synthetic placeholder* J-V curve
rather than a fresh simulator run. Reasons:

1. The legacy `nip_MAPbI3_singleGB.yaml` preset drives absorption through
   Beer–Lambert with fixed α and Φ, so a bare `Eg`-field shift would not
   actually move the optical absorption edge — the apparent bandgap of
   the simulated device would be unchanged.

2. A meaningful candidate run needs (a) TMM optics (n, k for the
   candidate material), (b) a candidate-derived parameter set
   (mobility, dielectric, traps), neither of which is available yet.

3. The R5 2D run took ~22 minutes; rerunning it under R6 with no
   physically meaningful change would just duplicate that figure.

The placeholder mirrors R5's 2D shape but mildly perturbed (V_oc nudged
up to reflect a slightly wider effective bandgap, FF unchanged), and
flags `placeholder: true` so downstream figure scripts can annotate the
caveat on slide 19.

When real DFT-screened params land, replace this script with a real
2D sim driven by candidate.yaml under TMM optics.
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures" / "data" / "r6_candidate_jv.json"
R5_JSON = OUT.parent / "r5_jv_curves.json"

CANDIDATE_NOTE = (
    "PLACEHOLDER candidate — synthetic curve based on R5's 2D MAPbI3 baseline "
    "with V_oc nudged +30 mV (representative of a slightly wider-gap candidate). "
    "Real DFT-screened parameters slot in once R3/R4 close (spec § 9). "
    "See module docstring for why a literal Eg-shift on the legacy preset "
    "would not produce a physically meaningful difference."
)

V_OC_BUMP = 0.030  # V — placeholder bandgap-shift surrogate


def main():
    base = json.loads(R5_JSON.read_text())
    v_2d = list(base["v_2d"])
    j_2d = list(base["j_2d"])
    m_2d = base["metrics_2d"]

    # Shift the J-V curve to the right by V_OC_BUMP (rigid translation in V).
    # Drop the trailing point that falls past V_max; keep length consistent.
    v = [v + V_OC_BUMP for v in v_2d]

    payload = {
        "placeholder": True,
        "note": CANDIDATE_NOTE,
        "v": v,
        "j": j_2d,
        "metrics": {
            "voc_V": m_2d["voc_V"] + V_OC_BUMP,
            "jsc_A_per_m2": m_2d["jsc_A_per_m2"],
            "ff": m_2d["ff"],
            # PCE = J_sc * V_oc * FF / 1000  (J in A/m², V in V, P_in = 1000 W/m²)
            "pce_pct": (m_2d["jsc_A_per_m2"] *
                        (m_2d["voc_V"] + V_OC_BUMP) *
                        m_2d["ff"] / 1000.0) * 100.0,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    print(f"  candidate (placeholder): "
          f"V_oc={payload['metrics']['voc_V']:.3f} V, "
          f"PCE={payload['metrics']['pce_pct']:.2f} %")


if __name__ == "__main__":
    main()
