"""Per-point SS interface-states solve for one CHI_ETL offset (argv: delta_E_C).
Prints one JSON line: {brk, Voc, Jsc(mA/cm2), FF(%), PCE(%)}. brk=False when the
SS-iface Newton + its certified transient fallback cannot reach the point
(deep-CBO: the 1 eV cliff destroys the junction and the SS root is unreachable by
any algebraic Newton — Gummel/Anderson/PTC all stall; see scope doc Section 0).
Run under a wall-clock timeout by the figure driver; on brk=False the figure fills
the point from the plain-transient fallback (SS == transient in the band-offset
regime, iface states inactive).
"""
import sys, json
from pathlib import Path
REPO = Path("/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/perovskite-sim")
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.experiments.steady_state import solve_voc_ss, run_jv_sweep_ss
from perovskite_sim.sweeps.device_parameter_sweep import SweepPoint, apply_sweep_point

CFG = REPO / "configs" / "scaps_mirror_v2.yaml"
dec = float(sys.argv[1])
base = load_scaps_yaml(CFG)                       # parity config: het_recomb_despike = 0.53
sw = apply_sweep_point(base, SweepPoint("p", "CHI_ETL", f"{dec:.3e}", {"etl_delta_ec_eV": dec}))
try:
    voc = solve_voc_ss(sw, N_grid=30, iface_states=True, V_hi=1.6)
    r = run_jv_sweep_ss(sw, N_grid=30, iface_states=True, stop_after_voc=True,
                        V_max=min(1.6, voc + 0.05), n_points=40)
    m = r.metrics
    print(json.dumps(dict(brk=bool(m.voc_bracketed), Voc=float(m.V_oc),
                          Jsc=float(m.J_sc) / 10, FF=float(m.FF) * 100, PCE=float(m.PCE) * 100)))
except Exception as e:
    print(json.dumps(dict(brk=False, err=f"{type(e).__name__}: {str(e)[:80]}")))
