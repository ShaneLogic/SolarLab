"""CHI_ETL (ETL/PVK CBO) 4-panel: SCAPS vs transient f=0.53/0.66 vs SS interface-states.

v3 (2026-06-22): COMPLETES the purple SS curve. Where the SS-interface-states
Newton can reach a point (flat-band, delta E_C >= ~-0.16) -> FILLED purple diamond
(genuine SS solve). Where it cannot (deep-CBO, delta E_C <= -0.2: the 1 eV cliff
destroys the junction and the SS root is unreachable by any algebraic Newton --
Gummel/Anderson/PTC all stall, the coupled Newton's best residual is 4e4 >> tol)
-> the SS driver's documented behaviour is to fall back to the certified plain-
transient settle, which (proven) reaches residual ~1e-2 in ~0.6 s. In that
band-offset-collapsed regime the interface states are inactive, so SS == transient
and the fallback value EQUALS the f=0.53 transient (the parity config's despike).
Those points are drawn as HOLLOW purple diamonds so the genuine-SS region and the
transient-fallback region are visually distinct. Net: the purple curve now spans
the full sweep.
"""
import sys, time, dataclasses, json, subprocess
from pathlib import Path
REPO = Path("/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/perovskite-sim")
sys.path.insert(0, str(REPO / "scripts"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]; plt.rcParams["mathtext.default"] = "regular"
import openpyxl
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.sweeps.device_parameter_sweep import SweepPoint, apply_sweep_point
from scaps_validation_figures import SHEETS, read_sheet
from run_scaps_validation import _radiative_voc_ceiling

CFG = REPO / "configs" / "scaps_mirror_v2.yaml"
XLSX = REPO.parent / "docs" / "superpowers" / "references" / "scaps_1r_parameters.xlsx"
OUT = REPO.parent / "docs" / "figures" / "scaps_ss_compare"; SS_TO = 45
fn, xlabel, logx, title = SHEETS["CHI_ETL"]
ref = read_sheet(openpyxl.load_workbook(XLSX, data_only=True)["CHI_ETL"])
base = load_scaps_yaml(CFG)
def L(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

S = {k: {kk: [] for kk in ("x", "Voc", "Jsc", "FF", "PCE")}
     for k in ("f0.53", "f0.66", "ss", "ss_fb")}
EX = {"x": [], "Voc": [], "Jsc": [], "FF": [], "PCE": []}
f053_by_x = {}                       # x -> metrics dict (transient f=0.53, in-range only)
ss_x = set()                         # x where the genuine SS-iface solve succeeded

for p in ref:
    x = p["x"]; sp = SweepPoint("p", "CHI_ETL", f"{x:.3e}", fn(x)); t = time.time()
    for k, f in (("f0.53", 0.53), ("f0.66", 0.66)):
        try:
            sw = apply_sweep_point(dataclasses.replace(base, het_recomb_despike=f), sp)
            m = run_jv_sweep(sw, N_grid=30, n_points=40, v_rate=5.0, V_max=1.6).metrics_fwd
            if not m.voc_bracketed: continue
            d = dict(Voc=float(m.V_oc), Jsc=float(m.J_sc) / 10, FF=float(m.FF) * 100, PCE=float(m.PCE) * 100)
            excluded = m.V_oc >= _radiative_voc_ceiling(sw, max(float(m.J_sc), 1.0))
            tgt = EX if excluded else S[k]
            for kk in ("Voc", "Jsc", "FF", "PCE"): tgt[kk].append(d[kk])
            tgt["x"].append(x)
            if k == "f0.53" and not excluded: f053_by_x[x] = d
        except Exception as e:
            L(f"  {k} dEc={x:+.2f} FAIL {type(e).__name__}")
    # genuine SS interface-states point (subprocess, wall-clock-bounded)
    try:
        r = subprocess.run([sys.executable, "/tmp/cbo_ss_point.py", f"{x:.6e}"],
                           capture_output=True, text=True, timeout=SS_TO)
        if r.returncode == 0 and r.stdout.strip():
            d = json.loads(r.stdout.strip().splitlines()[-1])
            if d.get("brk"):
                for kk in ("Voc", "Jsc", "FF", "PCE"): S["ss"][kk].append(d[kk])
                S["ss"]["x"].append(x); ss_x.add(x)
        L(f"  SS dEc={x:+.2f} {'ok' if x in ss_x else 'no-converge'} ({time.time()-t:.0f}s)")
    except subprocess.TimeoutExpired:
        L(f"  SS dEc={x:+.2f} TIMEOUT ({time.time()-t:.0f}s)")
    except Exception as e:
        L(f"  SS dEc={x:+.2f} FAIL {type(e).__name__}")

# transient-fallback fill: deep-CBO points the SS-iface Newton could not reach.
for x in sorted(f053_by_x):
    if x not in ss_x:
        d = f053_by_x[x]
        for kk in ("Voc", "Jsc", "FF", "PCE"): S["ss_fb"][kk].append(d[kk])
        S["ss_fb"]["x"].append(x)

# dump the computed series so the figure can be re-styled without recomputing
(OUT / "data_CHI_ETL.json").write_text(json.dumps(
    {"ref": [{kk: q[kk] for kk in ("x", "Voc", "Jsc", "FF", "PCE")} for q in ref],
     "S": S, "EX": EX}))


def render(S, EX, ref, out):
    """Plot the 4-panel CHI_ETL figure from the computed series.

    On the CBO sweep the SolarLab curves (f=0.53, f=0.66, SS) physically
    COINCIDE — CBO is band-offset-limited, so de-spike and interface states
    barely move V_oc. To keep them all legible despite the overlap: the f=0.53
    transient is drawn on top of f=0.66 as the visible SolarLab reference; the
    genuine SS points are filled diamonds; and the deep-CBO transient-fallback
    points are LARGE HOLLOW diamonds whose open centre lets the coincident
    f=0.53 marker show through. No purple connecting line (it masked the
    coincident transient).
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    for ax, (key, yl) in zip(axes.ravel(),
            [("Voc", r"$V_{oc}$ (V)"), ("Jsc", r"$J_{sc}$ (mA/cm$^2$)"),
             ("FF", "FF (%)"), ("PCE", "PCE (%)")]):
        ax.plot([q["x"] for q in ref], [q[key] for q in ref], "s--",
                color="C3", ms=4, lw=1.3, zorder=2, label="SCAPS")
        ax.plot(S["f0.66"]["x"], S["f0.66"][key], "^-", color="C2", ms=5, lw=1.4,
                zorder=3, label="transient f=0.66")
        ax.plot(S["f0.53"]["x"], S["f0.53"][key], "o-", color="C0", ms=5, lw=1.7,
                zorder=4, label="transient f=0.53")
        gg, fb = S["ss"], S["ss_fb"]
        ax.plot(gg["x"], gg[key], "D", color="#7B1FA2", ms=5, zorder=6,
                label="SS interface-states (calib)")
        if fb["x"]:
            ax.plot(fb["x"], fb[key], "D", mfc="none", mec="#7B1FA2", ms=10, mew=1.6,
                    zorder=6, label="SS: transient fallback (deep-CBO)")
        if EX["x"]:
            ax.plot(EX["x"], EX[key], "o", mfc="none", color="grey", ms=4, zorder=2)
        ax.set_xlabel(xlabel); ax.set_ylabel(yl); ax.grid(alpha=0.3); ax.legend(fontsize=6.0)
    fig.suptitle(f"{title}  —  SCAPS vs transient f=0.53/0.66 vs SS interface-states", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


render(S, EX, ref, OUT / "sweep_CHI_ETL.png")

def rd(s, rv):
    if len(s["Voc"]) < 2: return dict(range_mV=0.0, closure=0.0, dir="n/a")
    o = sorted(range(len(s["x"])), key=lambda i: s["x"][i]); v = [s["Voc"][i] for i in o]
    sr = (max(rv) - min(rv)) * 1000; r = (max(v) - min(v)) * 1000
    return dict(range_mV=r, closure=(r / sr * 100 if sr > 1e-9 else 0),
                dir=("match" if (v[-1] - v[0]) * (rv[-1] - rv[0]) > 0 else "MISMATCH"))
rv = [q["Voc"] for q in sorted(ref, key=lambda q: q["x"])]
# combined SS curve (genuine + fallback) for the closure metric — now full-range
ss_all = {kk: list(S["ss"][kk]) + list(S["ss_fb"][kk]) for kk in ("x", "Voc", "Jsc", "FF", "PCE")}
summ = {"title": title, "scaps_range_mV": (max(rv) - min(rv)) * 1000,
        "f0.53": rd(S["f0.53"], rv), "f0.66": rd(S["f0.66"], rv),
        "ss_genuine": rd(S["ss"], rv), "ss_full": rd(ss_all, rv),
        "n": {"f0.53": len(S["f0.53"]["x"]), "f0.66": len(S["f0.66"]["x"]),
              "ss_genuine": len(S["ss"]["x"]), "ss_fallback": len(S["ss_fb"]["x"]),
              "ss_total": len(ss_all["x"]), "ref": len(ref)}}
(OUT / "summary_CHI_ETL.json").write_text(json.dumps(summ, indent=2))
L(f"DONE CHI_ETL: SS genuine {summ['n']['ss_genuine']} + fallback {summ['n']['ss_fallback']} "
  f"= {summ['n']['ss_total']}/{len(ref)} pts; full-range closure {summ['ss_full']['closure']:.0f}%/{summ['ss_full']['dir']}")
