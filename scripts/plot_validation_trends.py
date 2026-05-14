"""Generate multi-panel validation figure from the physical trend tests.

Run from perovskite-sim/:  python ../scripts/plot_validation_trends.py
Output: validation_trends.png
"""

from __future__ import annotations

from dataclasses import replace
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import linregress

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.experiments.jv_sweep import run_jv_sweep, JVResult
from perovskite_sim.constants import K_B, Q

matplotlib.use("Agg")

# ── helpers (mirrors test_physical_trends.py) ──────────────────────────

EG_REF, NI_REF = 1.55, 3.2e13
EG_SWEEP = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
THICKNESS_SWEEP_NM = [100, 200, 400, 700, 1000]
MOBILITY_SWEEP_CM2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def _vary_absorber_param(stack, param_name, values):
    absorber_idx = next(i for i, layer in enumerate(stack.layers) if layer.role == "absorber")
    layer = stack.layers[absorber_idx]
    stacks = []
    for v in values:
        new_params = replace(layer.params, **{param_name: v})
        new_layer = replace(layer, params=new_params)
        new_layers = list(stack.layers)
        new_layers[absorber_idx] = new_layer
        stacks.append(replace(stack, layers=tuple(new_layers)))
    return stacks


def _vary_all_layers_param(stack, param_name, value):
    new_layers = []
    for layer in stack.layers:
        if layer.params is not None:
            new_layers.append(replace(layer, params=replace(layer.params, **{param_name: value})))
        else:
            new_layers.append(layer)
    return replace(stack, layers=tuple(new_layers))


def _vary_absorber_thickness(stack, thicknesses):
    absorber_idx = next(i for i, layer in enumerate(stack.layers) if layer.role == "absorber")
    layer = stack.layers[absorber_idx]
    return [replace(stack, layers=tuple(
        replace(layer, thickness=t) if i == absorber_idx else l
        for i, l in enumerate(stack.layers)
    )) for t in thicknesses]


def _above_gap_flux(eg):
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "perovskite-sim",
                             "perovskite_sim", "data", "am15g.csv")
    raw = np.loadtxt(data_path, delimiter=",", skiprows=6)
    wavelength_nm, spectral_flux = raw[:, 0], raw[:, 1]
    photon_energy_eV = 1240.0 / wavelength_nm
    above = photon_energy_eV >= eg
    if not np.any(above):
        return 0.0
    _integrate = getattr(np, "trapezoid", getattr(np, "trapz"))
    return float(_integrate(spectral_flux[above], wavelength_nm[above] * 1e-9))


def _ni_for_eg(eg):
    V_T = K_B * 300.0 / Q
    return float(NI_REF * np.exp((EG_REF - eg) / (2.0 * V_T)))


def _run_jv(stack):
    return run_jv_sweep(stack, N_grid=60, n_points=20, v_rate=5.0, V_max=1.5)


# ── data collection ────────────────────────────────────────────────────

print("Loading baseline...")
baseline = load_device_from_yaml("configs/nip_MAPbI3.yaml")

# Trend 1 & 5: Eg sweep
print("Sweeping Eg...")
eg_results = []
for eg in EG_SWEEP:
    s = _vary_all_layers_param(baseline, "Eg", eg)
    s = _vary_absorber_param(s, "ni", [_ni_for_eg(eg)])[0]
    s = replace(s, Phi=_above_gap_flux(eg))
    eg_results.append((eg, _run_jv(s)))

# Trend 2: thickness
print("Sweeping thickness...")
thick_stacks = _vary_absorber_thickness(baseline, [t * 1e-9 for t in THICKNESS_SWEEP_NM])
thick_results = [_run_jv(s) for s in thick_stacks]

# Trend 3: mobility
print("Sweeping mobility...")
mob_results = []
for mu_cm2 in MOBILITY_SWEEP_CM2:
    mu_m2 = mu_cm2 * 1e-4
    s = _vary_absorber_param(baseline, "mu_n", [mu_m2])[0]
    s = _vary_absorber_param(s, "mu_p", [mu_m2])[0]
    mob_results.append(_run_jv(s))

# Trend 4: ideality
print("Running dark J-V...")
ill_result = _run_jv(baseline)
j_sc = ill_result.metrics_fwd.J_sc
dark = run_jv_sweep(baseline, N_grid=60, n_points=30, v_rate=1.0, V_max=1.5, illuminated=False)
V_rev = np.asarray(dark.V_rev)
J_rev = np.abs(np.asarray(dark.J_rev))
floor, threshold = max(j_sc / 500, 0.5), j_sc / 8
lo = (J_rev > floor) & (J_rev < threshold)
V_lo, J_lo = V_rev[lo], J_rev[lo]
n_slope, n_int, n_r, _, _ = linregress(V_lo, np.log(J_lo))
n_id = 1.0 / (n_slope * 0.02585)

# Trend 6: Suns-V_oc
print("Running Suns-V_oc...")
from perovskite_sim.experiments.suns_voc import run_suns_voc

suns_levels = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
suns_result = run_suns_voc(baseline, suns_levels=suns_levels, N_grid=60, t_settle=0.1)
valid = np.isfinite(suns_result.V_oc) & (suns_result.V_oc > 0)
suns_x = np.log(np.asarray(suns_result.suns)[valid])
suns_y = np.asarray(suns_result.V_oc)[valid]
sun_slope, _, sun_r, _, _ = linregress(suns_x, suns_y)

# ── plotting ───────────────────────────────────────────────────────────

print("Plotting...")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
(ax1, ax2, ax3), (ax4, ax5, ax6) = axes

# Panel 1: V_oc vs Eg
eg_vals = np.array([e for e, _ in eg_results])
voc_vals = np.array([r.metrics_fwd.V_oc for _, r in eg_results])
dv = eg_vals - voc_vals
ax1.plot(eg_vals, voc_vals, "ko-", ms=5)
ax1.set(xlabel="Bandgap (eV)", ylabel="V$_{oc}$ (V)", title="V$_{oc}$ vs Bandgap")
for e, v in zip(eg_vals, voc_vals):
    ax1.annotate(f"{v:.3f}", (e, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)

ax1b = ax1.twinx()
ax1b.plot(eg_vals, dv * 1000, "rs--", ms=5, mfc="none")
ax1b.set(ylabel=r"$\Delta V$ = E$_g$ − V$_{oc}$ (mV)")
s, _, r, _, _ = linregress(eg_vals, dv)
ax1b.annotate(f"slope={s:.3f}, r={r:.3f}", xy=(0.95, 0.05), xycoords="axes fraction",
              ha="right", fontsize=7, color="red")

# Panel 2: J_sc vs Eg
jsc_vals = np.array([r.metrics_fwd.J_sc for _, r in eg_results])
ax2.plot(eg_vals, jsc_vals, "ko-", ms=5)
ax2.set(xlabel="Bandgap (eV)", ylabel="J$_{sc}$ (A/m²)", title="J$_{sc}$ vs Bandgap")
ax2.annotate(f"ratio 2.2/1.2 = {jsc_vals[-1]/jsc_vals[0]:.3f}", xy=(0.95, 0.05),
             xycoords="axes fraction", ha="right", fontsize=7)

# Panel 3: V_oc vs Thickness
thick_voc = [r.metrics_fwd.V_oc for r in thick_results]
log_t = np.log10(THICKNESS_SWEEP_NM)
ax3.plot(THICKNESS_SWEEP_NM, thick_voc, "ko-", ms=5)
ax3.set(xlabel="Thickness (nm)", ylabel="V$_{oc}$ (V)", title="V$_{oc}$ vs Thickness")
s_t, _, r_t, _, _ = linregress(log_t, thick_voc)
ax3.annotate(f"dV$_{{oc}}$/d log$_{{10}}$(t) = {s_t*1000:.1f} mV/dec\nr = {r_t:.3f}",
             xy=(0.95, 0.05), xycoords="axes fraction", ha="right", fontsize=7)

# Panel 4: FF vs Mobility
ff_vals = [r.metrics_fwd.FF for r in mob_results]
ax4.semilogx(MOBILITY_SWEEP_CM2, ff_vals, "ko-", ms=5)
ax4.set(xlabel="Mobility (cm²/Vs)", ylabel="FF", title="FF vs Mobility")
ax4.annotate(f"ΔFF = {ff_vals[-1]-ff_vals[0]:.4f}", xy=(0.95, 0.05),
             xycoords="axes fraction", ha="right", fontsize=7)

# Panel 5: Ideality Factor
V_fit = np.linspace(V_lo.min() * 0.98, V_lo.max() * 1.02, 50)
J_fit = np.exp(n_slope * V_fit + n_int)
ax5.semilogy(V_rev, J_rev, "k.", ms=4, alpha=0.5, label="dark |J|")
ax5.semilogy(V_lo, J_lo, "bo", ms=5, mfc="none", label="fit region")
ax5.semilogy(V_fit, J_fit, "r-", lw=1, label=f"n$_{{id}}$ = {n_id:.2f}")
ax5.axhline(floor, color="gray", ls=":", lw=0.8)
ax5.axhline(threshold, color="gray", ls=":", lw=0.8)
ax5.set(xlabel="V (V)", ylabel="|J| (A/m²)", title="Ideality Factor (dark J-V)")
ax5.legend(fontsize=7, loc="upper left")

# Panel 6: Suns-V_oc
suns_fit_x = np.linspace(suns_x.min() * 1.05, suns_x.max() * 0.95, 50)
suns_fit_y = sun_slope * suns_fit_x + suns_result.V_oc[-1] - sun_slope * suns_x[-1]
ax6.semilogx(np.exp(suns_x), suns_y * 1000, "ko-", ms=5)
ax6.set(xlabel="Illumination (suns)", ylabel="V$_{oc}$ (mV)",
        title=f"Suns-V$_{{oc}}$  (slope = {sun_slope*1000:.1f} mV/dec, r = {sun_r:.3f})")

fig.tight_layout()
out = "validation_trends.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
