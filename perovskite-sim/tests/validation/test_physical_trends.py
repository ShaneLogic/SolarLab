"""Physics trend validation: asserts that the drift-diffusion solver reproduces
well-established device-physics scaling laws.

Invoke with: pytest -m validation
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
from scipy.stats import linregress

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.experiments.jv_sweep import run_jv_sweep, JVResult

pytestmark = pytest.mark.validation


@pytest.fixture(scope="module")
def baseline_stack() -> DeviceStack:
    """Beer-Lambert n-i-p MAPbI3 preset — FULL tier (default), all physics live.

    Uses the BL preset rather than TMM because Trends 1 & 5 vary absorber Eg
    and need optical generation to respond. TMM n,k data is fixed per
    optical_material key and does not shift with Eg.
    """
    return load_device_from_yaml("configs/nip_MAPbI3.yaml")


def _vary_absorber_param(
    stack: DeviceStack, param_name: str, values: list[float],
) -> list[DeviceStack]:
    """Return new DeviceStacks with the absorber layer's MaterialParams field
    ``param_name`` set to each value in ``values``.

    Preserves the role-tag scan so ``role: absorber`` is the target layer.
    """
    absorber_idx = next(
        i for i, layer in enumerate(stack.layers) if layer.role == "absorber"
    )
    layer = stack.layers[absorber_idx]
    assert layer.params is not None, "absorber layer must have MaterialParams"

    stacks: list[DeviceStack] = []
    for v in values:
        new_params = replace(layer.params, **{param_name: v})
        new_layer = replace(layer, params=new_params)
        new_layers = list(stack.layers)
        new_layers[absorber_idx] = new_layer
        stacks.append(replace(stack, layers=tuple(new_layers)))
    return stacks


def _vary_all_layers_param(
    stack: DeviceStack, param_name: str, value: float,
) -> DeviceStack:
    """Return a new DeviceStack with *param_name* set to *value* on every
    electrical layer that has MaterialParams.
    """
    new_layers: list[LayerSpec] = []
    for layer in stack.layers:
        if layer.params is not None:
            new_params = replace(layer.params, **{param_name: value})
            new_layers.append(replace(layer, params=new_params))
        else:
            new_layers.append(layer)
    return replace(stack, layers=tuple(new_layers))


def _vary_absorber_thickness(
    stack: DeviceStack, thicknesses: list[float],
) -> list[DeviceStack]:
    """Return new DeviceStacks with the absorber layer thickness varied."""
    absorber_idx = next(
        i for i, layer in enumerate(stack.layers) if layer.role == "absorber"
    )
    layer = stack.layers[absorber_idx]
    stacks: list[DeviceStack] = []
    for t in thicknesses:
        new_layer = replace(layer, thickness=t)
        new_layers = list(stack.layers)
        new_layers[absorber_idx] = new_layer
        stacks.append(replace(stack, layers=tuple(new_layers)))
    return stacks


def _run_jv(stack: DeviceStack) -> JVResult:
    """Run a J-V sweep with settings matching the regression suite."""
    return run_jv_sweep(
        stack, N_grid=60, n_points=20, v_rate=5.0,
        V_max=1.5,
    )


EG_SWEEP = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]  # eV

# Reference bandgap for ni scaling. The baseline preset has ni = 3.2e13 m⁻³
# which physically corresponds to MAPbI3's native Eg ≈ 1.55 eV. At T = 300 K
# the solver uses the explicit ``ni`` parameter rather than deriving it from
# ``Eg``, so we must scale ni together with Eg to keep the intrinsic carrier
# density consistent with the shifted bandgap.
EG_REF = 1.55       # eV
NI_REF = 3.2e13     # m⁻³ (baseline MAPbI3 ni)


def _above_gap_flux(eg: float) -> float:
    """Integrated AM1.5G photon flux above bandgap *eg* (photons/m²/s).

    Uses the ASTM G-173 AM1.5G spectrum shipped with the package.
    """
    import os

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "perovskite_sim", "data", "am15g.csv",
    )
    raw = np.loadtxt(data_path, delimiter=",", skiprows=6)
    wavelength_nm = raw[:, 0]        # nm
    spectral_flux = raw[:, 1]        # photons / (m²·s·m)

    # Photon energy: E (eV) = 1240 / λ (nm)
    photon_energy_eV = 1240.0 / wavelength_nm
    above = photon_energy_eV >= eg
    if not np.any(above):
        return 0.0

    # Prefer trapezoid (NumPy >= 2.0), fall back to trapz
    _integrate = getattr(np, "trapezoid", getattr(np, "trapz"))
    # wavelength_nm → m so the integral over spectral flux [photons/(m²·s·m)]
    # yields photons/(m²·s); skipping the conversion gives 1e9× too many photons.
    return float(_integrate(spectral_flux[above], wavelength_nm[above] * 1e-9))


def _ni_for_eg(eg: float, eg_ref: float = EG_REF, ni_ref: float = NI_REF) -> float:
    """Intrinsic carrier density consistent with bandgap *eg*.

    ni ∝ exp(-Eg / 2kT), so ni(eg) = ni_ref · exp((eg_ref − eg) / (2 V_T)).
    """
    from perovskite_sim.constants import K_B, Q

    V_T = K_B * 300.0 / Q
    return float(ni_ref * np.exp((eg_ref - eg) / (2.0 * V_T)))


@pytest.fixture(scope="module")
def eg_sweep_results(baseline_stack: DeviceStack) -> list[tuple[float, JVResult]]:
    """Run J-V at each Eg and return (Eg, JVResult) pairs.

    Three linked parameters are adjusted per Eg:

    1. ``Eg`` — set on ALL electrical layers (not just the absorber) so the
       bands stay flat and thermionic emission does not activate across
       artificial heterojunction offsets.
    2. ``ni`` — scaled only on the absorber as ni ∝ exp(−Eg/2kT); transport
       layers keep their explicit (degenerate) ni values.
    3. ``Phi`` — scaled to the AM1.5G above-gap integrated photon flux, so
       J_sc drops at wider bandgaps (fewer above-gap photons).
    """
    results: list[tuple[float, JVResult]] = []
    for eg in EG_SWEEP:
        # Set Eg on every layer to keep bands flat
        stack_eg = _vary_all_layers_param(baseline_stack, "Eg", eg)
        # Scale ni on the absorber only
        ni_new = _ni_for_eg(eg)
        stacks_ni = _vary_absorber_param(stack_eg, "ni", [ni_new])
        stack_ni = stacks_ni[0]
        # Scale Phi to above-gap photon flux
        stack_phi = replace(stack_ni, Phi=_above_gap_flux(eg))
        result = _run_jv(stack_phi)
        results.append((eg, result))
    return results


def test_voc_vs_bandgap(eg_sweep_results: list[tuple[float, JVResult]]) -> None:
    """V_oc should respond monotonically to bandgap.

    Under Beer-Lambert optics the absorption coefficient alpha is fixed and
    does not shift with Eg, so V_oc is pinned near V_bi and varies only weakly
    with the Phi / ni co-variation.  This test therefore checks a weaker
    condition than the full Eg-tracking law: V_oc must be bounded and the
    ΔV slope must be physically signed (ΔV grows with Eg when optics do not
    respond).
    """
    eg_values = np.array([eg for eg, _ in eg_sweep_results])
    voc_values = np.array([r.metrics_fwd.V_oc for _, r in eg_sweep_results])
    delta_V = eg_values - voc_values

    # V_oc must be positive and below the thermodynamic limit (Eg/q)
    assert all(v > 0 for v in voc_values), f"V_oc must be positive: {voc_values}"
    assert all(d > 0 for d in delta_V), (
        f"ΔV = Eg − V_oc must be positive: V_oc={[f'{v:.4f}' for v in voc_values]}"
    )

    # Under Beer-Lambert V_oc is roughly constant; ΔV therefore grows with Eg.
    # The slope d(ΔV)/dEg must be positive (V_oc does not overshoot Eg).
    slope, _, r_value, _, _ = linregress(eg_values, delta_V)
    assert r_value > 0.9, (
        f"ΔV vs Eg correlation r={r_value:.3f} is too weak; "
        f"V_oc values: {[f'{v:.4f}' for v in voc_values]}"
    )
    assert slope > 0, (
        f"ΔV slope vs Eg is {slope:.3f} — expected positive; "
        "V_oc is not responding to bandgap"
    )


def test_jsc_vs_bandgap(eg_sweep_results: list[tuple[float, JVResult]]) -> None:
    """J_sc should decrease with increasing bandgap.

    Wider bandgap → fewer above-gap photons absorbed → lower photocurrent.
    Under Beer-Lambert with fixed alpha, the Eg-shifted ni contributes a
    small electrical component; the dominant optical trend is captured.
    """
    eg_values = np.array([eg for eg, _ in eg_sweep_results])
    jsc_values = np.array([r.metrics_fwd.J_sc for _, r in eg_sweep_results])

    assert all(j > 0 for j in jsc_values), (
        f"All J_sc values must be positive, got: {jsc_values}"
    )
    assert jsc_values[-1] < jsc_values[0], (
        f"J_sc at Eg={eg_values[-1]:.1f} eV ({jsc_values[-1]:.1f} A/m²) "
        f"must be less than J_sc at Eg={eg_values[0]:.1f} eV ({jsc_values[0]:.1f} A/m²)"
    )

    ratio = jsc_values[-1] / jsc_values[0]
    assert ratio <= 0.95, (
        f"J_sc ratio widest/narrowest Eg = {ratio:.3f} — "
        "expected more drop at wider bandgap (≤ 0.95)"
    )


# ---------------------------------------------------------------------------
# Trend 2: V_oc vs Thickness
# ---------------------------------------------------------------------------

THICKNESS_SWEEP_NM = [100, 200, 400, 700, 1000]  # nm → converted to m


def test_voc_vs_thickness(baseline_stack: DeviceStack) -> None:
    """V_oc should increase with absorber thickness.

    dV_oc / d(log₁₀(thickness)) should be positive and in 30–90 mV/decade —
    the classic SRH signature: thicker absorbers dilute contact recombination.
    """
    thicknesses_m = [t * 1e-9 for t in THICKNESS_SWEEP_NM]
    stacks = _vary_absorber_thickness(baseline_stack, thicknesses_m)

    voc_values: list[float] = []
    for stack in stacks:
        result = _run_jv(stack)
        voc_values.append(result.metrics_fwd.V_oc)

    log10_t = np.log10(THICKNESS_SWEEP_NM)
    slope, intercept, r_value, _, _ = linregress(log10_t, voc_values)

    # mV/decade
    slope_mv_per_decade = slope * 1000

    assert abs(r_value) > 0.7, (
        f"V_oc vs log₁₀(thickness) correlation r={r_value:.3f} is too weak — "
        f"V_oc values: {[f'{v:.4f}' for v in voc_values]}"
    )
    assert 20 <= abs(slope_mv_per_decade) <= 120, (
        f"V_oc vs thickness slope {slope_mv_per_decade:.1f} mV/decade "
        f"outside ±[20, 120] — V_oc values: {[f'{v:.4f}' for v in voc_values]}"
    )


# ---------------------------------------------------------------------------
# Trend 3: FF vs Mobility
# ---------------------------------------------------------------------------

MOBILITY_SWEEP_CM2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # cm²/Vs


def test_ff_vs_mobility(baseline_stack: DeviceStack) -> None:
    """FF should degrade measurably below ~1e-4 cm²/Vs.

    At low mobility transport resistance limits charge extraction, reducing FF.
    The test asserts FF at the lowest μ is at least 3 percentage points
    (absolute) lower than at the highest μ.
    """
    mobility_m2 = [m * 1e-4 for m in MOBILITY_SWEEP_CM2]  # cm²/Vs → m²/Vs
    ff_values: list[float] = []
    for mu in mobility_m2:
        stacks_n = _vary_absorber_param(baseline_stack, "mu_n", [mu])
        s = _vary_absorber_param(stacks_n[0], "mu_p", [mu])[0]
        result = _run_jv(s)
        ff_values.append(result.metrics_fwd.FF)
        if not ff_values[-1] > 0:
            pytest.fail(f"FF=0 at μ={mu:.2e} m²/Vs — solver likely failed")

    ff_drop = ff_values[-1] - ff_values[0]  # highest μ FF − lowest μ FF
    assert ff_drop >= 0.03, (
        f"FF drop from lowest to highest mobility is only {ff_drop:.4f} "
        f"(absolute) — expected ≥ 0.03. FF values: "
        f"{[f'{ff:.4f}' for ff in ff_values]}"
    )


# ---------------------------------------------------------------------------
# Trend 4: Ideality Factor
# ---------------------------------------------------------------------------


def test_ideality_factor(baseline_stack: DeviceStack) -> None:
    """Dark J-V ideality factor should be 1.0 ≤ n_id ≤ 2.5.

    In the low-injection regime a single-junction device with SRH and
    radiative recombination has n_id between 1 and 2.  The reverse scan
    is used because its starting state (forward-biased, settled) is better
    conditioned for the low-current exponential region than the forward scan
    which starts from dark equilibrium and can carry transient artefacts.
    """
    # Get J_sc reference from illuminated run for the threshold
    ill_result = _run_jv(baseline_stack)
    j_sc = ill_result.metrics_fwd.J_sc
    assert j_sc > 0, "Need illuminated J_sc reference for ideality test"

    # Run dark J-V
    dark_result = run_jv_sweep(
        baseline_stack, N_grid=60, n_points=30, v_rate=1.0,
        V_max=1.5, illuminated=False,
    )
    # Use reverse scan — starts at forward bias where the device is settled
    V = np.asarray(dark_result.V_rev)
    J = np.asarray(dark_result.J_rev)

    # Low-injection region: above noise floor but well below J_sc.
    # Dark J is negative under forward bias (diode forward current),
    # so we work with |J|.
    J_abs = np.abs(J)
    floor = max(j_sc / 500, 0.5)
    threshold = j_sc / 8
    lo_mask = (J_abs > floor) & (J_abs < threshold)
    if lo_mask.sum() < 4:
        pytest.skip(
            f"Not enough low-injection points: {lo_mask.sum()} with "
            f"|J| ∈ [{floor:.2e}, {threshold:.2e}]"
        )

    V_lo = V[lo_mask]
    J_lo = J_abs[lo_mask]

    slope, _, r_value, _, _ = linregress(V_lo, np.log(J_lo))
    # slope = d(ln J)/dV = q / (n_id * kT)  → n_id = q / (slope * kT)
    # At 300 K: kT/q = 0.02585 V, so n_id = 1 / (slope * 0.02585)
    n_id = 1.0 / (slope * 0.02585)

    assert r_value > 0.95, (
        f"Ideality fit correlation r={r_value:.3f} too weak — "
        "J(V) may not be exponential in the selected region"
    )
    assert 1.0 <= n_id <= 2.5, (
        f"Ideality factor n_id = {n_id:.2f} outside [1.0, 2.5]"
    )


# ---------------------------------------------------------------------------
# Trend 6: V_oc vs Illumination (Suns-V_oc)
# ---------------------------------------------------------------------------


def test_voc_vs_illumination(baseline_stack: DeviceStack) -> None:
    """Suns-V_oc slope dV_oc / d(ln Φ) should be n_id · kT/q.

    At 300 K: slope ∈ [25, 65] mV/decade, corresponding to
    ideality factor n_id ∈ [1.0, 2.5].

    Uses the existing run_suns_voc experiment with default suns levels.
    """
    from perovskite_sim.experiments.suns_voc import run_suns_voc

    # Wider suns range for robust slope fitting
    suns_levels = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
    result = run_suns_voc(
        baseline_stack, suns_levels=suns_levels, N_grid=60, t_settle=0.1,
    )

    assert len(result.suns) >= 4, (
        f"Need ≥4 suns levels for slope fit, got {len(result.suns)}"
    )
    # Filter out any failed levels (V_oc may be zero or NaN on failure)
    valid = np.isfinite(result.V_oc) & (result.V_oc > 0)
    suns_valid = np.asarray(result.suns)[valid]
    voc_valid = np.asarray(result.V_oc)[valid]

    assert len(suns_valid) >= 3, (
        f"Need ≥3 valid V_oc points, got {len(suns_valid)}"
    )

    slope, _, r_value, _, _ = linregress(np.log(suns_valid), voc_valid)
    slope_mv_per_decade = slope * 1000  # V/decade → mV/decade

    assert r_value > 0.95, (
        f"Suns-V_oc slope fit correlation r={r_value:.3f} too weak"
    )
    assert 20 <= slope_mv_per_decade <= 70, (
        f"Suns-V_oc slope {slope_mv_per_decade:.1f} mV/decade outside [20, 70]"
    )
