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
    return float(_integrate(spectral_flux[above], wavelength_nm[above]))


def _ni_for_eg(eg: float, eg_ref: float = EG_REF, ni_ref: float = NI_REF) -> float:
    """Intrinsic carrier density consistent with bandgap *eg*.

    ni ∝ exp(-Eg / 2kT), so ni(eg) = ni_ref · exp((eg_ref − eg) / (2 V_T)).
    """
    from perovskite_sim.constants import K_B, Q

    V_T = K_B * 300.0 / Q
    return float(ni_ref * np.exp((eg_ref - eg) / (2.0 * V_T)))


@pytest.fixture(scope="module")
def eg_sweep_results(baseline_stack: DeviceStack) -> list[tuple[float, JVResult]]:
    """Run J-V at each absorber Eg and return (Eg, JVResult) pairs.

    Three linked parameters are adjusted per Eg so the Beer-Lambert model
    captures the dominant Eg-driven physics:

    1. ``Eg`` — sets the band offsets at heterointerfaces.
    2. ``ni`` — scaled as ni ∝ exp(−Eg/2kT) because at T = 300 K the solver
       uses the explicit ``ni`` field and does not derive it from ``Eg``.
    3. ``Phi`` — scaled to the AM1.5G above-gap integrated photon flux, so
       J_sc drops at wider bandgaps (fewer above-gap photons).
    """
    results: list[tuple[float, JVResult]] = []
    stacks = _vary_absorber_param(baseline_stack, "Eg", EG_SWEEP)
    for eg, stack in zip(EG_SWEEP, stacks):
        ni_new = _ni_for_eg(eg)
        stacks_ni = _vary_absorber_param(stack, "ni", [ni_new])
        stack_ni = stacks_ni[0]
        # Scale Phi to above-gap photon flux
        stack_phi = replace(stack_ni, Phi=_above_gap_flux(eg))
        result = _run_jv(stack_phi)
        results.append((eg, result))
    return results


def test_voc_vs_bandgap(eg_sweep_results: list[tuple[float, JVResult]]) -> None:
    """V_oc loss ΔV = Eg/q − V_oc should be roughly constant with bandgap.

    In a physically correct simulator V_oc tracks Eg with slope ≈ 1,
    so the non-radiative loss stays in a narrow band (0.25–0.55 V).
    A simulator where V_oc does not respond to Eg would show growing ΔV.
    """
    eg_values = np.array([eg for eg, _ in eg_sweep_results])
    voc_values = np.array([r.metrics_fwd.V_oc for _, r in eg_sweep_results])
    delta_V = eg_values - voc_values

    median_loss = float(np.median(delta_V))
    assert 0.25 <= median_loss <= 0.55, (
        f"Median V_oc loss {median_loss:.3f} V outside [0.25, 0.55] V — "
        f"V_oc values: {[f'{v:.4f}' for v in voc_values]}"
    )

    slope, _, _, _, _ = linregress(eg_values, delta_V)
    assert abs(slope) <= 0.15, (
        f"ΔV slope vs Eg is {slope:.3f} — should be near zero; "
        "V_oc is not tracking bandgap correctly"
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
