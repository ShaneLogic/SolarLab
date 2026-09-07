"""Explicit continuous J-V histories using the existing density-state engine.

This research driver does not claim the standard staircase driver's local
branch certificate. Physical history, numerical settings and inventory checks
are returned separately; no physical device parameters are overridden.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from collections.abc import Mapping
from typing import Any

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.experiments import jv_sweep as jv
from perovskite_sim.experiments.waveform_jacobian import (
    WaveformJacobianCapabilityError, build_waveform_density_jacobian,
)
from perovskite_sim.experiments.protocol import (
    DCSettleCriterion, ExperimentProtocol, IlluminationStep, SamplingProtocol,
    ScanProtocol, _finite_float, _require_exact_keys, resolve_experiment_protocol,
)
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.physics.generation import dual_cell_faces, dual_cell_widths


# Calado zero-source ramps hit arithmetic floors with the old 1 m^-3 default.
# This research-driver value is explicit in results; core ODE defaults stay unchanged.
DEFAULT_DENSITY_ATOL_M3 = 100.0


@dataclass(frozen=True, slots=True)
class JVWaveform:
    start_voltage_V: float = -1.0
    dark_seed_s: float = 120.0
    dark_prep_s: float = 30.0
    branch_dwell_s: float = 0.5
    turnaround_s: float = 0.0
    turnaround_dark: bool = True
    uniform_generation_rate_m3_s: float | None = None
    schema_version: int = 1

    def __post_init__(self):
        for name in ("start_voltage_V", "dark_seed_s", "dark_prep_s", "branch_dwell_s", "turnaround_s"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if self.start_voltage_V > 0:
            raise ValueError("start_voltage_V must be <= 0 for Jsc extraction")
        if min(self.dark_seed_s, self.dark_prep_s, self.turnaround_s) < 0 or self.branch_dwell_s <= 0:
            raise ValueError("history times must be nonnegative and branch_dwell_s positive")
        if not isinstance(self.turnaround_dark, bool):
            raise TypeError("turnaround_dark must be boolean")
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("unsupported J-V waveform schema_version")
        if self.uniform_generation_rate_m3_s is not None:
            value = _finite_float(self.uniform_generation_rate_m3_s, "uniform_generation_rate_m3_s")
            if value < 0:
                raise ValueError("uniform_generation_rate_m3_s must be nonnegative")
            object.__setattr__(self, "uniform_generation_rate_m3_s", value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JVWaveform:
        if not isinstance(value, Mapping):
            raise TypeError("waveform must be an object")
        _require_exact_keys(value, {field.name for field in fields(cls)}, "JVWaveform")
        return cls(**value)


@dataclass(frozen=True, kw_only=True)
class WaveformJVResult(jv.JVResult):
    waveform: JVWaveform
    hysteresis_index_paper: float | None
    numerical_scope: str = "finite_time_diagnostic"
    numerical_controls: dict[str, float]
    inventory_relative_drift: tuple[float, ...]
    generation_budget_A_m2: float
    waveform_protocol_sha256: str
    jacobian_evaluator: str = "finite_difference"
    jacobian_fallback_reason: str | None = None


def voltage_samples(waveform: JVWaveform, n_points: int, v_rate: float, V_max: float):
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)) or n_points < 2:
        raise ValueError("n_points must be an integer >= 2")
    if _finite_float(v_rate, "v_rate") <= 0:
        raise ValueError("v_rate must be positive")
    if _finite_float(V_max, "V_max") <= 0:
        raise ValueError("V_max must be positive")
    return np.linspace(waveform.start_voltage_V, V_max, n_points)


def build_waveform_protocol(stack, waveform, *, n_points, v_rate, V_max, illuminated=True):
    if not isinstance(illuminated, bool):
        raise TypeError("illuminated must be boolean for a continuous waveform")
    forward = voltage_samples(waveform, n_points, v_rate, V_max)
    scan_s = (V_max - waveform.start_voltage_V) / v_rate
    source = (
        "device_optics" if waveform.uniform_generation_rate_m3_s is None
        else f"uniform_absorber_G_m3_s:{waveform.uniform_generation_rate_m3_s:.17g}"
    )
    light = "baseline" if illuminated else "dark"
    phases = (
        IlluminationStep("dark_seed_at_0V", "dark", waveform.dark_seed_s),
        IlluminationStep("dark_prebias_at_scan_start", "dark", waveform.dark_prep_s),
        IlluminationStep("forward_start_dwell", light, waveform.branch_dwell_s, source_reference=source),
        IlluminationStep("forward_continuous_ramp", light, scan_s, source_reference=source),
        IlluminationStep("turnaround_at_scan_stop", "dark" if waveform.turnaround_dark else light,
                         waveform.turnaround_s, source_reference=source),
        IlluminationStep("reverse_start_dwell", light, waveform.branch_dwell_s, source_reference=source),
        IlluminationStep("reverse_continuous_ramp", light, scan_s, source_reference=source),
    )
    return ExperimentProtocol(
        experiment="jv_hysteresis", initial_state_source="finite_time_dc_preconditioned",
        pre_bias_V=waveform.start_voltage_V, soak_duration_s=waveform.dark_prep_s,
        dwell_duration_s=waveform.branch_dwell_s, illumination_history=phases,
        temperature_K=stack.T,
        scan=ScanProtocol("voltage_V", "ascending_then_descending", waveform.start_voltage_V, V_max, v_rate),
        ac_excitation=None,
        dc_settle=(DCSettleCriterion("finite_time", waveform.dark_prep_s)
                   if waveform.dark_prep_s else DCSettleCriterion("not_applicable")),
        sampling=SamplingProtocol("voltage_V", "piecewise_linear", tuple(np.r_[forward, forward[::-1]])),
    )


def uniform_absorber_generation(x, stack, rate):
    """Exact integral of a piecewise-constant source on the existing dual mesh."""
    faces, widths = dual_cell_faces(x), dual_cell_widths(x)
    generation = np.zeros_like(x)
    left = float(x[0])
    found = False
    for layer in electrical_layers(stack):
        right = left + layer.thickness
        if layer.role == "absorber":
            overlap = np.maximum(0.0, np.minimum(faces[1:], right) - np.maximum(faces[:-1], left))
            generation += rate * overlap / widths
            found = True
        left = right
    if not found:
        raise ValueError("uniform generation requires an absorber layer")
    return generation


def run_waveform_jv(
    stack: DeviceStack, waveform: JVWaveform, *, N_grid=60, n_points=111,
    v_rate=0.04, V_max=1.2, illuminated=True, rtol=1e-4, atol=DEFAULT_DENSITY_ATOL_M3,
    save_snapshots=False, decompose_currents=False, progress=None,
    experiment_protocol=None, protocol_mode="compatibility",
) -> WaveformJVResult:
    jv.require_jv_driver_capability(stack, requested_driver="transient")
    if stack.interface_charge_closure != "off":
        raise ValueError("charged interface closure requires its dedicated driver")
    if not isinstance(waveform, JVWaveform):
        raise TypeError("waveform must be a JVWaveform")
    if not isinstance(illuminated, bool):
        raise TypeError("illuminated must be boolean")
    for name, value in (("rtol", rtol), ("atol", atol)):
        if _finite_float(value, name) <= 0:
            raise ValueError(f"{name} must be positive")
    forward_v = voltage_samples(waveform, n_points, v_rate, V_max)
    expected = build_waveform_protocol(stack, waveform, n_points=n_points, v_rate=v_rate,
                                       V_max=V_max, illuminated=illuminated)
    # The complete waveform object is already an explicit history declaration.
    protocol = resolve_experiment_protocol(experiment_protocol or expected,
        replace(expected, implicit_legacy_protocol=True), mode=protocol_mode)
    x = jv.build_electrical_grid(stack, N_grid)
    jv.require_thick_layer_interface_resolution(x, stack, N_grid=N_grid, allow_underresolved_grid=False)
    mat = jv.build_material_arrays(x, stack)
    if waveform.uniform_generation_rate_m3_s is not None:
        mat = replace(mat, G_optical=uniform_absorber_generation(x, stack, waveform.uniform_generation_rate_m3_s))
    y = jv.solve_equilibrium(x, stack)
    active_bias = 0.0

    def jacobian_bias(time):
        return active_bias(time) if callable(active_bias) else active_bias

    jacobian_fallback_reason = None
    try:
        jacobian = build_waveform_density_jacobian(x, stack, mat, jacobian_bias)
        jacobian_evaluator = "analytic_single_ion_density"
    except WaveformJacobianCapabilityError as exc:
        jacobian = None
        jacobian_evaluator = "finite_difference"
        jacobian_fallback_reason = str(exc)
    jacobian_kwargs = {"jacobian": jacobian} if jacobian is not None else {}
    widths = dual_cell_widths(x)
    initial = jv.StateVec.unpack(y, len(x), mat.N_iface_state)
    inventories = tuple(float(block @ widths) for block in (initial.P, initial.P_neg) if block is not None)
    mass_error = np.zeros(len(inventories))

    def advance(left, right, duration, light):
        nonlocal y, active_bias
        if duration == 0:
            return
        active_bias = left if left == right else lambda t: left + (right-left)*t/duration
        if left == right:
            y = jv._integrate_step(x, y, stack, mat, left, 0.0, duration, rtol, atol,
                                  illuminated=light, **jacobian_kwargs)
        else:
            solved = jv.run_transient(
                x=x, y0=y, stack=stack, t_span=(0.0, duration), t_eval=np.array([duration]),
                V_app=active_bias,
                mat=mat, illuminated=light, rtol=rtol, atol=atol,
                max_step=duration / 20.0, max_nfev=100_000,
                **jacobian_kwargs,
            )
            if not solved.success:
                raise RuntimeError(f"continuous voltage segment failed: {solved.message}")
            y = solved.y[:, -1]
        if not np.all(np.isfinite(y)):
            raise RuntimeError("nonfinite state in waveform history")
        state = jv.StateVec.unpack(y, len(x), mat.N_iface_state)
        for index, block in enumerate(block for block in (state.P, state.P_neg) if block is not None):
            drift = abs(float(block @ widths) - inventories[index]) / max(abs(inventories[index]), 1.0)
            mass_error[index] = max(mass_error[index], drift)
            if drift > 1e-6:
                raise RuntimeError(f"ion inventory changed by {drift:.3g} relative")

    def report(stage, index, total, message):
        if progress is not None:
            progress(stage, index, total, message)

    report("preconditioning", 0, 2, "Dark 0 V history")
    advance(0.0, 0.0, waveform.dark_seed_s, False)
    report("preconditioning", 1, 2, "Dark prebias history")
    advance(waveform.start_voltage_V, waveform.start_voltage_V, waveform.dark_prep_s, False)

    def sweep(voltage, name):
        current = np.empty_like(voltage)
        snapshots, decomposed = [], []
        previous_v = float(voltage[0])
        for index, value in enumerate(voltage):
            old = y.copy()
            duration = waveform.branch_dwell_s if index == 0 else abs(float(value) - previous_v) / v_rate
            advance(previous_v, float(value), duration, illuminated)
            components = jv.compute_current_components(x, y, stack, float(value),
                y_prev=old, dt=duration, mat=mat, V_app_prev=previous_v)
            current[index] = components.J_total[0]
            if not np.isfinite(current[index]):
                raise RuntimeError(f"nonfinite {name} current at {value} V")
            if decompose_currents:
                decomposed.append([getattr(components, key)[0] for key in ("J_n", "J_p", "J_ion", "J_disp", "J_total")])
            if save_snapshots:
                snapshots.append(jv.extract_spatial_snapshot(x, y, stack, float(value), mat=mat))
            report(name, index + 1, len(voltage), f"V={value:.4g} V")
            previous_v = float(value)
        decomposition = jv.JVCurrentDecomp(*np.asarray(decomposed).T) if decomposed else None
        return current, tuple(snapshots) if save_snapshots else None, decomposition

    fwd_j, fwd_snapshots, fwd_decomp = sweep(forward_v, "jv_forward")
    report("turnaround", 0, 1, "High-bias dwell")
    advance(V_max, V_max, waveform.turnaround_s, illuminated and not waveform.turnaround_dark)
    reverse_v = forward_v[::-1].copy()
    rev_j, rev_snapshots, rev_decomp = sweep(reverse_v, "jv_reverse")
    ceiling = jv.thermodynamic_voc_ceiling(stack)
    fwd = jv.compute_metrics(forward_v, fwd_j, V_oc_max=ceiling)
    rev = jv.compute_metrics(reverse_v[::-1], rev_j[::-1], V_oc_max=ceiling)
    if mat.G_optical is None:
        from perovskite_sim.physics.generation import beer_lambert_generation
        generation = beer_lambert_generation(x, mat.alpha, stack.Phi)
    else:
        generation = mat.G_optical
    return WaveformJVResult(
        V_fwd=forward_v, J_fwd=fwd_j, V_rev=reverse_v, J_rev=rev_j,
        metrics_fwd=fwd, metrics_rev=rev,
        hysteresis_index=(rev.PCE - fwd.PCE) / rev.PCE if rev.PCE else 0.0,
        hysteresis_index_paper=rev.PCE / fwd.PCE - 1.0 if fwd.PCE else None,
        snapshots_fwd=fwd_snapshots, snapshots_rev=rev_snapshots,
        decomp_fwd=fwd_decomp, decomp_rev=rev_decomp, protocol=protocol, waveform=waveform,
        numerical_controls={"rtol": rtol, "atol_m3": atol, "requested_N_grid": N_grid, "actual_N_grid": len(x)},
        inventory_relative_drift=tuple(float(value) for value in mass_error),
        generation_budget_A_m2=float(Q * (generation @ widths)) if illuminated else 0.0,
        waveform_protocol_sha256=protocol.protocol_hash,
        jacobian_evaluator=jacobian_evaluator,
        jacobian_fallback_reason=jacobian_fallback_reason,
    )
