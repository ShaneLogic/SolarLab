"""Metastable configuration preparation, frozen measurement, and operator-split transients.

Implements the explicit two-stage SCAPS-like metastable defect workflow on
``CompiledMetastableDevice1D`` + ``solve_dynamic_defect_steady_1d``:

1. ``prepare_metastable_state``: clamped outer fixed-point ``y = Y(n(y), p(y))``
   at the declared preparation working point, followed by a mandatory unclamped
   refinement step so a clamped iterate can never masquerade as converged.
2. ``solve_frozen_metastable_steady`` / ``solve_frozen_metastable_jv``: bias
   measurement at ``measurement_temperature_K`` with frozen configuration
   fractions ``y`` strictly preserved.
3. ``solve_metastable_transient``: sequential operator-split integration
   advancing ``y(t)`` analytically via ``advance_metastable_configuration``,
   gating both exact two-charge configuration transfer bookkeeping and
   halved-``dt`` operator-splitting error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import frozen_array
from solarlab.device.compile_dynamic_defect_1d import compile_metastable_device_1d
from solarlab.device.defects import MetastablePreparationInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.dynamic_defects_1d import (
    DynamicDefectSteadyOptions,
    DynamicDefectSteadyResult,
    solve_dynamic_defect_steady_1d,
)
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.dynamic_defect_1d import (
    CompiledDynamicDefectDevice1D,
    CompiledMetastableDevice1D,
)
from solarlab.physics.defects import (
    MultivalentSpeciesSpec,
    advance_metastable_configuration,
    configuration_charge_transfer_C_m3,
    configuration_from_logit,
    configuration_logit,
    evaluate_metastable_configuration_closure,
    evaluate_multivalent_defect_closure,
)

Vector: TypeAlias = NDArray[np.float64]

METASTABLE_TRANSIENT_SCHEME = "solarlab.metastable-operator-split-transient.v1"

__all__ = [
    "METASTABLE_TRANSIENT_SCHEME",
    "FrozenMetastableJVPoint",
    "FrozenMetastableJVResult",
    "MetastableCapabilityError",
    "MetastablePreparationError",
    "MetastableTransientCertificate",
    "MetastableTransientError",
    "MetastableTransientResult",
    "PreparedMetastableState",
    "prepare_metastable_state",
    "solve_frozen_metastable_jv",
    "solve_frozen_metastable_steady",
    "solve_metastable_transient",
]


class MetastablePreparationError(RuntimeError):
    """Raised when metastable configuration preparation or frozen provenance fails."""


class MetastableTransientError(RuntimeError):
    """Raised when operator-split metastable transient integration fails."""


class MetastableCapabilityError(MetastableTransientError, CapabilityError):
    """Raised when a device violates metastable preparation/transient capabilities."""


def _array_digest(label: str, *arrays: ArrayLike) -> str:
    h = sha256(label.encode("ascii"))
    for arr in arrays:
        contig = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
        h.update(str(contig.shape).encode("ascii"))
        h.update(contig.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class PreparedMetastableState:
    """Immutable prepared metastable configuration split and provenance."""

    compiled_metastable: CompiledMetastableDevice1D
    frozen_measurement_device: CompiledDynamicDefectDevice1D
    donor_fractions: tuple[Vector, ...]
    active_nodes: NDArray[np.bool_]
    grid_m: Vector
    preparation_steady: DynamicDefectSteadyResult
    outer_iterations: int
    clamped_iterations: int
    final_relative_change: float
    unclamped_refinement_change: float
    preparation_voltage_V: float
    preparation_temperature_K: float
    preparation_illumination_suns: float
    measurement_temperature_K: float
    freeze_configuration_during_measurement: bool
    protocol_sha256: str
    state_sha256: str
    scientific_qualification: bool = False
    physical_certificate_status: str = "not_evaluated"

    def __post_init__(self) -> None:
        frozen_fracs = tuple(frozen_array(f) for f in self.donor_fractions)
        object.__setattr__(self, "donor_fractions", frozen_fracs)
        object.__setattr__(self, "grid_m", frozen_array(self.grid_m))
        act = np.array(self.active_nodes, dtype=bool, copy=True)
        act.setflags(write=False)
        object.__setattr__(self, "active_nodes", act)

    @property
    def donor_fraction(self) -> Vector:
        if len(self.donor_fractions) != 1:
            raise ValueError("donor_fraction convenience property requires a single metastable species")
        return self.donor_fractions[0]


@dataclass(frozen=True)
class FrozenMetastableJVPoint:
    """One bias point on a frozen metastable J-V sweep."""

    voltage_V: float
    current_density_A_m2: float
    steady: DynamicDefectSteadyResult


@dataclass(frozen=True)
class FrozenMetastableJVResult:
    """J-V sweep measured against a strictly frozen metastable configuration."""

    voltages_V: Vector
    current_densities_A_m2: Vector
    points: tuple[FrozenMetastableJVPoint, ...]
    donor_fractions: tuple[Vector, ...]
    measurement_temperature_K: float
    generation_scale: float
    preparation_state_sha256: str
    scientific_qualification: bool = False
    physical_certificate_status: str = "not_evaluated"

    def __post_init__(self) -> None:
        object.__setattr__(self, "voltages_V", frozen_array(self.voltages_V))
        object.__setattr__(
            self, "current_densities_A_m2", frozen_array(self.current_densities_A_m2)
        )
        object.__setattr__(
            self, "donor_fractions", tuple(frozen_array(f) for f in self.donor_fractions)
        )


@dataclass(frozen=True, slots=True)
class MetastableTransientCertificate:
    """Numerical splitting and charge-transfer audit for one metastable transient."""

    scheme: str
    passed: bool
    reasons: tuple[str, ...]
    step_count: int
    minimum_fraction: float
    maximum_fraction: float
    maximum_state_residual: float
    maximum_charge_transfer_relative_error: float
    maximum_refinement_fraction_change: float
    maximum_step_over_relaxation_time: float
    minimum_relaxation_time_s: float


@dataclass(frozen=True)
class MetastableTransientResult:
    """Operator-split dynamic metastable configuration trajectory."""

    times_s: Vector
    donor_fractions: tuple[Vector, ...]
    stationary_fractions: tuple[Vector, ...]
    relaxation_times_s: tuple[Vector, ...]
    configuration_charge_transfer_C_m2: Vector
    terminal_current_density_A_m2: Vector
    states: tuple[DynamicDefectSteadyResult, ...]
    certificate: MetastableTransientCertificate
    preparation_protocol_sha256: str
    preparation_state_sha256: str
    scientific_qualification: bool = False
    physical_certificate_status: str = "not_evaluated"

    def __post_init__(self) -> None:
        object.__setattr__(self, "times_s", frozen_array(self.times_s))
        object.__setattr__(
            self, "donor_fractions", tuple(frozen_array(f) for f in self.donor_fractions)
        )
        object.__setattr__(
            self,
            "stationary_fractions",
            tuple(frozen_array(f) for f in self.stationary_fractions),
        )
        object.__setattr__(
            self, "relaxation_times_s", tuple(frozen_array(f) for f in self.relaxation_times_s)
        )
        object.__setattr__(
            self,
            "configuration_charge_transfer_C_m2",
            frozen_array(self.configuration_charge_transfer_C_m2),
        )
        object.__setattr__(
            self,
            "terminal_current_density_A_m2",
            frozen_array(self.terminal_current_density_A_m2),
        )

    @property
    def donor_fraction(self) -> Vector:
        if len(self.donor_fractions) != 1:
            raise ValueError("donor_fraction convenience property requires a single metastable species")
        return self.donor_fractions[0]


def prepare_metastable_state(
    device_or_prepared: CompiledMetastableDevice1D | PreparedDevice,
    *,
    preparation: MetastablePreparationInput | None = None,
    intervals_per_layer: int | Sequence[int] = 24,
    steady_options: DynamicDefectSteadyOptions | None = None,
) -> PreparedMetastableState:
    """Solve clamped outer fixed point + mandatory unclamped refinement for ``y``."""
    if isinstance(device_or_prepared, CompiledMetastableDevice1D):
        meta_dev = device_or_prepared
    elif isinstance(device_or_prepared, PreparedDevice):
        try:
            meta_dev = compile_metastable_device_1d(
                device_or_prepared, intervals_per_layer=intervals_per_layer
            )
        except CapabilityError as error:
            raise MetastableCapabilityError(str(error)) from error
    else:
        raise TypeError(
            "device_or_prepared must be CompiledMetastableDevice1D or PreparedDevice"
        )

    proto = preparation if preparation is not None else meta_dev.preparation
    if proto is None:
        raise MetastablePreparationError(
            "metastable preparation requires a MetastablePreparationInput protocol"
        )

    if hasattr(proto, "model_dump"):
        proto_dict = proto.model_dump()
    elif hasattr(proto, "to_mapping"):
        proto_dict = proto.to_mapping()
    elif isinstance(proto, Mapping):
        proto_dict = dict(proto)
    else:
        raise TypeError("unsupported metastable preparation protocol type")
    numerics = proto_dict["numerics"]
    prep_v = float(proto_dict["preparation_voltage_V"])
    prep_t = float(proto_dict["preparation_temperature_K"])
    prep_suns = float(proto_dict["preparation_illumination_suns"])
    meas_t = float(proto_dict["measurement_temperature_K"])
    freeze_flag = bool(proto_dict["freeze_configuration_during_measurement"])

    init_guess = float(numerics["initial_donor_fraction_guess"])
    if not 0.0 < init_guess < 1.0:
        raise MetastablePreparationError(
            "initial_donor_fraction_guess must lie strictly inside (0, 1)"
        )
    clamp = float(numerics["clamping_factor"])
    tol = float(numerics["relative_tolerance"])
    max_iters = int(numerics["max_iterations"])

    m_active = meta_dev.active_node_count
    mask = meta_dev.active_nodes
    specs = meta_dev.metastable_specs

    fractions: list[Vector] = [np.full(m_active, init_guess, dtype=np.float64) for _ in specs]

    def evaluate_outer(
        current_fracs: Sequence[Vector],
    ) -> tuple[DynamicDefectSteadyResult, list[Vector]]:
        frozen_dev = meta_dev.compile_frozen_device(current_fracs, temperature_K=prep_t)
        assert frozen_dev.stationary_defects is not None
        prep_reg = frozen_dev.stationary_defects.metastable_regions[0]
        eg_prep = float(prep_reg.band_gap_eV[0])
        nc_prep = float(prep_reg.effective_conduction_dos_m3[0])
        nv_prep = float(prep_reg.effective_valence_dos_m3[0])
        try:
            steady = solve_dynamic_defect_steady_1d(
                frozen_dev,
                voltage_V=prep_v,
                generation_scale=prep_suns,
                options=steady_options,
            )
        except RuntimeError as error:
            raise MetastablePreparationError(
                f"metastable preparation electronic solve failed: {error}"
            ) from error
        n_act = steady.state.electronic_state.electron_density_m3[mask]
        p_act = steady.state.electronic_state.hole_density_m3[mask]
        targets: list[Vector] = []
        for sp in specs:
            cl = evaluate_metastable_configuration_closure(
                n_act,
                p_act,
                sp,
                band_gap_eV=eg_prep,
                effective_conduction_dos_m3=nc_prep,
                effective_valence_dos_m3=nv_prep,
                temperature_K=prep_t,
            )
            targets.append(np.array(cl.donor_fraction, dtype=np.float64, copy=True))
        return steady, targets

    clamped_iters = 0
    rel_change = math.inf
    steady_curr: DynamicDefectSteadyResult | None = None

    for outer_iter in range(1, max_iters + 1):
        steady_curr, targets = evaluate_outer(fractions)
        rel_change = max(
            float(np.max(np.abs(tgt - cur))) / max(float(np.max(np.abs(cur))), 1.0e-12)
            for cur, tgt in zip(fractions, targets)
        )
        if clamp < 1.0:
            clamped_iters += 1
        fractions = [cur + clamp * (tgt - cur) for cur, tgt in zip(fractions, targets)]
        if any(not np.all(np.isfinite(f)) for f in fractions):
            raise MetastablePreparationError(
                "metastable preparation produced non-finite donor fraction"
            )
        if rel_change <= tol:
            break
    else:
        raise MetastablePreparationError(
            f"metastable preparation did not converge within {max_iters} iterations "
            f"(last change {rel_change:.4e} > {tol:.4e})"
        )

    # Mandatory unclamped refinement step:
    _, unclamped_targets = evaluate_outer(fractions)
    unclamped_change = max(
        float(np.max(np.abs(tgt - cur))) / max(float(np.max(np.abs(cur))), 1.0e-12)
        for cur, tgt in zip(fractions, unclamped_targets)
    )
    fractions = unclamped_targets
    if unclamped_change > tol:
        raise MetastablePreparationError(
            f"final unclamped metastable refinement exceeded tolerance: "
            f"{unclamped_change:.4e} > {tol:.4e}"
        )

    final_steady, _ = evaluate_outer(fractions)
    meas_dev = meta_dev.compile_frozen_device(fractions, temperature_K=meas_t)

    proto_sha = sha256(
        json.dumps(proto_dict, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    state_sha = _array_digest(
        "solarlab-metastable-prepared-state-v1",
        meta_dev.base_electronic_device.nodes_m,
        *fractions,
        final_steady.state.u,
        np.array([prep_v, prep_t, prep_suns, meas_t]),
    )
    return PreparedMetastableState(
        compiled_metastable=meta_dev,
        frozen_measurement_device=meas_dev,
        donor_fractions=tuple(fractions),
        active_nodes=mask,
        grid_m=meta_dev.base_electronic_device.nodes_m,
        preparation_steady=final_steady,
        outer_iterations=outer_iter,
        clamped_iterations=clamped_iters,
        final_relative_change=rel_change,
        unclamped_refinement_change=unclamped_change,
        preparation_voltage_V=prep_v,
        preparation_temperature_K=prep_t,
        preparation_illumination_suns=prep_suns,
        measurement_temperature_K=meas_t,
        freeze_configuration_during_measurement=freeze_flag,
        protocol_sha256=proto_sha,
        state_sha256=state_sha,
    )


def solve_frozen_metastable_steady(
    prepared_state: PreparedMetastableState,
    *,
    voltage_V: float = 0.0,
    generation_scale: float = 0.0,
    steady_options: DynamicDefectSteadyOptions | None = None,
) -> DynamicDefectSteadyResult:
    """Solve one steady working point against a strictly frozen configuration."""
    if not isinstance(prepared_state, PreparedMetastableState):
        raise TypeError("prepared_state must be PreparedMetastableState")
    if not prepared_state.freeze_configuration_during_measurement:
        raise MetastablePreparationError(
            "measurement requires freeze_configuration_during_measurement=True"
        )
    try:
        return solve_dynamic_defect_steady_1d(
            prepared_state.frozen_measurement_device,
            voltage_V=voltage_V,
            generation_scale=generation_scale,
            options=steady_options,
        )
    except RuntimeError as error:
        raise MetastablePreparationError(
            f"frozen metastable steady solve failed: {error}"
        ) from error


def solve_frozen_metastable_jv(
    prepared_state: PreparedMetastableState,
    voltages_V: ArrayLike,
    *,
    generation_scale: float = 0.0,
    steady_options: DynamicDefectSteadyOptions | None = None,
) -> FrozenMetastableJVResult:
    """Sweep J-V at ``measurement_temperature_K`` without updating ``donor_fractions``."""
    if not isinstance(prepared_state, PreparedMetastableState):
        raise TypeError("prepared_state must be PreparedMetastableState")
    if not prepared_state.freeze_configuration_during_measurement:
        raise MetastablePreparationError(
            "J-V measurement requires freeze_configuration_during_measurement=True"
        )
    v_arr = np.asarray(voltages_V, dtype=np.float64)
    if v_arr.ndim != 1 or v_arr.size < 1 or not np.all(np.isfinite(v_arr)):
        raise ValueError("voltages_V must be a non-empty 1D array of finite biases")

    f_dev = prepared_state.frozen_measurement_device
    points: list[FrozenMetastableJVPoint] = []
    j_arr = np.empty(v_arr.size, dtype=np.float64)
    prev_u: Vector | None = None
    prev_v = 0.0

    for idx, v_val in enumerate(v_arr):
        vf = float(v_val)
        seed_u: Vector | None = None
        if prev_u is not None and abs(vf - prev_v) <= 0.15:
            seed_u = f_dev.rebase_coordinates(prev_u, prev_v, vf)
            N = f_dev.node_count
            phi_l, phi_r = f_dev.electronic_component.contact_potentials(vf)
            dv = (phi_r - phi_l) - (seed_u[N - 1] - seed_u[0])
            nodes = f_dev.nodes_m
            seed_u[:N] += dv * (nodes - nodes[0]) / (nodes[-1] - nodes[0])
        try:
            st = solve_dynamic_defect_steady_1d(
                f_dev,
                voltage_V=vf,
                generation_scale=generation_scale,
                initial_u=seed_u,
                options=steady_options,
            )
        except RuntimeError:
            st = solve_dynamic_defect_steady_1d(
                f_dev,
                voltage_V=vf,
                generation_scale=generation_scale,
                options=steady_options,
            )
        points.append(
            FrozenMetastableJVPoint(
                voltage_V=vf,
                current_density_A_m2=st.terminal_current_density_A_m2,
                steady=st,
            )
        )
        j_arr[idx] = st.terminal_current_density_A_m2
        prev_u = st.state.u
        prev_v = vf

    return FrozenMetastableJVResult(
        voltages_V=v_arr,
        current_densities_A_m2=j_arr,
        points=tuple(points),
        donor_fractions=prepared_state.donor_fractions,
        measurement_temperature_K=prepared_state.measurement_temperature_K,
        generation_scale=float(generation_scale),
        preparation_state_sha256=prepared_state.state_sha256,
    )


def _run_operator_split_trace(
    prepared_state: PreparedMetastableState,
    times: Vector,
    *,
    voltage_V: float,
    generation_scale: float,
    keep_states: bool,
    steady_options: DynamicDefectSteadyOptions | None,
) -> tuple[
    tuple[Vector, ...],
    tuple[Vector, ...],
    tuple[Vector, ...],
    list[float],
    list[DynamicDefectSteadyResult],
    list[float],
    Vector,
    Vector,
]:
    meta_dev = prepared_state.compiled_metastable
    specs = meta_dev.metastable_specs
    mask = prepared_state.active_nodes
    grid = prepared_state.grid_m
    meas_t = prepared_state.measurement_temperature_K
    assert prepared_state.frozen_measurement_device.stationary_defects is not None
    meas_reg = prepared_state.frozen_measurement_device.stationary_defects.metastable_regions[0]
    eg = float(meas_reg.band_gap_eV[0])
    nc = float(meas_reg.effective_conduction_dos_m3[0])
    nv = float(meas_reg.effective_valence_dos_m3[0])

    # Round-trip through logit to reject boundary-saturated prepared fractions:
    curr_fracs = [
        configuration_from_logit(configuration_logit(np.asarray(f, dtype=np.float64)))
        for f in prepared_state.donor_fractions
    ]

    T = times.size
    m_act = meta_dev.active_node_count
    frac_hist = [np.zeros((T, m_act), dtype=np.float64) for _ in specs]
    stat_hist = [np.zeros((T - 1, m_act), dtype=np.float64) for _ in specs]
    tau_hist = [np.zeros((T - 1, m_act), dtype=np.float64) for _ in specs]
    for s_idx, cf in enumerate(curr_fracs):
        frac_hist[s_idx][0] = cf

    charge_errors: list[float] = []
    states: list[DynamicDefectSteadyResult] = []
    step_ratios: list[float] = []
    areal_transfers = np.zeros(T - 1, dtype=np.float64)
    j_trace = np.zeros(T, dtype=np.float64)

    for k in range(T):
        dev_k = meta_dev.compile_frozen_device(curr_fracs, temperature_K=meas_t)
        try:
            steady_k = solve_dynamic_defect_steady_1d(
                dev_k,
                voltage_V=voltage_V,
                generation_scale=generation_scale,
                options=steady_options,
            )
        except RuntimeError as error:
            raise MetastableTransientError(
                f"metastable transient step {k} failed electronic solve: {error}"
            ) from error

        j_trace[k] = steady_k.terminal_current_density_A_m2
        if keep_states:
            states.append(steady_k)
        if k == T - 1:
            break

        dt = float(times[k + 1] - times[k])
        n_full = steady_k.state.electronic_state.electron_density_m3
        p_full = steady_k.state.electronic_state.hole_density_m3
        n_act = n_full[mask]
        p_act = p_full[mask]

        next_fracs: list[Vector] = []
        pred_delta_rho: Vector = np.zeros(m_act, dtype=np.float64)
        for s_idx, (sp, cf) in enumerate(zip(specs, curr_fracs)):
            cl = evaluate_metastable_configuration_closure(
                n_act,
                p_act,
                sp,
                band_gap_eV=eg,
                effective_conduction_dos_m3=nc,
                effective_valence_dos_m3=nv,
                temperature_K=meas_t,
            )
            adv = advance_metastable_configuration(cf, cl, dt)
            donor_cl = evaluate_multivalent_defect_closure(
                n_act,
                p_act,
                MultivalentSpeciesSpec(f"{sp.identifier}:donor", 1.0, sp.donor_configuration),
                band_gap_eV=eg,
                effective_conduction_dos_m3=nc,
                effective_valence_dos_m3=nv,
                temperature_K=meas_t,
            )
            acc_cl = evaluate_multivalent_defect_closure(
                n_act,
                p_act,
                MultivalentSpeciesSpec(
                    f"{sp.identifier}:acceptor", 1.0, sp.acceptor_configuration
                ),
                band_gap_eV=eg,
                effective_conduction_dos_m3=nc,
                effective_valence_dos_m3=nv,
                temperature_K=meas_t,
            )
            next_fracs.append(np.array(adv.next_donor_fraction, dtype=np.float64, copy=True))
            frac_hist[s_idx][k + 1] = adv.next_donor_fraction
            stat_hist[s_idx][k] = adv.stationary_donor_fraction
            tau_hist[s_idx][k] = 1.0 / adv.total_conversion_rate_s1
            pred_delta_rho += configuration_charge_transfer_C_m3(
                sp.total_density_m3,
                donor_cl.mean_charge_e,
                acc_cl.mean_charge_e,
                adv.delta_donor_fraction,
            )
            step_ratios.append(float(np.max(dt * adv.total_conversion_rate_s1)))

        assert dev_k.stationary_defects is not None
        dev_next = meta_dev.compile_frozen_device(next_fracs, temperature_K=meas_t)
        assert dev_next.stationary_defects is not None
        rho_before = dev_k.stationary_defects.evaluate(n_full, p_full).charge_density_C_m3[mask]
        rho_after = dev_next.stationary_defects.evaluate(n_full, p_full).charge_density_C_m3[mask]
        measured_delta = rho_after - rho_before
        scale = max(float(np.max(np.abs(rho_before))), np.finfo(float).tiny)
        charge_errors.append(float(np.max(np.abs(measured_delta - pred_delta_rho)) / scale))

        areal_vec = np.zeros(grid.size, dtype=np.float64)
        areal_vec[mask] = pred_delta_rho
        areal_transfers[k] = float(np.trapezoid(areal_vec, grid))
        curr_fracs = next_fracs

    return (
        tuple(frac_hist),
        tuple(stat_hist),
        tuple(tau_hist),
        charge_errors,
        states,
        step_ratios,
        areal_transfers,
        j_trace,
    )


def solve_metastable_transient(
    prepared_state: PreparedMetastableState,
    times_s: ArrayLike,
    *,
    voltage_V: float = 0.0,
    generation_scale: float = 0.0,
    maximum_charge_transfer_relative_error: float = 1.0e-10,
    maximum_refinement_fraction_change: float = 5.0e-3,
    require_certificate: bool = True,
    steady_options: DynamicDefectSteadyOptions | None = None,
) -> MetastableTransientResult:
    """Integrate dynamic metastable configuration evolution via operator splitting + dt/2 audit."""
    if not isinstance(prepared_state, PreparedMetastableState):
        raise TypeError("prepared_state must be PreparedMetastableState")
    times = np.asarray(times_s, dtype=np.float64)
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or times[0] != 0.0
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("times_s must start at 0.0 and strictly increase")
    if not math.isfinite(voltage_V) or not math.isfinite(generation_scale) or generation_scale < 0.0:
        raise ValueError("voltage_V must be finite and generation_scale >= 0")
    if maximum_charge_transfer_relative_error <= 0.0 or maximum_refinement_fraction_change <= 0.0:
        raise ValueError("certificate tolerances must be positive")

    (
        fracs,
        stats,
        taus,
        charge_errs,
        states,
        ratios,
        areal_transfers,
        j_trace,
    ) = _run_operator_split_trace(
        prepared_state,
        times,
        voltage_V=voltage_V,
        generation_scale=generation_scale,
        keep_states=True,
        steady_options=steady_options,
    )

    refined_times = np.unique(np.concatenate([times, 0.5 * (times[:-1] + times[1:])]))
    refined_fracs, *_ = _run_operator_split_trace(
        prepared_state,
        refined_times,
        voltage_V=voltage_V,
        generation_scale=generation_scale,
        keep_states=False,
        steady_options=steady_options,
    )
    sampled = np.searchsorted(refined_times, times)
    refinement_change = max(
        float(np.max(np.abs(rf[sampled] - cf)))
        for rf, cf in zip(refined_fracs, fracs)
    )

    min_frac = min(float(np.min(cf)) for cf in fracs)
    max_frac = max(float(np.max(cf)) for cf in fracs)
    max_q_err = max(charge_errs) if charge_errs else 0.0
    max_res = max(st.residual_inf_norm for st in states)
    min_tau = min(float(np.min(tv)) for tv in taus) if taus else math.inf

    reasons: list[str] = []
    if min_frac <= 0.0 or max_frac >= 1.0:
        reasons.append("configuration_fraction_left_open_interval")
    if max_q_err > maximum_charge_transfer_relative_error:
        reasons.append("configuration_charge_transfer_mismatch")
    if refinement_change > maximum_refinement_fraction_change:
        reasons.append("operator_splitting_not_resolved")
    if not all(st.converged for st in states):
        reasons.append("unconverged_carrier_state")

    cert = MetastableTransientCertificate(
        scheme=METASTABLE_TRANSIENT_SCHEME,
        passed=not reasons,
        reasons=tuple(reasons),
        step_count=int(times.size),
        minimum_fraction=min_frac,
        maximum_fraction=max_frac,
        maximum_state_residual=max_res,
        maximum_charge_transfer_relative_error=max_q_err,
        maximum_refinement_fraction_change=refinement_change,
        maximum_step_over_relaxation_time=max(ratios) if ratios else 0.0,
        minimum_relaxation_time_s=min_tau,
    )
    if require_certificate and not cert.passed:
        raise MetastableTransientError(
            "metastable transient failed numerical audit: " + ", ".join(cert.reasons)
        )

    return MetastableTransientResult(
        times_s=times,
        donor_fractions=fracs,
        stationary_fractions=stats,
        relaxation_times_s=taus,
        configuration_charge_transfer_C_m2=areal_transfers,
        terminal_current_density_A_m2=j_trace,
        states=tuple(states),
        certificate=cert,
        preparation_protocol_sha256=prepared_state.protocol_sha256,
        preparation_state_sha256=prepared_state.state_sha256,
    )
