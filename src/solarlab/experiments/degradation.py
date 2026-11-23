"""Coupled ionic-stress degradation trajectories and frozen-ion J-V snapshots.

Contract (P05-04 clause 3):
1. Between snapshot times ``t_eval_s``, the live ``CompiledIonicDevice1D``
   evolves under constant electrical bias ``V_bias_V`` and illumination
   ``generation_scale`` via conservative Backward Euler steps of size
   ``<= dt_max_s``.
2. At each snapshot ``t_k``:
   - ``freeze_prepared_ions`` constructs an immobile-ion projection
     (``D_ion = 0``, ``D_ion_neg = 0``) that preserves ``P0``, ``P0_neg``,
     ``P_lim``, ``P_lim_neg``, and ``ion_site_contract`` bit-for-bit.
   - ``measure_frozen_ion_snapshot`` solves the stationary ``3 * N`` electronic
     system across ``probe_voltages_V`` against the exact frozen spatial ion
     coordinate slice ``u_ion = state.u[3*N:]``, verifying 0-ULP ion-profile
     invariance, identically zero ionic flux ``J_ion == 0``, and spatial current
     continuity on every probe bias.
3. Material parameters (absorber ``tau_n``, ``tau_p``) are cold-rebound ONLY
   when irreversible ``damage`` actually changes; zero-damage trajectories
   perform zero redundant material rebinds (``material_rebind_count == 0``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, cast

import numpy as np
from scipy.sparse import csc_matrix

from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.equations.implementation import use_assembled
from solarlab.equations.ionic_1d import (
    CompiledIonicDevice1D,
    Ionic1DProblem,
    IonicState1D,
)
from solarlab.experiments.electrothermal import has_declared_ions
from solarlab.experiments.jv.steady import SteadyJVMetrics, steady_jv_metrics
from solarlab.solvers.nonlinear.ionic_1d import (
    IonicSteadyOptions,
    solve_ionic_steady_state,
)
from solarlab.solvers.nonlinear.newton import NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import (
    CallableProblem,
    Matrix,
    ResidualScaling,
    SolveBudget,
    StepControl,
)
from solarlab.solvers.nonlinear.production import (
    HISTORICAL_MAX_ROW_V1,
    POISSON_CONTACT_BALANCE_V1,
    production_convergence_norm,
    production_coordinates,
    production_residual_scaling,
)
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
from solarlab.solvers.time.assembled_ionic_backward_euler import (
    AssembledIonicBackwardEuler1DStep,
)
from solarlab.solvers.time.backward_euler import BackwardEuler1DStep
from solarlab.solvers.time.carrier_seed import transient_density_basin_seed
from solarlab.solvers.time.ionic_backward_euler import (
    IonicBackwardEuler1DStep,
    step_accounting as ionic_step_accounting,
)

__all__ = [
    "DegradationDamageProtocol",
    "DegradationError",
    "DegradationProtocol",
    "DegradationResult",
    "FrozenIonSnapshot",
    "apply_absorber_damage",
    "freeze_prepared_ions",
    "measure_frozen_ion_snapshot",
    "run_degradation",
]


class DegradationError(RuntimeError):
    """Failed ionic stress step or frozen-ion snapshot certification."""


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(
        value, bool
    )


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DegradationDamageProtocol:
    """Optional irreversible ion-motion / pileup lifetime degradation law."""

    motion_gain_m2_C: float = 0.0
    segregation_gain_s_1: float = 0.0
    lifetime_degradation_alpha: float = 1.0
    damage_rebind_threshold: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "motion_gain_m2_C",
            "segregation_gain_s_1",
            "lifetime_degradation_alpha",
            "damage_rebind_threshold",
        ):
            raw = getattr(self, name)
            if not _is_real_number(raw) or not math.isfinite(float(raw)) or float(raw) < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, float(raw))


@dataclass(frozen=True)
class DegradationProtocol:
    """Explicit constant-bias stress and frozen-ion J-V snapshot controls."""

    V_bias_V: float = 0.9
    generation_scale: float = 1.0
    probe_generation_scale: float = 1.0
    precondition_time_s: float = 1e-3
    dt_max_s: float = 1.0
    settle_time_s: float = 0.0
    settle_steps: int = 4
    n_jv_points: int = 31
    V_max_V: float = 1.3
    incident_power_W_m2: float = 1000.0
    intervals_per_layer: int = 24
    ion_domain_layer: str | None = None
    ion_domain_endpoints: tuple[str, str] | None = None
    damage_protocol: DegradationDamageProtocol = DegradationDamageProtocol()
    poisson_tolerance: float = 1e-10
    continuity_tolerance_A_m2: float = 1e-7
    contact_log_tolerance: float = 1e-12
    max_current_error_A_m2: float = 1e-6
    ion_balance_tolerance_m2_s: float = 1e10

    def __post_init__(self) -> None:
        if not _is_real_number(self.V_bias_V) or not math.isfinite(float(self.V_bias_V)):
            raise ValueError("V_bias_V must be a finite real scalar")
        object.__setattr__(self, "V_bias_V", float(self.V_bias_V))

        for name in (
            "generation_scale",
            "probe_generation_scale",
            "precondition_time_s",
            "settle_time_s",
        ):
            raw = getattr(self, name)
            if not _is_real_number(raw) or not math.isfinite(float(raw)) or float(raw) < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, float(raw))

        for name in (
            "dt_max_s",
            "V_max_V",
            "incident_power_W_m2",
            "poisson_tolerance",
            "continuity_tolerance_A_m2",
            "contact_log_tolerance",
            "max_current_error_A_m2",
            "ion_balance_tolerance_m2_s",
        ):
            raw = getattr(self, name)
            if not _is_real_number(raw) or not math.isfinite(float(raw)) or float(raw) <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
            object.__setattr__(self, name, float(raw))

        if type(self.settle_steps) is not int or self.settle_steps < 1:
            raise ValueError("settle_steps must be a positive integer")
        if type(self.n_jv_points) is not int or self.n_jv_points < 3:
            raise ValueError("n_jv_points must be an integer >= 3")
        if type(self.intervals_per_layer) is not int or self.intervals_per_layer < 2:
            raise ValueError("intervals_per_layer must be an integer >= 2")
        if not isinstance(self.damage_protocol, DegradationDamageProtocol):
            raise TypeError("damage_protocol must be a DegradationDamageProtocol")


@dataclass(frozen=True)
class FrozenIonSnapshot:
    """Certified stationary electronic J-V probe at frozen spatial ion profiles."""

    time_s: float
    damage: float
    voltage_V: np.ndarray
    output_current_A_m2: np.ndarray
    metrics: SteadyJVMetrics
    positive_ion_density_m3: np.ndarray
    negative_ion_density_m3: np.ndarray
    positive_reference_m3: np.ndarray
    negative_reference_m3: np.ndarray
    max_ion_flux_m2_s: float
    max_current_spread_A_m2: float
    frozen_profile_exact: bool
    live_compiled_identity: str
    frozen_compiled_identity: str


@dataclass(frozen=True)
class DegradationResult:
    """Time series of frozen-ion J-V snapshots under coupled ionic stress."""

    t_s: np.ndarray
    V_oc_V: np.ndarray
    J_sc_A_m2: np.ndarray
    FF: np.ndarray
    PCE: np.ndarray
    damage: np.ndarray
    snapshots: tuple[FrozenIonSnapshot, ...]
    material_rebind_count: int
    snapshot_live_compiled_identities: tuple[str, ...]
    snapshot_frozen_compiled_identities: tuple[str, ...]
    base_prepared_sha256: str
    result_sha256: str


def freeze_prepared_ions(prepared: PreparedDevice) -> PreparedDevice:
    """Return a cold-resolved ``PreparedDevice`` with ``D_ion = D_ion_neg = 0``.

    Preserves ``P0``, ``P0_neg``, ``P_lim``, ``P_lim_neg``, and steric flags
    unchanged so compensating background charge and finite-site coordinate maps
    match the live ionic device bit-for-bit.
    """
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("prepared must be a PreparedDevice")
    editing = prepared.to_input().editing_data()
    for mat in editing.get("materials", ()):
        mat_p = mat.get("parameters")
        if isinstance(mat_p, dict):
            if "D_ion" in mat_p:
                mat_p["D_ion"] = 0.0
            if "D_ion_neg" in mat_p:
                mat_p["D_ion_neg"] = 0.0
    for lyr in editing.get("layers", ()):
        lyr_p = lyr.setdefault("parameters", {})
        lyr_p["D_ion"] = 0.0
        lyr_p["D_ion_neg"] = 0.0
    return resolve_device(
        DeviceInput.model_validate(editing),
        prepared.defaults,
        prepared.resources,
        sources=prepared.sources,
    )


def apply_absorber_damage(
    base_prepared: PreparedDevice,
    damage: float,
    *,
    alpha: float = 1.0,
    target_layer_id: str | None = None,
) -> PreparedDevice:
    """Cold-rebind absorber ``tau_n, tau_p = tau_0 / (1 + alpha * damage)``."""
    if not isinstance(base_prepared, PreparedDevice):
        raise TypeError("base_prepared must be a PreparedDevice")
    if not _is_real_number(damage) or not math.isfinite(float(damage)) or float(damage) < 0.0:
        raise ValueError("damage must be finite and >= 0")
    if not _is_real_number(alpha) or not math.isfinite(float(alpha)) or float(alpha) < 0.0:
        raise ValueError("alpha must be finite and >= 0")

    scale = 1.0 + float(alpha) * float(damage)
    if scale == 1.0:
        return base_prepared

    base_layers = {lyr.id: dict(lyr.parameters) for lyr in base_prepared.layers}
    editing = base_prepared.to_input().editing_data()
    updated = False
    for lyr in editing.get("layers", ()):
        lid = str(lyr["id"])
        is_target = (
            (lid == target_layer_id)
            if target_layer_id is not None
            else (lyr.get("role") == "absorber")
        )
        if not is_target:
            continue
        orig = base_layers[lid]
        tau_n0 = float(cast(float, orig["tau_n"]))
        tau_p0 = float(cast(float, orig["tau_p"]))
        lyr_p = lyr.setdefault("parameters", {})
        lyr_p["tau_n"] = tau_n0 / scale
        lyr_p["tau_p"] = tau_p0 / scale
        updated = True

    if not updated:
        raise DegradationError(
            "apply_absorber_damage found no matching absorber layer to update"
        )
    return resolve_device(
        DeviceInput.model_validate(editing),
        base_prepared.defaults,
        base_prepared.resources,
        sources=base_prepared.sources,
    )


def _compile_ionic_pair(
    prepared: PreparedDevice,
    *,
    intervals_per_layer: int,
    ion_domain_layer: str | None,
    ion_domain_endpoints: tuple[str, str] | None,
) -> tuple[CompiledIonicDevice1D, CompiledIonicDevice1D]:
    has_ions, inferred_layer = has_declared_ions(prepared)
    if not has_ions:
        raise DegradationError(
            "Degradation snapshots require a PreparedDevice with a declared ionic population"
        )
    elec_layers = tuple(lyr for lyr in prepared.layers if lyr.role != "substrate")
    chosen_layer = ion_domain_layer if ion_domain_layer is not None else inferred_layer
    chosen_end = (
        ion_domain_endpoints
        if ion_domain_endpoints is not None
        else (("blocking", "blocking") if len(elec_layers) > 1 else None)
    )
    live_compiled = compile_ionic_device_1d(
        prepared,
        intervals_per_layer=intervals_per_layer,
        ion_domain_layer=chosen_layer,
        ion_domain_endpoints=chosen_end,
    )
    frozen_prep = freeze_prepared_ions(prepared)
    frozen_compiled = compile_ionic_device_1d(
        frozen_prep,
        intervals_per_layer=intervals_per_layer,
        ion_domain_layer=chosen_layer,
        ion_domain_endpoints=chosen_end,
    )
    return live_compiled, frozen_compiled


def _step_coupled_ionic(
    compiled: CompiledIonicDevice1D,
    prev_state: IonicState1D,
    *,
    voltage_V: float,
    generation_scale: float,
    dt_s: float,
    poisson_tol: float,
    continuity_tol: float,
    contact_tol: float,
    ion_balance_tol: float,
    predictor: str = "previous",
    _depth: int = 0,
) -> tuple[IonicState1D, dict[str, object]]:
    assembled = use_assembled()
    step = (
        cast(
            IonicBackwardEuler1DStep,
            AssembledIonicBackwardEuler1DStep(
                AssembledIonic1DProblem(compiled, voltage_V, generation_scale),
                prev_state,
                dt_s,
            ),
        )
        if assembled
        else IonicBackwardEuler1DStep(
            Ionic1DProblem(compiled, voltage_V, generation_scale),
            prev_state,
            dt_s,
        )
    )
    n_nodes, n_ion = compiled.node_count, compiled.ion_node_count
    weights = np.r_[
        production_residual_scaling(
            compiled.electronic_component,
            poisson_tolerance=poisson_tol,
            continuity_tolerance_A_m2=continuity_tol,
            contact_log_tolerance=contact_tol,
        ).row_weights,
        np.full(
            n_ion * compiled.ion_species_count,
            n_ion / ion_balance_tol,
        ),
    ]
    steps = np.r_[
        production_coordinates(
            compiled.electronic_component,
            potential_step_V=0.1,
            quasi_fermi_step=2.0,
        ).max_step,
        np.full(n_ion * compiled.ion_species_count, 2.0),
    ]
    settings = NewtonSettings(
        ResidualScaling(weights),
        1.0,
        StepControl(steps),
        SolveBudget(80, 640, 82, 24),
        (),
        False,
    )
    elec_norm = production_convergence_norm(
        compiled.electronic_component,
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )
    assert elec_norm is not None

    def conv_norm(scaled: np.ndarray) -> float:
        if scaled.shape != (compiled.state_size,) or not np.all(np.isfinite(scaled)):
            return math.inf
        return max(
            elec_norm(scaled[: 3 * n_nodes]),
            *(
                math.fsum(abs(scaled[block]).tolist()) / n_ion
                for block in compiled.ion_species_slices
            ),
        )

    base_seed = step.initial_coordinates("previous", voltage_lift=True)
    raw_e_step = BackwardEuler1DStep(
        DC1DProblem(compiled.electronic_component, voltage_V, generation_scale),
        prev_state.electronic_state,
        dt_s,
    )
    if predictor == "generation_inventory" and generation_scale > 0.0:
        try:
            seeded_e = transient_density_basin_seed(
                raw_e_step,
                base_seed[: 3 * n_nodes],
                device=compiled.electronic_component,
                voltage_V=voltage_V,
            )
            seed = np.r_[seeded_e, base_seed[3 * n_nodes :]]
        except ValueError:
            seed = step.initial_coordinates(predictor, voltage_lift=True)
    else:
        seed = base_seed

    newton = solve_newton(
        CallableProblem(step.residual, step.jacobian),
        seed,
        settings,
        convergence_norm=conv_norm,
    )
    if (
        (not newton.accepted or not np.all(np.isfinite(newton.state)))
        and predictor != "generation_inventory"
        and generation_scale > 0.0
    ):
        try:
            retry_e = transient_density_basin_seed(
                raw_e_step,
                base_seed[: 3 * n_nodes],
                device=compiled.electronic_component,
                voltage_V=voltage_V,
            )
            newton = solve_newton(
                CallableProblem(step.residual, step.jacobian),
                np.r_[retry_e, base_seed[3 * n_nodes :]],
                settings,
                convergence_norm=conv_norm,
            )
        except ValueError:
            pass
    if not newton.accepted or not np.all(np.isfinite(newton.state)):
        if _depth < 4 and dt_s >= 4e-5:
            v_mid = 0.5 * (prev_state.voltage_V + voltage_V)
            mid_state, _ = _step_coupled_ionic(
                compiled,
                prev_state,
                voltage_V=v_mid,
                generation_scale=generation_scale,
                dt_s=0.5 * dt_s,
                poisson_tol=poisson_tol,
                continuity_tol=continuity_tol,
                contact_tol=contact_tol,
                ion_balance_tol=ion_balance_tol,
                predictor=predictor,
                _depth=_depth + 1,
            )
            return _step_coupled_ionic(
                compiled,
                mid_state,
                voltage_V=voltage_V,
                generation_scale=generation_scale,
                dt_s=0.5 * dt_s,
                poisson_tol=poisson_tol,
                continuity_tol=continuity_tol,
                contact_tol=contact_tol,
                ion_balance_tol=ion_balance_tol,
                predictor="previous",
                _depth=_depth + 1,
            )
        raise DegradationError(
            f"Ionic Backward Euler step failed at V={voltage_V:.4f} V: {newton.termination.value}"
        )
    next_state = compiled.state(newton.state, voltage_V=voltage_V)
    acc = ionic_step_accounting(step, newton.state)
    return next_state, acc


def _lift_electronic_voltage(
    frozen_compiled: CompiledIonicDevice1D,
    u_elec: np.ndarray,
    v_old: float,
    v_new: float,
) -> np.ndarray:
    """Rebase carrier references and harmonically lift potential from v_old to v_new."""
    elec_comp = frozen_compiled.electronic_component
    values = np.array(
        elec_comp.rebase_coordinates(u_elec, v_old, v_new), dtype=float, copy=True
    )
    if v_new != v_old:
        n_nodes = elec_comp.node_count
        res_f = 1.0 / elec_comp.poisson.face_capacitance_F_m2
        prof = np.r_[0.0, np.cumsum(res_f)]
        prof /= prof[-1]
        change = -elec_comp.voltage_polarity * (v_new - v_old)
        before = values[:n_nodes].copy()
        values[:n_nodes] += change * prof
        shift = (values[:n_nodes] - before) / elec_comp.thermal_voltage_V
        values[n_nodes : 2 * n_nodes] -= shift
        values[2 * n_nodes :] -= shift
    return values


def _solve_frozen_electronic_point(
    frozen_compiled: CompiledIonicDevice1D,
    u_elec_seed: np.ndarray,
    u_ion_frozen: np.ndarray,
    *,
    voltage_V: float,
    generation_scale: float,
    poisson_tol: float,
    continuity_tol: float,
    contact_tol: float,
) -> tuple[np.ndarray, float, float, float]:
    """Solve stationary ``3 * N`` electronic rows at exact frozen ``u_ion_frozen``."""
    n_elec = 3 * frozen_compiled.node_count
    prob = (
        AssembledIonic1DProblem(frozen_compiled, voltage_V, generation_scale)
        if use_assembled()
        else Ionic1DProblem(frozen_compiled, voltage_V, generation_scale)
    )
    weights = production_residual_scaling(
        frozen_compiled.electronic_component,
        poisson_tolerance=poisson_tol,
        continuity_tolerance_A_m2=continuity_tol,
        contact_log_tolerance=contact_tol,
    ).row_weights
    steps = production_coordinates(
        frozen_compiled.electronic_component,
        potential_step_V=0.1,
        quasi_fermi_step=2.0,
    ).max_step
    settings = NewtonSettings(
        ResidualScaling(weights),
        1.0,
        StepControl(steps),
        SolveBudget(100, 10000, 2000, 40),
        (),
        False,
    )
    elec_norm = production_convergence_norm(
        frozen_compiled.electronic_component,
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )

    def res_fn(u_e: np.ndarray) -> Matrix:
        return prob.residual(np.r_[u_e, u_ion_frozen])[:n_elec]

    def jac_fn(u_e: np.ndarray) -> Matrix:
        full_j = prob.jacobian(np.r_[u_e, u_ion_frozen])
        return cast(csc_matrix, full_j[:n_elec, :n_elec]).tocsc()

    newton = solve_newton(
        CallableProblem(res_fn, jac_fn),
        u_elec_seed,
        settings,
        convergence_norm=elec_norm,
    )
    if not newton.accepted or not np.all(np.isfinite(newton.state)):
        raise DegradationError(
            f"Frozen-ion electronic steady solve failed at V={voltage_V:.4f} V: {newton.termination.value}"
        )

    u_full = np.r_[newton.state, u_ion_frozen]
    raw_prob = Ionic1DProblem(frozen_compiled, voltage_V, generation_scale)
    j_ion_pos = raw_prob.ion_fluxes(u_full, species="positive")
    max_j_ion = float(np.max(np.abs(j_ion_pos)))
    if frozen_compiled.ion_species_count == 2:
        j_ion_neg = raw_prob.ion_fluxes(u_full, species="negative")
        max_j_ion = max(max_j_ion, float(np.max(np.abs(j_ion_neg))))

    elec_prob = raw_prob.electronic_problem
    j_interior = elec_prob.current_density(newton.state)
    j_contacts = elec_prob.contact_current_density(newton.state)
    j_faces = np.r_[j_contacts[0], j_interior, j_contacts[1]]
    spread = float(np.ptp(j_faces))
    polarity = float(frozen_compiled.electronic_component.voltage_polarity)
    j_out = float(-polarity * j_faces[0])
    return newton.state, j_out, max_j_ion, spread


def measure_frozen_ion_snapshot(
    live_compiled: CompiledIonicDevice1D,
    frozen_compiled: CompiledIonicDevice1D,
    live_state: IonicState1D,
    probe_voltages_V: object,
    *,
    time_s: float = 0.0,
    damage: float = 0.0,
    generation_scale: float = 1.0,
    settle_time_s: float = 0.0,
    settle_steps: int = 4,
    incident_power_W_m2: float = 1000.0,
    poisson_tolerance: float = 1e-10,
    continuity_tolerance_A_m2: float = 1e-7,
    contact_log_tolerance: float = 1e-12,
    max_current_error_A_m2: float = 1e-6,
    ion_balance_tolerance_m2_s: float = 1e10,
) -> FrozenIonSnapshot:
    """Measure a stationary J-V curve with spatial ion profiles frozen bit-for-bit."""
    if live_state.device_sha256 != live_compiled.identity:
        raise ValueError("live_state does not belong to live_compiled")
    if (
        not np.all(frozen_compiled.positive_ion_diffusivity_faces_m2_s == 0.0)
        or (
            frozen_compiled.ion_species_count == 2
            and not np.all(frozen_compiled.negative_ion_diffusivity_faces_m2_s == 0.0)
        )
    ):
        raise ValueError("frozen_compiled must have identically zero ion diffusivity")

    v_probe = np.array(probe_voltages_V, dtype=float, copy=True)
    if (
        v_probe.ndim != 1
        or v_probe.size < 2
        or not np.all(np.isfinite(v_probe))
        or not (np.all(np.diff(v_probe) > 0.0) or np.all(np.diff(v_probe) < 0.0))
    ):
        raise ValueError("probe_voltages_V must be strictly monotonic 1-D finite voltages")
    v_probe.setflags(write=False)

    n_elec = 3 * live_compiled.node_count
    u_ion_snap = np.array(live_state.u[n_elec:], dtype=float, copy=True)
    u_ion_snap.setflags(write=False)

    # Smoothly continue electronic coordinates from live_state.voltage_V to v0
    # while holding u_ion_snap bit-identical (0 ULP).
    v0 = float(v_probe[0])
    u_elec_start = np.array(live_state.u[:n_elec], dtype=float, copy=True)
    snap_state = frozen_compiled.state(
        np.r_[
            frozen_compiled.electronic_component.rebase_coordinates(
                u_elec_start, live_state.voltage_V, v0
            ),
            u_ion_snap,
        ],
        voltage_V=v0,
    )

    # Verify 0-ULP bit-identical frozen ion profiles & reference backgrounds
    if not np.array_equal(
        snap_state.positive_ion_density_m3, live_state.positive_ion_density_m3
    ) or not np.array_equal(
        snap_state.negative_ion_density_m3, live_state.negative_ion_density_m3
    ):
        raise DegradationError(
            "Frozen-ion device mapping altered spatial ion density profiles"
        )
    if not np.array_equal(
        frozen_compiled.positive_ion_reference_m3,
        live_compiled.positive_ion_reference_m3,
    ) or not np.array_equal(
        frozen_compiled.negative_ion_reference_m3,
        live_compiled.negative_ion_reference_m3,
    ):
        raise DegradationError(
            "Frozen-ion device altered compensating P0 / P0_neg background density"
        )

    def advance_frozen_v(
        u_prev: np.ndarray,
        v_from: float,
        v_to: float,
        _depth: int = 0,
    ) -> tuple[np.ndarray, float, float, float]:
        lifted = _lift_electronic_voltage(frozen_compiled, u_prev, v_from, v_to)
        try:
            return _solve_frozen_electronic_point(
                frozen_compiled,
                lifted,
                u_ion_snap,
                voltage_V=v_to,
                generation_scale=generation_scale,
                poisson_tol=poisson_tolerance,
                continuity_tol=continuity_tolerance_A_m2,
                contact_tol=contact_log_tolerance,
            )
        except DegradationError:
            if _depth >= 5 or abs(v_to - v_from) < 1e-4:
                raise
            v_mid = 0.5 * (v_from + v_to)
            u_mid, _, _, _ = advance_frozen_v(u_prev, v_from, v_mid, _depth + 1)
            return advance_frozen_v(u_mid, v_mid, v_to, _depth + 1)

    # Relax electronic carriers at live_state.voltage_V first, then step smoothly to v0
    u_bridge, _, _, _ = advance_frozen_v(
        u_elec_start, live_state.voltage_V, live_state.voltage_V
    )
    if v0 != live_state.voltage_V:
        n_bridge = max(1, int(math.ceil(abs(v0 - live_state.voltage_V) / 0.08)))
        v_bridge = np.linspace(live_state.voltage_V, v0, n_bridge + 1)
        for k_br in range(1, v_bridge.size):
            u_bridge, _, _, _ = advance_frozen_v(
                u_bridge, float(v_bridge[k_br - 1]), float(v_bridge[k_br])
            )
    snap_state = frozen_compiled.state(np.r_[u_bridge, u_ion_snap], voltage_V=v0)

    if settle_time_s > 0.0:
        dt_settle = settle_time_s / settle_steps
        for _ in range(settle_steps):
            snap_state, _ = _step_coupled_ionic(
                frozen_compiled,
                snap_state,
                voltage_V=v0,
                generation_scale=generation_scale,
                dt_s=dt_settle,
                poisson_tol=poisson_tolerance,
                continuity_tol=continuity_tolerance_A_m2,
                contact_tol=contact_log_tolerance,
                ion_balance_tol=ion_balance_tolerance_m2_s,
            )
            if not np.array_equal(snap_state.u[n_elec:], u_ion_snap):
                raise DegradationError(
                    "Frozen-ion carrier relaxation settle mutated immobile ion coordinates"
                )

    u_elec_cur = np.array(snap_state.u[:n_elec], dtype=float, copy=True)
    v_prev = v0
    j_out = np.empty(v_probe.size, dtype=float)
    max_ion_flux = 0.0
    max_spread = 0.0

    for idx, v_val in enumerate(v_probe):
        v_f = float(v_val)
        u_elec_cur, j_k, flux_k, spread_k = advance_frozen_v(
            u_elec_cur, v_prev, v_f
        )
        if flux_k != 0.0:
            raise DegradationError(
                f"Nonzero ionic flux {flux_k:.3e} m^-2 s^-1 detected during frozen-ion probe"
            )
        if spread_k > max_current_error_A_m2:
            raise DegradationError(
                f"Frozen-ion probe current spread {spread_k:.3e} A/m^2 > {max_current_error_A_m2:.3e} A/m^2 at V={v_f:.4f} V"
            )
        probe_check = frozen_compiled.state(np.r_[u_elec_cur, u_ion_snap], voltage_V=v_f)
        if not np.array_equal(
            probe_check.positive_ion_density_m3, live_state.positive_ion_density_m3
        ) or not np.array_equal(
            probe_check.negative_ion_density_m3, live_state.negative_ion_density_m3
        ):
            raise DegradationError(
                "Frozen-ion density drifted during voltage probe sweep"
            )

        j_out[idx] = j_k
        max_ion_flux = max(max_ion_flux, flux_k)
        max_spread = max(max_spread, spread_k)
        v_prev = v_f

    j_out.setflags(write=False)
    metrics = steady_jv_metrics(
        v_probe, j_out, incident_power_W_m2=incident_power_W_m2
    )
    pos_prof = np.array(live_state.positive_ion_density_m3, dtype=float, copy=True)
    neg_prof = np.array(live_state.negative_ion_density_m3, dtype=float, copy=True)
    pos_ref = np.array(
        frozen_compiled.positive_ion_reference_m3, dtype=float, copy=True
    )
    neg_ref = np.array(
        frozen_compiled.negative_ion_reference_m3, dtype=float, copy=True
    )
    for arr in (pos_prof, neg_prof, pos_ref, neg_ref):
        arr.setflags(write=False)

    return FrozenIonSnapshot(
        time_s=float(time_s),
        damage=float(damage),
        voltage_V=v_probe,
        output_current_A_m2=j_out,
        metrics=metrics,
        positive_ion_density_m3=pos_prof,
        negative_ion_density_m3=neg_prof,
        positive_reference_m3=pos_ref,
        negative_reference_m3=neg_ref,
        max_ion_flux_m2_s=max_ion_flux,
        max_current_spread_A_m2=max_spread,
        frozen_profile_exact=True,
        live_compiled_identity=live_compiled.identity,
        frozen_compiled_identity=frozen_compiled.identity,
    )


def _advance_damage(
    damage_prev: float,
    acc: dict[str, object],
    state: IonicState1D,
    compiled: CompiledIonicDevice1D,
    dt_s: float,
    dmg_prot: DegradationDamageProtocol,
) -> float:
    inc = 0.0
    if dmg_prot.motion_gain_m2_C > 0.0:
        j_ion = np.asarray(acc["ion_conduction_current_density_A_m2"], dtype=float)
        mean_abs_j_ion = float(np.mean(np.abs(j_ion)))
        inc += dmg_prot.motion_gain_m2_C * mean_abs_j_ion * dt_s
    if dmg_prot.segregation_gain_s_1 > 0.0:
        rel_pos = np.max(
            np.abs(state.positive_ion_density_m3 - compiled.positive_ion_reference_m3)
            / compiled.positive_ion_reference_m3
        )
        rel_max = float(rel_pos)
        if compiled.ion_species_count == 2:
            rel_neg = np.max(
                np.abs(
                    state.negative_ion_density_m3 - compiled.negative_ion_reference_m3
                )
                / compiled.negative_ion_reference_m3
            )
            rel_max = max(rel_max, float(rel_neg))
        inc += dmg_prot.segregation_gain_s_1 * rel_max * dt_s
    return damage_prev + inc


def run_degradation(
    prepared: PreparedDevice,
    t_eval_s: object,
    protocol: DegradationProtocol | None = None,
) -> DegradationResult:
    """Evolve coupled ionic stress and record frozen-ion J-V snapshots at ``t_eval_s``."""
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("prepared must be a PreparedDevice")
    prot = DegradationProtocol() if protocol is None else protocol
    if not isinstance(prot, DegradationProtocol):
        raise TypeError("protocol must be a DegradationProtocol")

    if np.iscomplexobj(t_eval_s):
        raise ValueError("t_eval_s must be real-valued")
    times = np.array(t_eval_s, dtype=float, copy=True)
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or float(times[0]) < 0.0
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError(
            "t_eval_s must be a 1-D strictly increasing nonnegative array with >= 2 samples"
        )
    times.setflags(write=False)

    live_compiled, frozen_compiled = _compile_ionic_pair(
        prepared,
        intervals_per_layer=prot.intervals_per_layer,
        ion_domain_layer=prot.ion_domain_layer,
        ion_domain_endpoints=prot.ion_domain_endpoints,
    )

    # Start from dark 0 V equilibrium, then step cleanly to stress bias at t = 0
    steady_opts = IonicSteadyOptions(
        electronic=SteadyOptions(
            poisson_tolerance=prot.poisson_tolerance,
            continuity_tolerance_A_m2=prot.continuity_tolerance_A_m2,
            contact_log_tolerance=prot.contact_log_tolerance,
            convergence_policy=HISTORICAL_MAX_ROW_V1,
        ),
        inventory_relative_tolerance=1e-11,
        ion_balance_tolerance_m2_s=prot.ion_balance_tolerance_m2_s,
        ion_flux_tolerance_m2_s=prot.ion_balance_tolerance_m2_s,
        ion_logit_step=2.0,
        inventory_row_node=live_compiled.ion_node_count - 1,
        budget=SolveBudget(120, 960, 122, 12),
        wall_s=30.0,
    )
    init_sol = solve_ionic_steady_state(
        live_compiled,
        voltage_V=0.0,
        generation_scale=0.0,
        options=steady_opts,
        initial_state=None,
    )
    if not init_sol.accepted or init_sol.state is None:
        raise DegradationError(
            f"Initial 0 V dark ionic steady state failed: {init_sol.termination}"
        )
    cur_state = init_sol.state
    if prot.precondition_time_s > 0.0 and (
        prot.V_bias_V != 0.0 or prot.generation_scale != 0.0
    ):
        n_v_steps = max(4, int(math.ceil(abs(prot.V_bias_V) / 0.15)))
        total_pre = max(prot.precondition_time_s, 5e-4)
        # First 3 steps smoothly turn on illumination at 0 V with dt >= 2e-5 s,
        # followed by smooth voltage ramp to V_bias_V.
        if prot.generation_scale > 0.0:
            light_steps = [
                (0.0, prot.generation_scale * 1e-3, 2e-5),
                (0.0, prot.generation_scale * 1e-1, 8e-5),
                (0.0, prot.generation_scale, 2e-4),
            ]
        else:
            light_steps = []
        rem_pre = max(total_pre - 3e-4, 2e-4)
        v_ramp = np.linspace(0.0, prot.V_bias_V, n_v_steps + 1)[1:]
        dt_v = rem_pre / n_v_steps
        bias_steps = [(float(v_step), prot.generation_scale, dt_v) for v_step in v_ramp]
        for idx_pre, (v_p, g_p, dt_pre) in enumerate(light_steps + bias_steps):
            cur_state, _ = _step_coupled_ionic(
                live_compiled,
                cur_state,
                voltage_V=v_p,
                generation_scale=g_p,
                dt_s=dt_pre,
                poisson_tol=prot.poisson_tolerance,
                continuity_tol=prot.continuity_tolerance_A_m2,
                contact_tol=prot.contact_log_tolerance,
                ion_balance_tol=prot.ion_balance_tolerance_m2_s,
                predictor=(
                    "generation_inventory"
                    if (idx_pre == 0 and g_p > 0.0)
                    else "previous"
                ),
            )

    # Advance from t = 0.0 to times[0] if times[0] > 0.0, or rebase electronic bias at t = 0
    t_cur = 0.0
    damage = 0.0
    cached_damage = 0.0
    rebind_count = 0

    def maybe_rebind_for_damage(
        dmg_val: float, state_in: IonicState1D
    ) -> tuple[CompiledIonicDevice1D, CompiledIonicDevice1D, IonicState1D]:
        nonlocal cached_damage, rebind_count, live_compiled, frozen_compiled
        dmg_prot = prot.damage_protocol
        if (
            dmg_prot.lifetime_degradation_alpha == 0.0
            or abs(dmg_val - cached_damage) <= dmg_prot.damage_rebind_threshold
            or dmg_val == cached_damage
        ):
            return live_compiled, frozen_compiled, state_in

        damaged_prep = apply_absorber_damage(
            prepared,
            dmg_val,
            alpha=dmg_prot.lifetime_degradation_alpha,
            target_layer_id=live_compiled.domain_layer_id,
        )
        new_live, new_frozen = _compile_ionic_pair(
            damaged_prep,
            intervals_per_layer=prot.intervals_per_layer,
            ion_domain_layer=prot.ion_domain_layer,
            ion_domain_endpoints=prot.ion_domain_endpoints,
        )
        rebound_state = new_live.state(state_in.u, voltage_V=state_in.voltage_V)
        live_compiled = new_live
        frozen_compiled = new_frozen
        cached_damage = dmg_val
        rebind_count += 1
        return live_compiled, frozen_compiled, rebound_state

    def advance_interval(target_t: float, state_in: IonicState1D) -> IonicState1D:
        nonlocal t_cur, damage
        span = target_t - t_cur
        if span <= 0.0:
            return state_in
        n_chunks = max(1, int(math.ceil(span / prot.dt_max_s)))
        dt_chunk = span / n_chunks
        st = state_in
        for _ in range(n_chunks):
            st, acc = _step_coupled_ionic(
                live_compiled,
                st,
                voltage_V=prot.V_bias_V,
                generation_scale=prot.generation_scale,
                dt_s=dt_chunk,
                poisson_tol=prot.poisson_tolerance,
                continuity_tol=prot.continuity_tolerance_A_m2,
                contact_tol=prot.contact_log_tolerance,
                ion_balance_tol=prot.ion_balance_tolerance_m2_s,
            )
            t_cur += dt_chunk
            damage = _advance_damage(
                damage, acc, st, live_compiled, dt_chunk, prot.damage_protocol
            )
            _, _, st = maybe_rebind_for_damage(damage, st)
        t_cur = target_t
        return st

    if float(times[0]) > 0.0:
        cur_state = advance_interval(float(times[0]), cur_state)

    v_probe = np.linspace(0.0, prot.V_max_V, prot.n_jv_points)
    snapshots: list[FrozenIonSnapshot] = []

    for idx_snap, t_snap in enumerate(times):
        if idx_snap > 0:
            cur_state = advance_interval(float(t_snap), cur_state)
        snap = measure_frozen_ion_snapshot(
            live_compiled,
            frozen_compiled,
            cur_state,
            v_probe,
            time_s=float(t_snap),
            damage=damage,
            generation_scale=prot.probe_generation_scale,
            settle_time_s=prot.settle_time_s,
            settle_steps=prot.settle_steps,
            incident_power_W_m2=prot.incident_power_W_m2,
            poisson_tolerance=prot.poisson_tolerance,
            continuity_tolerance_A_m2=prot.continuity_tolerance_A_m2,
            contact_log_tolerance=prot.contact_log_tolerance,
            max_current_error_A_m2=prot.max_current_error_A_m2,
            ion_balance_tolerance_m2_s=prot.ion_balance_tolerance_m2_s,
        )
        snapshots.append(snap)

    voc_arr = np.array(
        [
            float("nan")
            if s.metrics.open_circuit_voltage_V is None
            else s.metrics.open_circuit_voltage_V
            for s in snapshots
        ],
        dtype=float,
    )
    jsc_arr = np.array(
        [
            float("nan")
            if s.metrics.short_circuit_current_A_m2 is None
            else s.metrics.short_circuit_current_A_m2
            for s in snapshots
        ],
        dtype=float,
    )
    ff_arr = np.array(
        [
            float("nan") if s.metrics.fill_factor is None else s.metrics.fill_factor
            for s in snapshots
        ],
        dtype=float,
    )
    pce_arr = np.array(
        [
            float("nan") if s.metrics.efficiency is None else s.metrics.efficiency
            for s in snapshots
        ],
        dtype=float,
    )
    dmg_arr = np.array([s.damage for s in snapshots], dtype=float)
    for arr in (voc_arr, jsc_arr, ff_arr, pce_arr, dmg_arr):
        arr.setflags(write=False)

    live_ids = tuple(s.live_compiled_identity for s in snapshots)
    frozen_ids = tuple(s.frozen_compiled_identity for s in snapshots)

    sha = _canonical_sha256(
        {
            "base_prepared_sha256": prepared.content_sha256,
            "t_s": [float(x).hex() for x in times],
            "damage": [float(x).hex() for x in dmg_arr],
            "material_rebind_count": rebind_count,
            "live_ids": list(live_ids),
            "frozen_ids": list(frozen_ids),
            "metrics": [asdict(s.metrics) for s in snapshots],
        }
    )
    return DegradationResult(
        t_s=times,
        V_oc_V=voc_arr,
        J_sc_A_m2=jsc_arr,
        FF=ff_arr,
        PCE=pce_arr,
        damage=dmg_arr,
        snapshots=tuple(snapshots),
        material_rebind_count=rebind_count,
        snapshot_live_compiled_identities=live_ids,
        snapshot_frozen_compiled_identities=frozen_ids,
        base_prepared_sha256=prepared.content_sha256,
        result_sha256=sha,
    )
