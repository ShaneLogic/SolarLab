"""Coupled steady-state electrothermal operating-point solver (P04-13 / P05-04).

At every trial temperature ``T`` visited by the bracketed root finder:
1. ``PreparedDevice`` is cold-rebound from its immutable declaration by setting
   ``settings.T = T`` (leaving all 300 K layer parameters untouched so
   ``compile_layer_parameters`` / ``compile_ionic_device_1d`` apply the exact
   temperature laws ``Nc(T), Nv(T), Eg(T), ni(T), mu(T), B_rad(T), D_ion(T)``).
2. A fresh ``CompiledDevice1D`` or ``CompiledIonicDevice1D`` is compiled and
   evaluated from a fresh equilibrium/steady state at ``T``: neither carrier nor
   ionic state is ever inherited across trial temperatures.
3. The intrinsic J-V sweep is mapped through ``ExternalCircuitProtocol`` and the
   operating-point terminal power ``P_elec_out(T)`` (sampled terminal MPP,
   open-circuit, or fixed load) is inserted into ``LumpedThermalProtocol``.

Electrothermal self-heating is mutually exclusive with an externally prescribed
temperature schedule ``T(t)``; non-isothermal PDE time stepping remains gated by
``TIME_DOMAIN_TEMPERATURE_QUALIFIED`` (``False``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Literal, cast

import numpy as np
from scipy.optimize import brentq

from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem
from solarlab.equations.implementation import use_assembled
from solarlab.equations.ionic_1d import (
    CompiledIonicDevice1D,
    Ionic1DProblem,
    IonicState1D,
)
from solarlab.experiments.external_circuit import (
    ExternalCircuitJVResult,
    ExternalCircuitProtocol,
    apply_external_circuit,
)
from solarlab.experiments.jv.steady import run_steady_jv
from solarlab.experiments.thermal_balance import (
    LumpedThermalProtocol,
    ThermalPowerLedger,
    thermal_power_ledger,
)
from solarlab.experiments.transient import (
    TransientOptions,
    TransientProtocol,
    solve_transient,
)
from solarlab.physics.band_structure import TIME_DOMAIN_TEMPERATURE_QUALIFIED
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
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    density_basin_seed,
    solve_steady_state,
)
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
    "ElectrothermalCapabilityError",
    "ElectrothermalConvergenceError",
    "ElectrothermalError",
    "ElectrothermalJVProtocol",
    "ElectrothermalOperatingPointProtocol",
    "ElectrothermalOperatingPointResult",
    "ElectrothermalSourceError",
    "ElectrothermalTemperatureEvaluation",
    "compile_prepared_at_temperature",
    "has_declared_ions",
    "rebind_prepared_temperature",
    "run_fresh_jv_at_temperature",
    "solve_electrothermal_operating_point",
]


class ElectrothermalError(RuntimeError):
    """Base error for electrothermal operating-point coupling."""


class ElectrothermalCapabilityError(ElectrothermalError):
    """Unsupported or mutually exclusive electrothermal capability request."""


class ElectrothermalSourceError(ElectrothermalError):
    """Intrinsic J-V or external-circuit evaluation failed certification."""


class ElectrothermalConvergenceError(ElectrothermalError):
    """Bracketed electrothermal root failed to converge within residual tolerances."""


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(
        value, bool
    )


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def has_declared_ions(prepared: PreparedDevice) -> tuple[bool, str | None]:
    active_layers: list[str] = []
    for layer in prepared.layers:
        if layer.role == "substrate":
            continue
        params = dict(layer.parameters)
        if any(
            float(cast(float, params[name])) != 0.0
            for name in ("P0", "P0_neg", "D_ion", "D_ion_neg")
        ):
            active_layers.append(layer.id)
    if not active_layers:
        return False, None
    if len(active_layers) == 1:
        return True, active_layers[0]
    return True, None


def rebind_prepared_temperature(
    prepared: PreparedDevice,
    temperature_K: float,
) -> PreparedDevice:
    """Return a cold-rebound ``PreparedDevice`` at ``temperature_K``."""
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("prepared must be a PreparedDevice")
    if (
        not _is_real_number(temperature_K)
        or not math.isfinite(float(temperature_K))
        or float(temperature_K) <= 0.0
    ):
        raise ValueError("temperature_K must be a finite positive real scalar")

    editing = prepared.to_input().editing_data()
    editing.setdefault("settings", {})["T"] = float(temperature_K)
    return resolve_device(
        DeviceInput.model_validate(editing),
        prepared.defaults,
        prepared.resources,
        sources=prepared.sources,
    )


def compile_prepared_at_temperature(
    prepared: PreparedDevice,
    temperature_K: float,
    *,
    intervals_per_layer: int = 24,
    ion_domain_layer: str | None = None,
    ion_domain_endpoints: tuple[str, str] | None = None,
) -> tuple[PreparedDevice, CompiledDevice1D | CompiledIonicDevice1D]:
    """Cold-rebind and freshly compile ``prepared`` at ``temperature_K``."""
    rebound = rebind_prepared_temperature(prepared, temperature_K)
    has_ions, inferred_layer = has_declared_ions(rebound)
    if not has_ions:
        return rebound, compile_device_1d(
            rebound, intervals_per_layer=intervals_per_layer
        )

    electrical_layers = tuple(
        layer for layer in rebound.layers if layer.role != "substrate"
    )
    chosen_layer = ion_domain_layer if ion_domain_layer is not None else inferred_layer
    chosen_endpoints = ion_domain_endpoints
    if chosen_endpoints is None and len(electrical_layers) > 1:
        chosen_endpoints = ("blocking", "blocking")
    return rebound, compile_ionic_device_1d(
        rebound,
        intervals_per_layer=intervals_per_layer,
        ion_domain_layer=chosen_layer,
        ion_domain_endpoints=chosen_endpoints,
    )


@dataclass(frozen=True)
class ElectrothermalJVProtocol:
    """Explicit fresh-state J-V evaluation contract at each trial temperature."""

    intervals_per_layer: int = 24
    voltage_points_per_branch: int = 41
    scan_rate_V_s: float = 1.0
    voltage_max_V: float = 1.4
    substeps_per_point: int = 1
    precondition_time_s: float = 0.0
    incident_power_W_m2: float = 1000.0
    driver: Literal["transient_jv_strict", "steady_jv"] = "transient_jv_strict"
    temperature_initialization: Literal["fresh_state_per_temperature"] = (
        "fresh_state_per_temperature"
    )
    ion_domain_layer: str | None = None
    ion_domain_endpoints: tuple[str, str] | None = None
    poisson_tolerance: float = 1e-10
    continuity_tolerance_A_m2: float = 1e-7
    contact_log_tolerance: float = 1e-12
    ion_balance_tolerance_m2_s: float = 1e10
    wall_s_per_temperature: float = 30.0

    def __post_init__(self) -> None:
        if type(self.intervals_per_layer) is not int or self.intervals_per_layer < 2:
            raise ValueError("intervals_per_layer must be an integer >= 2")
        if (
            type(self.voltage_points_per_branch) is not int
            or self.voltage_points_per_branch < 3
        ):
            raise ValueError("voltage_points_per_branch must be an integer >= 3")
        if type(self.substeps_per_point) is not int or self.substeps_per_point < 1:
            raise ValueError("substeps_per_point must be an integer >= 1")
        for name in (
            "scan_rate_V_s",
            "voltage_max_V",
            "incident_power_W_m2",
            "poisson_tolerance",
            "continuity_tolerance_A_m2",
            "contact_log_tolerance",
            "ion_balance_tolerance_m2_s",
            "wall_s_per_temperature",
        ):
            raw = getattr(self, name)
            if not _is_real_number(raw) or not math.isfinite(float(raw)) or float(raw) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(raw))
        if (
            not _is_real_number(self.precondition_time_s)
            or not math.isfinite(float(self.precondition_time_s))
            or float(self.precondition_time_s) < 0.0
        ):
            raise ValueError("precondition_time_s must be finite and >= 0")
        object.__setattr__(self, "precondition_time_s", float(self.precondition_time_s))
        if self.driver not in ("transient_jv_strict", "steady_jv"):
            raise ValueError(
                "driver must be 'transient_jv_strict' or 'steady_jv'"
            )
        if self.temperature_initialization != "fresh_state_per_temperature":
            raise ValueError(
                "temperature_initialization must be 'fresh_state_per_temperature'"
            )

    @property
    def sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class ElectrothermalOperatingPointProtocol:
    """Operating-point selection and Brent root controls."""

    temperature_absolute_tolerance_K: float = 1e-8
    power_residual_tolerance_W_m2: float = 1e-6
    maximum_root_iterations: int = 50
    operating_branch: Literal["forward", "reverse"] = "forward"
    operating_point: Literal[
        "sampled_terminal_maximum_power", "open_circuit", "fixed_load"
    ] = "sampled_terminal_maximum_power"
    fixed_terminal_voltage_V: float | None = None
    load_resistance_ohm_m2: float | None = None
    circuit_control_volume: Literal["device_plus_lumped_parasitics"] = (
        "device_plus_lumped_parasitics"
    )

    def __post_init__(self) -> None:
        for name in (
            "temperature_absolute_tolerance_K",
            "power_residual_tolerance_W_m2",
        ):
            raw = getattr(self, name)
            if not _is_real_number(raw) or not math.isfinite(float(raw)) or float(raw) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(raw))
        if (
            type(self.maximum_root_iterations) is not int
            or self.maximum_root_iterations < 1
        ):
            raise ValueError("maximum_root_iterations must be a positive integer")
        if self.operating_branch not in ("forward", "reverse"):
            raise ValueError("operating_branch must be 'forward' or 'reverse'")
        if self.operating_point not in (
            "sampled_terminal_maximum_power",
            "open_circuit",
            "fixed_load",
        ):
            raise ValueError(
                f"Unsupported operating_point {self.operating_point!r}"
            )
        if self.circuit_control_volume != "device_plus_lumped_parasitics":
            raise ValueError(
                "circuit_control_volume must be 'device_plus_lumped_parasitics'"
            )
        if self.operating_point == "fixed_load":
            has_v = self.fixed_terminal_voltage_V is not None
            has_r = self.load_resistance_ohm_m2 is not None
            if has_v == has_r:
                raise ValueError(
                    "fixed_load requires exactly one of fixed_terminal_voltage_V or load_resistance_ohm_m2"
                )
            if has_v:
                assert self.fixed_terminal_voltage_V is not None
                if not _is_real_number(
                    self.fixed_terminal_voltage_V
                ) or not math.isfinite(float(self.fixed_terminal_voltage_V)):
                    raise ValueError("fixed_terminal_voltage_V must be finite")
                object.__setattr__(
                    self,
                    "fixed_terminal_voltage_V",
                    float(self.fixed_terminal_voltage_V),
                )
            if has_r:
                assert self.load_resistance_ohm_m2 is not None
                if (
                    not _is_real_number(self.load_resistance_ohm_m2)
                    or not math.isfinite(float(self.load_resistance_ohm_m2))
                    or float(self.load_resistance_ohm_m2) <= 0.0
                ):
                    raise ValueError(
                        "load_resistance_ohm_m2 must be finite and positive"
                    )
                object.__setattr__(
                    self,
                    "load_resistance_ohm_m2",
                    float(self.load_resistance_ohm_m2),
                )
        else:
            if (
                self.fixed_terminal_voltage_V is not None
                or self.load_resistance_ohm_m2 is not None
            ):
                raise ValueError(
                    "fixed_terminal_voltage_V / load_resistance_ohm_m2 are allowed only when operating_point='fixed_load'"
                )

    @property
    def sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class ElectrothermalTemperatureEvaluation:
    """Single cold-rebound trial-temperature evaluation record."""

    temperature_K: float
    rebound_prepared_sha256: str
    compiled_identity: str
    terminal_voltage_V: float
    terminal_current_A_m2: float
    terminal_power_W_m2: float
    power_balance_residual_W_m2: float
    external_result_sha256: str
    fresh_state_initialized: bool
    source_certified: bool


@dataclass(frozen=True)
class ElectrothermalOperatingPointResult:
    """Certified coupled steady-state electrothermal operating point."""

    operating_temperature_K: float
    temperature_rise_K: float
    terminal_voltage_V: float
    terminal_current_A_m2: float
    terminal_power_W_m2: float
    power_balance_residual_W_m2: float
    thermal_ledger: ThermalPowerLedger
    final_external_result: ExternalCircuitJVResult
    temperature_evaluations: tuple[ElectrothermalTemperatureEvaluation, ...]
    electrical_evaluations: int
    root_iterations: int
    base_prepared_sha256: str
    thermal_protocol: LumpedThermalProtocol
    circuit_protocol: ExternalCircuitProtocol
    electrical_protocol: ElectrothermalJVProtocol
    operating_protocol: ElectrothermalOperatingPointProtocol
    time_domain_temperature_qualified: bool
    mapping_sha256: str
    certified: bool


def _step_ionic_be(
    compiled: CompiledIonicDevice1D,
    prev_state: IonicState1D,
    *,
    voltage_V: float,
    generation_scale: float,
    dt_s: float,
    elec_prot: ElectrothermalJVProtocol,
    predictor: str = "previous",
    _depth: int = 0,
) -> tuple[IonicState1D, float]:
    """Advance one conservative Backward Euler step on ``CompiledIonicDevice1D``."""
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
    count, ion_count = compiled.node_count, compiled.ion_node_count
    weights = np.r_[
        production_residual_scaling(
            compiled.electronic_component,
            poisson_tolerance=elec_prot.poisson_tolerance,
            continuity_tolerance_A_m2=elec_prot.continuity_tolerance_A_m2,
            contact_log_tolerance=elec_prot.contact_log_tolerance,
        ).row_weights,
        np.full(
            ion_count * compiled.ion_species_count,
            ion_count / elec_prot.ion_balance_tolerance_m2_s,
        ),
    ]
    steps = np.r_[
        production_coordinates(
            compiled.electronic_component,
            potential_step_V=0.1,
            quasi_fermi_step=2.0,
        ).max_step,
        np.full(ion_count * compiled.ion_species_count, 2.0),
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
            elec_norm(scaled[: 3 * count]),
            *(
                math.fsum(abs(scaled[block]).tolist()) / ion_count
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
                base_seed[: 3 * count],
                device=compiled.electronic_component,
                voltage_V=voltage_V,
            )
            seed = np.r_[seeded_e, base_seed[3 * count :]]
        except ValueError:
            seed = step.initial_coordinates(predictor, voltage_lift=True)
    else:
        seed = base_seed

    def eval_res(y: np.ndarray) -> Matrix:
        return step.residual(y)

    def eval_jac(y: np.ndarray) -> Matrix:
        return step.jacobian(y)

    newton = solve_newton(
        CallableProblem(eval_res, eval_jac),
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
                base_seed[: 3 * count],
                device=compiled.electronic_component,
                voltage_V=voltage_V,
            )
            newton = solve_newton(
                CallableProblem(eval_res, eval_jac),
                np.r_[retry_e, base_seed[3 * count :]],
                settings,
                convergence_norm=conv_norm,
            )
        except ValueError:
            pass
    if not newton.accepted or not np.all(np.isfinite(newton.state)):
        if _depth < 4 and dt_s >= 4e-5:
            v_mid = 0.5 * (prev_state.voltage_V + voltage_V)
            mid_state, _ = _step_ionic_be(
                compiled,
                prev_state,
                voltage_V=v_mid,
                generation_scale=generation_scale,
                dt_s=0.5 * dt_s,
                elec_prot=elec_prot,
                predictor=predictor,
                _depth=_depth + 1,
            )
            return _step_ionic_be(
                compiled,
                mid_state,
                voltage_V=voltage_V,
                generation_scale=generation_scale,
                dt_s=0.5 * dt_s,
                elec_prot=elec_prot,
                predictor="previous",
                _depth=_depth + 1,
            )
        raise ElectrothermalSourceError(
            f"Ionic Backward Euler step failed at V={voltage_V:.4f} V: {newton.termination.value}"
        )
    next_state = compiled.state(newton.state, voltage_V=voltage_V)
    acc = ionic_step_accounting(step, newton.state)
    term_total = np.asarray(acc["terminal_total_current_density_A_m2"], dtype=float)
    polarity = float(compiled.electronic_component.voltage_polarity)
    return next_state, float(-polarity * term_total[0])


def run_fresh_jv_at_temperature(
    compiled: CompiledDevice1D | CompiledIonicDevice1D,
    elec_prot: ElectrothermalJVProtocol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a fresh-state forward & reverse J-V sweep on ``compiled``."""
    v_fwd = np.linspace(0.0, elec_prot.voltage_max_V, elec_prot.voltage_points_per_branch)
    v_rev = v_fwd[::-1].copy()

    if elec_prot.driver == "steady_jv":
        if isinstance(compiled, CompiledDevice1D):
            steady_opts = SteadyOptions(
                poisson_tolerance=elec_prot.poisson_tolerance,
                continuity_tolerance_A_m2=elec_prot.continuity_tolerance_A_m2,
                contact_log_tolerance=elec_prot.contact_log_tolerance,
            )
            res_fwd = run_steady_jv(
                compiled,
                v_fwd,
                generation_scale=1.0,
                incident_power_W_m2=elec_prot.incident_power_W_m2,
                options=steady_opts,
            )
            res_rev = run_steady_jv(
                compiled,
                v_rev,
                generation_scale=1.0,
                incident_power_W_m2=elec_prot.incident_power_W_m2,
                options=steady_opts,
            )
            return v_fwd, res_fwd.output_current_A_m2, v_rev, res_rev.output_current_A_m2

        ionic_opts = IonicSteadyOptions(
            electronic=SteadyOptions(
                poisson_tolerance=elec_prot.poisson_tolerance,
                continuity_tolerance_A_m2=elec_prot.continuity_tolerance_A_m2,
                contact_log_tolerance=elec_prot.contact_log_tolerance,
                convergence_policy=HISTORICAL_MAX_ROW_V1,
            ),
            inventory_relative_tolerance=1e-11,
            ion_balance_tolerance_m2_s=elec_prot.ion_balance_tolerance_m2_s,
            ion_flux_tolerance_m2_s=elec_prot.ion_balance_tolerance_m2_s,
            ion_logit_step=2.0,
            inventory_row_node=compiled.ion_node_count - 1,
            budget=SolveBudget(120, 960, 122, 12),
            wall_s=elec_prot.wall_s_per_temperature,
        )
        polarity = float(compiled.electronic_component.voltage_polarity)

        def sweep_ionic_steady(voltages: np.ndarray) -> np.ndarray:
            currents: np.ndarray = np.empty(voltages.size, dtype=float)
            prev_u: np.ndarray | None = None
            prev_v: float | None = None
            for idx, v_val in enumerate(voltages):
                v_f = float(v_val)
                seed_state: IonicState1D | None = None
                if prev_u is None:
                    dark_init = solve_ionic_steady_state(
                        compiled,
                        voltage_V=v_f,
                        generation_scale=0.0,
                        options=ionic_opts,
                        initial_state=None,
                    )
                    if dark_init.accepted and dark_init.state is not None:
                        try:
                            seeded_e = density_basin_seed(
                                DC1DProblem(compiled.electronic_component, v_f, 1.0),
                                dark_init.state.electronic_state.u,
                            )
                            seed_state = compiled.state(
                                np.r_[
                                    seeded_e,
                                    dark_init.state.u[3 * compiled.node_count :],
                                ],
                                voltage_V=v_f,
                            )
                        except ValueError:
                            seed_state = dark_init.state
                elif prev_v is not None:
                    elec_u = compiled.electronic_component.rebase_coordinates(
                        prev_u[: 3 * compiled.node_count], prev_v, v_f
                    )
                    seed_state = compiled.state(
                        np.r_[elec_u, prev_u[3 * compiled.node_count :]],
                        voltage_V=v_f,
                    )
                sol = solve_ionic_steady_state(
                    compiled,
                    voltage_V=v_f,
                    generation_scale=1.0,
                    options=ionic_opts,
                    initial_state=seed_state,
                )
                if not sol.accepted or sol.state is None or sol.diagnostics is None:
                    raise ElectrothermalSourceError(
                        f"Ionic steady solve failed at V={v_f:.4f} V: {sol.termination}"
                    )
                tot = np.asarray(
                    sol.diagnostics["total_current_density_A_m2"], dtype=float
                )
                currents[idx] = -polarity * float(tot[0])
                prev_u = sol.state.u
                prev_v = v_f
            return currents

        return v_fwd, sweep_ionic_steady(v_fwd), v_rev, sweep_ionic_steady(v_rev)

    # driver == "transient_jv_strict": triangular Backward Euler forward -> reverse
    sub = elec_prot.substeps_per_point
    dv = float(v_fwd[1] - v_fwd[0])
    dt_sub = dv / (elec_prot.scan_rate_V_s * sub)

    if isinstance(compiled, CompiledDevice1D):
        steady_init_opts = SteadyOptions(
            poisson_tolerance=elec_prot.poisson_tolerance,
            continuity_tolerance_A_m2=elec_prot.continuity_tolerance_A_m2,
            contact_log_tolerance=elec_prot.contact_log_tolerance,
        )
        steady0 = solve_steady_state(
            compiled,
            initial_state=None,
            voltage_V=0.0,
            generation_scale=1.0,
            options=steady_init_opts,
        )
        if not steady0.converged:
            raise ElectrothermalSourceError(
                "Initial 0 V steady state did not converge for transient_jv_strict"
            )
        polarity = float(compiled.voltage_polarity)
        j0 = float(-polarity * steady0.current_density_A_m2[0])

        times_list: list[float] = [0.0]
        volts_list: list[float] = [0.0]
        t_cur = 0.0
        if elec_prot.precondition_time_s > 0.0:
            n_pre = max(4, int(math.ceil(elec_prot.precondition_time_s / max(dt_sub, 1e-3))))
            for k in range(1, n_pre + 1):
                t_cur = elec_prot.precondition_time_s * k / n_pre
                times_list.append(t_cur)
                volts_list.append(0.0)

        fwd_sample_indices: list[int] = [len(times_list) - 1]
        for i in range(1, v_fwd.size):
            v_a = float(v_fwd[i - 1])
            v_b = float(v_fwd[i])
            for s in range(1, sub + 1):
                t_cur += dt_sub
                times_list.append(t_cur)
                volts_list.append(v_a + (v_b - v_a) * (s / sub))
            fwd_sample_indices.append(len(times_list) - 1)

        rev_sample_indices: list[int] = [len(times_list) - 1]
        for i in range(1, v_rev.size):
            v_a = float(v_rev[i - 1])
            v_b = float(v_rev[i])
            for s in range(1, sub + 1):
                t_cur += dt_sub
                times_list.append(t_cur)
                volts_list.append(v_a + (v_b - v_a) * (s / sub))
            rev_sample_indices.append(len(times_list) - 1)

        t_arr: np.ndarray = np.asarray(times_list, dtype=float)
        v_arr: np.ndarray = np.asarray(volts_list, dtype=float)
        g_arr: np.ndarray = np.ones_like(t_arr)
        tr_prot = TransientProtocol(times_s=t_arr, voltage_V=v_arr, generation_scale=g_arr)
        tr_opts = TransientOptions(
            poisson_tolerance=elec_prot.poisson_tolerance,
            continuity_tolerance_A_m2=elec_prot.continuity_tolerance_A_m2,
            contact_log_tolerance=elec_prot.contact_log_tolerance,
            potential_step_V=0.25,
            quasi_fermi_step=2.0,
            newton_budget=SolveBudget(60, 480, 62, 12),
            max_steps=t_arr.size + 4,
            wall_s=elec_prot.wall_s_per_temperature,
            predictor="previous",
            convergence_policy=POISSON_CONTACT_BALANCE_V1,
        )
        tr_res = solve_transient(
            compiled,
            steady0,
            tr_prot,
            options=tr_opts,
            initial_options=steady_init_opts,
            method="backward_euler",
        )
        if not tr_res.completed:
            raise ElectrothermalSourceError(
                "transient_jv_strict Backward Euler sweep failed certification"
            )

        step_currents: np.ndarray = np.empty(t_arr.size, dtype=float)
        step_currents[0] = j0
        for idx_step, attempt in enumerate(tr_res.attempts, start=1):
            assert attempt.accounting is not None
            term = np.asarray(
                attempt.accounting["terminal_total_current_density_A_m2"], dtype=float
            )
            step_currents[idx_step] = -polarity * float(term[0])

        j_fwd = step_currents[np.asarray(fwd_sample_indices, dtype=int)]
        j_rev = step_currents[np.asarray(rev_sample_indices, dtype=int)]
        return v_fwd, j_fwd, v_rev, j_rev

    # Ionic transient_jv_strict: fresh dark 0 V equilibrium followed by explicit
    # finite-time 0 V illuminated preconditioning and Backward Euler J-V scan
    ionic_steady_opts = IonicSteadyOptions(
        electronic=SteadyOptions(
            poisson_tolerance=elec_prot.poisson_tolerance,
            continuity_tolerance_A_m2=elec_prot.continuity_tolerance_A_m2,
            contact_log_tolerance=elec_prot.contact_log_tolerance,
            convergence_policy=HISTORICAL_MAX_ROW_V1,
        ),
        inventory_relative_tolerance=1e-11,
        ion_balance_tolerance_m2_s=elec_prot.ion_balance_tolerance_m2_s,
        ion_flux_tolerance_m2_s=elec_prot.ion_balance_tolerance_m2_s,
        ion_logit_step=2.0,
        inventory_row_node=compiled.ion_node_count - 1,
        budget=SolveBudget(120, 960, 122, 12),
        wall_s=elec_prot.wall_s_per_temperature,
    )
    init_sol = solve_ionic_steady_state(
        compiled,
        voltage_V=0.0,
        generation_scale=0.0,
        options=ionic_steady_opts,
        initial_state=None,
    )
    if not init_sol.accepted or init_sol.state is None or init_sol.diagnostics is None:
        raise ElectrothermalSourceError(
            f"Initial dark 0 V ionic steady state failed: {init_sol.termination}"
        )
    cur_state = init_sol.state
    settle_s = (
        elec_prot.precondition_time_s
        if elec_prot.precondition_time_s > 0.0
        else 1e-3
    )
    pre_steps: list[tuple[float, float]] = [
        (1e-3, 2e-5),
        (1e-1, 8e-5),
        (1.0, 2e-4),
    ]
    rem_s = max(settle_s - 3e-4, 2e-4)
    n_rem = max(2, int(math.ceil(rem_s / max(dt_sub, 1e-3))))
    for _ in range(n_rem):
        pre_steps.append((1.0, rem_s / n_rem))

    j_cur = 0.0
    for idx_pre, (g_pre, dt_pre) in enumerate(pre_steps):
        cur_state, j_cur = _step_ionic_be(
            compiled,
            cur_state,
            voltage_V=0.0,
            generation_scale=g_pre,
            dt_s=dt_pre,
            elec_prot=elec_prot,
            predictor="generation_inventory" if idx_pre == 0 else "previous",
        )

    j_fwd = np.empty(v_fwd.size, dtype=float)
    j_fwd[0] = j_cur
    for i in range(1, v_fwd.size):
        v_a = float(v_fwd[i - 1])
        v_b = float(v_fwd[i])
        for s in range(1, sub + 1):
            v_step = v_a + (v_b - v_a) * (s / sub)
            cur_state, j_cur = _step_ionic_be(
                compiled,
                cur_state,
                voltage_V=v_step,
                generation_scale=1.0,
                dt_s=dt_sub,
                elec_prot=elec_prot,
            )
        j_fwd[i] = j_cur

    j_rev = np.empty(v_rev.size, dtype=float)
    j_rev[0] = j_cur
    for i in range(1, v_rev.size):
        v_a = float(v_rev[i - 1])
        v_b = float(v_rev[i])
        for s in range(1, sub + 1):
            v_step = v_a + (v_b - v_a) * (s / sub)
            cur_state, j_cur = _step_ionic_be(
                compiled,
                cur_state,
                voltage_V=v_step,
                generation_scale=1.0,
                dt_s=dt_sub,
                elec_prot=elec_prot,
            )
        j_rev[i] = j_cur

    return v_fwd, j_fwd, v_rev, j_rev


def _extract_operating_point(
    external: ExternalCircuitJVResult,
    op_prot: ElectrothermalOperatingPointProtocol,
) -> tuple[float, float, float]:
    """Extract ``(V_term, J_term, P_term)`` according to ``op_prot``."""
    is_fwd = op_prot.operating_branch == "forward"
    v_branch = external.V_terminal_fwd if is_fwd else external.V_terminal_rev
    j_branch = external.J_terminal_fwd if is_fwd else external.J_terminal_rev
    metrics = external.metrics_fwd if is_fwd else external.metrics_rev

    order = np.argsort(v_branch)
    v_sorted = v_branch[order]
    j_sorted = j_branch[order]

    if op_prot.operating_point == "open_circuit":
        if metrics.open_circuit_voltage_V is None:
            raise ElectrothermalSourceError(
                f"{op_prot.operating_branch} terminal branch does not bracket open circuit"
            )
        return float(metrics.open_circuit_voltage_V), 0.0, 0.0

    if op_prot.operating_point == "sampled_terminal_maximum_power":
        if metrics.open_circuit_voltage_V is None:
            raise ElectrothermalSourceError(
                f"{op_prot.operating_branch} terminal branch does not bracket open circuit"
            )
        mask = (v_branch >= 0.0) & (v_branch <= metrics.open_circuit_voltage_V)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            raise ElectrothermalSourceError(
                f"{op_prot.operating_branch} terminal branch has no power-quadrant sample"
            )
        p_branch = v_branch * j_branch
        idx = int(indices[int(np.argmax(p_branch[indices]))])
        v_sel = float(v_branch[idx])
        j_sel = float(j_branch[idx])
        p_sel = float(p_branch[idx])
        if not math.isfinite(p_sel) or p_sel < 0.0:
            raise ElectrothermalSourceError("non-physical terminal maximum power")
        return v_sel, j_sel, p_sel

    # fixed_load
    if op_prot.fixed_terminal_voltage_V is not None:
        v_req = float(op_prot.fixed_terminal_voltage_V)
        if not (float(v_sorted[0]) <= v_req <= float(v_sorted[-1])):
            raise ElectrothermalSourceError(
                f"fixed_terminal_voltage_V={v_req:.4f} V lies outside terminal sweep "
                f"[{float(v_sorted[0]):.4f}, {float(v_sorted[-1]):.4f}] V"
            )
        j_req = float(np.interp(v_req, v_sorted, j_sorted))
        return v_req, j_req, v_req * j_req

    assert op_prot.load_resistance_ohm_m2 is not None
    r_load = float(op_prot.load_resistance_ohm_m2)
    if metrics.open_circuit_voltage_V is None or metrics.short_circuit_current_A_m2 is None:
        raise ElectrothermalSourceError(
            "load_resistance_ohm_m2 operating point requires bracketed J_sc and V_oc"
        )
    v_hi = float(metrics.open_circuit_voltage_V)

    def load_mismatch(v_val: float) -> float:
        j_interp = float(np.interp(v_val, v_sorted, j_sorted))
        return v_val - r_load * j_interp

    v_load = float(brentq(load_mismatch, 0.0, v_hi, xtol=1e-12, maxiter=100))
    j_load = float(np.interp(v_load, v_sorted, j_sorted))
    return v_load, j_load, v_load * j_load


def solve_electrothermal_operating_point(
    prepared: PreparedDevice,
    thermal_protocol: LumpedThermalProtocol,
    circuit_protocol: ExternalCircuitProtocol | None = None,
    electrical_protocol: ElectrothermalJVProtocol | None = None,
    operating_protocol: ElectrothermalOperatingPointProtocol | None = None,
    *,
    external_temperature_K: object = None,
    temperature_schedule_K: object = None,
) -> ElectrothermalOperatingPointResult:
    """Solve the coupled steady-state electrothermal operating point on ``[T_amb, T_max]``."""
    if external_temperature_K is not None or temperature_schedule_K is not None:
        raise ElectrothermalCapabilityError(
            "electrothermal self-heating is mutually exclusive with an external "
            "temperature override or T(t) schedule"
        )
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("prepared must be a PreparedDevice")
    if not isinstance(thermal_protocol, LumpedThermalProtocol):
        raise TypeError("thermal_protocol must be a LumpedThermalProtocol")

    circ_prot = ExternalCircuitProtocol() if circuit_protocol is None else circuit_protocol
    if not isinstance(circ_prot, ExternalCircuitProtocol):
        raise TypeError("circuit_protocol must be an ExternalCircuitProtocol")

    elec_prot = ElectrothermalJVProtocol() if electrical_protocol is None else electrical_protocol
    if not isinstance(elec_prot, ElectrothermalJVProtocol):
        raise TypeError("electrical_protocol must be an ElectrothermalJVProtocol")

    op_prot = (
        ElectrothermalOperatingPointProtocol()
        if operating_protocol is None
        else operating_protocol
    )
    if not isinstance(op_prot, ElectrothermalOperatingPointProtocol):
        raise TypeError("operating_protocol must be an ElectrothermalOperatingPointProtocol")

    evaluations: list[ElectrothermalTemperatureEvaluation] = []
    external_by_temp: dict[float, ExternalCircuitJVResult] = {}
    eval_by_temp: dict[float, ElectrothermalTemperatureEvaluation] = {}

    def evaluate(temperature_K: float) -> ElectrothermalTemperatureEvaluation:
        t_val = float(temperature_K)
        cached = eval_by_temp.get(t_val)
        if cached is not None:
            return cached

        rebound, compiled = compile_prepared_at_temperature(
            prepared,
            t_val,
            intervals_per_layer=elec_prot.intervals_per_layer,
            ion_domain_layer=elec_prot.ion_domain_layer,
            ion_domain_endpoints=elec_prot.ion_domain_endpoints,
        )
        v_fwd, j_fwd, v_rev, j_rev = run_fresh_jv_at_temperature(compiled, elec_prot)
        ext_res = apply_external_circuit(
            (v_fwd, j_fwd, v_rev, j_rev),
            circ_prot,
            incident_power_W_m2=elec_prot.incident_power_W_m2,
        )
        v_op, j_op, p_op = _extract_operating_point(ext_res, op_prot)
        ledger = thermal_power_ledger(
            t_val,
            thermal_protocol,
            exported_electrical_power_W_m2=p_op,
        )
        record = ElectrothermalTemperatureEvaluation(
            temperature_K=t_val,
            rebound_prepared_sha256=rebound.content_sha256,
            compiled_identity=compiled.identity,
            terminal_voltage_V=v_op,
            terminal_current_A_m2=j_op,
            terminal_power_W_m2=p_op,
            power_balance_residual_W_m2=ledger.net_storage_rate_W_m2,
            external_result_sha256=ext_res.result_sha256,
            fresh_state_initialized=True,
            source_certified=True,
        )
        evaluations.append(record)
        eval_by_temp[t_val] = record
        external_by_temp[t_val] = ext_res
        return record

    t_amb = thermal_protocol.ambient_temperature_K
    t_max = thermal_protocol.maximum_temperature_K
    p_tol = op_prot.power_residual_tolerance_W_m2

    lo_eval = evaluate(t_amb)
    root_iterations = 0
    if abs(lo_eval.power_balance_residual_W_m2) <= p_tol:
        t_star = t_amb
    else:
        if lo_eval.power_balance_residual_W_m2 < 0.0:
            raise ElectrothermalConvergenceError(
                f"Electrothermal storage rate is negative at T_amb={t_amb:.3f} K "
                f"({lo_eval.power_balance_residual_W_m2:.4g} W/m^2)"
            )
        hi_eval = evaluate(t_max)
        if abs(hi_eval.power_balance_residual_W_m2) <= p_tol:
            t_star = t_max
        else:
            if hi_eval.power_balance_residual_W_m2 > 0.0:
                raise ElectrothermalConvergenceError(
                    f"Electrothermal root is not bracketed below T_max={t_max:.3f} K "
                    f"(f(T_max)={hi_eval.power_balance_residual_W_m2:.4g} W/m^2 > 0)"
                )
            root_val, root_info = brentq(
                lambda temp: evaluate(float(temp)).power_balance_residual_W_m2,
                t_amb,
                t_max,
                xtol=op_prot.temperature_absolute_tolerance_K,
                rtol=4.0 * float(np.finfo(float).eps),
                maxiter=op_prot.maximum_root_iterations,
                full_output=True,
                disp=False,
            )
            if not root_info.converged:
                raise ElectrothermalConvergenceError(
                    "Electrothermal Brent root finder did not converge"
                )
            t_star = float(root_val)
            root_iterations = int(root_info.iterations)

    final_eval = evaluate(t_star)
    if abs(final_eval.power_balance_residual_W_m2) > p_tol:
        raise ElectrothermalConvergenceError(
            f"Electrothermal power residual {abs(final_eval.power_balance_residual_W_m2):.3e} W/m^2 "
            f"exceeds {p_tol:.3e} W/m^2"
        )
    final_ext = external_by_temp[final_eval.temperature_K]
    final_ledger = thermal_power_ledger(
        final_eval.temperature_K,
        thermal_protocol,
        exported_electrical_power_W_m2=final_eval.terminal_power_W_m2,
    )
    frozen_evals = tuple(evaluations)
    mapping_sha = _canonical_sha256(
        {
            "base_prepared_sha256": prepared.content_sha256,
            "thermal_protocol": asdict(thermal_protocol),
            "circuit_protocol": asdict(circ_prot),
            "electrical_protocol_sha256": elec_prot.sha256,
            "operating_protocol_sha256": op_prot.sha256,
            "operating_temperature_K": final_eval.temperature_K.hex(),
            "final_external_sha256": final_ext.result_sha256,
            "evaluation_count": len(frozen_evals),
        }
    )
    return ElectrothermalOperatingPointResult(
        operating_temperature_K=final_eval.temperature_K,
        temperature_rise_K=final_eval.temperature_K - t_amb,
        terminal_voltage_V=final_eval.terminal_voltage_V,
        terminal_current_A_m2=final_eval.terminal_current_A_m2,
        terminal_power_W_m2=final_eval.terminal_power_W_m2,
        power_balance_residual_W_m2=final_eval.power_balance_residual_W_m2,
        thermal_ledger=final_ledger,
        final_external_result=final_ext,
        temperature_evaluations=frozen_evals,
        electrical_evaluations=len(frozen_evals),
        root_iterations=root_iterations,
        base_prepared_sha256=prepared.content_sha256,
        thermal_protocol=thermal_protocol,
        circuit_protocol=circ_prot,
        electrical_protocol=elec_prot,
        operating_protocol=op_prot,
        time_domain_temperature_qualified=bool(TIME_DOMAIN_TEMPERATURE_QUALIFIED),
        mapping_sha256=mapping_sha,
        certified=True,
    )
