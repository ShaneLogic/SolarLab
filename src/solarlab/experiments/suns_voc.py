"""Suns-V_oc characterization on the unified 1D DC residual.

Each illumination level ``X > 0`` solves genuine DC operating points on the
assembled drift-diffusion (or coupled ionic-electronic) equations rather than
finite-time transient relaxation:

1. Short-circuit operating point at ``V = 0`` with ``generation_scale = X``
   (optionally subtracting the matched dark ``V = 0`` baseline offset).
2. Open-circuit root ``V_oc(X)`` refined by ``solve_direct_voc`` (or bounded
   coupled ionic DC bisection) inside an accepted sign-change voltage bracket.

Non-converged or unbracketed points record ``np.nan`` and an explicit entry in
``point_reasons``; ``0.0`` is never substituted for a missing operating point.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem
from solarlab.equations.implementation import DCProblem, use_assembled
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, IonicState1D
from solarlab.experiments.jv.operating_points import (
    DirectVocError,
    VocBracketError,
    solve_direct_voc,
)
from solarlab.experiments.jv.steady import SteadyJVResult, steady_jv_metrics
from solarlab.reproducibility.job_provenance import current_provenance
from solarlab.solvers.nonlinear.ionic_1d import (
    IonicSteadyOptions,
    IonicSteadyResult,
    solve_ionic_steady_state,
)
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    SteadyState1D,
    SteadyStateError,
    density_basin_seed,
    solve_steady_state,
)

__all__ = [
    "DEFAULT_SUNS",
    "SunsVocOptions",
    "SunsVocResult",
    "compute_pseudo_ff",
    "compute_suns_voc_ideality",
    "default_ionic_steady_options",
    "run_suns_voc",
    "solve_ionic_dc_operating_point",
]

BOLTZMANN_J_K = 1.380649e-23
DEFAULT_SUNS: tuple[float, ...] = (0.01, 0.1, 1.0, 5.0, 10.0)
ProgressCallback = Callable[[str, int, int, str], None]


def default_ionic_steady_options(
    ion_node_count: int,
    *,
    electronic: SteadyOptions | None = None,
) -> IonicSteadyOptions:
    """Explicit production coupled ionic DC controls when none are supplied."""
    elec = (
        SteadyOptions(
            continuity_tolerance_A_m2=1e-7,
            current_spread_tolerance_A_m2=1e-7,
        )
        if electronic is None
        else electronic
    )
    return IonicSteadyOptions(
        electronic=elec,
        inventory_relative_tolerance=1e-12,
        ion_balance_tolerance_m2_s=1e6,
        ion_flux_tolerance_m2_s=1e6,
        ion_logit_step=1.0,
        inventory_row_node=max(int(ion_node_count) - 1, 0),
        budget=SolveBudget(60, 600, 60, 24),
        wall_s=30.0,
    )


def _ro_float(value: Any) -> np.ndarray:
    arr = np.array(value, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


def _ro_bool(value: Any) -> np.ndarray:
    arr = np.array(value, dtype=np.bool_, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class SunsVocOptions:
    """Explicit numerical controls for DC Suns-V_oc sweeps."""

    steady: SteadyOptions = field(
        default_factory=lambda: SteadyOptions(
            continuity_tolerance_A_m2=1e-7,
            current_spread_tolerance_A_m2=1e-7,
        )
    )
    ionic_steady: IonicSteadyOptions | None = None
    voc_voltage_tol_V: float = 1e-6
    voc_current_tol_A_m2: float = 1e-6
    voc_bracket_step_V: float = 0.05
    voc_bracket_max_V: float = 3.0
    max_voc_solve_calls: int = 64
    max_continuation_subdivisions: int = 16
    subtract_dark_offset_at_jsc: bool = True
    require_all_bracketed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.steady, SteadyOptions):
            raise ValueError("steady must be an explicit SteadyOptions instance")
        if self.ionic_steady is not None and not isinstance(
            self.ionic_steady, IonicSteadyOptions
        ):
            raise ValueError(
                "ionic_steady must be None or an explicit IonicSteadyOptions"
            )
        for name, value in (
            ("voc_voltage_tol_V", self.voc_voltage_tol_V),
            ("voc_current_tol_A_m2", self.voc_current_tol_A_m2),
            ("voc_bracket_step_V", self.voc_bracket_step_V),
            ("voc_bracket_max_V", self.voc_bracket_max_V),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.voc_bracket_step_V > self.voc_bracket_max_V:
            raise ValueError("voc_bracket_step_V cannot exceed voc_bracket_max_V")
        if type(self.max_voc_solve_calls) is not int or self.max_voc_solve_calls < 4:
            raise ValueError("max_voc_solve_calls must be an integer >= 4")
        if (
            type(self.max_continuation_subdivisions) is not int
            or self.max_continuation_subdivisions < 2
        ):
            raise ValueError("max_continuation_subdivisions must be an integer >= 2")


@dataclass(frozen=True)
class SunsVocResult:
    """Genuine DC Suns-V_oc sweep, ideality diagnostics, and Sinton pseudo-JV."""

    suns_levels: np.ndarray
    V_oc_V: np.ndarray
    J_sc_A_m2: np.ndarray
    dark_short_circuit_A_m2: float
    converged_mask: np.ndarray
    voc_bracketed_mask: np.ndarray
    point_reasons: tuple[str | None, ...]
    local_ideality_factor: np.ndarray
    ideality_factor_fit: float
    ideality_r_squared: float
    J_pseudo_V: np.ndarray
    J_pseudo_A_m2: np.ndarray
    pseudo_FF: float
    pseudo_ff_reason: str | None
    reference_index: int
    max_current_spread_A_m2: float
    temperature_K: float
    voltage_polarity: float
    compiled_identity: str
    solver_controls: dict[str, Any]
    job_provenance: dict[str, Any] | None = field(default_factory=current_provenance)

    def __post_init__(self) -> None:
        for name in (
            "suns_levels",
            "V_oc_V",
            "J_sc_A_m2",
            "local_ideality_factor",
            "J_pseudo_V",
            "J_pseudo_A_m2",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))
        for mask_name in ("converged_mask", "voc_bracketed_mask"):
            object.__setattr__(self, mask_name, _ro_bool(getattr(self, mask_name)))

    @property
    def suns(self) -> np.ndarray:
        return self.suns_levels

    @property
    def V_oc(self) -> np.ndarray:
        return self.V_oc_V

    @property
    def J_sc(self) -> np.ndarray:
        return self.J_sc_A_m2

    @property
    def J_pseudo_J(self) -> np.ndarray:
        return self.J_pseudo_A_m2

    @property
    def J_spread_max(self) -> float:
        return self.max_current_spread_A_m2

    @property
    def completed(self) -> bool:
        return bool(np.all(self.converged_mask & self.voc_bracketed_mask))


def compute_suns_voc_ideality(
    suns_levels: Sequence[float] | np.ndarray,
    V_oc_V: Sequence[float] | np.ndarray,
    *,
    temperature_K: float,
    valid_mask: Sequence[bool] | np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Compute local ``n_id(X)`` and global log-linear fit over valid points."""
    suns: np.ndarray = np.asarray(suns_levels, dtype=float)
    voc: np.ndarray = np.asarray(V_oc_V, dtype=float)
    if suns.ndim != 1 or voc.shape != suns.shape or suns.size == 0:
        raise ValueError("suns_levels and V_oc_V must be matching 1D non-empty arrays")
    if not math.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError("temperature_K must be finite and positive")
    v_thermal = BOLTZMANN_J_K * float(temperature_K) / ELEMENTARY_CHARGE_C

    mask = (
        np.isfinite(suns) & (suns > 0.0) & np.isfinite(voc)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
        & np.isfinite(suns)
        & (suns > 0.0)
        & np.isfinite(voc)
    )
    local: np.ndarray = np.full(suns.shape, np.nan, dtype=float)
    if int(np.count_nonzero(mask)) < 2:
        return local, float("nan"), float("nan")

    ln_x = np.log(suns[mask])
    y = voc[mask]
    if np.any(np.diff(ln_x) <= 0.0):
        raise ValueError("valid suns_levels must be strictly ascending")

    local_valid = np.gradient(y, ln_x) / v_thermal
    local[mask] = local_valid

    slope, intercept = np.polyfit(ln_x, y, 1)
    n_fit = float(slope / v_thermal)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - (slope * ln_x + intercept)) ** 2))
    r_sq = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 1.0
    return local, n_fit, r_sq


def compute_pseudo_ff(
    J_pseudo_V: Sequence[float] | np.ndarray,
    J_pseudo_A_m2: Sequence[float] | np.ndarray,
    *,
    V_oc_ref_V: float,
    J_sc_ref_A_m2: float,
) -> tuple[float, str | None]:
    """Compute Sinton pseudo-FF on the valid power-quadrant points."""
    if (
        not math.isfinite(V_oc_ref_V)
        or not math.isfinite(J_sc_ref_A_m2)
        or V_oc_ref_V <= 0.0
        or J_sc_ref_A_m2 <= 0.0
    ):
        return float("nan"), "missing_or_nonpositive_reference"

    v_arr: np.ndarray = np.asarray(J_pseudo_V, dtype=float)
    j_arr: np.ndarray = np.asarray(J_pseudo_A_m2, dtype=float)
    finite = np.isfinite(v_arr) & np.isfinite(j_arr)
    mask = (
        finite
        & (v_arr > 0.0)
        & (v_arr < V_oc_ref_V)
        & (j_arr > 0.0)
        & (j_arr < J_sc_ref_A_m2)
    )
    if not np.any(mask):
        return float("nan"), "no_valid_power_quadrant_points"

    v_quad = np.concatenate(([0.0], v_arr[mask], [V_oc_ref_V]))
    j_quad = np.concatenate(([J_sc_ref_A_m2], j_arr[mask], [0.0]))
    order = np.argsort(v_quad)
    v_sorted = v_quad[order]
    j_sorted = j_quad[order]

    v_grid = np.linspace(0.0, V_oc_ref_V, 500)
    j_grid = np.interp(v_grid, v_sorted, j_sorted)
    p_max = float(np.max(v_grid * j_grid))
    return float(p_max / (V_oc_ref_V * J_sc_ref_A_m2)), None


def _normalize_suns_levels(suns_levels: Sequence[float] | np.ndarray) -> np.ndarray:
    raw = [float(level) for level in suns_levels]
    if not raw:
        raise ValueError("suns_levels must be non-empty")
    if any(not math.isfinite(val) or val <= 0.0 for val in raw):
        raise ValueError(f"suns_levels must be finite and positive, got {raw}")
    return np.array(sorted(set(raw)), dtype=float)


def _electronic_output_and_spread(
    solution: SteadyState1D, polarity: float
) -> tuple[float, float]:
    currents = np.asarray(solution.current_density_A_m2, dtype=float)
    return float(-polarity * currents[0]), float(np.ptp(currents))


def _solve_electronic_step_with_continuation(
    compiled: CompiledDevice1D,
    seed: SteadyState1D,
    target_V: float,
    generation_scale: float,
    options: SteadyOptions,
    max_subdivisions: int,
) -> SteadyState1D:
    """Advance from an accepted DC solution to ``target_V`` with bounded halving."""
    v_curr = seed.voltage_V
    current_sol = seed
    try:
        sol = solve_steady_state(
            compiled,
            voltage_V=target_V,
            generation_scale=generation_scale,
            initial_state=current_sol.state,
            options=options,
        )
        if sol.converged:
            return sol
    except SteadyStateError as first_err:
        last_err = first_err
    else:
        last_err = SteadyStateError(f"unconverged at {target_V} V", sol.newton)

    subdivisions = 2
    while subdivisions <= max_subdivisions:
        v_base = v_curr
        step_sol = current_sol
        chain_ok = True
        for step in range(1, subdivisions + 1):
            v_mid = v_base + (target_V - v_base) * (step / subdivisions)
            try:
                mid_sol = solve_steady_state(
                    compiled,
                    voltage_V=v_mid,
                    generation_scale=generation_scale,
                    initial_state=step_sol.state,
                    options=options,
                )
            except SteadyStateError as err:
                last_err = err
                chain_ok = False
                break
            if not mid_sol.converged:
                last_err = SteadyStateError(f"unconverged at {v_mid} V", mid_sol.newton)
                chain_ok = False
                break
            step_sol = mid_sol
            current_sol = step_sol
            v_curr = v_mid
        if chain_ok:
            return step_sol
        subdivisions *= 2
    raise last_err


def _solve_electronic_voc_point(
    compiled: CompiledDevice1D,
    suns: float,
    sol_zero: SteadyState1D,
    warm_voc_V: float,
    options: SunsVocOptions,
) -> tuple[float, bool, bool, str | None, float]:
    """Bracket and refine ``V_oc(suns)`` via ``solve_direct_voc``."""
    polarity = float(compiled.voltage_polarity)
    j_zero, spread_max = _electronic_output_and_spread(sol_zero, polarity)
    if abs(j_zero) <= options.voc_current_tol_A_m2:
        return 0.0, True, True, None, spread_max
    if j_zero < 0.0:
        return (
            float("nan"),
            True,
            False,
            "voc_not_bracketed:nonpositive_short_circuit_current",
            spread_max,
        )

    low_sol = sol_zero
    step_v = options.voc_bracket_step_V
    v_probe = max(step_v, warm_voc_V - step_v)
    if v_probe > step_v:
        # Step smoothly from 0 V to just below warm_voc_V so Newton stays in basin.
        n_coarse = max(1, int(math.ceil(v_probe / max(step_v, 0.1))))
        for idx in range(1, n_coarse + 1):
            v_next = v_probe * (idx / n_coarse)
            try:
                cand = _solve_electronic_step_with_continuation(
                    compiled,
                    low_sol,
                    v_next,
                    suns,
                    options.steady,
                    options.max_continuation_subdivisions,
                )
            except SteadyStateError as exc:
                return (
                    float("nan"),
                    False,
                    False,
                    f"voc_not_converged:{exc}",
                    spread_max,
                )
            j_cand, sp_cand = _electronic_output_and_spread(cand, polarity)
            spread_max = max(spread_max, sp_cand)
            if j_cand <= 0.0:
                high_sol = cand
                break
            low_sol = cand
        else:
            high_sol = None
    else:
        high_sol = None

    v_cursor = low_sol.voltage_V
    while high_sol is None and v_cursor < options.voc_bracket_max_V - 1e-12:
        v_next = min(options.voc_bracket_max_V, v_cursor + step_v)
        try:
            cand = _solve_electronic_step_with_continuation(
                compiled,
                low_sol,
                v_next,
                suns,
                options.steady,
                options.max_continuation_subdivisions,
            )
        except SteadyStateError as exc:
            return (
                float("nan"),
                False,
                False,
                f"voc_not_converged:{exc}",
                spread_max,
            )
        j_cand, sp_cand = _electronic_output_and_spread(cand, polarity)
        spread_max = max(spread_max, sp_cand)
        if j_cand <= 0.0:
            high_sol = cand
            break
        low_sol = cand
        v_cursor = v_next

    if high_sol is None:
        return (
            float("nan"),
            True,
            False,
            f"voc_not_bracketed:exceeded_max_voltage_{options.voc_bracket_max_V:.4g}V",
            spread_max,
        )

    voltages = np.array([low_sol.voltage_V, high_sol.voltage_V], dtype=float)
    currents = np.array(
        [low_sol.current_density_A_m2, high_sol.current_density_A_m2], dtype=float
    )
    outputs = -polarity * currents[:, 0]
    bracket = SteadyJVResult(
        voltage_V=voltages,
        current_density_A_m2=currents,
        output_current_A_m2=outputs,
        solutions=(low_sol, high_sol),
        metrics=steady_jv_metrics(voltages, outputs),
        generation_scale=suns,
        incident_power_W_m2=None,
        voltage_polarity=polarity,
        compiled_identity=compiled.identity,
        nodes_m=compiled.nodes_m,
        solver_controls=asdict(options.steady),
    )
    try:
        refined = solve_direct_voc(
            compiled,
            bracket,
            voltage_tolerance_V=options.voc_voltage_tol_V,
            max_solve_calls=options.max_voc_solve_calls,
            max_continuation_subdivisions=options.max_continuation_subdivisions,
            options=options.steady,
        )
    except (VocBracketError, DirectVocError) as exc:
        return float("nan"), False, True, f"voc_refinement_failed:{exc}", spread_max

    _, sp_voc = _electronic_output_and_spread(refined.solution, polarity)
    return float(refined.voltage_V), True, True, None, max(spread_max, sp_voc)


def _dc_problem(
    device: CompiledDevice1D, voltage_V: float, generation_scale: float
) -> DCProblem:
    if use_assembled():
        return AssembledDC1DProblem(device, voltage_V, generation_scale)
    return DC1DProblem(device, voltage_V, generation_scale)


def _rebase_ionic_state(
    compiled: CompiledIonicDevice1D, state: IonicState1D, voltage_V: float
) -> IonicState1D:
    if state.voltage_V == voltage_V:
        return state
    coords = state.u.copy()
    n_elec = 3 * compiled.node_count
    coords[:n_elec] = compiled.electronic_component.rebase_coordinates(
        coords[:n_elec], state.voltage_V, voltage_V
    )
    return compiled.state(coords, voltage_V=voltage_V)


def _solve_ionic_voltage_step(
    compiled: CompiledIonicDevice1D,
    seed_state: IonicState1D,
    target_V: float,
    generation_scale: float,
    ionic_controls: IonicSteadyOptions,
    max_subdivisions: int = 16,
) -> IonicSteadyResult:
    """Advance coupled ionic DC from ``seed_state`` to ``target_V`` with bounded halving."""
    v_curr = float(seed_state.voltage_V)
    curr_state = seed_state
    cand = solve_ionic_steady_state(
        compiled,
        voltage_V=target_V,
        generation_scale=generation_scale,
        options=ionic_controls,
        initial_state=_rebase_ionic_state(compiled, curr_state, target_V),
    )
    if cand.accepted and cand.state is not None:
        return cand

    subdivisions = 2
    last_failed = cand
    while subdivisions <= max_subdivisions:
        step_state = curr_state
        v_base = v_curr
        ok_chain = True
        for step in range(1, subdivisions + 1):
            v_mid = v_base + (target_V - v_base) * (step / subdivisions)
            mid_res = solve_ionic_steady_state(
                compiled,
                voltage_V=v_mid,
                generation_scale=generation_scale,
                options=ionic_controls,
                initial_state=_rebase_ionic_state(compiled, step_state, v_mid),
            )
            last_failed = mid_res
            if not mid_res.accepted or mid_res.state is None:
                ok_chain = False
                break
            step_state = mid_res.state
            curr_state = step_state
            v_curr = v_mid
        if ok_chain:
            return last_failed
        subdivisions *= 2
    return last_failed


def solve_ionic_dc_operating_point(
    compiled: CompiledIonicDevice1D,
    *,
    voltage_V: float = 0.0,
    generation_scale: float = 0.0,
    options: IonicSteadyOptions,
    base_state: IonicState1D | None = None,
    base_generation_scale: float = 0.0,
    max_subdivisions: int = 16,
) -> IonicSteadyResult:
    """Solve a coupled ionic DC operating point using physical-density light seeding."""
    target_v = float(voltage_V)
    target_gen = float(generation_scale)
    n_elec = 3 * compiled.node_count
    max_gen_source = float(
        np.max(np.abs(compiled.electronic_component.generation.source_m3_s))
    )

    if base_state is None:
        if target_gen == 0.0 and target_v == 0.0:
            return solve_ionic_steady_state(
                compiled, voltage_V=0.0, generation_scale=0.0, options=options
            )
        dark_zero = solve_ionic_steady_state(
            compiled, voltage_V=0.0, generation_scale=0.0, options=options
        )
        if not dark_zero.accepted or dark_zero.state is None:
            return dark_zero
        curr_state = dark_zero.state
        curr_gen = 0.0
        last_res: IonicSteadyResult | None = dark_zero
    else:
        curr_state = base_state
        curr_gen = float(base_generation_scale)
        last_res = None

    if target_gen != curr_gen:
        if target_gen == 0.0 or max_gen_source == 0.0:
            res_gen = solve_ionic_steady_state(
                compiled,
                voltage_V=float(curr_state.voltage_V),
                generation_scale=target_gen,
                options=options,
                initial_state=curr_state,
            )
            if not res_gen.accepted or res_gen.state is None:
                return res_gen
            curr_state = res_gen.state
            curr_gen = target_gen
            last_res = res_gen
        else:
            light_steps = (
                [frac * target_gen for frac in (1e-8, 1e-6, 1e-4, 1e-2, 1.0)]
                if curr_gen == 0.0
                else [target_gen]
            )
            for light in light_steps:
                v_hold = float(curr_state.voltage_V)
                prob = _dc_problem(compiled.electronic_component, v_hold, light)
                try:
                    seeded_elec = density_basin_seed(prob, curr_state.u[:n_elec])
                    joined = np.r_[seeded_elec, curr_state.u[n_elec:]]
                    seed_ion = compiled.state(joined, voltage_V=v_hold)
                except ValueError:
                    seed_ion = curr_state
                res_light = solve_ionic_steady_state(
                    compiled,
                    voltage_V=v_hold,
                    generation_scale=light,
                    options=options,
                    initial_state=seed_ion,
                )
                if not res_light.accepted or res_light.state is None:
                    return res_light
                curr_state = res_light.state
                curr_gen = light
                last_res = res_light

    v_curr = float(curr_state.voltage_V)
    if v_curr == target_v and last_res is not None:
        return last_res

    max_step = max(float(options.electronic.max_bias_step_V), 0.02)
    n_bias = max(1, int(math.ceil(abs(target_v - v_curr) / max_step)))
    res_bias: IonicSteadyResult | None = None
    for idx in range(1, n_bias + 1):
        v_step = v_curr + (target_v - v_curr) * (idx / n_bias)
        res_bias = _solve_ionic_voltage_step(
            compiled,
            curr_state,
            v_step,
            target_gen,
            options,
            max_subdivisions=max_subdivisions,
        )
        if not res_bias.accepted or res_bias.state is None:
            return res_bias
        curr_state = res_bias.state
    assert res_bias is not None
    return res_bias


def _ionic_output_and_spread(
    result: IonicSteadyResult, polarity: float
) -> tuple[float, float]:
    assert result.diagnostics is not None
    currents = np.asarray(result.diagnostics["total_current_density_A_m2"], dtype=float)
    return float(-polarity * currents[0]), float(np.ptp(currents))


def _solve_ionic_voc_point(
    compiled: CompiledIonicDevice1D,
    suns: float,
    sol_zero: IonicSteadyResult,
    warm_voc_V: float,
    options: SunsVocOptions,
    ionic_controls: IonicSteadyOptions,
) -> tuple[float, bool, bool, str | None, float]:
    """Bracket and bisect coupled ionic DC ``V_oc(suns)``."""
    polarity = float(compiled.electronic_component.voltage_polarity)
    j_zero, spread_max = _ionic_output_and_spread(sol_zero, polarity)
    if abs(j_zero) <= options.voc_current_tol_A_m2:
        return 0.0, True, True, None, spread_max
    if j_zero < 0.0:
        return (
            float("nan"),
            True,
            False,
            "voc_not_bracketed:nonpositive_short_circuit_current",
            spread_max,
        )

    assert sol_zero.state is not None
    low_state = sol_zero.state
    low_v = 0.0
    low_j = j_zero
    high_state: IonicState1D | None = None
    high_v = 0.0
    high_j = 0.0

    step_v = options.voc_bracket_step_V
    v_cursor = 0.0
    target_start = max(0.0, warm_voc_V - step_v)
    if target_start > step_v:
        n_coarse = max(1, int(math.ceil(target_start / max(step_v, 0.1))))
        for idx in range(1, n_coarse + 1):
            v_next = target_start * (idx / n_coarse)
            cand = _solve_ionic_voltage_step(
                compiled,
                low_state,
                v_next,
                suns,
                ionic_controls,
                max_subdivisions=options.max_continuation_subdivisions,
            )
            if not cand.accepted or cand.state is None:
                reason = cand.error or cand.termination
                return (
                    float("nan"),
                    False,
                    False,
                    f"voc_not_converged:{reason}",
                    spread_max,
                )
            j_cand, sp_cand = _ionic_output_and_spread(cand, polarity)
            spread_max = max(spread_max, sp_cand)
            if j_cand <= 0.0:
                high_state, high_v, high_j = cand.state, v_next, j_cand
                break
            low_state, low_v, low_j = cand.state, v_next, j_cand
            v_cursor = v_next

    while high_state is None and v_cursor < options.voc_bracket_max_V - 1e-12:
        v_next = min(options.voc_bracket_max_V, v_cursor + step_v)
        cand = _solve_ionic_voltage_step(
            compiled,
            low_state,
            v_next,
            suns,
            ionic_controls,
            max_subdivisions=options.max_continuation_subdivisions,
        )
        if not cand.accepted or cand.state is None:
            reason = cand.error or cand.termination
            return (
                float("nan"),
                False,
                False,
                f"voc_not_converged:{reason}",
                spread_max,
            )
        j_cand, sp_cand = _ionic_output_and_spread(cand, polarity)
        spread_max = max(spread_max, sp_cand)
        if j_cand <= 0.0:
            high_state, high_v, high_j = cand.state, v_next, j_cand
            break
        low_state, low_v, low_j = cand.state, v_next, j_cand
        v_cursor = v_next

    if high_state is None:
        return (
            float("nan"),
            True,
            False,
            f"voc_not_bracketed:exceeded_max_voltage_{options.voc_bracket_max_V:.4g}V",
            spread_max,
        )

    calls = 0
    while (high_v - low_v) > options.voc_voltage_tol_V:
        if calls >= options.max_voc_solve_calls:
            return (
                float("nan"),
                False,
                True,
                "voc_refinement_failed:exhausted_solve_calls",
                spread_max,
            )
        calls += 1
        mid_v = 0.5 * (low_v + high_v)
        seed_base = low_state if (mid_v - low_v) <= (high_v - mid_v) else high_state
        mid_res = _solve_ionic_voltage_step(
            compiled,
            seed_base,
            mid_v,
            suns,
            ionic_controls,
            max_subdivisions=options.max_continuation_subdivisions,
        )
        if not mid_res.accepted or mid_res.state is None:
            reason = mid_res.error or mid_res.termination
            return (
                float("nan"),
                False,
                True,
                f"voc_refinement_failed:{reason}",
                spread_max,
            )
        j_mid, sp_mid = _ionic_output_and_spread(mid_res, polarity)
        spread_max = max(spread_max, sp_mid)
        if abs(j_mid) <= options.voc_current_tol_A_m2:
            return mid_v, True, True, None, spread_max
        if j_mid > 0.0:
            low_state, low_v, low_j = mid_res.state, mid_v, j_mid
        else:
            high_state, high_v, high_j = mid_res.state, mid_v, j_mid

    best_v = low_v if abs(low_j) <= abs(high_j) else high_v
    return float(best_v), True, True, None, spread_max


def run_suns_voc(
    compiled: CompiledDevice1D | CompiledIonicDevice1D,
    suns_levels: Sequence[float] | np.ndarray = DEFAULT_SUNS,
    *,
    options: SunsVocOptions | None = None,
    progress: ProgressCallback | None = None,
) -> SunsVocResult:
    """Run a genuine DC Suns-V_oc sweep on an electronic or coupled ionic device."""
    opts = SunsVocOptions() if options is None else options
    suns_sorted = _normalize_suns_levels(suns_levels)
    n_pts = int(suns_sorted.size)

    voc_arr: np.ndarray = np.full(n_pts, np.nan, dtype=float)
    jsc_arr: np.ndarray = np.full(n_pts, np.nan, dtype=float)
    conv_mask: np.ndarray = np.zeros(n_pts, dtype=bool)
    brack_mask: np.ndarray = np.zeros(n_pts, dtype=bool)
    reasons: list[str | None] = [None] * n_pts
    spread_max = 0.0
    warm_voc = 0.0

    if isinstance(compiled, CompiledDevice1D):
        polarity = float(compiled.voltage_polarity)
        temperature_K = float(
            compiled.thermal_voltage_V * ELEMENTARY_CHARGE_C / BOLTZMANN_J_K
        )
        controls_record: dict[str, Any] = {
            "mode": "electronic_dc",
            "steady": asdict(opts.steady),
            "voc_voltage_tol_V": opts.voc_voltage_tol_V,
            "voc_bracket_step_V": opts.voc_bracket_step_V,
            "voc_bracket_max_V": opts.voc_bracket_max_V,
        }
        j_dark = 0.0
        if opts.subtract_dark_offset_at_jsc:
            try:
                dark_sol = solve_steady_state(
                    compiled, voltage_V=0.0, generation_scale=0.0, options=opts.steady
                )
                if dark_sol.converged:
                    j_dark, sp_dark = _electronic_output_and_spread(dark_sol, polarity)
                    spread_max = max(spread_max, sp_dark)
                else:
                    j_dark = float("nan")
            except SteadyStateError:
                j_dark = float("nan")

        for k, suns in enumerate(suns_sorted):
            try:
                sol_zero = solve_steady_state(
                    compiled,
                    voltage_V=0.0,
                    generation_scale=float(suns),
                    options=opts.steady,
                )
                if not sol_zero.converged:
                    reasons[k] = "jsc_not_converged:unconverged_steady_result"
                    continue
            except SteadyStateError as exc:
                reasons[k] = f"jsc_not_converged:{exc}"
                continue

            j_tot, sp_zero = _electronic_output_and_spread(sol_zero, polarity)
            spread_max = max(spread_max, sp_zero)
            jsc_arr[k] = (
                j_tot - j_dark
                if (opts.subtract_dark_offset_at_jsc and math.isfinite(j_dark))
                else j_tot
            )
            voc_k, conv_k, brack_k, reason_k, sp_voc = _solve_electronic_voc_point(
                compiled, float(suns), sol_zero, warm_voc, opts
            )
            spread_max = max(spread_max, sp_voc)
            voc_arr[k] = voc_k
            conv_mask[k] = conv_k
            brack_mask[k] = brack_k
            reasons[k] = reason_k
            if conv_k and brack_k and math.isfinite(voc_k):
                warm_voc = max(warm_voc, voc_k)
            if progress is not None:
                progress(
                    "suns_voc",
                    k + 1,
                    n_pts,
                    f"X={suns:.3g}, V_oc={voc_k:.4f} V, J_sc={jsc_arr[k]:.2f} A/m^2",
                )

    elif isinstance(compiled, CompiledIonicDevice1D):
        polarity = float(compiled.electronic_component.voltage_polarity)
        temperature_K = float(
            compiled.electronic_component.thermal_voltage_V
            * ELEMENTARY_CHARGE_C
            / BOLTZMANN_J_K
        )
        ionic_controls = (
            default_ionic_steady_options(
                compiled.ion_node_count, electronic=opts.steady
            )
            if opts.ionic_steady is None
            else opts.ionic_steady
        )
        controls_record = {
            "mode": "ionic_dc",
            "ionic_steady": asdict(ionic_controls),
            "voc_voltage_tol_V": opts.voc_voltage_tol_V,
            "voc_bracket_step_V": opts.voc_bracket_step_V,
            "voc_bracket_max_V": opts.voc_bracket_max_V,
        }
        j_dark = 0.0
        prev_ionic_zero: IonicState1D | None = None
        prev_suns = 0.0
        if opts.subtract_dark_offset_at_jsc:
            dark_res = solve_ionic_steady_state(
                compiled,
                voltage_V=0.0,
                generation_scale=0.0,
                options=ionic_controls,
            )
            if dark_res.accepted and dark_res.state is not None:
                j_dark, sp_dark = _ionic_output_and_spread(dark_res, polarity)
                spread_max = max(spread_max, sp_dark)
                prev_ionic_zero = dark_res.state
                prev_suns = 0.0
            else:
                j_dark = float("nan")

        for k, suns in enumerate(suns_sorted):
            sol_zero_ion = solve_ionic_dc_operating_point(
                compiled,
                voltage_V=0.0,
                generation_scale=float(suns),
                options=ionic_controls,
                base_state=prev_ionic_zero,
                base_generation_scale=prev_suns,
                max_subdivisions=opts.max_continuation_subdivisions,
            )
            if not sol_zero_ion.accepted or sol_zero_ion.state is None:
                reason = sol_zero_ion.error or sol_zero_ion.termination
                reasons[k] = f"jsc_not_converged:{reason}"
                continue

            prev_ionic_zero = sol_zero_ion.state
            prev_suns = float(suns)
            j_tot, sp_zero = _ionic_output_and_spread(sol_zero_ion, polarity)
            spread_max = max(spread_max, sp_zero)
            jsc_arr[k] = (
                j_tot - j_dark
                if (opts.subtract_dark_offset_at_jsc and math.isfinite(j_dark))
                else j_tot
            )
            voc_k, conv_k, brack_k, reason_k, sp_voc = _solve_ionic_voc_point(
                compiled, float(suns), sol_zero_ion, warm_voc, opts, ionic_controls
            )
            spread_max = max(spread_max, sp_voc)
            voc_arr[k] = voc_k
            conv_mask[k] = conv_k
            brack_mask[k] = brack_k
            reasons[k] = reason_k
            if conv_k and brack_k and math.isfinite(voc_k):
                warm_voc = max(warm_voc, voc_k)
            if progress is not None:
                progress(
                    "suns_voc",
                    k + 1,
                    n_pts,
                    f"X={suns:.3g}, V_oc={voc_k:.4f} V, J_sc={jsc_arr[k]:.2f} A/m^2",
                )
    else:
        raise TypeError(
            "compiled must be a CompiledDevice1D or CompiledIonicDevice1D"
        )

    valid_mask = conv_mask & brack_mask
    if opts.require_all_bracketed and not bool(np.all(valid_mask)):
        bad = [
            f"X={suns_sorted[i]:.4g} ({reasons[i]})"
            for i in range(n_pts)
            if not valid_mask[i]
        ]
        raise RuntimeError(
            "Suns-V_oc failed to converge and bracket every intensity: "
            + ", ".join(bad)
        )

    local_n, n_fit, r_sq = compute_suns_voc_ideality(
        suns_sorted, voc_arr, temperature_K=temperature_K, valid_mask=valid_mask
    )

    ref_idx = int(np.argmin(np.abs(suns_sorted - 1.0)))
    voc_ref = float(voc_arr[ref_idx]) if valid_mask[ref_idx] else float("nan")
    jsc_ref = float(jsc_arr[ref_idx]) if valid_mask[ref_idx] else float("nan")

    j_pseudo_v = np.where(valid_mask, voc_arr, np.nan)
    j_pseudo_j = (
        np.where(valid_mask, jsc_ref - jsc_arr, np.nan)
        if math.isfinite(jsc_ref)
        else np.full(n_pts, np.nan, dtype=float)
    )
    pff, pff_reason = compute_pseudo_ff(
        j_pseudo_v,
        j_pseudo_j,
        V_oc_ref_V=voc_ref,
        J_sc_ref_A_m2=jsc_ref,
    )

    return SunsVocResult(
        suns_levels=suns_sorted,
        V_oc_V=voc_arr,
        J_sc_A_m2=jsc_arr,
        dark_short_circuit_A_m2=float(j_dark),
        converged_mask=conv_mask,
        voc_bracketed_mask=brack_mask,
        point_reasons=tuple(reasons),
        local_ideality_factor=local_n,
        ideality_factor_fit=n_fit,
        ideality_r_squared=r_sq,
        J_pseudo_V=j_pseudo_v,
        J_pseudo_A_m2=j_pseudo_j,
        pseudo_FF=pff,
        pseudo_ff_reason=pff_reason,
        reference_index=ref_idx,
        max_current_spread_A_m2=float(spread_max),
        temperature_K=temperature_K,
        voltage_polarity=polarity,
        compiled_identity=compiled.identity,
        solver_controls=controls_record,
    )
