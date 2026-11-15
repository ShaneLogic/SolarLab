"""Unified storage-form BDF1/BDF2 transient drivers for dynamic bulk/interface defects and ions.

Decomposes the legacy combinatorial transient runners (`bulk_defect_transient`,
`interface_defect_transient`, `bulk_defect_ion_transient`,
`interface_defect_ion_transient`, and the narrow-gate production
`dynamic_defect_transient`) onto one unified ``DynamicDefectStorage1D`` +
``BDFStep`` + ``dynamic_defect_step_accounting`` integration engine while
preserving every explicit capability gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu

from solarlab.contracts import frozen_array
from solarlab.device.compile_dynamic_defect_1d import compile_dynamic_defect_device_1d
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.assembly.dynamic_defects_1d import (
    AssembledDynamicDefect1DProblem,
    DynamicDefectSteadyOptions,
    DynamicDefectStorage1D,
    dynamic_defect_step_accounting,
    dynamic_defect_system_1d,
    solve_dynamic_defect_steady_1d,
)
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.dynamic_defect_1d import CompiledDynamicDefectDevice1D
from solarlab.solvers.time.bdf import BDFStep, HistoryPoint

Vector: TypeAlias = NDArray[np.float64]

VoltageInterpolation = Literal["right_continuous_step_and_hold", "piecewise_linear"]
DynamicDefectTransientCapability = Literal[
    "bulk_defect_transient",
    "interface_defect_transient",
    "bulk_defect_ion_transient",
    "interface_defect_ion_transient",
    "interface_defect_plus_positive_ions",
]

__all__ = [
    "DynamicDefectTransientCapability",
    "DynamicDefectTransientCapabilityError",
    "DynamicDefectTransientError",
    "DynamicDefectTransientOptions",
    "DynamicDefectTransientResult",
    "VoltageInterpolation",
    "classify_dynamic_defect_transient_capability",
    "solve_bulk_defect_ion_transient",
    "solve_bulk_defect_transient",
    "solve_dynamic_defect_transient",
    "solve_interface_defect_ion_transient",
    "solve_interface_defect_transient",
]


class DynamicDefectTransientError(RuntimeError):
    """Raised when dynamic-defect transient integration or validation fails."""


class DynamicDefectTransientCapabilityError(DynamicDefectTransientError, CapabilityError):
    """Raised when a device or protocol violates a transient capability gate."""


@dataclass(frozen=True, slots=True)
class DynamicDefectTransientOptions:
    """Discretization and nonlinear solver controls for dynamic-defect transients."""

    intervals_per_layer: int | tuple[int, ...] = 24
    substeps_per_interval: int = 1
    bdf_order: int = 1
    voltage_interpolation: VoltageInterpolation = "right_continuous_step_and_hold"
    ion_domain_endpoints: tuple[str, str] = ("blocking", "blocking")
    newton_tolerance: float = 1.0e-10
    max_newton_iterations: int = 50
    max_step_halvings: int = 8
    steady_options: DynamicDefectSteadyOptions | None = None

    def __post_init__(self) -> None:
        if isinstance(self.intervals_per_layer, int):
            if isinstance(self.intervals_per_layer, bool) or self.intervals_per_layer < 2:
                raise ValueError("intervals_per_layer must be >= 2")
        else:
            tup = tuple(int(v) for v in self.intervals_per_layer)
            if not tup or any(v < 2 for v in tup):
                raise ValueError("intervals_per_layer entries must be >= 2")
            object.__setattr__(self, "intervals_per_layer", tup)
        if isinstance(self.substeps_per_interval, bool) or self.substeps_per_interval < 1:
            raise ValueError("substeps_per_interval must be >= 1")
        if self.bdf_order not in {1, 2}:
            raise ValueError("bdf_order must be 1 or 2")
        if self.voltage_interpolation not in {
            "right_continuous_step_and_hold",
            "piecewise_linear",
        }:
            raise ValueError(f"unsupported voltage_interpolation {self.voltage_interpolation!r}")
        if not math.isfinite(self.newton_tolerance) or self.newton_tolerance <= 0.0:
            raise ValueError("newton_tolerance must be finite and positive")
        if self.max_newton_iterations < 5 or self.max_step_halvings < 0:
            raise ValueError("invalid max_newton_iterations or max_step_halvings")


@dataclass(frozen=True)
class DynamicDefectTransientResult:
    """Conservative all-face current, carrier, ion, and trap transient trajectory."""

    capability: DynamicDefectTransientCapability
    device: CompiledDynamicDefectDevice1D
    grid_m: Vector
    times_s: Vector
    voltage_V: Vector
    generation_scale: Vector
    terminal_current_density_A_m2: Vector
    conduction_current_faces_A_m2: Vector
    ionic_current_faces_A_m2: Vector
    displacement_current_faces_A_m2: Vector
    total_current_faces_A_m2: Vector
    maxwell_spread_A_m2: Vector
    potential_V: Vector
    electron_density_m3: Vector
    hole_density_m3: Vector
    bulk_trap_occupancy: Vector
    interface_trap_occupancy: Vector
    interface_sheet_charges_C_m2: Vector
    positive_ion_density_m3: Vector
    positive_ion_centroid_m: Vector
    maximum_ion_inventory_relative_drift: float
    maximum_maxwell_spread_A_m2: float
    scientific_qualification: bool = False
    physical_certificate_status: str = "not_evaluated"

    def __post_init__(self) -> None:
        for attr in (
            "grid_m",
            "times_s",
            "voltage_V",
            "generation_scale",
            "terminal_current_density_A_m2",
            "conduction_current_faces_A_m2",
            "ionic_current_faces_A_m2",
            "displacement_current_faces_A_m2",
            "total_current_faces_A_m2",
            "maxwell_spread_A_m2",
            "potential_V",
            "electron_density_m3",
            "hole_density_m3",
            "bulk_trap_occupancy",
            "interface_trap_occupancy",
            "interface_sheet_charges_C_m2",
            "positive_ion_density_m3",
            "positive_ion_centroid_m",
        ):
            object.__setattr__(self, attr, frozen_array(getattr(self, attr)))


def _validate_traces(
    times_s: ArrayLike,
    voltage_V: ArrayLike,
    generation_scale: float | ArrayLike,
) -> tuple[Vector, Vector, Vector]:
    t_arr = np.asarray(times_s, dtype=np.float64)
    v_arr = np.asarray(voltage_V, dtype=np.float64)
    if t_arr.ndim != 1 or t_arr.size < 2 or not np.all(np.isfinite(t_arr)):
        raise ValueError("times_s must be a 1D array with at least two finite values")
    if t_arr[0] != 0.0:
        raise ValueError("transient times_s must start at t = 0.0")
    if np.any(np.diff(t_arr) <= 0.0):
        raise ValueError("times_s must be strictly increasing")
    if v_arr.shape != t_arr.shape or not np.all(np.isfinite(v_arr)):
        raise ValueError("voltage_V must be finite and aligned with times_s")
    if np.ndim(generation_scale) == 0:
        g_val = float(generation_scale)
        if not math.isfinite(g_val) or g_val < 0.0:
            raise ValueError("generation_scale must be finite and non-negative")
        g_arr = np.full(t_arr.size, g_val, dtype=np.float64)
    else:
        g_arr = np.asarray(generation_scale, dtype=np.float64)
        if g_arr.shape != t_arr.shape or not np.all(np.isfinite(g_arr)) or np.any(g_arr < 0.0):
            raise ValueError("generation_scale trace must be finite, non-negative, and aligned")
    return t_arr, v_arr, g_arr


def _require_uncalibrated_interface_defects(prepared: PreparedDevice) -> int:
    count = 0
    for iface in prepared.interfaces:
        if not iface.electrical:
            continue
        entry = iface.to_mapping()
        defect = entry.get("defect")
        if defect is None:
            continue
        count += 1
        if float(defect["calibration_factor"]) != 1.0 or float(
            defect["iface_state_calibration_factor"]
        ) != 1.0:
            raise DynamicDefectTransientCapabilityError(
                f"interfaces.{entry['id']}: empirical calibration factors are refused"
            )
        raw_vn = entry.get("v_n")
        raw_vp = entry.get("v_p")
        micro_pair = tuple(float(v) for v in entry["microscopic_capture_velocities_m_s"])
        if raw_vn is not None and raw_vp is not None and not (
            math.isclose(float(raw_vn), micro_pair[0], rel_tol=1.0e-12, abs_tol=0.0)
            and math.isclose(float(raw_vp), micro_pair[1], rel_tol=1.0e-12, abs_tol=0.0)
        ):
            raise DynamicDefectTransientCapabilityError(
                f"interfaces.{entry['id']}: compatibility SRV disagrees with microscopic capture"
            )
    return count


def _layer_param_float(layer: object, key: str) -> float:
    params = dict(getattr(layer, "parameters", ()))
    val = params.get(key)
    return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else 0.0


def classify_dynamic_defect_transient_capability(
    prepared: PreparedDevice,
) -> Literal["interface_defect_plus_positive_ions"]:
    """Enforce the exact narrow v1 production dynamic-defect transient gate."""
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("classify_dynamic_defect_transient_capability requires PreparedDevice")
    elec_layers = [ly for ly in prepared.layers if ly.role != "substrate"]
    if len(elec_layers) != 2:
        raise DynamicDefectTransientCapabilityError(
            "v1 production transient requires exactly two electrical layers"
        )
    for ly in elec_layers:
        if (
            ly.defects
            or ly.bulk_trap_distribution_json is not None
            or ly.metastable_document is not None
        ):
            raise DynamicDefectTransientCapabilityError(
                "v1 production transient excludes simultaneous bulk explicit defects"
            )

    iface_count = _require_uncalibrated_interface_defects(prepared)
    if iface_count != 1:
        raise DynamicDefectTransientCapabilityError(
            "v1 production transient requires exactly one microscopic interface defect"
        )

    settings = prepared.to_mapping()["settings"]
    if str(settings.get("interface_charge_closure", "off")) != "equilibrium_referenced":
        raise DynamicDefectTransientCapabilityError(
            "v1 production transient requires equilibrium-referenced interface charge"
        )
    if not bool(settings.get("interface_charge_rebaseline_acknowledged", False)):
        raise DynamicDefectTransientCapabilityError(
            "v1 production transient requires interface charge rebaseline acknowledgement"
        )

    pos_layers = [
        idx
        for idx, ly in enumerate(elec_layers)
        if _layer_param_float(ly, "D_ion") > 0.0
        and _layer_param_float(ly, "P0") > 0.0
    ]
    neg_layers = [
        idx
        for idx, ly in enumerate(elec_layers)
        if _layer_param_float(ly, "D_ion_neg") > 0.0
        and _layer_param_float(ly, "P0_neg") > 0.0
    ]
    if len(pos_layers) != 1:
        raise DynamicDefectTransientCapabilityError(
            "v1 production transient requires exactly one active positive-ion layer"
        )
    if str(elec_layers[pos_layers[0]].role).lower() != "absorber":
        raise DynamicDefectTransientCapabilityError(
            "the active positive-ion layer must have role='absorber'"
        )
    if neg_layers:
        raise DynamicDefectTransientCapabilityError(
            "v1 production transient excludes active negative ions"
        )
    return "interface_defect_plus_positive_ions"


def _solve_bdf_step_newton(
    step: BDFStep,
    u_init: Vector,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[Vector, bool]:
    u = np.array(u_init, dtype=np.float64, copy=True)
    adapter = step.endpoint
    assert isinstance(adapter, DynamicDefectStorage1D)
    N = adapter.device.node_count
    phi_l, phi_r = adapter.device.electronic_component.contact_potentials(adapter.voltage_V)
    u[0] = phi_l
    u[N - 1] = phi_r

    for iteration in range(max_iterations + 1):
        try:
            f = np.asarray(step.residual(u), dtype=np.float64)
            jac = csc_matrix(step.jacobian(u))
        except (FloatingPointError, ValueError):
            return u, False
        if not np.all(np.isfinite(f)):
            return u, False
        row_max = np.asarray(np.abs(jac).max(axis=1).todense()).ravel()
        row_scale = np.where(row_max > 0.0, 1.0 / row_max, 1.0)
        fs = row_scale * f
        if float(np.max(np.abs(fs))) <= tolerance:
            return u, True
        if iteration == max_iterations:
            break
        j_row = diags(row_scale) @ jac
        col_max = np.asarray(np.abs(j_row).max(axis=0).todense()).ravel()
        col_scale = np.where(col_max > 0.0, 1.0 / col_max, 1.0)
        try:
            du = -col_scale * splu((j_row @ diags(col_scale)).tocsc()).solve(fs)
        except RuntimeError:
            return u, False
        if not np.all(np.isfinite(du)):
            return u, False
        max_dphi = float(np.max(np.abs(du[:N])))
        max_dlog = float(np.max(np.abs(du[N:]))) if du.size > N else 0.0
        clip = 1.0
        if max_dphi > 0.5:
            clip = min(clip, 0.5 / max_dphi)
        if max_dlog > 8.0:
            clip = min(clip, 8.0 / max_dlog)
        merit = float(np.linalg.norm(fs))
        alpha = clip
        accepted = False
        for _ in range(24):
            u_try = u + alpha * du
            try:
                f_try = np.asarray(step.residual(u_try), dtype=np.float64)
            except (FloatingPointError, ValueError):
                alpha *= 0.5
                continue
            if np.all(np.isfinite(f_try)):
                m_try = float(np.linalg.norm(row_scale * f_try))
                if m_try <= (1.0 - 1.0e-4 * alpha) * merit or m_try <= tolerance:
                    u = u_try
                    accepted = True
                    break
            alpha *= 0.5
        if not accepted:
            return u, False
    return u, False


def _advance_substep_recursive(
    device: CompiledDynamicDefectDevice1D,
    sys_binding: Any,
    history: Sequence[HistoryPoint],
    t_start: float,
    t_end: float,
    v_target: float,
    g_target: float,
    order: int,
    options: DynamicDefectTransientOptions,
    halvings_left: int,
) -> tuple[HistoryPoint, BDFStep]:
    prob = AssembledDynamicDefect1DProblem(device, v_target, g_target, None, sys_binding)
    adapter = DynamicDefectStorage1D(prob)
    prev = history[0]
    assert isinstance(prev.problem, DynamicDefectStorage1D)
    u_guess = device.rebase_coordinates(prev.u, prev.problem.voltage_V, v_target)

    eff_order = 2 if (order == 2 and len(history) >= 2) else 1
    hist_slice = (history[0], history[1]) if eff_order == 2 else (history[0],)
    step = BDFStep(adapter, hist_slice, t_end, eff_order)
    u_sol, ok = _solve_bdf_step_newton(
        step,
        u_guess,
        tolerance=options.newton_tolerance,
        max_iterations=options.max_newton_iterations,
    )
    if not ok and abs(v_target - prev.problem.voltage_V) > 1.0e-9:
        u_cap = u_guess.copy()
        N = device.node_count
        vt = device.thermal_voltage_V
        phi_l, phi_r = device.electronic_component.contact_potentials(v_target)
        dv = (phi_r - phi_l) - (u_cap[N - 1] - u_cap[0])
        nodes = device.nodes_m
        dphi_vec = dv * (nodes - nodes[0]) / (nodes[-1] - nodes[0])
        u_cap[:N] += dphi_vec
        u_cap[N : 2 * N] -= dphi_vec / vt
        u_cap[2 * N : 3 * N] += dphi_vec / vt
        u_sol, ok = _solve_bdf_step_newton(
            step,
            u_cap,
            tolerance=options.newton_tolerance,
            max_iterations=options.max_newton_iterations,
        )
    if ok:
        hp = HistoryPoint(t_end, u_sol, adapter.storage(u_sol), adapter)
        return hp, step

    if halvings_left <= 0:
        raise DynamicDefectTransientError(
            f"BDF transient failed to converge on [{t_start:.4e}, {t_end:.4e}] s at V={v_target:.4f} V"
        )
    t_mid = 0.5 * (t_start + t_end)
    v_mid = (
        0.5 * (prev.problem.voltage_V + v_target)
        if options.voltage_interpolation == "piecewise_linear"
        else v_target
    )
    g_mid = (
        0.5 * (prev.problem.generation_scale + g_target)
        if options.voltage_interpolation == "piecewise_linear"
        else g_target
    )
    hp_mid, _ = _advance_substep_recursive(
        device,
        sys_binding,
        history,
        t_start,
        t_mid,
        v_mid,
        g_mid,
        1,
        options,
        halvings_left - 1,
    )
    return _advance_substep_recursive(
        device,
        sys_binding,
        (hp_mid, *history),
        t_mid,
        t_end,
        v_target,
        g_target,
        order,
        options,
        halvings_left - 1,
    )


def _run_unified_transient(
    capability: DynamicDefectTransientCapability,
    compiled: CompiledDynamicDefectDevice1D,
    times: Vector,
    voltages: Vector,
    generations: Vector,
    options: DynamicDefectTransientOptions,
) -> DynamicDefectTransientResult:
    steady0 = solve_dynamic_defect_steady_1d(
        compiled,
        voltage_V=float(voltages[0]),
        generation_scale=float(generations[0]),
        options=options.steady_options,
    )
    sys_tr = dynamic_defect_system_1d(compiled, inventory_row_node=None)
    prob0 = AssembledDynamicDefect1DProblem(
        compiled, float(voltages[0]), float(generations[0]), None, sys_tr
    )
    adapter0 = DynamicDefectStorage1D(prob0)

    T = times.size
    N = compiled.node_count
    bt_size = compiled.bulk_trap_dof_count
    it_size = len(compiled.dynamic_interface_traps)
    if_count = len(compiled.electronic_component.geometry.interfaces)
    has_pos_ions = compiled.ionic_component is not None

    j_term = np.zeros(T, dtype=np.float64)
    j_cond = np.zeros((T, N + 1), dtype=np.float64)
    j_ion = np.zeros((T, N + 1), dtype=np.float64)
    j_disp = np.zeros((T, N + 1), dtype=np.float64)
    j_tot = np.zeros((T, N + 1), dtype=np.float64)
    spreads = np.zeros(T, dtype=np.float64)

    pot = np.zeros((T, N), dtype=np.float64)
    n_arr = np.zeros((T, N), dtype=np.float64)
    p_arr = np.zeros((T, N), dtype=np.float64)
    bt_occ = np.zeros((T, bt_size), dtype=np.float64)
    it_occ = np.zeros((T, it_size), dtype=np.float64)
    sheets = np.zeros((T, if_count), dtype=np.float64)
    pos_ion = np.zeros((T, N if has_pos_ions else 0), dtype=np.float64)
    pos_centroid: Vector = np.zeros(T if has_pos_ions else 0, dtype=np.float64)

    def record_state(k: int, u_vec: Vector, v_val: float) -> float:
        st = compiled.state(u_vec, voltage_V=v_val)
        pot[k] = st.electronic_state.potential_V
        n_arr[k] = st.electronic_state.electron_density_m3
        p_arr[k] = st.electronic_state.hole_density_m3
        if bt_size:
            bt_occ[k] = st.bulk_trap_occupancy
        if it_size:
            it_occ[k] = st.interface_trap_occupancy
        if if_count:
            sheets[k] = st.interface_sheet_charges_C_m2
        inv_drift = 0.0
        if compiled.ionic_component is not None and st.ionic_state is not None:
            ion_c = compiled.ionic_component
            pos_ion[k] = ion_c.project_ion_density(st.ionic_state.positive_ion_density_m3)
            w_ion = ion_c.ion_geometry.cell_measures_m
            x_ion = ion_c.ion_geometry.nodes_m
            dens_pos = st.ionic_state.positive_ion_density_m3
            inv_pos = float(np.sum(w_ion * dens_pos))
            pos_centroid[k] = float(np.sum(w_ion * x_ion * dens_pos) / inv_pos)
            ref_pos = ion_c.reference_inventory_m2
            inv_drift = abs(inv_pos - ref_pos) / ref_pos
            if ion_c.ion_species_count == 2:
                dens_neg = st.ionic_state.negative_ion_density_m3
                inv_neg = float(np.sum(w_ion * dens_neg))
                ref_neg = ion_c.negative_reference_inventory_m2
                inv_drift = max(inv_drift, abs(inv_neg - ref_neg) / ref_neg)
        return inv_drift

    max_inv_drift = record_state(0, steady0.state.u, float(voltages[0]))
    j_term[0] = steady0.terminal_current_density_A_m2
    j_cond[0] = steady0.all_face_current_density_A_m2
    j_tot[0] = steady0.all_face_current_density_A_m2
    spreads[0] = steady0.maxwell_spread_A_m2

    history: tuple[HistoryPoint, ...] = (
        HistoryPoint(0.0, steady0.state.u, adapter0.storage(steady0.state.u), adapter0),
    )

    n_sub = options.substeps_per_interval
    for k in range(1, T):
        t0 = float(times[k - 1])
        t1 = float(times[k])
        v0 = float(voltages[k - 1])
        v1 = float(voltages[k])
        g0 = float(generations[k - 1])
        g1 = float(generations[k])
        abrupt = options.voltage_interpolation == "right_continuous_step_and_hold" and (
            v1 != v0 or g1 != g0
        )

        last_step: BDFStep | None = None
        for s_idx in range(1, n_sub + 1):
            ta = t0 + (t1 - t0) * ((s_idx - 1) / n_sub)
            tb = t0 + (t1 - t0) * (s_idx / n_sub)
            if options.voltage_interpolation == "piecewise_linear":
                v_sub = v0 + (v1 - v0) * (s_idx / n_sub)
                g_sub = g0 + (g1 - g0) * (s_idx / n_sub)
            else:
                v_sub = v1
                g_sub = g1

            req_order = 1 if (abrupt and s_idx == 1) else options.bdf_order
            hp_new, last_step = _advance_substep_recursive(
                compiled,
                sys_tr,
                history,
                ta,
                tb,
                v_sub,
                g_sub,
                req_order,
                options,
                options.max_step_halvings,
            )
            history = (hp_new, history[0])

        assert last_step is not None
        u_now = history[0].u
        acc = dynamic_defect_step_accounting(last_step, u_now)
        j_cond[k] = acc["conduction_current_density_A_m2"]
        j_ion[k] = acc["ionic_current_density_A_m2"]
        j_disp[k] = acc["displacement_current_density_A_m2"]
        j_tot[k] = acc["total_current_density_A_m2"]
        j_term[k] = acc["terminal_current_density_A_m2"]
        spreads[k] = acc["maxwell_spread_A_m2"]
        max_inv_drift = max(max_inv_drift, record_state(k, u_now, v1))

    return DynamicDefectTransientResult(
        capability=capability,
        device=compiled,
        grid_m=compiled.nodes_m,
        times_s=times,
        voltage_V=voltages,
        generation_scale=generations,
        terminal_current_density_A_m2=j_term,
        conduction_current_faces_A_m2=j_cond,
        ionic_current_faces_A_m2=j_ion,
        displacement_current_faces_A_m2=j_disp,
        total_current_faces_A_m2=j_tot,
        maxwell_spread_A_m2=spreads,
        potential_V=pot,
        electron_density_m3=n_arr,
        hole_density_m3=p_arr,
        bulk_trap_occupancy=bt_occ,
        interface_trap_occupancy=it_occ,
        interface_sheet_charges_C_m2=sheets,
        positive_ion_density_m3=pos_ion,
        positive_ion_centroid_m=pos_centroid,
        maximum_ion_inventory_relative_drift=max_inv_drift,
        maximum_maxwell_spread_A_m2=float(np.max(spreads)),
    )


def solve_dynamic_defect_transient(
    prepared: PreparedDevice,
    times_s: ArrayLike,
    voltage_V: ArrayLike,
    *,
    illuminated: bool = False,
    options: DynamicDefectTransientOptions | None = None,
) -> DynamicDefectTransientResult:
    """Solve the narrow v1 production dark interface-defect + positive-ion transient."""
    controls = DynamicDefectTransientOptions() if options is None else options
    if not isinstance(illuminated, (bool, np.bool_)):
        raise TypeError("illuminated must be boolean")
    if bool(illuminated):
        raise DynamicDefectTransientCapabilityError(
            "the v1 production dynamic-defect transient capability is dark-only"
        )
    if controls.voltage_interpolation != "right_continuous_step_and_hold":
        raise DynamicDefectTransientCapabilityError(
            "v1 production dynamic-defect transient requires right_continuous_step_and_hold"
        )
    capability = classify_dynamic_defect_transient_capability(prepared)
    times, voltages, generations = _validate_traces(times_s, voltage_V, 0.0)

    elec_layers = [ly for ly in prepared.layers if ly.role != "substrate"]
    ion_layer = next(
        ly.id
        for ly in elec_layers
        if _layer_param_float(ly, "D_ion") > 0.0
        and _layer_param_float(ly, "P0") > 0.0
    )
    try:
        compiled = compile_dynamic_defect_device_1d(
            prepared,
            intervals_per_layer=controls.intervals_per_layer,
            dynamic_bulk_defects=False,
            dynamic_interface_defects=True,
            ion_domain_layer=ion_layer,
            ion_domain_endpoints=controls.ion_domain_endpoints,
        )
    except CapabilityError as error:
        raise DynamicDefectTransientCapabilityError(str(error)) from error

    return _run_unified_transient(capability, compiled, times, voltages, generations, controls)


def solve_bulk_defect_transient(
    prepared: PreparedDevice,
    times_s: ArrayLike,
    voltage_V: ArrayLike,
    *,
    generation_scale: float | ArrayLike = 0.0,
    options: DynamicDefectTransientOptions | None = None,
) -> DynamicDefectTransientResult:
    """Solve charged single-level non-spatial ion-free bulk-defect transient (D6-E1)."""
    controls = DynamicDefectTransientOptions() if options is None else options
    times, voltages, generations = _validate_traces(times_s, voltage_V, generation_scale)

    elec_layers = [ly for ly in prepared.layers if ly.role != "substrate"]
    if _require_uncalibrated_interface_defects(prepared) > 0:
        raise DynamicDefectTransientCapabilityError(
            "solve_bulk_defect_transient excludes interface defects"
        )
    has_bulk = False
    for ly in elec_layers:
        if (
            _layer_param_float(ly, "D_ion") > 0.0
            and _layer_param_float(ly, "P0") > 0.0
        ) or (
            _layer_param_float(ly, "D_ion_neg") > 0.0
            and _layer_param_float(ly, "P0_neg") > 0.0
        ):
            raise DynamicDefectTransientCapabilityError(
                "solve_bulk_defect_transient excludes active mobile ions"
            )
        if ly.bulk_trap_distribution_json is not None or ly.metastable_document is not None:
            raise DynamicDefectTransientCapabilityError(
                f"layers.{ly.id}: solve_bulk_defect_transient supports only explicit single-level bulk defects"
            )
        if ly.defect_schema_version == "solarlab-explicit-bulk-defects-v4":
            raise DynamicDefectTransientCapabilityError(
                f"layers.{ly.id}: multivalent bulk defects are refused in D6-E1 bulk transient"
            )
        for raw_bd in ly.defects:
            bd = raw_bd.to_mapping()
            has_bulk = True
            if str(bd.get("charge_transition")) == "neutral":
                raise DynamicDefectTransientCapabilityError(
                    f"layers.{ly.id}.{bd['id']}: charge-coupled bulk transient rejects neutral transitions"
                )
            dist = bd.get("distribution", {})
            if str(dist.get("kind", "single_level")) != "single_level":
                raise DynamicDefectTransientCapabilityError(
                    f"layers.{ly.id}.{bd['id']}: D6-E1 bulk transient requires single_level distribution"
                )
            sp = bd.get("spatial_profile")
            if sp is not None and str(sp.get("kind", "uniform")) != "uniform":
                raise DynamicDefectTransientCapabilityError(
                    f"layers.{ly.id}.{bd['id']}: D6-E1 bulk transient rejects non-uniform spatial_profile"
                )
    if not has_bulk:
        raise DynamicDefectTransientCapabilityError(
            "solve_bulk_defect_transient requires at least one explicit charged bulk defect"
        )

    try:
        compiled = compile_dynamic_defect_device_1d(
            prepared,
            intervals_per_layer=controls.intervals_per_layer,
            dynamic_bulk_defects=True,
            dynamic_interface_defects=False,
        )
    except CapabilityError as error:
        raise DynamicDefectTransientCapabilityError(str(error)) from error

    return _run_unified_transient(
        "bulk_defect_transient", compiled, times, voltages, generations, controls
    )


def solve_interface_defect_transient(
    prepared: PreparedDevice,
    times_s: ArrayLike,
    voltage_V: ArrayLike,
    *,
    generation_scale: float | ArrayLike = 0.0,
    options: DynamicDefectTransientOptions | None = None,
) -> DynamicDefectTransientResult:
    """Solve ion-free two-sided dynamic interface-defect transient (D6-E2)."""
    controls = DynamicDefectTransientOptions() if options is None else options
    times, voltages, generations = _validate_traces(times_s, voltage_V, generation_scale)

    for ly in prepared.layers:
        if ly.role == "substrate":
            continue
        if ly.defects or ly.bulk_trap_distribution_json is not None or ly.metastable_document is not None:
            raise DynamicDefectTransientCapabilityError(
                "solve_interface_defect_transient excludes bulk explicit defects"
            )
        if (
            _layer_param_float(ly, "D_ion") > 0.0
            and _layer_param_float(ly, "P0") > 0.0
        ) or (
            _layer_param_float(ly, "D_ion_neg") > 0.0
            and _layer_param_float(ly, "P0_neg") > 0.0
        ):
            raise DynamicDefectTransientCapabilityError(
                "solve_interface_defect_transient excludes active mobile ions"
            )
    if _require_uncalibrated_interface_defects(prepared) < 1:
        raise DynamicDefectTransientCapabilityError(
            "solve_interface_defect_transient requires at least one microscopic interface defect"
        )
    settings = prepared.to_mapping()["settings"]
    if str(settings.get("interface_charge_closure", "off")) != "equilibrium_referenced":
        raise DynamicDefectTransientCapabilityError(
            "solve_interface_defect_transient requires equilibrium_referenced interface charge"
        )
    if not bool(settings.get("interface_charge_rebaseline_acknowledged", False)):
        raise DynamicDefectTransientCapabilityError(
            "solve_interface_defect_transient requires interface_charge_rebaseline_acknowledged=True"
        )

    try:
        compiled = compile_dynamic_defect_device_1d(
            prepared,
            intervals_per_layer=controls.intervals_per_layer,
            dynamic_bulk_defects=False,
            dynamic_interface_defects=True,
        )
    except CapabilityError as error:
        raise DynamicDefectTransientCapabilityError(str(error)) from error

    return _run_unified_transient(
        "interface_defect_transient", compiled, times, voltages, generations, controls
    )


def _single_ion_layer(prepared: PreparedDevice) -> str:
    layers = [
        ly.id
        for ly in prepared.layers
        if ly.role != "substrate"
        and (
            (_layer_param_float(ly, "D_ion") > 0.0 and _layer_param_float(ly, "P0") > 0.0)
            or (
                _layer_param_float(ly, "D_ion_neg") > 0.0
                and _layer_param_float(ly, "P0_neg") > 0.0
            )
        )
    ]
    if len(layers) != 1:
        raise DynamicDefectTransientCapabilityError(
            "coupled defect-ion transient requires exactly one active mobile-ion layer"
        )
    return layers[0]


def solve_bulk_defect_ion_transient(
    prepared: PreparedDevice,
    times_s: ArrayLike,
    voltage_V: ArrayLike,
    *,
    generation_scale: float | ArrayLike = 0.0,
    options: DynamicDefectTransientOptions | None = None,
) -> DynamicDefectTransientResult:
    """Solve coupled dynamic bulk-defect + mobile-ion transient (D6-E3a)."""
    controls = DynamicDefectTransientOptions() if options is None else options
    times, voltages, generations = _validate_traces(times_s, voltage_V, generation_scale)
    if _require_uncalibrated_interface_defects(prepared) > 0:
        raise DynamicDefectTransientCapabilityError(
            "solve_bulk_defect_ion_transient excludes interface defects"
        )
    for ly in prepared.layers:
        if ly.role == "substrate":
            continue
        for raw_bd in ly.defects:
            bd_map = raw_bd.to_mapping()
            if str(bd_map.get("charge_transition", "")) == "neutral":
                raise DynamicDefectTransientCapabilityError(
                    f"layers.{ly.id}.{bd_map['id']}: charge-coupled bulk-ion transient rejects neutral transitions"
                )
    ion_layer = _single_ion_layer(prepared)
    try:
        compiled = compile_dynamic_defect_device_1d(
            prepared,
            intervals_per_layer=controls.intervals_per_layer,
            dynamic_bulk_defects=True,
            dynamic_interface_defects=False,
            ion_domain_layer=ion_layer,
            ion_domain_endpoints=controls.ion_domain_endpoints,
        )
    except CapabilityError as error:
        raise DynamicDefectTransientCapabilityError(str(error)) from error
    if not compiled.has_dynamic_bulk_traps:
        raise DynamicDefectTransientCapabilityError(
            "solve_bulk_defect_ion_transient requires explicit bulk defects"
        )

    return _run_unified_transient(
        "bulk_defect_ion_transient", compiled, times, voltages, generations, controls
    )


def solve_interface_defect_ion_transient(
    prepared: PreparedDevice,
    times_s: ArrayLike,
    voltage_V: ArrayLike,
    *,
    generation_scale: float | ArrayLike = 0.0,
    options: DynamicDefectTransientOptions | None = None,
) -> DynamicDefectTransientResult:
    """Solve coupled dynamic interface-defect + mobile-ion transient (D6-E3b)."""
    controls = DynamicDefectTransientOptions() if options is None else options
    times, voltages, generations = _validate_traces(times_s, voltage_V, generation_scale)
    for ly in prepared.layers:
        if ly.role != "substrate" and (
            ly.defects or ly.bulk_trap_distribution_json is not None or ly.metastable_document is not None
        ):
            raise DynamicDefectTransientCapabilityError(
                "solve_interface_defect_ion_transient excludes bulk explicit defects"
            )
    if _require_uncalibrated_interface_defects(prepared) < 1:
        raise DynamicDefectTransientCapabilityError(
            "solve_interface_defect_ion_transient requires at least one microscopic interface defect"
        )
    settings = prepared.to_mapping()["settings"]
    if str(settings.get("interface_charge_closure", "off")) != "equilibrium_referenced":
        raise DynamicDefectTransientCapabilityError(
            "solve_interface_defect_ion_transient requires equilibrium_referenced interface charge"
        )
    if not bool(settings.get("interface_charge_rebaseline_acknowledged", False)):
        raise DynamicDefectTransientCapabilityError(
            "solve_interface_defect_ion_transient requires interface_charge_rebaseline_acknowledged=True"
        )
    ion_layer = _single_ion_layer(prepared)
    try:
        compiled = compile_dynamic_defect_device_1d(
            prepared,
            intervals_per_layer=controls.intervals_per_layer,
            dynamic_bulk_defects=False,
            dynamic_interface_defects=True,
            ion_domain_layer=ion_layer,
            ion_domain_endpoints=controls.ion_domain_endpoints,
        )
    except CapabilityError as error:
        raise DynamicDefectTransientCapabilityError(str(error)) from error

    return _run_unified_transient(
        "interface_defect_ion_transient", compiled, times, voltages, generations, controls
    )
