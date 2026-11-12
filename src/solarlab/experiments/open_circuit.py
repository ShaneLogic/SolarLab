"""Charge-conserving floating-terminal (open-circuit) Backward Euler integrator.

At genuine open circuit, zero external terminal current means zero Gauss-Maxwell
total port current density:

    I_total,port = I_carrier,port + I_ion,port + dD_port / dt = 0

NEVER steady conduction current ``I_cond = 0`` (which drops ``dD_port / dt`` and
turns dynamic photovoltage relaxation into an unphysical algebraic clamp).

At each Backward Euler step ``[t_{k-1}, t_k]``, the ``(M + 1)``-unknown system
``z = [y; V_terminal]`` solves all ``M`` physical PDE Backward Euler rows
simultaneously with ``I_total,left(y, V_terminal) = 0`` via bordered sparse
Newton iteration, certifying all-face Maxwell current closure and telescoping
port-charge conservation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast

import numpy as np
from scipy.sparse import bmat, csc_matrix, diags
from scipy.sparse.linalg import splu

from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem, State1D
from solarlab.equations.ionic_1d import (
    CompiledIonicDevice1D,
    Ionic1DProblem,
    IonicState1D,
)
from solarlab.solvers.nonlinear.production import (
    POISSON_CONTACT_BALANCE_V1,
    production_convergence_norm,
    production_coordinates,
    production_residual_scaling,
)
from solarlab.solvers.time.backward_euler import (
    BackwardEuler1DStep,
    step_accounting as electronic_step_accounting,
)
from solarlab.solvers.time.carrier_seed import transient_density_basin_seed
from solarlab.solvers.time.ionic_backward_euler import (
    IonicBackwardEuler1DStep,
    step_accounting as ionic_step_accounting,
)

__all__ = [
    "OpenCircuitError",
    "OpenCircuitOptions",
    "OpenCircuitStepDiagnostics",
    "OpenCircuitTrajectory",
    "geometric_series_capacitance_F_m2",
    "integrate_open_circuit",
]


class OpenCircuitError(RuntimeError):
    """Floating-terminal Backward Euler step failed convergence or conservation."""


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(
        value, bool
    )


def geometric_series_capacitance_F_m2(
    device: CompiledDevice1D | CompiledIonicDevice1D,
) -> float:
    """Return total series geometric capacitance ``(sum_f 1 / C_f)^(-1)`` [F/m^2]."""
    comp = (
        device.electronic_component
        if isinstance(device, CompiledIonicDevice1D)
        else device
    )
    inv_sum = float(np.sum(1.0 / comp.poisson.face_capacitance_F_m2))
    return 1.0 / inv_sum


@dataclass(frozen=True)
class OpenCircuitOptions:
    """Numerical tolerances and Newton controls for floating-terminal integration."""

    poisson_tolerance: float = 1e-12
    continuity_tolerance_A_m2: float = 1e-8
    contact_log_tolerance: float = 1e-12
    ion_balance_tolerance_m2_s: float = 1e9
    max_current_error_A_m2: float = 1e-7
    max_voltage_error_V: float = 1e-6
    potential_step_V: float = 0.1
    quasi_fermi_step: float = 2.0
    ion_logit_step: float = 2.0
    terminal_voltage_step_V: float = 0.15
    max_newton_iterations: int = 80
    fd_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "poisson_tolerance",
            "continuity_tolerance_A_m2",
            "contact_log_tolerance",
            "ion_balance_tolerance_m2_s",
            "max_current_error_A_m2",
            "max_voltage_error_V",
            "potential_step_V",
            "quasi_fermi_step",
            "ion_logit_step",
            "terminal_voltage_step_V",
            "fd_epsilon",
        ):
            raw = getattr(self, name)
            if not _is_real_number(raw) or not math.isfinite(float(raw)) or float(raw) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(raw))
        if type(self.max_newton_iterations) is not int or self.max_newton_iterations < 4:
            raise ValueError("max_newton_iterations must be an integer >= 4")


@dataclass(frozen=True)
class OpenCircuitStepDiagnostics:
    """Per-step conservation and solver diagnostics at floating terminal bias."""

    start_s: float
    end_s: float
    dt_s: float
    generation_scale: float
    terminal_voltage_V: float
    newton_iterations: int
    scaled_residual: float
    left_conduction_current_A_m2: float
    left_displacement_current_A_m2: float
    left_total_current_A_m2: float
    max_face_total_current_A_m2: float
    interval_charge_residual_C_m2: float


@dataclass(frozen=True)
class OpenCircuitTrajectory:
    """Certified floating-terminal trajectory ``V_oc(t)`` and conservation ledger."""

    times_s: np.ndarray
    generation_scale: np.ndarray
    terminal_voltage_V: np.ndarray
    left_conduction_current_A_m2: np.ndarray
    left_displacement_current_A_m2: np.ndarray
    left_total_current_A_m2: np.ndarray
    max_face_total_current_A_m2: np.ndarray
    cumulative_charge_residual_C_m2: float
    charge_voltage_error_V: float
    geometric_capacitance_F_m2: float
    states: tuple[State1D | IonicState1D, ...]
    step_diagnostics: tuple[OpenCircuitStepDiagnostics, ...]
    compiled_identity: str
    certified: bool


def _left_port_support_indices(
    device: CompiledDevice1D | CompiledIonicDevice1D,
) -> tuple[int, ...]:
    """Exact state indices entering ``I_total,left(y, V)``."""
    comp = (
        device.electronic_component
        if isinstance(device, CompiledIonicDevice1D)
        else device
    )
    n_nodes = comp.node_count
    cols: list[int] = [0, 1, n_nodes, n_nodes + 1, 2 * n_nodes, 2 * n_nodes + 1]
    if isinstance(device, CompiledIonicDevice1D) and int(device.ion_node_indices[0]) == 0:
        for block in device.ion_species_slices:
            cols.append(int(block.start))
    return tuple(cols)


def _eval_step_and_port(
    device: CompiledDevice1D | CompiledIonicDevice1D,
    prev_state: State1D | IonicState1D,
    dt_s: float,
    generation_scale: float,
    y: np.ndarray,
    voltage_V: float,
) -> tuple[
    BackwardEuler1DStep | IonicBackwardEuler1DStep,
    np.ndarray,
    dict[str, object],
    float,
]:
    if isinstance(device, CompiledDevice1D):
        assert isinstance(prev_state, State1D)
        step_e = BackwardEuler1DStep(
            DC1DProblem(device, voltage_V=voltage_V, generation_scale=generation_scale),
            prev_state,
            dt_s,
        )
        acc = electronic_step_accounting(step_e, y)
        res_pde = np.asarray(acc["backward_euler_residual"], dtype=float)
        term_tot = np.asarray(acc["terminal_total_current_density_A_m2"], dtype=float)
        return step_e, res_pde, acc, float(term_tot[0])

    assert isinstance(prev_state, IonicState1D)
    step_i = IonicBackwardEuler1DStep(
        Ionic1DProblem(device, voltage_V=voltage_V, generation_scale=generation_scale),
        prev_state,
        dt_s,
    )
    acc_i = ionic_step_accounting(step_i, y)
    res_pde_i = np.asarray(acc_i["backward_euler_residual"], dtype=float)
    term_tot_i = np.asarray(acc_i["terminal_total_current_density_A_m2"], dtype=float)
    return step_i, res_pde_i, acc_i, float(term_tot_i[0])


def _solve_floating_step(
    device: CompiledDevice1D | CompiledIonicDevice1D,
    prev_state: State1D | IonicState1D,
    prev_voltage_V: float,
    dt_s: float,
    generation_scale: float,
    options: OpenCircuitOptions,
    support_cols: tuple[int, ...],
    predictor: str = "previous",
) -> tuple[State1D | IonicState1D, float, OpenCircuitStepDiagnostics, float, float]:
    """Solve one bordered ``(M + 1)`` open-circuit Backward Euler step."""
    comp = (
        device.electronic_component
        if isinstance(device, CompiledIonicDevice1D)
        else device
    )
    m_size = (
        device.state_size
        if isinstance(device, CompiledIonicDevice1D)
        else 3 * comp.node_count
    )

    # Build row weights and max coordinate step bounds
    elec_weights = production_residual_scaling(
        comp,
        poisson_tolerance=options.poisson_tolerance,
        continuity_tolerance_A_m2=options.continuity_tolerance_A_m2,
        contact_log_tolerance=options.contact_log_tolerance,
    ).row_weights
    elec_steps = production_coordinates(
        comp,
        potential_step_V=options.potential_step_V,
        quasi_fermi_step=options.quasi_fermi_step,
    ).max_step
    if isinstance(device, CompiledIonicDevice1D):
        ion_len = device.ion_node_count * device.ion_species_count
        pde_weights = np.r_[
            elec_weights,
            np.full(ion_len, device.ion_node_count / options.ion_balance_tolerance_m2_s),
        ]
        pde_steps = np.r_[elec_steps, np.full(ion_len, options.ion_logit_step)]
    else:
        pde_weights = elec_weights
        pde_steps = elec_steps

    row_weights = np.r_[pde_weights, 1.0 / options.max_current_error_A_m2]
    row_scale = diags(row_weights, format="csc")
    max_steps = np.r_[pde_steps, options.terminal_voltage_step_V]
    elec_norm = production_convergence_norm(
        comp,
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )
    assert elec_norm is not None
    n_elec = 3 * comp.node_count

    def eval_scaled_norm(scaled_vec: np.ndarray) -> float:
        if not np.all(np.isfinite(scaled_vec)):
            return math.inf
        val = max(elec_norm(scaled_vec[:n_elec]), abs(float(scaled_vec[-1])))
        if isinstance(device, CompiledIonicDevice1D):
            n_ion = device.ion_node_count
            for block in device.ion_species_slices:
                val = max(
                    val,
                    math.fsum(abs(scaled_vec[block]).tolist()) / n_ion,
                )
        return val

    # Seed at previous terminal voltage (zero voltage jump, exact previous inventory)
    if isinstance(device, CompiledDevice1D):
        assert isinstance(prev_state, State1D)
        seed_step = BackwardEuler1DStep(
            DC1DProblem(
                device, voltage_V=prev_voltage_V, generation_scale=generation_scale
            ),
            prev_state,
            dt_s,
        )
        y_cur = seed_step.initial_coordinates("previous", voltage_lift=False)
        if predictor == "generation_inventory" and generation_scale > 0.0:
            try:
                y_cur = transient_density_basin_seed(
                    seed_step, y_cur, device=device, voltage_V=prev_voltage_V
                )
            except ValueError:
                y_cur = seed_step.initial_coordinates(predictor, voltage_lift=False)
    else:
        assert isinstance(prev_state, IonicState1D)
        seed_step_i = IonicBackwardEuler1DStep(
            Ionic1DProblem(
                device, voltage_V=prev_voltage_V, generation_scale=generation_scale
            ),
            prev_state,
            dt_s,
        )
        y_cur = seed_step_i.initial_coordinates("previous", voltage_lift=False)
        if predictor == "generation_inventory" and generation_scale > 0.0:
            try:
                seeded_e = transient_density_basin_seed(
                    seed_step_i.electronic_step,
                    y_cur[:n_elec],
                    device=comp,
                    voltage_V=prev_voltage_V,
                )
                y_cur = np.r_[seeded_e, y_cur[n_elec:]]
            except ValueError:
                y_cur = seed_step_i.initial_coordinates(predictor, voltage_lift=False)

    v_cur = float(prev_voltage_V)
    eps = options.fd_epsilon

    # Harmonic potential lift + carrier reference rebase tangent vector g_vec:
    # Perturbing y -> y + dv * g_vec as V -> V + dv updates contact potentials
    # harmonically while holding bulk physical densities (n, p, P) fixed.
    n_nodes = comp.node_count
    res_faces = 1.0 / comp.poisson.face_capacitance_F_m2
    harm_prof = np.r_[0.0, np.cumsum(res_faces)]
    harm_prof /= harm_prof[-1]
    d_phi_dv = -comp.voltage_polarity * harm_prof
    ref_0 = comp.carrier_reference_energies(0.0)
    ref_1 = comp.carrier_reference_energies(1.0)
    d_ref_n = ref_1[0] - ref_0[0]
    d_ref_p = ref_1[1] - ref_0[1]
    g_vec: np.ndarray = np.zeros(m_size, dtype=float)
    g_vec[:n_nodes] = d_phi_dv
    g_vec[n_nodes : 2 * n_nodes] = (-d_ref_n - d_phi_dv) / comp.thermal_voltage_V
    g_vec[2 * n_nodes : 3 * n_nodes] = (-d_ref_p - d_phi_dv) / comp.thermal_voltage_V

    def extract_step_currents(
        acc_map: dict[str, object],
    ) -> tuple[float, float, float, float]:
        if isinstance(device, CompiledDevice1D):
            t_cond = float(
                np.asarray(
                    acc_map["terminal_conduction_current_density_A_m2"], dtype=float
                )[0]
            )
            t_disp = float(
                np.asarray(
                    acc_map["terminal_displacement_current_density_A_m2"], dtype=float
                )[0]
            )
            int_t = np.asarray(acc_map["total_current_density_A_m2"], dtype=float)
            term_t = np.asarray(
                acc_map["terminal_total_current_density_A_m2"], dtype=float
            )
            all_t = np.r_[term_t[0], int_t, term_t[1]]
            d_left = t_disp * dt_s
        else:
            all_c = np.asarray(acc_map["conduction_current_density_A_m2"], dtype=float)
            all_d = np.asarray(
                acc_map["displacement_current_density_A_m2"], dtype=float
            )
            all_t = np.asarray(acc_map["total_current_density_A_m2"], dtype=float)
            int_d = np.asarray(acc_map["interval_displacement_C_m2"], dtype=float)
            t_cond = float(all_c[0])
            t_disp = float(all_d[0])
            d_left = float(int_d[0])
        return t_cond, t_disp, d_left, float(np.max(np.abs(all_t)))

    for it in range(1, options.max_newton_iterations + 1):
        step_obj, res_pde, acc, i_left = _eval_step_and_port(
            device, prev_state, dt_s, generation_scale, y_cur, v_cur
        )
        full_res = np.r_[res_pde, i_left]
        scaled_norm = eval_scaled_norm(row_weights * full_res)
        term_cond, term_disp, delta_d_left, max_face = extract_step_currents(acc)
        if (
            scaled_norm <= 1.0
            and abs(i_left) <= options.max_current_error_A_m2
            and max_face <= options.max_current_error_A_m2
        ):
            new_state: State1D | IonicState1D = device.state(y_cur, voltage_V=v_cur)
            dq_step = dt_s * term_cond + delta_d_left
            diag = OpenCircuitStepDiagnostics(
                start_s=0.0,
                end_s=0.0,
                dt_s=dt_s,
                generation_scale=generation_scale,
                terminal_voltage_V=v_cur,
                newton_iterations=it - 1,
                scaled_residual=scaled_norm,
                left_conduction_current_A_m2=term_cond,
                left_displacement_current_A_m2=term_disp,
                left_total_current_A_m2=i_left,
                max_face_total_current_A_m2=max_face,
                interval_charge_residual_C_m2=dq_step,
            )
            return new_state, v_cur, diag, dq_step, max_face

        # Assemble bordered Jacobian in physical-density lifted chart (dw, dV)
        # where dy = dw + dV * g_vec:
        #   [ J_yy    dR_pde/dV|_{dw=0} ]
        #   [ dI/dy   dI/dV|_{dw=0}     ]
        j_yy = step_obj.jacobian(y_cur)

        # Row M: dI_left / dy on support_cols via central differences
        row_data: list[float] = []
        row_cols: list[int] = []
        for col in support_cols:
            h_y = eps * max(1.0, abs(float(y_cur[col])))
            y_plus = y_cur.copy()
            y_plus[col] += h_y
            y_minus = y_cur.copy()
            y_minus[col] -= h_y
            _, _, _, i_p = _eval_step_and_port(
                device, prev_state, dt_s, generation_scale, y_plus, v_cur
            )
            _, _, _, i_m = _eval_step_and_port(
                device, prev_state, dt_s, generation_scale, y_minus, v_cur
            )
            deriv = (i_p - i_m) / (2.0 * h_y)
            if deriv != 0.0:
                row_cols.append(col)
                row_data.append(deriv)

        j_vy = csc_matrix(
            (
                np.asarray(row_data, dtype=float),
                (np.zeros(len(row_cols), dtype=int), np.asarray(row_cols, dtype=int)),
            ),
            shape=(1, m_size),
        )

        # Column M: directional derivative along (g_vec, 1) via central difference in V
        h_v = eps * max(1.0, abs(v_cur))
        _, res_vp, _, i_vp = _eval_step_and_port(
            device,
            prev_state,
            dt_s,
            generation_scale,
            y_cur + h_v * g_vec,
            v_cur + h_v,
        )
        _, res_vm, _, i_vm = _eval_step_and_port(
            device,
            prev_state,
            dt_s,
            generation_scale,
            y_cur - h_v * g_vec,
            v_cur - h_v,
        )
        d_res_dv = ((res_vp - res_vm) / (2.0 * h_v)).reshape(m_size, 1)
        d_i_dv = np.array([[(i_vp - i_vm) / (2.0 * h_v)]], dtype=float)

        scaled_res = row_weights * full_res
        bordered = cast(
            csc_matrix,
            row_scale
            @ bmat(
                [[j_yy, csc_matrix(d_res_dv)], [j_vy, csc_matrix(d_i_dv)]],
                format="csc",
            ),
        )
        try:
            lu = splu(bordered)
            delta_chart = np.asarray(lu.solve(-scaled_res), dtype=float)
        except RuntimeError as exc:
            raise OpenCircuitError(f"Singular bordered Jacobian: {exc}") from exc
        if delta_chart.shape != (m_size + 1,) or not np.all(np.isfinite(delta_chart)):
            raise OpenCircuitError("Nonfinite bordered Newton step in integrate_open_circuit")

        delta_w = delta_chart[:m_size]
        delta_v = float(delta_chart[m_size])
        delta_y = delta_w + delta_v * g_vec

        # Coordinate-wise step bound on chart increment (dw, dV) + Deuflhard monotonicity
        step_norm = float(np.max(np.abs(delta_chart) / max_steps))
        alpha = 1.0 / max(1.0, step_norm)

        accepted_line = False
        best_simp = math.inf
        best_y = y_cur + alpha * delta_y
        best_v = float(v_cur + alpha * delta_v)
        for _ls in range(12):
            cand_y = y_cur + alpha * delta_y
            cand_v = float(v_cur + alpha * delta_v)
            try:
                _, cand_pde, cand_acc, cand_i = _eval_step_and_port(
                    device, prev_state, dt_s, generation_scale, cand_y, cand_v
                )
                cand_scaled = row_weights * np.r_[cand_pde, cand_i]
                cand_scaled_norm = eval_scaled_norm(cand_scaled)
                _, _, _, cand_max_face = extract_step_currents(cand_acc)
                simp_dz = np.asarray(lu.solve(-cand_scaled), dtype=float)
                simp_norm = float(np.max(np.abs(simp_dz) / max_steps))
            except (ValueError, FloatingPointError, RuntimeError):
                cand_scaled_norm = math.inf
                cand_max_face = math.inf
                simp_norm = math.inf
            if math.isfinite(simp_norm) and simp_norm < best_simp:
                best_simp = simp_norm
                best_y, best_v = cand_y, cand_v
            if math.isfinite(simp_norm) and (
                simp_norm <= (1.0 - 1e-4 * alpha) * step_norm
                or cand_scaled_norm < (1.0 - 1e-4 * alpha) * scaled_norm
                or (cand_scaled_norm <= 1.0 and step_norm <= 1e-2)
                or (
                    cand_scaled_norm <= 1.0
                    and abs(cand_i) <= options.max_current_error_A_m2
                    and cand_max_face <= options.max_current_error_A_m2
                )
            ):
                best_y, best_v = cand_y, cand_v
                accepted_line = True
                break
            alpha *= 0.5
        if not accepted_line and not math.isfinite(best_simp):
            raise OpenCircuitError("Floating-terminal Newton backtracking failed")
        y_cur, v_cur = best_y, best_v

    raise OpenCircuitError(
        f"Open-circuit Backward Euler step failed to converge in {options.max_newton_iterations} Newton iterations"
    )


def _advance_floating_interval(
    device: CompiledDevice1D | CompiledIonicDevice1D,
    prev_state: State1D | IonicState1D,
    prev_voltage_V: float,
    t_prev: float,
    t_next: float,
    g_prev: float,
    g_next: float,
    options: OpenCircuitOptions,
    support_cols: tuple[int, ...],
    _depth: int = 0,
) -> tuple[State1D | IonicState1D, float, OpenCircuitStepDiagnostics, float, float]:
    dt_s = t_next - t_prev
    pred = "generation_inventory" if (g_prev == 0.0 and g_next > 0.0) else "previous"
    try:
        return _solve_floating_step(
            device,
            prev_state,
            prev_voltage_V,
            dt_s,
            g_next,
            options,
            support_cols,
            predictor=pred,
        )
    except OpenCircuitError:
        if _depth >= 5 or dt_s < 2e-8:
            raise
        t_mid = 0.5 * (t_prev + t_next)
        g_mid = 0.5 * (g_prev + g_next) if g_prev != g_next else g_next
        mid_state, mid_v, diag1, dq1, mf1 = _advance_floating_interval(
            device,
            prev_state,
            prev_voltage_V,
            t_prev,
            t_mid,
            g_prev,
            g_mid,
            options,
            support_cols,
            _depth=_depth + 1,
        )
        end_state, end_v, diag2, dq2, mf2 = _advance_floating_interval(
            device,
            mid_state,
            mid_v,
            t_mid,
            t_next,
            g_mid,
            g_next,
            options,
            support_cols,
            _depth=_depth + 1,
        )
        dq_tot = dq1 + dq2
        mf_tot = max(mf1, mf2)
        combined = OpenCircuitStepDiagnostics(
            start_s=t_prev,
            end_s=t_next,
            dt_s=dt_s,
            generation_scale=g_next,
            terminal_voltage_V=end_v,
            newton_iterations=diag1.newton_iterations + diag2.newton_iterations,
            scaled_residual=max(diag1.scaled_residual, diag2.scaled_residual),
            left_conduction_current_A_m2=diag2.left_conduction_current_A_m2,
            left_displacement_current_A_m2=diag2.left_displacement_current_A_m2,
            left_total_current_A_m2=diag2.left_total_current_A_m2,
            max_face_total_current_A_m2=mf_tot,
            interval_charge_residual_C_m2=dq_tot,
        )
        return end_state, end_v, combined, dq_tot, mf_tot


def integrate_open_circuit(
    device: CompiledDevice1D | CompiledIonicDevice1D,
    initial_state: State1D | IonicState1D,
    times_s: object,
    generation_scale: object,
    *,
    initial_voltage_V: float | None = None,
    options: OpenCircuitOptions | None = None,
) -> OpenCircuitTrajectory:
    """Integrate ``device`` along ``times_s`` under zero total external port current."""
    if not isinstance(device, (CompiledDevice1D, CompiledIonicDevice1D)):
        raise TypeError("device must be a CompiledDevice1D or CompiledIonicDevice1D")
    if isinstance(device, CompiledDevice1D) and not isinstance(initial_state, State1D):
        raise TypeError("CompiledDevice1D requires an initial State1D")
    if isinstance(device, CompiledIonicDevice1D) and not isinstance(
        initial_state, IonicState1D
    ):
        raise TypeError("CompiledIonicDevice1D requires an initial IonicState1D")
    if initial_state.device_sha256 != device.identity:
        raise ValueError("initial_state belongs to a different compiled device")

    opts = OpenCircuitOptions() if options is None else options
    if not isinstance(opts, OpenCircuitOptions):
        raise TypeError("options must be an OpenCircuitOptions")

    if np.iscomplexobj(times_s) or np.iscomplexobj(generation_scale):
        raise ValueError("times_s and generation_scale must be real-valued")
    t_arr = np.array(times_s, dtype=float, copy=True)
    g_arr = np.array(generation_scale, dtype=float, copy=True)
    if (
        t_arr.ndim != 1
        or t_arr.size < 2
        or not np.all(np.isfinite(t_arr))
        or np.any(np.diff(t_arr) <= 0.0)
    ):
        raise ValueError("times_s must be a 1-D strictly increasing finite array")
    if g_arr.shape != t_arr.shape or not np.all(np.isfinite(g_arr)) or np.any(g_arr < 0.0):
        raise ValueError(
            "generation_scale must match times_s shape with finite nonnegative values"
        )
    t_arr.setflags(write=False)
    g_arr.setflags(write=False)

    v0 = float(initial_state.voltage_V if initial_voltage_V is None else initial_voltage_V)
    if not math.isfinite(v0):
        raise ValueError("initial_voltage_V must be finite")
    if v0 != initial_state.voltage_V:
        if isinstance(device, CompiledDevice1D):
            assert isinstance(initial_state, State1D)
            rebased_u = device.rebase_coordinates(
                initial_state.u, initial_state.voltage_V, v0
            )
            cur_state: State1D | IonicState1D = device.state(rebased_u, voltage_V=v0)
        else:
            assert isinstance(initial_state, IonicState1D)
            n_e = 3 * device.node_count
            rebased_e = device.electronic_component.rebase_coordinates(
                initial_state.u[:n_e], initial_state.voltage_V, v0
            )
            cur_state = device.state(
                np.r_[rebased_e, initial_state.u[n_e:]], voltage_V=v0
            )
    else:
        cur_state = initial_state

    support_cols = _left_port_support_indices(device)
    c_geom = geometric_series_capacitance_F_m2(device)

    n_pts = t_arr.size
    v_hist = np.empty(n_pts, dtype=float)
    i_cond_hist = np.empty(n_pts, dtype=float)
    i_disp_hist = np.empty(n_pts, dtype=float)
    i_tot_hist = np.empty(n_pts, dtype=float)
    max_face_hist = np.empty(n_pts, dtype=float)

    v_hist[0] = v0
    i_cond_hist[0] = 0.0
    i_disp_hist[0] = 0.0
    i_tot_hist[0] = 0.0
    max_face_hist[0] = 0.0

    states_out: list[State1D | IonicState1D] = [cur_state]
    diags_out: list[OpenCircuitStepDiagnostics] = []
    cum_dq = 0.0
    cur_v = v0

    for k in range(1, n_pts):
        t_prev = float(t_arr[k - 1])
        t_next = float(t_arr[k])
        dt_s = t_next - t_prev
        g_prev = float(g_arr[k - 1])
        g_next = float(g_arr[k])

        next_state, next_v, raw_diag, dq_step, max_face = _advance_floating_interval(
            device,
            cur_state,
            cur_v,
            t_prev,
            t_next,
            g_prev,
            g_next,
            opts,
            support_cols,
        )
        if max_face > opts.max_current_error_A_m2:
            raise OpenCircuitError(
                f"Step [{t_prev:.3e}, {t_next:.3e}] s violated all-face Maxwell open-circuit gate: "
                f"max|I_total|={max_face:.3e} A/m^2 > {opts.max_current_error_A_m2:.3e} A/m^2"
            )
        cum_dq += dq_step
        step_diag = OpenCircuitStepDiagnostics(
            start_s=t_prev,
            end_s=t_next,
            dt_s=dt_s,
            generation_scale=g_next,
            terminal_voltage_V=next_v,
            newton_iterations=raw_diag.newton_iterations,
            scaled_residual=raw_diag.scaled_residual,
            left_conduction_current_A_m2=raw_diag.left_conduction_current_A_m2,
            left_displacement_current_A_m2=raw_diag.left_displacement_current_A_m2,
            left_total_current_A_m2=raw_diag.left_total_current_A_m2,
            max_face_total_current_A_m2=max_face,
            interval_charge_residual_C_m2=dq_step,
        )
        v_hist[k] = next_v
        i_cond_hist[k] = step_diag.left_conduction_current_A_m2
        i_disp_hist[k] = step_diag.left_displacement_current_A_m2
        i_tot_hist[k] = step_diag.left_total_current_A_m2
        max_face_hist[k] = max_face
        states_out.append(next_state)
        diags_out.append(step_diag)
        cur_state = next_state
        cur_v = next_v

    v_err = abs(cum_dq) / c_geom
    if v_err > opts.max_voltage_error_V:
        raise OpenCircuitError(
            f"Cumulative telescoping charge voltage error {v_err:.3e} V exceeds {opts.max_voltage_error_V:.3e} V"
        )

    for arr in (v_hist, i_cond_hist, i_disp_hist, i_tot_hist, max_face_hist):
        arr.setflags(write=False)

    return OpenCircuitTrajectory(
        times_s=t_arr,
        generation_scale=g_arr,
        terminal_voltage_V=v_hist,
        left_conduction_current_A_m2=i_cond_hist,
        left_displacement_current_A_m2=i_disp_hist,
        left_total_current_A_m2=i_tot_hist,
        max_face_total_current_A_m2=max_face_hist,
        cumulative_charge_residual_C_m2=cum_dq,
        charge_voltage_error_V=v_err,
        geometric_capacitance_F_m2=c_geom,
        states=tuple(states_out),
        step_diagnostics=tuple(diags_out),
        compiled_identity=device.identity,
        certified=True,
    )
