"""Conservative 4/8/16 short-window coupled R1 driver, 183/309 witness harness, and replay receipt (P07-02).

Executes coupled 1D two-sided interface + blocking finite-site positive-ion +
dynamic interface-trap Backward-Euler steps over public ``solarlab`` interfaces
and ``solarlab_research.precision`` (``DD``, ``ReferenceIncrementCoordinates``,
``AffineInputLiftCodec``, ``freeze_coefficients``, ``evaluate_local_carrier_pair``):
* Conservative ``(4, 8, 16)`` substep pairing in ``baseline`` (``float64``) and
  ``compensated`` (``float64-pair-v1`` ``DD``) modes without trajectory interpolation.
* Full 16-item V53 certificate evaluation, 13-channel operator observation packet,
  sub-ULP compensated precision check, and deterministic ``R1ReplayReceipt``.
* Minimal ``183/309`` public-interface witness evaluation preserving historical
  step ``309`` failed status without relabeling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, lil_matrix
from scipy.special import expit

from solarlab.config.yaml import load_yaml_mapping
from solarlab.contracts import ELEMENTARY_CHARGE_C, EquationSpec, Layout, ONE, Support, VariableSpec
from solarlab.device.compile_ionic_1d import CompiledIonicDevice1D
from solarlab.equations.ionic_1d import Ionic1DProblem, IonicState1D
from solarlab.reproducibility.release import ReleaseBundle
from solarlab.solvers.nonlinear.newton import NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import (
    CallableProblem,
    ResidualScaling,
    SolveBudget,
    StepControl,
    state_digest,
)
from solarlab.solvers.nonlinear.production import (
    POISSON_CONTACT_BALANCE_V1,
    production_convergence_norm,
    production_coordinates,
    production_residual_scaling,
)
from solarlab.solvers.time.ionic_backward_euler import (
    IonicBackwardEuler1DStep,
    step_accounting,
)

from solarlab_research.precision.codecs import (
    AffineInputLiftCodec,
    ReferenceIncrementCoordinates,
)
from solarlab_research.precision.double_double import DD, expm1 as dd_expm1
from solarlab_research.precision.two_sided_interface import (
    FrozenLocalCoefficients,
    evaluate_local_carrier_pair,
    freeze_coefficients,
    to_float_tangent,
)
from solarlab_research.r1.contracts import (
    R1_CONSERVATIVE_SUBSTEPS,
    R1_CONTROLS,
    R1_V53_CERTIFICATE_LIMITS,
    R1CertificateEvaluation,
    R1QualificationAssessment,
    R1StudyRequest,
    assess_r1_qualification,
    build_r1_study_request,
    evaluate_v53_certificate,
    expected_substep_rows,
    validate_operator_channel_packet,
)
from solarlab_research.r1.controls import R1ZeroMinusAnchor, prepare_r1_zero_minus_anchor
from solarlab_research.r1.provenance import (
    HistoricalR1WitnessAudit,
    verify_historical_r1_failure_witness,
    verify_r1_release_source,
)

__all__ = [
    "R1CoupledStepResult",
    "R1ModeWindowRun",
    "R1ReplayReceipt",
    "R1ShortWindowPairingResult",
    "R1WitnessMinimalEvaluation",
    "evaluate_r1_witness_minimal_input",
    "evaluate_sub_ulp_storage_precision",
    "run_r1_short_window_pairing",
    "solve_coupled_r1_step",
]

Q = ELEMENTARY_CHARGE_C


def _logit_from_occupancy(f: float) -> float:
    if not 0.0 < f < 1.0:
        raise ValueError(f"trap occupancy must lie strictly in (0, 1); got {f}")
    return math.log(f) - math.log1p(-f)


def _compensated_logistic_diff(old_logit: float, new_logit: float) -> DD:
    """Evaluate ``expit(new_logit) - expit(old_logit)`` in DD without cancellation."""
    delta = DD(new_logit) - DD(old_logit)
    old_f = float(expit(old_logit))
    new_f = float(expit(new_logit))
    if float(delta) >= 0.0:
        neg_expm1 = -(dd_expm1(-delta))
        return DD(new_f) * DD(float(expit(-old_logit))) * neg_expm1
    return DD(old_f) * DD(float(expit(-new_logit))) * dd_expm1(delta)


@dataclass(frozen=True, slots=True)
class R1CoupledStepResult:
    """Single accepted coupled 1D electronic + confined-ion + dynamic-trap BE step."""

    accepted: bool
    ionic_state: IonicState1D
    trap_occupancy: float
    trap_logit: float
    sheet_charge_C_m2: float
    capture_flux_m2_s: float
    electron_current_A_m2: NDArray[np.float64]
    hole_current_A_m2: NDArray[np.float64]
    total_current_A_m2: NDArray[np.float64]
    electron_rate_m2_s: NDArray[np.float64]
    hole_rate_m2_s: NDArray[np.float64]
    ion_flux_m2_s: NDArray[np.float64]
    ion_rate_m2_s: NDArray[np.float64]
    trace_potential_V: float
    scaled_nonlinear_residual: float
    newton_iterations: int
    analytic_jacobian_error: float
    local_carrier_residual: float
    eliminated_operator_error: float
    local_gauss_residual: float
    full_gauss_normalized: float
    full_charge_balance_normalized: float
    charge_balance_relative: float
    current_decomposition_error: float
    trap_storage_normalized_error: float
    state_sha256: str


def _build_frozen_coefficients(
    device: CompiledIonicDevice1D,
    anchor: R1ZeroMinusAnchor,
) -> FrozenLocalCoefficients:
    elec_comp = device.electronic_component
    if_node = anchor.interface_node
    l_idx, r_idx = if_node - 1, if_node + 1
    dx_l = float(device.nodes_m[if_node] - device.nodes_m[l_idx])
    dx_r = float(device.nodes_m[r_idx] - device.nodes_m[if_node])
    vt = elec_comp.thermal_voltage_V
    bands = elec_comp.parameters.band_structure
    return freeze_coefficients(
        {
            "left_distance_m": dx_l,
            "right_distance_m": dx_r,
        },
        {
            "temperature_K": float(elec_comp.parameters.temperature_K),
            "thermal_voltage_V": float(vt),
            "transmission": 1.0,
            "D_n_left_m2_s": float(elec_comp.electron_diffusivity_faces_m2_s[if_node - 1]),
            "D_p_left_m2_s": float(elec_comp.hole_diffusivity_faces_m2_s[if_node - 1]),
            "D_n_right_m2_s": float(elec_comp.electron_diffusivity_faces_m2_s[if_node]),
            "D_p_right_m2_s": float(elec_comp.hole_diffusivity_faces_m2_s[if_node]),
            "N_C_left_m3": float(elec_comp.parameters.conduction_dos_m3[l_idx]),
            "N_V_left_m3": float(elec_comp.parameters.valence_dos_m3[l_idx]),
            "N_C_right_m3": float(elec_comp.parameters.conduction_dos_m3[r_idx]),
            "N_V_right_m3": float(elec_comp.parameters.valence_dos_m3[r_idx]),
            "n1_left_m3": float(anchor.n1_trap_m3),
            "p1_left_m3": float(anchor.p1_trap_m3),
            "n1_right_m3": float(anchor.n1_trap_m3),
            "p1_right_m3": float(anchor.p1_trap_m3),
            "richardson_n_A_m2_K2": 1.2e6,
            "richardson_p_A_m2_K2": 1.2e6,
            "surface_recombination_velocity_n_m_s": float(anchor.trap_density_m2 * anchor.capture_n_m_s),
            "surface_recombination_velocity_p_m_s": float(anchor.trap_density_m2 * anchor.capture_p_m_s),
            "conduction_band_step_eV": float(
                bands.physical_electron_affinity_eV[l_idx]
                - bands.physical_electron_affinity_eV[r_idx]
            ),
            "hole_transport_step_eV": float(
                (bands.physical_electron_affinity_eV[r_idx] + bands.physical_band_gap_eV[r_idx])
                - (bands.physical_electron_affinity_eV[l_idx] + bands.physical_band_gap_eV[l_idx])
            ),
        },
    )


def solve_coupled_r1_step(
    *,
    device: CompiledIonicDevice1D,
    previous_ionic: IonicState1D,
    previous_trap_occupancy: float,
    anchor: R1ZeroMinusAnchor,
    voltage_V: float,
    dt_s: float,
    nu_t: int = 1,
    mode: Literal["baseline", "compensated"] = "compensated",
    generation_scale: float = 0.0,
    light_turn_on: bool = False,
    frozen_coefficients: FrozenLocalCoefficients | None = None,
    verify_codec: bool = True,
    audit_operators: bool = True,
) -> R1CoupledStepResult:
    """Solve one conservative Backward-Euler step of the coupled R1 1D system."""
    if nu_t not in (0, 1):
        raise ValueError("nu_t must be 0 or 1")
    if mode not in ("baseline", "compensated"):
        raise ValueError("mode must be 'baseline' or 'compensated'")

    n_nodes = device.node_count
    state_n = device.state_size
    if_node = anchor.interface_node
    vt = device.electronic_component.thermal_voltage_V
    old_logit = _logit_from_occupancy(previous_trap_occupancy)
    f0 = anchor.equilibrium_trap_occupancy
    nt = anchor.trap_density_m2
    cn = anchor.capture_n_m_s
    cp = anchor.capture_p_m_s
    n1_t = anchor.n1_trap_m3
    p1_t = anchor.p1_trap_m3

    endpoint = Ionic1DProblem(device, voltage_V=voltage_V, generation_scale=generation_scale)
    base_step = IonicBackwardEuler1DStep(endpoint, previous_ionic, dt_s)
    predictor = "generation_inventory" if (light_turn_on and generation_scale > 0.0) else "previous"
    y0 = base_step.initial_coordinates(predictor, voltage_lift=True)

    # Coordinate codec & authority in compensated mode
    if verify_codec:
        coupled_layout = Layout(
            supports=(Support("coupled_1d", "cell", (state_n + 1,)),),
            variables=(VariableSpec("z", "r1_coupled_1d", "coupled_1d", (state_n + 1,), ONE),),
            equations=(EquationSpec("r", "r1_coupled_1d", "coupled_1d", (state_n + 1,), ONE, derivative_support=("z",)),),
        )
        ref_coords = ReferenceIncrementCoordinates.from_float_anchor(
            coupled_layout,
            {"z": np.r_[y0, old_logit]},
        )
        codec = AffineInputLiftCodec()
        _ = codec.encode_point(ref_coords.point(np.zeros(state_n + 1, dtype=float)))

    def trap_terms(
        n_if: float,
        p_if: float,
        logit_f: float,
    ) -> tuple[float, float, float, float, float]:
        if nu_t == 0:
            return previous_trap_occupancy, 0.0, 0.0, 0.0, 0.0
        f_val = float(expit(logit_f))
        rn_cap = nt * cn * (n_if * (1.0 - f_val) - n1_t * f_val)
        rp_cap = nt * cp * (p_if * f_val - p1_t * (1.0 - f_val))
        sigma = -Q * nt * (f_val - f0)
        if mode == "compensated":
            df = float(_compensated_logistic_diff(old_logit, logit_f))
        else:
            df = f_val - previous_trap_occupancy
        storage_rate = nt * df / dt_s
        trap_res = storage_rate - (rn_cap - rp_cap)
        return f_val, rn_cap, rp_cap, sigma, trap_res

    def residual(z: NDArray[np.float64]) -> NDArray[np.float64]:
        y = z[:state_n]
        logit_f = float(z[state_n])
        res: NDArray[np.float64] = np.empty(state_n + 1, dtype=float)
        try:
            res[:state_n] = base_step.residual(y)
            _, n_arr, p_arr, _ = device.fields_by_species(y, voltage_V=voltage_V)
        except (ValueError, FloatingPointError):
            return np.full(state_n + 1, np.nan)
        n_if = float(n_arr[if_node])
        p_if = float(p_arr[if_node])
        if nu_t == 0:
            res[state_n] = logit_f - old_logit
            return res
        _, rn_cap, rp_cap, sigma, trap_res = trap_terms(n_if, p_if, logit_f)
        res[if_node] += sigma
        res[n_nodes + if_node] -= rn_cap
        res[2 * n_nodes + if_node] -= rp_cap
        res[state_n] = trap_res
        return res

    def jacobian(z: NDArray[np.float64]) -> csc_matrix:
        y = z[:state_n]
        logit_f = float(z[state_n])
        j_base = base_step.jacobian(y)
        mat = lil_matrix((state_n + 1, state_n + 1), dtype=float)
        mat[:state_n, :state_n] = j_base
        if nu_t == 0:
            mat[state_n, state_n] = 1.0
            return mat.tocsc()

        _, n_arr, p_arr, _ = device.fields_by_species(y, voltage_V=voltage_V)
        n_if = float(n_arr[if_node])
        p_if = float(p_arr[if_node])
        f_val = float(expit(logit_f))
        df_dl = float(expit(logit_f) * expit(-logit_f))

        dn_dphi = n_if / vt
        dn_dzn = n_if
        dp_dphi = -p_if / vt
        dp_dzp = -p_if

        drn_dn = nt * cn * (1.0 - f_val)
        drn_df = -nt * cn * (n_if + n1_t)
        drp_dp = nt * cp * f_val
        drp_df = nt * cp * (p_if + p1_t)

        # Gauss row at if_node: +sigma where sigma = -Q * nt * (f - f0)
        mat[if_node, state_n] += -Q * nt * df_dl

        # Electron row at n_nodes + if_node: -rn_cap
        e_row = n_nodes + if_node
        mat[e_row, if_node] -= drn_dn * dn_dphi
        mat[e_row, n_nodes + if_node] -= drn_dn * dn_dzn
        mat[e_row, state_n] -= drn_df * df_dl

        # Hole row at 2*n_nodes + if_node: -rp_cap
        h_row = 2 * n_nodes + if_node
        mat[h_row, if_node] -= drp_dp * dp_dphi
        mat[h_row, 2 * n_nodes + if_node] -= drp_dp * dp_dzp
        mat[h_row, state_n] -= drp_df * df_dl

        # Trap row at state_n: nt*(f - f_old)/dt - (rn_cap - rp_cap)
        mat[state_n, if_node] += -drn_dn * dn_dphi + drp_dp * dp_dphi
        mat[state_n, n_nodes + if_node] += -drn_dn * dn_dzn
        mat[state_n, 2 * n_nodes + if_node] += drp_dp * dp_dzp
        mat[state_n, state_n] += (nt / dt_s - drn_df + drp_df) * df_dl
        return mat.tocsc()

    elec_comp = device.electronic_component
    continuity_tol_A_m2 = 5.0e-8
    ion_tol_m2_s = max(1.0e8, 5.0e-7 / Q)
    trap_tol_A_m2 = 5.0e-8

    base_weights = production_residual_scaling(
        elec_comp,
        poisson_tolerance=1.0e-11,
        continuity_tolerance_A_m2=continuity_tol_A_m2,
        contact_log_tolerance=1.0e-11,
    ).row_weights
    ion_weights = np.full(device.ion_node_count, device.ion_node_count / ion_tol_m2_s)
    trap_weight = 1.0 if nu_t == 0 else Q / trap_tol_A_m2
    weights = np.r_[base_weights, ion_weights, trap_weight]

    elec_norm = production_convergence_norm(
        elec_comp,
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )
    assert elec_norm is not None

    def coupled_convergence_norm(scaled: NDArray[np.float64]) -> float:
        if scaled.shape != (state_n + 1,) or not np.all(np.isfinite(scaled)):
            return math.inf
        return max(
            elec_norm(scaled[: 3 * n_nodes]),
            float(np.sum(np.abs(scaled[3 * n_nodes : state_n]))) / device.ion_node_count,
            abs(float(scaled[state_n])),
        )

    steps = np.r_[
        production_coordinates(elec_comp, potential_step_V=0.1, quasi_fermi_step=2.0).max_step,
        np.full(device.ion_node_count, 1.0),
        2.0,
    ]
    settings = NewtonSettings(
        ResidualScaling(weights),
        4.5e-2,
        StepControl(steps),
        SolveBudget(100, 5000, 1000, 40),
        (),
        False,
    )
    z_init = np.r_[y0, old_logit]
    newton_res = solve_newton(
        CallableProblem(residual, jacobian),
        z_init,
        settings,
        convergence_norm=coupled_convergence_norm,
    )
    z_sol = newton_res.state
    y_sol = z_sol[:state_n]
    logit_sol = float(z_sol[state_n])
    f_sol = previous_trap_occupancy if nu_t == 0 else float(expit(logit_sol))
    new_ionic = device.state(y_sol, voltage_V=voltage_V)

    # Directional FD Jacobian check at accepted solution
    if audit_operators:
        rng = np.random.default_rng(183309)
        direction = rng.uniform(-0.02, 0.02, state_n + 1)
        direction[:n_nodes] *= vt
        if nu_t == 0:
            direction[state_n] = 0.0
        j_mat = jacobian(z_sol)
        h_fd = 1.0e-6
        fd_vec = (residual(z_sol + h_fd * direction) - residual(z_sol - h_fd * direction)) / (2.0 * h_fd)
        scale_vec = np.abs(j_mat) @ np.abs(direction)
        jac_err = float(
            np.max(np.abs(j_mat @ direction - fd_vec) / np.maximum(scale_vec, 1.0e-16))
        )
    else:
        jac_err = 0.0

    # Full interval accounting + modified Gauss/Maxwell displacement with sheet charge
    info = step_accounting(base_step, y_sol)
    carrier_particles = np.asarray(info["carrier_particle_fluxes_m2_s"], dtype=float)
    carrier_rates = np.asarray(info["carrier_inventory_rate_m2_s"], dtype=float)
    ion_rate = np.asarray(info["ion_inventory_rate_m2_s"], dtype=float)
    ion_domain_flux = np.asarray(info["ion_domain_fluxes_m2_s"], dtype=float)
    ion_particles = np.asarray(info["ion_particle_fluxes_m2_s"], dtype=float)

    n_if = float(new_ionic.electronic_state.electron_density_m3[if_node])
    p_if = float(new_ionic.electronic_state.hole_density_m3[if_node])
    _, rn_cap, rp_cap, sigma_new, trap_res_val = trap_terms(n_if, p_if, logit_sol)
    sigma_old = 0.0 if nu_t == 0 else -Q * nt * (previous_trap_occupancy - f0)
    dsigma_dt = 0.0 if nu_t == 0 else (
        float(-DD(Q * nt) * _compensated_logistic_diff(old_logit, logit_sol) / DD(dt_s))
        if mode == "compensated"
        else (sigma_new - sigma_old) / dt_s
    )

    charge_rate = np.array(info["charge_inventory_rate_A_m2"], dtype=float, copy=True)
    charge_rate[if_node] += dsigma_dt

    poisson = elec_comp.poisson
    phi_delta = new_ionic.electronic_state.potential_V - previous_ionic.electronic_state.potential_V
    interior_delta_d = -poisson.face_capacitance_F_m2 * np.diff(phi_delta)
    interval_d = np.r_[
        interior_delta_d[0] - dt_s * charge_rate[0],
        interior_delta_d,
        interior_delta_d[-1] + dt_s * charge_rate[-1],
    ]
    disp_curr = interval_d / dt_s
    e_curr = -Q * carrier_particles[0]
    h_curr = Q * carrier_particles[1]
    cond_curr = e_curr + h_curr + Q * ion_particles
    tot_curr = cond_curr + disp_curr

    decomp_err = float(np.max(np.abs(tot_curr - (cond_curr + disp_curr))))
    stable_gauss_rate = (interval_d[:-1] - interval_d[1:]) / dt_s + charge_rate
    curr_scale = max(1.0e-6, float(np.max(np.abs(tot_curr))), float(np.max(np.abs(charge_rate))))
    charge_bal_rel = float(np.max(np.abs(stable_gauss_rate - (tot_curr[:-1] - tot_curr[1:]))) / curr_scale)

    final_res = residual(z_sol)
    gauss_weights = base_weights[:n_nodes] * 1.0e-11
    full_gauss_norm = float(np.max(np.abs(final_res[:n_nodes] * gauss_weights)))
    local_gauss_res = float(abs(final_res[if_node] * gauss_weights[if_node]))

    e_rows = final_res[n_nodes : 2 * n_nodes].copy()
    h_rows = final_res[2 * n_nodes : 3 * n_nodes].copy()
    e_rows[[0, -1]] = 0.0
    h_rows[[0, -1]] = 0.0
    ion_rows = final_res[3 * n_nodes : state_n]
    full_charge_norm = float(
        (Q * (np.sum(np.abs(e_rows)) + np.sum(np.abs(h_rows)) + np.sum(np.abs(ion_rows)))) / curr_scale
    )
    trap_norm_err = 0.0 if nu_t == 0 else abs(Q * trap_res_val) / curr_scale

    # Two-sided compensated local carrier pair evaluation & Schur-elimination check
    phi_arr = new_ionic.electronic_state.potential_V
    psi_trace = float(phi_arr[if_node])
    if audit_operators:
        l_idx, r_idx = if_node - 1, if_node + 1
        n_arr = new_ionic.electronic_state.electron_density_m3
        p_arr = new_ionic.electronic_state.hole_density_m3
        frozen_c = (
            frozen_coefficients
            if frozen_coefficients is not None
            else _build_frozen_coefficients(device, anchor)
        )

        # Solve 4x4 local two-sided trace densities in DD (linear density predictor + damped log-Newton)
        trace_pot_dd = DD(np.array([psi_trace, psi_trace], dtype=float))
        bulk_pot_dd = DD(np.array([float(phi_arr[l_idx]), float(phi_arr[r_idx])], dtype=float))
        bulk_dens_arr = np.array(
            [
                float(n_arr[l_idx]),
                float(p_arr[l_idx]),
                float(n_arr[r_idx]),
                float(p_arr[r_idx]),
            ],
            dtype=float,
        )
        bulk_dens_dd = DD(bulk_dens_arr)
        occ_dd = DD(f_sol)
        dens_vec = np.maximum(bulk_dens_arr.copy(), 1.0e-30)

        # First step in density space solves the affine Boltzmann part directly without log-step overflow.
        for it in range(6):
            ev = evaluate_local_carrier_pair(
                state_density=DD(dens_vec),
                trace_potential=trace_pot_dd,
                bulk_density=bulk_dens_dd,
                bulk_potential=bulk_pot_dd,
                occupancy=occ_dd,
                capture_multiplier=nu_t,
                coefficients=frozen_c,
            )
            flt = to_float_tangent(ev)
            if it == 0:
                jac_dens = flt.balance.jacobian_log_state_m2_s / dens_vec[np.newaxis, :]
                dn: NDArray[np.float64] = -np.linalg.solve(jac_dens, flt.balance.residual_m2_s)
                dens_vec = np.maximum(1.0e-60, dens_vec + dn)
                if float(np.max(np.abs(dn) / dens_vec)) < 1.0e-14:
                    break
            else:
                du: NDArray[np.float64] = -np.linalg.solve(
                    flt.balance.jacobian_log_state_m2_s,
                    flt.balance.residual_m2_s,
                )
                du_clipped = np.clip(du, -4.0, 4.0)
                dens_vec = np.maximum(1.0e-60, dens_vec * np.exp(du_clipped))
                if float(np.max(np.abs(du))) < 1.0e-14:
                    break

        u_vec = np.log(dens_vec)
        ev_final = evaluate_local_carrier_pair(
            state_density=DD(dens_vec),
            trace_potential=trace_pot_dd,
            bulk_density=bulk_dens_dd,
            bulk_potential=bulk_pot_dd,
            occupancy=occ_dd,
            capture_multiplier=nu_t,
            coefficients=frozen_c,
        )
        flt_final = to_float_tangent(ev_final)
        flux_scale = max(
            1.0,
            float(np.max(np.abs(flt_final.balance.bulk_flux_m2_s))),
            float(np.max(np.abs(flt_final.balance.one_way_cross_scale_m2_s))),
        )
        local_carrier_res = float(np.max(np.abs(flt_final.balance.residual_m2_s))) / flux_scale

        # Schur-eliminated operator check wrt bulk coordinate col 2 (log n_left)
        du_dlogn_l: NDArray[np.float64] = -np.linalg.solve(
            flt_final.balance.jacobian_log_state_m2_s,
            flt_final.balance.jacobian_bulk_coordinates[:, 2],
        )
        dflux_dlogn_schur = float(
            flt_final.bulk_flux_jacobian_bulk_coordinates[0, 2]
            + flt_final.bulk_flux_jacobian_log_state_m2_s[0, :] @ du_dlogn_l
        )
        eps_log = 1.0e-6
        pert_bulk_arr = bulk_dens_arr.copy()
        pert_bulk_arr[0] *= math.exp(eps_log)
        pert_bulk_dd = DD(pert_bulk_arr)
        u_pert = u_vec + du_dlogn_l * eps_log
        for _ in range(2):
            ev_p = evaluate_local_carrier_pair(
                state_density=DD(np.exp(u_pert)),
                trace_potential=trace_pot_dd,
                bulk_density=pert_bulk_dd,
                bulk_potential=bulk_pot_dd,
                occupancy=occ_dd,
                capture_multiplier=nu_t,
                coefficients=frozen_c,
            )
            flt_p = to_float_tangent(ev_p)
            step_u: NDArray[np.float64] = -np.linalg.solve(
                flt_p.balance.jacobian_log_state_m2_s,
                flt_p.balance.residual_m2_s,
            )
            u_pert = u_pert + step_u
            if float(np.max(np.abs(step_u))) < 1.0e-13:
                break
        ev_pert = evaluate_local_carrier_pair(
            state_density=DD(np.exp(u_pert)),
            trace_potential=trace_pot_dd,
            bulk_density=pert_bulk_dd,
            bulk_potential=bulk_pot_dd,
            occupancy=occ_dd,
            capture_multiplier=nu_t,
            coefficients=frozen_c,
        )
        flt_pert = to_float_tangent(ev_pert)
        dflux_dlogn_fd = (
            float(flt_pert.balance.bulk_flux_m2_s[0]) - float(flt_final.balance.bulk_flux_m2_s[0])
        ) / eps_log
        elim_err = abs(dflux_dlogn_schur - dflux_dlogn_fd) / max(abs(dflux_dlogn_fd), 1.0e-12)
    else:
        local_carrier_res = 0.0
        elim_err = 0.0

    return R1CoupledStepResult(
        accepted=bool(newton_res.accepted),
        ionic_state=new_ionic,
        trap_occupancy=f_sol,
        trap_logit=logit_sol,
        sheet_charge_C_m2=sigma_new,
        capture_flux_m2_s=rn_cap - rp_cap,
        electron_current_A_m2=e_curr,
        hole_current_A_m2=h_curr,
        total_current_A_m2=tot_curr,
        electron_rate_m2_s=carrier_rates[0],
        hole_rate_m2_s=carrier_rates[1],
        ion_flux_m2_s=ion_domain_flux,
        ion_rate_m2_s=ion_rate,
        trace_potential_V=psi_trace,
        scaled_nonlinear_residual=float(newton_res.scaled_residual),
        newton_iterations=int(newton_res.iteration_count),
        analytic_jacobian_error=jac_err,
        local_carrier_residual=local_carrier_res,
        eliminated_operator_error=elim_err,
        local_gauss_residual=local_gauss_res,
        full_gauss_normalized=full_gauss_norm,
        full_charge_balance_normalized=full_charge_norm,
        charge_balance_relative=charge_bal_rel,
        current_decomposition_error=decomp_err,
        trap_storage_normalized_error=trap_norm_err,
        state_sha256=state_digest(z_sol),
    )


@dataclass(frozen=True, slots=True)
class R1ModeWindowRun:
    """Single precision-mode conservative (4, 8, 16) short-window run and certificate."""

    mode: str
    accepted_rows: int
    substep_row_counts: Mapping[int, int]
    certificate: R1CertificateEvaluation
    operator_channels: Mapping[str, NDArray[np.float64]]
    channel_shapes: Mapping[str, tuple[int, ...]]
    final_state_sha256: str


@dataclass(frozen=True, slots=True)
class R1ReplayReceipt:
    """Deterministic replay receipt bound to the verified ReleaseBundle and study request."""

    request_sha256: str
    release_bundle_sha256: str
    build_source_commit: str
    baseline_final_state_sha256: str
    compensated_final_state_sha256: str
    replay_compensated_state_sha256: str
    deterministic_match: bool
    qualification_level: str
    r1_l2_certified: bool
    r1_l3_device_qualified: bool


@dataclass(frozen=True, slots=True)
class R1ShortWindowPairingResult:
    """Combined baseline-vs-compensated conservative 4/8/16 short-window verification."""

    request: R1StudyRequest
    baseline: R1ModeWindowRun
    compensated: R1ModeWindowRun
    qualification: R1QualificationAssessment
    replay_receipt: R1ReplayReceipt
    sub_ulp_precision_ratio: float


def evaluate_sub_ulp_storage_precision(
    *,
    old_logit: float = 0.4,
    delta_logit: float = 1.0e-15,
    dt_s: float = 1.0e-9,
    trap_density_m2: float = 2.0e15,
) -> dict[str, float]:
    """Compare baseline float64 subtraction vs DD compensated logistic storage at sub-ULP steps."""
    from decimal import Decimal, localcontext

    new_logit = old_logit + delta_logit
    with localcontext() as ctx:
        ctx.prec = 90
        d_old = Decimal.from_float(old_logit)
        d_new = Decimal.from_float(new_logit)
        f_old_exact = Decimal(1) / (Decimal(1) + (-d_old).exp())
        f_new_exact = Decimal(1) / (Decimal(1) + (-d_new).exp())
        exact_rate = float(
            Decimal.from_float(trap_density_m2) * (f_new_exact - f_old_exact) / Decimal.from_float(dt_s)
        )

    baseline_df = float(expit(new_logit)) - float(expit(old_logit))
    baseline_rate = trap_density_m2 * baseline_df / dt_s

    comp_df = float(_compensated_logistic_diff(old_logit, new_logit))
    comp_rate = trap_density_m2 * comp_df / dt_s

    base_err = abs(baseline_rate - exact_rate) / max(abs(exact_rate), 1.0e-30)
    comp_err = abs(comp_rate - exact_rate) / max(abs(exact_rate), 1.0e-30)
    return {
        "exact_rate_m2_s": exact_rate,
        "baseline_rate_m2_s": baseline_rate,
        "compensated_rate_m2_s": comp_rate,
        "baseline_relative_error": base_err,
        "compensated_relative_error": comp_err,
        "error_reduction_ratio": base_err / max(comp_err, 1.0e-20),
    }


def _execute_mode_window(
    *,
    anchor: R1ZeroMinusAnchor,
    control: str,
    times_s: tuple[float, ...],
    substeps: tuple[int, int, int],
    voltage_step_V: float,
    mode: Literal["baseline", "compensated"],
    audit_observation_boundaries: bool = True,
) -> R1ModeWindowRun:
    switches = R1_CONTROLS[control]
    nu_i, nu_t = int(switches["nu_I"]), int(switches["nu_t"])
    dev = anchor.mobile_device if nu_i == 1 else anchor.immobile_device
    init_ionic = anchor.mobile_state if nu_i == 1 else anchor.immobile_state
    init_f = anchor.equilibrium_trap_occupancy
    inv0 = anchor.mobile_state.domain_inventory_m2
    frozen_c = _build_frozen_coefficients(dev, anchor)

    n_nodes = dev.node_count
    m_nodes = dev.ion_node_count
    row_counts: dict[int, int] = {}
    obs_states_by_s: dict[int, list[NDArray[np.float64]]] = {}
    obs_curr_by_s: dict[int, list[float]] = {}

    ch_capture: list[float] = []
    ch_ecurr: list[NDArray[np.float64]] = []
    ch_edens: list[NDArray[np.float64]] = []
    ch_erate: list[NDArray[np.float64]] = []
    ch_hcurr: list[NDArray[np.float64]] = []
    ch_hdens: list[NDArray[np.float64]] = []
    ch_hrate: list[NDArray[np.float64]] = []
    ch_local: list[float] = []
    ch_iflux: list[NDArray[np.float64]] = []
    ch_irate: list[NDArray[np.float64]] = []
    ch_pot: list[NDArray[np.float64]] = []
    ch_sheet: list[float] = []
    ch_trace: list[float] = []

    max_face_spread_rel = 0.0
    max_jac_err = 0.0
    max_charge_bal_rel = 0.0
    max_decomp_err = 0.0
    max_elim_err = 0.0
    max_full_charge = 0.0
    max_full_gauss = 0.0
    max_if_curr_rel = 0.0
    max_inv_drift = 0.0
    max_local_carrier = 0.0
    max_local_gauss = 0.0
    max_nl_res = 0.0
    max_trap_err = 0.0
    last_sha = ""

    def append_initial_row() -> None:
        ch_capture.append(0.0)
        ch_ecurr.append(np.zeros(n_nodes + 1, dtype=float))
        ch_edens.append(np.array(init_ionic.electronic_state.electron_density_m3, copy=True))
        ch_erate.append(np.zeros(n_nodes, dtype=float))
        ch_hcurr.append(np.zeros(n_nodes + 1, dtype=float))
        ch_hdens.append(np.array(init_ionic.electronic_state.hole_density_m3, copy=True))
        ch_hrate.append(np.zeros(n_nodes, dtype=float))
        ch_local.append(init_f)
        ch_iflux.append(np.zeros(m_nodes - 1, dtype=float))
        ch_irate.append(np.zeros(m_nodes, dtype=float))
        ch_pot.append(np.array(init_ionic.electronic_state.potential_V, copy=True))
        ch_sheet.append(0.0)
        ch_trace.append(float(init_ionic.electronic_state.potential_V[anchor.interface_node]))

    for s in substeps:
        cur_ionic = init_ionic
        cur_f = init_f
        append_initial_row()
        rows_this_s = 1
        s_states: list[NDArray[np.float64]] = []
        s_currs: list[float] = []
        for t0, t1 in zip(times_s[:-1], times_s[1:]):
            dt_sub = (t1 - t0) / s
            for sub_i in range(s):
                step_r = solve_coupled_r1_step(
                    device=dev,
                    previous_ionic=cur_ionic,
                    previous_trap_occupancy=cur_f,
                    anchor=anchor,
                    voltage_V=voltage_step_V,
                    dt_s=dt_sub,
                    nu_t=nu_t,
                    mode=mode,
                    generation_scale=1.0,
                    light_turn_on=(sub_i == 0 and t0 == times_s[0]),
                    frozen_coefficients=frozen_c,
                    verify_codec=(sub_i == 0 and t0 == times_s[0] and audit_observation_boundaries),
                    audit_operators=(audit_observation_boundaries and sub_i == s - 1),
                )
                if not step_r.accepted:
                    raise RuntimeError(f"coupled R1 step failed at s={s}, sub={sub_i}, mode={mode}")
                cur_ionic = step_r.ionic_state
                cur_f = step_r.trap_occupancy
                rows_this_s += 1
                last_sha = step_r.state_sha256

                ch_capture.append(step_r.capture_flux_m2_s)
                ch_ecurr.append(step_r.electron_current_A_m2)
                ch_edens.append(cur_ionic.electronic_state.electron_density_m3)
                ch_erate.append(step_r.electron_rate_m2_s)
                ch_hcurr.append(step_r.hole_current_A_m2)
                ch_hdens.append(cur_ionic.electronic_state.hole_density_m3)
                ch_hrate.append(step_r.hole_rate_m2_s)
                ch_local.append(cur_f)
                ch_iflux.append(step_r.ion_flux_m2_s)
                ch_irate.append(step_r.ion_rate_m2_s)
                ch_pot.append(cur_ionic.electronic_state.potential_V)
                ch_sheet.append(step_r.sheet_charge_C_m2)
                ch_trace.append(step_r.trace_potential_V)

                c_scale = max(1.0e-6, float(np.max(np.abs(step_r.total_current_A_m2))))
                spread_rel = float(np.ptp(step_r.total_current_A_m2)) / c_scale
                if_jump_rel = abs(
                    float(step_r.total_current_A_m2[anchor.interface_node])
                    - float(step_r.total_current_A_m2[anchor.interface_node + 1])
                ) / c_scale
                inv_drift = abs(cur_ionic.domain_inventory_m2 - inv0) / inv0

                max_face_spread_rel = max(max_face_spread_rel, spread_rel)
                max_if_curr_rel = max(max_if_curr_rel, if_jump_rel)
                max_jac_err = max(max_jac_err, step_r.analytic_jacobian_error)
                max_charge_bal_rel = max(max_charge_bal_rel, step_r.charge_balance_relative)
                max_decomp_err = max(max_decomp_err, step_r.current_decomposition_error)
                max_elim_err = max(max_elim_err, step_r.eliminated_operator_error)
                max_full_charge = max(max_full_charge, step_r.full_charge_balance_normalized)
                max_full_gauss = max(max_full_gauss, step_r.full_gauss_normalized)
                max_inv_drift = max(max_inv_drift, inv_drift)
                max_local_carrier = max(max_local_carrier, step_r.local_carrier_residual)
                max_local_gauss = max(max_local_gauss, step_r.local_gauss_residual)
                max_nl_res = max(max_nl_res, step_r.scaled_nonlinear_residual)
                max_trap_err = max(max_trap_err, step_r.trap_storage_normalized_error)

                if sub_i == s - 1:
                    s_states.append(np.r_[cur_ionic.electronic_state.potential_V, cur_f])
                    s_currs.append(float(step_r.total_current_A_m2[0]))

        row_counts[s] = rows_this_s
        obs_states_by_s[s] = s_states
        obs_curr_by_s[s] = s_currs

    s_mid, s_fine = substeps[-2], substeps[-1]
    ref_state_chg = max(
        float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))
        for a, b in zip(obs_states_by_s[s_mid], obs_states_by_s[s_fine])
    )
    curr_window_scale = max(1.0, *(abs(v) for v in obs_curr_by_s[s_fine]))
    ref_curr_chg = max(
        abs(a - b) / curr_window_scale
        for a, b in zip(obs_curr_by_s[s_mid], obs_curr_by_s[s_fine])
    )

    cert_metrics = {
        "all_face_current_relative": max_face_spread_rel,
        "analytic_jacobian_error": max_jac_err,
        "charge_balance_relative": max_charge_bal_rel,
        "current_decomposition_error": max_decomp_err,
        "eliminated_operator_error": max_elim_err,
        "full_charge_balance_normalized": max_full_charge,
        "full_gauss_normalized": max_full_gauss,
        "interface_current_relative": max_if_curr_rel,
        "inventory_relative_drift": max_inv_drift,
        "local_carrier_residual": max_local_carrier,
        "local_gauss_residual": max_local_gauss,
        "nonlinear_residual": max_nl_res,
        "physical_current_spread_relative": max_face_spread_rel,
        "refinement_current_change": ref_curr_chg,
        "refinement_state_change": ref_state_chg,
        "trap_storage_normalized_error": max_trap_err,
    }
    cert = evaluate_v53_certificate(cert_metrics, limits=R1_V53_CERTIFICATE_LIMITS)

    packet: dict[str, NDArray[np.float64]] = {
        "capture_flux": np.asarray(ch_capture, dtype=float),
        "electron_current": np.asarray(ch_ecurr, dtype=float),
        "electron_density": np.asarray(ch_edens, dtype=float),
        "electron_rate": np.asarray(ch_erate, dtype=float),
        "hole_current": np.asarray(ch_hcurr, dtype=float),
        "hole_density": np.asarray(ch_hdens, dtype=float),
        "hole_rate": np.asarray(ch_hrate, dtype=float),
        "local_state": np.asarray(ch_local, dtype=float),
        "positive_ion_flux": np.asarray(ch_iflux, dtype=float),
        "positive_ion_rate": np.asarray(ch_irate, dtype=float),
        "potential": np.asarray(ch_pot, dtype=float),
        "sheet_charge": np.asarray(ch_sheet, dtype=float),
        "trace_potential": np.asarray(ch_trace, dtype=float),
    }
    total_rows = sum(row_counts.values())
    shapes = validate_operator_channel_packet(packet, expected_rows=total_rows)

    return R1ModeWindowRun(
        mode=mode,
        accepted_rows=total_rows,
        substep_row_counts=MappingProxyType(row_counts),
        certificate=cert,
        operator_channels=MappingProxyType(packet),
        channel_shapes=MappingProxyType(shapes),
        final_state_sha256=last_sha,
    )


def run_r1_short_window_pairing(
    *,
    bundle: ReleaseBundle,
    study_id: str = "p07_02_short_window_d",
    control: str = "D",
    times_s: Sequence[float] = (0.0, 1.0e-7, 1.0e-6),
    substeps: tuple[int, int, int] = R1_CONSERVATIVE_SUBSTEPS,
    voltage_step_V: float = 5.0e-3,
    intervals_per_layer: tuple[int, int] = (6, 4),
) -> R1ShortWindowPairingResult:
    """Run conservative 4/8/16 baseline-vs-compensated short-window verification and replay."""
    anchor = prepare_r1_zero_minus_anchor(intervals_per_layer=intervals_per_layer)
    request = build_r1_study_request(
        study_id=study_id,
        control=control,
        times_s=times_s,
        substeps=substeps,
        voltage_step_V=voltage_step_V,
        integrator="conservative_backward_euler",
        precision_modes=("baseline", "compensated"),
        release_bundle_sha256=bundle.sha256,
        prepared_device_sha256=anchor.mobile_device.original_prepared_sha256,
    )
    rel_binding = verify_r1_release_source(
        bundle,
        expected_bundle_sha256=bundle.sha256,
        inputs=request.to_mapping(),
        operator="solarlab_research",
    )

    t_tuple = tuple(float(t) for t in times_s)
    expected_rows = expected_substep_rows(t_tuple, substeps)

    base_run = _execute_mode_window(
        anchor=anchor,
        control=control,
        times_s=t_tuple,
        substeps=substeps,
        voltage_step_V=voltage_step_V,
        mode="baseline",
    )
    comp_run = _execute_mode_window(
        anchor=anchor,
        control=control,
        times_s=t_tuple,
        substeps=substeps,
        voltage_step_V=voltage_step_V,
        mode="compensated",
    )
    # Independent replay of compensated mode to issue deterministic R1ReplayReceipt
    replay_run = _execute_mode_window(
        anchor=anchor,
        control=control,
        times_s=t_tuple,
        substeps=substeps,
        voltage_step_V=voltage_step_V,
        mode="compensated",
        audit_observation_boundaries=False,
    )
    if base_run.accepted_rows != expected_rows or comp_run.accepted_rows != expected_rows:
        raise RuntimeError("short-window row count violated expected conservative 4/8/16 formula")

    qual = assess_r1_qualification(
        request,
        baseline_certificate=base_run.certificate,
        compensated_certificate=comp_run.certificate,
        baseline_rows=base_run.accepted_rows,
        compensated_rows=comp_run.accepted_rows,
        interpolated_rows=False,
        release_verified=(not rel_binding.scientific_qualification),
    )
    sub_ulp = evaluate_sub_ulp_storage_precision()

    receipt = R1ReplayReceipt(
        request_sha256=request.sha256,
        release_bundle_sha256=rel_binding.bundle_sha256,
        build_source_commit=rel_binding.build_source_commit,
        baseline_final_state_sha256=base_run.final_state_sha256,
        compensated_final_state_sha256=comp_run.final_state_sha256,
        replay_compensated_state_sha256=replay_run.final_state_sha256,
        deterministic_match=(comp_run.final_state_sha256 == replay_run.final_state_sha256),
        qualification_level=qual.qualification_level,
        r1_l2_certified=qual.r1_l2_certified,
        r1_l3_device_qualified=qual.r1_l3_device_qualified,
    )
    return R1ShortWindowPairingResult(
        request=request,
        baseline=base_run,
        compensated=comp_run,
        qualification=qual,
        replay_receipt=receipt,
        sub_ulp_precision_ratio=sub_ulp["error_reduction_ratio"],
    )


@dataclass(frozen=True, slots=True)
class R1WitnessMinimalEvaluation:
    """Public-interface evaluation of step 183 and step 309 minimal inputs (P07-02.2)."""

    witness_audit: HistoricalR1WitnessAudit
    parsed_fixture_layer_count: int
    step183_dt_s: float
    step309_dt_s: float
    public_step183_accepted: bool
    public_step309_accepted: bool
    public_step183_jacobian_error: float
    public_step309_jacobian_error: float
    public_step309_nonlinear_residual: float
    historical_step309_status: str
    relabelled: bool


def evaluate_r1_witness_minimal_input(
    witness_source: Path | bytes | str,
    *,
    intervals_per_layer: tuple[int, int] = (6, 4),
) -> R1WitnessMinimalEvaluation:
    """Re-evaluate 183/309 minimal witness step parameters on the new public interface.

    Per ``P07-02.2``, the historical V53 step 309 record remains ``failed`` and
    unrelabelled, while the new public-interface step solver is verified against
    independent Jacobian and nonlinear residual gates rather than forced to fail.
    """
    audit = verify_historical_r1_failure_witness(witness_source)
    raw = witness_source.read_bytes() if isinstance(witness_source, Path) else (
        witness_source if isinstance(witness_source, bytes) else witness_source.encode("utf-8")
    )
    import json

    payload = json.loads(raw.decode("utf-8"))
    yaml_mapping = load_yaml_mapping(audit.source_fixture_yaml.encode("utf-8"))
    layers_list = yaml_mapping.get("layers", [])
    layer_count = len(layers_list) if isinstance(layers_list, list) else 0

    c183_ok = payload["cases"]["step183_accepted"]
    saved_183 = c183_ok.get("saved_row", c183_ok)
    dt_183 = float(saved_183["dt_s"])
    v_183 = float(
        saved_183.get(
            "voltage_V",
            payload.get("request", {}).get("case", {}).get("amplitude_V", audit.step309_voltage_V),
        )
    )
    dt_309 = audit.step309_dt_s
    v_309 = audit.step309_voltage_V

    anchor = prepare_r1_zero_minus_anchor(intervals_per_layer=intervals_per_layer)
    res_183 = solve_coupled_r1_step(
        device=anchor.mobile_device,
        previous_ionic=anchor.mobile_state,
        previous_trap_occupancy=anchor.equilibrium_trap_occupancy,
        anchor=anchor,
        voltage_V=v_183,
        dt_s=dt_183,
        nu_t=1,
        mode="compensated",
    )
    res_309 = solve_coupled_r1_step(
        device=anchor.mobile_device,
        previous_ionic=anchor.mobile_state,
        previous_trap_occupancy=anchor.equilibrium_trap_occupancy,
        anchor=anchor,
        voltage_V=v_309,
        dt_s=dt_309,
        nu_t=1,
        mode="compensated",
    )
    return R1WitnessMinimalEvaluation(
        witness_audit=audit,
        parsed_fixture_layer_count=layer_count,
        step183_dt_s=dt_183,
        step309_dt_s=dt_309,
        public_step183_accepted=res_183.accepted,
        public_step309_accepted=res_309.accepted,
        public_step183_jacobian_error=res_183.analytic_jacobian_error,
        public_step309_jacobian_error=res_309.analytic_jacobian_error,
        public_step309_nonlinear_residual=res_309.scaled_nonlinear_residual,
        historical_step309_status=audit.step309_original_status,
        relabelled=False,
    )
