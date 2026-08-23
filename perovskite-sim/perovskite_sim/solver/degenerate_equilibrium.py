"""Restricted equilibrium solver for homogeneous degenerate p-n junctions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import solve_banded

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.physics.contacts import (
    SemiconductorContactState,
    build_semiconductor_contact_state,
)
from perovskite_sim.physics.degenerate_transport import (
    generalized_carrier_face_statistics,
    generalized_sg_fluxes_n,
    generalized_sg_fluxes_p,
)
from perovskite_sim.physics.generation import dual_cell_widths
from perovskite_sim.physics.statistics import (
    DISCRETE_LEVEL,
    carrier_density_derivative_reduced_fermi_level,
    carrier_density_from_reduced_fermi_level,
    dopant_charge_state,
)
from perovskite_sim.solver.mol import (
    DEGENERATE_TRANSPORT_RESEARCH_RECOMBINATION_OFF,
    MaterialArrays,
    StateVec,
    assemble_rhs,
    build_material_arrays,
)


@dataclass(frozen=True, slots=True)
class DegeneratePNEquilibriumResult:
    """State and internal numerical evidence for one dark p+/n+ junction."""

    state: np.ndarray
    potential_V: np.ndarray
    electron_density_m3: np.ndarray
    hole_density_m3: np.ndarray
    band_gap_narrowing_eV: np.ndarray
    ionized_acceptor_density_m3: np.ndarray
    ionized_donor_density_m3: np.ndarray
    electron_face_current_A_m2: np.ndarray
    hole_face_current_A_m2: np.ndarray
    left_contact: SemiconductorContactState
    right_contact: SemiconductorContactState
    newton_iterations: int
    maximum_normalized_poisson_residual: float
    maximum_normalized_carrier_rate: float
    maximum_relative_face_current: float
    maximum_absolute_face_current_A_m2: float
    charge_balance_relative_error: float
    depletion_width_m: float
    analytic_depletion_width_m: float
    peak_electric_field_V_m: float
    analytic_peak_electric_field_V_m: float

    @property
    def depletion_width_relative_error(self) -> float:
        return abs(
            self.depletion_width_m - self.analytic_depletion_width_m
        ) / self.analytic_depletion_width_m

    @property
    def peak_field_relative_error(self) -> float:
        return abs(
            self.peak_electric_field_V_m
            - self.analytic_peak_electric_field_V_m
        ) / self.analytic_peak_electric_field_V_m


@dataclass(frozen=True, slots=True)
class _EquilibriumDensityState:
    electron_density_m3: np.ndarray
    hole_density_m3: np.ndarray
    electron_density_derivative_eta_n_m3: np.ndarray
    hole_density_derivative_eta_p_m3: np.ndarray
    ionized_donor_density_m3: np.ndarray
    ionized_acceptor_density_m3: np.ndarray
    donor_density_derivative_eta_n_m3: np.ndarray
    acceptor_density_derivative_eta_p_m3: np.ndarray

    def charge_density_C_m3(self) -> np.ndarray:
        return Q * (
            self.hole_density_m3
            - self.electron_density_m3
            + self.ionized_donor_density_m3
            - self.ionized_acceptor_density_m3
        )

    def charge_derivative_potential_C_m3_V(self, thermal_V: float) -> np.ndarray:
        return Q * (
            -self.electron_density_derivative_eta_n_m3
            - self.hole_density_derivative_eta_p_m3
            + self.donor_density_derivative_eta_n_m3
            + self.acceptor_density_derivative_eta_p_m3
        ) / thermal_V


def _contact_states(
    stack: DeviceStack,
    mat: MaterialArrays,
) -> tuple[SemiconductorContactState, SemiconductorContactState]:
    from perovskite_sim.models.mode import resolve_mode

    layers = electrical_layers(stack)
    mode = resolve_mode(stack.mode)
    return (
        build_semiconductor_contact_state(
            layers[0].params,
            temperature_K=mat.T_device,
            use_temperature_scaling=mode.use_temperature_scaling,
        ),
        build_semiconductor_contact_state(
            layers[-1].params,
            temperature_K=mat.T_device,
            use_temperature_scaling=mode.use_temperature_scaling,
        ),
    )


def _density_state(
    potential_V: np.ndarray,
    mat: MaterialArrays,
    left_work_function_eV: float,
) -> _EquilibriumDensityState:
    thermal = mat.V_T_device
    chi = np.asarray(mat.chi_phys, dtype=float)
    gap = np.asarray(mat.Eg_phys, dtype=float)
    conduction_dos = float(mat.N_C_physical[0])
    valence_dos = float(mat.N_V_physical[0])
    statistics = mat.carrier_statistics
    eta_n = (-left_work_function_eV + potential_V + chi) / thermal
    eta_p = (left_work_function_eV - potential_V - chi - gap) / thermal
    n = np.asarray(
        [
            carrier_density_from_reduced_fermi_level(
                eta,
                conduction_dos,
                statistics=statistics,
            )
            for eta in eta_n
        ],
        dtype=float,
    )
    p = np.asarray(
        [
            carrier_density_from_reduced_fermi_level(
                eta,
                valence_dos,
                statistics=statistics,
            )
            for eta in eta_p
        ],
        dtype=float,
    )
    dn_deta = np.asarray(
        [
            carrier_density_derivative_reduced_fermi_level(
                eta,
                conduction_dos,
                statistics=statistics,
            )
            for eta in eta_n
        ],
        dtype=float,
    )
    dp_deta = np.asarray(
        [
            carrier_density_derivative_reduced_fermi_level(
                eta,
                valence_dos,
                statistics=statistics,
            )
            for eta in eta_p
        ],
        dtype=float,
    )
    ionized_donors = np.asarray(mat.N_D, dtype=float)
    ionized_acceptors = np.asarray(mat.N_A, dtype=float)
    donor_derivative = np.zeros_like(ionized_donors)
    acceptor_derivative = np.zeros_like(ionized_acceptors)
    if mat.dopant_ionization_model == DISCRETE_LEVEL:
        required = (
            mat.donor_binding_energy_eV,
            mat.acceptor_binding_energy_eV,
            mat.donor_degeneracy,
            mat.acceptor_degeneracy,
        )
        if any(value is None for value in required):
            raise ValueError(
                "discrete-level equilibrium requires dopant parameter arrays"
            )
        dopant_states = tuple(
            dopant_charge_state(
                reduced_electron_fermi_level=float(eta_n[index]),
                reduced_hole_fermi_level=float(eta_p[index]),
                donor_density_m3=float(mat.N_D[index]),
                acceptor_density_m3=float(mat.N_A[index]),
                thermal_voltage_V=thermal,
                model=DISCRETE_LEVEL,
                donor_binding_energy_eV=float(
                    mat.donor_binding_energy_eV[index]
                ),
                acceptor_binding_energy_eV=float(
                    mat.acceptor_binding_energy_eV[index]
                ),
                donor_degeneracy=float(mat.donor_degeneracy[index]),
                acceptor_degeneracy=float(mat.acceptor_degeneracy[index]),
            )
            for index in range(potential_V.size)
        )
        ionized_donors = np.asarray(
            [state.ionized_donor_density_m3 for state in dopant_states],
            dtype=float,
        )
        ionized_acceptors = np.asarray(
            [state.ionized_acceptor_density_m3 for state in dopant_states],
            dtype=float,
        )
        donor_derivative = np.asarray(
            [
                state.donor_density_derivative_eta_n_m3
                for state in dopant_states
            ],
            dtype=float,
        )
        acceptor_derivative = np.asarray(
            [
                state.acceptor_density_derivative_eta_p_m3
                for state in dopant_states
            ],
            dtype=float,
        )
    return _EquilibriumDensityState(
        electron_density_m3=n,
        hole_density_m3=p,
        electron_density_derivative_eta_n_m3=dn_deta,
        hole_density_derivative_eta_p_m3=dp_deta,
        ionized_donor_density_m3=ionized_donors,
        ionized_acceptor_density_m3=ionized_acceptors,
        donor_density_derivative_eta_n_m3=donor_derivative,
        acceptor_density_derivative_eta_p_m3=acceptor_derivative,
    )


def _poisson_residual(
    potential_V: np.ndarray,
    charge_density_C_m3: np.ndarray,
    mat: MaterialArrays,
) -> np.ndarray:
    factor = mat.poisson_factor
    return (
        factor.C[:-1] * (potential_V[:-2] - potential_V[1:-1])
        + factor.C[1:] * (potential_V[2:] - potential_V[1:-1])
        + charge_density_C_m3[1:-1] * factor.h_cell
    )


def _normalized_poisson_residual(
    residual: np.ndarray,
    mat: MaterialArrays,
) -> float:
    factor = mat.poisson_factor
    density_reference = np.maximum(
        np.maximum(mat.N_A[1:-1], mat.N_D[1:-1]),
        1.0,
    )
    voltage_reference = max(abs(float(mat.V_bi_bc)), mat.V_T_device)
    scale = (
        Q * density_reference * factor.h_cell
        + (factor.C[:-1] + factor.C[1:]) * voltage_reference
    )
    return float(np.max(np.abs(residual) / scale))


def _transport_residual_metrics(
    *,
    coordinate_m: np.ndarray,
    electron_density_m3: np.ndarray,
    hole_density_m3: np.ndarray,
    electron_current_A_m2: np.ndarray,
    hole_current_A_m2: np.ndarray,
    physical_rhs: np.ndarray,
    mat: MaterialArrays,
) -> tuple[float, float]:
    """Scale equilibrium roundoff by the local generalized-SG transport."""
    spacing = np.diff(coordinate_m)
    electron_face = generalized_carrier_face_statistics(
        electron_density_m3,
        float(mat.N_C_physical[0]),
        statistics=mat.carrier_statistics,
    )
    hole_face = generalized_carrier_face_statistics(
        hole_density_m3,
        float(mat.N_V_physical[0]),
        statistics=mat.carrier_statistics,
    )
    electron_scale = (
        Q
        * mat.D_n_face
        * electron_face.diffusion_enhancement
        * np.maximum(electron_density_m3[:-1], electron_density_m3[1:])
        / spacing
    )
    hole_scale = (
        Q
        * mat.D_p_face
        * hole_face.diffusion_enhancement
        * np.maximum(hole_density_m3[:-1], hole_density_m3[1:])
        / spacing
    )
    relative_face_current = max(
        float(
            np.max(
                np.abs(electron_current_A_m2)
                / np.maximum(electron_scale, np.finfo(float).tiny)
            )
        ),
        float(
            np.max(
                np.abs(hole_current_A_m2)
                / np.maximum(hole_scale, np.finfo(float).tiny)
            )
        ),
    )

    cell_width = dual_cell_widths(coordinate_m)
    electron_rate_scale = np.zeros_like(electron_density_m3)
    hole_rate_scale = np.zeros_like(hole_density_m3)
    electron_rate_scale[1:-1] = (
        electron_scale[:-1] + electron_scale[1:]
    ) / (Q * cell_width[1:-1])
    hole_rate_scale[1:-1] = (
        hole_scale[:-1] + hole_scale[1:]
    ) / (Q * cell_width[1:-1])
    node_count = coordinate_m.size
    electron_rhs = physical_rhs[:node_count]
    hole_rhs = physical_rhs[node_count : 2 * node_count]
    normalized_carrier_rate = max(
        float(
            np.max(
                np.abs(electron_rhs[1:-1])
                / np.maximum(
                    electron_rate_scale[1:-1],
                    np.finfo(float).tiny,
                )
            )
        ),
        float(
            np.max(
                np.abs(hole_rhs[1:-1])
                / np.maximum(
                    hole_rate_scale[1:-1],
                    np.finfo(float).tiny,
                )
            )
        ),
    )
    return normalized_carrier_rate, relative_face_current


def solve_degenerate_pn_equilibrium(
    x: np.ndarray,
    stack: DeviceStack,
    *,
    poisson_tolerance: float = 1.0e-10,
    max_newton_iterations: int = 100,
    max_potential_step_V: float = 0.1,
    max_line_search_backtracks: int = 20,
) -> DegeneratePNEquilibriumResult:
    """Solve one dark homogeneous p-left/n-right junction at equilibrium.

    Recombination is explicitly disabled because the repository's historical
    ``np-ni^2`` closures are Maxwell-Boltzmann laws. The resulting lane
    certifies FD charge, Poisson, and generalized transport only.
    """
    coordinate = np.asarray(x, dtype=float)
    if (
        coordinate.ndim != 1
        or coordinate.size < 5
        or not np.all(np.isfinite(coordinate))
        or np.any(np.diff(coordinate) <= 0.0)
    ):
        raise ValueError("degenerate PN grid must be finite and strictly increasing")
    if (
        not math.isfinite(poisson_tolerance)
        or poisson_tolerance <= 0.0
        or max_newton_iterations <= 0
        or not math.isfinite(max_potential_step_V)
        or max_potential_step_V <= 0.0
        or max_line_search_backtracks < 0
    ):
        raise ValueError("degenerate PN solver controls are invalid")
    layers = electrical_layers(stack)
    if len(layers) != 2:
        raise ValueError("degenerate PN equilibrium requires exactly two layers")
    if (
        float(layers[0].params.N_A - layers[0].params.N_D) <= 0.0
        or float(layers[1].params.N_D - layers[1].params.N_A) <= 0.0
    ):
        raise ValueError("degenerate PN equilibrium requires p-left and n-right")

    mat = build_material_arrays(
        coordinate,
        stack,
        carrier_statistics_transport=(
            DEGENERATE_TRANSPORT_RESEARCH_RECOMBINATION_OFF
        ),
    )
    left_contact, right_contact = _contact_states(stack, mat)
    left_net = float(
        left_contact.neutrality.ionized_acceptor_density_m3
        - left_contact.neutrality.ionized_donor_density_m3
    )
    right_net = float(
        right_contact.neutrality.ionized_donor_density_m3
        - right_contact.neutrality.ionized_acceptor_density_m3
    )
    if left_net <= 0.0 or right_net <= 0.0:
        raise ValueError("ionized contact charge does not preserve p-left/n-right")
    phi_left = float(stack.phi_left)
    phi_right = phi_left + float(mat.V_bi_bc)
    potential = np.linspace(phi_left, phi_right, coordinate.size)
    factor = mat.poisson_factor
    last_normalized = math.inf

    for iteration in range(1, max_newton_iterations + 1):
        density = _density_state(
            potential,
            mat,
            left_contact.work_function_eV,
        )
        charge = density.charge_density_C_m3()
        residual = _poisson_residual(potential, charge, mat)
        normalized = _normalized_poisson_residual(
            residual,
            mat,
        )
        if normalized <= poisson_tolerance:
            break
        charge_derivative = density.charge_derivative_potential_C_m3_V(
            mat.V_T_device
        )
        banded = np.zeros((3, coordinate.size - 2), dtype=float)
        banded[0, 1:] = factor.C[1:-1]
        banded[1] = -(
            factor.C[:-1] + factor.C[1:]
        ) + charge_derivative[1:-1] * factor.h_cell
        banded[2, :-1] = factor.C[1:-1]
        step = solve_banded((1, 1), banded, -residual)
        infinity_norm = float(np.max(np.abs(step)))
        if not np.isfinite(infinity_norm):
            raise RuntimeError("degenerate PN Newton step is non-finite")
        if infinity_norm > max_potential_step_V:
            step *= max_potential_step_V / infinity_norm

        accepted = False
        damping = 1.0
        for _ in range(max_line_search_backtracks + 1):
            trial = potential.copy()
            trial[1:-1] += damping * step
            trial_density = _density_state(
                trial,
                mat,
                left_contact.work_function_eV,
            )
            trial_charge = trial_density.charge_density_C_m3()
            trial_residual = _poisson_residual(trial, trial_charge, mat)
            trial_normalized = _normalized_poisson_residual(
                trial_residual,
                mat,
            )
            if trial_normalized < normalized:
                potential = trial
                last_normalized = trial_normalized
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            raise RuntimeError(
                "degenerate PN Newton line search stalled at normalized "
                f"residual {normalized:.6g}"
            )
    else:
        raise RuntimeError(
            "degenerate PN Newton exceeded max iterations at normalized "
            f"residual {last_normalized:.6g}"
        )

    density = _density_state(
        potential,
        mat,
        left_contact.work_function_eV,
    )
    n = density.electron_density_m3
    p = density.hole_density_m3
    charge = density.charge_density_C_m3()
    residual = _poisson_residual(potential, charge, mat)
    normalized = _normalized_poisson_residual(
        residual,
        mat,
    )
    state = StateVec.pack(n, p, mat.P_ion0.copy())
    physical_rhs = assemble_rhs(
        0.0,
        state,
        coordinate,
        stack,
        mat,
        illuminated=False,
        V_app=0.0,
        phi_frozen=potential,
    )
    spacing = np.diff(coordinate)
    electron_current = generalized_sg_fluxes_n(
        potential + mat.chi_phys,
        n,
        spacing,
        mat.D_n_face / mat.V_T_device,
        mat.V_T_device,
        float(mat.N_C_physical[0]),
        statistics=mat.carrier_statistics,
    )
    hole_current = generalized_sg_fluxes_p(
        potential + mat.chi_phys + mat.Eg_phys,
        p,
        spacing,
        mat.D_p_face / mat.V_T_device,
        mat.V_T_device,
        float(mat.N_V_physical[0]),
        statistics=mat.carrier_statistics,
    )
    maximum_normalized_rate, maximum_relative_face_current = (
        _transport_residual_metrics(
            coordinate_m=coordinate,
            electron_density_m3=n,
            hole_density_m3=p,
            electron_current_A_m2=electron_current,
            hole_current_A_m2=hole_current,
            physical_rhs=physical_rhs,
            mat=mat,
        )
    )

    junction = float(layers[0].thickness)
    weights = dual_cell_widths(coordinate)
    left_mask = coordinate < junction
    right_mask = coordinate > junction
    negative_charge = -float(
        np.sum(np.minimum(charge[left_mask], 0.0) * weights[left_mask])
    )
    positive_charge = float(
        np.sum(np.maximum(charge[right_mask], 0.0) * weights[right_mask])
    )
    depletion_left = negative_charge / (Q * left_net)
    depletion_right = positive_charge / (Q * right_net)
    depletion_width = depletion_left + depletion_right
    charge_balance = abs(positive_charge - negative_charge) / max(
        positive_charge,
        negative_charge,
        np.finfo(float).tiny,
    )
    built_in = abs(float(mat.V_bi_bc))
    permittivity = EPS_0 * float(mat.eps_r[0])
    analytic_width = math.sqrt(
        2.0
        * permittivity
        * built_in
        * (left_net + right_net)
        / (Q * left_net * right_net)
    )
    analytic_field = Q * left_net * (
        analytic_width * right_net / (left_net + right_net)
    ) / permittivity
    electric_field = -np.diff(potential) / spacing

    return DegeneratePNEquilibriumResult(
        state=state,
        potential_V=potential,
        electron_density_m3=n,
        hole_density_m3=p,
        band_gap_narrowing_eV=(
            np.zeros_like(n)
            if mat.band_gap_narrowing_eV is None
            else np.asarray(mat.band_gap_narrowing_eV, dtype=float).copy()
        ),
        ionized_acceptor_density_m3=density.ionized_acceptor_density_m3,
        ionized_donor_density_m3=density.ionized_donor_density_m3,
        electron_face_current_A_m2=electron_current,
        hole_face_current_A_m2=hole_current,
        left_contact=left_contact,
        right_contact=right_contact,
        newton_iterations=iteration,
        maximum_normalized_poisson_residual=normalized,
        maximum_normalized_carrier_rate=maximum_normalized_rate,
        maximum_relative_face_current=maximum_relative_face_current,
        maximum_absolute_face_current_A_m2=float(
            max(
                np.max(np.abs(electron_current)),
                np.max(np.abs(hole_current)),
            )
        ),
        charge_balance_relative_error=charge_balance,
        depletion_width_m=depletion_width,
        analytic_depletion_width_m=analytic_width,
        peak_electric_field_V_m=float(np.max(np.abs(electric_field))),
        analytic_peak_electric_field_V_m=analytic_field,
    )


__all__ = [
    "DegeneratePNEquilibriumResult",
    "solve_degenerate_pn_equilibrium",
]
