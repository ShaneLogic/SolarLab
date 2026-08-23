"""Restricted equilibrium solver for energy-distributed charged bulk traps."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import solve_banded

from perovskite_sim.constants import Q
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.models.mode import resolve_mode
from perovskite_sim.physics.bulk_traps import (
    BulkTrapState,
    evaluate_bulk_trap_state,
)
from perovskite_sim.physics.contacts import (
    SemiconductorContactState,
    build_semiconductor_contact_state,
)
from perovskite_sim.physics.degenerate_transport import (
    generalized_sg_fluxes_n,
    generalized_sg_fluxes_p,
)
from perovskite_sim.physics.generation import dual_cell_widths
from perovskite_sim.solver.mol import (
    BULK_TRAP_CHARGE_RESEARCH_EQUILIBRIUM,
    MaterialArrays,
    StateVec,
    build_material_arrays,
)


@dataclass(frozen=True, slots=True)
class BulkTrapPNEquilibriumResult:
    """Dark equilibrium state plus P4.3 numerical and physical evidence."""

    state: np.ndarray
    potential_V: np.ndarray
    electron_density_m3: np.ndarray
    hole_density_m3: np.ndarray
    trap_occupancy: np.ndarray
    trap_charge_density_C_m3: np.ndarray
    trap_recombination_rate_m3_s: np.ndarray
    electron_face_current_A_m2: np.ndarray
    hole_face_current_A_m2: np.ndarray
    left_contact: SemiconductorContactState
    right_contact: SemiconductorContactState
    quadrature_order: int
    newton_iterations: int
    maximum_normalized_poisson_residual: float
    maximum_relative_face_current: float
    maximum_absolute_face_current_A_m2: float
    maximum_mass_action_relative_error: float
    gauss_law_relative_error: float
    integrated_bulk_trap_charge_C_m2: float
    peak_electric_field_V_m: float
    minimum_trap_occupancy: float
    maximum_trap_occupancy: float


@dataclass(frozen=True, slots=True)
class _DensityState:
    electron_density_m3: np.ndarray
    hole_density_m3: np.ndarray
    trap: BulkTrapState

    def charge_density_C_m3(self, mat: MaterialArrays) -> np.ndarray:
        return Q * (
            self.hole_density_m3
            - self.electron_density_m3
            + mat.N_D
            - mat.N_A
        ) + self.trap.charge_density_C_m3

    def charge_derivative_potential_C_m3_V(
        self,
        thermal_voltage_V: float,
    ) -> np.ndarray:
        return Q * (
            -(self.electron_density_m3 + self.hole_density_m3)
            + self.trap.charge_number_derivative_potential_m3_V
            * thermal_voltage_V
        ) / thermal_voltage_V


def _density_state(
    potential_V: np.ndarray,
    mat: MaterialArrays,
    left_work_function_eV: float,
    *,
    quadrature_order: int,
) -> _DensityState:
    if mat.bulk_trap_distribution is None:
        raise ValueError("bulk-trap equilibrium material lacks a distribution")
    if mat.N_C_physical is None or mat.N_V_physical is None:
        raise ValueError("bulk-trap equilibrium requires physical DOS arrays")
    thermal = float(mat.V_T_device)
    conduction_dos = float(mat.N_C_physical[0])
    valence_dos = float(mat.N_V_physical[0])
    chi = np.asarray(mat.chi_phys, dtype=float)
    gap = np.asarray(mat.Eg_phys, dtype=float)
    eta_n = (-left_work_function_eV + potential_V + chi) / thermal
    eta_p = (left_work_function_eV - potential_V - chi - gap) / thermal
    n = conduction_dos * np.exp(eta_n)
    p = valence_dos * np.exp(eta_p)
    trap = evaluate_bulk_trap_state(
        n,
        p,
        mat.bulk_trap_distribution,
        band_gap_eV=float(gap[0]),
        effective_conduction_dos_m3=conduction_dos,
        effective_valence_dos_m3=valence_dos,
        temperature_K=float(mat.T_device),
        quadrature_order=quadrature_order,
    )
    return _DensityState(
        electron_density_m3=n,
        hole_density_m3=p,
        trap=trap,
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
    trap_density = float(mat.bulk_trap_distribution.total_density_m3)
    density_reference = np.maximum.reduce((
        np.asarray(mat.N_A[1:-1], dtype=float),
        np.asarray(mat.N_D[1:-1], dtype=float),
        np.full(mat.N_A.size - 2, trap_density),
        np.ones(mat.N_A.size - 2),
    ))
    voltage_reference = max(abs(float(mat.V_bi_bc)), mat.V_T_device)
    scale = (
        Q * density_reference * factor.h_cell
        + (factor.C[:-1] + factor.C[1:]) * voltage_reference
    )
    return float(np.max(np.abs(residual) / scale))


def _relative_face_current(
    current: np.ndarray,
    density: np.ndarray,
    diffusivity: np.ndarray,
    spacing: np.ndarray,
) -> float:
    scale = (
        Q
        * diffusivity
        * np.maximum(density[:-1], density[1:])
        / spacing
    )
    return float(
        np.max(np.abs(current) / np.maximum(scale, np.finfo(float).tiny))
    )


def solve_bulk_trap_pn_equilibrium(
    x: np.ndarray,
    stack: DeviceStack,
    *,
    quadrature_order: int = 64,
    poisson_tolerance: float = 1.0e-10,
    max_newton_iterations: int = 100,
    max_potential_step_V: float = 0.1,
    max_line_search_backtracks: int = 20,
) -> BulkTrapPNEquilibriumResult:
    """Solve one MB, fully-ionized, charged-trap p/n equilibrium slice."""
    coordinate = np.asarray(x, dtype=float)
    if (
        coordinate.ndim != 1
        or coordinate.size < 5
        or not np.all(np.isfinite(coordinate))
        or np.any(np.diff(coordinate) <= 0.0)
    ):
        raise ValueError("bulk-trap PN grid must be finite and strictly increasing")
    if (
        isinstance(quadrature_order, bool)
        or not isinstance(quadrature_order, (int, np.integer))
        or int(quadrature_order) < 1
    ):
        raise ValueError("quadrature_order must be a positive integer")
    if (
        isinstance(max_newton_iterations, bool)
        or not isinstance(max_newton_iterations, (int, np.integer))
        or int(max_newton_iterations) <= 0
        or isinstance(max_line_search_backtracks, bool)
        or not isinstance(max_line_search_backtracks, (int, np.integer))
        or int(max_line_search_backtracks) < 0
    ):
        raise ValueError("bulk-trap PN iteration controls must be integers")
    if (
        not math.isfinite(poisson_tolerance)
        or poisson_tolerance <= 0.0
        or not math.isfinite(max_potential_step_V)
        or max_potential_step_V <= 0.0
    ):
        raise ValueError("bulk-trap PN solver controls are invalid")
    layers = electrical_layers(stack)
    if len(layers) != 2:
        raise ValueError("bulk-trap PN equilibrium requires exactly two layers")
    if any(layer.params is None for layer in layers):
        raise ValueError(
            "bulk-trap PN equilibrium requires material parameters on both layers"
        )
    expected_span = float(sum(layer.thickness for layer in layers))
    span_tolerance = max(1.0e-18, 1.0e-12 * expected_span)
    if (
        abs(float(coordinate[0])) > span_tolerance
        or abs(float(coordinate[-1]) - expected_span) > span_tolerance
    ):
        raise ValueError("bulk-trap PN grid must span the full electrical stack")
    if (
        float(layers[0].params.N_A - layers[0].params.N_D) <= 0.0
        or float(layers[1].params.N_D - layers[1].params.N_A) <= 0.0
    ):
        raise ValueError("bulk-trap PN equilibrium requires p-left and n-right")
    order = int(quadrature_order)
    mat = build_material_arrays(
        coordinate,
        stack,
        bulk_trap_charge_closure=(
            BULK_TRAP_CHARGE_RESEARCH_EQUILIBRIUM
        ),
    )
    mode = resolve_mode(stack.mode)
    left_contact = build_semiconductor_contact_state(
        layers[0].params,
        temperature_K=mat.T_device,
        use_temperature_scaling=mode.use_temperature_scaling,
        bulk_trap_quadrature_order=order,
    )
    right_contact = build_semiconductor_contact_state(
        layers[-1].params,
        temperature_K=mat.T_device,
        use_temperature_scaling=mode.use_temperature_scaling,
        bulk_trap_quadrature_order=order,
    )
    if left_contact.bulk_trap_state is None or right_contact.bulk_trap_state is None:
        raise ValueError("bulk-trap contact states did not consume the distribution")
    potential_left = float(stack.phi_left)
    potential_right = potential_left + (
        left_contact.work_function_eV - right_contact.work_function_eV
    )
    potential = np.linspace(potential_left, potential_right, coordinate.size)
    factor = mat.poisson_factor
    last_normalized = math.inf

    for iteration in range(1, int(max_newton_iterations) + 1):
        density = _density_state(
            potential,
            mat,
            left_contact.work_function_eV,
            quadrature_order=order,
        )
        charge = density.charge_density_C_m3(mat)
        residual = _poisson_residual(potential, charge, mat)
        normalized = _normalized_poisson_residual(residual, mat)
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
        if not math.isfinite(infinity_norm):
            raise RuntimeError("bulk-trap PN Newton step is non-finite")
        if infinity_norm > max_potential_step_V:
            step *= max_potential_step_V / infinity_norm
        accepted = False
        damping = 1.0
        for _ in range(int(max_line_search_backtracks) + 1):
            trial = potential.copy()
            trial[1:-1] += damping * step
            trial_density = _density_state(
                trial,
                mat,
                left_contact.work_function_eV,
                quadrature_order=order,
            )
            trial_residual = _poisson_residual(
                trial,
                trial_density.charge_density_C_m3(mat),
                mat,
            )
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
                "bulk-trap PN Newton line search stalled at normalized "
                f"residual {normalized:.6g}"
            )
    else:
        raise RuntimeError(
            "bulk-trap PN Newton exceeded max iterations at normalized "
            f"residual {last_normalized:.6g}"
        )

    density = _density_state(
        potential,
        mat,
        left_contact.work_function_eV,
        quadrature_order=order,
    )
    n = density.electron_density_m3
    p = density.hole_density_m3
    charge = density.charge_density_C_m3(mat)
    residual = _poisson_residual(potential, charge, mat)
    normalized = _normalized_poisson_residual(residual, mat)
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
    relative_current = max(
        _relative_face_current(
            electron_current,
            n,
            mat.D_n_face,
            spacing,
        ),
        _relative_face_current(
            hole_current,
            p,
            mat.D_p_face,
            spacing,
        ),
    )
    intrinsic_product = float(mat.N_C_physical[0] * mat.N_V_physical[0]) * math.exp(
        -float(mat.Eg_phys[0]) / mat.V_T_device
    )
    mass_action_error = float(
        np.max(
            np.abs(n * p - intrinsic_product)
            / max(intrinsic_product, np.finfo(float).tiny)
        )
    )
    displacement = -factor.C * np.diff(potential)
    integrated_charge = float(np.sum(charge[1:-1] * factor.h_cell))
    displacement_jump = float(displacement[-1] - displacement[0])
    gauss_scale = max(
        float(np.sum(np.abs(charge[1:-1]) * factor.h_cell)),
        abs(integrated_charge),
        abs(displacement_jump),
        np.finfo(float).tiny,
    )
    gauss_error = abs(displacement_jump - integrated_charge) / gauss_scale
    weights = dual_cell_widths(coordinate)
    electric_field = -np.diff(potential) / spacing
    return BulkTrapPNEquilibriumResult(
        state=StateVec.pack(n, p, mat.P_ion0.copy()),
        potential_V=potential,
        electron_density_m3=n,
        hole_density_m3=p,
        trap_occupancy=density.trap.occupancy,
        trap_charge_density_C_m3=density.trap.charge_density_C_m3,
        trap_recombination_rate_m3_s=density.trap.recombination_rate_m3_s,
        electron_face_current_A_m2=electron_current,
        hole_face_current_A_m2=hole_current,
        left_contact=left_contact,
        right_contact=right_contact,
        quadrature_order=order,
        newton_iterations=iteration,
        maximum_normalized_poisson_residual=normalized,
        maximum_relative_face_current=relative_current,
        maximum_absolute_face_current_A_m2=float(max(
            np.max(np.abs(electron_current)),
            np.max(np.abs(hole_current)),
        )),
        maximum_mass_action_relative_error=mass_action_error,
        gauss_law_relative_error=gauss_error,
        integrated_bulk_trap_charge_C_m2=float(
            np.sum(density.trap.charge_density_C_m3 * weights)
        ),
        peak_electric_field_V_m=float(np.max(np.abs(electric_field))),
        minimum_trap_occupancy=float(np.min(density.trap.occupancy)),
        maximum_trap_occupancy=float(np.max(density.trap.occupancy)),
    )


__all__ = [
    "BulkTrapPNEquilibriumResult",
    "solve_bulk_trap_pn_equilibrium",
]
