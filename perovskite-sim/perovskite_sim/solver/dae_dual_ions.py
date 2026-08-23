"""Research-only semi-explicit DAE for two mobile-ion species.

This topology retains positive and negative ion densities plus Poisson
potential explicitly.  It supports one electrical layer, ohmic carrier
contacts, blocking ion boundaries, and no physical or interface-state
interfaces.  Production transient routes remain in :mod:`solver.mol`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.special import expit

from perovskite_sim.constants import Q
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.dae import DAECapabilityError
from perovskite_sim.solver.mol import (
    MaterialArrays,
    StateVec,
    assemble_rhs,
    build_material_arrays,
    poisson_right_boundary,
)


def _readonly_f64(value: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _state_sha256(label: str, *arrays: np.ndarray) -> str:
    digest = hashlib.sha256(label.encode("ascii"))
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _shifted_logistic_density(
    reference_m3: np.ndarray,
    limit_m3: np.ndarray,
    logit_reference: np.ndarray,
    coordinate: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    theta = expit(logit_reference + coordinate)
    theta_reference = reference_m3 / limit_m3
    positive_delta = (
        theta
        * (1.0 - theta_reference)
        * (-np.expm1(-np.maximum(coordinate, 0.0)))
    )
    negative_delta = (
        -theta_reference
        * (1.0 - theta)
        * (-np.expm1(np.minimum(coordinate, 0.0)))
    )
    density = reference_m3 + limit_m3 * np.where(
        coordinate >= 0.0,
        positive_delta,
        negative_delta,
    )
    if np.any(theta <= 0.0) or np.any(theta >= 1.0):
        raise ValueError(f"dual-ion {label} logit coordinate saturated")
    return density


@dataclass(frozen=True, slots=True)
class DualIonDAELayout:
    """Coordinate layout for ``(log n, log p, ion+, ion-, phi)``."""

    node_count: int
    shared_site: bool
    electron_reference_m3: np.ndarray
    hole_reference_m3: np.ndarray
    positive_ion_reference_m3: np.ndarray
    negative_ion_reference_m3: np.ndarray
    positive_ion_site_limit_m3: np.ndarray
    negative_ion_site_limit_m3: np.ndarray
    positive_ion_coordinate_reference: np.ndarray
    negative_ion_coordinate_reference: np.ndarray
    electron_rate_scale_m3_s: np.ndarray
    hole_rate_scale_m3_s: np.ndarray
    positive_ion_rate_scale_m3_s: np.ndarray
    negative_ion_rate_scale_m3_s: np.ndarray
    poisson_scale_C_m2: np.ndarray
    potential_scale_V: float
    differential_mask: np.ndarray
    algebraic_mask: np.ndarray

    @property
    def size(self) -> int:
        return 5 * self.node_count

    @property
    def electron_slice(self) -> slice:
        return slice(0, self.node_count)

    @property
    def hole_slice(self) -> slice:
        return slice(self.node_count, 2 * self.node_count)

    @property
    def positive_ion_slice(self) -> slice:
        return slice(2 * self.node_count, 3 * self.node_count)

    @property
    def negative_ion_slice(self) -> slice:
        return slice(3 * self.node_count, 4 * self.node_count)

    @property
    def potential_slice(self) -> slice:
        return slice(4 * self.node_count, 5 * self.node_count)


@dataclass(frozen=True, slots=True)
class DualIonDAEResidualReport:
    """Differential, algebraic, and per-species inventory evidence."""

    normalized_residual: np.ndarray
    electron_rate_residual_m3_s: np.ndarray
    hole_rate_residual_m3_s: np.ndarray
    positive_ion_rate_residual_m3_s: np.ndarray
    negative_ion_rate_residual_m3_s: np.ndarray
    poisson_residual_C_m2: np.ndarray
    carrier_boundary_residual_log: np.ndarray
    potential_boundary_residual_V: np.ndarray
    positive_ion_inventory_residual_m2_s: float
    negative_ion_inventory_residual_m2_s: float
    positive_ion_rhs_inventory_rate_m2_s: float
    negative_ion_rhs_inventory_rate_m2_s: float
    max_normalized_carrier_residual: float
    max_normalized_positive_ion_residual: float
    max_normalized_negative_ion_residual: float
    max_normalized_differential_residual: float
    max_normalized_algebraic_residual: float
    max_normalized_residual: float


@dataclass(frozen=True, slots=True)
class DualIonDAEConsistentInitialCondition:
    """Reproducible state/derivative pair satisfying every dual-ion row."""

    coordinate: np.ndarray
    derivative: np.ndarray
    physical_state: np.ndarray
    potential_V: np.ndarray
    report: DualIonDAEResidualReport
    certified: bool
    state_sha256: str


@dataclass(frozen=True, slots=True)
class DualIonDAE:
    """Narrow DAE retaining both mobile-ion species and Poisson explicitly."""

    grid_m: np.ndarray
    stack: DeviceStack
    material: MaterialArrays
    layout: DualIonDAELayout
    V_app_V: float
    illuminated: bool

    def _shared_site_fractions(
        self,
        coordinate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        layout = self.layout
        positive_log_ratio = (
            layout.positive_ion_coordinate_reference
            + coordinate[layout.positive_ion_slice]
        )
        negative_log_ratio = (
            layout.negative_ion_coordinate_reference
            + coordinate[layout.negative_ion_slice]
        )
        offset = np.maximum.reduce(
            (
                np.zeros(layout.node_count),
                positive_log_ratio,
                negative_log_ratio,
            )
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            positive_weight = np.exp(positive_log_ratio - offset)
            negative_weight = np.exp(negative_log_ratio - offset)
            vacancy_weight = np.exp(-offset)
        denominator = positive_weight + negative_weight + vacancy_weight
        positive_fraction = positive_weight / denominator
        negative_fraction = negative_weight / denominator
        vacancy_fraction = vacancy_weight / denominator
        fractions = (positive_fraction, negative_fraction, vacancy_fraction)
        if any(
            not np.all(np.isfinite(value)) or np.any(value <= 0.0)
            for value in fractions
        ):
            raise ValueError("dual-ion shared-site coordinate saturated")
        return fractions

    def physical_fields(
        self,
        coordinate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        value = np.asarray(coordinate, dtype=float)
        if value.shape != (self.layout.size,) or not np.all(np.isfinite(value)):
            raise ValueError("dual-ion DAE coordinate must be finite and layout-sized")
        layout = self.layout
        with np.errstate(over="ignore", invalid="ignore"):
            n = layout.electron_reference_m3 * np.exp(value[layout.electron_slice])
            p = layout.hole_reference_m3 * np.exp(value[layout.hole_slice])
        if layout.shared_site:
            positive_fraction, negative_fraction, _vacancy = (
                self._shared_site_fractions(value)
            )
            site_limit = layout.positive_ion_site_limit_m3
            positive_ion = site_limit * positive_fraction
            negative_ion = site_limit * negative_fraction
        else:
            positive_ion = _shifted_logistic_density(
                layout.positive_ion_reference_m3,
                layout.positive_ion_site_limit_m3,
                layout.positive_ion_coordinate_reference,
                value[layout.positive_ion_slice],
                label="positive-ion",
            )
            negative_ion = _shifted_logistic_density(
                layout.negative_ion_reference_m3,
                layout.negative_ion_site_limit_m3,
                layout.negative_ion_coordinate_reference,
                value[layout.negative_ion_slice],
                label="negative-ion",
            )
        phi = np.asarray(value[layout.potential_slice], dtype=float)
        arrays = (n, p, positive_ion, negative_ion, phi)
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("dual-ion DAE coordinate mapping is non-finite")
        return n, p, positive_ion, negative_ion, phi

    def ion_coordinate_jacobian_m3(self, coordinate: np.ndarray) -> np.ndarray:
        """Return node-local ``d(P+, P-)/d(u+, u-)`` mass matrices."""
        _n, _p, positive_ion, negative_ion, _phi = self.physical_fields(
            coordinate
        )
        layout = self.layout
        result = np.zeros((layout.node_count, 2, 2), dtype=float)
        if layout.shared_site:
            limit = layout.positive_ion_site_limit_m3
            positive_fraction = positive_ion / limit
            negative_fraction = negative_ion / limit
            coupling = positive_ion * negative_fraction
            result[:, 0, 0] = positive_ion * (1.0 - positive_fraction)
            result[:, 0, 1] = -coupling
            result[:, 1, 0] = -coupling
            result[:, 1, 1] = negative_ion * (1.0 - negative_fraction)
        else:
            result[:, 0, 0] = positive_ion * (
                1.0 - positive_ion / layout.positive_ion_site_limit_m3
            )
            result[:, 1, 1] = negative_ion * (
                1.0 - negative_ion / layout.negative_ion_site_limit_m3
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("dual-ion coordinate mass matrix is non-finite")
        return result

    def ion_coordinate_hessian_m3(self, coordinate: np.ndarray) -> np.ndarray:
        """Return node-local second derivatives of both ion densities.

        The result is indexed ``[node, physical_species, first_coordinate,
        second_coordinate]``.  Shared-site entries include every softmax cross
        derivative; distinct-sublattice entries are diagonal logistic terms.
        """
        _n, _p, positive_ion, negative_ion, _phi = self.physical_fields(
            coordinate
        )
        layout = self.layout
        result = np.zeros((layout.node_count, 2, 2, 2), dtype=float)
        if layout.shared_site:
            limit = layout.positive_ion_site_limit_m3
            fractions = np.stack(
                (positive_ion / limit, negative_ion / limit),
                axis=1,
            )
            for species in range(2):
                for first in range(2):
                    for second in range(2):
                        delta_species_second = float(species == second)
                        delta_species_first = float(species == first)
                        delta_first_second = float(first == second)
                        result[:, species, first, second] = (
                            limit
                            * fractions[:, species]
                            * (
                                (
                                    delta_species_second
                                    - fractions[:, second]
                                )
                                * (
                                    delta_species_first
                                    - fractions[:, first]
                                )
                                - fractions[:, first]
                                * (
                                    delta_first_second
                                    - fractions[:, second]
                                )
                            )
                        )
        else:
            positive_fraction = (
                positive_ion / layout.positive_ion_site_limit_m3
            )
            negative_fraction = (
                negative_ion / layout.negative_ion_site_limit_m3
            )
            result[:, 0, 0, 0] = (
                positive_ion
                * (1.0 - positive_fraction)
                * (1.0 - 2.0 * positive_fraction)
            )
            result[:, 1, 1, 1] = (
                negative_ion
                * (1.0 - negative_fraction)
                * (1.0 - 2.0 * negative_fraction)
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("dual-ion coordinate Hessian is non-finite")
        return result

    def packed_physical_state(self, coordinate: np.ndarray) -> np.ndarray:
        n, p, positive_ion, negative_ion, _phi = self.physical_fields(coordinate)
        return StateVec.pack(n, p, positive_ion, negative_ion)

    def compatible_derivative(self, coordinate: np.ndarray) -> np.ndarray:
        """Return qdot satisfying every differential row at one coordinate."""
        layout = self.layout
        n, p, _positive_ion, _negative_ion, phi = self.physical_fields(coordinate)
        rhs = StateVec.unpack(
            assemble_rhs(
                0.0,
                self.packed_physical_state(coordinate),
                self.grid_m,
                self.stack,
                self.material,
                illuminated=self.illuminated,
                V_app=self.V_app_V,
                phi_frozen=phi,
            ),
            layout.node_count,
        )
        if rhs.P_neg is None:
            raise ValueError("dual-ion RHS omitted the negative-ion species")
        derivative = np.zeros(layout.size, dtype=float)
        interior = slice(1, layout.node_count - 1)
        derivative[1 : layout.node_count - 1] = rhs.n[interior] / n[interior]
        derivative[
            layout.node_count + 1 : 2 * layout.node_count - 1
        ] = rhs.p[interior] / p[interior]
        ion_rhs = np.stack((rhs.P, rhs.P_neg), axis=1)
        try:
            ion_rate = np.linalg.solve(
                self.ion_coordinate_jacobian_m3(coordinate),
                ion_rhs[..., None],
            )[..., 0]
        except np.linalg.LinAlgError as exc:
            raise ValueError("dual-ion coordinate mass matrix is singular") from exc
        derivative[layout.positive_ion_slice] = ion_rate[:, 0]
        derivative[layout.negative_ion_slice] = ion_rate[:, 1]
        if not np.all(np.isfinite(derivative)):
            raise ValueError("dual-ion compatible derivative is non-finite")
        return derivative

    def residual_report(
        self,
        coordinate: np.ndarray,
        derivative: np.ndarray,
    ) -> DualIonDAEResidualReport:
        layout = self.layout
        rate = np.asarray(derivative, dtype=float)
        if rate.shape != (layout.size,) or not np.all(np.isfinite(rate)):
            raise ValueError("dual-ion DAE derivative must be layout-sized and finite")
        n, p, positive_ion, negative_ion, phi = self.physical_fields(coordinate)
        rhs = StateVec.unpack(
            assemble_rhs(
                0.0,
                StateVec.pack(n, p, positive_ion, negative_ion),
                self.grid_m,
                self.stack,
                self.material,
                illuminated=self.illuminated,
                V_app=self.V_app_V,
                phi_frozen=phi,
            ),
            layout.node_count,
        )
        if rhs.P_neg is None:
            raise ValueError("dual-ion RHS omitted the negative-ion species")
        interior = slice(1, layout.node_count - 1)
        electron_rate = n * rate[layout.electron_slice] - rhs.n
        hole_rate = p * rate[layout.hole_slice] - rhs.p
        ion_coordinate_rate = np.stack(
            (
                rate[layout.positive_ion_slice],
                rate[layout.negative_ion_slice],
            ),
            axis=1,
        )
        physical_ion_rate = np.einsum(
            "nij,nj->ni",
            self.ion_coordinate_jacobian_m3(coordinate),
            ion_coordinate_rate,
        )
        positive_ion_rate = physical_ion_rate[:, 0] - rhs.P
        negative_ion_rate = physical_ion_rate[:, 1] - rhs.P_neg

        normalized = np.zeros(layout.size, dtype=float)
        normalized[1 : layout.node_count - 1] = (
            electron_rate[interior] / layout.electron_rate_scale_m3_s[interior]
        )
        normalized[
            layout.node_count + 1 : 2 * layout.node_count - 1
        ] = hole_rate[interior] / layout.hole_rate_scale_m3_s[interior]
        normalized[layout.positive_ion_slice] = (
            positive_ion_rate / layout.positive_ion_rate_scale_m3_s
        )
        normalized[layout.negative_ion_slice] = (
            negative_ion_rate / layout.negative_ion_rate_scale_m3_s
        )

        boundary_nodes = np.array([0, layout.node_count - 1], dtype=int)
        electron_target = np.array(
            [self.material.n_L, self.material.n_R], dtype=float
        )
        hole_target = np.array(
            [self.material.p_L, self.material.p_R], dtype=float
        )
        electron_boundary = np.log(n[boundary_nodes] / electron_target)
        hole_boundary = np.log(p[boundary_nodes] / hole_target)
        normalized[0] = electron_boundary[0]
        normalized[layout.node_count - 1] = electron_boundary[1]
        normalized[layout.node_count] = hole_boundary[0]
        normalized[2 * layout.node_count - 1] = hole_boundary[1]

        if self.material.P_ion0_neg is None:
            raise ValueError("negative-ion reference charge is unavailable")
        rho = Q * (
            p
            - n
            + positive_ion
            - self.material.P_ion0
            - negative_ion
            + self.material.P_ion0_neg
            + self.material.N_D
            - self.material.N_A
        )
        capacitance = self.material.poisson_factor.C
        poisson = (
            capacitance[1:] * (phi[2:] - phi[1:-1])
            - capacitance[:-1] * (phi[1:-1] - phi[:-2])
            + rho[1:-1] * self.material.poisson_factor.h_cell
        )
        potential_boundary = np.array(
            [
                phi[0],
                phi[-1] - poisson_right_boundary(self.material, self.V_app_V),
            ],
            dtype=float,
        )
        potential_rows = normalized[layout.potential_slice]
        potential_rows[0] = potential_boundary[0] / layout.potential_scale_V
        potential_rows[-1] = potential_boundary[1] / layout.potential_scale_V
        potential_rows[1:-1] = poisson / layout.poisson_scale_C_m2

        carrier_rows = np.concatenate(
            (
                normalized[1 : layout.node_count - 1],
                normalized[
                    layout.node_count + 1 : 2 * layout.node_count - 1
                ],
            )
        )
        positive_rows = normalized[layout.positive_ion_slice]
        negative_rows = normalized[layout.negative_ion_slice]
        differential = normalized[layout.differential_mask]
        algebraic = normalized[layout.algebraic_mask]
        carrier_boundary = np.concatenate((electron_boundary, hole_boundary))
        return DualIonDAEResidualReport(
            normalized_residual=_readonly_f64(
                normalized,
                shape=(layout.size,),
                name="dual-ion normalized residual",
            ),
            electron_rate_residual_m3_s=_readonly_f64(
                electron_rate[interior],
                shape=(layout.node_count - 2,),
                name="electron rate residual",
            ),
            hole_rate_residual_m3_s=_readonly_f64(
                hole_rate[interior],
                shape=(layout.node_count - 2,),
                name="hole rate residual",
            ),
            positive_ion_rate_residual_m3_s=_readonly_f64(
                positive_ion_rate,
                shape=(layout.node_count,),
                name="positive-ion rate residual",
            ),
            negative_ion_rate_residual_m3_s=_readonly_f64(
                negative_ion_rate,
                shape=(layout.node_count,),
                name="negative-ion rate residual",
            ),
            poisson_residual_C_m2=_readonly_f64(
                poisson,
                shape=(layout.node_count - 2,),
                name="Poisson residual",
            ),
            carrier_boundary_residual_log=_readonly_f64(
                carrier_boundary,
                shape=(4,),
                name="carrier boundary residual",
            ),
            potential_boundary_residual_V=_readonly_f64(
                potential_boundary,
                shape=(2,),
                name="potential boundary residual",
            ),
            positive_ion_inventory_residual_m2_s=dual_cell_integral(
                self.grid_m,
                positive_ion_rate,
            ),
            negative_ion_inventory_residual_m2_s=dual_cell_integral(
                self.grid_m,
                negative_ion_rate,
            ),
            positive_ion_rhs_inventory_rate_m2_s=dual_cell_integral(
                self.grid_m,
                rhs.P,
            ),
            negative_ion_rhs_inventory_rate_m2_s=dual_cell_integral(
                self.grid_m,
                rhs.P_neg,
            ),
            max_normalized_carrier_residual=float(
                np.max(np.abs(carrier_rows), initial=0.0)
            ),
            max_normalized_positive_ion_residual=float(
                np.max(np.abs(positive_rows), initial=0.0)
            ),
            max_normalized_negative_ion_residual=float(
                np.max(np.abs(negative_rows), initial=0.0)
            ),
            max_normalized_differential_residual=float(
                np.max(np.abs(differential), initial=0.0)
            ),
            max_normalized_algebraic_residual=float(
                np.max(np.abs(algebraic), initial=0.0)
            ),
            max_normalized_residual=float(
                np.max(np.abs(normalized), initial=0.0)
            ),
        )

    def residual(self, coordinate: np.ndarray, derivative: np.ndarray) -> np.ndarray:
        return self.residual_report(coordinate, derivative).normalized_residual

    def derivative_jacobian(self, coordinate: np.ndarray) -> np.ndarray:
        """Return exact ``dF/d(qdot)`` including shared-site cross terms."""
        layout = self.layout
        n, p, _positive_ion, _negative_ion, _phi = self.physical_fields(
            coordinate
        )
        result = np.zeros((layout.size, layout.size), dtype=float)
        interior = np.arange(1, layout.node_count - 1)
        result[interior, interior] = (
            n[interior] / layout.electron_rate_scale_m3_s[interior]
        )
        hole_rows = layout.node_count + interior
        result[hole_rows, hole_rows] = (
            p[interior] / layout.hole_rate_scale_m3_s[interior]
        )
        ion_mass = self.ion_coordinate_jacobian_m3(coordinate)
        positive_offset = 2 * layout.node_count
        negative_offset = 3 * layout.node_count
        for node in range(layout.node_count):
            result[positive_offset + node, positive_offset + node] = (
                ion_mass[node, 0, 0]
                / layout.positive_ion_rate_scale_m3_s[node]
            )
            result[positive_offset + node, negative_offset + node] = (
                ion_mass[node, 0, 1]
                / layout.positive_ion_rate_scale_m3_s[node]
            )
            result[negative_offset + node, positive_offset + node] = (
                ion_mass[node, 1, 0]
                / layout.negative_ion_rate_scale_m3_s[node]
            )
            result[negative_offset + node, negative_offset + node] = (
                ion_mass[node, 1, 1]
                / layout.negative_ion_rate_scale_m3_s[node]
            )
        return result

    def algebraic_state_jacobian(self, coordinate: np.ndarray) -> np.ndarray:
        n, p, _positive_ion, _negative_ion, _phi = self.physical_fields(
            coordinate
        )
        ion_mass = self.ion_coordinate_jacobian_m3(coordinate)
        ion_charge_derivative = ion_mass[:, 0, :] - ion_mass[:, 1, :]
        layout = self.layout
        count = layout.node_count
        result = np.zeros((layout.size, layout.size), dtype=float)
        for index in (0, count - 1, count, 2 * count - 1):
            result[index, index] = 1.0

        positive_offset = 2 * count
        negative_offset = 3 * count
        potential_offset = 4 * count
        result[potential_offset, potential_offset] = 1.0 / layout.potential_scale_V
        result[-1, -1] = 1.0 / layout.potential_scale_V
        capacitance = self.material.poisson_factor.C
        widths = self.material.poisson_factor.h_cell
        for local, node in enumerate(range(1, count - 1)):
            row = potential_offset + node
            scale = layout.poisson_scale_C_m2[local]
            result[row, node] = -Q * n[node] * widths[local] / scale
            result[row, count + node] = Q * p[node] * widths[local] / scale
            result[row, positive_offset + node] = (
                Q * ion_charge_derivative[node, 0] * widths[local] / scale
            )
            result[row, negative_offset + node] = (
                Q * ion_charge_derivative[node, 1] * widths[local] / scale
            )
            result[row, potential_offset + node - 1] = (
                capacitance[node - 1] / scale
            )
            result[row, potential_offset + node] = -(
                capacitance[node - 1] + capacitance[node]
            ) / scale
            result[row, potential_offset + node + 1] = capacitance[node] / scale
        return result


def _validate_dual_ion_capability(
    stack: DeviceStack,
    material: MaterialArrays,
    packed_state: np.ndarray,
    node_count: int,
) -> tuple[StateVec, bool]:
    violations: list[str] = []
    if len(stack.layers) != 1:
        violations.append("exactly one electrical layer is required")
    if material.interface_nodes:
        violations.append("physical interfaces are not supported")
    if material.N_iface_state:
        violations.append("dynamic interface states are not supported")
    if material.iface_qss_exclusive_transport:
        violations.append("algebraic QSS interfaces are not supported")
    if material.has_selective_contacts:
        violations.append("selective contacts are not supported")
    if not material.has_dual_ions:
        violations.append("both positive and negative mobile ions are required")
    required_negative = (
        material.D_ion_neg_node,
        material.D_ion_neg_face,
        material.P_ion0_neg,
        material.P_lim_neg_node,
        material.P_lim_neg_face,
    )
    if any(value is None for value in required_negative):
        violations.append("negative-ion material arrays are incomplete")
    if np.any(material.D_ion_node <= 0.0):
        violations.append("positive ions must be active at every node")
    if material.D_ion_neg_node is not None and np.any(
        material.D_ion_neg_node <= 0.0
    ):
        violations.append("negative ions must be active at every node")
    if (
        np.any(material.P_ion0 <= 0.0)
        or np.any(material.P_lim_node <= material.P_ion0)
    ):
        violations.append("positive-ion reference must lie inside site limits")
    if (
        material.P_ion0_neg is not None
        and material.P_lim_neg_node is not None
        and (
            np.any(material.P_ion0_neg <= 0.0)
            or np.any(material.P_lim_neg_node <= material.P_ion0_neg)
        )
    ):
        violations.append("negative-ion reference must lie inside site limits")
    shared_site = bool(
        material.ion_steric_diffusion_only and material.ion_steric_shared_site
    )
    if (
        shared_site
        and material.P_lim_neg_node is not None
        and not np.array_equal(material.P_lim_node, material.P_lim_neg_node)
    ):
        violations.append("shared-site dual ions require one common site limit")
    if violations:
        raise DAECapabilityError("; ".join(violations))

    state = np.asarray(packed_state, dtype=float)
    if state.shape != (4 * node_count,) or not np.all(np.isfinite(state)):
        raise ValueError("reference_state must be a finite dual-ion-layout vector")
    unpacked = StateVec.unpack(state, node_count)
    if unpacked.P_neg is None:
        raise ValueError("reference_state must contain the negative-ion species")
    if np.any(unpacked.n <= 0.0) or np.any(unpacked.p <= 0.0):
        raise ValueError("reference carrier densities must be strictly positive")
    if np.any(unpacked.P <= 0.0) or np.any(unpacked.P >= material.P_lim_node):
        raise ValueError("reference positive-ion density must be inside site limits")
    assert material.P_lim_neg_node is not None
    if np.any(unpacked.P_neg <= 0.0) or np.any(
        unpacked.P_neg >= material.P_lim_neg_node
    ):
        raise ValueError("reference negative-ion density must be inside site limits")
    if shared_site and np.any(
        unpacked.P + unpacked.P_neg >= material.P_lim_node
    ):
        raise ValueError("reference shared-site total occupancy must be below one")
    return unpacked, shared_site


def build_dual_ion_dae(
    grid_m: np.ndarray,
    stack: DeviceStack,
    reference_state: np.ndarray,
    *,
    V_app_V: float = 0.0,
    illuminated: bool = False,
    carrier_reference_time_s: float = 1.0e-9,
    ion_reference_time_s: float = 1.0,
    material: MaterialArrays | None = None,
) -> DualIonDAE:
    """Build the parked dual-mobile-ion DAE capability."""
    grid = np.asarray(grid_m, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 3
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("grid_m must be finite and strictly increasing")
    for name, value in (
        ("carrier_reference_time_s", carrier_reference_time_s),
        ("ion_reference_time_s", ion_reference_time_s),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(V_app_V):
        raise ValueError("V_app_V must be finite")
    mat = build_material_arrays(grid, stack) if material is None else material
    if mat.poisson_factor.N != grid.size:
        raise ValueError("material Poisson factor does not match the DAE grid")
    state, shared_site = _validate_dual_ion_capability(
        stack,
        mat,
        reference_state,
        grid.size,
    )
    assert state.P_neg is not None
    assert mat.P_lim_neg_node is not None
    assert mat.P_ion0_neg is not None
    n = np.array(state.n, copy=True)
    p = np.array(state.p, copy=True)
    n[[0, -1]] = (mat.n_L, mat.n_R)
    p[[0, -1]] = (mat.p_L, mat.p_R)
    positive_ion = np.array(state.P, copy=True)
    negative_ion = np.array(state.P_neg, copy=True)
    if shared_site:
        vacancy_fraction = 1.0 - (
            positive_ion + negative_ion
        ) / mat.P_lim_node
        positive_coordinate_reference = np.log(
            positive_ion / mat.P_lim_node
        ) - np.log(vacancy_fraction)
        negative_coordinate_reference = np.log(
            negative_ion / mat.P_lim_node
        ) - np.log(vacancy_fraction)
    else:
        positive_fraction = positive_ion / mat.P_lim_node
        negative_fraction = negative_ion / mat.P_lim_neg_node
        positive_coordinate_reference = np.log(positive_fraction) - np.log1p(
            -positive_fraction
        )
        negative_coordinate_reference = np.log(negative_fraction) - np.log1p(
            -negative_fraction
        )

    potential_scale = max(float(mat.V_T_device), 1.0e-3)
    charge_scale = Q * (
        np.abs(n[1:-1])
        + np.abs(p[1:-1])
        + np.abs(mat.N_D[1:-1])
        + np.abs(mat.N_A[1:-1])
        + np.abs(mat.P_ion0[1:-1])
        + np.abs(mat.P_ion0_neg[1:-1])
    ) * mat.poisson_factor.h_cell
    dielectric_charge = (
        mat.poisson_factor.C[:-1] + mat.poisson_factor.C[1:]
    ) * potential_scale
    poisson_scale = np.maximum.reduce(
        (
            charge_scale,
            dielectric_charge,
            np.full(grid.size - 2, np.finfo(float).tiny),
        )
    )
    differential_mask = np.zeros(5 * grid.size, dtype=bool)
    differential_mask[1 : grid.size - 1] = True
    differential_mask[grid.size + 1 : 2 * grid.size - 1] = True
    differential_mask[2 * grid.size : 4 * grid.size] = True
    algebraic_mask = ~differential_mask
    for array in (differential_mask, algebraic_mask):
        array.setflags(write=False)

    layout = DualIonDAELayout(
        node_count=grid.size,
        shared_site=shared_site,
        electron_reference_m3=_readonly_f64(
            n, shape=(grid.size,), name="electron reference"
        ),
        hole_reference_m3=_readonly_f64(
            p, shape=(grid.size,), name="hole reference"
        ),
        positive_ion_reference_m3=_readonly_f64(
            positive_ion,
            shape=(grid.size,),
            name="positive-ion reference",
        ),
        negative_ion_reference_m3=_readonly_f64(
            negative_ion,
            shape=(grid.size,),
            name="negative-ion reference",
        ),
        positive_ion_site_limit_m3=_readonly_f64(
            mat.P_lim_node,
            shape=(grid.size,),
            name="positive-ion site limit",
        ),
        negative_ion_site_limit_m3=_readonly_f64(
            mat.P_lim_neg_node,
            shape=(grid.size,),
            name="negative-ion site limit",
        ),
        positive_ion_coordinate_reference=_readonly_f64(
            positive_coordinate_reference,
            shape=(grid.size,),
            name="positive-ion coordinate reference",
        ),
        negative_ion_coordinate_reference=_readonly_f64(
            negative_coordinate_reference,
            shape=(grid.size,),
            name="negative-ion coordinate reference",
        ),
        electron_rate_scale_m3_s=_readonly_f64(
            np.maximum(n / carrier_reference_time_s, 1.0),
            shape=(grid.size,),
            name="electron rate scale",
        ),
        hole_rate_scale_m3_s=_readonly_f64(
            np.maximum(p / carrier_reference_time_s, 1.0),
            shape=(grid.size,),
            name="hole rate scale",
        ),
        positive_ion_rate_scale_m3_s=_readonly_f64(
            np.maximum(positive_ion / ion_reference_time_s, 1.0),
            shape=(grid.size,),
            name="positive-ion rate scale",
        ),
        negative_ion_rate_scale_m3_s=_readonly_f64(
            np.maximum(negative_ion / ion_reference_time_s, 1.0),
            shape=(grid.size,),
            name="negative-ion rate scale",
        ),
        poisson_scale_C_m2=_readonly_f64(
            poisson_scale,
            shape=(grid.size - 2,),
            name="Poisson scale",
        ),
        potential_scale_V=potential_scale,
        differential_mask=differential_mask,
        algebraic_mask=algebraic_mask,
    )
    return DualIonDAE(
        grid_m=_readonly_f64(grid, shape=(grid.size,), name="grid"),
        stack=stack,
        material=mat,
        layout=layout,
        V_app_V=float(V_app_V),
        illuminated=bool(illuminated),
    )


def project_dual_ion_algebraic_state(
    model: DualIonDAE,
    coordinate: np.ndarray,
) -> np.ndarray:
    """Pin ohmic reservoirs and solve Poisson at fixed carrier/ion coordinates."""
    value = np.asarray(coordinate, dtype=float)
    if value.shape != (model.layout.size,) or not np.all(np.isfinite(value)):
        raise ValueError("coordinate must be finite and layout-sized")
    result = np.array(value, copy=True)
    count = model.layout.node_count
    result[0] = 0.0
    result[count - 1] = 0.0
    result[count] = 0.0
    result[2 * count - 1] = 0.0
    n, p, positive_ion, negative_ion, _phi = model.physical_fields(result)
    assert model.material.P_ion0_neg is not None
    rho = Q * (
        p
        - n
        + positive_ion
        - model.material.P_ion0
        - negative_ion
        + model.material.P_ion0_neg
        + model.material.N_D
        - model.material.N_A
    )
    result[model.layout.potential_slice] = solve_poisson_prefactored(
        model.material.poisson_factor,
        rho,
        phi_left=0.0,
        phi_right=poisson_right_boundary(model.material, model.V_app_V),
    )
    return result


def build_dual_ion_consistent_initial_condition(
    model: DualIonDAE,
    *,
    residual_tolerance: float = 1.0e-10,
) -> DualIonDAEConsistentInitialCondition:
    """Construct a deterministic Poisson- and rate-compatible initial pair."""
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    coordinate = project_dual_ion_algebraic_state(
        model,
        np.zeros(model.layout.size, dtype=float),
    )
    derivative = model.compatible_derivative(coordinate)
    report = model.residual_report(coordinate, derivative)
    certified = report.max_normalized_residual <= residual_tolerance
    physical = model.packed_physical_state(coordinate)
    _n, _p, _positive_ion, _negative_ion, potential = model.physical_fields(
        coordinate
    )
    coordinate_ro = _readonly_f64(
        coordinate,
        shape=(model.layout.size,),
        name="dual-ion consistent coordinate",
    )
    derivative_ro = _readonly_f64(
        derivative,
        shape=(model.layout.size,),
        name="dual-ion consistent derivative",
    )
    physical_ro = _readonly_f64(
        physical,
        shape=(4 * model.layout.node_count,),
        name="dual-ion consistent physical state",
    )
    potential_ro = _readonly_f64(
        potential,
        shape=(model.layout.node_count,),
        name="dual-ion consistent potential",
    )
    return DualIonDAEConsistentInitialCondition(
        coordinate=coordinate_ro,
        derivative=derivative_ro,
        physical_state=physical_ro,
        potential_V=potential_ro,
        report=report,
        certified=bool(certified),
        state_sha256=_state_sha256(
            "dual-mobile-ion-dae-initial-v1",
            model.grid_m,
            coordinate_ro,
            derivative_ro,
            physical_ro,
            potential_ro,
        ),
    )


def finite_difference_dual_ion_state_jacobian(
    model: DualIonDAE,
    coordinate: np.ndarray,
    derivative: np.ndarray,
    *,
    relative_step: float = 1.0e-6,
) -> np.ndarray:
    """Independent central reference for dual-ion ``dF/dq``."""
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    value = np.asarray(coordinate, dtype=float)
    if value.shape != (model.layout.size,):
        raise ValueError("coordinate does not match the dual-ion layout")
    result = np.empty((model.layout.size, model.layout.size), dtype=float)
    potential_start = model.layout.potential_slice.start
    assert potential_start is not None
    for column in range(model.layout.size):
        scale = 1.0 if column < potential_start else model.layout.potential_scale_V
        step = relative_step * max(abs(value[column]), scale)
        plus = value.copy()
        minus = value.copy()
        plus[column] += step
        minus[column] -= step
        result[:, column] = (
            model.residual(plus, derivative) - model.residual(minus, derivative)
        ) / (2.0 * step)
    return result


def finite_difference_dual_ion_derivative_jacobian(
    model: DualIonDAE,
    coordinate: np.ndarray,
    derivative: np.ndarray,
    *,
    relative_step: float = 1.0e-6,
) -> np.ndarray:
    """Independent central reference for dual-ion ``dF/d(qdot)``."""
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    rate = np.asarray(derivative, dtype=float)
    if rate.shape != (model.layout.size,):
        raise ValueError("derivative does not match the dual-ion layout")
    analytic_scale = model.derivative_jacobian(coordinate)
    result = np.empty((model.layout.size, model.layout.size), dtype=float)
    for column in range(model.layout.size):
        column_norm = float(np.max(np.abs(analytic_scale[:, column]), initial=0.0))
        rate_scale = 1.0 / column_norm if column_norm > 0.0 else 1.0
        step = relative_step * max(abs(rate[column]), rate_scale)
        plus = rate.copy()
        minus = rate.copy()
        plus[column] += step
        minus[column] -= step
        result[:, column] = (
            model.residual(coordinate, plus) - model.residual(coordinate, minus)
        ) / (2.0 * step)
    return result


__all__ = [
    "DualIonDAE",
    "DualIonDAEConsistentInitialCondition",
    "DualIonDAELayout",
    "DualIonDAEResidualReport",
    "build_dual_ion_consistent_initial_condition",
    "build_dual_ion_dae",
    "finite_difference_dual_ion_derivative_jacobian",
    "finite_difference_dual_ion_state_jacobian",
    "project_dual_ion_algebraic_state",
]
