"""Research-only semi-explicit DAE backbone.

The first Phase-4 slice keeps carrier densities differential and exposes the
electrostatic potential as an algebraic variable.  It intentionally supports
only a single electrical layer with ohmic contacts, no mobile ions, and no
interface state.  Production transients continue to use ``solver.mol``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.mol import (
    MaterialArrays,
    StateVec,
    assemble_rhs,
    build_material_arrays,
    poisson_right_boundary,
)


class DAECapabilityError(ValueError):
    """The requested stack is outside the first research DAE slice."""


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


@dataclass(frozen=True, slots=True)
class SemiExplicitDAELayout:
    """Coordinate layout and row classification for ``(log n, log p, phi)``."""

    node_count: int
    electron_reference_m3: np.ndarray
    hole_reference_m3: np.ndarray
    electron_rate_scale_m3_s: np.ndarray
    hole_rate_scale_m3_s: np.ndarray
    poisson_scale_C_m2: np.ndarray
    potential_scale_V: float
    differential_mask: np.ndarray
    algebraic_mask: np.ndarray

    @property
    def size(self) -> int:
        return 3 * self.node_count

    @property
    def electron_slice(self) -> slice:
        return slice(0, self.node_count)

    @property
    def hole_slice(self) -> slice:
        return slice(self.node_count, 2 * self.node_count)

    @property
    def potential_slice(self) -> slice:
        return slice(2 * self.node_count, 3 * self.node_count)


@dataclass(frozen=True, slots=True)
class DAEResidualReport:
    """Separate differential and algebraic residual evidence."""

    normalized_residual: np.ndarray
    electron_rate_residual_m3_s: np.ndarray
    hole_rate_residual_m3_s: np.ndarray
    poisson_residual_C_m2: np.ndarray
    carrier_boundary_residual_log: np.ndarray
    potential_boundary_residual_V: np.ndarray
    max_normalized_differential_residual: float
    max_normalized_algebraic_residual: float
    max_normalized_residual: float


@dataclass(frozen=True, slots=True)
class DAEConsistentInitialCondition:
    """A reproducible state/derivative pair satisfying every DAE row."""

    coordinate: np.ndarray
    derivative: np.ndarray
    physical_state: np.ndarray
    potential_V: np.ndarray
    report: DAEResidualReport
    certified: bool
    state_sha256: str


@dataclass(frozen=True, slots=True)
class NoIonNoInterfaceDAE:
    """Narrow reference DAE with explicit Poisson algebraic coordinates."""

    grid_m: np.ndarray
    stack: DeviceStack
    material: MaterialArrays
    layout: SemiExplicitDAELayout
    V_app_V: float
    illuminated: bool

    def physical_fields(
        self,
        coordinate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        value = np.asarray(coordinate, dtype=float)
        if value.shape != (self.layout.size,) or not np.all(np.isfinite(value)):
            raise ValueError("DAE coordinate must be a finite layout-sized vector")
        with np.errstate(over="ignore", invalid="ignore"):
            n = self.layout.electron_reference_m3 * np.exp(
                value[self.layout.electron_slice]
            )
            p = self.layout.hole_reference_m3 * np.exp(
                value[self.layout.hole_slice]
            )
        phi = np.asarray(value[self.layout.potential_slice], dtype=float)
        if not np.all(np.isfinite(n)) or not np.all(np.isfinite(p)):
            raise ValueError("DAE log-density coordinate overflowed")
        return n, p, phi

    def packed_physical_state(self, coordinate: np.ndarray) -> np.ndarray:
        n, p, _phi = self.physical_fields(coordinate)
        return StateVec.pack(n, p, np.zeros(self.layout.node_count))

    def residual_report(
        self,
        coordinate: np.ndarray,
        derivative: np.ndarray,
    ) -> DAEResidualReport:
        layout = self.layout
        rate = np.asarray(derivative, dtype=float)
        if rate.shape != (layout.size,) or not np.all(np.isfinite(rate)):
            raise ValueError("DAE derivative must be a finite layout-sized vector")
        n, p, phi = self.physical_fields(coordinate)
        packed = StateVec.pack(n, p, np.zeros(layout.node_count))
        rhs = StateVec.unpack(
            assemble_rhs(
                0.0,
                packed,
                self.grid_m,
                self.stack,
                self.material,
                illuminated=self.illuminated,
                V_app=self.V_app_V,
                phi_frozen=phi,
            ),
            layout.node_count,
        )

        electron_rate = n * rate[layout.electron_slice] - rhs.n
        hole_rate = p * rate[layout.hole_slice] - rhs.p
        normalized = np.zeros(layout.size, dtype=float)
        interior = slice(1, -1)
        normalized[1 : layout.node_count - 1] = (
            electron_rate[interior]
            / layout.electron_rate_scale_m3_s[interior]
        )
        normalized[layout.node_count + 1 : 2 * layout.node_count - 1] = (
            hole_rate[interior]
            / layout.hole_rate_scale_m3_s[interior]
        )

        electron_target = np.array(
            [self.material.n_L, self.material.n_R], dtype=float
        )
        hole_target = np.array(
            [self.material.p_L, self.material.p_R], dtype=float
        )
        boundary_nodes = np.array([0, layout.node_count - 1], dtype=int)
        electron_boundary = np.log(n[boundary_nodes] / electron_target)
        hole_boundary = np.log(p[boundary_nodes] / hole_target)
        normalized[0] = electron_boundary[0]
        normalized[layout.node_count - 1] = electron_boundary[1]
        normalized[layout.node_count] = hole_boundary[0]
        normalized[2 * layout.node_count - 1] = hole_boundary[1]

        rho = Q * (p - n + self.material.N_D - self.material.N_A)
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

        differential = normalized[layout.differential_mask]
        algebraic = normalized[layout.algebraic_mask]
        carrier_boundary = np.concatenate((electron_boundary, hole_boundary))
        return DAEResidualReport(
            normalized_residual=_readonly_f64(
                normalized,
                shape=(layout.size,),
                name="normalized DAE residual",
            ),
            electron_rate_residual_m3_s=_readonly_f64(
                electron_rate[1:-1],
                shape=(layout.node_count - 2,),
                name="electron rate residual",
            ),
            hole_rate_residual_m3_s=_readonly_f64(
                hole_rate[1:-1],
                shape=(layout.node_count - 2,),
                name="hole rate residual",
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

    def residual(
        self,
        coordinate: np.ndarray,
        derivative: np.ndarray,
    ) -> np.ndarray:
        return self.residual_report(coordinate, derivative).normalized_residual

    def derivative_jacobian(self, coordinate: np.ndarray) -> np.ndarray:
        """Return exact ``dF/d(qdot)`` in scaled log-density coordinates."""
        n, p, _phi = self.physical_fields(coordinate)
        layout = self.layout
        result = np.zeros((layout.size, layout.size), dtype=float)
        indices = np.arange(1, layout.node_count - 1)
        result[indices, indices] = (
            n[indices] / layout.electron_rate_scale_m3_s[indices]
        )
        hole_indices = layout.node_count + indices
        result[hole_indices, hole_indices] = (
            p[indices] / layout.hole_rate_scale_m3_s[indices]
        )
        return result

    def algebraic_state_jacobian(self, coordinate: np.ndarray) -> np.ndarray:
        """Return exact boundary and Poisson rows of ``dF/dq``."""
        n, p, _phi = self.physical_fields(coordinate)
        layout = self.layout
        count = layout.node_count
        result = np.zeros((layout.size, layout.size), dtype=float)
        result[0, 0] = 1.0
        result[count - 1, count - 1] = 1.0
        result[count, count] = 1.0
        result[2 * count - 1, 2 * count - 1] = 1.0

        potential_offset = 2 * count
        result[potential_offset, potential_offset] = 1.0 / layout.potential_scale_V
        result[-1, -1] = 1.0 / layout.potential_scale_V
        capacitance = self.material.poisson_factor.C
        widths = self.material.poisson_factor.h_cell
        for local, node in enumerate(range(1, count - 1)):
            row = potential_offset + node
            scale = layout.poisson_scale_C_m2[local]
            result[row, node] = -Q * n[node] * widths[local] / scale
            result[row, count + node] = Q * p[node] * widths[local] / scale
            result[row, potential_offset + node - 1] = (
                capacitance[node - 1] / scale
            )
            result[row, potential_offset + node] = -(
                capacitance[node - 1] + capacitance[node]
            ) / scale
            result[row, potential_offset + node + 1] = capacitance[node] / scale
        return result


def _validate_first_slice_capability(
    material: MaterialArrays,
    packed_state: np.ndarray,
    node_count: int,
) -> StateVec:
    violations: list[str] = []
    if material.interface_nodes:
        violations.append("physical interfaces are not supported")
    if material.N_iface_state:
        violations.append("dynamic interface states are not supported")
    if material.iface_qss_exclusive_transport:
        violations.append("algebraic QSS interfaces are not supported")
    if material.has_selective_contacts:
        violations.append("selective contacts are not supported")
    if material.has_dual_ions:
        violations.append("dual mobile ions are not supported")
    if np.any(material.D_ion_node != 0.0) or np.any(material.P_ion0 != 0.0):
        violations.append("mobile ions are not supported")
    if violations:
        raise DAECapabilityError("; ".join(violations))
    state = np.asarray(packed_state, dtype=float)
    if state.shape != (3 * node_count,) or not np.all(np.isfinite(state)):
        raise ValueError("reference_state must be a finite single-ion-layout vector")
    unpacked = StateVec.unpack(state, node_count)
    if np.any(unpacked.P != 0.0):
        raise DAECapabilityError("the structural ion block must be exactly zero")
    if np.any(unpacked.n <= 0.0) or np.any(unpacked.p <= 0.0):
        raise ValueError("reference carrier densities must be strictly positive")
    return unpacked


def build_no_ion_no_interface_dae(
    grid_m: np.ndarray,
    stack: DeviceStack,
    reference_state: np.ndarray,
    *,
    V_app_V: float = 0.0,
    illuminated: bool = False,
    reference_time_s: float = 1.0e-6,
    material: MaterialArrays | None = None,
) -> NoIonNoInterfaceDAE:
    """Build the parked Phase-4 DAE reference without changing MoL routes."""
    grid = np.asarray(grid_m, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 3
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("grid_m must be finite and strictly increasing")
    if not np.isfinite(V_app_V):
        raise ValueError("V_app_V must be finite")
    if not np.isfinite(reference_time_s) or reference_time_s <= 0.0:
        raise ValueError("reference_time_s must be finite and positive")
    mat = build_material_arrays(grid, stack) if material is None else material
    if mat.poisson_factor.N != grid.size:
        raise ValueError("material Poisson factor does not match the DAE grid")
    state = _validate_first_slice_capability(
        mat,
        reference_state,
        grid.size,
    )
    n = np.array(state.n, copy=True)
    p = np.array(state.p, copy=True)
    n[[0, -1]] = (mat.n_L, mat.n_R)
    p[[0, -1]] = (mat.p_L, mat.p_R)
    potential_scale = max(float(mat.V_T_device), 1.0e-3)
    carrier_charge = Q * (
        np.abs(n[1:-1])
        + np.abs(p[1:-1])
        + np.abs(mat.N_D[1:-1])
        + np.abs(mat.N_A[1:-1])
    ) * mat.poisson_factor.h_cell
    dielectric_charge = (
        mat.poisson_factor.C[:-1] + mat.poisson_factor.C[1:]
    ) * potential_scale
    poisson_scale = np.maximum.reduce(
        (
            carrier_charge,
            dielectric_charge,
            np.full(grid.size - 2, np.finfo(float).tiny),
        )
    )
    differential_mask = np.zeros(3 * grid.size, dtype=bool)
    differential_mask[1 : grid.size - 1] = True
    differential_mask[grid.size + 1 : 2 * grid.size - 1] = True
    algebraic_mask = ~differential_mask
    for array in (differential_mask, algebraic_mask):
        array.setflags(write=False)
    layout = SemiExplicitDAELayout(
        node_count=grid.size,
        electron_reference_m3=_readonly_f64(
            n, shape=(grid.size,), name="electron reference"
        ),
        hole_reference_m3=_readonly_f64(
            p, shape=(grid.size,), name="hole reference"
        ),
        electron_rate_scale_m3_s=_readonly_f64(
            np.maximum(n / reference_time_s, 1.0),
            shape=(grid.size,),
            name="electron rate scale",
        ),
        hole_rate_scale_m3_s=_readonly_f64(
            np.maximum(p / reference_time_s, 1.0),
            shape=(grid.size,),
            name="hole rate scale",
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
    return NoIonNoInterfaceDAE(
        grid_m=_readonly_f64(grid, shape=(grid.size,), name="grid"),
        stack=stack,
        material=mat,
        layout=layout,
        V_app_V=float(V_app_V),
        illuminated=bool(illuminated),
    )


def project_algebraic_state(
    model: NoIonNoInterfaceDAE,
    coordinate: np.ndarray,
) -> np.ndarray:
    """Enforce carrier boundary values and solve the exact Poisson constraint."""
    value = np.asarray(coordinate, dtype=float)
    if value.shape != (model.layout.size,) or not np.all(np.isfinite(value)):
        raise ValueError("coordinate must be a finite layout-sized vector")
    result = np.array(value, copy=True)
    count = model.layout.node_count
    result[0] = 0.0
    result[count - 1] = 0.0
    result[count] = 0.0
    result[2 * count - 1] = 0.0
    n, p, _phi = model.physical_fields(result)
    rho = Q * (p - n + model.material.N_D - model.material.N_A)
    result[model.layout.potential_slice] = solve_poisson_prefactored(
        model.material.poisson_factor,
        rho,
        phi_left=0.0,
        phi_right=poisson_right_boundary(model.material, model.V_app_V),
    )
    return result


def build_consistent_initial_condition(
    model: NoIonNoInterfaceDAE,
    *,
    residual_tolerance: float = 1.0e-10,
) -> DAEConsistentInitialCondition:
    """Construct deterministic algebraic state and compatible carrier rates."""
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    coordinate = project_algebraic_state(
        model,
        np.zeros(model.layout.size, dtype=float),
    )
    packed = model.packed_physical_state(coordinate)
    _n, _p, phi = model.physical_fields(coordinate)
    rhs = StateVec.unpack(
        assemble_rhs(
            0.0,
            packed,
            model.grid_m,
            model.stack,
            model.material,
            illuminated=model.illuminated,
            V_app=model.V_app_V,
            phi_frozen=phi,
        ),
        model.layout.node_count,
    )
    derivative = np.zeros(model.layout.size, dtype=float)
    derivative[1 : model.layout.node_count - 1] = (
        rhs.n[1:-1] / _n[1:-1]
    )
    derivative[
        model.layout.node_count + 1 : 2 * model.layout.node_count - 1
    ] = (
        rhs.p[1:-1] / _p[1:-1]
    )
    report = model.residual_report(coordinate, derivative)
    certified = report.max_normalized_residual <= residual_tolerance
    coordinate_ro = _readonly_f64(
        coordinate,
        shape=(model.layout.size,),
        name="consistent coordinate",
    )
    derivative_ro = _readonly_f64(
        derivative,
        shape=(model.layout.size,),
        name="consistent derivative",
    )
    packed_ro = _readonly_f64(
        packed,
        shape=(3 * model.layout.node_count,),
        name="consistent physical state",
    )
    potential_ro = _readonly_f64(
        phi,
        shape=(model.layout.node_count,),
        name="consistent potential",
    )
    return DAEConsistentInitialCondition(
        coordinate=coordinate_ro,
        derivative=derivative_ro,
        physical_state=packed_ro,
        potential_V=potential_ro,
        report=report,
        certified=bool(certified),
        state_sha256=_state_sha256(
            "no-ion-no-interface-dae-initial-v1",
            model.grid_m,
            coordinate_ro,
            derivative_ro,
            packed_ro,
            potential_ro,
        ),
    )


def finite_difference_state_jacobian(
    model: NoIonNoInterfaceDAE,
    coordinate: np.ndarray,
    derivative: np.ndarray,
    *,
    relative_step: float = 1.0e-6,
) -> np.ndarray:
    """Independent central reference for ``dF/dq``."""
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    value = np.asarray(coordinate, dtype=float)
    if value.shape != (model.layout.size,):
        raise ValueError("coordinate does not match the DAE layout")
    result = np.empty((model.layout.size, model.layout.size), dtype=float)
    for column in range(model.layout.size):
        scale = (
            1.0
            if column < 2 * model.layout.node_count
            else model.layout.potential_scale_V
        )
        step = relative_step * max(abs(value[column]), scale)
        plus = value.copy()
        minus = value.copy()
        plus[column] += step
        minus[column] -= step
        result[:, column] = (
            model.residual(plus, derivative) - model.residual(minus, derivative)
        ) / (2.0 * step)
    return result


def finite_difference_derivative_jacobian(
    model: NoIonNoInterfaceDAE,
    coordinate: np.ndarray,
    derivative: np.ndarray,
    *,
    relative_step: float = 1.0e-6,
) -> np.ndarray:
    """Independent central reference for ``dF/d(qdot)``."""
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    rate = np.asarray(derivative, dtype=float)
    if rate.shape != (model.layout.size,):
        raise ValueError("derivative does not match the DAE layout")
    result = np.empty((model.layout.size, model.layout.size), dtype=float)
    count = model.layout.node_count
    for column in range(model.layout.size):
        if column < count:
            rate_scale = (
                model.layout.electron_rate_scale_m3_s[column]
                / model.layout.electron_reference_m3[column]
            )
        elif column < 2 * count:
            local = column - count
            rate_scale = (
                model.layout.hole_rate_scale_m3_s[local]
                / model.layout.hole_reference_m3[local]
            )
        else:
            rate_scale = 1.0
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
    "DAECapabilityError",
    "DAEConsistentInitialCondition",
    "DAEResidualReport",
    "NoIonNoInterfaceDAE",
    "SemiExplicitDAELayout",
    "build_consistent_initial_condition",
    "build_no_ion_no_interface_dae",
    "finite_difference_derivative_jacobian",
    "finite_difference_state_jacobian",
    "project_algebraic_state",
]
