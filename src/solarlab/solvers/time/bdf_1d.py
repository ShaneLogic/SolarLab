"""Opt-in electronic/single-positive-ion adapters for physical-storage BDF.

All constitutive laws, admission controls and Newton row/step policies come
from the unchanged BE paths. Order1 uses their exact accounting implementation.
Order2 combines physical secants and the corresponding displacement derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np

from solarlab.contracts import frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.physics.electrostatics import contact_displacement
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.solvers.nonlinear.problem import Matrix, Vector
from solarlab.solvers.time.backward_euler import BackwardEuler1DStep, storage_mask, inventory_m2, step_accounting
from solarlab.solvers.time.ionic_backward_euler import (
    IonicBackwardEuler1DStep, step_accounting as ionic_accounting,
)
from solarlab.solvers.time.bdf import (
    BDFStep, HistoryPoint, StorageProblem,
)


class BDFProtocol(Protocol):
    """Read-only drive arrays, without dependence on an experiment value type."""

    @property
    def times_s(self) -> Vector: ...

    @property
    def voltage_V(self) -> Vector: ...

    @property
    def generation_scale(self) -> Vector: ...


@dataclass(frozen=True)
class BDFDrive:
    """Separate protocol knots and left/right voltage/generation values.

    Between knots interpolate from right[i] to left[i+1]. Unequal left/right
    values at a knot declare a discontinuity and trigger a consistent restart.
    Output samples are supplied independently through TransientProtocol.times_s.
    """

    times_s: Vector
    left_values: np.ndarray  # (knots,2): voltage[V], generation scale
    right_values: np.ndarray

    def __post_init__(self) -> None:
        for name in ("times_s", "left_values", "right_values"):
            if np.iscomplexobj(getattr(self, name)):
                raise ValueError("protocol values must be real")
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        if (self.times_s.ndim != 1 or len(self.times_s) < 2 or np.any(np.diff(self.times_s) <= 0)
                or self.left_values.shape != (len(self.times_s), 2) or self.right_values.shape != self.left_values.shape
                or np.any(self.left_values[:, 1] < 0) or np.any(self.right_values[:, 1] < 0)):
            raise ValueError("ordered protocol knots and matched nonnegative generation values required")

    @classmethod
    def from_protocol(cls, protocol: BDFProtocol) -> BDFDrive:
        values = np.column_stack((protocol.voltage_V, protocol.generation_scale))
        return cls(protocol.times_s, values, values)

    def values(self, t: float, side: str = "right") -> tuple[float, float]:
        exact = np.flatnonzero(self.times_s == t)
        if exact.size:
            data = self.left_values if side == "left" else self.right_values
            return float(data[exact[0], 0]), float(data[exact[0], 1])
        index = int(np.searchsorted(self.times_s, t)) - 1
        if index < 0 or index >= len(self.times_s) - 1:
            raise ValueError("drive evaluation outside its declared protocol")
        fraction = (t - self.times_s[index]) / (self.times_s[index + 1] - self.times_s[index])
        value = self.right_values[index] + fraction * (self.left_values[index + 1] - self.right_values[index])
        return float(value[0]), float(value[1])

    @property
    def slope_breakpoints_s(self) -> tuple[float, ...]:
        """Continuous protocol knots with different adjacent segment slopes."""
        slopes = (self.left_values[1:] - self.right_values[:-1]) / np.diff(self.times_s)[:, None]
        return tuple(float(self.times_s[index]) for index in range(1, len(self.times_s) - 1)
            if np.array_equal(self.left_values[index], self.right_values[index])
            and not np.array_equal(slopes[index - 1], slopes[index]))


@dataclass(frozen=True)
class ElectronicStorage1D:
    problem: DC1DProblem

    def residual(self, u: Vector) -> Vector:
        return self.problem.residual(u)

    def jacobian(self, u: Vector) -> Matrix:
        return self.problem.jacobian(u)

    def physical_storage(self, u: Vector) -> Vector:
        device = self.problem.device
        state = device.state(u, voltage_V=self.problem.voltage_V)
        return np.r_[np.zeros(device.node_count), inventory_m2(device, state).ravel()]

    def storage(self, u: Vector) -> Vector:
        result = self.physical_storage(u)
        result[self.problem.device.node_count:] *= storage_mask(self.problem.device).ravel()
        return result

    def storage_tangent(self, u: Vector) -> Matrix:
        return -self.problem.storage_jacobian(u)

    def physical_increment(self, u: Vector, old: ElectronicStorage1D, old_u: Vector) -> Vector:
        state = old.problem.device.state(old_u, voltage_V=old.problem.voltage_V)
        change = BackwardEuler1DStep(self.problem, state, 1.0).inventory_rate_m2_s(u)
        return np.r_[np.zeros(self.problem.device.node_count), change.ravel()]

    def storage_increment(self, u: Vector, old_problem: StorageProblem, old_u: Vector) -> Vector:
        if not isinstance(old_problem, ElectronicStorage1D):
            raise ValueError("electronic BDF history must keep its original reference problem")
        change = self.physical_increment(u, old_problem, old_u)
        change[self.problem.device.node_count:] *= storage_mask(self.problem.device).ravel()
        return change


@dataclass(frozen=True)
class IonicStorage1D:
    problem: Ionic1DProblem

    def __post_init__(self) -> None:
        if self.problem.device.ion_species_count != 1:
            raise ValueError("this BDF adapter supports exactly one positive ion population; dual/shared-site is unadmitted")

    @property
    def electronic(self) -> ElectronicStorage1D:
        return ElectronicStorage1D(self.problem.electronic_problem)

    def residual(self, u: Vector) -> Vector:
        return self.problem.residual(u)

    def jacobian(self, u: Vector) -> Matrix:
        return self.problem.jacobian(u)

    def physical_storage(self, u: Vector) -> Vector:
        device = self.problem.device
        state = device.state(u, voltage_V=self.problem.voltage_V)
        return np.r_[self.electronic.physical_storage(u[:3 * device.node_count]),
                     device.ion_geometry.cell_measures_m * state.positive_ion_density_m3]

    def storage(self, u: Vector) -> Vector:
        device = self.problem.device
        state = device.state(u, voltage_V=self.problem.voltage_V)
        return np.r_[self.electronic.storage(u[:3 * device.node_count]),
                     device.ion_geometry.cell_measures_m * state.positive_ion_density_m3]

    def storage_tangent(self, u: Vector) -> Matrix:
        return -self.problem.storage_jacobian(u)

    def physical_increment(self, u: Vector, old: IonicStorage1D, old_u: Vector) -> Vector:
        state = old.problem.device.state(old_u, voltage_V=old.problem.voltage_V)
        carrier, ion = IonicBackwardEuler1DStep(self.problem, state, 1.0).inventory_rate_m2_s(u)
        return np.r_[np.zeros(self.problem.device.node_count), carrier.ravel(), ion]

    def storage_increment(self, u: Vector, old_problem: StorageProblem, old_u: Vector) -> Vector:
        if not isinstance(old_problem, IonicStorage1D):
            raise ValueError("ionic BDF history must keep its original physical reference")
        change = self.physical_increment(u, old_problem, old_u)
        count = self.problem.device.node_count
        change[count:3 * count] *= storage_mask(self.problem.device.electronic_component).ravel()
        return change


def _be_step(step: BDFStep) -> BackwardEuler1DStep | IonicBackwardEuler1DStep:
    new, old = step.endpoint, step.history[0]
    if isinstance(new, ElectronicStorage1D) and isinstance(old.problem, ElectronicStorage1D):
        return BackwardEuler1DStep(new.problem, old.problem.problem.device.state(old.u,
            voltage_V=old.problem.problem.voltage_V), step.dt_s)
    if isinstance(new, IonicStorage1D) and isinstance(old.problem, IonicStorage1D):
        return IonicBackwardEuler1DStep(new.problem, old.problem.problem.device.state(old.u,
            voltage_V=old.problem.problem.voltage_V), step.dt_s)
    raise TypeError("matched electronic or single-ion BDF history required")


def bdf_seed(step: BDFStep, predictor: str, *, voltage_lift: bool) -> Vector:
    """Read the matched BE seed without changing its predictor or arithmetic."""
    return _be_step(step).initial_coordinates(predictor, voltage_lift=voltage_lift)


def _full_increment(new: ElectronicStorage1D | IonicStorage1D, u: Vector,
                    old: StorageProblem, old_u: Vector) -> Vector:
    if isinstance(new, ElectronicStorage1D) and isinstance(old, ElectronicStorage1D):
        return new.physical_increment(u, old, old_u)
    if isinstance(new, IonicStorage1D) and isinstance(old, IonicStorage1D):
        return new.physical_increment(u, old, old_u)
    raise ValueError("physical inventory history must keep the same adapter kind")


def _potential(adapter: ElectronicStorage1D | IonicStorage1D, u: Vector) -> Vector:
    if isinstance(adapter, ElectronicStorage1D):
        return adapter.problem.device.state(u, voltage_V=adapter.problem.voltage_V).potential_V
    return adapter.problem.device.state(u, voltage_V=adapter.problem.voltage_V).electronic_state.potential_V


def _displacement(adapter: ElectronicStorage1D | IonicStorage1D, u: Vector) -> tuple[Vector, Vector]:
    if isinstance(adapter, ElectronicStorage1D):
        state = adapter.problem.device.state(u, voltage_V=adapter.problem.voltage_V)
        device = adapter.problem.device
        rho = device.charge_ledger(state.electron_density_m3, state.hole_density_m3).space_charge_C_m3()
    else:
        state_i = adapter.problem.device.state(u, voltage_V=adapter.problem.voltage_V)
        state = state_i.electronic_state
        device = adapter.problem.device.electronic_component
        rho = adapter.problem.device.charge_ledger(state.electron_density_m3, state.hole_density_m3,
            state_i.positive_ion_density_m3).space_charge_C_m3()
    inner = device.poisson.face_displacement(state.potential_V)
    ports = contact_displacement(inner, rho, device.geometry.cell_measures_m)
    return np.r_[ports[0], inner, ports[1]], rho * device.geometry.cell_measures_m


def bdf_accounting(step: BDFStep, u: Vector) -> dict[str, Any]:
    """Original order1 readout, or full coefficient-consistent BDF2 accounting."""
    be = _be_step(step)
    data: dict[str, Any] = dict(step_accounting(be, u) if isinstance(be, BackwardEuler1DStep)
                               else ionic_accounting(be, u))
    if step.order == 1:
        return data
    be_comparison = dict(data)
    adapter = step.endpoint
    assert isinstance(adapter, (ElectronicStorage1D, IonicStorage1D))
    previous = step.history[0]
    older = step.history[1]
    assert isinstance(previous.problem, type(adapter)) and isinstance(older.problem, type(adapter))
    coefficient = step.coefficients
    rate = coefficient[0] * _full_increment(adapter, u, previous.problem, previous.u) / step.dt_s
    rate -= coefficient[2] * _full_increment(previous.problem, previous.u, older.problem, older.u) / step.dt_s
    new_inventory = adapter.physical_storage(u)
    previous_inventory = previous.problem.physical_storage(previous.u)
    older_inventory = older.problem.physical_storage(older.u)
    rounded_rate = (coefficient[0] * (new_inventory - previous_inventory)
                    - coefficient[2] * (previous_inventory - older_inventory)) / step.dt_s
    ionic = isinstance(adapter, IonicStorage1D)
    if isinstance(adapter, IonicStorage1D):
        device = adapter.problem.device.electronic_component
    else:
        device = adapter.problem.device
    count = device.node_count
    carrier_rate = rate[count:3 * count].reshape(2, count)
    if ionic:
        assert isinstance(adapter, IonicStorage1D)
        assert isinstance(be, IonicBackwardEuler1DStep)
        electronic_data = step_accounting(be.electronic_step, u[:3 * count])
        old_rate = np.asarray(data["carrier_inventory_rate_m2_s"])
        particles = np.array(data["carrier_particle_fluxes_m2_s"], copy=True)
    else:
        electronic_data = data
        old_rate = np.asarray(data["inventory_rate_m2_s"])
        particles = np.array(data["particle_fluxes_m2_s"], copy=True)
    for species, carrier in enumerate(("n", "p")):
        if device.channel(carrier, "left").kind == "fixed_reservoir":
            particles[species, 0] += carrier_rate[species, 0] - old_rate[species, 0]
        if device.channel(carrier, "right").kind == "fixed_reservoir":
            particles[species, -1] -= carrier_rate[species, -1] - old_rate[species, -1]
    pair_source = np.asarray(electronic_data["net_pair_source_m2_s"])
    carrier_balance = particles[:, :-1] - particles[:, 1:] + pair_source - carrier_rate
    conduction = Q * (particles[1] - particles[0])
    charge_rate = Q * (carrier_rate[1] - carrier_rate[0])
    charge_defect = Q * (carrier_balance[1] - carrier_balance[0])
    if ionic:
        assert isinstance(adapter, IonicStorage1D)
        ion_rate = rate[3 * count:]
        ion_flux = np.asarray(data["ion_domain_fluxes_m2_s"])
        ion_balance = -np.diff(np.r_[0.0, ion_flux, 0.0]) - ion_rate
        indices = adapter.problem.device.ion_node_indices
        charge_rate[indices] += Q * ion_rate
        charge_defect[indices] += Q * ion_balance
        conduction += np.asarray(data["ion_conduction_current_density_A_m2"])
        data.update(ion_inventory_rate_m2_s=ion_rate, ion_balance_m2_s=ion_balance,
                    global_ion_inventory_rate_m2_s=math.fsum(ion_rate.tolist()),
                    global_ion_balance_m2_s=math.fsum(ion_balance.tolist()),
                    carrier_inventory_rate_m2_s=carrier_rate, carrier_particle_fluxes_m2_s=particles,
                    rounded_ion_inventory_rate_m2_s=rounded_rate[3 * count:],
                    ion_inventory_readback_error_m2_s=ion_rate - rounded_rate[3 * count:],
                    rounded_carrier_inventory_rate_m2_s=rounded_rate[count:3 * count].reshape(2, count),
                    carrier_inventory_readback_error_m2_s=carrier_rate - rounded_rate[count:3 * count].reshape(2, count),
                    carrier_conduction_current_density_A_m2=Q * (particles[1] - particles[0]))
    else:
        data.update(inventory_rate_m2_s=carrier_rate, particle_fluxes_m2_s=particles,
                    rounded_inventory_subtraction_rate_m2_s=rounded_rate[count:].reshape(2, count),
                    inventory_subtraction_roundoff_m2_s=carrier_rate - rounded_rate[count:].reshape(2, count))
    def phi(point: HistoryPoint) -> Vector:
        a = point.problem
        assert isinstance(a, (ElectronicStorage1D, IonicStorage1D))
        return _potential(a, point.u)
    new_phi = _potential(adapter, u)
    phi_rate = (coefficient[0] * (new_phi - phi(previous))
                - coefficient[2] * (phi(previous) - phi(older))) / step.dt_s
    interior = -device.poisson.face_capacitance_F_m2 * np.diff(phi_rate)
    displacement_rate = np.r_[interior[0] - charge_rate[0], interior, interior[-1] + charge_rate[-1]]
    current = conduction + displacement_rate
    gauss_rate = displacement_rate[:-1] - displacement_rate[1:] + charge_rate
    new_d, new_charge = _displacement(adapter, u)
    previous_d, previous_charge = _displacement(previous.problem, previous.u)
    older_d, older_charge = _displacement(older.problem, older.u)
    rounded_displacement_rate = (coefficient[0] * (new_d - previous_d)
                                - coefficient[2] * (previous_d - older_d)) / step.dt_s
    rounded_charge_rate = (coefficient[0] * (new_charge - previous_charge)
                          - coefficient[2] * (previous_charge - older_charge)) / step.dt_s
    displacement_roundoff = displacement_rate - rounded_displacement_rate
    charge_roundoff = charge_rate - rounded_charge_rate
    rounded_gauss_rate = rounded_displacement_rate[:-1] - rounded_displacement_rate[1:] + rounded_charge_rate
    jump = current[:-1] - current[1:]
    residual = step.residual(u)
    mask = storage_mask(device)
    continuity = Q * float(np.sum(abs(residual[count:3 * count].reshape(2, count)[mask])))
    if ionic:
        continuity += Q * math.fsum(abs(residual[3 * count:]).tolist())
    data.update(bdf_order=2, bdf_coefficients=coefficient, bdf_residual=residual,
        backward_euler_comparison=be_comparison,
        bdf_physical_inventory_rate=rate, carrier_balance_m2_s=carrier_balance,
        bdf_rounded_inventory_rate=rounded_rate, displacement_C_m2=new_d, charge_inventory_C_m2=new_charge,
        charge_inventory_rate_A_m2=charge_rate, bdf_displacement_rate_A_m2=displacement_rate,
        charge_subtraction_roundoff_A_m2=charge_roundoff,
        bdf_rounded_displacement_rate_A_m2=rounded_displacement_rate,
        bdf_displacement_readback_error_A_m2=displacement_roundoff,
        actual_endpoint_displacement_increment_C_m2=new_d - previous_d,
        bdf_weighted_displacement_increment_C_m2=step.dt_s * displacement_rate,
        bdf_weighted_rounded_displacement_increment_C_m2=step.dt_s * rounded_displacement_rate,
        bdf_weighted_displacement_readback_error_C_m2=step.dt_s * displacement_roundoff,
        bdf_conduction_current_A_m2=conduction, bdf_total_current_A_m2=current,
        terminal_total_current_density_A_m2=current[[0, -1]],
        stable_gauss_defect_rate_A_m2=gauss_rate, gauss_defect_rate_A_m2=rounded_gauss_rate,
        maxwell_cell_jump_A_m2=jump,
        predicted_maxwell_cell_jump_A_m2=charge_defect + gauss_rate,
        maxwell_identity_error_A_m2=jump - charge_defect - gauss_rate,
        readback_predicted_maxwell_cell_jump_A_m2=charge_defect + rounded_gauss_rate + charge_roundoff
            + displacement_roundoff[:-1] - displacement_roundoff[1:],
        readback_maxwell_identity_error_A_m2=jump - charge_defect - rounded_gauss_rate - charge_roundoff
            - displacement_roundoff[:-1] + displacement_roundoff[1:],
        maxwell_spread_A_m2=float(np.ptp(current)), continuity_defect_bound_A_m2=continuity,
        global_particle_balance_m2_s=np.sum(carrier_rate, axis=1) - (particles[:, 0] - particles[:, -1] + np.sum(pair_source)),
        global_charge_balance_A_m2=float(np.sum(charge_rate) - (conduction[0] - conduction[-1])))
    if ionic:
        data.update(conduction_current_density_A_m2=conduction, displacement_current_density_A_m2=displacement_rate,
                    total_current_density_A_m2=current,
                    interval_displacement_C_m2=be_comparison["interval_displacement_C_m2"],
                    rounded_interval_displacement_C_m2=be_comparison["rounded_interval_displacement_C_m2"],
                    displacement_readback_error_C_m2=be_comparison["displacement_readback_error_C_m2"])
    else:
        data.update(conduction_current_density_A_m2=conduction[1:-1],
                    displacement_current_density_A_m2=displacement_rate[1:-1],
                    total_current_density_A_m2=current[1:-1],
                    terminal_conduction_current_density_A_m2=conduction[[0, -1]],
                    terminal_displacement_current_density_A_m2=displacement_rate[[0, -1]])
    return data
