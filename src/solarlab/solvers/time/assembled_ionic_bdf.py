"""Physical BDF on the canonical assembled single/multiple ionic BE storage.

Every species secant and tangent is supplied by AssembledIonicStorage1D.
D0 species retain their full physical mass, as in the original BE time rows;
only genuinely algebraic/constraint rows have zero equation storage. The
original BE observations supply fluxes, sources, charge and domain geometry.
No experiment selection or production default is changed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, TypeAlias, cast

import numpy as np

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem, AssembledMultipleIonic1DProblem
from solarlab.equations.dc_1d import DC1DProblem, State1D
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, IonicState1D
from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D, MultipleIonicState1D
from solarlab.physics.electrostatics import contact_displacement
from solarlab.solvers.nonlinear.problem import Vector
from solarlab.solvers.time.assembled_ionic_backward_euler import (
    AssembledIonicBackwardEuler1DStep, AssembledMultipleIonicBackwardEuler1DStep,
)
from solarlab.solvers.time.assembled_storage import AssembledIonicStorage1D
from solarlab.solvers.time.backward_euler import BackwardEuler1DStep
from solarlab.solvers.time.bdf import BDFStep
from solarlab.solvers.time.ionic_backward_euler import IonicBackwardEuler1DStep, ionic_step_accounting
from solarlab.solvers.time.multiple_ionic_backward_euler import MultipleIonicBackwardEuler1DStep, multiple_ionic_step_accounting

__all__ = ["assembled_ionic_bdf_accounting", "assembled_ionic_bdf_seed"]

IonicDevice: TypeAlias = CompiledIonicDevice1D | CompiledMultipleIonicDevice1D
IonicState: TypeAlias = IonicState1D | MultipleIonicState1D
IonicBEStep: TypeAlias = AssembledIonicBackwardEuler1DStep | AssembledMultipleIonicBackwardEuler1DStep


def _adapter(step: BDFStep) -> AssembledIonicStorage1D:
    if not isinstance(step.endpoint, AssembledIonicStorage1D):
        raise TypeError("canonical assembled ionic storage is required")
    return step.endpoint


def _be(step: BDFStep) -> IonicBEStep:
    endpoint = _adapter(step)
    old = step.history[0]
    if not isinstance(old.problem, AssembledIonicStorage1D):
        raise TypeError("ionic history must retain the canonical assembled storage")
    previous = old.problem.device.state(old.u, voltage_V=old.problem.voltage_V)
    if isinstance(endpoint.problem, AssembledIonic1DProblem):
        return AssembledIonicBackwardEuler1DStep(endpoint.problem, cast(IonicState1D, previous), step.dt_s)
    return AssembledMultipleIonicBackwardEuler1DStep(endpoint.problem, cast(MultipleIonicState1D, previous), step.dt_s)


def assembled_ionic_bdf_seed(step: BDFStep) -> Vector:
    """Reuse the original harmonic carrier lift and unchanged ion coordinates."""
    be = _be(step)
    return (be.initial_coordinates("previous", voltage_lift=True)
            if isinstance(be, AssembledIonicBackwardEuler1DStep) else be.initial_coordinates())


def _physical_rate(step: BDFStep, u: Vector) -> Vector:
    adapter = _adapter(step)
    old = step.history[0]
    rate = step.coefficients[0] * adapter.physical_increment(u, old.problem, old.u) / step.dt_s
    if step.order == 2:
        older = step.history[1]
        if not isinstance(old.problem, AssembledIonicStorage1D):
            raise TypeError("ionic history must retain its storage binding")
        rate -= step.coefficients[2] * old.problem.physical_increment(old.u, older.problem, older.u) / step.dt_s
    return rate


@dataclass(frozen=True)
class _CarrierRateStep(BackwardEuler1DStep):
    """Readout-only view with the complete BDF carrier inventory rate."""

    physical_rate: np.ndarray

    def inventory_rate_m2_s(self, u: np.ndarray) -> np.ndarray:
        return self.physical_rate


@dataclass(frozen=True)
class _AccountingStep:
    """Supply canonical BDF rates to the unchanged BE flux/charge readers."""

    step: BDFStep
    be: IonicBEStep
    rate: Vector

    @property
    def endpoint(self) -> AssembledIonic1DProblem | AssembledMultipleIonic1DProblem:
        return self.be.endpoint

    @property
    def previous(self) -> IonicState:
        return self.be.previous

    @property
    def dt_s(self) -> float:
        return self.step.dt_s

    @property
    def electronic_step(self) -> _CarrierRateStep:
        original = self.be.electronic_step
        n = self.endpoint.device.node_count
        return _CarrierRateStep(cast(DC1DProblem, original.endpoint), cast(State1D, original.previous),
                                self.dt_s, self.rate[n:3 * n].reshape(2, n))

    def inventory_rates_m2_s(self, u: Vector) -> Any:
        device = self.endpoint.device
        n = device.node_count
        if isinstance(device, CompiledIonicDevice1D):
            ions: Any = tuple(self.rate[block] for block in device.ion_species_slices)
        else:
            ions = {name: {species: self.rate[block] for species, block in blocks.items()}
                    for name, blocks in device.ion_species_slices.items()}
        return self.rate[n:3 * n].reshape(2, n), ions

    def residual(self, u: Vector) -> Vector:
        return self.step.residual(u)


def _domains(device: IonicDevice) -> tuple[tuple[CompiledIonicDevice1D, dict[str, slice]], ...]:
    if isinstance(device, CompiledIonicDevice1D):
        return ((device, dict(zip(device.ion_species_names, device.ion_species_slices))),)
    return tuple((domain, dict(device.ion_species_slices[domain.domain_layer_id])) for domain in device.domains)


def _electrostatic(adapter: AssembledIonicStorage1D, u: Vector) -> tuple[Vector, Vector, Vector]:
    device = adapter.device
    state = device.state(u, voltage_V=adapter.voltage_V)
    e = state.electronic_state
    if isinstance(device, CompiledIonicDevice1D):
        assert isinstance(state, IonicState1D)
        ledger = device.charge_ledger(e.electron_density_m3, e.hole_density_m3,
            state.positive_ion_density_m3,
            state.negative_ion_density_m3 if device.ion_species_count == 2 else None)
    else:
        populations = device.fields_by_domain(u, voltage_V=adapter.voltage_V)[3]
        ledger = device.charge_ledger(e.electron_density_m3, e.hole_density_m3, populations)
    geometry, poisson = device.electronic_component.geometry, device.electronic_component.poisson
    rho = ledger.space_charge_C_m3()
    interior = poisson.face_displacement(e.potential_V)
    ports = contact_displacement(interior, rho, geometry.cell_measures_m)
    return e.potential_V, np.r_[ports[0], interior, ports[1]], rho * geometry.cell_measures_m


def _domain_records(device: IonicDevice, data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(device, CompiledMultipleIonicDevice1D):
        return cast(dict[str, Any], data["domains"])
    names = data.get("ion_species", {"positive": data})
    records = {}
    for species in device.ion_species_names:
        values = names[species]
        records[species] = {
            "inventory_m2": values["ion_inventory_m2"],
            "inventory_rate_m2_s": values["ion_inventory_rate_m2_s"],
            "balance_m2_s": values["ion_balance_m2_s"],
            "balance_l1_m2_s": math.fsum(abs(np.asarray(values["ion_balance_m2_s"])).tolist()),
            "domain_flux_m2_s": np.r_[0.0, values["ion_domain_fluxes_m2_s"], 0.0],
        }
    return {device.domain_layer_id: {"species": records,
            "electrical_node_indices": device.ion_node_indices,
            "physical_weights_m": device.ion_geometry.cell_measures_m}}


def assembled_ionic_bdf_accounting(step: BDFStep, u: Vector) -> dict[str, Any]:
    """Original BE readouts at order one; full coefficient-consistent BDF2."""
    adapter, be = _adapter(step), _be(step)
    device = adapter.device
    original = dict(be.accounting(u))
    if step.order == 1:
        original["domains"] = _domain_records(device, original)
        return original
    rate = _physical_rate(step, u)
    view = _AccountingStep(step, be, rate)
    data = dict(ionic_step_accounting(cast(IonicBackwardEuler1DStep, view), u)
                if isinstance(be, AssembledIonicBackwardEuler1DStep)
                else multiple_ionic_step_accounting(cast(MultipleIonicBackwardEuler1DStep, view), u))
    domains = _domain_records(device, data)
    old, older = step.history[:2]
    assert isinstance(old.problem, AssembledIonicStorage1D)
    assert isinstance(older.problem, AssembledIonicStorage1D)
    coefficients, h = step.coefficients, step.dt_s
    new_s, old_s, older_s = adapter.physical_storage(u), old.problem.physical_storage(old.u), older.problem.physical_storage(older.u)
    rounded_rate = (coefficients[0] * (new_s - old_s) - coefficients[2] * (old_s - older_s)) / h
    actual_increment = adapter.physical_increment(u, old.problem, old.u)
    ion_balance = np.zeros(device.node_count)
    for domain, blocks in _domains(device):
        for species, block in blocks.items():
            record = domains[domain.domain_layer_id]["species"][species]
            sign = 1.0 if species == "positive" else -1.0
            ion_balance[domain.ion_node_indices] += sign * np.asarray(record["balance_m2_s"])
            record.update(inventory_rate_m2_s=rate[block], rounded_inventory_rate_m2_s=rounded_rate[block],
                inventory_readback_error_m2_s=rate[block] - rounded_rate[block],
                physical_inventory_increment_m2=actual_increment[block],
                bdf_weighted_inventory_increment_m2=h * rate[block],
                native_BDF_rows_m2_s=step.residual(u)[block])
            if isinstance(device, CompiledMultipleIonicDevice1D):
                record["native_BE_rows_m2_s"] = original["domains"][domain.domain_layer_id]["species"][species]["native_BE_rows_m2_s"]
            else:
                named = data.get("ion_species", {"positive": data})[species]
                named["rounded_ion_inventory_rate_m2_s"] = rounded_rate[block]
                named["ion_inventory_readback_error_m2_s"] = rate[block] - rounded_rate[block]
    phi, displacement, charge = _electrostatic(adapter, u)
    old_phi, old_d, old_charge = _electrostatic(old.problem, old.u)
    older_phi, older_d, older_charge = _electrostatic(older.problem, older.u)
    phi_rate = (coefficients[0] * (phi - old_phi) - coefficients[2] * (old_phi - older_phi)) / h
    interior = -device.electronic_component.poisson.face_capacitance_F_m2 * np.diff(phi_rate)
    charge_rate = np.asarray(data["charge_inventory_rate_A_m2"])
    displacement_rate = np.r_[interior[0] - charge_rate[0], interior, interior[-1] + charge_rate[-1]]
    rounded_d_rate = (coefficients[0] * (displacement - old_d) - coefficients[2] * (old_d - older_d)) / h
    rounded_charge_rate = (coefficients[0] * (charge - old_charge) - coefficients[2] * (old_charge - older_charge)) / h
    d_error, q_error = displacement_rate - rounded_d_rate, charge_rate - rounded_charge_rate
    gauss = displacement_rate[:-1] - displacement_rate[1:] + charge_rate
    rounded_gauss = rounded_d_rate[:-1] - rounded_d_rate[1:] + rounded_charge_rate
    carrier_balance = np.asarray(data["carrier_balance_m2_s"])
    defect = Q * (carrier_balance[1] - carrier_balance[0] + ion_balance)
    current = np.asarray(data["conduction_current_density_A_m2"]) + displacement_rate
    jump = current[:-1] - current[1:]
    data.update(domains=domains, bdf_order=2, bdf_coefficients=coefficients, bdf_residual=step.residual(u),
        backward_euler_comparison=original, backward_euler_residual=original["backward_euler_residual"],
        bdf_physical_inventory_rate_m2_s=rate, bdf_rounded_inventory_rate_m2_s=rounded_rate,
        carrier_inventory_rate_m2_s=rate[device.node_count:3 * device.node_count].reshape(2, device.node_count),
        rounded_carrier_inventory_rate_m2_s=rounded_rate[device.node_count:3 * device.node_count].reshape(2, device.node_count),
        carrier_inventory_readback_error_m2_s=(rate - rounded_rate)[device.node_count:3 * device.node_count].reshape(2, device.node_count),
        bdf_displacement_rate_A_m2=displacement_rate, bdf_rounded_displacement_rate_A_m2=rounded_d_rate,
        bdf_displacement_readback_error_A_m2=d_error,
        bdf_weighted_displacement_increment_C_m2=h * displacement_rate,
        bdf_actual_physical_inventory_increment_m2=actual_increment,
        displacement_C_m2=displacement, charge_inventory_C_m2=charge,
        charge_inventory_rate_A_m2=charge_rate, charge_readback_error_A_m2=q_error,
        charge_subtraction_roundoff_A_m2=q_error,
        interval_displacement_C_m2=original["interval_displacement_C_m2"],
        rounded_interval_displacement_C_m2=original["rounded_interval_displacement_C_m2"],
        displacement_readback_error_C_m2=original["displacement_readback_error_C_m2"],
        displacement_current_density_A_m2=displacement_rate, total_current_density_A_m2=current,
        terminal_total_current_density_A_m2=current[[0, -1]],
        stable_gauss_defect_rate_A_m2=gauss, rounded_gauss_defect_rate_A_m2=rounded_gauss,
        gauss_defect_rate_A_m2=rounded_gauss,
        maxwell_cell_jump_A_m2=jump, predicted_maxwell_cell_jump_A_m2=defect + gauss,
        readback_predicted_maxwell_cell_jump_A_m2=defect + rounded_gauss + q_error + d_error[:-1] - d_error[1:],
        maxwell_identity_error_A_m2=jump - defect - gauss,
        readback_maxwell_identity_error_A_m2=jump - defect - rounded_gauss - q_error - d_error[:-1] + d_error[1:],
        maxwell_spread_A_m2=float(np.ptp(current)))
    if isinstance(device, CompiledIonicDevice1D):
        positive = device.ion_species_slices[0]
        data["rounded_ion_inventory_rate_m2_s"] = rounded_rate[positive]
        data["ion_inventory_readback_error_m2_s"] = rate[positive] - rounded_rate[positive]
    return data
