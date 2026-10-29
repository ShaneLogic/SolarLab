"""Native physical backward Euler and Maxwell accounting for both ion domains."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.equations.multiple_ionic_1d import MultipleIonic1DProblem, MultipleIonicState1D
from solarlab.physics.electrostatics import contact_displacement
from solarlab.solvers.time.backward_euler import BackwardEuler1DStep, storage_mask, step_accounting as electronic_accounting
from solarlab.solvers.time.ionic_backward_euler import IonicBackwardEuler1DStep

__all__ = ["MultipleIonicBackwardEuler1DStep", "multiple_ionic_step_accounting"]


@dataclass(frozen=True)
class MultipleIonicBackwardEuler1DStep:
    """Full R=F_DC-delta(S)/dt and J=F_y+K/dt; no steady row replacement."""

    endpoint: MultipleIonic1DProblem
    previous: MultipleIonicState1D
    dt_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, MultipleIonic1DProblem):
            raise TypeError("the original multiple-domain endpoint equations are required")
        if isinstance(self.dt_s, bool) or not isinstance(self.dt_s, Real) or not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("dt_s must be a finite positive real scalar")
        self.endpoint.device.validate_state(self.previous)

    @property
    def electronic_step(self) -> BackwardEuler1DStep:
        return BackwardEuler1DStep(self.endpoint.electronic_problem, self.previous.electronic_state, self.dt_s)

    def inventory_rates_m2_s(self, y: np.ndarray) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
        device, size = self.endpoint.device, 3 * self.endpoint.device.node_count
        values = device._values(y)
        device.fields_by_domain(values, voltage_V=self.endpoint.voltage_V)
        carrier = self.electronic_step.inventory_rate_m2_s(values[:size])
        rates: dict[str, dict[str, np.ndarray]] = {}
        for domain, previous in zip(device.domains, self.previous.domain_component_states):
            local = IonicBackwardEuler1DStep(Ionic1DProblem(domain, self.endpoint.voltage_V,
                                                          self.endpoint.generation_scale), previous, self.dt_s)
            _, ionic = local.inventory_rates_m2_s(device.local_coordinates(values, domain.domain_layer_id))
            rates[domain.domain_layer_id] = dict(zip(domain.ion_species_names, ionic))
        return carrier, rates

    def residual(self, y: np.ndarray) -> np.ndarray:
        device, count = self.endpoint.device, self.endpoint.device.node_count
        result = self.endpoint.residual(y)
        carrier, ionic = self.inventory_rates_m2_s(y)
        rows = result[count:3 * count].reshape(2, count)
        mask = storage_mask(device.electronic_component)
        rows[mask] -= carrier[mask]
        for name, species_rates in ionic.items():
            for species, rate in species_rates.items():
                result[device.ion_species_slices[name][species]] -= rate
        return result

    def jacobian(self, y: np.ndarray) -> csc_matrix:
        return self.endpoint.jacobian(y) + self.endpoint.storage_jacobian(y) / self.dt_s

    def initial_coordinates(self) -> np.ndarray:
        """Existing harmonic voltage lift; all ionic unknowns stay in the solve."""
        electronic = self.electronic_step.initial_coordinates("previous", voltage_lift=True)
        return np.r_[electronic, self.previous.u[3 * self.endpoint.device.node_count:]]


def multiple_ionic_step_accounting(step: MultipleIonicBackwardEuler1DStep, y: np.ndarray) -> dict[str, Any]:
    """All internal/external +x currents and separate conservative inventories."""
    problem, device, old = step.endpoint, step.endpoint.device, step.previous
    count, dt = device.node_count, step.dt_s
    new = device.state(y, voltage_V=problem.voltage_V)
    carrier_info = electronic_accounting(step.electronic_step, np.asarray(y)[:3 * count])
    carrier_rate, ionic_rates = step.inventory_rates_m2_s(y)
    carrier_balance = np.asarray(carrier_info["carrier_balance_m2_s"])
    particles = np.asarray(carrier_info["particle_fluxes_m2_s"])
    carrier_conduction = Q * (particles[1] - particles[0])
    ionic_conduction, ion_charge_rate = np.zeros(count + 1), np.zeros(count)
    ion_charge_balance = np.zeros(count)
    domains: dict[str, Any] = {}
    residual = step.residual(y)
    for domain, before, after in zip(device.domains, old.domain_component_states, new.domain_component_states):
        name, indices, weights = domain.domain_layer_id, domain.ion_node_indices, domain.ion_geometry.cell_measures_m
        species_records: dict[str, Any] = {}
        for species in domain.ion_species_names:
            sign = 1.0 if species == "positive" else -1.0
            rate = ionic_rates[name][species]
            flux = problem.ion_fluxes(y, domain_id=name, species=species)
            global_flux = np.r_[0.0, problem.ion_fluxes_on_electronic_faces(y, domain_id=name, species=species), 0.0]
            balance = -np.diff(np.r_[0.0, flux, 0.0]) - rate
            ion_charge_rate[indices] += sign * Q * rate
            ion_charge_balance[indices] += sign * balance
            ionic_conduction += sign * Q * global_flux
            old_inventory = weights * getattr(before, species + "_ion_density_m3")
            new_inventory = weights * getattr(after, species + "_ion_density_m3")
            rounded = (new_inventory - old_inventory) / dt
            rounding_bound = 8 * np.finfo(float).eps * (
                math.fsum(abs(flux).tolist()) + math.fsum(abs(rate).tolist()))
            species_records[species] = {
                "inventory_m2": new_inventory, "inventory_rate_m2_s": rate,
                "physical_inventory_increment_m2": dt * rate,
                "rounded_inventory_rate_m2_s": rounded, "inventory_readback_error_m2_s": rate - rounded,
                "balance_m2_s": balance, "balance_l1_m2_s": math.fsum(abs(balance).tolist()),
                "native_BE_rows_m2_s": residual[device.ion_species_slices[name][species]],
                "domain_flux_m2_s": np.r_[0.0, flux, 0.0], "global_particle_flux_m2_s": global_flux,
                "signed_conduction_current_A_m2": sign * Q * global_flux,
                "inventory_rate_sum_m2_s": math.fsum(rate.tolist()),
                "balance_sum_m2_s": math.fsum(balance.tolist()),
                "inventory_balance_identity_error_m2_s": math.fsum(balance.tolist()) + math.fsum(rate.tolist()),
                "inventory_balance_roundoff_bound_m2_s": rounding_bound,
            }
        domains[name] = {"species": species_records, "electrical_node_indices": indices, "physical_weights_m": weights}
    charge_rate = Q * (carrier_rate[1] - carrier_rate[0]) + ion_charge_rate
    conduction = carrier_conduction + ionic_conduction
    poisson, widths = device.electronic_component.poisson, device.electronic_component.geometry.cell_measures_m

    def readback(state: MultipleIonicState1D) -> tuple[np.ndarray, np.ndarray]:
        e = state.electronic_state
        populations = device.fields_by_domain(state.u, voltage_V=state.voltage_V)[3]
        rho = device.charge_ledger(e.electron_density_m3, e.hole_density_m3, populations).space_charge_C_m3()
        internal = poisson.face_displacement(e.potential_V)
        ports = contact_displacement(internal, rho, widths)
        return np.r_[ports[0], internal, ports[1]], widths * rho

    old_d, old_charge = readback(old)
    new_d, new_charge = readback(new)
    phi_delta = new.electronic_state.potential_V - old.electronic_state.potential_V
    internal_delta = -poisson.face_capacitance_F_m2 * np.diff(phi_delta)
    interval_d = np.r_[internal_delta[0] - dt * charge_rate[0], internal_delta,
                       internal_delta[-1] + dt * charge_rate[-1]]
    rounded_d = new_d - old_d
    d_error = interval_d - rounded_d
    displacement_current = interval_d / dt
    maxwell = conduction + displacement_current
    old_gauss = old_d[:-1] - old_d[1:] + old_charge
    new_gauss = new_d[:-1] - new_d[1:] + new_charge
    stable_gauss_rate = (interval_d[:-1] - interval_d[1:]) / dt + charge_rate
    rounded_gauss_rate = (new_gauss - old_gauss) / dt
    charge_readback_error = charge_rate - (new_charge - old_charge) / dt
    charge_defect = Q * (carrier_balance[1] - carrier_balance[0] + ion_charge_balance)
    jump = maxwell[:-1] - maxwell[1:]
    predicted = charge_defect + stable_gauss_rate
    rounded_predicted = charge_defect + rounded_gauss_rate + charge_readback_error + (d_error[:-1] - d_error[1:]) / dt
    mask = storage_mask(device.electronic_component)
    return {
        "domains": domains, "backward_euler_residual": residual, "dc_residual": problem.residual(y),
        "carrier_inventory_rate_m2_s": carrier_rate, "carrier_balance_m2_s": carrier_balance,
        "carrier_particle_fluxes_m2_s": particles, "carrier_conduction_current_A_m2": carrier_conduction,
        "ionic_conduction_current_A_m2": ionic_conduction, "conduction_current_density_A_m2": conduction,
        "interval_displacement_C_m2": interval_d, "rounded_interval_displacement_C_m2": rounded_d,
        "displacement_readback_error_C_m2": d_error, "displacement_current_density_A_m2": displacement_current,
        "total_current_density_A_m2": maxwell, "terminal_total_current_density_A_m2": maxwell[[0, -1]],
        "charge_inventory_rate_A_m2": charge_rate, "charge_inventory_C_m2": new_charge,
        "charge_readback_error_A_m2": charge_readback_error,
        "old_gauss_defect_C_m2": old_gauss, "gauss_defect_C_m2": new_gauss,
        "stable_gauss_defect_rate_A_m2": stable_gauss_rate, "rounded_gauss_defect_rate_A_m2": rounded_gauss_rate,
        "maxwell_cell_jump_A_m2": jump, "predicted_maxwell_cell_jump_A_m2": predicted,
        "maxwell_identity_error_A_m2": jump - predicted, "readback_maxwell_identity_error_A_m2": jump - rounded_predicted,
        "maxwell_spread_A_m2": float(np.ptp(maxwell)),
        "global_charge_balance_A_m2": float(np.sum(charge_rate) - (conduction[0] - conduction[-1])),
        "continuity_defect_bound_A_m2": Q * (float(np.sum(abs(residual[count:3 * count].reshape(2, count)[mask])))
            + math.fsum(abs(residual[3 * count:]).tolist())),
        "storage_mask": mask, "previous_voltage_V": old.voltage_V, "voltage_V": problem.voltage_V,
        "dt_s": dt, "compiled_identity": device.identity, "physical_certificate_status": "not_evaluated",
    }
