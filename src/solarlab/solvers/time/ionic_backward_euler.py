"""Physical backward Euler for the integrated electronic/confined-ion model.

R=F_DC-(S_new-S_old)/dt and J=F_y+K/dt. All M balances per species remain;
only fixed-reservoir carrier rows are algebraic. Diagnostics carry actual
motion currents and explicit floating readback errors, not acceptance labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import csc_matrix
from scipy.special import expit

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, Ionic1DProblem, IonicState1D
from solarlab.physics.electrostatics import contact_displacement
from solarlab.physics.ions.shared_site import shared_site_secant_m3
from solarlab.solvers.time.backward_euler import (
    BackwardEuler1DStep, _storage_mask, inventory_m2 as electronic_inventory_m2,
    step_accounting as electronic_step_accounting,
)

__all__ = ["IonicBackwardEuler1DStep", "finite_site_secant_m3", "inventory_m2", "step_accounting", "ionic_step_accounting"]


def finite_site_secant_m3(
    capacity_m3: np.ndarray, logit_reference: np.ndarray,
    old_coordinate: np.ndarray, new_coordinate: np.ndarray,
) -> np.ndarray:
    """Stable physical logistic difference, including arbitrary nonzero old logits.

    The represented coordinate difference is formed before adding the common
    reference. expit(-logit) retains the vacancy fraction near capacity, and
    expm1 retains tiny motion whose rounded density subtraction can be zero.
    """
    arrays = [np.asarray(value, dtype=float) for value in (
        capacity_m3, logit_reference, old_coordinate, new_coordinate,
    )]
    capacity, reference, old, new = arrays
    if (capacity.ndim != 1 or any(value.shape != capacity.shape or not np.all(np.isfinite(value)) for value in arrays)
            or np.any(capacity <= 0.0)):
        raise ValueError("finite-site secant requires finite matched vectors and positive capacities")
    old_logit, new_logit = reference + old, reference + new
    old_theta, new_theta = expit(old_logit), expit(new_logit)
    if np.any(old_theta <= 0.0) or np.any(old_theta >= 1.0) or np.any(new_theta <= 0.0) or np.any(new_theta >= 1.0):
        raise ValueError("finite-site secant coordinate saturated")
    delta = new - old
    positive = capacity * new_theta * expit(-old_logit) * (-np.expm1(-np.maximum(delta, 0.0)))
    negative = capacity * old_theta * expit(-new_logit) * np.expm1(np.minimum(delta, 0.0))
    result = np.where(delta >= 0.0, positive, negative)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("finite-site physical secant is nonfinite")
    return result


def inventory_m2(
    device: CompiledIonicDevice1D, state: IonicState1D, *, species: str = "positive",
) -> tuple[np.ndarray, np.ndarray]:
    """Physical carrier (2,N) and named ion (M,) inventories, including endpoints."""
    if not isinstance(state, IonicState1D) or state.device_sha256 != device.identity:
        raise ValueError("inventory requires a full same-device ionic state")
    if species not in device.ion_species_names:
        raise ValueError("requested ion species is not assembled in this device")
    carrier = electronic_inventory_m2(device.electronic_component, state.electronic_state)
    return carrier, device.ion_geometry.cell_measures_m * getattr(state, f"{species}_ion_density_m3")


@dataclass(frozen=True)
class IonicBackwardEuler1DStep:
    """One full nonlinear physical step; no steady inventory-row substitution."""

    endpoint: Ionic1DProblem
    previous: IonicState1D
    dt_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, Ionic1DProblem) or not isinstance(self.previous, IonicState1D):
            raise ValueError("a full ionic endpoint and previous ionic state are required")
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        device = self.endpoint.device
        if self.previous.device_sha256 != device.identity:
            raise ValueError("previous state must belong to the same complete ionic device")
        actual = device.state(self.previous.u, voltage_V=self.previous.voltage_V)
        if self.previous.electronic_state.device_sha256 != device.electronic_component.identity:
            raise ValueError("previous electronic component identity is inconsistent")
        for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
            if not np.array_equal(getattr(actual.electronic_state, name), getattr(self.previous.electronic_state, name)):
                raise ValueError("previous electronic fields disagree with their voltage reference")
        if (any(not np.array_equal(getattr(actual, f"{species}_ion_density_m3"),
                                   getattr(self.previous, f"{species}_ion_density_m3"))
                for species in device.ion_species_names)
                or actual.domain_inventories_m2 != self.previous.domain_inventories_m2):
            raise ValueError("previous physical ion fields/inventory disagree with the bound coordinates")

    @property
    def electronic_step(self) -> BackwardEuler1DStep:
        return BackwardEuler1DStep(self.endpoint.electronic_problem, self.previous.electronic_state, self.dt_s)

    def inventory_rates_m2_s(self, y: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Carrier rates and physical site-law secants for every species, in block order."""
        device, N = self.endpoint.device, self.endpoint.device.node_count
        values = np.asarray(y, dtype=float)
        device.fields_by_species(values, voltage_V=self.endpoint.voltage_V)
        carrier = self.electronic_step.inventory_rate_m2_s(values[:3 * N])
        if device.ion_site_contract == "shared_finite_sites":
            positive, negative = device.ion_species_slices
            changes = shared_site_secant_m3(
                device.positive_ion_site_limit_m3,
                device.positive_ion_logit_reference, device.negative_ion_logit_reference,
                self.previous.u[positive], self.previous.u[negative], values[positive], values[negative],
            )
            return carrier, tuple(device.ion_geometry.cell_measures_m * change / self.dt_s for change in changes)
        rates = []
        for species, block in zip(device.ion_species_names, device.ion_species_slices):
            ion_change = finite_site_secant_m3(
                getattr(device, f"{species}_ion_site_limit_m3"), getattr(device, f"{species}_ion_logit_reference"),
                self.previous.u[block], values[block],
            )
            rates.append(device.ion_geometry.cell_measures_m * ion_change / self.dt_s)
        return carrier, tuple(rates)

    def inventory_rate_m2_s(self, y: np.ndarray, *, species: str = "positive") -> tuple[np.ndarray, np.ndarray]:
        """Historical positive-rate reader, with an explicit negative-species selector."""
        if species not in self.endpoint.device.ion_species_names:
            raise ValueError("requested ion species is not assembled in this device")
        carrier, rates = self.inventory_rates_m2_s(y)
        return carrier, rates[self.endpoint.device.ion_species_names.index(species)]

    def residual(self, y: np.ndarray) -> np.ndarray:
        result = self.endpoint.residual(y)
        N = self.endpoint.device.node_count
        carrier, ions = self.inventory_rates_m2_s(y)
        rows = result[N:3 * N].reshape(2, N)
        mask = _storage_mask(self.endpoint.device.electronic_component)
        rows[mask] -= carrier[mask]
        for block, ion in zip(self.endpoint.device.ion_species_slices, ions):
            result[block] -= ion
        return result

    def jacobian(self, y: np.ndarray) -> csc_matrix:
        return self.endpoint.jacobian(y) + self.endpoint.storage_jacobian(y) / self.dt_s

    def initial_coordinates(self, predictor: str = "previous", *, voltage_lift: bool = False) -> np.ndarray:
        """Rebase carrier references; ion seed is unchanged, but every ion row is solved."""
        electronic = self.electronic_step.initial_coordinates(predictor, voltage_lift=voltage_lift)
        return np.r_[electronic, self.previous.u[3 * self.endpoint.device.node_count:]]


def step_accounting(step: IonicBackwardEuler1DStep, y: np.ndarray) -> dict[str, object]:
    """All-face conduction/displacement and independent interval balance diagnostics.

    All currents point along +x in A/m^2. Ion number rates are domain-local
    particles/(m^2*s); their charge is scattered without a second volume factor.
    Stable interval displacement and separately rounded state readbacks are
    both exposed, including their effect on the complete Maxwell identity.
    Historical unnamed ion arrays refer to the positive species; dual evidence
    also records both named species. Ionic conduction/charge arrays sum signs.
    """
    problem, device, old = step.endpoint, step.endpoint.device, step.previous
    N, dt, idx = device.node_count, step.dt_s, device.ion_node_indices
    new = device.state(y, voltage_V=problem.voltage_V)
    carrier_info = electronic_step_accounting(step.electronic_step, np.asarray(y)[:3 * N])
    carrier_rate, ion_rates = step.inventory_rates_m2_s(y)
    ion_rate = ion_rates[0]
    carrier_balance = np.asarray(carrier_info["carrier_balance_m2_s"])
    carrier_particles = np.asarray(carrier_info["particle_fluxes_m2_s"])
    carrier_conduction = Q * (carrier_particles[1] - carrier_particles[0])
    ion_domain_flux = problem.ion_fluxes(y)
    ion_particles = np.r_[0.0, problem.ion_fluxes_on_electronic_faces(y), 0.0]
    ion_balance = -np.diff(np.r_[0.0, ion_domain_flux, 0.0]) - ion_rate
    ion_charge_rate = np.zeros(N)
    ion_charge_rate[idx] = Q * ion_rate
    signed_ion_particles = ion_particles
    if device.ion_species_count == 2:
        negative_rate = ion_rates[1]
        negative_flux = problem.ion_fluxes(y, species="negative")
        negative_particles = np.r_[0.0, problem.ion_fluxes_on_electronic_faces(y, species="negative"), 0.0]
        negative_balance = -np.diff(np.r_[0.0, negative_flux, 0.0]) - negative_rate
        signed_ion_particles = ion_particles - negative_particles
        ion_charge_rate[idx] = Q * (ion_rate - negative_rate)
    charge_rate = Q * (carrier_rate[1] - carrier_rate[0]) + ion_charge_rate
    conduction = carrier_conduction + Q * signed_ion_particles
    widths = device.electronic_component.geometry.cell_measures_m
    poisson = device.electronic_component.poisson

    def readback(state: IonicState1D) -> tuple[np.ndarray, np.ndarray]:
        e = state.electronic_state
        rho = device.charge_ledger(e.electron_density_m3, e.hole_density_m3,
                                  state.positive_ion_density_m3,
                                  state.negative_ion_density_m3 if device.ion_species_count == 2 else None).space_charge_C_m3()
        interior = poisson.face_displacement(e.potential_V)
        ports = contact_displacement(interior, rho, widths)
        return np.r_[ports[0], interior, ports[1]], widths * rho

    old_d, old_charge = readback(old)
    new_d, new_charge = readback(new)
    phi_delta = new.electronic_state.potential_V - old.electronic_state.potential_V
    interior_delta = -poisson.face_capacitance_F_m2 * np.diff(phi_delta)
    interval_d = np.r_[interior_delta[0] - dt * charge_rate[0], interior_delta,
                       interior_delta[-1] + dt * charge_rate[-1]]
    rounded_interval_d = new_d - old_d
    displacement_readback_error = interval_d - rounded_interval_d
    displacement_current = interval_d / dt
    maxwell = conduction + displacement_current
    old_gauss = old_d[:-1] - old_d[1:] + old_charge
    new_gauss = new_d[:-1] - new_d[1:] + new_charge
    rounded_gauss_rate = (new_gauss - old_gauss) / dt
    stable_gauss_rate = (interval_d[:-1] - interval_d[1:]) / dt + charge_rate
    charge_readback_error = charge_rate - (new_charge - old_charge) / dt
    ion_balance_global = np.zeros(N)
    ion_balance_global[idx] = ion_balance
    if device.ion_species_count == 2:
        ion_balance_global[idx] = ion_balance - negative_balance
    charge_defect = Q * (carrier_balance[1] - carrier_balance[0] + ion_balance_global)
    jump = maxwell[:-1] - maxwell[1:]
    predicted = charge_defect + stable_gauss_rate
    rounded_predicted = (charge_defect + rounded_gauss_rate + charge_readback_error
                         + (displacement_readback_error[:-1] - displacement_readback_error[1:]) / dt)
    old_carrier_inventory, old_ion_inventory = inventory_m2(device, old)
    new_carrier_inventory, new_ion_inventory = inventory_m2(device, new)
    rounded_carrier_rate = (new_carrier_inventory - old_carrier_inventory) / dt
    rounded_ion_rate = (new_ion_inventory - old_ion_inventory) / dt
    residual = step.residual(y)
    mask = _storage_mask(device.electronic_component)
    continuity_bound = Q * (float(np.sum(abs(residual[N:3 * N].reshape(2, N)[mask])))
                            + math.fsum(abs(residual[3 * N:]).tolist()))
    result = {
        "physical_certificate_status": "not_evaluated",
        "carrier_inventory_m2": new_carrier_inventory, "ion_inventory_m2": new_ion_inventory,
        "carrier_inventory_rate_m2_s": carrier_rate, "ion_inventory_rate_m2_s": ion_rate,
        "rounded_carrier_inventory_rate_m2_s": rounded_carrier_rate,
        "rounded_ion_inventory_rate_m2_s": rounded_ion_rate,
        "carrier_inventory_readback_error_m2_s": carrier_rate - rounded_carrier_rate,
        "ion_inventory_readback_error_m2_s": ion_rate - rounded_ion_rate,
        "carrier_particle_fluxes_m2_s": carrier_particles, "ion_particle_fluxes_m2_s": ion_particles,
        "ion_domain_fluxes_m2_s": ion_domain_flux,
        "carrier_balance_m2_s": carrier_balance, "ion_balance_m2_s": ion_balance,
        "charge_inventory_C_m2": new_charge, "charge_inventory_rate_A_m2": charge_rate,
        "charge_subtraction_roundoff_A_m2": charge_readback_error,
        "carrier_conduction_current_density_A_m2": carrier_conduction,
        "ion_conduction_current_density_A_m2": Q * signed_ion_particles,
        "conduction_current_density_A_m2": conduction,
        "interval_displacement_C_m2": interval_d,
        "rounded_interval_displacement_C_m2": rounded_interval_d,
        "displacement_readback_error_C_m2": displacement_readback_error,
        "displacement_current_density_A_m2": displacement_current,
        "total_current_density_A_m2": maxwell,
        "terminal_total_current_density_A_m2": maxwell[[0, -1]],
        "displacement_C_m2": new_d,
        "old_gauss_defect_C_m2": old_gauss, "gauss_defect_C_m2": new_gauss,
        "gauss_defect_rate_A_m2": rounded_gauss_rate, "stable_gauss_defect_rate_A_m2": stable_gauss_rate,
        "maxwell_cell_jump_A_m2": jump, "predicted_maxwell_cell_jump_A_m2": predicted,
        "readback_predicted_maxwell_cell_jump_A_m2": rounded_predicted,
        "maxwell_identity_error_A_m2": jump - predicted,
        "readback_maxwell_identity_error_A_m2": jump - rounded_predicted,
        "global_ion_inventory_rate_m2_s": math.fsum(ion_rate.tolist()),
        "global_ion_balance_m2_s": math.fsum(ion_balance.tolist()),
        "global_charge_balance_A_m2": float(np.sum(charge_rate) - (conduction[0] - conduction[-1])),
        "maxwell_spread_A_m2": float(np.ptp(maxwell)),
        "continuity_defect_bound_A_m2": continuity_bound,
        "backward_euler_residual": residual, "dc_residual": problem.residual(y), "storage_mask": mask,
        "compiled_identity": device.identity, "previous_compiled_identity": old.device_sha256,
        "previous_voltage_V": old.voltage_V, "voltage_V": problem.voltage_V, "dt_s": dt,
    }
    if device.ion_species_count == 2:
        positive_keys = (
            "ion_inventory_m2", "ion_inventory_rate_m2_s", "rounded_ion_inventory_rate_m2_s",
            "ion_inventory_readback_error_m2_s", "ion_particle_fluxes_m2_s", "ion_domain_fluxes_m2_s",
            "ion_balance_m2_s", "global_ion_inventory_rate_m2_s", "global_ion_balance_m2_s",
        )
        positive = {key: result[key] for key in positive_keys}
        positive["ion_conduction_current_density_A_m2"] = Q * ion_particles
        _, old_negative_inventory = inventory_m2(device, old, species="negative")
        _, new_negative_inventory = inventory_m2(device, new, species="negative")
        rounded_negative_rate = (new_negative_inventory - old_negative_inventory) / dt
        negative = {
            "ion_inventory_m2": new_negative_inventory, "ion_inventory_rate_m2_s": negative_rate,
            "rounded_ion_inventory_rate_m2_s": rounded_negative_rate,
            "ion_inventory_readback_error_m2_s": negative_rate - rounded_negative_rate,
            "ion_particle_fluxes_m2_s": negative_particles, "ion_domain_fluxes_m2_s": negative_flux,
            "ion_balance_m2_s": negative_balance,
            "ion_conduction_current_density_A_m2": -Q * negative_particles,
            "global_ion_inventory_rate_m2_s": math.fsum(negative_rate.tolist()),
            "global_ion_balance_m2_s": math.fsum(negative_balance.tolist()),
        }
        result["ion_species"] = {"positive": positive, "negative": negative}
    if device.ion_site_contract == "shared_finite_sites":
        occupied = (new.positive_ion_density_m3 + new.negative_ion_density_m3) / device.positive_ion_site_limit_m3
        result["shared_site"] = {
            "summed_site_fraction": occupied, "vacancy_fraction": 1.0 - occupied,
            "maximum_summed_site_fraction": float(np.max(occupied)),
            "minimum_vacancy_fraction": float(np.min(1.0 - occupied)),
        }
    return result


ionic_step_accounting = step_accounting
