"""Physical-inventory backward Euler for the existing ion-free 1D equations.

For positive carrier inventories S=(0,w*n,w*p), R_BE=F_DC-(S_new-S_old)/dt.
Fixed-reservoir rows have zero equation storage, but their actual endpoint
inventories remain in port accounting. K=F_udot=-dS/du, hence J_BE=J_DC+K/dt.
No coordinate time difference or nonlinear coordinate normalization is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import csc_matrix

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem, State1D
from solarlab.physics.electrostatics import contact_displacement
from solarlab.solvers.nonlinear.production import poisson_row_weights

__all__ = ["BackwardEuler1DStep", "inventory_m2", "step_accounting"]


def inventory_m2(device: CompiledDevice1D, state: State1D) -> np.ndarray:
    """Both physical particle inventories per area, including endpoint cells."""
    if state.device_sha256 != device.identity:
        raise ValueError("inventory requires a state from the same compiled device")
    return np.stack((state.electron_density_m3, state.hole_density_m3)) * device.geometry.cell_measures_m


def _storage_mask(device: CompiledDevice1D) -> np.ndarray:
    mask: np.ndarray = np.ones((2, device.node_count), dtype=bool)
    for species, carrier in enumerate(("n", "p")):
        for side, node in (("left", 0), ("right", device.node_count - 1)):
            if device.channel(carrier, side).kind == "fixed_reservoir":
                mask[species, node] = False
    return mask


@dataclass(frozen=True)
class BackwardEuler1DStep:
    """One physical step with independently bound old/new voltage references."""

    endpoint: DC1DProblem
    previous: State1D
    dt_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if self.previous.device_sha256 != self.endpoint.device.identity:
            raise ValueError("the old state must belong to this same device")
        actual = self.endpoint.device.state(self.previous.u, voltage_V=self.previous.voltage_V)
        for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
            if not np.array_equal(getattr(actual, name), getattr(self.previous, name)):
                raise ValueError("old physical fields disagree with their coordinate reference")

    def inventory_rate_m2_s(self, u: np.ndarray) -> np.ndarray:
        """Nonlinear physical inventory secant, evaluated without cancellation.

        For each carrier S_new-S_old = S_old*expm1(log(c_new/c_old)).
        expm1 is essential: replacing it by the exponent would instead be
        coordinate differencing, which is not backward Euler in inventory.
        """
        device = self.endpoint.device
        count = device.node_count
        values = np.asarray(u, dtype=float)
        delta = values - self.previous.u
        old_ref = device.carrier_reference_energies(self.previous.voltage_V)
        new_ref = device.carrier_reference_energies(self.endpoint.voltage_V)
        log_ratio = np.stack((
            (delta[:count] + (new_ref[0] - old_ref[0])) / device.thermal_voltage_V + delta[count:2 * count],
            -(delta[:count] + (new_ref[1] - old_ref[1])) / device.thermal_voltage_V - delta[2 * count:],
        ))
        return inventory_m2(device, self.previous) * np.expm1(log_ratio) / self.dt_s

    def residual(self, u: np.ndarray) -> np.ndarray:
        result = self.endpoint.residual(u)
        count = self.endpoint.device.node_count
        rate = self.inventory_rate_m2_s(u)
        rows = result[count:].reshape(2, count)
        mask = _storage_mask(self.endpoint.device)
        rows[mask] -= rate[mask]
        return result

    def jacobian(self, u: np.ndarray) -> csc_matrix:
        return self.endpoint.jacobian(u) + self.endpoint.storage_jacobian(u) / self.dt_s

    def initial_coordinates(self, predictor: str = "previous") -> np.ndarray:
        """Explicit Newton seed; it never changes the old physical inventory.

        generation_inventory adds dt*G_new equally to the two differential
        carrier populations, retaining old phi. Transport/recombination are
        solved by implicit Newton, not included in this positive source seed.
        This is neither an accepted time step nor a steady settling stage.
        """
        device = self.endpoint.device
        values = device.rebase_coordinates(
            self.previous.u, self.previous.voltage_V, self.endpoint.voltage_V,
        )
        if predictor == "previous":
            return values
        if predictor != "generation_inventory":
            raise ValueError("unknown explicit Newton predictor")
        increment = self.dt_s * self.endpoint.generation_scale * device.generation.source_m3_s
        mask = _storage_mask(device)
        count = device.node_count
        for species, (density, offset, sign) in enumerate((
            (self.previous.electron_density_m3, count, 1.0),
            (self.previous.hole_density_m3, 2 * count, -1.0),
        )):
            addition = np.where(mask[species], increment, 0.0)
            values[offset:offset + count] += sign * np.log1p(addition / density)
        if not np.all(np.isfinite(values)):
            raise ValueError("generation-inventory predictor is nonfinite; no clipping or fallback")
        return values


def step_accounting(step: BackwardEuler1DStep, u: np.ndarray) -> dict[str, object]:
    """Raw conservative diagnostics, never a physical qualification verdict.

All face currents point along +x, including left/right external faces. Number
inventories are particles/m^2; charge and D are C/m^2; currents are A/m^2.
Fixed-reservoir particle fluxes are reconstructed from physical endpoint
inventory rates, not the overwritten algebraic continuity rows.
"""
    problem, old = step.endpoint, step.previous
    device, count = problem.device, problem.device.node_count
    new = device.state(u, voltage_V=problem.voltage_V)
    widths = device.geometry.cell_measures_m
    if not np.array_equal(widths[1:-1], device.poisson.interior_widths_m):
        raise ValueError("carrier and Gauss interior control volumes must agree")
    rate = step.inventory_rate_m2_s(u)
    pair_source = widths * (problem.generation_scale * device.generation.source_m3_s
                           - problem.recombination_rate(new.electron_density_m3,
                                                        new.hole_density_m3,
                                                        derivatives=False).rate)
    internal = problem.fluxes(u)
    particles = np.empty((2, count + 1))
    for species, (carrier, flux) in enumerate((("n", internal.electron), ("p", internal.hole))):
        particles[species, 1:-1] = flux.flux
        for side, node, face, target, sign in (
            ("left", 0, 0, 0, 1.0), ("right", count - 1, -1, -1, -1.0),
        ):
            channel = device.channel(carrier, side)
            if channel.kind == "fixed_reservoir":
                exchange = flux.flux[face] + sign * (rate[species, node] - pair_source[node])
            else:
                exchange, _ = problem._contact_exchange(u, channel, node)
            particles[species, target] = exchange
    balance = particles[:, :-1] - particles[:, 1:] + pair_source - rate
    conduction = Q * (particles[1] - particles[0])

    def displacement(state: State1D) -> tuple[np.ndarray, np.ndarray]:
        rho = device.charge_ledger(state.electron_density_m3, state.hole_density_m3).space_charge_C_m3()
        interior = device.poisson.face_displacement(state.potential_V)
        ports = contact_displacement(interior, rho, widths)
        return np.r_[ports[0], interior, ports[1]], rho * widths

    old_d, old_charge = displacement(old)
    new_d, new_charge = displacement(new)
    displacement_current = (new_d - old_d) / step.dt_s
    maxwell = conduction + displacement_current
    old_gauss = old_d[:-1] - old_d[1:] + old_charge
    new_gauss = new_d[:-1] - new_d[1:] + new_charge
    gauss_rate = (new_gauss - old_gauss) / step.dt_s
    charge_defect = Q * (balance[1] - balance[0])
    # A separately rounded charge-ledger subtraction and the stable physical
    # secant can differ by roundoff. Expose it in the identity, never erase it.
    charge_subtraction_roundoff = Q * (rate[1] - rate[0]) - (new_charge - old_charge) / step.dt_s
    predicted_jump = charge_defect + gauss_rate + charge_subtraction_roundoff
    actual_jump = maxwell[:-1] - maxwell[1:]
    dc_residual, be_residual = problem.residual(u), step.residual(u)
    mask = _storage_mask(device)
    return {
        "physical_certificate_status": "not_evaluated",
        "inventory_m2": inventory_m2(device, new),
        "inventory_rate_m2_s": rate,
        "rounded_inventory_subtraction_rate_m2_s": (
            inventory_m2(device, new) - inventory_m2(device, old)) / step.dt_s,
        "inventory_subtraction_roundoff_m2_s": rate - (
            inventory_m2(device, new) - inventory_m2(device, old)) / step.dt_s,
        "net_pair_source_m2_s": pair_source,
        "particle_fluxes_m2_s": particles,
        "carrier_balance_m2_s": balance,
        "charge_inventory_C_m2": new_charge,
        "charge_inventory_rate_A_m2": Q * (rate[1] - rate[0]),
        "charge_subtraction_roundoff_A_m2": charge_subtraction_roundoff,
        "conduction_current_density_A_m2": conduction[1:-1],
        "displacement_current_density_A_m2": displacement_current[1:-1],
        "total_current_density_A_m2": maxwell[1:-1],
        "terminal_conduction_current_density_A_m2": conduction[[0, -1]],
        "terminal_displacement_current_density_A_m2": displacement_current[[0, -1]],
        "terminal_total_current_density_A_m2": maxwell[[0, -1]],
        "displacement_C_m2": new_d,
        "old_gauss_defect_C_m2": old_gauss,
        "gauss_defect_C_m2": new_gauss,
        "gauss_defect_rate_A_m2": gauss_rate,
        "maxwell_cell_jump_A_m2": actual_jump,
        "predicted_maxwell_cell_jump_A_m2": predicted_jump,
        "maxwell_identity_error_A_m2": actual_jump - predicted_jump,
        "global_particle_balance_m2_s": np.sum(rate, axis=1) - (
            particles[:, 0] - particles[:, -1] + np.sum(pair_source)),
        "global_charge_balance_A_m2": float(np.sum(Q * (rate[1] - rate[0]))
                                             - (conduction[0] - conduction[-1])),
        "maxwell_spread_A_m2": float(np.ptp(maxwell)),
        "continuity_defect_bound_A_m2": float(Q * np.sum(np.abs(be_residual[count:].reshape(2, count)[mask]))),
        "poisson_normalized_max": float(np.max(np.abs(dc_residual[:count] * poisson_row_weights(device)))),
        "dc_residual": dc_residual,
        "backward_euler_residual": be_residual,
        "storage_mask": mask,
    }
