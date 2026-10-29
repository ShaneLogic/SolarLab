"""Mobile-positive AC for two separated blocking domains on one 1D mesh.

The production multiple-domain J/K are assembled once. Local single-domain
operators supply mathematical tangents only; their component states are not
claimed independent equilibria. All physical charge and Maxwell faces are
assembled globally, retaining both original domain balance blocks.
"""

from __future__ import annotations

from solarlab.reproducibility.job_provenance import admitted_source_pins

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.equations.ionic_small_signal import IonicSmallSignal1D, frequency_voltage
from solarlab.equations.multiple_ionic_1d import (
    CompiledMultipleIonicDevice1D, MultipleIonic1DProblem, MultipleIonicState1D,
)

__all__ = ["MultipleIonicACSystem", "MultipleIonicSmallSignal1D", "build_multiple_ionic_small_signal", "scope"]


def scope(device: CompiledMultipleIonicDevice1D, ion_mode: str = "mobile") -> None:
    if not isinstance(device, CompiledMultipleIonicDevice1D):
        raise ValueError("an actual complete multiple-domain ionic device is required")
    if ion_mode != "mobile":
        raise CapabilityError("two-domain AC currently supports only mobile positive ions")
    if len(device.domains) != 2:
        raise CapabilityError("two-domain AC requires exactly two separated blocking domains")
    if any(channel.kind != "fixed_reservoir" for channel in device.electronic_component.channels):
        raise CapabilityError("two-domain AC requires all fixed-reservoir carrier contacts")
    if device.electronic_component.field_transport is not None:
        raise CapabilityError("combined CT and multiple-domain AC is outside this initial scope")
    for domain in device.domains:
        if domain.ion_species_names != ("positive",) or domain.ion_site_contract != "independent_finite_sites":
            raise CapabilityError("two-domain AC requires one independent positive species per domain")
        if np.any(domain.positive_ion_diffusivity_faces_m2_s <= 0.0):
            raise CapabilityError("two-domain AC requires one connected positive-D population in each domain")
        record = domain.binding_manifest.get("ion_domain")
        if not isinstance(record, dict) or record.get("contacts") != ["blocking", "blocking"]:
            raise CapabilityError("two-domain AC requires both endpoints of every domain explicitly blocking")




def _sum_complex(values: ArrayLike) -> complex:
    array = np.asarray(values, dtype=complex)
    return complex(math.fsum(array.real.ravel().tolist()), math.fsum(array.imag.ravel().tolist()))


@dataclass(frozen=True)
class MultipleIonicACSystem:
    frequency_Hz: float
    voltage_phasor_V: complex
    jacobian: csc_matrix
    storage_jacobian: csc_matrix
    rhs: np.ndarray
    rhs_rate: np.ndarray
    fixed_coordinates: dict[int, complex]
    constraint_rows: dict[str, dict[str, int]]
    original_rows_enforced: tuple[int, ...]
    coordinate_form: str
    coordinate_lift_per_V: np.ndarray
    ion_mode: str = "mobile"


@dataclass(frozen=True)
class MultipleIonicSmallSignal1D:
    problem: MultipleIonic1DProblem
    operating_state: MultipleIonicState1D
    coordinate_form: str = "moving_boundary_lift_v1"
    jacobian: csc_matrix = field(init=False, repr=False)
    storage_jacobian: csc_matrix = field(init=False, repr=False)
    reference_direction_per_V: np.ndarray = field(init=False, repr=False)
    coordinate_lift_per_V: np.ndarray = field(init=False, repr=False)
    source_sha256: dict[str, str] = field(init=False)
    working_state_digest: str = field(init=False)
    working_state_identity: str = field(init=False)
    _local_operators: tuple[IonicSmallSignal1D, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.problem, MultipleIonic1DProblem):
            raise ValueError("the full production multiple-domain problem is required")
        device, state = self.problem.device, self.operating_state
        scope(device)
        if self.coordinate_form not in {"moving_boundary_lift_v1", "fixed_reference_v1"}:
            raise ValueError("an explicit supported AC coordinate representation is required")
        device.validate_state(state)
        if state.voltage_V != self.problem.voltage_V:
            raise ValueError("AC state and production operating voltage must agree")
        local = tuple(IonicSmallSignal1D(Ionic1DProblem(domain, state.voltage_V, self.problem.generation_scale),
            component_state, self.coordinate_form)
            for domain, component_state in zip(device.domains, state.domain_component_states))
        object.__setattr__(self, "_local_operators", local)
        object.__setattr__(self, "jacobian", self.problem.jacobian(state.u))
        object.__setattr__(self, "storage_jacobian", self.problem.storage_jacobian(state.u))
        N = device.node_count
        for name in ("reference_direction_per_V", "coordinate_lift_per_V"):
            values = np.zeros(device.state_size)
            values[:3 * N] = getattr(local[0], name)[:3 * N]
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        digest = hashlib.sha256(np.asarray(state.u, dtype=np.float64).tobytes()).hexdigest()
        object.__setattr__(self, "working_state_digest", digest)
        object.__setattr__(self, "working_state_identity", hashlib.sha256(json.dumps((
            device.identity, float(state.voltage_V).hex(), float(self.problem.generation_scale).hex(), digest,
        )).encode()).hexdigest())
        object.__setattr__(self, "source_sha256", admitted_source_pins())

    @property
    def device(self) -> CompiledMultipleIonicDevice1D:
        return self.problem.device

    def _moving_voltage_partial(self, direction: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray]:
        """One electronic drive plus each domain's actual J_ion,phi lift force."""
        device, N = self.device, self.device.node_count
        moving = np.zeros(device.state_size) if direction is None else np.asarray(direction, dtype=float)
        if moving.shape != (device.state_size,) or np.any(moving[3 * N:] != 0.0):
            raise ValueError("the voltage lift must keep every ionic coordinate unchanged")
        first = self._local_operators[0]
        partial, drive = first._moving_voltage_partial(np.r_[moving[:3 * N],
            np.zeros(first.device.state_size - 3 * N)])
        result, rate = np.zeros(device.state_size), np.zeros(device.state_size)
        result[:3 * N], rate[:3 * N] = partial[:3 * N], drive[:3 * N]
        result[3 * N:] = self.jacobian[3 * N:, :N] @ moving[:N]
        return result, rate

    def system(self, frequency_Hz: float, *, voltage_phasor_V: complex = 1.0,
               ion_mode: str = "mobile") -> MultipleIonicACSystem:
        scope(self.device, ion_mode)
        frequency, voltage = frequency_voltage(frequency_Hz, voltage_phasor_V, "mobile")
        device, N = self.device, self.device.node_count
        J, K = self.jacobian.copy().tolil(), self.storage_jacobian.copy().tolil()
        rhs: np.ndarray = np.zeros(device.state_size, dtype=complex)
        rhs[N - 1] = -device.electronic_component.voltage_polarity * voltage
        known = {0: 0.0j, N - 1: complex(rhs[N - 1])}
        for offset in (N, 2 * N):
            known[offset] = 0.0j
            known[offset + N - 1] = -rhs[N - 1] / device.electronic_component.thermal_voltage_V
        rhs_rate = np.zeros_like(rhs)
        if self.coordinate_form == "moving_boundary_lift_v1":
            partial, drive = self._moving_voltage_partial(self.coordinate_lift_per_V)
            rhs, rhs_rate = -partial * voltage, drive * voltage
            known = {index: 0.0j for index in known}
        constraints: dict[str, dict[str, int]] = {}
        if frequency == 0.0:
            for name, blocks in device.ion_species_slices.items():
                constraints[name] = {}
                for species, block in blocks.items():
                    row = block.start
                    gradient = -np.asarray(self.storage_jacobian[block, :].sum(axis=0)).ravel()
                    nonzero = np.flatnonzero(gradient)
                    J.rows[row], J.data[row] = nonzero.tolist(), gradient[nonzero].tolist()
                    K.rows[row], K.data[row] = [], []
                    rhs[row] = rhs_rate[row] = 0.0j
                    constraints[name][species] = row
        return MultipleIonicACSystem(frequency, voltage, J.tocsc(), K.tocsc(), rhs, rhs_rate,
            known, constraints, tuple(range(device.state_size)), self.coordinate_form,
            self.coordinate_lift_per_V)

    def original_residual(self, fixed_perturbation: ArrayLike, frequency_Hz: float,
                          *, voltage_phasor_V: complex = 1.0) -> np.ndarray:
        frequency, voltage = frequency_voltage(frequency_Hz, voltage_phasor_V, "mobile")
        x = np.asarray(fixed_perturbation, dtype=complex)
        if x.shape != (self.device.state_size,) or not np.all(np.isfinite(x)):
            raise ValueError("a finite complete two-domain perturbation is required")
        rhs = np.zeros_like(x)
        rhs[self.device.node_count - 1] = -self.device.electronic_component.voltage_polarity * voltage
        return (self.jacobian + 2j * math.pi * frequency * self.storage_jacobian) @ x - rhs

    def observe(self, perturbation: ArrayLike, frequency_Hz: float, *, voltage_phasor_V: complex = 1.0,
                ion_mode: str = "mobile", coordinate_form: str | None = None) -> dict[str, Any]:
        scope(self.device, ion_mode)
        frequency, voltage = frequency_voltage(frequency_Hz, voltage_phasor_V, "mobile")
        device, N = self.device, self.device.node_count
        form = self.coordinate_form if coordinate_form is None else coordinate_form
        x = np.asarray(perturbation, dtype=complex)
        if form not in {"moving_boundary_lift_v1", "fixed_reference_v1"}:
            raise ValueError("unknown AC coordinate representation")
        if x.shape != (device.state_size,) or not np.all(np.isfinite(x)):
            raise ValueError("a finite complete two-domain perturbation is required")
        fixed_x = x if form == "fixed_reference_v1" else x + (
            self.coordinate_lift_per_V - self.reference_direction_per_V) * voltage
        moving_x = x + (self.reference_direction_per_V if form == "fixed_reference_v1"
                        else self.coordinate_lift_per_V) * voltage
        pieces = tuple(cast(dict[str, Any], operator.observe(np.r_[x[:3 * N], x[device.domain_slices[name]]], frequency,
            voltage_phasor_V=voltage, coordinate_form=form))
            for name, operator in zip(device.domain_ids, self._local_operators))
        first, component = pieces[0], device.electronic_component
        carrier = np.asarray(first["carrier_particle_fluxes_m2_s"])
        carrier_balance = np.asarray(first["carrier_balance_m2_s"])
        dn, dp = first["electron_density_perturbation_m3"], first["hole_density_perturbation_m3"]
        charge = Q * component.geometry.cell_measures_m * (dp - dn)
        charge_balance = Q * (carrier_balance[1] - carrier_balance[0])
        conduction = Q * (carrier[1] - carrier[0])
        ionic_current: np.ndarray = np.zeros(N + 1, dtype=complex)
        base_ionic_charge, fixed_background = np.zeros(N), np.zeros(N)
        domains = {}
        for domain, part in zip(device.domains, pieces):
            name = domain.domain_layer_id
            density = part["ion_density_perturbation_m3"]["positive"]
            particles = part["ion_particle_fluxes_m2_s"]["positive"]
            balance = part["ion_balance_m2_s"]["positive"]
            weighted = domain.ion_geometry.cell_measures_m * density
            inventory = _sum_complex(weighted)
            deposited = Q * weighted
            charge[domain.ion_node_indices] += deposited
            charge_balance[domain.ion_node_indices] += Q * balance
            conduction += Q * particles
            ionic_current += Q * particles
            base_ionic_charge += part["base_ion_charge_density_C_m3"]
            fixed_background -= Q * domain.project_ion_density(domain.positive_ion_reference_m3)
            domains[name] = {"electrical_node_indices": domain.ion_node_indices,
                "physical_weights_m": domain.ion_geometry.cell_measures_m,
                "compiled_component_identity": domain.identity,
                "species": {"positive": {"density_perturbation_m3": density,
                    "particle_fluxes_m2_s": particles, "native_balance_m2_s": balance,
                    "inventory_perturbation_m2": inventory,
                    "charge_inventory_perturbation_C_m2": deposited,
                    "charge_inventory_roundoff_C_m2": _sum_complex(deposited) - Q * inventory,
                    "base_density_m3": part["base_ion_density_m3"]["positive"],
                    "reference_density_m3": domain.positive_ion_reference_m3,
                    "reference_inventory_m2": device.reference_inventories_m2[name]["positive"],
                    "blocking_face_indices": (int(domain.ion_node_indices[0]), int(domain.ion_node_indices[-1] + 1))}}}
        omega = 2 * math.pi * frequency
        phi = fixed_x[:N]
        interior_D = -component.poisson.face_capacitance_F_m2 * np.diff(phi)
        D = np.r_[interior_D[0] - charge[0], interior_D, interior_D[-1] + charge[-1]]
        displacement = 1j * omega * D
        total = conduction + displacement
        gauss = D[:-1] - D[1:] + charge
        identity = total[:-1] - total[1:] - charge_balance - 1j * omega * gauss
        matrix = self.jacobian + 1j * omega * self.storage_jacobian
        fixed_rhs: np.ndarray = np.zeros(device.state_size, dtype=complex)
        fixed_rhs[N - 1] = -component.voltage_polarity * voltage
        fixed_residual = matrix @ fixed_x - fixed_rhs
        partial, _ = self._moving_voltage_partial()
        moving_rhs = -partial * voltage + 1j * omega * (self.storage_jacobian @ self.reference_direction_per_V) * voltage
        moving_residual = matrix @ moving_x - moving_rhs
        errors = []
        for values, rhs, residual in ((fixed_x, fixed_rhs, fixed_residual), (moving_x, moving_rhs, moving_residual)):
            denominator = abs(matrix) @ abs(values) + abs(rhs)
            errors.append(float(np.max(np.divide(abs(residual), denominator,
                out=np.zeros_like(denominator), where=denominator != 0.0))))
        bounds = Q * (float(np.sum(abs(carrier_balance))) + math.fsum(
            float(np.sum(abs(part["ion_balance_m2_s"]["positive"]))) for part in pieces))
        bounds += omega * float(np.sum(abs(gauss)))
        positive_electrode = 0 if component.voltage_polarity > 0.0 else -1
        input_current = component.voltage_polarity * total[positive_electrode]
        return {"frequency_Hz": frequency, "voltage_phasor_V": voltage, "ion_mode": "mobile",
            "coordinate_form": form, "solve_coordinate_perturbation": x.copy(),
            "fixed_coordinate_perturbation": fixed_x, "moving_coordinate_perturbation": moving_x,
            "reference_direction_per_V": self.reference_direction_per_V,
            "coordinate_lift_per_V": self.coordinate_lift_per_V,
            "potential_perturbation_V": phi, "electric_field_perturbation_V_m": -np.diff(phi) / np.diff(component.nodes_m),
            "electron_density_perturbation_m3": dn, "hole_density_perturbation_m3": dp,
            "carrier_particle_fluxes_m2_s": carrier, "carrier_balance_m2_s": carrier_balance,
            "domains": domains, "charge_inventory_perturbation_C_m2": charge,
            "charge_balance_C_m2_s": charge_balance,
            "base_ion_charge_density_C_m3": base_ionic_charge, "fixed_ion_background_charge_C_m3": fixed_background,
            "ionic_conduction_current_density_A_m2": ionic_current,
            "conduction_current_density_A_m2": conduction, "displacement_current_density_A_m2": displacement,
            "total_current_density_A_m2": total, "terminal_current_density_A_m2": total[[0, -1]],
            "terminal_conduction_current_density_A_m2": conduction[[0, -1]],
            "terminal_displacement_current_density_A_m2": displacement[[0, -1]],
            "input_current_density_A_m2": input_current, "admittance_density_S_m2": input_current / voltage,
            "maxwell_spread_A_m2_V": float(np.max(abs(total - total[0]))) / abs(voltage),
            "current_scale_A_m2_V": float(np.max(abs(total))) / abs(voltage),
            "residual_balance_bound_A_m2_V": bounds / abs(voltage),
            "gauss_defect_C_m2": gauss, "maxwell_identity_error_A_m2": identity,
            "original_fixed_residual": fixed_residual, "original_residual": moving_residual,
            "original_componentwise_backward_error": max(errors),
            "original_rows_enforced": tuple(range(device.state_size)),
            "original_ion_force_residual": {name: {species: moving_residual[block] for species, block in blocks.items()}
                for name, blocks in device.ion_species_slices.items()},
            "compiled_identity": device.identity, "working_state_identity": self.working_state_identity,
            "working_state_digest": self.working_state_digest, "source_sha256": dict(self.source_sha256),
            "physical_certificate_status": "not_evaluated"}


def build_multiple_ionic_small_signal(compiled: CompiledMultipleIonicDevice1D, state: MultipleIonicState1D,
                                     *, generation_scale: float = 0.0,
                                     coordinate_form: str = "moving_boundary_lift_v1") -> MultipleIonicSmallSignal1D:
    scope(compiled)
    if not isinstance(state, MultipleIonicState1D):
        raise ValueError("a complete bound multiple-domain operating state is required")
    return MultipleIonicSmallSignal1D(MultipleIonic1DProblem(compiled, state.voltage_V, generation_scale),
                                      state, coordinate_form)
