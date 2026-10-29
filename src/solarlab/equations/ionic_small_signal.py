"""Fixed-reservoir ionic AC operator and full Maxwell-current observation.

Phasors use exp(+i*omega*t). The fixed physical carrier-reference coordinate
x=du_moving-c*Vhat gives (J+i*omega*K)x=-polarity*e_phi_right*Vhat:
the moving storage drive -K*c*Vdot is counted once and cancels on conversion.
Main admits/revalidates accepted saved DC results; this layer checks the full
bound state and supplies mathematical operators, not an acceptance certificate.
The default solve coordinates are small boundary-lifted moving coordinates;
known voltage terms remain separate through physical density/current evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.dc_1d import CapabilityError, quasi_fermi_potential_tangents
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, Ionic1DProblem, IonicState1D
from solarlab.equations.ionic_closure import declared_immobile_profiles, immobile_profile_binding
from solarlab.physics.ions import ion_face_flux_jacobian

__all__ = ["IonicACSystem", "IonicSmallSignal1D", "build_ionic_small_signal", "frequency_voltage"]


def frequency_voltage(frequency_Hz: float, voltage_phasor_V: complex, ion_mode: str) -> tuple[float, complex]:
    frequency, voltage = float(frequency_Hz), complex(voltage_phasor_V)
    if not math.isfinite(frequency) or frequency < 0.0:
        raise ValueError("frequency must be finite nonnegative Hz")
    if not np.isfinite(voltage) or voltage == 0.0:
        raise ValueError("voltage phasor must be finite and nonzero; 1 V denotes a derivative")
    if ion_mode not in {"mobile", "frozen"}:
        raise ValueError("ion_mode must explicitly select mobile or frozen ions")
    return frequency, voltage


def _source_pins(*, field_transport: bool = False) -> dict[str, str]:
    names: tuple[str, ...] = (__name__, "solarlab.equations.ionic_1d", "solarlab.equations.dc_1d",
             "solarlab.physics.ions.migration", "solarlab.physics.ions.finite_site",
             "solarlab.physics.ions.shared_site", "solarlab.physics.transport.electrochemical_flux",
             "solarlab.physics.electrostatics.poisson", "solarlab.equations.ionic_closure")
    if field_transport:
        names += ("solarlab.equations.field_transport_1d",
                  "solarlab.physics.transport.field_mobility",
                  "solarlab.physics.transport.field_mobility_tangents")
    result = {}
    for name in names:
        path = sys.modules[name].__file__
        if path is None:
            raise ValueError("ionic AC source must be bound to actual source files")
        result[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result


@dataclass(frozen=True)
class IonicACSystem:
    """One amplitude-bound system for the unchanged complex sparse solver.

    rhs/fixed_coordinates already include voltage_phasor_V. Inventory rows
    constrain physical particle number; original mobile rows stay diagnostic.
    """

    frequency_Hz: float
    voltage_phasor_V: complex
    ion_mode: str
    jacobian: csc_matrix
    storage_jacobian: csc_matrix
    rhs: np.ndarray
    rhs_rate: np.ndarray
    fixed_coordinates: dict[int, complex]
    constraint_rows: dict[str, int]
    original_rows_enforced: tuple[int, ...]
    coordinate_form: str
    coordinate_lift_per_V: np.ndarray
    immobile_constraint_rows: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def coordinate_representation(self) -> str:
        return self.coordinate_form

    @property
    def rhs_per_V(self) -> np.ndarray:
        return self.rhs / self.voltage_phasor_V

    @property
    def rhs_rate_per_V(self) -> np.ndarray:
        return self.rhs_rate / self.voltage_phasor_V

    @property
    def fixed_coordinates_per_V(self) -> dict[int, complex]:
        return {index: value / self.voltage_phasor_V for index, value in self.fixed_coordinates.items()}


@dataclass(frozen=True)
class IonicSmallSignal1D:
    """Production linearization about a complete, externally admitted DC state.

    Bound-field consistency is checked here. The experiment driver must first
    reject missing/failed DC results and revalidate its original physical gates.
    """

    problem: Ionic1DProblem
    operating_state: IonicState1D
    coordinate_form: str = "moving_boundary_lift_v1"
    jacobian: csc_matrix = field(init=False, repr=False)
    storage_jacobian: csc_matrix = field(init=False, repr=False)
    reference_direction_per_V: np.ndarray = field(init=False, repr=False)
    coordinate_lift_per_V: np.ndarray = field(init=False, repr=False)
    source_sha256: dict[str, str] = field(init=False)
    working_state_digest: str = field(init=False)
    working_state_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.problem, Ionic1DProblem) or not isinstance(self.operating_state, IonicState1D):
            raise ValueError("ionic AC requires a full ionic problem and bound operating state")
        if self.coordinate_form not in {"moving_boundary_lift_v1", "fixed_reference_v1"}:
            raise ValueError("ionic AC requires an explicit supported coordinate representation")
        device, state = self.problem.device, self.operating_state
        if state.device_sha256 != device.identity or state.voltage_V != self.problem.voltage_V:
            raise ValueError("ionic AC state must have the same complete device and operating voltage")
        if any(channel.kind != "fixed_reservoir" for channel in device.electronic_component.channels):
            raise CapabilityError("ionic AC currently requires all fixed-reservoir carrier contacts")
        manifest = device.binding_manifest
        domain, component = manifest.get("ion_domain"), manifest.get("electronic_component")
        if (hashlib.sha256(device.binding_manifest_json.encode()).hexdigest() != device.identity
                or not isinstance(domain, dict) or domain.get("contacts") != ["blocking", "blocking"]
                or not isinstance(component, dict) or component.get("compiled_sha256") != device.electronic_component.identity):
            raise ValueError("ionic AC requires the original full binding and one declared blocking domain")
        actual = device.state(state.u, voltage_V=state.voltage_V)
        for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
            if not np.array_equal(getattr(state.electronic_state, name), getattr(actual.electronic_state, name)):
                raise ValueError("ionic AC operating fields disagree with their bound coordinates")
        if (state.electronic_state.device_sha256 != device.electronic_component.identity
                or not np.array_equal(actual.positive_ion_density_m3, state.positive_ion_density_m3)
                or not np.array_equal(actual.negative_ion_density_m3, state.negative_ion_density_m3)
                or actual.domain_inventories_m2 != state.domain_inventories_m2):
            raise ValueError("ionic AC requires complete consistent species fields/inventories")
        object.__setattr__(self, "jacobian", self.problem.jacobian(state.u))
        object.__setattr__(self, "storage_jacobian", self.problem.storage_jacobian(state.u))
        count = device.node_count
        direction = np.zeros(device.state_size)
        before, after = device.electronic_component.carrier_reference_energies(0.0), device.electronic_component.carrier_reference_energies(1.0)
        for offset, zero, one in zip((count, 2 * count), before, after):
            direction[offset:offset + count] = -(one - zero) / device.electronic_component.thermal_voltage_V
        direction.setflags(write=False)
        object.__setattr__(self, "reference_direction_per_V", direction)
        lift = np.zeros(device.state_size)
        lift[count - 1] = -device.electronic_component.voltage_polarity
        for offset in (count, 2 * count):
            for node in (0, count - 1):
                lift[offset + node] = -lift[node] / device.electronic_component.thermal_voltage_V + direction[offset + node]
        lift.setflags(write=False)
        object.__setattr__(self, "coordinate_lift_per_V", lift)
        digest = hashlib.sha256(np.asarray(state.u, dtype=np.float64).tobytes()).hexdigest()
        object.__setattr__(self, "working_state_digest", digest)
        object.__setattr__(self, "working_state_identity", hashlib.sha256(json.dumps((
            device.identity, float(state.voltage_V).hex(), float(self.problem.generation_scale).hex(), digest,
        )).encode()).hexdigest())
        object.__setattr__(self, "source_sha256", _source_pins(field_transport=device.electronic_component.field_transport is not None))

    @property
    def device(self) -> CompiledIonicDevice1D:
        return self.problem.device

    @property
    def coordinate_representation(self) -> str:
        return self.coordinate_form

    def system(self, frequency_Hz: float, *, voltage_phasor_V: complex = 1.0, ion_response: str = "mobile", ion_mode: str | None = None) -> IonicACSystem:
        ion_mode = ion_response if ion_mode is None else ion_mode
        frequency, voltage = frequency_voltage(frequency_Hz, voltage_phasor_V, ion_mode)
        device, N = self.device, self.device.node_count
        J, K = self.jacobian.copy().tolil(), self.storage_jacobian.copy().tolil()
        rhs: np.ndarray = np.zeros(device.state_size, dtype=complex)
        rhs[N - 1] = -device.electronic_component.voltage_polarity * voltage
        phi_right = complex(rhs[N - 1])
        known = {0: 0.0j, N - 1: phi_right}
        for offset in (N, 2 * N):
            known[offset] = 0.0j
            known[offset + N - 1] = -phi_right / device.electronic_component.thermal_voltage_V
        rhs_rate = np.zeros_like(rhs)
        if self.coordinate_form == "moving_boundary_lift_v1":
            partial, drive = self._moving_voltage_partial(self.coordinate_lift_per_V)
            rhs = -partial * voltage
            rhs_rate = drive * voltage
            known = {index: 0.0j for index in known}
        rows: dict[str, int] = {}
        profile_rows: dict[str, tuple[int, ...]] = {}
        profiles = declared_immobile_profiles(device)
        enforced: np.ndarray = np.ones(device.state_size, dtype=bool)
        if ion_mode == "frozen":
            for index in range(3 * N, device.state_size):
                J.rows[index], J.data[index] = [index], [1.0]
                K.rows[index], K.data[index] = [], []
                known[index] = 0.0j
                rhs[index] = rhs_rate[index] = 0.0j
            enforced[3 * N:] = False
        elif frequency == 0.0:
            for species, block in zip(device.ion_species_names, device.ion_species_slices):
                if species in profiles:
                    local_rows = tuple(range(block.start, block.stop))
                    for row in local_rows:
                        gradient = -self.storage_jacobian[row, :].toarray()[0]
                        nonzero = np.flatnonzero(gradient)
                        J.rows[row], J.data[row] = nonzero.tolist(), gradient[nonzero].tolist()
                        K.rows[row], K.data[row] = [], []
                        rhs[row] = rhs_rate[row] = 0.0j
                    profile_rows[species] = local_rows
                    continue
                if np.any(getattr(device, species + "_ion_diffusivity_faces_m2_s") <= 0.0):
                    raise CapabilityError("zero-Hz mobile ions require one connected positive-D domain per species")
                row = block.start
                gradient = -np.asarray(self.storage_jacobian[block, :].sum(axis=0)).ravel()
                nonzero = np.flatnonzero(gradient)
                J.rows[row], J.data[row] = nonzero.tolist(), gradient[nonzero].tolist()
                K.rows[row], K.data[row] = [], []
                rhs[row] = rhs_rate[row] = 0.0j
                rows[species] = row
        return IonicACSystem(frequency, voltage, ion_mode, J.tocsc(), K.tocsc(), rhs,
                             rhs_rate, known, rows, tuple(int(index) for index in np.flatnonzero(enforced)),
                             self.coordinate_form, self.coordinate_lift_per_V, profile_rows)

    def original_residual(self, fixed_perturbation: ArrayLike, frequency_Hz: float, *, voltage_phasor_V: complex = 1.0) -> np.ndarray:
        frequency, voltage = frequency_voltage(frequency_Hz, voltage_phasor_V, "mobile")
        x = np.asarray(fixed_perturbation, dtype=complex)
        if x.shape != (self.device.state_size,) or not np.all(np.isfinite(x)):
            raise ValueError("ionic AC perturbation must be finite and complete")
        rhs = np.zeros_like(x)
        rhs[self.device.node_count - 1] = -self.device.electronic_component.voltage_polarity * voltage
        return (self.jacobian + 2j * math.pi * frequency * self.storage_jacobian) @ x - rhs

    def _carrier_particles(self, x: np.ndarray, reference_drive: complex = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Production quasi-Fermi tangent, grouping exact flux homogeneity modes."""
        component, N = self.device.electronic_component, self.device.node_count
        u = self.operating_state.u[:3 * N]
        fluxes = self.problem.electronic_problem.fluxes(u)
        transport = None if component.field_transport is None else component.field_transport.evaluate(u[:N])
        state = self.operating_state.electronic_state
        before, after = component.carrier_reference_energies(0.0), component.carrier_reference_energies(1.0)
        values = []
        for offset, density, flux, sign, reference in ((N, state.electron_density_m3, fluxes.electron, 1.0, after[0] - before[0]),
                                                      (2 * N, state.hole_density_m3, fluxes.hole, -1.0, after[1] - before[1])):
            drive = sign * np.diff(u[offset:offset + N])
            left_phi, _ = quasi_fermi_potential_tangents(flux, drive, sign, component.thermal_voltage_V)
            if transport is not None:
                diffusivity = (transport.electron_diffusivity_m2_s if offset == N
                               else transport.hole_diffusivity_m2_s)
                d_diffusivity = (transport.electron_d_diffusivity_d_field if offset == N
                                 else transport.hole_d_diffusivity_d_field)
                # Same antisymmetric CT chain as production Fy. A moving
                # carrier energy reference alone has no electric-field drive.
                left_phi = left_phi + (flux.flux / diffusivity) * d_diffusivity / np.diff(component.nodes_m)
            left_z = sign * flux.left_density_derivative * density[:-1]
            values.append(left_phi * (x[:N - 1] - x[1:N])
                          + left_z * (x[offset:offset + N - 1] - x[offset + 1:offset + N])
                          + sign * flux.flux * (x[offset + 1:offset + N] + (x[1:N] + reference * reference_drive) / component.thermal_voltage_V))
        return values[0], values[1]

    def _moving_voltage_partial(self, direction: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Grouped actual F_V+J*direction and K*(c-direction), before cancellation.

        Only the declared zero ion-reference/ion-lift path is used. The potential
        lift's ionic endpoint force remains through the production Jion*lift.
        """
        device, N = self.device, self.device.node_count
        component, state = device.electronic_component, self.operating_state.electronic_state
        moving = np.zeros(device.state_size) if direction is None else np.asarray(direction)
        if np.any(moving[3 * N:] != 0.0):
            raise ValueError("this AC voltage lift cannot change ion coordinates")
        before, after = component.carrier_reference_energies(0.0), component.carrier_reference_energies(1.0)
        references = (after[0] - before[0], after[1] - before[1])
        dn = state.electron_density_m3 * ((moving[:N] + references[0]) / component.thermal_voltage_V + moving[N:2 * N])
        dp = -state.hole_density_m3 * ((moving[:N] + references[1]) / component.thermal_voltage_V + moving[2 * N:3 * N])
        rate = self.problem.electronic_problem.recombination_rate(state.electron_density_m3, state.hole_density_m3)
        assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
        source = -(rate.d_rate_dn * dn + rate.d_rate_dp * dp)
        fn, fp = self._carrier_particles(moving, 1.0)
        result: np.ndarray = np.zeros(device.state_size)
        storage_drive: np.ndarray = np.zeros(device.state_size)
        result[:N] = component.poisson.jacobian() @ moving[:N] + component.poisson.charge_jacobian() * Q * (dp - dn)
        result[N - 1] += component.voltage_polarity
        for offset, population, flux, sign, reference in ((N, state.electron_density_m3, fn, 1.0, references[0]),
                                                          (2 * N, state.hole_density_m3, fp, -1.0, references[1])):
            rows = -np.diff(np.r_[0.0, flux, 0.0]) + component.geometry.cell_measures_m * source
            for node in (0, N - 1):
                rows[node] = sign * ((moving[node] + reference) / component.thermal_voltage_V + moving[offset + node])
            result[offset:offset + N] = rows
            storage_drive[offset:offset + N] = component.geometry.cell_measures_m * sign * population * (
                (moving[:N] + reference) / component.thermal_voltage_V + moving[offset:offset + N]
            )
            storage_drive[offset] = storage_drive[offset + N - 1] = 0.0
        result[3 * N:] = self.jacobian[3 * N:, :N] @ moving[:N]
        return result, storage_drive

    def observe(self, fixed_perturbation: ArrayLike | IonicACSystem, frequency_Hz: float | ArrayLike, *, voltage_phasor_V: complex = 1.0, ion_mode: str = "mobile", coordinate_form: str | None = None) -> dict[str, object]:
        if isinstance(fixed_perturbation, IonicACSystem):
            descriptor = fixed_perturbation
            fixed_perturbation, frequency_Hz, ion_mode = frequency_Hz, descriptor.frequency_Hz, descriptor.ion_mode
            coordinate_form = descriptor.coordinate_form
        frequency, voltage = frequency_voltage(frequency_Hz, voltage_phasor_V, ion_mode)
        device, N = self.device, self.device.node_count
        x = np.asarray(fixed_perturbation, dtype=complex)
        form = self.coordinate_form if coordinate_form is None else coordinate_form
        if form not in {"moving_boundary_lift_v1", "fixed_reference_v1"}:
            raise ValueError("unknown AC perturbation coordinate representation")
        if x.shape != (device.state_size,) or not np.all(np.isfinite(x)):
            raise ValueError("ionic AC perturbation must be finite and complete")
        solve_coordinates = x.copy()
        moving_x = x + self.reference_direction_per_V * voltage
        if form == "moving_boundary_lift_v1":
            moving_x = x + self.coordinate_lift_per_V * voltage
            fixed_x = x + (self.coordinate_lift_per_V - self.reference_direction_per_V) * voltage
        else:
            fixed_x = x
        fixed_residual = self.original_residual(fixed_x, frequency, voltage_phasor_V=voltage)
        if ion_mode == "frozen" and np.any(x[3 * N:] != 0.0):
            raise ValueError("frozen-ion response requires every ionic coordinate perturbation to be zero")
        omega = 2.0 * math.pi * frequency
        component, state = device.electronic_component, self.operating_state
        phi = fixed_x[:N]
        n, p = state.electronic_state.electron_density_m3, state.electronic_state.hole_density_m3
        dn = n * (phi / component.thermal_voltage_V + x[N:2 * N])
        dp = -p * (phi / component.thermal_voltage_V + x[2 * N:3 * N])
        if form == "moving_boundary_lift_v1":
            before, after = component.carrier_reference_energies(0.0), component.carrier_reference_energies(1.0)
            lift = self.coordinate_lift_per_V
            dn = n * (x[:N] / component.thermal_voltage_V + x[N:2 * N]) + n * ((lift[:N] + after[0] - before[0]) / component.thermal_voltage_V + lift[N:2 * N]) * voltage
            dp = -p * (x[:N] / component.thermal_voltage_V + x[2 * N:3 * N]) - p * ((lift[:N] + after[1] - before[1]) / component.thermal_voltage_V + lift[2 * N:3 * N]) * voltage
        mass = device.ion_coordinate_jacobian_m3(state.u, voltage_V=state.voltage_V)
        coordinates = np.stack([x[block] for block in device.ion_species_slices], axis=1)
        densities = np.sum(mass * coordinates[:, None, :], axis=2).T
        fn, fp = self._carrier_particles(x)
        if form == "moving_boundary_lift_v1":
            known_n, known_p = self._carrier_particles(self.coordinate_lift_per_V, 1.0)
            fn, fp = fn + known_n * voltage, fp + known_p * voltage
        rate = self.problem.electronic_problem.recombination_rate(n, p)
        assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
        pair = -component.geometry.cell_measures_m * (rate.d_rate_dn * dn + rate.d_rate_dp * dp)
        carrier: np.ndarray = np.empty((2, N + 1), dtype=complex)
        for index, (flux, density) in enumerate(((fn, dn), (fp, dp))):
            carrier[index, 1:-1] = flux
            carrier[index, 0] = flux[0] - pair[0] + 1j * omega * component.geometry.cell_measures_m[0] * density[0]
            carrier[index, -1] = flux[-1] + pair[-1] - 1j * omega * component.geometry.cell_measures_m[-1] * density[-1]
        carrier_balance = -np.diff(carrier, axis=1) + pair - 1j * omega * component.geometry.cell_measures_m * np.stack((dn, dp))
        populations = (state.positive_ion_density_m3,) if device.ion_species_count == 1 else (state.positive_ion_density_m3, state.negative_ion_density_m3)
        ion_particles, ion_balance, base_profiles, ion_inventory = {}, {}, {}, {}
        charge = Q * component.geometry.cell_measures_m * (dp - dn)
        charge_balance = Q * (carrier_balance[1] - carrier_balance[0])
        conduction = Q * (carrier[1] - carrier[0])
        ionic_conduction: np.ndarray = np.zeros(N + 1, dtype=complex)
        base_ion_charge = np.zeros(N)
        for index, (species, density, block) in enumerate(zip(device.ion_species_names, populations, device.ion_species_slices)):
            sign = 1.0 if index == 0 else -1.0
            tangent = ion_face_flux_jacobian(
                state.electronic_state.potential_V[device.ion_node_indices], density, np.diff(device.ion_geometry.nodes_m),
                getattr(device, species + "_ion_diffusivity_faces_m2_s"), component.thermal_voltage_V,
                getattr(device, species + "_ion_site_limit_m3")[:-1],
                steric_diffusion_only=device.steric_model == "historical_diffusion_only_steric_v1",
                P_lim_node=getattr(device, species + "_ion_site_limit_m3"),
                P_other_node=populations[1 - index] if device.ion_site_contract == "shared_finite_sites" else None,
                drift_sign=sign,
            )
            if not np.all(tangent.differentiable_faces):
                raise ValueError("ionic AC tangent is undefined at a retained constitutive kink")
            perturbation = densities[index]
            other = densities[1 - index] if device.ion_species_count == 2 else np.zeros_like(perturbation)
            interior = (tangent.potential_left_derivative * phi[device.ion_node_indices[:-1]]
                        + tangent.potential_right_derivative * phi[device.ion_node_indices[1:]]
                        + tangent.density_left_derivative * perturbation[:-1] + tangent.density_right_derivative * perturbation[1:]
                        + tangent.partner_left_derivative * other[:-1] + tangent.partner_right_derivative * other[1:])
            if ion_mode == "frozen":
                interior = np.zeros_like(interior)
            full: np.ndarray = np.zeros(N + 1, dtype=complex)
            full[device.ion_node_indices[:-1] + 1] = interior
            balance = -np.diff(np.r_[0.0j, interior, 0.0j]) - 1j * omega * device.ion_geometry.cell_measures_m * perturbation
            ion_particles[species], ion_balance[species], base_profiles[species] = full, balance, density
            ion_inventory[species] = complex(np.sum(device.ion_geometry.cell_measures_m * perturbation))
            charge[device.ion_node_indices] += sign * Q * device.ion_geometry.cell_measures_m * perturbation
            charge_balance[device.ion_node_indices] += sign * Q * balance
            conduction += sign * Q * full
            ionic_conduction += sign * Q * full
            reference = getattr(device, species + "_ion_reference_m3")
            base_ion_charge += sign * Q * (device.project_ion_density(density) - device.project_ion_density(reference))
        internal_D = -component.poisson.face_capacitance_F_m2 * np.diff(phi)
        D = np.r_[internal_D[0] - charge[0], internal_D, internal_D[-1] + charge[-1]]
        displacement = 1j * omega * D
        total = conduction + displacement
        gauss = D[:-1] - D[1:] + charge
        identity = total[:-1] - total[1:] - charge_balance - 1j * omega * gauss
        enforced: np.ndarray = np.ones(device.state_size, dtype=bool)
        if ion_mode == "frozen":
            enforced[3 * N:] = False
        matrix = self.jacobian + 1j * omega * self.storage_jacobian
        rhs: np.ndarray = np.zeros(device.state_size, dtype=complex)
        rhs[N - 1] = -component.voltage_polarity * voltage
        denominator = abs(matrix) @ abs(fixed_x) + abs(rhs)
        fixed_errors = np.divide(abs(fixed_residual), denominator, out=np.zeros_like(denominator), where=denominator != 0.0)
        original_partial, _drive = self._moving_voltage_partial()
        moving_rhs = -original_partial * voltage + 1j * omega * (self.storage_jacobian @ self.reference_direction_per_V) * voltage
        residual = matrix @ moving_x - moving_rhs
        moving_denominator = abs(matrix) @ abs(moving_x) + abs(moving_rhs)
        moving_errors = np.divide(abs(residual), moving_denominator, out=np.zeros_like(moving_denominator), where=moving_denominator != 0.0)
        errors = np.maximum(fixed_errors, moving_errors)
        positive_electrode = 0 if component.voltage_polarity > 0.0 else -1
        input_current = component.voltage_polarity * total[positive_electrode]
        bound = (Q * (float(np.sum(abs(carrier_balance))) + sum(float(np.sum(abs(v))) for v in ion_balance.values()))
                 + omega * float(np.sum(abs(gauss)))) / abs(voltage)
        if _source_pins(field_transport=component.field_transport is not None) != self.source_sha256:
            raise RuntimeError("ionic AC physical source changed during observation")
        result: dict[str, object] = {
            "frequency_Hz": frequency, "voltage_phasor_V": voltage, "ion_mode": ion_mode,
            "coordinate_form": form, "coordinate_representation": form, "solve_coordinate_perturbation": solve_coordinates,
            "coordinate_lift_per_V": self.coordinate_lift_per_V,
            "fixed_coordinate_perturbation": fixed_x, "moving_coordinate_perturbation": moving_x,
            "reference_direction_per_V": self.reference_direction_per_V,
            "potential_perturbation_V": phi, "electric_field_perturbation_V_m": -np.diff(phi) / np.diff(component.nodes_m),
            "electron_density_perturbation_m3": dn, "hole_density_perturbation_m3": dp,
            "ion_density_perturbation_m3": dict(zip(device.ion_species_names, densities)),
            "ion_inventory_perturbation_m2": ion_inventory, "carrier_particle_fluxes_m2_s": carrier,
            "ion_particle_fluxes_m2_s": ion_particles, "carrier_balance_m2_s": carrier_balance,
            "ion_balance_m2_s": ion_balance, "charge_inventory_perturbation_C_m2": charge,
            "conduction_current_density_A_m2": conduction, "displacement_current_density_A_m2": displacement,
            "ionic_conduction_current_density_A_m2": ionic_conduction,
            "total_current_density_A_m2": total, "terminal_current_density_A_m2": total[[0, -1]],
            "terminal_conduction_current_density_A_m2": conduction[[0, -1]],
            "terminal_displacement_current_density_A_m2": displacement[[0, -1]],
            "input_current_density_A_m2": input_current, "admittance_density_S_m2": input_current / voltage,
            "maxwell_spread_A_m2_V": float(np.max(abs(total - total[0]))) / abs(voltage),
            "current_scale_A_m2_V": float(np.max(abs(total))) / abs(voltage),
            "residual_balance_bound_A_m2_V": bound, "gauss_defect_C_m2": gauss,
            "maxwell_identity_error_A_m2": identity, "original_residual": residual,
            "original_fixed_residual": fixed_residual,
            "original_rows_enforced": tuple(int(index) for index in np.flatnonzero(enforced)), "original_componentwise_backward_error": float(np.max(errors[enforced])),
            "original_ion_force_residual": {name: residual[block] for name, block in zip(device.ion_species_names, device.ion_species_slices)},
            "base_ion_density_m3": base_profiles, "base_ion_charge_density_C_m3": base_ion_charge,
            "compiled_identity": device.identity, "working_state_digest": self.working_state_digest,
            "working_state_identity": self.working_state_identity, "source_sha256": dict(self.source_sha256),
            "physical_certificate_status": "not_evaluated",
        }
        profiles = declared_immobile_profiles(device)
        if profiles:
            result["immobile_profile_binding"] = immobile_profile_binding(device)
            result["immobile_density_perturbation_relative_per_V"] = {
                species: float(np.max(abs(densities[device.ion_species_names.index(species)]) / target)) / abs(voltage)
                for species, target in profiles.items()}
            result["immobile_base_profile_relative_error"] = {
                species: float(np.max(abs(populations[device.ion_species_names.index(species)] - target) / target))
                for species, target in profiles.items()}
        return result


def build_ionic_small_signal(compiled: CompiledIonicDevice1D, state: IonicState1D, *, generation_scale: float = 0.0, coordinate_form: str = "moving_boundary_lift_v1") -> IonicSmallSignal1D:
    if not isinstance(compiled, CompiledIonicDevice1D) or not isinstance(state, IonicState1D):
        raise ValueError("a complete compiled ionic device and bound operating state are required")
    return IonicSmallSignal1D(Ionic1DProblem(compiled, state.voltage_V, generation_scale), state, coordinate_form)
