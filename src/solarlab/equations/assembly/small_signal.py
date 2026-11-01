"""Small-signal (AC) analysis on the assembled ionic 1D systems (decision Q3).

Phasors use exp(+i*omega*t). The linearization is the assembled ``J`` and
``M = -S_u`` at an admitted operating state; nothing is assembled twice.
The analysis-level row operations are explicit, applied in a fixed order,
and recorded with their identities in every :class:`AssembledACSystem`:

1. ``contact_drive`` / ``known_contact_coordinates``: the voltage drive
   ``-polarity * V`` on the right Gauss row, and the prescribed contact
   potential and fixed-reservoir carrier coordinates.
2. ``moving_boundary_lift`` (moving-lift form only): the drive becomes the
   grouped ``-(F_V + J c) V`` and rate ``K (c - r) V`` of the coordinate lift
   ``c``, and every prescribed coordinate becomes zero.
3. ``frozen_ion_rows`` (frozen ions): ion rows become identity rows with no
   storage and zero drive, and are left out of the enforced original rows.
4. ``zero_hz_inventory_rows`` / ``zero_hz_immobile_profile_rows`` (0 Hz,
   mobile ions): one row per connected mobile species becomes its physical
   inventory gradient; every row of a zero-diffusivity species becomes its
   profile gradient.

Every number reproduces the original ``IonicSmallSignal1D`` and
``MultipleIonicSmallSignal1D`` arithmetic in its written order (pinned bit for
bit by tests/unit/assembly). Content identities replace in-module source pins
(Q6): the operator identity binds the assembled system, the working state and
the coordinate form.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem, AssembledMultipleIonic1DProblem
from solarlab.equations.assembly.quasi_fermi import quasi_fermi_potential_tangents
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, IonicState1D
from solarlab.equations.ionic_closure import declared_immobile_profiles, immobile_profile_binding
from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D, MultipleIonicState1D
from solarlab.physics.ions import DIFFUSION_ONLY_STERIC_V1, ion_face_flux_jacobian

__all__ = [
    "AssembledACSystem", "AssembledIonicSmallSignal1D", "AssembledMultipleIonicSmallSignal1D",
    "RowOperation", "assembled_ionic_small_signal", "assembled_multiple_ionic_small_signal",
]

SCHEMA = "solarlab.assembled_small_signal.v1"
MOVING_LIFT = "moving_boundary_lift_v1"
FIXED_REFERENCE = "fixed_reference_v1"
FORMS = frozenset({MOVING_LIFT, FIXED_REFERENCE})

Vector: TypeAlias = NDArray[np.float64]
ComplexVector: TypeAlias = NDArray[np.complex128]


def _frequency_voltage(frequency_Hz: float, voltage_phasor_V: complex, ion_mode: str) -> tuple[float, complex]:
    frequency, voltage = float(frequency_Hz), complex(voltage_phasor_V)
    if not math.isfinite(frequency) or frequency < 0.0:
        raise ValueError("frequency must be finite nonnegative Hz")
    if not np.isfinite(voltage) or voltage == 0.0:
        raise ValueError("voltage phasor must be finite and nonzero; 1 V denotes a derivative")
    if ion_mode not in {"mobile", "frozen"}:
        raise ValueError("ion_mode must explicitly select mobile or frozen ions")
    return frequency, voltage


def _sum_complex(values: ArrayLike) -> complex:
    array = np.asarray(values, dtype=complex)
    return complex(math.fsum(array.real.ravel().tolist()), math.fsum(array.imag.ravel().tolist()))


def _digest(values: ArrayLike) -> str:
    """Content digest of a real or complex vector (binary64 words)."""
    array = np.ascontiguousarray(values)
    return sha256(array.dtype.str.encode() + array.tobytes()).hexdigest()


def _complex_word(value: complex) -> list[str]:
    return [float(value.real).hex(), float(value.imag).hex()]


@dataclass(frozen=True)
class RowOperation:
    """One declared analysis-level operation on the assembled ``J``, ``M`` and drive."""

    kind: str
    rows: tuple[int, ...]
    detail: str = ""

    @property
    def identity(self) -> str:
        return sha256(json.dumps([SCHEMA, self.kind, list(self.rows), self.detail],
                                 separators=(",", ":")).encode()).hexdigest()


def _operation(kind: str, rows: ArrayLike, **detail: Any) -> RowOperation:
    indices = tuple(int(row) for row in np.asarray(rows, dtype=np.intp).ravel())
    return RowOperation(kind, indices, json.dumps(detail, sort_keys=True, separators=(",", ":")))


@dataclass(frozen=True)
class AssembledACSystem:
    """One amplitude-bound frequency system and the row operations that formed it.

    ``rhs``/``fixed_coordinates`` already include ``voltage_phasor_V``.
    ``identity`` binds the operator, frequency, phasor, ion mode, coordinate
    form and every operation identity, in application order.
    """

    frequency_Hz: float
    voltage_phasor_V: complex
    ion_mode: str
    jacobian: csc_matrix
    storage_jacobian: csc_matrix
    rhs: ComplexVector
    rhs_rate: ComplexVector
    fixed_coordinates: dict[int, complex]
    constraint_rows: Mapping[str, Any]
    original_rows_enforced: tuple[int, ...]
    coordinate_form: str
    coordinate_lift_per_V: Vector
    immobile_constraint_rows: dict[str, tuple[int, ...]] = field(default_factory=dict)
    row_operations: tuple[RowOperation, ...] = ()
    operator_identity: str = ""

    @property
    def identity(self) -> str:
        record = [SCHEMA, self.operator_identity, float(self.frequency_Hz).hex(),
                  _complex_word(self.voltage_phasor_V), self.ion_mode, self.coordinate_form,
                  [operation.identity for operation in self.row_operations]]
        return sha256(json.dumps(record, separators=(",", ":")).encode()).hexdigest()

    @property
    def coordinate_representation(self) -> str:
        return self.coordinate_form

    @property
    def rhs_per_V(self) -> ComplexVector:
        result: ComplexVector = self.rhs / self.voltage_phasor_V
        return result

    @property
    def rhs_rate_per_V(self) -> ComplexVector:
        result: ComplexVector = self.rhs_rate / self.voltage_phasor_V
        return result

    @property
    def fixed_coordinates_per_V(self) -> dict[int, complex]:
        return {index: value / self.voltage_phasor_V for index, value in self.fixed_coordinates.items()}


def _system_record(system: AssembledACSystem | None, operator_identity: str, frequency: float, voltage: complex,
                   ion_mode: str, form: str) -> dict[str, object]:
    """The identities an observation records: its frequency system and that system's row operations."""
    if system is None:
        return {}
    if (system.operator_identity != operator_identity or system.frequency_Hz != frequency
            or system.voltage_phasor_V != voltage or system.ion_mode != ion_mode or system.coordinate_form != form):
        raise ValueError("the AC system belongs to another operator, frequency, drive or representation")
    return {"ac_system_identity": system.identity,
            "ac_row_operations": tuple((operation.kind, operation.identity) for operation in system.row_operations)}


def _contact_drive(device: CompiledIonicDevice1D | CompiledMultipleIonicDevice1D, voltage: complex,
                   operations: list[RowOperation]) -> tuple[ComplexVector, dict[int, complex]]:
    """Operation 1: the right-contact Gauss drive and the known contact coordinates.

    The carrier contact values keep each original's own expression: the
    single-domain operator divides the Python ``complex`` drive, the
    two-domain operator the NumPy ``complex128`` one.
    """
    N, component = device.node_count, device.electronic_component
    rhs: ComplexVector = np.zeros(device.state_size, dtype=complex)
    rhs[N - 1] = -component.voltage_polarity * voltage
    phi_right = complex(rhs[N - 1])
    known: dict[int, complex] = {0: 0.0j, N - 1: phi_right}
    for offset in (N, 2 * N):
        known[offset] = 0.0j
        known[offset + N - 1] = (-rhs[N - 1] / component.thermal_voltage_V
                                 if isinstance(device, CompiledMultipleIonicDevice1D)
                                 else -phi_right / component.thermal_voltage_V)
    operations.append(_operation("contact_drive", [N - 1], value=_complex_word(phi_right)))
    operations.append(_operation("known_contact_coordinates", sorted(known),
                                 values={str(row): _complex_word(value) for row, value in sorted(known.items())}))
    return rhs, known


def _moving_lift(lift: Vector, reference: Vector, partial: Vector, drive: Vector, voltage: complex,
                 known: dict[int, complex], operations: list[RowOperation]) -> tuple[
        ComplexVector, ComplexVector, dict[int, complex]]:
    """Operation 2: the grouped lift drive and rate, and zero prescribed coordinates."""
    rhs: ComplexVector = -partial * voltage
    rhs_rate: ComplexVector = drive * voltage
    operations.append(_operation("moving_boundary_lift", np.arange(partial.size), lift=_digest(lift),
                                 reference_direction=_digest(reference), drive_per_V=_digest(partial),
                                 rate_per_V=_digest(drive)))
    return rhs, rhs_rate, {index: 0.0j for index in known}


@dataclass(frozen=True)
class AssembledIonicSmallSignal1D:
    """Linearization of the assembled single-domain rows about an admitted state.

    The surface of ``IonicSmallSignal1D``: ``jacobian``/``storage_jacobian``,
    the reference direction and coordinate lift, ``system``,
    ``original_residual`` and ``observe``. Bound-field consistency is checked
    here; the experiment admits the DC state and its physical gates.
    """

    problem: AssembledIonic1DProblem
    operating_state: IonicState1D
    coordinate_form: str = MOVING_LIFT
    jacobian: csc_matrix = field(init=False, repr=False)
    storage_jacobian: csc_matrix = field(init=False, repr=False)
    reference_direction_per_V: Vector = field(init=False, repr=False)
    coordinate_lift_per_V: Vector = field(init=False, repr=False)
    working_state_digest: str = field(init=False)
    working_state_identity: str = field(init=False)
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.problem, AssembledIonic1DProblem) or not isinstance(self.operating_state, IonicState1D):
            raise ValueError("ionic AC requires a full ionic problem and bound operating state")
        if self.coordinate_form not in FORMS:
            raise ValueError("ionic AC requires an explicit supported coordinate representation")
        device, state = self.problem.device, self.operating_state
        if state.device_sha256 != device.identity or state.voltage_V != self.problem.voltage_V:
            raise ValueError("ionic AC state must have the same complete device and operating voltage")
        if any(channel.kind != "fixed_reservoir" for channel in device.electronic_component.channels):
            raise CapabilityError("ionic AC currently requires all fixed-reservoir carrier contacts")
        manifest = device.binding_manifest
        domain, component = manifest.get("ion_domain"), manifest.get("electronic_component")
        if (sha256(device.binding_manifest_json.encode()).hexdigest() != device.identity
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
        component = device.electronic_component
        before, after = component.carrier_reference_energies(0.0), component.carrier_reference_energies(1.0)
        for offset, zero, one in zip((count, 2 * count), before, after):
            direction[offset:offset + count] = -(one - zero) / component.thermal_voltage_V
        direction.setflags(write=False)
        object.__setattr__(self, "reference_direction_per_V", direction)
        lift = np.zeros(device.state_size)
        lift[count - 1] = -component.voltage_polarity
        for offset in (count, 2 * count):
            for node in (0, count - 1):
                lift[offset + node] = -lift[node] / component.thermal_voltage_V + direction[offset + node]
        lift.setflags(write=False)
        object.__setattr__(self, "coordinate_lift_per_V", lift)
        digest = sha256(np.asarray(state.u, dtype=np.float64).tobytes()).hexdigest()
        object.__setattr__(self, "working_state_digest", digest)
        object.__setattr__(self, "working_state_identity", sha256(json.dumps((
            device.identity, float(state.voltage_V).hex(), float(self.problem.generation_scale).hex(), digest,
        )).encode()).hexdigest())
        object.__setattr__(self, "identity", sha256(json.dumps(
            [SCHEMA, "ionic_1d", self.problem.binding.identity, self.working_state_identity, self.coordinate_form],
            separators=(",", ":")).encode()).hexdigest())

    @property
    def device(self) -> CompiledIonicDevice1D:
        return self.problem.device

    @property
    def coordinate_representation(self) -> str:
        return self.coordinate_form

    def system(self, frequency_Hz: float, *, voltage_phasor_V: complex = 1.0, ion_response: str = "mobile",
               ion_mode: str | None = None) -> AssembledACSystem:
        ion_mode = ion_response if ion_mode is None else ion_mode
        frequency, voltage = _frequency_voltage(frequency_Hz, voltage_phasor_V, ion_mode)
        device, N = self.device, self.device.node_count
        J, K = self.jacobian.copy().tolil(), self.storage_jacobian.copy().tolil()
        operations: list[RowOperation] = []
        rhs, known = _contact_drive(device, voltage, operations)
        rhs_rate = np.zeros_like(rhs)
        if self.coordinate_form == MOVING_LIFT:
            partial, drive = self._moving_voltage_partial(self.coordinate_lift_per_V)
            rhs, rhs_rate, known = _moving_lift(self.coordinate_lift_per_V, self.reference_direction_per_V,
                                                partial, drive, voltage, known, operations)
        rows: dict[str, int] = {}
        profile_rows: dict[str, tuple[int, ...]] = {}
        profiles = declared_immobile_profiles(device)
        enforced: NDArray[np.bool_] = np.ones(device.state_size, dtype=bool)
        if ion_mode == "frozen":
            for index in range(3 * N, device.state_size):
                J.rows[index], J.data[index] = [index], [1.0]
                K.rows[index], K.data[index] = [], []
                known[index] = 0.0j
                rhs[index] = rhs_rate[index] = 0.0j
            enforced[3 * N:] = False
            operations.append(_operation("frozen_ion_rows", np.arange(3 * N, device.state_size)))
        elif frequency == 0.0:
            for species, block in zip(device.ion_species_names, device.ion_species_slices):
                if species in profiles:
                    local_rows = tuple(range(cast(int, block.start), cast(int, block.stop)))
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
                row = cast(int, block.start)
                gradient = -np.asarray(self.storage_jacobian[block, :].sum(axis=0)).ravel()
                nonzero = np.flatnonzero(gradient)
                J.rows[row], J.data[row] = nonzero.tolist(), gradient[nonzero].tolist()
                K.rows[row], K.data[row] = [], []
                rhs[row] = rhs_rate[row] = 0.0j
                rows[species] = row
            if rows:
                operations.append(_operation("zero_hz_inventory_rows", sorted(rows.values()), species=rows))
            if profile_rows:
                operations.append(_operation("zero_hz_immobile_profile_rows",
                                             sorted(row for part in profile_rows.values() for row in part),
                                             species={name: list(part) for name, part in profile_rows.items()}))
        return AssembledACSystem(frequency, voltage, ion_mode, J.tocsc(), K.tocsc(), rhs, rhs_rate, known, rows,
                                 tuple(int(index) for index in np.flatnonzero(enforced)), self.coordinate_form,
                                 self.coordinate_lift_per_V, profile_rows, tuple(operations), self.identity)

    def original_residual(self, fixed_perturbation: ArrayLike, frequency_Hz: float, *,
                          voltage_phasor_V: complex = 1.0) -> ComplexVector:
        frequency, voltage = _frequency_voltage(frequency_Hz, voltage_phasor_V, "mobile")
        x = np.asarray(fixed_perturbation, dtype=complex)
        if x.shape != (self.device.state_size,) or not np.all(np.isfinite(x)):
            raise ValueError("ionic AC perturbation must be finite and complete")
        rhs = np.zeros_like(x)
        rhs[self.device.node_count - 1] = -self.device.electronic_component.voltage_polarity * voltage
        result: ComplexVector = (self.jacobian + 2j * math.pi * frequency * self.storage_jacobian) @ x - rhs
        return result

    def _carrier_particles(self, x: ArrayLike, reference_drive: complex = 0.0) -> tuple[ComplexVector, ComplexVector]:
        """Production quasi-Fermi tangent, grouping exact flux homogeneity modes."""
        component, N = self.device.electronic_component, self.device.node_count
        values_x = np.asarray(x)
        u = self.operating_state.u[:3 * N]
        fluxes = self.problem.electronic_problem.fluxes(u)
        transport = None if component.field_transport is None else component.field_transport.evaluate(u[:N])
        state = self.operating_state.electronic_state
        before, after = component.carrier_reference_energies(0.0), component.carrier_reference_energies(1.0)
        values = []
        for offset, density, flux, sign, reference in (
                (N, state.electron_density_m3, fluxes.electron, 1.0, after[0] - before[0]),
                (2 * N, state.hole_density_m3, fluxes.hole, -1.0, after[1] - before[1])):
            drive = sign * np.diff(u[offset:offset + N])
            left_phi, _ = quasi_fermi_potential_tangents(flux, drive, sign, component.thermal_voltage_V)
            if transport is not None:
                diffusivity = (transport.electron_diffusivity_m2_s if offset == N
                               else transport.hole_diffusivity_m2_s)
                d_diffusivity = (transport.electron_d_diffusivity_d_field if offset == N
                                 else transport.hole_d_diffusivity_d_field)
                # Same antisymmetric CT chain as the steady Jacobian. A moving
                # carrier energy reference alone has no electric-field drive.
                left_phi = left_phi + (flux.flux / diffusivity) * d_diffusivity / np.diff(component.nodes_m)
            left_z = sign * flux.left_density_derivative * density[:-1]
            values.append(left_phi * (values_x[:N - 1] - values_x[1:N])
                          + left_z * (values_x[offset:offset + N - 1] - values_x[offset + 1:offset + N])
                          + sign * flux.flux * (values_x[offset + 1:offset + N]
                                                + (values_x[1:N] + reference * reference_drive) / component.thermal_voltage_V))
        return values[0], values[1]

    def _moving_voltage_partial(self, direction: ArrayLike | None = None) -> tuple[Vector, Vector]:
        """Grouped actual F_V+J*direction and K*(c-direction), before cancellation.

        Only the declared zero ion-reference/ion-lift path is used. The potential
        lift's ionic endpoint force remains through the assembled Jion*lift.
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
        result: Vector = np.zeros(device.state_size)
        storage_drive: Vector = np.zeros(device.state_size)
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

    def observe(self, fixed_perturbation: ArrayLike | AssembledACSystem, frequency_Hz: float | ArrayLike, *,
                voltage_phasor_V: complex = 1.0, ion_mode: str = "mobile", coordinate_form: str | None = None,
                system: AssembledACSystem | None = None) -> dict[str, object]:
        """Physical response of one solve; with ``system`` (or the system first) its identities are recorded."""
        descriptor = system
        if isinstance(fixed_perturbation, AssembledACSystem):
            descriptor = fixed_perturbation
            fixed_perturbation, frequency_Hz, ion_mode = cast(ArrayLike, frequency_Hz), descriptor.frequency_Hz, descriptor.ion_mode
            coordinate_form = descriptor.coordinate_form
        frequency, voltage = _frequency_voltage(cast(float, frequency_Hz), voltage_phasor_V, ion_mode)
        device, N = self.device, self.device.node_count
        x = np.asarray(fixed_perturbation, dtype=complex)
        form = self.coordinate_form if coordinate_form is None else coordinate_form
        if form not in FORMS:
            raise ValueError("unknown AC perturbation coordinate representation")
        if x.shape != (device.state_size,) or not np.all(np.isfinite(x)):
            raise ValueError("ionic AC perturbation must be finite and complete")
        solve_coordinates = x.copy()
        moving_x = x + self.reference_direction_per_V * voltage
        if form == MOVING_LIFT:
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
        if form == MOVING_LIFT:
            before, after = component.carrier_reference_energies(0.0), component.carrier_reference_energies(1.0)
            lift = self.coordinate_lift_per_V
            dn = n * (x[:N] / component.thermal_voltage_V + x[N:2 * N]) + n * ((lift[:N] + after[0] - before[0]) / component.thermal_voltage_V + lift[N:2 * N]) * voltage
            dp = -p * (x[:N] / component.thermal_voltage_V + x[2 * N:3 * N]) - p * ((lift[:N] + after[1] - before[1]) / component.thermal_voltage_V + lift[2 * N:3 * N]) * voltage
        mass = device.ion_coordinate_jacobian_m3(state.u, voltage_V=state.voltage_V)
        coordinates = np.stack([x[block] for block in device.ion_species_slices], axis=1)
        densities = np.sum(mass * coordinates[:, None, :], axis=2).T
        fn, fp = self._carrier_particles(x)
        if form == MOVING_LIFT:
            known_n, known_p = self._carrier_particles(self.coordinate_lift_per_V, 1.0)
            fn, fp = fn + known_n * voltage, fp + known_p * voltage
        rate = self.problem.electronic_problem.recombination_rate(n, p)
        assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
        pair = -component.geometry.cell_measures_m * (rate.d_rate_dn * dn + rate.d_rate_dp * dp)
        carrier: ComplexVector = np.empty((2, N + 1), dtype=complex)
        for index, (flux, density) in enumerate(((fn, dn), (fp, dp))):
            carrier[index, 1:-1] = flux
            carrier[index, 0] = flux[0] - pair[0] + 1j * omega * component.geometry.cell_measures_m[0] * density[0]
            carrier[index, -1] = flux[-1] + pair[-1] - 1j * omega * component.geometry.cell_measures_m[-1] * density[-1]
        carrier_balance = -np.diff(carrier, axis=1) + pair - 1j * omega * component.geometry.cell_measures_m * np.stack((dn, dp))
        populations = ((state.positive_ion_density_m3,) if device.ion_species_count == 1
                       else (state.positive_ion_density_m3, state.negative_ion_density_m3))
        ion_particles, ion_balance, base_profiles, ion_inventory = {}, {}, {}, {}
        charge = Q * component.geometry.cell_measures_m * (dp - dn)
        charge_balance = Q * (carrier_balance[1] - carrier_balance[0])
        conduction = Q * (carrier[1] - carrier[0])
        ionic_conduction: ComplexVector = np.zeros(N + 1, dtype=complex)
        base_ion_charge = np.zeros(N)
        for index, (species, density, block) in enumerate(zip(device.ion_species_names, populations, device.ion_species_slices)):
            sign = 1.0 if index == 0 else -1.0
            tangent = ion_face_flux_jacobian(
                state.electronic_state.potential_V[device.ion_node_indices], density, np.diff(device.ion_geometry.nodes_m),
                getattr(device, species + "_ion_diffusivity_faces_m2_s"), component.thermal_voltage_V,
                getattr(device, species + "_ion_site_limit_m3")[:-1],
                steric_diffusion_only=device.steric_model == DIFFUSION_ONLY_STERIC_V1,
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
            full: ComplexVector = np.zeros(N + 1, dtype=complex)
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
        enforced: NDArray[np.bool_] = np.ones(device.state_size, dtype=bool)
        if ion_mode == "frozen":
            enforced[3 * N:] = False
        matrix = self.jacobian + 1j * omega * self.storage_jacobian
        rhs: ComplexVector = np.zeros(device.state_size, dtype=complex)
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
            "original_rows_enforced": tuple(int(index) for index in np.flatnonzero(enforced)),
            "original_componentwise_backward_error": float(np.max(errors[enforced])),
            "original_ion_force_residual": {name: residual[block] for name, block in zip(device.ion_species_names, device.ion_species_slices)},
            "base_ion_density_m3": base_profiles, "base_ion_charge_density_C_m3": base_ion_charge,
            "compiled_identity": device.identity, "working_state_digest": self.working_state_digest,
            "working_state_identity": self.working_state_identity, "analysis_identity": self.identity,
            "physical_certificate_status": "not_evaluated",
        }
        result.update(_system_record(descriptor, self.identity, frequency, voltage, ion_mode, form))
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


def _scope(device: CompiledMultipleIonicDevice1D, ion_mode: str = "mobile") -> None:
    """The two-domain AC scope (as ``multiple_ionic_small_signal``)."""
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


@dataclass(frozen=True)
class AssembledMultipleIonicSmallSignal1D:
    """Mobile-positive AC of two separated blocking domains on the assembled rows.

    The surface of ``MultipleIonicSmallSignal1D``. The per-domain operators
    supply mathematical tangents and local observations only.
    """

    problem: AssembledMultipleIonic1DProblem
    operating_state: MultipleIonicState1D
    coordinate_form: str = MOVING_LIFT
    jacobian: csc_matrix = field(init=False, repr=False)
    storage_jacobian: csc_matrix = field(init=False, repr=False)
    reference_direction_per_V: Vector = field(init=False, repr=False)
    coordinate_lift_per_V: Vector = field(init=False, repr=False)
    working_state_digest: str = field(init=False)
    working_state_identity: str = field(init=False)
    identity: str = field(init=False)
    _local_operators: tuple[AssembledIonicSmallSignal1D, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.problem, AssembledMultipleIonic1DProblem):
            raise ValueError("the full production multiple-domain problem is required")
        device, state = self.problem.device, self.operating_state
        _scope(device)
        if self.coordinate_form not in FORMS:
            raise ValueError("an explicit supported AC coordinate representation is required")
        device.validate_state(state)
        if state.voltage_V != self.problem.voltage_V:
            raise ValueError("AC state and production operating voltage must agree")
        local = tuple(AssembledIonicSmallSignal1D(
            AssembledIonic1DProblem(domain, state.voltage_V, self.problem.generation_scale),
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
        digest = sha256(np.asarray(state.u, dtype=np.float64).tobytes()).hexdigest()
        object.__setattr__(self, "working_state_digest", digest)
        object.__setattr__(self, "working_state_identity", sha256(json.dumps((
            device.identity, float(state.voltage_V).hex(), float(self.problem.generation_scale).hex(), digest,
        )).encode()).hexdigest())
        object.__setattr__(self, "identity", sha256(json.dumps(
            [SCHEMA, "multiple_ionic_1d", self.problem.binding.identity, self.working_state_identity,
             self.coordinate_form, [operator.identity for operator in local]],
            separators=(",", ":")).encode()).hexdigest())

    @property
    def device(self) -> CompiledMultipleIonicDevice1D:
        return self.problem.device

    @property
    def coordinate_representation(self) -> str:
        return self.coordinate_form

    def _moving_voltage_partial(self, direction: ArrayLike | None = None) -> tuple[Vector, Vector]:
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
               ion_mode: str = "mobile") -> AssembledACSystem:
        _scope(self.device, ion_mode)
        frequency, voltage = _frequency_voltage(frequency_Hz, voltage_phasor_V, "mobile")
        device = self.device
        J, K = self.jacobian.copy().tolil(), self.storage_jacobian.copy().tolil()
        operations: list[RowOperation] = []
        rhs, known = _contact_drive(device, voltage, operations)
        rhs_rate = np.zeros_like(rhs)
        if self.coordinate_form == MOVING_LIFT:
            partial, drive = self._moving_voltage_partial(self.coordinate_lift_per_V)
            rhs, rhs_rate, known = _moving_lift(self.coordinate_lift_per_V, self.reference_direction_per_V,
                                                partial, drive, voltage, known, operations)
        constraints: dict[str, dict[str, int]] = {}
        if frequency == 0.0:
            for name, blocks in device.ion_species_slices.items():
                constraints[name] = {}
                for species, block in blocks.items():
                    row = cast(int, block.start)
                    gradient = -np.asarray(self.storage_jacobian[block, :].sum(axis=0)).ravel()
                    nonzero = np.flatnonzero(gradient)
                    J.rows[row], J.data[row] = nonzero.tolist(), gradient[nonzero].tolist()
                    K.rows[row], K.data[row] = [], []
                    rhs[row] = rhs_rate[row] = 0.0j
                    constraints[name][species] = row
            operations.append(_operation("zero_hz_inventory_rows",
                                         sorted(row for part in constraints.values() for row in part.values()),
                                         domains=constraints))
        return AssembledACSystem(frequency, voltage, "mobile", J.tocsc(), K.tocsc(), rhs, rhs_rate, known,
                                 constraints, tuple(range(device.state_size)), self.coordinate_form,
                                 self.coordinate_lift_per_V, {}, tuple(operations), self.identity)

    def original_residual(self, fixed_perturbation: ArrayLike, frequency_Hz: float,
                          *, voltage_phasor_V: complex = 1.0) -> ComplexVector:
        frequency, voltage = _frequency_voltage(frequency_Hz, voltage_phasor_V, "mobile")
        x = np.asarray(fixed_perturbation, dtype=complex)
        if x.shape != (self.device.state_size,) or not np.all(np.isfinite(x)):
            raise ValueError("a finite complete two-domain perturbation is required")
        rhs = np.zeros_like(x)
        rhs[self.device.node_count - 1] = -self.device.electronic_component.voltage_polarity * voltage
        result: ComplexVector = (self.jacobian + 2j * math.pi * frequency * self.storage_jacobian) @ x - rhs
        return result

    def observe(self, perturbation: ArrayLike, frequency_Hz: float, *, voltage_phasor_V: complex = 1.0,
                ion_mode: str = "mobile", coordinate_form: str | None = None,
                system: AssembledACSystem | None = None) -> dict[str, Any]:
        """Physical response of one solve; with ``system`` its identities are recorded."""
        _scope(self.device, ion_mode)
        frequency, voltage = _frequency_voltage(frequency_Hz, voltage_phasor_V, "mobile")
        device, N = self.device, self.device.node_count
        form = self.coordinate_form if coordinate_form is None else coordinate_form
        x = np.asarray(perturbation, dtype=complex)
        if form not in FORMS:
            raise ValueError("unknown AC coordinate representation")
        if x.shape != (device.state_size,) or not np.all(np.isfinite(x)):
            raise ValueError("a finite complete two-domain perturbation is required")
        fixed_x = x if form == FIXED_REFERENCE else x + (
            self.coordinate_lift_per_V - self.reference_direction_per_V) * voltage
        moving_x = x + (self.reference_direction_per_V if form == FIXED_REFERENCE
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
        ionic_current: ComplexVector = np.zeros(N + 1, dtype=complex)
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
        fixed_rhs: ComplexVector = np.zeros(device.state_size, dtype=complex)
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
            "working_state_digest": self.working_state_digest, "analysis_identity": self.identity,
            "physical_certificate_status": "not_evaluated",
            **_system_record(system, self.identity, frequency, voltage, "mobile", form)}


def assembled_ionic_small_signal(compiled: CompiledIonicDevice1D, state: IonicState1D, *,
                                 generation_scale: float = 0.0,
                                 coordinate_form: str = MOVING_LIFT) -> AssembledIonicSmallSignal1D:
    """The AC operator of one admitted single-domain state (as ``build_ionic_small_signal``)."""
    if not isinstance(compiled, CompiledIonicDevice1D) or not isinstance(state, IonicState1D):
        raise ValueError("a complete compiled ionic device and bound operating state are required")
    return AssembledIonicSmallSignal1D(AssembledIonic1DProblem(compiled, state.voltage_V, generation_scale),
                                       state, coordinate_form)


def assembled_multiple_ionic_small_signal(compiled: CompiledMultipleIonicDevice1D, state: MultipleIonicState1D, *,
                                          generation_scale: float = 0.0,
                                          coordinate_form: str = MOVING_LIFT) -> AssembledMultipleIonicSmallSignal1D:
    """The AC operator of one admitted two-domain state (as ``build_multiple_ionic_small_signal``)."""
    _scope(compiled)
    if not isinstance(state, MultipleIonicState1D):
        raise ValueError("a complete bound multiple-domain operating state is required")
    return AssembledMultipleIonicSmallSignal1D(AssembledMultipleIonic1DProblem(compiled, state.voltage_V,
                                                                               generation_scale),
                                               state, coordinate_form)
