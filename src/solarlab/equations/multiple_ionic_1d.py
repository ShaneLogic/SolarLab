"""Two explicitly separated blocking ion domains on one electronic mesh.

Each domain retains the integrated finite-site map, constitutive law and
physical storage. This is an assembly binding, without a steady/time driver.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import block_diag, bmat, csc_matrix

from solarlab.contracts import frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem, State1D, Vector
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, Ionic1DProblem, IonicState1D
from solarlab.physics.electrostatics.charge import ChargeEntry, ChargeLedger

__all__ = ["CompiledMultipleIonicDevice1D", "MultipleIonic1DProblem", "MultipleIonicState1D"]


def _finite_scalar(value: float, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True)
class MultipleIonicState1D:
    """Full physical trial coordinates and both identified component states."""

    u: Vector
    electronic_state: State1D
    domain_ids: tuple[str, ...]
    domain_component_states: tuple[IonicState1D, ...]
    device_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "u", frozen_array(self.u))
        object.__setattr__(self, "domain_ids", tuple(self.domain_ids))
        object.__setattr__(self, "domain_component_states", tuple(self.domain_component_states))
        if (not isinstance(self.electronic_state, State1D)
                or not all(isinstance(state, IonicState1D) for state in self.domain_component_states)
                or len(self.domain_ids) != 2 or len(set(self.domain_ids)) != 2
                or not all(isinstance(name, str) and name for name in self.domain_ids)
                or len(self.domain_component_states) != 2):
            raise ValueError("the state requires exactly two distinct ordered domains")
        electronic_size = self.electronic_state.u.size
        expected = np.r_[self.electronic_state.u, *[state.u[electronic_size:] for state in self.domain_component_states]]
        if not isinstance(self.device_sha256, str) or not self.device_sha256 or not np.array_equal(self.u, expected):
            raise ValueError("multiple-domain state shape/coordinates must bind every component")
        for state in self.domain_component_states:
            if (state.voltage_V != self.voltage_V
                    or state.electronic_state.device_sha256 != self.electronic_state.device_sha256
                    or not np.array_equal(state.electronic_state.u, self.electronic_state.u)):
                raise ValueError("all ionic domains must share the same electronic state and voltage")

    @property
    def voltage_V(self) -> float:
        return self.electronic_state.voltage_V

    @property
    def domain_inventories_m2(self) -> dict[str, dict[str, float]]:
        return {name: state.domain_inventories_m2
                for name, state in zip(self.domain_ids, self.domain_component_states)}


@dataclass(frozen=True)
class CompiledMultipleIonicDevice1D:
    """Source-bound composition of two local ionic components, never ion-free."""

    device_id: str
    sha256: str
    original_prepared_sha256: str
    electronic_component: CompiledDevice1D
    domains: tuple[CompiledIonicDevice1D, ...]
    binding_manifest_json: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "domains", tuple(self.domains))
        if any(not isinstance(value, str) or not value for value in (
            self.device_id, self.sha256, self.original_prepared_sha256, self.binding_manifest_json
        )):
            raise ValueError("multiple-domain device and source identities must be nonempty strings")
        if (not isinstance(self.electronic_component, CompiledDevice1D)
                or not all(isinstance(domain, CompiledIonicDevice1D) for domain in self.domains)
                or len(self.domains) != 2 or len(set(self.domain_ids)) != 2):
            raise ValueError("exactly two distinct blocking ion domains are required")
        if self.domains[0].ion_node_indices[-1] >= self.domains[1].ion_node_indices[0]:
            raise ValueError("domains must be ordered and separated by an inactive electrical region")
        for domain in self.domains:
            if domain.electronic_component is not self.electronic_component:
                raise ValueError("both domains must use the same actual compiled electronic component")
        manifest = self.binding_manifest
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if (self.sha256 != hashlib.sha256(encoded.encode()).hexdigest()
                or manifest.get("original_prepared_sha256") != self.original_prepared_sha256
                or manifest.get("electronic_component_sha256") != self.electronic_component.identity
                or manifest.get("state_size") != self.state_size):
            raise ValueError("multiple-domain source, component or state identity is inconsistent")
        declarations = manifest.get("original_ionic_inputs")
        records = manifest.get("domains")
        if not isinstance(declarations, list) or not isinstance(records, list) or len(records) != 2:
            raise ValueError("all original ion declarations and both domain records are required")
        for domain, record in zip(self.domains, records):
            if not isinstance(record, dict):
                raise ValueError("invalid domain binding record")
            if (record.get("layer") != domain.domain_layer_id
                    or record.get("endpoints") != ["blocking", "blocking"]
                    or record.get("component_sha256") != domain.identity
                    or record.get("component_manifest") != domain.binding_manifest
                    or record.get("reference_inventories_m2") != self.reference_inventories_m2[domain.domain_layer_id]):
                raise ValueError("domain identities, endpoints or inventories differ from their binding")
            original = [entry for entry in declarations if isinstance(entry, dict)
                        and entry.get("layer") == domain.domain_layer_id]
            local = [entry for entry in cast(list[dict[str, Any]], domain.binding_manifest["original_ionic_inputs"])
                     if entry["layer"] == domain.domain_layer_id]
            if len(original) != 1 or original != local:
                raise ValueError("an isolated domain must preserve all its original ionic parameters")
        for entry in declarations:
            if not isinstance(entry, dict):
                raise ValueError("invalid original ionic declaration")
            if entry.get("layer") not in self.domain_ids and any(
                entry.get(name) != 0.0 for name in ("P0", "P0_neg", "D_ion", "D_ion_neg")
            ):
                raise ValueError("undeclared ionic populations or mobility cannot be stripped")
        expected_layout = [{"domain": name, "species": species, "start": block.start, "stop": block.stop,
                            "coordinate": "original_finite_site_coordinate"}
                           for name, species_blocks in self.ion_species_slices.items()
                           for species, block in species_blocks.items()]
        if manifest.get("ionic_state_blocks") != expected_layout:
            raise ValueError("every ordered domain/species state block must match the actual coordinates")
        if any(domain.binding_manifest.get("original_ionic_settings") != manifest.get("original_ionic_settings")
               for domain in self.domains):
            raise ValueError("each domain must preserve the original global ionic constitutive flags")

    @property
    def identity(self) -> str:
        return self.sha256

    @property
    def binding_manifest(self) -> dict[str, Any]:
        result = json.loads(self.binding_manifest_json)
        if not isinstance(result, dict):
            raise ValueError("multiple-domain manifest must be an object")
        return result

    @property
    def node_count(self) -> int:
        return self.electronic_component.node_count

    @property
    def nodes_m(self) -> Vector:
        return self.electronic_component.nodes_m

    @property
    def domain_ids(self) -> tuple[str, ...]:
        return tuple(domain.domain_layer_id for domain in self.domains)

    @property
    def state_size(self) -> int:
        return 3 * self.node_count + sum(domain.state_size - 3 * self.node_count for domain in self.domains)

    @property
    def domain_slices(self) -> dict[str, slice]:
        offset = 3 * self.node_count
        result: dict[str, slice] = {}
        for domain in self.domains:
            size = domain.state_size - 3 * self.node_count
            result[domain.domain_layer_id] = slice(offset, offset + size)
            offset += size
        return result

    @property
    def ion_species_slices(self) -> dict[str, dict[str, slice]]:
        start = 3 * self.node_count
        result: dict[str, dict[str, slice]] = {}
        for domain in self.domains:
            result[domain.domain_layer_id] = {}
            for species in domain.ion_species_names:
                result[domain.domain_layer_id][species] = slice(start, start + domain.ion_node_count)
                start += domain.ion_node_count
        return result

    @property
    def reference_inventories_m2(self) -> dict[str, dict[str, float]]:
        return {domain.domain_layer_id: {
            species: math.fsum((domain.ion_geometry.cell_measures_m * getattr(domain, species + "_ion_reference_m3")).tolist())
            for species in domain.ion_species_names} for domain in self.domains}

    def _values(self, y: ArrayLike) -> Vector:
        if np.asarray(y).dtype.kind not in {"i", "u", "f"}:
            raise ValueError("multiple-domain coordinates and rates must be real numeric arrays")
        values = np.asarray(y, dtype=float)
        if values.shape != (self.state_size,) or not np.all(np.isfinite(values)):
            raise ValueError("coordinates must contain 3N and all species of both ionic domains")
        return values

    def local_coordinates(self, y: ArrayLike, domain_id: str) -> Vector:
        values = self._values(y)
        if domain_id not in self.domain_ids:
            raise ValueError("requested ion domain is not assembled")
        return np.r_[values[:3 * self.node_count], values[self.domain_slices[domain_id]]]

    def fields_by_domain(self, y: ArrayLike, *, voltage_V: float = 0.0) -> tuple[Vector, Vector, Vector, dict[str, tuple[Vector, ...]]]:
        voltage_V = _finite_scalar(voltage_V, "voltage_V")
        values = self._values(y)
        phi, n, p = self.electronic_component.fields(values[:3 * self.node_count], voltage_V=voltage_V)
        populations = {domain.domain_layer_id: domain.fields_by_species(
            self.local_coordinates(values, domain.domain_layer_id), voltage_V=voltage_V)[3]
            for domain in self.domains}
        return phi, n, p, populations

    def state(self, y: ArrayLike, *, voltage_V: float = 0.0) -> MultipleIonicState1D:
        voltage_V = _finite_scalar(voltage_V, "voltage_V")
        values = self._values(y)
        component = self.electronic_component.state(values[:3 * self.node_count], voltage_V=voltage_V)
        local = tuple(domain.state(self.local_coordinates(values, domain.domain_layer_id), voltage_V=voltage_V)
                      for domain in self.domains)
        return MultipleIonicState1D(values, component, self.domain_ids, local, self.identity)

    def validate_state(self, state: MultipleIonicState1D) -> None:
        """Verify identity, shape and readback of a trial; no physical acceptance."""
        if (not isinstance(state, MultipleIonicState1D) or state.device_sha256 != self.identity
                or state.domain_ids != self.domain_ids):
            raise ValueError("state must bind this complete multiple-domain device and ordered domains")
        expected = self.state(state.u, voltage_V=state.voltage_V)
        for actual, required in zip(state.domain_component_states, expected.domain_component_states):
            if (actual.device_sha256 != required.device_sha256
                    or actual.domain_inventories_m2 != required.domain_inventories_m2
                    or any(not np.array_equal(getattr(actual, name), getattr(required, name))
                           for name in ("u", "positive_ion_density_m3", "negative_ion_density_m3"))):
                raise ValueError("ionic component identities, coordinates or physical inventories disagree")
        for electronic_actual in (state.electronic_state, *[value.electronic_state for value in state.domain_component_states]):
            if (electronic_actual.device_sha256 != expected.electronic_state.device_sha256
                    or any(not np.array_equal(getattr(electronic_actual, name), getattr(expected.electronic_state, name))
                           for name in ("u", "potential_V", "electron_density_m3", "hole_density_m3"))):
                raise ValueError("electronic component identity or physical fields disagree")

    def initial_state(self, *, voltage_V: float = 0.0) -> MultipleIonicState1D:
        voltage_V = _finite_scalar(voltage_V, "voltage_V")
        electronic = self.domains[0].initial_state(voltage_V=voltage_V).u[:3 * self.node_count]
        return self.state(np.r_[electronic, np.zeros(self.state_size - electronic.size)], voltage_V=voltage_V)

    def charge_ledger(self, n: Vector, p: Vector, populations: dict[str, tuple[Vector, ...]]) -> ChargeLedger:
        if set(populations) != set(self.domain_ids):
            raise ValueError("charge requires every declared ionic domain")
        electronic = self.electronic_component.charge_ledger(n, p)
        positive: list[ChargeEntry] = []
        negative: list[ChargeEntry] = []
        for domain in self.domains:
            values = populations[domain.domain_layer_id]
            if len(values) != domain.ion_species_count:
                raise ValueError("charge requires every species of every domain")
            for index, species in enumerate(domain.ion_species_names):
                entry = ChargeEntry(species + "_ion." + domain.domain_layer_id, "ion_charge_projection",
                    1 if species == "positive" else -1, domain.project_ion_density(values[index]),
                    domain.project_ion_density(getattr(domain, species + "_ion_reference_m3")),
                    group="space_charge" if species == "positive" else "negative_ion_space_charge")
                (positive if species == "positive" else negative).append(entry)
        return ChargeLedger(self.node_count, electronic.interface_count,
            electronic.entries[:2] + tuple(positive) + electronic.entries[2:] + tuple(negative), electronic.sheets)


@dataclass(frozen=True)
class MultipleIonic1DProblem:
    """Original electronic rows and every local closed-domain balance."""

    device: CompiledMultipleIonicDevice1D
    voltage_V: float = 0.0
    generation_scale: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.device, CompiledMultipleIonicDevice1D):
            raise TypeError("a full source-bound multiple-domain device is required")
        object.__setattr__(self, "voltage_V", _finite_scalar(self.voltage_V, "voltage_V"))
        object.__setattr__(self, "generation_scale", _finite_scalar(self.generation_scale, "generation_scale", nonnegative=True))
        DC1DProblem(self.device.electronic_component, self.voltage_V, self.generation_scale)

    @property
    def electronic_problem(self) -> DC1DProblem:
        return DC1DProblem(self.device.electronic_component, self.voltage_V, self.generation_scale)

    def _derivative(self, ydot: ArrayLike | None) -> Vector:
        return np.zeros(self.device.state_size) if ydot is None else self.device._values(ydot)

    def ion_fluxes(self, y: ArrayLike, *, domain_id: str, species: str = "positive") -> Vector:
        if domain_id not in self.device.domain_ids:
            raise ValueError("requested ion domain is not assembled")
        domain = self.device.domains[self.device.domain_ids.index(domain_id)]
        return Ionic1DProblem(domain, self.voltage_V, self.generation_scale).ion_fluxes(
            self.device.local_coordinates(y, domain_id), species=species)

    def ion_fluxes_on_electronic_faces(self, y: ArrayLike, *, domain_id: str, species: str = "positive") -> Vector:
        flux = self.ion_fluxes(y, domain_id=domain_id, species=species)
        domain = self.device.domains[self.device.domain_ids.index(domain_id)]
        result = np.zeros(self.device.node_count - 1)
        result[domain.ion_node_indices[:-1]] = flux
        return result

    def ion_current_density_A_m2(self, y: ArrayLike) -> Vector:
        result = np.zeros(self.device.node_count - 1)
        for domain in self.device.domains:
            for species in domain.ion_species_names:
                result += (Q if species == "positive" else -Q) * self.ion_fluxes_on_electronic_faces(
                    y, domain_id=domain.domain_layer_id, species=species)
        return result

    def residual(self, y: ArrayLike, ydot: ArrayLike | None = None) -> Vector:
        device, count = self.device, self.device.node_count
        values, derivative = device._values(y), self._derivative(ydot)
        phi, n, p, populations = device.fields_by_domain(values, voltage_V=self.voltage_V)
        result = np.empty(device.state_size)
        result[:3 * count] = self.electronic_problem.residual(values[:3 * count], derivative[:3 * count])
        rho = device.charge_ledger(n, p, populations).space_charge_C_m3()
        left, right = device.electronic_component.contact_potentials(self.voltage_V)
        result[:count] = device.electronic_component.poisson.residual(phi, rho, left, right)
        for domain in device.domains:
            name = domain.domain_layer_id
            local = device.local_coordinates(values, name)
            mass = domain.ion_coordinate_jacobian_m3(local, voltage_V=self.voltage_V)
            blocks = device.ion_species_slices[name]
            for index, species in enumerate(domain.ion_species_names):
                rate = sum(mass[:, index, column] * derivative[blocks[partner]]
                           for column, partner in enumerate(domain.ion_species_names))
                result[blocks[species]] = -np.diff(np.r_[0.0, self.ion_fluxes(values, domain_id=name, species=species), 0.0])
                result[blocks[species]] -= domain.ion_geometry.cell_measures_m * rate
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("multiple-domain residual is outside the finite envelope")
        return result

    def physical_storage(self, y: ArrayLike) -> Vector:
        """Physical S in particles/m²; algebraic rows have zero storage."""
        device, count = self.device, self.device.node_count
        _, n, p, populations = device.fields_by_domain(y, voltage_V=self.voltage_V)
        result = np.zeros(device.state_size)
        for offset, carrier, density in ((count, "n", n), (2 * count, "p", p)):
            result[offset:offset + count] = device.electronic_component.geometry.cell_measures_m * density
            for side, node in (("left", 0), ("right", count - 1)):
                if device.electronic_component.channel(carrier, side).kind == "fixed_reservoir":
                    result[offset + node] = 0.0
        for domain in device.domains:
            for species, density in zip(domain.ion_species_names, populations[domain.domain_layer_id]):
                result[device.ion_species_slices[domain.domain_layer_id][species]] = domain.ion_geometry.cell_measures_m * density
        return result

    def storage_jacobian(self, y: ArrayLike) -> csc_matrix:
        device, size = self.device, 3 * self.device.node_count
        values = device._values(y)
        electronic = self.electronic_problem.storage_jacobian(values[:size])
        ionic = [Ionic1DProblem(domain, self.voltage_V, self.generation_scale).storage_jacobian(
            device.local_coordinates(values, domain.domain_layer_id))[size:, size:] for domain in device.domains]
        return block_diag((electronic, *ionic), format="csc")

    def inventory_gradients(self, y: ArrayLike) -> dict[str, dict[str, Vector]]:
        """Full physical gradients, with shared-site cross terms and no row replacement."""
        storage = self.storage_jacobian(y)
        return {name: {species: -np.asarray(storage[block, :].sum(axis=0)).ravel()
                       for species, block in blocks.items()}
                for name, blocks in self.device.ion_species_slices.items()}

    def jacobian(self, y: ArrayLike, ydot: ArrayLike | None = None, *, alpha: float = 0.0) -> csc_matrix:
        alpha = _finite_scalar(alpha, "alpha")
        device, size = self.device, 3 * self.device.node_count
        values, derivative = device._values(y), self._derivative(ydot)
        electronic = self.electronic_problem.jacobian(values[:size], derivative[:size], alpha=alpha)
        local = [Ionic1DProblem(domain, self.voltage_V, self.generation_scale).jacobian(
            device.local_coordinates(values, domain.domain_layer_id),
            device.local_coordinates(derivative, domain.domain_layer_id), alpha=alpha) for domain in device.domains]
        return bmat([[electronic, local[0][:size, size:], local[1][:size, size:]],
                     [local[0][size:, :size], local[0][size:, size:], None],
                     [local[1][size:, :size], None, local[1][size:, size:]]], format="csc")
