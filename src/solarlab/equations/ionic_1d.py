"""Executable single-positive-ion extension of the classical electronic rows.

This binding admits one declared layer-wide ion domain, finite sites, blocking
domain endpoints, and a static operating voltage/temperature. It assembles F(y, ydot),
Fy and Fydot; it does not supply a qualified ionic steady or time driver.
The electronic component retains its own named geometry. Ionic storage uses
geometric half cells and the original P0 compensation background.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import block_diag, csc_matrix, diags, lil_matrix

from solarlab.contracts import frozen_array
from solarlab.contracts.arrays import integer_indices
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem, State1D, Vector
from solarlab.mesh.geometry import ControlVolumeGeometry1D, HALF_ENDPOINT_V1
from solarlab.physics.electrostatics.charge import ChargeEntry, ChargeLedger
from solarlab.physics.ions import (
    DIFFUSION_ONLY_STERIC_V1, WHOLE_FLUX_STERIC_V1,
    ion_face_flux, ion_face_flux_jacobian, single_site_density,
)

__all__ = ["CompiledIonicDevice1D", "Ionic1DProblem", "IonicState1D"]
Q = ELEMENTARY_CHARGE_C


@dataclass(frozen=True)
class IonicState1D:
    """Physical trial state bound to the complete ionic device, not just its component."""

    u: Vector
    electronic_state: State1D
    positive_ion_density_m3: Vector
    device_sha256: str
    domain_inventory_m2: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "u", frozen_array(self.u))
        object.__setattr__(self, "positive_ion_density_m3", frozen_array(self.positive_ion_density_m3))
        count = self.electronic_state.potential_V.size
        ion_count = self.positive_ion_density_m3.size
        if self.u.shape != (3 * count + ion_count,) or not np.array_equal(self.u[:3 * count], self.electronic_state.u):
            raise ValueError("ionic coordinates must include their complete electronic component")
        if self.positive_ion_density_m3.ndim != 1 or ion_count < 3 or np.any(self.positive_ion_density_m3 <= 0.0):
            raise ValueError("positive-ion density must be positive on its declared domain")
        if not self.device_sha256 or not math.isfinite(self.domain_inventory_m2) or self.domain_inventory_m2 <= 0.0:
            raise ValueError("ionic state requires complete device identity and finite positive inventory")

    @property
    def voltage_V(self) -> float:
        return self.electronic_state.voltage_V


@dataclass(frozen=True)
class CompiledIonicDevice1D:
    """One explicit electronic component and one closed positive-ion domain.

    The manifest binds the original prepared document, all ionic declarations,
    the electronic projection, both geometries and the inventory. The final
    device never advertises the component's ion-free capability as its own.
    """

    device_id: str
    sha256: str
    original_prepared_sha256: str
    electronic_component: CompiledDevice1D
    ion_geometry: ControlVolumeGeometry1D
    domain_layer_id: str
    positive_ion_reference_m3: Vector
    positive_ion_site_limit_m3: Vector
    positive_ion_logit_reference: Vector
    positive_ion_diffusivity_faces_m2_s: Vector
    steric_model: str
    binding_manifest_json: str
    capabilities: tuple[str, ...] = (
        "classical_1d_electronic_component", "single_positive_ion_1d_residual",
        "finite_sites", "blocking_ion_contacts", "physical_ion_storage",
    )
    ion_node_indices: NDArray[np.intp] = field(default_factory=lambda: np.empty(0, dtype=np.intp))

    def __post_init__(self) -> None:
        count = self.ion_node_count
        indices = integer_indices(self.ion_node_indices)
        if indices.size == 0:
            if count != self.node_count:
                raise ValueError("confined ion domain requires an explicit electrical-node map")
            indices = np.arange(count, dtype=np.intp)
        if (indices.shape != (count,) or np.any(np.diff(indices) != 1)
                or indices[0] < 0 or indices[-1] >= self.node_count):
            raise ValueError("ion-node map must be ordered, contiguous and inside the electronic grid")
        object.__setattr__(self, "ion_node_indices", np.frombuffer(indices.tobytes(), dtype=np.intp))
        layer_ids = [layer.name for layer in self.electronic_component.parameters.layers]
        if layer_ids.count(self.domain_layer_id) != 1:
            raise ValueError("ionic binding: domain layer is absent from the electronic component")
        layer_index = layer_ids.index(self.domain_layer_id)
        planes = self.electronic_component.geometry.interfaces
        left = 0 if layer_index == 0 else planes[layer_index - 1].shared_node
        right = self.node_count - 1 if layer_index == len(layer_ids) - 1 else planes[layer_index].shared_node
        if left is None or right is None or not np.array_equal(indices, np.arange(left, right + 1)):
            raise ValueError("ionic binding: node map must cover both physical layer endpoints")
        for name in ("positive_ion_reference_m3", "positive_ion_site_limit_m3", "positive_ion_logit_reference"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} must match the declared ion nodes")
        object.__setattr__(self, "positive_ion_diffusivity_faces_m2_s", frozen_array(self.positive_ion_diffusivity_faces_m2_s))
        if self.positive_ion_diffusivity_faces_m2_s.shape != (count - 1,) or np.any(self.positive_ion_diffusivity_faces_m2_s < 0.0):
            raise ValueError("ionic diffusivity must be non-negative on every domain face")
        if np.any(self.positive_ion_reference_m3 <= 0.0) or np.any(self.positive_ion_reference_m3 >= self.positive_ion_site_limit_m3):
            raise ValueError("finite-site reference must lie strictly between zero and capacity")
        if self.ion_geometry.name != HALF_ENDPOINT_V1 or self.ion_geometry.interfaces:
            raise ValueError("first ionic domain requires geometric half cells without interfaces")
        if not np.array_equal(self.ion_geometry.nodes_m, self.nodes_m[indices]):
            raise ValueError("ion and electronic nodes must coincide")
        if not np.array_equal(self.ion_geometry.cell_measures_m[1:-1],
                              self.electronic_component.geometry.cell_measures_m[indices[1:-1]]):
            raise ValueError("ion charge widths must match the original Poisson interiors")
        if self.steric_model not in {WHOLE_FLUX_STERIC_V1, DIFFUSION_ONLY_STERIC_V1}:
            raise ValueError("unsupported ionic constitutive law")
        if not self.sha256 or not self.original_prepared_sha256 or not self.domain_layer_id:
            raise ValueError("ionic device needs complete original and domain identities")
        manifest = json.loads(self.binding_manifest_json)
        declarations = manifest.get("original_ionic_inputs") if isinstance(manifest, dict) else None
        if not isinstance(declarations, list):
            raise ValueError("ionic binding requires original ionic declarations")
        matching = [entry for entry in declarations if isinstance(entry, dict) and entry.get("layer") == self.domain_layer_id]
        if len(matching) != 1:
            raise ValueError("ionic binding requires exactly one original domain-layer declaration")
        try:
            reference = float(matching[0]["P0"])
            capacity = float(matching[0]["P_lim"])
            occupancy = reference / capacity
            logit = math.log(occupancy) - math.log1p(-occupancy)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError("ionic binding requires valid original finite-site parameters") from exc
        if not np.array_equal(self.positive_ion_reference_m3, np.full(count, reference)):
            raise ValueError("ionic binding: reference density differs from the original P0 declaration")
        if not np.array_equal(self.positive_ion_site_limit_m3, np.full(count, capacity)):
            raise ValueError("ionic binding: site capacity differs from the original P_lim declaration")
        if not np.array_equal(self.positive_ion_logit_reference, np.full(count, logit)):
            raise ValueError("ionic binding: logit reference differs from the original P0/P_lim relationship")
        domain = manifest.get("ion_domain")
        if (not isinstance(domain, dict) or domain.get("layer") != self.domain_layer_id
                or not np.array_equal(domain.get("nodes_m"), self.ion_geometry.nodes_m)
                or not np.array_equal(domain.get("cell_measures_m"), self.ion_geometry.cell_measures_m)):
            raise ValueError("ionic binding: domain geometry differs from the bound declaration")
        if not np.array_equal(domain.get("electrical_node_indices"), indices):
            raise ValueError("ionic binding: confined node map differs from the declared domain")
        if domain.get("contacts") != ["blocking", "blocking"]:
            raise ValueError("ionic binding: this domain requires two declared blocking endpoints")
        declaration = domain.get("declaration")
        if (not isinstance(declaration, dict) or declaration.get("layer") != self.domain_layer_id
                or declaration.get("endpoints") != ["blocking", "blocking"]
                or (self.ion_node_count != self.node_count and declaration.get("source") != "compile_arguments")):
            raise ValueError("ionic binding: domain endpoint configuration is incomplete")
        for entry in declarations:
            if not isinstance(entry, dict):
                raise ValueError("ionic binding: invalid original ion declaration")
            if any(entry.get(name) != 0.0 for name in ("P0_neg", "D_ion_neg")):
                raise ValueError("ionic binding: negative or dual ions are unsupported")
            if entry.get("layer") != self.domain_layer_id and any(entry.get(name) != 0.0 for name in ("P0", "D_ion")):
                raise ValueError("ionic binding: outside-domain populations or mobility are unsupported")

    @property
    def identity(self) -> str:
        return self.sha256

    @property
    def node_count(self) -> int:
        return self.electronic_component.node_count

    @property
    def ion_node_count(self) -> int:
        return self.ion_geometry.node_count

    @property
    def state_size(self) -> int:
        return 3 * self.node_count + self.ion_node_count

    @property
    def nodes_m(self) -> Vector:
        return self.electronic_component.nodes_m

    @property
    def domain_bounds_m(self) -> tuple[float, float]:
        return float(self.ion_geometry.nodes_m[0]), float(self.ion_geometry.nodes_m[-1])

    @property
    def reference_inventory_m2(self) -> float:
        return math.fsum((self.ion_geometry.cell_measures_m * self.positive_ion_reference_m3).tolist())

    @property
    def ion_charge_weight_ratio(self) -> Vector:
        """Deposit physical ion-cell charge onto the electronic ledger volumes."""
        return frozen_array(self.ion_geometry.cell_measures_m / self.electronic_component.geometry.cell_measures_m[self.ion_node_indices])

    def project_ion_density(self, density: ArrayLike) -> Vector:
        """Charge-deposition density (W_P/W_e)*P on electronic cells, zero outside."""
        values = np.asarray(density, dtype=float)
        if values.shape != (self.ion_node_count,) or not np.all(np.isfinite(values)):
            raise ValueError("ion density projection requires finite domain-local values")
        result = np.zeros(self.node_count)
        result[self.ion_node_indices] = self.ion_charge_weight_ratio * values
        return result

    @property
    def binding_manifest(self) -> dict[str, object]:
        return json.loads(self.binding_manifest_json)

    def fields(self, y: ArrayLike, *, voltage_V: float = 0.0) -> tuple[Vector, Vector, Vector, Vector]:
        values = np.asarray(y, dtype=float)
        count = self.node_count
        if values.shape != (self.state_size,) or not np.all(np.isfinite(values)):
            raise ValueError("ionic coordinates must be finite with shape (3*N + M,)")
        phi, n, p = self.electronic_component.fields(values[:3 * count], voltage_V=voltage_V)
        density = single_site_density(
            self.positive_ion_reference_m3, self.positive_ion_site_limit_m3,
            self.positive_ion_logit_reference, values[3 * count:],
        )
        if any(not np.all(np.isfinite(field)) or np.any(field <= 0.0) for field in (n, p, density)):
            raise ValueError("ionic/electronic trial density is outside the finite positive domain")
        if np.any(density >= self.positive_ion_site_limit_m3):
            raise ValueError("single-ion finite-site density saturated")
        return phi, n, p, density

    def state(self, y: ArrayLike, *, voltage_V: float = 0.0) -> IonicState1D:
        values = np.asarray(y, dtype=float)
        _, _, _, density = self.fields(values, voltage_V=voltage_V)
        component = self.electronic_component.state(values[:3 * self.node_count], voltage_V=voltage_V)
        inventory = math.fsum((self.ion_geometry.cell_measures_m * density).tolist())
        return IonicState1D(values, component, density, self.identity, inventory)

    def initial_state(self, *, voltage_V: float = 0.0) -> IonicState1D:
        """Cold physical reference with exact configured ion density; no settling solve."""
        left, right = self.electronic_component.contact_potentials(voltage_V)
        position = (self.nodes_m - self.nodes_m[0]) / self.electronic_component.geometry.extent_m
        phi = left + (right - left) * position
        return self.state(np.r_[phi, np.zeros(2 * self.node_count + self.ion_node_count)], voltage_V=voltage_V)

    def charge_ledger(self, n: Vector, p: Vector, density: Vector) -> ChargeLedger:
        component = self.electronic_component.charge_ledger(n, p)
        # Preserve the original positive-ion compensation and declaration order.
        ion = ChargeEntry("positive_ion", "ion_charge_projection", 1,
                          self.project_ion_density(density), self.project_ion_density(self.positive_ion_reference_m3))
        return ChargeLedger(self.node_count, component.interface_count, component.entries[:2] + (ion,) + component.entries[2:])


@dataclass(frozen=True)
class Ionic1DProblem:
    """Actual electronic/ionic F, Fy+alpha*Fydot and storage partials."""

    device: CompiledIonicDevice1D
    voltage_V: float = 0.0
    generation_scale: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.device, CompiledIonicDevice1D):
            raise TypeError("Ionic1DProblem requires the complete compiled ionic device")
        DC1DProblem(self.device.electronic_component, self.voltage_V, self.generation_scale)

    @property
    def electronic_problem(self) -> DC1DProblem:
        return DC1DProblem(self.device.electronic_component, self.voltage_V, self.generation_scale)

    def _derivative(self, ydot: ArrayLike | None) -> Vector:
        if ydot is None:
            return np.zeros(self.device.state_size)
        derivative = np.asarray(ydot, dtype=float)
        if derivative.shape != (self.device.state_size,) or not np.all(np.isfinite(derivative)):
            raise ValueError("ydot must be finite and match the complete ionic coordinates")
        return derivative

    def _ion_partial(self, density: Vector) -> Vector:
        # Same finite-site chain derivative as the implemented single-ion map.
        return density * (1.0 - density / self.device.positive_ion_site_limit_m3)

    def ion_fluxes(self, y: ArrayLike) -> Vector:
        phi, _, _, density = self.device.fields(y, voltage_V=self.voltage_V)
        return ion_face_flux(
            phi[self.device.ion_node_indices], density, np.diff(self.device.ion_geometry.nodes_m), self.device.positive_ion_diffusivity_faces_m2_s,
            self.device.electronic_component.thermal_voltage_V, self.device.positive_ion_site_limit_m3[:-1],
            steric_diffusion_only=self.device.steric_model == DIFFUSION_ONLY_STERIC_V1,
            P_lim_node=self.device.positive_ion_site_limit_m3,
        )

    def ion_fluxes_on_electronic_faces(self, y: ArrayLike) -> Vector:
        """Scatter domain face exchange without inventing cross-interface transmission."""
        result = np.zeros(self.device.node_count - 1)
        result[self.device.ion_node_indices[:-1]] = self.ion_fluxes(y)
        return result

    def residual(self, y: ArrayLike, ydot: ArrayLike | None = None) -> Vector:
        count = self.device.node_count
        values = np.asarray(y, dtype=float)
        phi, n, p, density = self.device.fields(values, voltage_V=self.voltage_V)
        derivative = self._derivative(ydot)
        result = np.empty(self.device.state_size)
        result[:3 * count] = self.electronic_problem.residual(values[:3 * count], derivative[:3 * count])
        rho = self.device.charge_ledger(n, p, density).space_charge_C_m3()
        left, right = self.device.electronic_component.contact_potentials(self.voltage_V)
        result[:count] = self.device.electronic_component.poisson.residual(phi, rho, left, right)
        # Every internal face contributes -Gamma to its left cell, +Gamma to right.
        result[3 * count:] = -np.diff(np.r_[0.0, self.ion_fluxes(values), 0.0])
        result[3 * count:] -= self.device.ion_geometry.cell_measures_m * self._ion_partial(density) * derivative[3 * count:]
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("ionic residual is outside the finite numerical envelope")
        return result

    def storage_jacobian(self, y: ArrayLike) -> csc_matrix:
        values = np.asarray(y, dtype=float)
        _, _, _, density = self.device.fields(values, voltage_V=self.voltage_V)
        electronic = self.electronic_problem.storage_jacobian(values[:3 * self.device.node_count])
        ionic = diags(-self.device.ion_geometry.cell_measures_m * self._ion_partial(density), format="csc")
        return block_diag((electronic, ionic), format="csc")

    def jacobian(self, y: ArrayLike, ydot: ArrayLike | None = None, *, alpha: float = 0.0) -> csc_matrix:
        if not math.isfinite(alpha):
            raise ValueError("alpha must be finite")
        count = self.device.node_count
        values = np.asarray(y, dtype=float)
        phi, _, _, density = self.device.fields(values, voltage_V=self.voltage_V)
        derivative = self._derivative(ydot)
        partial = self._ion_partial(density)
        flux = ion_face_flux_jacobian(
            phi[self.device.ion_node_indices], density, np.diff(self.device.ion_geometry.nodes_m), self.device.positive_ion_diffusivity_faces_m2_s,
            self.device.electronic_component.thermal_voltage_V, self.device.positive_ion_site_limit_m3[:-1],
            steric_diffusion_only=self.device.steric_model == DIFFUSION_ONLY_STERIC_V1,
            P_lim_node=self.device.positive_ion_site_limit_m3,
        )
        if not np.all(flux.differentiable_faces):
            raise ValueError("ionic Jacobian is undefined at a retained historical regularization kink")
        matrix = lil_matrix((self.device.state_size, self.device.state_size), dtype=float)
        matrix[:3 * count, :3 * count] = self.electronic_problem.jacobian(values[:3 * count], derivative[:3 * count])
        indices = self.device.ion_node_indices
        charge = Q * self.device.electronic_component.poisson.charge_jacobian()[indices] * self.device.ion_charge_weight_ratio * partial
        for local, node in enumerate(indices):
            matrix[node, 3 * count + local] = charge[local]
        for face in range(self.device.ion_node_count - 1):
            for orientation, node in ((-1.0, face), (1.0, face + 1)):
                row = 3 * count + node
                matrix[row, indices[face]] += orientation * flux.potential_left_derivative[face]
                matrix[row, indices[face + 1]] += orientation * flux.potential_right_derivative[face]
                matrix[row, 3 * count + face] += orientation * flux.density_left_derivative[face] * partial[face]
                matrix[row, 3 * count + face + 1] += orientation * flux.density_right_derivative[face] * partial[face + 1]
        second = partial * (1.0 - 2.0 * density / self.device.positive_ion_site_limit_m3)
        matrix[3 * count:, 3 * count:] -= diags(self.device.ion_geometry.cell_measures_m * second * derivative[3 * count:])
        result = matrix.tocsc()
        if alpha != 0.0:
            result = result + alpha * self.storage_jacobian(values)
        if not np.all(np.isfinite(result.data)):
            raise FloatingPointError("ionic Jacobian is outside the finite numerical envelope")
        return result

    def inventory_rate_m2_s(self, y: ArrayLike, ydot: ArrayLike) -> float:
        """Actual d(sum(W_P*P))/dt; blocking face transport has zero net input."""
        _, _, _, density = self.device.fields(y, voltage_V=self.voltage_V)
        derivative = self._derivative(ydot)[3 * self.device.node_count:]
        return math.fsum((self.device.ion_geometry.cell_measures_m * self._ion_partial(density) * derivative).tolist())
