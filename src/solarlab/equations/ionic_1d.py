"""Executable independent/shared finite-site extension of classical electronic rows.

This binding admits one declared layer-wide ion domain, finite sites, blocking
domain endpoints, and a static operating voltage/temperature. It assembles F(y, ydot),
Fy and Fydot; it does not supply a qualified ionic steady or time driver.
The electronic component retains its own named geometry. Ionic storage uses
geometric half cells and the original P0/P0_neg compensation backgrounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import block_diag, bmat, csc_matrix, diags, lil_matrix

from solarlab.contracts import frozen_array
from solarlab.contracts.arrays import integer_indices
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem, State1D, Vector
from solarlab.mesh.geometry import ControlVolumeGeometry1D, HALF_ENDPOINT_V1
from solarlab.physics.band_structure.temperature import ion_diffusivity_at_temperature
from solarlab.physics.electrostatics.charge import ChargeEntry, ChargeLedger
from solarlab.physics.ions import (
    DIFFUSION_ONLY_STERIC_V1, WHOLE_FLUX_STERIC_V1,
    ion_face_flux, ion_face_flux_jacobian, shared_site_densities, shifted_logistic_density, single_site_density,
)
from solarlab.physics.ions.shared_site import shared_site_coordinate_jacobian_m3, shared_site_coordinate_hessian_m3

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
    negative_ion_density_m3: Vector = field(default_factory=lambda: np.empty(0))
    negative_domain_inventory_m2: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "u", frozen_array(self.u))
        object.__setattr__(self, "positive_ion_density_m3", frozen_array(self.positive_ion_density_m3))
        object.__setattr__(self, "negative_ion_density_m3", frozen_array(self.negative_ion_density_m3))
        count = self.electronic_state.potential_V.size
        ion_count = self.positive_ion_density_m3.size
        negative_count = self.negative_ion_density_m3.size
        if self.u.shape != (3 * count + ion_count + negative_count,) or not np.array_equal(self.u[:3 * count], self.electronic_state.u):
            raise ValueError("ionic coordinates must include their complete electronic component")
        if self.positive_ion_density_m3.ndim != 1 or ion_count < 3 or np.any(self.positive_ion_density_m3 <= 0.0):
            raise ValueError("positive-ion density must be positive on its declared domain")
        if not self.device_sha256 or not math.isfinite(self.domain_inventory_m2) or self.domain_inventory_m2 <= 0.0:
            raise ValueError("ionic state requires complete device identity and finite positive inventory")
        if negative_count:
            if (self.negative_ion_density_m3.shape != (ion_count,) or np.any(self.negative_ion_density_m3 <= 0.0)
                    or not math.isfinite(self.negative_domain_inventory_m2) or self.negative_domain_inventory_m2 <= 0.0):
                raise ValueError("negative-ion density and inventory must be positive on the complete same domain")
        elif self.negative_domain_inventory_m2 != 0.0:
            raise ValueError("an absent negative population cannot carry an inventory")

    @property
    def voltage_V(self) -> float:
        return self.electronic_state.voltage_V

    @property
    def domain_inventories_m2(self) -> dict[str, float]:
        if self.negative_ion_density_m3.size:
            return {"positive": self.domain_inventory_m2, "negative": self.negative_domain_inventory_m2}
        return {"positive": self.domain_inventory_m2}


@dataclass(frozen=True)
class CompiledIonicDevice1D:
    """One electronic component and one closed domain with independent ion sites.

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
    negative_ion_reference_m3: Vector = field(default_factory=lambda: np.empty(0))
    negative_ion_site_limit_m3: Vector = field(default_factory=lambda: np.empty(0))
    negative_ion_logit_reference: Vector = field(default_factory=lambda: np.empty(0))
    negative_ion_diffusivity_faces_m2_s: Vector = field(default_factory=lambda: np.empty(0))
    ion_site_contract: str = "independent_finite_sites"

    def __post_init__(self) -> None:
        count = self.ion_node_count
        if self.ion_site_contract not in {"independent_finite_sites", "shared_finite_sites"}:
            raise ValueError("unsupported ionic site contract")
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
        for name in ("negative_ion_reference_m3", "negative_ion_site_limit_m3", "negative_ion_logit_reference",
                     "negative_ion_diffusivity_faces_m2_s"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        negative_count = count if self.negative_ion_reference_m3.size else 0
        if any(getattr(self, name).shape != (negative_count,) for name in (
            "negative_ion_reference_m3", "negative_ion_site_limit_m3", "negative_ion_logit_reference",
        )) or self.negative_ion_diffusivity_faces_m2_s.shape != (count - 1 if negative_count else 0,):
            raise ValueError("negative-ion arrays must all match the complete domain or all be absent")
        if np.any(self.negative_ion_diffusivity_faces_m2_s < 0.0):
            raise ValueError("negative-ion diffusivity must be non-negative on every domain face")
        if self.ion_site_contract == "shared_finite_sites" and not negative_count:
            raise ValueError("ionic binding: shared sites require both complete populations")
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
        expected_positive_logit = np.full(count, logit)
        if self.ion_site_contract == "shared_finite_sites":
            source_negative = float(matching[0]["P0_neg"])
            source_negative_capacity = float(matching[0]["P_lim_neg"])
            if source_negative_capacity != capacity or source_negative <= 0.0 or reference + source_negative >= capacity:
                raise ValueError("ionic binding: shared sites require common capacity and positive source vacancy")
            source_vacancy = 1.0 - (np.full(count, reference) + np.full(count, source_negative)) / capacity
            expected_positive_logit = np.log(np.full(count, reference) / capacity) - np.log(source_vacancy)
        if not np.array_equal(self.positive_ion_logit_reference, expected_positive_logit):
            raise ValueError("ionic binding: logit reference differs from the original P0/P_lim relationship")
        negative_reference = float(matching[0]["P0_neg"])
        negative_diffusion = float(matching[0]["D_ion_neg"])
        if negative_count:
            try:
                negative_capacity = float(matching[0]["P_lim_neg"])
                negative_occupancy = negative_reference / negative_capacity
                negative_logit = math.log(negative_occupancy) - math.log1p(-negative_occupancy)
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                raise ValueError("ionic binding requires valid original negative finite-site parameters") from exc
            for name, value in (("negative_ion_reference_m3", negative_reference),
                                ("negative_ion_site_limit_m3", negative_capacity)):
                if not np.array_equal(getattr(self, name), np.full(count, value)):
                    raise ValueError(f"ionic binding: {name} differs from its original declaration")
            expected_negative_logit = np.full(count, negative_logit)
            if self.ion_site_contract == "shared_finite_sites":
                expected_negative_logit = np.log(np.full(count, negative_reference) / capacity) - np.log(source_vacancy)
            if not np.array_equal(self.negative_ion_logit_reference, expected_negative_logit):
                raise ValueError("ionic binding: negative log-ratio reference differs from its original declaration")
            model = manifest.get("ion_model")
            if (not isinstance(model, dict) or model.get("site_contract") != self.ion_site_contract
                    or not np.array_equal(model.get("negative_face_diffusivity_m2_s"), self.negative_ion_diffusivity_faces_m2_s)):
                raise ValueError("ionic binding: negative transport must match the independent-site declaration")
            activation = float(matching[0]["E_a_ion"])
            if (not all(math.isfinite(value) for value in (negative_reference, negative_capacity, negative_diffusion, activation))
                    or negative_diffusion < 0.0 or activation < 0.0 or negative_reference <= 0.0
                    or not np.all(self.negative_ion_reference_m3 < self.negative_ion_site_limit_m3)):
                raise ValueError("ionic binding: negative population/transport is outside its finite-site domain")
            scaled_negative = ion_diffusivity_at_temperature(negative_diffusion, self.electronic_component.parameters.temperature_K, activation)
            if not np.array_equal(self.negative_ion_diffusivity_faces_m2_s, np.full(count - 1, scaled_negative)):
                raise ValueError("ionic binding: negative diffusivity differs from its original temperature-scaled declaration")
            settings = manifest.get("original_ionic_settings")
            if not isinstance(settings, dict) or settings.get("ion_steric_shared_site") is not (self.ion_site_contract == "shared_finite_sites"):
                raise ValueError("ionic binding: declared ionic site contract differs from its original flags")
            if self.ion_site_contract == "shared_finite_sites":
                if self.steric_model != DIFFUSION_ONLY_STERIC_V1 or settings.get("ion_steric_diffusion_only") is not True:
                    raise ValueError("ionic binding: shared sites require the original diffusion-only law")
                if (model.get("coordinate_map") != "historical_shared_site_softmax_v1"
                        or not np.array_equal(model.get("reference_vacancy_fraction"), source_vacancy)):
                    raise ValueError("ionic binding: shared coordinate/vacancy declaration differs from its source")
            layout = manifest.get("state_layout")
            if (not isinstance(layout, dict) or layout.get("ion_species") != ["positive", "negative"]
                    or layout.get("size") != self.state_size):
                raise ValueError("ionic binding: both species must remain in the declared state layout")
            object.__setattr__(self, "capabilities", tuple(
                ("shared_site_dual_ion_1d_residual" if self.ion_site_contract == "shared_finite_sites" else "independent_dual_ion_1d_residual")
                if name == "single_positive_ion_1d_residual" else name
                for name in self.capabilities
            ))
        elif negative_reference != 0.0 or negative_diffusion != 0.0:
            raise ValueError("ionic binding: declared negative population or mobility cannot be stripped")
        domain = manifest.get("ion_domain")
        if (not isinstance(domain, dict) or domain.get("layer") != self.domain_layer_id
                or not np.array_equal(domain.get("nodes_m"), self.ion_geometry.nodes_m)
                or not np.array_equal(domain.get("cell_measures_m"), self.ion_geometry.cell_measures_m)):
            raise ValueError("ionic binding: domain geometry differs from the bound declaration")
        if not np.array_equal(domain.get("electrical_node_indices"), indices):
            raise ValueError("ionic binding: confined node map differs from the declared domain")
        if domain.get("contacts") != ["blocking", "blocking"]:
            raise ValueError("ionic binding: this domain requires two declared blocking endpoints")
        if negative_count and domain.get("negative_reference_inventory_m2") != self.negative_reference_inventory_m2:
            raise ValueError("ionic binding: negative inventory differs from its original domain reference")
        declaration = domain.get("declaration")
        if (not isinstance(declaration, dict) or declaration.get("layer") != self.domain_layer_id
                or declaration.get("endpoints") != ["blocking", "blocking"]
                or (self.ion_node_count != self.node_count and declaration.get("source") != "compile_arguments")):
            raise ValueError("ionic binding: domain endpoint configuration is incomplete")
        for entry in declarations:
            if not isinstance(entry, dict):
                raise ValueError("ionic binding: invalid original ion declaration")
            if entry.get("layer") != self.domain_layer_id and any(entry.get(name) != 0.0 for name in ("P0", "D_ion", "P0_neg", "D_ion_neg")):
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
        return 3 * self.node_count + self.ion_species_count * self.ion_node_count

    @property
    def ion_species_names(self) -> tuple[str, ...]:
        return ("positive", "negative") if self.negative_ion_reference_m3.size else ("positive",)

    @property
    def ion_species_count(self) -> int:
        return len(self.ion_species_names)

    @property
    def ion_species_slices(self) -> tuple[slice, ...]:
        start, M = 3 * self.node_count, self.ion_node_count
        return tuple(slice(start + index * M, start + (index + 1) * M) for index in range(self.ion_species_count))

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
    def negative_reference_inventory_m2(self) -> float:
        if self.ion_species_count == 1:
            return 0.0
        return math.fsum((self.ion_geometry.cell_measures_m * self.negative_ion_reference_m3).tolist())

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
        """Historical positive-only reader; dual devices require fields_by_species."""
        if self.ion_species_count != 1:
            raise ValueError("dual-ion fields require fields_by_species; a population cannot be omitted")
        phi, n, p, populations = self.fields_by_species(y, voltage_V=voltage_V)
        return phi, n, p, populations[0]

    def fields_by_species(self, y: ArrayLike, *, voltage_V: float = 0.0) -> tuple[Vector, Vector, Vector, tuple[Vector, ...]]:
        """Physical electronic fields and every declared ion population, in block order."""
        values = np.asarray(y, dtype=float)
        count = self.node_count
        if values.shape != (self.state_size,) or not np.all(np.isfinite(values)):
            raise ValueError("ionic coordinates must be finite with shape (3*N + species_count*M,)")
        phi, n, p = self.electronic_component.fields(values[:3 * count], voltage_V=voltage_V)
        populations: list[Vector] = []
        if self.ion_site_contract == "shared_finite_sites":
            positive, negative = self.ion_species_slices
            populations.extend(shared_site_densities(
                self.positive_ion_site_limit_m3, self.positive_ion_logit_reference, self.negative_ion_logit_reference,
                values[positive], values[negative],
            ))
            if np.any(populations[0] + populations[1] >= self.positive_ion_site_limit_m3):
                raise ValueError("shared-site trial summed occupancy is outside the representable finite vacancy domain")
        for species, block in zip(self.ion_species_names, self.ion_species_slices):
            if self.ion_site_contract == "shared_finite_sites":
                break
            reference = getattr(self, species + "_ion_reference_m3")
            capacity = getattr(self, species + "_ion_site_limit_m3")
            logit = getattr(self, species + "_ion_logit_reference")
            if self.ion_species_count == 1:
                density = single_site_density(reference, capacity, logit, values[block])
            else:
                density = shifted_logistic_density(reference, capacity, logit, values[block], label=species)
            if np.any(density >= capacity):
                raise ValueError(f"{species}-ion finite-site density saturated")
            populations.append(density)
        if any(not np.all(np.isfinite(value)) or np.any(value <= 0.0) for value in (n, p, *populations)):
            raise ValueError("ionic/electronic trial density is outside the finite positive domain")
        return phi, n, p, tuple(populations)

    def ion_coordinate_jacobian_m3(self, y: ArrayLike, *, voltage_V: float = 0.0) -> np.ndarray:
        """Node-major complete physical ion derivatives [M,species,coordinate]."""
        populations = self.fields_by_species(y, voltage_V=voltage_V)[3]
        if self.ion_site_contract == "shared_finite_sites":
            return shared_site_coordinate_jacobian_m3(populations[0], populations[1], self.positive_ion_site_limit_m3)
        result = np.zeros((self.ion_node_count, self.ion_species_count, self.ion_species_count))
        for index, (species, density) in enumerate(zip(self.ion_species_names, populations)):
            result[:, index, index] = density * (1.0 - density / getattr(self, species + "_ion_site_limit_m3"))
        return result

    def ion_coordinate_hessian_m3(self, y: ArrayLike, *, voltage_V: float = 0.0) -> np.ndarray:
        """Node-major complete physical ion Hessian [M,species,coordinate,coordinate]."""
        populations = self.fields_by_species(y, voltage_V=voltage_V)[3]
        if self.ion_site_contract == "shared_finite_sites":
            return shared_site_coordinate_hessian_m3(populations[0], populations[1], self.positive_ion_site_limit_m3)
        result = np.zeros((self.ion_node_count, self.ion_species_count, self.ion_species_count, self.ion_species_count))
        for index, (species, density) in enumerate(zip(self.ion_species_names, populations)):
            capacity = getattr(self, species + "_ion_site_limit_m3")
            result[:, index, index, index] = density * (1.0 - density / capacity) * (1.0 - 2.0 * density / capacity)
        return result

    def state(self, y: ArrayLike, *, voltage_V: float = 0.0) -> IonicState1D:
        values = np.asarray(y, dtype=float)
        _, _, _, populations = self.fields_by_species(values, voltage_V=voltage_V)
        component = self.electronic_component.state(values[:3 * self.node_count], voltage_V=voltage_V)
        inventory = math.fsum((self.ion_geometry.cell_measures_m * populations[0]).tolist())
        negative = populations[1] if self.ion_species_count == 2 else np.empty(0)
        negative_inventory = math.fsum((self.ion_geometry.cell_measures_m * negative).tolist()) if negative.size else 0.0
        return IonicState1D(values, component, populations[0], self.identity, inventory, negative, negative_inventory)

    def initial_state(self, *, voltage_V: float = 0.0) -> IonicState1D:
        """Cold physical reference with exact configured ion density; no settling solve."""
        left, right = self.electronic_component.contact_potentials(voltage_V)
        position = (self.nodes_m - self.nodes_m[0]) / self.electronic_component.geometry.extent_m
        phi = left + (right - left) * position
        return self.state(np.r_[phi, np.zeros(2 * self.node_count + self.ion_species_count * self.ion_node_count)], voltage_V=voltage_V)

    def charge_ledger(self, n: Vector, p: Vector, density: Vector, negative_density: Vector | None = None) -> ChargeLedger:
        component = self.electronic_component.charge_ledger(n, p)
        # Preserve the original positive-ion compensation and declaration order.
        ion = ChargeEntry("positive_ion", "ion_charge_projection", 1,
                          self.project_ion_density(density), self.project_ion_density(self.positive_ion_reference_m3))
        ledger = ChargeLedger(self.node_count, component.interface_count, component.entries[:2] + (ion,) + component.entries[2:])
        if self.ion_species_count == 2:
            if negative_density is None:
                raise ValueError("dual-ion charge requires the declared negative population")
            negative = ChargeEntry("negative_ion", "ion_charge_projection", -1,
                                   self.project_ion_density(negative_density), self.project_ion_density(self.negative_ion_reference_m3),
                                   group="negative_ion_space_charge")
            return ledger.with_entry(negative)
        if negative_density is not None:
            raise ValueError("negative charge is not assembled for this positive-only device")
        return ledger


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

    def _ion_partial(self, density: Vector, species: str = "positive") -> Vector:
        # Same finite-site chain derivative as the implemented single-ion map.
        return density * (1.0 - density / getattr(self.device, species + "_ion_site_limit_m3"))

    def ion_fluxes(self, y: ArrayLike, *, species: str = "positive") -> Vector:
        """Particle flux for one explicitly named species, oriented along +x."""
        if species not in self.device.ion_species_names:
            raise ValueError("requested ion species is not assembled in this device")
        phi, _, _, populations = self.device.fields_by_species(y, voltage_V=self.voltage_V)
        index = self.device.ion_species_names.index(species)
        density = populations[index]
        partner = populations[1 - index] if self.device.ion_site_contract == "shared_finite_sites" else None
        capacity = getattr(self.device, species + "_ion_site_limit_m3")
        return ion_face_flux(
            phi[self.device.ion_node_indices], density, np.diff(self.device.ion_geometry.nodes_m), getattr(self.device, species + "_ion_diffusivity_faces_m2_s"),
            self.device.electronic_component.thermal_voltage_V, capacity[:-1],
            steric_diffusion_only=self.device.steric_model == DIFFUSION_ONLY_STERIC_V1,
            P_lim_node=capacity, P_other_node=partner, drift_sign=1.0 if species == "positive" else -1.0,
        )

    def ion_fluxes_on_electronic_faces(self, y: ArrayLike, *, species: str = "positive") -> Vector:
        """Scatter domain face exchange without inventing cross-interface transmission."""
        result = np.zeros(self.device.node_count - 1)
        result[self.device.ion_node_indices[:-1]] = self.ion_fluxes(y, species=species)
        return result

    def ion_current_density_A_m2(self, y: ArrayLike) -> Vector:
        """All internal electrical faces: q*(Gamma_positive-Gamma_negative)."""
        positive = self.ion_fluxes_on_electronic_faces(y)
        if self.device.ion_species_count == 1:
            return Q * positive
        return Q * (positive - self.ion_fluxes_on_electronic_faces(y, species="negative"))

    def residual(self, y: ArrayLike, ydot: ArrayLike | None = None) -> Vector:
        count = self.device.node_count
        values = np.asarray(y, dtype=float)
        phi, n, p, populations = self.device.fields_by_species(values, voltage_V=self.voltage_V)
        derivative = self._derivative(ydot)
        result = np.empty(self.device.state_size)
        result[:3 * count] = self.electronic_problem.residual(values[:3 * count], derivative[:3 * count])
        rho = self.device.charge_ledger(n, p, populations[0], populations[1] if len(populations) == 2 else None).space_charge_C_m3()
        left, right = self.device.electronic_component.contact_potentials(self.voltage_V)
        result[:count] = self.device.electronic_component.poisson.residual(phi, rho, left, right)
        # Every internal face contributes -Gamma to its left cell, +Gamma to right.
        for index, (species, density, block) in enumerate(zip(self.device.ion_species_names, populations, self.device.ion_species_slices)):
            result[block] = -np.diff(np.r_[0.0, self.ion_fluxes(values, species=species), 0.0])
            if self.device.ion_site_contract == "shared_finite_sites":
                mass = self.device.ion_coordinate_jacobian_m3(values, voltage_V=self.voltage_V)
                plus, minus = self.device.ion_species_slices
                rate = mass[:, index, 0] * derivative[plus] + mass[:, index, 1] * derivative[minus]
                result[block] -= self.device.ion_geometry.cell_measures_m * rate
            else:
                result[block] -= self.device.ion_geometry.cell_measures_m * self._ion_partial(density, species) * derivative[block]
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("ionic residual is outside the finite numerical envelope")
        return result

    def storage_jacobian(self, y: ArrayLike) -> csc_matrix:
        values = np.asarray(y, dtype=float)
        _, _, _, populations = self.device.fields_by_species(values, voltage_V=self.voltage_V)
        electronic = self.electronic_problem.storage_jacobian(values[:3 * self.device.node_count])
        if self.device.ion_site_contract == "shared_finite_sites":
            mass = self.device.ion_coordinate_jacobian_m3(values, voltage_V=self.voltage_V)
            coupled = bmat([[diags(-self.device.ion_geometry.cell_measures_m * mass[:, row, column], format="csc")
                             for column in range(2)] for row in range(2)], format="csc")
            return block_diag((electronic, coupled), format="csc")
        ionic = tuple(diags(-self.device.ion_geometry.cell_measures_m * self._ion_partial(density, species), format="csc")
                      for species, density in zip(self.device.ion_species_names, populations))
        return block_diag((electronic, *ionic), format="csc")

    def jacobian(self, y: ArrayLike, ydot: ArrayLike | None = None, *, alpha: float = 0.0) -> csc_matrix:
        if not math.isfinite(alpha):
            raise ValueError("alpha must be finite")
        if self.device.ion_site_contract == "shared_finite_sites":
            return self._shared_site_jacobian(y, ydot, alpha)
        count = self.device.node_count
        values = np.asarray(y, dtype=float)
        phi, _, _, populations = self.device.fields_by_species(values, voltage_V=self.voltage_V)
        derivative = self._derivative(ydot)
        matrix = lil_matrix((self.device.state_size, self.device.state_size), dtype=float)
        matrix[:3 * count, :3 * count] = self.electronic_problem.jacobian(values[:3 * count], derivative[:3 * count])
        indices = self.device.ion_node_indices
        for species, density, block in zip(self.device.ion_species_names, populations, self.device.ion_species_slices):
            start = block.start
            capacity = getattr(self.device, species + "_ion_site_limit_m3")
            partial = self._ion_partial(density, species)
            sign = 1.0 if species == "positive" else -1.0
            flux = ion_face_flux_jacobian(
                phi[indices], density, np.diff(self.device.ion_geometry.nodes_m), getattr(self.device, species + "_ion_diffusivity_faces_m2_s"),
                self.device.electronic_component.thermal_voltage_V, capacity[:-1],
                steric_diffusion_only=self.device.steric_model == DIFFUSION_ONLY_STERIC_V1,
                P_lim_node=capacity, drift_sign=sign,
            )
            if not np.all(flux.differentiable_faces):
                raise ValueError("ionic Jacobian is undefined at a retained historical regularization kink")
            charge = Q * self.device.electronic_component.poisson.charge_jacobian()[indices] * self.device.ion_charge_weight_ratio * partial
            if sign < 0.0:
                charge = -charge
            for local, node in enumerate(indices):
                matrix[node, start + local] = charge[local]
            for face in range(self.device.ion_node_count - 1):
                for orientation, node in ((-1.0, face), (1.0, face + 1)):
                    row = start + node
                    matrix[row, indices[face]] += orientation * flux.potential_left_derivative[face]
                    matrix[row, indices[face + 1]] += orientation * flux.potential_right_derivative[face]
                    matrix[row, start + face] += orientation * flux.density_left_derivative[face] * partial[face]
                    matrix[row, start + face + 1] += orientation * flux.density_right_derivative[face] * partial[face + 1]
            second = partial * (1.0 - 2.0 * density / capacity)
            matrix[block, block] -= diags(self.device.ion_geometry.cell_measures_m * second * derivative[block])
        result = matrix.tocsc()
        if alpha != 0.0:
            result = result + alpha * self.storage_jacobian(values)
        if not np.all(np.isfinite(result.data)):
            raise FloatingPointError("ionic Jacobian is outside the finite numerical envelope")
        return result

    def _shared_site_jacobian(self, y: ArrayLike, ydot: ArrayLike | None, alpha: float) -> csc_matrix:
        """Full existing-row transport, signed charge and coupled storage chain."""
        device, count = self.device, self.device.node_count
        values = np.asarray(y, dtype=float)
        phi, _, _, populations = device.fields_by_species(values, voltage_V=self.voltage_V)
        derivative = self._derivative(ydot)
        mass = device.ion_coordinate_jacobian_m3(values, voltage_V=self.voltage_V)
        hessian = device.ion_coordinate_hessian_m3(values, voltage_V=self.voltage_V)
        matrix = lil_matrix((device.state_size, device.state_size), dtype=float)
        matrix[:3 * count, :3 * count] = self.electronic_problem.jacobian(values[:3 * count], derivative[:3 * count])
        indices, blocks = device.ion_node_indices, device.ion_species_slices
        for column, block in enumerate(blocks):
            charge = Q * device.electronic_component.poisson.charge_jacobian()[indices] * device.ion_charge_weight_ratio * (mass[:, 0, column] - mass[:, 1, column])
            for local, node in enumerate(indices):
                matrix[node, block.start + local] = charge[local]
        for species, (name, density, block) in enumerate(zip(device.ion_species_names, populations, blocks)):
            other = 1 - species
            flux = ion_face_flux_jacobian(
                phi[indices], density, np.diff(device.ion_geometry.nodes_m), getattr(device, name + "_ion_diffusivity_faces_m2_s"),
                device.electronic_component.thermal_voltage_V, device.positive_ion_site_limit_m3[:-1],
                steric_diffusion_only=True, P_lim_node=device.positive_ion_site_limit_m3,
                P_other_node=populations[other], drift_sign=1.0 if species == 0 else -1.0,
            )
            if not np.all(flux.differentiable_faces):
                raise ValueError("ionic Jacobian is undefined at a retained historical regularization kink")
            for face in range(device.ion_node_count - 1):
                for orientation, local in ((-1.0, face), (1.0, face + 1)):
                    row = block.start + local
                    matrix[row, indices[face]] += orientation * flux.potential_left_derivative[face]
                    matrix[row, indices[face + 1]] += orientation * flux.potential_right_derivative[face]
                    for column, column_block in enumerate(blocks):
                        left = flux.density_left_derivative[face] * mass[face, species, column] + flux.partner_left_derivative[face] * mass[face, other, column]
                        right = flux.density_right_derivative[face] * mass[face + 1, species, column] + flux.partner_right_derivative[face] * mass[face + 1, other, column]
                        matrix[row, column_block.start + face] += orientation * left
                        matrix[row, column_block.start + face + 1] += orientation * right
            for column, column_block in enumerate(blocks):
                second = hessian[:, species, column, 0] * derivative[blocks[0]] + hessian[:, species, column, 1] * derivative[blocks[1]]
                matrix[block, column_block] -= diags(device.ion_geometry.cell_measures_m * second)
        result = matrix.tocsc()
        if alpha != 0.0:
            result = result + alpha * self.storage_jacobian(values)
        if not np.all(np.isfinite(result.data)):
            raise FloatingPointError("ionic Jacobian is outside the finite numerical envelope")
        return result

    def inventory_rate_m2_s(self, y: ArrayLike, ydot: ArrayLike, *, species: str = "positive") -> float:
        """Actual d(sum(W_P*P))/dt; blocking face transport has zero net input."""
        if species not in self.device.ion_species_names:
            raise ValueError("requested ion species is not assembled in this device")
        _, _, _, populations = self.device.fields_by_species(y, voltage_V=self.voltage_V)
        index = self.device.ion_species_names.index(species)
        if self.device.ion_site_contract == "shared_finite_sites":
            derivative = self._derivative(ydot)
            mass = self.device.ion_coordinate_jacobian_m3(y, voltage_V=self.voltage_V)
            plus, minus = self.device.ion_species_slices
            rate = mass[:, index, 0] * derivative[plus] + mass[:, index, 1] * derivative[minus]
            return math.fsum((self.device.ion_geometry.cell_measures_m * rate).tolist())
        derivative = self._derivative(ydot)[self.device.ion_species_slices[index]]
        return math.fsum((self.device.ion_geometry.cell_measures_m * self._ion_partial(populations[index], species) * derivative).tolist())
