"""Compiled 1D state, layout metadata, and constitutive bindings for dynamic defects.

Supports:
- Stationary QSS bulk defects (charged/distributed/spatial monovalent, multivalent,
  and frozen metastable configurations);
- Dynamic bulk trap occupancy modes (interior active nodes, logit-increment DOFs);
- Dynamic two-sided heterointerface trap occupancy DOFs with equilibrium-referenced
  or off sheet-charge closure;
- Coupled mobile ions + dynamic bulk/interface defects in one unified state vector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from typing import Any, Callable, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import frozen_array
from solarlab.contracts.arrays import integer_indices
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.device.defects import MetastablePreparationInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CompiledDevice1D, State1D
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, IonicState1D
from solarlab.mesh.geometry import ControlVolumeGeometry1D, TWO_SIDED_TRACE_V1
from solarlab.physics.defects import (
    DynamicBulkTrapLayout,
    DynamicInterfaceTrapSpec,
    FrozenMetastableRegion,
    MetastableDefectSpec,
    MonovalentDefectRegion,
    MultivalentDefectRegion,
    bulk_trap_charge_density,
    evaluate_frozen_metastable_region,
    evaluate_monovalent_region_closure,
    evaluate_multivalent_region_closure,
    logit_occupancy_secant,
    occupancy_from_logit_increment,
    one_minus_occupancy_from_logit_increment,
    stable_expit,
)
from solarlab.physics.electrostatics.charge import ChargeEntry, ChargeLedger
from solarlab.physics.electrostatics.interface_traces import TraceElectrostatics
from solarlab.physics.interface_transport import TwoSidedInterfacePhysics
from solarlab.physics.ions import (
    DIFFUSION_ONLY_STERIC_V1,
    WHOLE_FLUX_STERIC_V1,
    shared_site_densities,
    shifted_logistic_density,
    single_site_density,
)
from solarlab.physics.ions.shared_site import (
    shared_site_coordinate_hessian_m3,
    shared_site_coordinate_jacobian_m3,
)

Vector: TypeAlias = NDArray[np.float64]

__all__ = [
    "CompiledDynamicDefectDevice1D",
    "CompiledMetastableDevice1D",
    "DynamicDefectState1D",
    "DynamicInterfaceTrapBinding1D",
    "StationaryDefectEvaluation1D",
    "StationaryDefectModel1D",
    "TwoSidedIonicComponent1D",
]


@dataclass(frozen=True, slots=True)
class StationaryDefectEvaluation1D:
    """Grid-summed stationary defect space charge, recombination, and exact partials."""

    charge_density_C_m3: Vector
    d_charge_dn_C: Vector
    d_charge_dp_C: Vector
    d_charge_dphi_fixed_qf_C_m3_V: Vector
    recombination_rate_m3_s: Vector
    d_recombination_dn_s1: Vector
    d_recombination_dp_s1: Vector


@dataclass(frozen=True, slots=True)
class StationaryDefectModel1D:
    """Stationary QSS bulk defect model (monovalent, multivalent, frozen metastable)."""

    node_count: int
    identity_sha256: str
    monovalent_regions: tuple[MonovalentDefectRegion, ...] = ()
    multivalent_regions: tuple[MultivalentDefectRegion, ...] = ()
    metastable_regions: tuple[FrozenMetastableRegion, ...] = ()

    def __post_init__(self) -> None:
        if self.node_count < 2:
            raise ValueError("StationaryDefectModel1D requires node_count >= 2")
        if not self.identity_sha256:
            raise ValueError("StationaryDefectModel1D requires identity_sha256")
        if not (
            self.monovalent_regions
            or self.multivalent_regions
            or self.metastable_regions
        ):
            raise ValueError("StationaryDefectModel1D requires at least one region")

    def evaluate(
        self,
        electron_density_m3: ArrayLike,
        hole_density_m3: ArrayLike,
    ) -> StationaryDefectEvaluation1D:
        """Evaluate net stationary defect space charge and recombination on all nodes."""
        n = np.asarray(electron_density_m3, dtype=np.float64)
        p = np.asarray(hole_density_m3, dtype=np.float64)
        if n.shape != (self.node_count,) or p.shape != (self.node_count,):
            raise ValueError("carrier vectors must match StationaryDefectModel1D.node_count")

        rho: Vector = np.zeros(self.node_count, dtype=np.float64)
        drho_dn: Vector = np.zeros(self.node_count, dtype=np.float64)
        drho_dp: Vector = np.zeros(self.node_count, dtype=np.float64)
        drho_dphi: Vector = np.zeros(self.node_count, dtype=np.float64)
        rec: Vector = np.zeros(self.node_count, dtype=np.float64)
        drec_dn: Vector = np.zeros(self.node_count, dtype=np.float64)
        drec_dp: Vector = np.zeros(self.node_count, dtype=np.float64)

        for mono in self.monovalent_regions:
            ev_m = evaluate_monovalent_region_closure(mono, n, p)
            mask = mono.active_nodes
            rho[mask] += ev_m.total_charge_density_C_m3
            drho_dn[mask] += ev_m.total_charge_derivative_n_C
            drho_dp[mask] += ev_m.total_charge_derivative_p_C
            drho_dphi[mask] += ev_m.total_charge_derivative_fixed_qf_C_m3_V
            rec[mask] += ev_m.total_recombination_rate_m3_s
            drec_dn[mask] += ev_m.total_recombination_derivative_n_s1
            drec_dp[mask] += ev_m.total_recombination_derivative_p_s1

        for mv in self.multivalent_regions:
            mask = mv.active_nodes
            for ev_v in evaluate_multivalent_region_closure(mv, n, p):
                rho[mask] += ev_v.charge_density_C_m3
                drho_dn[mask] += ev_v.charge_derivative_n_C
                drho_dp[mask] += ev_v.charge_derivative_p_C
                drho_dphi[mask] += ev_v.charge_derivative_fixed_qf_C_m3_V
                rec[mask] += ev_v.total_recombination_rate_m3_s
                drec_dn[mask] += ev_v.total_recombination_derivative_n_s1
                drec_dp[mask] += ev_v.total_recombination_derivative_p_s1

        for meta in self.metastable_regions:
            ev_f = evaluate_frozen_metastable_region(meta, n, p)
            rho += ev_f.total_charge_density_C_m3
            drho_dn += ev_f.total_charge_derivative_n_C
            drho_dp += ev_f.total_charge_derivative_p_C
            drho_dphi += ev_f.total_charge_derivative_fixed_qf_C_m3_V
            rec += ev_f.total_recombination_rate_m3_s
            drec_dn += ev_f.total_recombination_derivative_n_s1
            drec_dp += ev_f.total_recombination_derivative_p_s1

        for arr in (rho, drho_dn, drho_dp, drho_dphi, rec, drec_dn, drec_dp):
            arr.setflags(write=False)

        return StationaryDefectEvaluation1D(
            charge_density_C_m3=rho,
            d_charge_dn_C=drho_dn,
            d_charge_dp_C=drho_dp,
            d_charge_dphi_fixed_qf_C_m3_V=drho_dphi,
            recombination_rate_m3_s=rec,
            d_recombination_dn_s1=drec_dn,
            d_recombination_dp_s1=drec_dp,
        )


@dataclass(frozen=True, slots=True)
class DynamicInterfaceTrapBinding1D:
    """Two-sided heterointerface with an explicit dynamic trap occupancy DOF."""

    interface_index: int
    interface_id: str
    face: int
    left_node: int
    right_node: int
    trace_electrostatics: TraceElectrostatics
    physics: TwoSidedInterfacePhysics
    trap_spec: DynamicInterfaceTrapSpec
    reference_logit: float
    weight_left: float = 0.0
    weight_right: float = 0.0

    def __post_init__(self) -> None:
        if self.right_node != self.left_node + 1 or self.face != self.left_node:
            raise ValueError("dynamic two-sided interface requires adjacent left/right nodes")
        if not math.isfinite(self.reference_logit):
            raise ValueError("reference_logit must be finite")
        if self.trap_spec.sheet_density_m2 <= 0.0 or (
            self.physics.surface_recombination_velocity_n_m_s <= 0.0
            and self.physics.surface_recombination_velocity_p_m_s <= 0.0
        ):
            raise ValueError("dynamic interface trap requires positive sheet density and capture")

    @property
    def has_sheet_charge(self) -> bool:
        return bool(self.trap_spec.equilibrium_referenced_charge)

    @property
    def trap_density_m2(self) -> float:
        return float(self.trap_spec.sheet_density_m2)

    @property
    def node_set(self) -> str:
        return f"iface_trap:{self.interface_index}"

    @property
    def block(self) -> str:
        return f"iface_trap:{self.interface_index}"

    @property
    def left_port(self) -> str:
        return f"iface_left:{self.interface_index}"

    def occupancy_from_increment(self, increment: float) -> float:
        inc = float(increment)
        if not math.isfinite(inc):
            raise ValueError("interface trap logit increment must be finite")
        return float(stable_expit(self.reference_logit + inc))

    def one_minus_occupancy_from_increment(self, increment: float) -> float:
        inc = float(increment)
        if not math.isfinite(inc):
            raise ValueError("interface trap logit increment must be finite")
        return float(stable_expit(-(self.reference_logit + inc)))

    def occupancy_minus_equilibrium_from_increment(self, increment: float) -> float:
        inc = float(increment)
        if not math.isfinite(inc):
            raise ValueError("interface trap logit increment must be finite")
        return float(
            logit_occupancy_secant(
                np.array([self.reference_logit + inc]),
                np.array([self.reference_logit]),
            )[0]
        )

    def sheet_charge_C_m2(self, occupancy: float) -> float:
        if not self.has_sheet_charge:
            return 0.0
        return -Q * self.trap_density_m2 * (
            float(occupancy) - self.trap_spec.equilibrium_occupancy
        )

    def sheet_charge_from_increment_C_m2(self, increment: float) -> float:
        if not self.has_sheet_charge:
            return 0.0
        return -Q * self.trap_density_m2 * self.occupancy_minus_equilibrium_from_increment(
            increment
        )


@dataclass(frozen=True)
class DynamicDefectState1D:
    """Physical trial/accepted state bound to CompiledDynamicDefectDevice1D."""

    u: Vector
    electronic_state: State1D
    device_sha256: str
    voltage_V: float
    ionic_state: IonicState1D | None = None
    bulk_trap_occupancy: Vector = field(default_factory=lambda: np.empty(0))
    interface_trap_occupancy: Vector = field(default_factory=lambda: np.empty(0))
    interface_sheet_charges_C_m2: Vector = field(default_factory=lambda: np.empty(0))

    def __post_init__(self) -> None:
        object.__setattr__(self, "u", frozen_array(self.u))
        object.__setattr__(self, "bulk_trap_occupancy", frozen_array(self.bulk_trap_occupancy))
        object.__setattr__(
            self, "interface_trap_occupancy", frozen_array(self.interface_trap_occupancy)
        )
        object.__setattr__(
            self,
            "interface_sheet_charges_C_m2",
            frozen_array(self.interface_sheet_charges_C_m2),
        )
        if not self.device_sha256 or not math.isfinite(self.voltage_V):
            raise ValueError("DynamicDefectState1D requires device_sha256 and finite voltage_V")


@dataclass(frozen=True)
class TwoSidedIonicComponent1D:
    """Blocking single-layer finite-site ion domain on a TWO_SIDED_TRACE_V1 grid."""

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
        "two_sided_trace_electronic_component",
        "two_sided_ionic_1d_residual",
        "finite_sites",
        "blocking_ion_contacts",
        "physical_ion_storage",
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
        if (
            indices.shape != (count,)
            or count < 3
            or np.any(np.diff(indices) != 1)
            or indices[0] < 0
            or indices[-1] >= self.node_count
        ):
            raise ValueError("two-sided ion-node map must be ordered, contiguous, >=3 nodes, inside grid")
        object.__setattr__(self, "ion_node_indices", np.frombuffer(indices.tobytes(), dtype=np.intp))
        layer_ids = [layer.name for layer in self.electronic_component.parameters.layers]
        if layer_ids.count(self.domain_layer_id) != 1:
            raise ValueError("two-sided ionic binding: domain layer is absent from electronic component")
        layer_index = layer_ids.index(self.domain_layer_id)
        planes = self.electronic_component.geometry.interfaces
        left = 0 if layer_index == 0 else planes[layer_index - 1].right_node
        right = self.node_count - 1 if layer_index == len(layer_ids) - 1 else planes[layer_index].left_node
        if not np.array_equal(indices, np.arange(left, right + 1)):
            raise ValueError("two-sided ionic binding: node map must cover the layer's two-sided trace nodes")
        if self.electronic_component.geometry.name != TWO_SIDED_TRACE_V1 or self.ion_geometry.name != TWO_SIDED_TRACE_V1:
            raise ValueError("TwoSidedIonicComponent1D requires TWO_SIDED_TRACE_V1 electronic and ion geometry")
        if not np.array_equal(self.ion_geometry.nodes_m, self.nodes_m[indices]):
            raise ValueError("ion and electronic nodes must coincide")
        if not np.array_equal(
            self.ion_geometry.cell_measures_m,
            self.electronic_component.geometry.cell_measures_m[indices],
        ):
            raise ValueError("two-sided ion cell measures must match electronic cell measures")
        for name in ("positive_ion_reference_m3", "positive_ion_site_limit_m3", "positive_ion_logit_reference"):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} must match the declared ion nodes")
        object.__setattr__(
            self, "positive_ion_diffusivity_faces_m2_s", frozen_array(self.positive_ion_diffusivity_faces_m2_s)
        )
        if self.positive_ion_diffusivity_faces_m2_s.shape != (count - 1,) or np.any(
            self.positive_ion_diffusivity_faces_m2_s < 0.0
        ):
            raise ValueError("ionic diffusivity must be non-negative on every domain face")
        for name in (
            "negative_ion_reference_m3",
            "negative_ion_site_limit_m3",
            "negative_ion_logit_reference",
            "negative_ion_diffusivity_faces_m2_s",
        ):
            object.__setattr__(self, name, frozen_array(getattr(self, name)))
        neg_count = count if self.negative_ion_reference_m3.size else 0
        if (
            any(
                getattr(self, name).shape != (neg_count,)
                for name in (
                    "negative_ion_reference_m3",
                    "negative_ion_site_limit_m3",
                    "negative_ion_logit_reference",
                )
            )
            or self.negative_ion_diffusivity_faces_m2_s.shape != (count - 1 if neg_count else 0,)
        ):
            raise ValueError("negative-ion arrays must all match the complete domain or all be absent")
        if np.any(self.negative_ion_diffusivity_faces_m2_s < 0.0):
            raise ValueError("negative-ion diffusivity must be non-negative on every domain face")
        if self.ion_site_contract == "shared_finite_sites" and not neg_count:
            raise ValueError("shared sites require both complete populations")
        if np.any(self.positive_ion_reference_m3 <= 0.0) or np.any(
            self.positive_ion_reference_m3 >= self.positive_ion_site_limit_m3
        ):
            raise ValueError("finite-site reference must lie strictly between zero and capacity")
        if self.steric_model not in {WHOLE_FLUX_STERIC_V1, DIFFUSION_ONLY_STERIC_V1}:
            raise ValueError("unsupported ionic constitutive law")
        if not self.sha256 or not self.original_prepared_sha256 or not self.domain_layer_id:
            raise ValueError("two-sided ionic component needs complete identities")

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
        start, m_nodes = 3 * self.node_count, self.ion_node_count
        return tuple(
            slice(start + index * m_nodes, start + (index + 1) * m_nodes)
            for index in range(self.ion_species_count)
        )

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
        return frozen_array(
            self.ion_geometry.cell_measures_m
            / self.electronic_component.geometry.cell_measures_m[self.ion_node_indices]
        )

    def project_ion_density(self, density: ArrayLike) -> Vector:
        values = np.asarray(density, dtype=float)
        if values.shape != (self.ion_node_count,) or not np.all(np.isfinite(values)):
            raise ValueError("ion density projection requires finite domain-local values")
        result = np.zeros(self.node_count)
        result[self.ion_node_indices] = self.ion_charge_weight_ratio * values
        return result

    @property
    def binding_manifest(self) -> dict[str, object]:
        loaded = json.loads(self.binding_manifest_json)
        assert isinstance(loaded, dict)
        return loaded

    def fields(self, y: ArrayLike, *, voltage_V: float = 0.0) -> tuple[Vector, Vector, Vector, Vector]:
        if self.ion_species_count != 1:
            raise ValueError("dual-ion fields require fields_by_species")
        phi, n, p, populations = self.fields_by_species(y, voltage_V=voltage_V)
        return phi, n, p, populations[0]

    def fields_by_species(
        self, y: ArrayLike, *, voltage_V: float = 0.0
    ) -> tuple[Vector, Vector, Vector, tuple[Vector, ...]]:
        values = np.asarray(y, dtype=float)
        count = self.node_count
        if values.shape != (self.state_size,) or not np.all(np.isfinite(values)):
            raise ValueError("ionic coordinates must be finite with shape (3*N + species_count*M,)")
        phi, n, p = self.electronic_component.fields(values[: 3 * count], voltage_V=voltage_V)
        populations: list[Vector] = []
        if self.ion_site_contract == "shared_finite_sites":
            positive, negative = self.ion_species_slices
            populations.extend(
                shared_site_densities(
                    self.positive_ion_site_limit_m3,
                    self.positive_ion_logit_reference,
                    self.negative_ion_logit_reference,
                    values[positive],
                    values[negative],
                )
            )
            if np.any(populations[0] + populations[1] >= self.positive_ion_site_limit_m3):
                raise ValueError("shared-site trial summed occupancy exceeded site capacity")
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

    def ion_coordinate_jacobian_m3(self, y: ArrayLike, *, voltage_V: float = 0.0) -> Vector:
        populations = self.fields_by_species(y, voltage_V=voltage_V)[3]
        if self.ion_site_contract == "shared_finite_sites":
            return shared_site_coordinate_jacobian_m3(
                populations[0], populations[1], self.positive_ion_site_limit_m3
            )
        result = np.zeros((self.ion_node_count, self.ion_species_count, self.ion_species_count))
        for index, (species, density) in enumerate(zip(self.ion_species_names, populations)):
            result[:, index, index] = density * (
                1.0 - density / getattr(self, species + "_ion_site_limit_m3")
            )
        return result

    def ion_coordinate_hessian_m3(self, y: ArrayLike, *, voltage_V: float = 0.0) -> Vector:
        populations = self.fields_by_species(y, voltage_V=voltage_V)[3]
        if self.ion_site_contract == "shared_finite_sites":
            return shared_site_coordinate_hessian_m3(
                populations[0], populations[1], self.positive_ion_site_limit_m3
            )
        result = np.zeros(
            (self.ion_node_count, self.ion_species_count, self.ion_species_count, self.ion_species_count)
        )
        for index, (species, density) in enumerate(zip(self.ion_species_names, populations)):
            capacity = getattr(self, species + "_ion_site_limit_m3")
            result[:, index, index, index] = (
                density * (1.0 - density / capacity) * (1.0 - 2.0 * density / capacity)
            )
        return result

    def state(self, y: ArrayLike, *, voltage_V: float = 0.0) -> IonicState1D:
        values = np.asarray(y, dtype=float)
        _, _, _, populations = self.fields_by_species(values, voltage_V=voltage_V)
        component = self.electronic_component.state(values[: 3 * self.node_count], voltage_V=voltage_V)
        inventory = math.fsum((self.ion_geometry.cell_measures_m * populations[0]).tolist())
        negative = populations[1] if self.ion_species_count == 2 else np.empty(0)
        neg_inv = (
            math.fsum((self.ion_geometry.cell_measures_m * negative).tolist())
            if negative.size
            else 0.0
        )
        return IonicState1D(values, component, populations[0], self.identity, inventory, negative, neg_inv)

    def initial_state(self, *, voltage_V: float = 0.0) -> IonicState1D:
        left, right = self.electronic_component.contact_potentials(voltage_V)
        position = (self.nodes_m - self.nodes_m[0]) / self.electronic_component.geometry.extent_m
        phi = left + (right - left) * position
        return self.state(
            np.r_[phi, np.zeros(2 * self.node_count + self.ion_species_count * self.ion_node_count)],
            voltage_V=voltage_V,
        )

    def charge_ledger(
        self, n: Vector, p: Vector, density: Vector, negative_density: Vector | None = None
    ) -> ChargeLedger:
        component = self.electronic_component.charge_ledger(n, p)
        ion = ChargeEntry(
            "positive_ion",
            "ion_charge_projection",
            1,
            self.project_ion_density(density),
            self.project_ion_density(self.positive_ion_reference_m3),
        )
        ledger = ChargeLedger(
            self.node_count,
            component.interface_count,
            component.entries[:2] + (ion,) + component.entries[2:],
        )
        if self.ion_species_count == 2:
            if negative_density is None:
                raise ValueError("dual-ion charge requires the declared negative population")
            negative = ChargeEntry(
                "negative_ion",
                "ion_charge_projection",
                -1,
                self.project_ion_density(negative_density),
                self.project_ion_density(self.negative_ion_reference_m3),
                group="negative_ion_space_charge",
            )
            return ledger.with_entry(negative)
        if negative_density is not None:
            raise ValueError("negative charge is not assembled for this positive-only device")
        return ledger


@dataclass(frozen=True)
class CompiledDynamicDefectDevice1D:
    """Unified compiled 1D device supporting stationary, dynamic, and ionic-defect models."""

    device_id: str
    sha256: str
    original_prepared_sha256: str
    electronic_component: CompiledDevice1D
    ionic_component: CompiledIonicDevice1D | TwoSidedIonicComponent1D | None = None
    stationary_defects: StationaryDefectModel1D | None = None
    dynamic_bulk_traps: DynamicBulkTrapLayout | None = None
    bulk_trap_reference_logit: Vector = field(default_factory=lambda: np.empty(0))
    dynamic_interface_traps: tuple[DynamicInterfaceTrapBinding1D, ...] = ()
    binding_manifest_json: str = "{}"
    capabilities: tuple[str, ...] = ()
    dark_equilibrium_seed: Vector = field(default_factory=lambda: np.empty(0), repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "bulk_trap_reference_logit", frozen_array(self.bulk_trap_reference_logit)
        )
        object.__setattr__(
            self, "dark_equilibrium_seed", frozen_array(self.dark_equilibrium_seed)
        )
        if self.dark_equilibrium_seed.size not in {0, self.state_size}:
            raise ValueError("dark_equilibrium_seed must be empty or match state_size")
        if self.ionic_component is not None:
            if (
                self.ionic_component.electronic_component.identity
                != self.electronic_component.identity
            ):
                raise ValueError(
                    "ionic_component must share the exact electronic_component"
                )
        if self.dynamic_bulk_traps is not None:
            if self.dynamic_bulk_traps.node_count != self.electronic_component.node_count:
                raise ValueError("dynamic_bulk_traps.node_count must match electronic_component")
            if self.bulk_trap_reference_logit.shape != (self.dynamic_bulk_traps.size,):
                raise ValueError("bulk_trap_reference_logit must match dynamic_bulk_traps.size")
        elif self.bulk_trap_reference_logit.size != 0:
            raise ValueError("absent dynamic_bulk_traps requires empty bulk_trap_reference_logit")
        if self.stationary_defects is not None:
            if self.stationary_defects.node_count != self.electronic_component.node_count:
                raise ValueError("stationary_defects.node_count must match electronic_component")
        seen_ifaces = set()
        for itrap in self.dynamic_interface_traps:
            if itrap.interface_index in seen_ifaces:
                raise ValueError("duplicate dynamic interface trap binding")
            seen_ifaces.add(itrap.interface_index)
            if not 0 < itrap.left_node < itrap.right_node < self.node_count - 1:
                raise ValueError("dynamic interface trap nodes must be interior")

    @property
    def identity(self) -> str:
        return self.sha256

    @property
    def node_count(self) -> int:
        return self.electronic_component.node_count

    @property
    def nodes_m(self) -> Vector:
        return self.electronic_component.nodes_m

    @property
    def thermal_voltage_V(self) -> float:
        return self.electronic_component.thermal_voltage_V

    @property
    def voltage_polarity(self) -> float:
        return self.electronic_component.voltage_polarity

    @property
    def has_ions(self) -> bool:
        return self.ionic_component is not None

    @property
    def has_dynamic_bulk_traps(self) -> bool:
        return self.dynamic_bulk_traps is not None

    @property
    def has_dynamic_interface_traps(self) -> bool:
        return len(self.dynamic_interface_traps) > 0

    @property
    def has_stationary_defects(self) -> bool:
        return self.stationary_defects is not None

    @property
    def electronic_slice(self) -> slice:
        return slice(0, 3 * self.node_count)

    @property
    def ion_dof_count(self) -> int:
        if self.ionic_component is None:
            return 0
        return self.ionic_component.state_size - 3 * self.node_count

    @property
    def ion_slice(self) -> slice:
        start = 3 * self.node_count
        return slice(start, start + self.ion_dof_count)

    @property
    def bulk_trap_dof_count(self) -> int:
        if self.dynamic_bulk_traps is None:
            return 0
        return self.dynamic_bulk_traps.size

    @property
    def bulk_trap_slice(self) -> slice:
        start = 3 * self.node_count + self.ion_dof_count
        return slice(start, start + self.bulk_trap_dof_count)

    @property
    def bulk_trap_mode_slices(self) -> tuple[slice, ...]:
        if self.dynamic_bulk_traps is None:
            return ()
        base = 3 * self.node_count + self.ion_dof_count
        return tuple(
            slice(base + m.flat_slice.start, base + m.flat_slice.stop)
            for m in self.dynamic_bulk_traps.modes
        )

    @property
    def interface_trap_dof_count(self) -> int:
        return 2 * len(self.dynamic_interface_traps)

    @property
    def interface_trap_slice(self) -> slice:
        start = 3 * self.node_count + self.ion_dof_count + self.bulk_trap_dof_count
        return slice(start, start + self.interface_trap_dof_count)

    @property
    def interface_trap_slices(self) -> tuple[slice, ...]:
        base = 3 * self.node_count + self.ion_dof_count + self.bulk_trap_dof_count
        return tuple(
            slice(base + 2 * i, base + 2 * (i + 1))
            for i in range(len(self.dynamic_interface_traps))
        )

    @property
    def state_size(self) -> int:
        return (
            3 * self.node_count
            + self.ion_dof_count
            + self.bulk_trap_dof_count
            + self.interface_trap_dof_count
        )

    @property
    def binding_manifest(self) -> dict[str, Any]:
        loaded = json.loads(self.binding_manifest_json)
        assert isinstance(loaded, dict)
        return loaded

    def rebase_coordinates(
        self, u: ArrayLike, old_voltage_V: float, new_voltage_V: float
    ) -> Vector:
        values = np.array(u, dtype=np.float64, copy=True)
        if values.shape != (self.state_size,):
            raise ValueError("coordinate vector shape does not match state_size")
        values[self.electronic_slice] = self.electronic_component.rebase_coordinates(
            values[self.electronic_slice], old_voltage_V, new_voltage_V
        )
        return values

    def bulk_trap_occupancies(self, u: ArrayLike) -> Vector:
        values = np.asarray(u, dtype=np.float64)
        if values.shape != (self.state_size,):
            raise ValueError("coordinate vector shape does not match state_size")
        if self.dynamic_bulk_traps is None:
            return np.empty(0, dtype=np.float64)
        return occupancy_from_logit_increment(
            self.bulk_trap_reference_logit,
            values[self.bulk_trap_slice],
            self.dynamic_bulk_traps,
        )

    def bulk_trap_one_minus_occupancies(self, u: ArrayLike) -> Vector:
        values = np.asarray(u, dtype=np.float64)
        if values.shape != (self.state_size,):
            raise ValueError("coordinate vector shape does not match state_size")
        if self.dynamic_bulk_traps is None:
            return np.empty(0, dtype=np.float64)
        return one_minus_occupancy_from_logit_increment(
            self.bulk_trap_reference_logit,
            values[self.bulk_trap_slice],
            self.dynamic_bulk_traps,
        )

    def interface_trap_occupancies(self, u: ArrayLike) -> Vector:
        values = np.asarray(u, dtype=np.float64)
        if values.shape != (self.state_size,):
            raise ValueError("coordinate vector shape does not match state_size")
        if not self.dynamic_interface_traps:
            return np.empty(0, dtype=np.float64)
        occ = np.array(
            [
                itrap.occupancy_from_increment(float(values[sl.start]))
                for itrap, sl in zip(self.dynamic_interface_traps, self.interface_trap_slices)
            ],
            dtype=np.float64,
        )
        occ.setflags(write=False)
        return occ

    def interface_trap_one_minus_occupancies(self, u: ArrayLike) -> Vector:
        values = np.asarray(u, dtype=np.float64)
        if values.shape != (self.state_size,):
            raise ValueError("coordinate vector shape does not match state_size")
        if not self.dynamic_interface_traps:
            return np.empty(0, dtype=np.float64)
        omf = np.array(
            [
                itrap.one_minus_occupancy_from_increment(float(values[sl.start]))
                for itrap, sl in zip(self.dynamic_interface_traps, self.interface_trap_slices)
            ],
            dtype=np.float64,
        )
        omf.setflags(write=False)
        return omf

    def interface_sheet_charges_C_m2(self, u: ArrayLike, *, voltage_V: float = 0.0) -> Vector:
        values = np.asarray(u, dtype=np.float64)
        if values.shape != (self.state_size,):
            raise ValueError("coordinate vector shape does not match state_size")
        sheets = np.array(
            self.electronic_component.interface_sheet_charges_C_m2(
                values[self.electronic_slice], voltage_V=voltage_V
            ),
            dtype=np.float64,
            copy=True,
        )
        if self.dynamic_interface_traps:
            for itrap, sl in zip(self.dynamic_interface_traps, self.interface_trap_slices):
                sheets[itrap.interface_index] = itrap.sheet_charge_from_increment_C_m2(
                    float(values[sl.start])
                )
        sheets.setflags(write=False)
        return sheets

    def space_charge_density_C_m3(
        self, n: Vector, p: Vector, u: ArrayLike
    ) -> Vector:
        """Total volumetric space charge density [C m^-3] on all device nodes."""
        values = np.asarray(u, dtype=np.float64)
        if self.ionic_component is not None:
            ion_u = values[: self.ionic_component.state_size]
            _, _, _, pops = self.ionic_component.fields_by_species(ion_u, voltage_V=0.0)
            base_ledger: ChargeLedger = self.ionic_component.charge_ledger(
                n, p, pops[0], pops[1] if len(pops) == 2 else None
            )
            rho = np.array(base_ledger.space_charge_C_m3(), dtype=np.float64, copy=True)
        else:
            rho = np.array(
                self.electronic_component.charge_ledger(n, p).space_charge_C_m3(),
                dtype=np.float64,
                copy=True,
            )
        if self.stationary_defects is not None:
            rho += self.stationary_defects.evaluate(n, p).charge_density_C_m3
        if self.dynamic_bulk_traps is not None:
            occ = self.bulk_trap_occupancies(values)
            omf = self.bulk_trap_one_minus_occupancies(values)
            rho += bulk_trap_charge_density(
                occ, self.dynamic_bulk_traps, one_minus_occupancy=omf
            )
        rho.setflags(write=False)
        return rho

    def state(self, u: ArrayLike, *, voltage_V: float = 0.0) -> DynamicDefectState1D:
        values = np.asarray(u, dtype=np.float64)
        if values.shape != (self.state_size,) or not np.all(np.isfinite(values)):
            raise ValueError("state coordinates must be finite and match state_size")
        elec_state = self.electronic_component.state(
            values[self.electronic_slice], voltage_V=voltage_V
        )
        ion_state: IonicState1D | None = None
        if self.ionic_component is not None:
            ion_state = self.ionic_component.state(
                values[: self.ionic_component.state_size], voltage_V=voltage_V
            )
        bulk_occ = self.bulk_trap_occupancies(values)
        iface_occ = self.interface_trap_occupancies(values)
        sheets = self.interface_sheet_charges_C_m2(values, voltage_V=voltage_V)
        return DynamicDefectState1D(
            u=values,
            electronic_state=elec_state,
            device_sha256=self.sha256,
            voltage_V=float(voltage_V),
            ionic_state=ion_state,
            bulk_trap_occupancy=bulk_occ,
            interface_trap_occupancy=iface_occ,
            interface_sheet_charges_C_m2=sheets,
        )

    def initial_equilibrium_seed(self, *, voltage_V: float = 0.0) -> Vector:
        """Construct zero-excess dark/reference coordinate seed at ``voltage_V``."""
        N = self.node_count
        if self.dark_equilibrium_seed.size == self.state_size:
            seed = np.array(self.dark_equilibrium_seed, dtype=np.float64, copy=True)
            if voltage_V != 0.0:
                seed = self.rebase_coordinates(seed, 0.0, voltage_V)
                phi_l, phi_r = self.electronic_component.contact_potentials(voltage_V)
                dv = (phi_r - phi_l) - (seed[N - 1] - seed[0])
                nodes = self.nodes_m
                seed[:N] += dv * (nodes - nodes[0]) / (nodes[-1] - nodes[0])
            return seed
        seed = np.zeros(self.state_size, dtype=np.float64)
        phi_l, phi_r = self.electronic_component.contact_potentials(voltage_V)
        nodes = self.nodes_m
        seed[:N] = phi_l + (phi_r - phi_l) * (nodes - nodes[0]) / (nodes[-1] - nodes[0])
        if voltage_V != 0.0:
            seed = self.rebase_coordinates(seed, 0.0, voltage_V)
            seed[:N] = phi_l + (phi_r - phi_l) * (nodes - nodes[0]) / (nodes[-1] - nodes[0])
        return seed


@dataclass(frozen=True)
class CompiledMetastableDevice1D:
    """Compiled metastable bulk-defect model supporting preparation, frozen J-V, and transients."""

    device_id: str
    sha256: str
    prepared: PreparedDevice
    metastable_layer_id: str
    metastable_layer_index: int
    metastable_specs: tuple[MetastableDefectSpec, ...]
    preparation: MetastablePreparationInput | None
    intervals_per_layer: int | tuple[int, ...]
    base_electronic_device: CompiledDevice1D
    active_nodes: NDArray[np.bool_]
    band_gap_eV: Vector
    conduction_dos_m3: Vector
    valence_dos_m3: Vector
    temperature_K: float
    freeze_builder: Callable[
        [Mapping[str, ArrayLike] | Sequence[ArrayLike], float | None],
        CompiledDynamicDefectDevice1D,
    ] = field(repr=False, compare=False)

    @property
    def identity(self) -> str:
        return self.sha256

    @property
    def active_node_count(self) -> int:
        return int(np.count_nonzero(self.active_nodes))

    @property
    def node_count(self) -> int:
        return self.base_electronic_device.node_count

    def compile_frozen_device(
        self,
        donor_fractions: Mapping[str, ArrayLike] | Sequence[ArrayLike],
        *,
        temperature_K: float | None = None,
    ) -> CompiledDynamicDefectDevice1D:
        """Compile a stationary frozen-configuration device at ``donor_fractions``."""
        return self.freeze_builder(donor_fractions, temperature_K)
