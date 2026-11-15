"""Bind a PreparedDevice to the unified dynamic-defect and metastable 1D models (P04-10).

Supported (everything else raises :class:`CapabilityError` before any solve):

* explicit monovalent bulk defects (charged, neutral, distributed, spatially graded)
  and legacy ``bulk_trap_distribution`` declarations as stationary quasi-steady
  inventories (``dynamic_bulk_defects=False``) or as explicit trap-occupancy
  storage DOFs on interior active nodes (``dynamic_bulk_defects=True``);
* explicit multivalent / amphoteric bulk defects (``solarlab-explicit-bulk-defects-v4``)
  in stationary grand-partition closure (``dynamic_bulk_defects=False``);
* contact-layer equilibrium charge-neutrality rebinding when an outer electrical
  layer carries explicit charged/distributed/multivalent bulk defects;
* dynamic two-sided heterointerface single-level traps with equilibrium-referenced
  or off sheet-charge closure (``dynamic_interface_defects=True``), including
  stacks mixing microscopic defect interfaces with passive interfaces;
* mobile finite-site ions coupled to bulk and/or interface defects on both the
  deduplicated topology (:class:`CompiledIonicDevice1D`) and the two-sided trace
  topology (:class:`TwoSidedIonicComponent1D`);
* two-configuration metastable bulk-defect inventories compiled via
  :func:`compile_metastable_device_1d` into :class:`CompiledMetastableDevice1D`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import math
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import ION_INPUTS, compile_ionic_device_1d
from solarlab.device.defects import (
    LegacyBulkTrapInput,
    MetastableDocumentInput,
    MetastablePreparationInput,
)
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice, PreparedLayer, PreparedMultivalentDefect
from solarlab.equations.dc_1d import (
    CapabilityError,
    CompiledDevice1D,
    CompiledInterfaceModel1D,
    TERegularizationName,
    TwoSidedInterfaceBindingSpec,
)
from solarlab.equations.dynamic_defect_1d import (
    CompiledDynamicDefectDevice1D,
    CompiledMetastableDevice1D,
    DynamicInterfaceTrapBinding1D,
    StationaryDefectModel1D,
    TwoSidedIonicComponent1D,
)
from solarlab.equations.ionic_1d import CompiledIonicDevice1D
from solarlab.mesh import RIGHT_LAYER_WINS
from solarlab.mesh.geometry import (
    ControlVolumeGeometry1D,
    NO_SHARED_NODES,
    TWO_SIDED_TRACE_V1,
)
from solarlab.mesh.grid import layer_node_masks
from solarlab.physics.band_structure.temperature import ion_diffusivity_at_temperature
from solarlab.physics.contacts.exchange import (
    STATISTICS_CLOSURE,
    CarrierExchangeChannel,
)
from solarlab.physics.defects import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    DefectDistributionSpec,
    DynamicInterfaceTrapSpec,
    FrozenMetastableRegion,
    MetastableConversionKineticsSpec,
    MetastableDefectSpec,
    MonovalentDefectRegion,
    MonovalentSpeciesSpec,
    MultivalentConfigurationSpec,
    MultivalentDefectRegion,
    MultivalentSpeciesSpec,
    MultivalentTransitionKineticsSpec,
    SpatialDensityProfileSpec,
    SpatialProfileKnot,
    WIDTH_GAUSSIAN_SIGMA,
    WIDTH_NOT_APPLICABLE,
    compile_dynamic_bulk_trap_layout,
    quasi_steady_bulk_trap_logit,
    solve_monovalent_defect_charge_neutrality,
    solve_multivalent_defect_charge_neutrality,
    validate_defect_energy_quadrature_order,
)
from solarlab.physics.electrostatics.interface_traces import sheet_charge_row_weights
from solarlab.physics.interface_transport import (
    EquilibriumReferencedSheetCharge,
    TwoSidedBulkState,
    solve_two_sided_interface_qss,
)
from solarlab.physics.ions import (
    DIFFUSION_ONLY_STERIC_V1,
    HISTORICAL_OCCUPANCY_CLIP,
    HISTORICAL_STERIC_FLOOR,
    WHOLE_FLUX_STERIC_V1,
)
from solarlab.physics.recombination import (
    BulkRecombinationCoefficients,
    BulkRecombinationReaction,
)

__all__ = [
    "CapabilityError",
    "CompiledDynamicDefectDevice1D",
    "CompiledMetastableDevice1D",
    "compile_dynamic_defect_device_1d",
    "compile_metastable_device_1d",
]

BINDING_SCHEMA = "solarlab.dynamic_defect_1d.binding.v2"
METASTABLE_BINDING_SCHEMA = "solarlab.metastable_defect_1d.binding.v1"
InterfaceTopologyName = Literal["deduplicated", "two_sided_trace"]


def _canonical(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
        ).encode()
    ).hexdigest()


def _mask_lifetime_srh(
    component: CompiledDevice1D, explicit_srh_nodes: NDArray[np.bool_]
) -> CompiledDevice1D:
    """Disable lifetime-form SRH on nodes owned by explicit bulk/metastable defects."""
    owned = np.asarray(explicit_srh_nodes, dtype=bool)
    if not np.any(owned):
        return component
    coeff = component.recombination.coefficients
    new_coeff = BulkRecombinationCoefficients(
        node_count=coeff.node_count,
        ni_sq_m6=coeff.ni_sq_m6,
        tau_n_s=np.where(owned, np.inf, coeff.tau_n_s),
        tau_p_s=np.where(owned, np.inf, coeff.tau_p_s),
        n1_m3=np.where(owned, np.maximum(coeff.n1_m3, 1.0), coeff.n1_m3),
        p1_m3=np.where(owned, np.maximum(coeff.p1_m3, 1.0), coeff.p1_m3),
        radiative_m3_s=coeff.radiative_m3_s,
        auger_n_m6_s=coeff.auger_n_m6_s,
        auger_p_m6_s=coeff.auger_p_m6_s,
    )
    new_recomb = BulkRecombinationReaction(
        new_coeff, component.recombination.explicit_defects
    )
    new_sha = _canonical(
        ["explicit_srh_mask_v1", component.identity, np.flatnonzero(owned).tolist()]
    )
    return replace(component, sha256=new_sha, recombination=new_recomb)


def _logit(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise CapabilityError(
            f"interface reference occupancy {value!r} saturated outside (0, 1)"
        )
    return math.log(value) - math.log1p(-value)


def _species_from_item(layer_id: str, item: dict[str, Any]) -> MonovalentSpeciesSpec:
    where = f"layers.{layer_id}.bulk_defects.{item['id']}"
    distribution = item["distribution"]
    kinetics = item["kinetics"]
    if item["charge_transition"] not in ("acceptor", "donor", "neutral"):
        raise CapabilityError(
            f"{where}: charge_transition must be acceptor, donor, or neutral"
        )
    if float(item["degeneracy"]) != 1.0:
        raise CapabilityError(f"{where}: monovalent defects require unit degeneracy")
    center = distribution["center_eV_above_vb"]
    if center is None:
        raise CapabilityError(
            f"{where}: a resolved energy above the valence band is required"
        )
    try:
        spec = DefectDistributionSpec(
            kind=str(distribution["kind"]),
            center_eV_above_vb=float(center),
            total_density_m3=float(distribution["total_density_m3"]),
            width_eV=(
                0.0
                if distribution.get("width_eV") is None
                else float(distribution["width_eV"])
            ),
            width_convention=(
                "not_applicable"
                if distribution.get("width_convention") in (None, "not_applicable")
                else str(distribution["width_convention"])
            ),
            support_width_multiplier=(
                None
                if distribution.get("support_width_multiplier") is None
                else float(distribution["support_width_multiplier"])
            ),
        )
        profile = item.get("spatial_profile")
        spatial = None
        if profile is not None:
            spatial = SpatialDensityProfileSpec(
                tuple(
                    SpatialProfileKnot(
                        float(knot["position_fraction"]),
                        float(knot["density_multiplier"]),
                    )
                    for knot in profile["knots"]
                )
            )
        return MonovalentSpeciesSpec(
            identifier=f"{layer_id}/{item['id']}",
            charge_transition=str(item["charge_transition"]),
            distribution=spec,
            sigma_n_m2=float(kinetics["sigma_n_m2"]),
            sigma_p_m2=float(kinetics["sigma_p_m2"]),
            thermal_velocity_n_m_s=float(kinetics["thermal_velocity_n_m_s"]),
            thermal_velocity_p_m_s=float(kinetics["thermal_velocity_p_m_s"]),
            spatial_profile=spatial,
        )
    except ValueError as error:
        raise CapabilityError(f"{where}: {error}") from error


def _species_from_legacy_bulk_trap(
    layer_id: str, trap_data: Mapping[str, Any]
) -> MonovalentSpeciesSpec:
    where = f"layers.{layer_id}.bulk_trap_distribution"
    kind = str(trap_data["distribution"])
    try:
        spec = DefectDistributionSpec(
            kind=kind,
            center_eV_above_vb=float(trap_data["center_eV_above_vb"]),
            total_density_m3=float(trap_data["total_density_m3"]),
            width_eV=(
                0.0
                if kind == "single_level"
                else float(trap_data["energy_sigma_eV"])
            ),
            width_convention=(
                WIDTH_NOT_APPLICABLE if kind == "single_level" else WIDTH_GAUSSIAN_SIGMA
            ),
            support_width_multiplier=None,
        )
        v_th = float(trap_data["thermal_velocity_m_s"])
        return MonovalentSpeciesSpec(
            identifier=f"{layer_id}/bulk_trap_distribution",
            charge_transition=str(trap_data["charge_transition"]),
            distribution=spec,
            sigma_n_m2=float(trap_data["sigma_n_m2"]),
            sigma_p_m2=float(trap_data["sigma_p_m2"]),
            thermal_velocity_n_m_s=v_th,
            thermal_velocity_p_m_s=v_th,
        )
    except ValueError as error:
        raise CapabilityError(f"{where}: {error}") from error


def _multivalent_config_from_mapping(
    where: str, cfg: Mapping[str, Any]
) -> MultivalentConfigurationSpec:
    try:
        kinetics = tuple(
            MultivalentTransitionKineticsSpec(
                sigma_n_m2=float(kin["sigma_n_m2"]),
                sigma_p_m2=float(kin["sigma_p_m2"]),
                thermal_velocity_n_m_s=float(kin["thermal_velocity_n_m_s"]),
                thermal_velocity_p_m_s=float(kin["thermal_velocity_p_m_s"]),
            )
            for kin in cfg["transition_kinetics"]
        )
        return MultivalentConfigurationSpec(
            family=str(cfg["family"]),
            charge_states_e=tuple(int(c) for c in cfg["charge_states_e"]),
            degeneracy_convention=str(cfg["degeneracy_convention"]),
            state_degeneracies=tuple(float(g) for g in cfg["state_degeneracies"]),
            transition_energies_eV_above_vb=tuple(
                float(e) for e in cfg["transition_energies_eV_above_vb"]
            ),
            transition_kinetics=kinetics,
        )
    except ValueError as error:
        raise CapabilityError(f"{where}: {error}") from error


def _multivalent_from_item(
    layer_id: str, item: Mapping[str, Any]
) -> MultivalentSpeciesSpec:
    where = f"layers.{layer_id}.bulk_defects.{item['id']}"
    cfg = _multivalent_config_from_mapping(where, item["configuration"])
    try:
        return MultivalentSpeciesSpec(
            identifier=f"{layer_id}/{item['id']}",
            total_density_m3=float(item["total_density_m3"]),
            configuration=cfg,
        )
    except ValueError as error:
        raise CapabilityError(f"{where}: {error}") from error


def _metastable_from_item(
    layer_id: str, item: Mapping[str, Any]
) -> MetastableDefectSpec:
    where = f"layers.{layer_id}.metastable_defects.{item['id']}"
    donor_cfg = _multivalent_config_from_mapping(
        f"{where}.donor_configuration", item["donor_configuration"]
    )
    acceptor_cfg = _multivalent_config_from_mapping(
        f"{where}.acceptor_configuration", item["acceptor_configuration"]
    )
    kin = item["conversion_kinetics"]
    try:
        conv = MetastableConversionKineticsSpec(
            transition_energy_eV_above_vb=float(kin["transition_energy_eV_above_vb"]),
            electron_capture_activation_eV=float(kin["electron_capture_activation_eV"]),
            electron_emission_activation_eV=float(
                kin["electron_emission_activation_eV"]
            ),
            hole_capture_activation_eV=float(kin["hole_capture_activation_eV"]),
            hole_emission_activation_eV=float(kin["hole_emission_activation_eV"]),
            electron_capture_path=str(kin["electron_capture_path"]),
            hole_capture_path=str(kin["hole_capture_path"]),
            capture_n_m3_s=float(kin["capture_n_m3_s"]),
            capture_p_m3_s=float(kin["capture_p_m3_s"]),
            phonon_frequency_Hz=float(kin["phonon_frequency_Hz"]),
        )
        return MetastableDefectSpec(
            identifier=f"{layer_id}/{item['id']}",
            total_density_m3=float(item["total_density_m3"]),
            donor_configuration=donor_cfg,
            acceptor_configuration=acceptor_cfg,
            donor_conversion_state_index=int(item["donor_conversion_state_index"]),
            acceptor_conversion_state_index=int(item["acceptor_conversion_state_index"]),
            conversion_kinetics=conv,
        )
    except ValueError as error:
        raise CapabilityError(f"{where}: {error}") from error


def _compile_two_sided_ionic_component(
    *,
    original_prepared: PreparedDevice,
    component_prepared: PreparedDevice,
    component: CompiledDevice1D,
    projection_changes: list[dict[str, object]],
    layers: tuple[PreparedLayer, ...],
    ion_domain_layer: str | None,
    ion_domain_endpoints: tuple[str, str] | None,
) -> TwoSidedIonicComponent1D:
    """Bind a single-layer blocking finite-site ion domain onto TWO_SIDED_TRACE_V1."""
    data = original_prepared.to_mapping()
    if ion_domain_layer is None:
        mobile_layers = [
            layer
            for layer in layers
            if any(
                float(cast(float, dict(layer.parameters)[name])) != 0.0
                for name in ("D_ion", "D_ion_neg")
            )
        ]
        active_layers = mobile_layers or [
            layer
            for layer in layers
            if any(
                float(cast(float, dict(layer.parameters)[name])) != 0.0
                for name in ("P0", "D_ion", "P0_neg", "D_ion_neg")
            )
        ]
        if len(layers) == 1:
            domain_layer = layers[0]
        elif len(active_layers) == 1 and ion_domain_endpoints == ("blocking", "blocking"):
            domain_layer = active_layers[0]
        else:
            raise CapabilityError(
                "two_sided_ionic_1d: multilayer device requires explicit ion_domain_layer and blocking endpoints"
            )
    else:
        selected = [layer for layer in layers if layer.id == ion_domain_layer]
        if len(selected) != 1:
            raise CapabilityError(
                "two_sided_ionic_1d: ion_domain_layer must name one electrical layer"
            )
        domain_layer = selected[0]

    if ion_domain_endpoints is None:
        if len(layers) != 1:
            raise CapabilityError(
                "two_sided_ionic_1d: multilayer ion domain requires explicit ('blocking', 'blocking') endpoints"
            )
    elif ion_domain_endpoints != ("blocking", "blocking"):
        raise CapabilityError(
            "two_sided_ionic_1d: ion interface exchange is unsupported; declare ('blocking', 'blocking')"
        )

    shared = bool(data["settings"]["ion_steric_shared_site"])
    if shared and not data["settings"]["ion_steric_diffusion_only"]:
        raise CapabilityError(
            "shared-site ions require the original diffusion-only crowding law"
        )
    for layer in original_prepared.layers:
        params = dict(layer.parameters)
        if layer.role == "substrate" and any(
            params[name] != 0.0 for name in ("P0", "D_ion", "P0_neg", "D_ion_neg")
        ):
            raise CapabilityError("two_sided_ionic_1d: substrate ion domains are unsupported")
        if layer.id != domain_layer.id and any(
            params[name] != 0.0 for name in ("D_ion", "D_ion_neg")
        ):
            raise CapabilityError(
                "two_sided_ionic_1d: multiple active ion layers are unsupported"
            )

    parameters = dict(domain_layer.parameters)
    reference = float(cast(float, parameters["P0"]))
    capacity = float(cast(float, parameters["P_lim"]))
    diffusion = float(cast(float, parameters["D_ion"]))
    activation = float(cast(float, parameters["E_a_ion"]))
    if not all(math.isfinite(v) for v in (reference, capacity, diffusion, activation)):
        raise CapabilityError("two_sided_ionic_1d: finite ionic parameters are required")
    if not 0.0 < reference < capacity:
        raise CapabilityError(
            "two_sided_ionic_1d: a strictly positive, unsaturated finite-site reference is required"
        )
    if diffusion < 0.0 or activation < 0.0:
        raise CapabilityError(
            "two_sided_ionic_1d: diffusivity/activation energy must be non-negative"
        )

    negative_reference = float(cast(float, parameters["P0_neg"]))
    negative_capacity = float(cast(float, parameters["P_lim_neg"]))
    negative_diffusion = float(cast(float, parameters["D_ion_neg"]))
    dual = negative_reference != 0.0 or negative_diffusion != 0.0
    if dual:
        if not all(
            math.isfinite(v)
            for v in (negative_reference, negative_capacity, negative_diffusion)
        ):
            raise CapabilityError(
                "two_sided_ionic_1d: finite negative-ion parameters are required"
            )
        if not 0.0 < negative_reference < negative_capacity:
            raise CapabilityError(
                "two_sided_ionic_1d: a positive unsaturated negative-ion reference is required"
            )
        if negative_diffusion < 0.0:
            raise CapabilityError(
                "two_sided_ionic_1d: negative-ion diffusivity must be non-negative"
            )
    if shared:
        if not dual:
            raise CapabilityError(
                "shared-site ions require both declared positive and negative references"
            )
        if capacity != negative_capacity:
            raise CapabilityError(
                "shared-site ions require one exactly common site capacity"
            )
        if (
            not math.isfinite(reference + negative_reference)
            or reference + negative_reference >= capacity
        ):
            raise CapabilityError(
                "shared-site reference summed occupancy must be below the common capacity"
            )

    domain_index = layers.index(domain_layer)
    planes = component.geometry.interfaces
    left = 0 if domain_index == 0 else int(planes[domain_index - 1].right_node)
    right = (
        component.node_count - 1
        if domain_index == len(layers) - 1
        else int(planes[domain_index].left_node)
    )
    indices: np.ndarray = np.arange(left, right + 1, dtype=np.intp)
    if indices.size < 3:
        raise CapabilityError(
            "two_sided_ionic_1d: ion domain requires at least 3 trace nodes"
        )
    ion_geometry = ControlVolumeGeometry1D(
        TWO_SIDED_TRACE_V1,
        component.geometry.nodes_m[indices],
        component.geometry.cell_measures_m[indices],
        component.geometry.cell_faces_m[left : right + 2],
        (),
        NO_SHARED_NODES,
        NO_SHARED_NODES,
    )
    count = ion_geometry.node_count
    scaled_diffusion = ion_diffusivity_at_temperature(
        diffusion, component.parameters.temperature_K, activation
    )
    if not math.isfinite(scaled_diffusion) or scaled_diffusion < 0.0:
        raise CapabilityError(
            "two_sided_ionic_1d: temperature-scaled ion diffusivity is outside the finite envelope"
        )
    occ = reference / capacity
    logit = math.log(occ) - math.log1p(-occ)
    steric_model = (
        DIFFUSION_ONLY_STERIC_V1
        if data["settings"]["ion_steric_diffusion_only"]
        else WHOLE_FLUX_STERIC_V1
    )
    ref_arr: np.ndarray = np.full(count, reference, dtype=np.float64)
    lim_arr: np.ndarray = np.full(count, capacity, dtype=np.float64)
    faces: np.ndarray = np.full(count - 1, scaled_diffusion, dtype=np.float64)
    pos_logit_arr: np.ndarray = np.full(count, logit, dtype=np.float64)
    neg_arrays: dict[str, np.ndarray] = {}
    if dual:
        scaled_neg = ion_diffusivity_at_temperature(
            negative_diffusion, component.parameters.temperature_K, activation
        )
        if not math.isfinite(scaled_neg) or scaled_neg < 0.0:
            raise CapabilityError(
                "two_sided_ionic_1d: temperature-scaled negative diffusivity is outside the finite envelope"
            )
        neg_occ = negative_reference / negative_capacity
        neg_logit = math.log(neg_occ) - math.log1p(-neg_occ)
        neg_arrays = {
            "negative_ion_reference_m3": np.full(count, negative_reference, dtype=np.float64),
            "negative_ion_site_limit_m3": np.full(count, negative_capacity, dtype=np.float64),
            "negative_ion_logit_reference": np.full(count, neg_logit, dtype=np.float64),
            "negative_ion_diffusivity_faces_m2_s": np.full(
                count - 1, scaled_neg, dtype=np.float64
            ),
        }
        if shared:
            vacancy = (
                1.0
                - (ref_arr + neg_arrays["negative_ion_reference_m3"]) / lim_arr
            )
            pos_logit_arr = np.log(ref_arr / lim_arr) - np.log(vacancy)
            neg_arrays["negative_ion_logit_reference"] = np.log(
                neg_arrays["negative_ion_reference_m3"] / lim_arr
            ) - np.log(vacancy)

    declarations = [
        {
            "layer": lyr.id,
            **{name: dict(lyr.parameters)[name] for name in ION_INPUTS},
        }
        for lyr in original_prepared.layers
    ]
    manifest: dict[str, Any] = {
        "binding": (
            "two_sided_shared_site_dual_ion_1d_v1"
            if shared
            else (
                "two_sided_independent_dual_ion_1d_v1"
                if dual
                else "two_sided_single_positive_ion_1d_v1"
            )
        ),
        "original_prepared_sha256": original_prepared.content_sha256,
        "original_ionic_inputs": declarations,
        "electronic_component": {
            "prepared_sha256": component_prepared.content_sha256,
            "compiled_sha256": component.identity,
            "projection": projection_changes,
            "geometry": component.geometry.name,
        },
        "ion_domain": {
            "layer": domain_layer.id,
            "electrical_node_indices": indices.tolist(),
            "geometry": ion_geometry.name,
            "contacts": ["blocking", "blocking"],
            "reference_inventory_m2": math.fsum(
                (ion_geometry.cell_measures_m * ref_arr).tolist()
            ),
        },
        "ion_model": {
            "steric_model": steric_model,
            "occupancy_clip": HISTORICAL_OCCUPANCY_CLIP,
            "steric_floor": HISTORICAL_STERIC_FLOOR,
            "activation_energy_eV": activation,
            "temperature_K": component.parameters.temperature_K,
            "face_diffusivity_m2_s": faces.tolist(),
        },
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    identity = hashlib.sha256(encoded.encode()).hexdigest()
    return TwoSidedIonicComponent1D(
        device_id=str(data["id"]),
        sha256=identity,
        original_prepared_sha256=original_prepared.content_sha256,
        electronic_component=component,
        ion_geometry=ion_geometry,
        domain_layer_id=domain_layer.id,
        positive_ion_reference_m3=ref_arr,
        positive_ion_site_limit_m3=lim_arr,
        positive_ion_logit_reference=pos_logit_arr,
        positive_ion_diffusivity_faces_m2_s=faces,
        steric_model=steric_model,
        binding_manifest_json=encoded,
        ion_node_indices=indices,
        negative_ion_reference_m3=neg_arrays.get("negative_ion_reference_m3", np.empty(0)),
        negative_ion_site_limit_m3=neg_arrays.get("negative_ion_site_limit_m3", np.empty(0)),
        negative_ion_logit_reference=neg_arrays.get(
            "negative_ion_logit_reference", np.empty(0)
        ),
        negative_ion_diffusivity_faces_m2_s=neg_arrays.get(
            "negative_ion_diffusivity_faces_m2_s", np.empty(0)
        ),
        ion_site_contract="shared_finite_sites" if shared else "independent_finite_sites",
    )


def _rebind_contact_neutrality(
    component: CompiledDevice1D,
    *,
    last_layer_index: int,
    layer_species: Mapping[int, Sequence[MonovalentSpeciesSpec]],
    layer_multivalent: Mapping[int, Sequence[MultivalentSpeciesSpec]],
    energy_quadrature_order: int,
) -> CompiledDevice1D:
    """Re-solve contact equilibrium neutrality when outer layers carry bulk defects."""
    left_has = 0 in layer_species or 0 in layer_multivalent
    right_has = last_layer_index in layer_species or last_layer_index in layer_multivalent
    if not left_has and not right_has:
        return component

    parameters = component.parameters
    bands = parameters.band_structure
    temp = float(parameters.temperature_K)
    vt = float(parameters.thermal_voltage_V)

    def solve_endpoint(
        layer_idx: int, node: int, position: float
    ) -> tuple[float, tuple[float, float]]:
        gap = float(bands.physical_band_gap_eV[node])
        chi = float(bands.physical_electron_affinity_eV[node])
        nc = float(parameters.conduction_dos_m3[node])
        nv = float(parameters.valence_dos_m3[node])
        nd = float(parameters.donor_density_m3[node])
        na = float(parameters.acceptor_density_m3[node])
        if layer_idx in layer_species:
            st = solve_monovalent_defect_charge_neutrality(
                temperature_K=temp,
                band_gap_eV=gap,
                effective_conduction_dos_m3=nc,
                effective_valence_dos_m3=nv,
                donor_density_m3=nd,
                acceptor_density_m3=na,
                species=layer_species[layer_idx],
                electron_affinity_eV=chi,
                energy_quadrature_order=energy_quadrature_order,
                normalized_layer_position=position,
            )
            assert st.work_function_eV is not None
            return float(st.work_function_eV), (
                float(st.electron_density_m3),
                float(st.hole_density_m3),
            )
        st_mv = solve_multivalent_defect_charge_neutrality(
            temperature_K=temp,
            band_gap_eV=gap,
            effective_conduction_dos_m3=nc,
            effective_valence_dos_m3=nv,
            donor_density_m3=nd,
            acceptor_density_m3=na,
            species=layer_multivalent[layer_idx],
            electron_affinity_eV=chi,
        )
        assert st_mv.work_function_eV is not None
        return float(st_mv.work_function_eV), (
            float(st_mv.electron_density_m3),
            float(st_mv.hole_density_m3),
        )

    base_work_left = -component.fermi_level_eV - component.phi_left_V
    base_work_right = base_work_left - component.built_in_potential_V
    if left_has:
        work_left, res_left = solve_endpoint(0, 0, 0.0)
    else:
        work_left = base_work_left
        res_left = (
            float(component.reference_electron_density_m3[0]),
            float(component.reference_hole_density_m3[0]),
        )
    if right_has:
        work_right, res_right = solve_endpoint(last_layer_index, -1, 1.0)
    else:
        work_right = base_work_right
        res_right = (
            float(component.reference_electron_density_m3[-1]),
            float(component.reference_hole_density_m3[-1]),
        )

    phi_left = component.phi_left_V
    built_in = work_left - work_right
    fermi = -work_left - phi_left
    nodes = component.nodes_m
    ref_phi = phi_left + built_in * (nodes - nodes[0]) / (nodes[-1] - nodes[0])
    ref_n = parameters.conduction_dos_m3 * np.exp(
        (fermi + ref_phi + bands.physical_electron_affinity_eV) / vt
    )
    ref_p = parameters.valence_dos_m3 * np.exp(
        -(
            fermi
            + ref_phi
            + bands.physical_electron_affinity_eV
            + bands.physical_band_gap_eV
        )
        / vt
    )
    ref_n[0], ref_p[0] = res_left
    ref_n[-1], ref_p[-1] = res_right
    reservoirs = (res_left, res_right)
    rebound_sides = {"left": left_has, "right": right_has}
    new_channels = tuple(
        CarrierExchangeChannel(
            ch.carrier,
            ch.side,
            ch.velocity_m_s,
            reservoirs[0 if ch.side == "left" else 1][0 if ch.carrier == "n" else 1],
            reservoir_source=(
                STATISTICS_CLOSURE if rebound_sides[ch.side] else ch.reservoir_source
            ),
        )
        for ch in component.channels
    )
    rebound_sha = _canonical(
        [
            "defect_contact_neutrality_v1",
            component.identity,
            work_left,
            work_right,
            res_left,
            res_right,
        ]
    )
    return replace(
        component,
        sha256=rebound_sha,
        built_in_potential_V=built_in,
        fermi_level_eV=fermi,
        reference_potential_V=ref_phi,
        reference_electron_density_m3=ref_n,
        reference_hole_density_m3=ref_p,
        channels=new_channels,
    )


def compile_dynamic_defect_device_1d(
    prepared: PreparedDevice,
    *,
    intervals_per_layer: int | Sequence[int] = 24,
    dynamic_bulk_defects: bool = False,
    dynamic_interface_defects: bool = False,
    energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    interface_topology: InterfaceTopologyName | None = None,
    ion_domain_layer: str | None = None,
    ion_domain_endpoints: tuple[str, str] | None = None,
    te_softness: float = 0.02,
    te_regularization_mode: TERegularizationName = "legacy_logistic",
    interface_transmission: float = 1.0,
) -> CompiledDynamicDefectDevice1D:
    """Compile stationary/dynamic bulk defects, dynamic interface traps, and ions."""
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("compile_dynamic_defect_device_1d requires an existing PreparedDevice")
    try:
        quad_order = validate_defect_energy_quadrature_order(energy_quadrature_order)
    except ValueError as error:
        raise CapabilityError(f"unsupported_dynamic_defect_1d: {error}") from error
    if interface_topology is not None and interface_topology not in (
        "deduplicated",
        "two_sided_trace",
    ):
        raise ValueError(
            "interface_topology must be 'deduplicated', 'two_sided_trace', or None"
        )

    data = prepared.to_mapping()
    settings = data["settings"]
    layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    refusals: list[str] = []

    layer_species: dict[int, list[MonovalentSpeciesSpec]] = {}
    layer_multivalent: dict[int, list[MultivalentSpeciesSpec]] = {}
    handled_ids: dict[str, set[str]] = {}
    cleared_legacy_traps: set[str] = set()

    for index, layer in enumerate(layers):
        if layer.metastable_document_json is not None:
            refusals.append(
                f"layers.{layer.id}: metastable inventories use compile_metastable_device_1d"
            )
            continue
        if layer.bulk_trap_distribution_json is not None:
            trap_data = LegacyBulkTrapInput.model_validate(
                json.loads(layer.bulk_trap_distribution_json.decode())
            ).resolved_data(dict(layer.parameters)["Eg"])
            layer_species.setdefault(index, []).append(
                _species_from_legacy_bulk_trap(layer.id, trap_data)
            )
            cleared_legacy_traps.add(layer.id)
        if layer.defects and layer.defect_model != "explicit_quasi_steady":
            refusals.append(
                f"layers.{layer.id}.bulk_defects: an explicit inventory requires explicit_quasi_steady"
            )
            continue
        for item_obj in layer.defects:
            if isinstance(item_obj, PreparedMultivalentDefect):
                if dynamic_bulk_defects:
                    refusals.append(
                        f"layers.{layer.id}.bulk_defects.{item_obj.values['id']}: "
                        "dynamic trap storage DOFs are not bound for multivalent v4 species"
                    )
                    continue
                item = item_obj.to_mapping()
                layer_multivalent.setdefault(index, []).append(
                    _multivalent_from_item(layer.id, item)
                )
                handled_ids.setdefault(layer.id, set()).add(str(item["id"]))
                continue

            item = item_obj.to_mapping()
            neutral_simple = (
                not dynamic_bulk_defects
                and item["charge_transition"] == "neutral"
                and item["distribution"]["kind"] == "single_level"
                and item.get("spatial_profile") is None
            )
            if neutral_simple:
                continue
            layer_species.setdefault(index, []).append(_species_from_item(layer.id, item))
            handled_ids.setdefault(layer.id, set()).add(str(item["id"]))

    if refusals:
        raise CapabilityError("unsupported_dynamic_defect_1d: " + "; ".join(refusals))
    if dynamic_bulk_defects and not layer_species:
        raise CapabilityError(
            "unsupported_dynamic_defect_1d: dynamic_bulk_defects=True requires explicit monovalent/distributed bulk defects"
        )

    closure = str(settings["interface_charge_closure"])
    if closure not in ("off", "equilibrium_referenced"):
        raise CapabilityError(
            f"unsupported_dynamic_defect_1d: unsupported interface_charge_closure={closure!r}"
        )
    if closure == "equilibrium_referenced" and not settings[
        "interface_charge_rebaseline_acknowledged"
    ]:
        raise CapabilityError(
            "unsupported_dynamic_defect_1d: equilibrium_referenced requires interface_charge_rebaseline_acknowledged=True"
        )
    if dynamic_interface_defects and interface_topology == "deduplicated":
        raise CapabilityError(
            "unsupported_dynamic_defect_1d: dynamic_interface_defects requires two_sided_trace topology"
        )
    use_two_sided = bool(
        dynamic_interface_defects
        or interface_topology == "two_sided_trace"
        or closure == "equilibrium_referenced"
    )
    if use_two_sided and (
        settings["interface_shared_occupancy"]
        or settings["interface_plane_projection"]
        or settings["interface_plane_closure"]
        or settings["interface_plane_generation"]
    ):
        raise CapabilityError(
            "unsupported_dynamic_defect_1d: shared-occupancy/plane flags are refused on two_sided_trace"
        )

    ion_active = any(
        float(cast(float, dict(layer.parameters)[name])) != 0.0
        for layer in prepared.layers
        for name in ("P0", "D_ion", "P0_neg", "D_ion_neg")
    )

    # ---- electronic projection ----------------------------------------------------
    projection = prepared.to_input().editing_data()
    changes: list[dict[str, object]] = []
    for layer_doc in projection["layers"]:
        lid = str(layer_doc["id"])
        removed = handled_ids.get(lid)
        if removed:
            kept = [
                entry
                for entry in layer_doc.get("bulk_defects", ())
                if entry["id"] not in removed
            ]
            layer_doc["bulk_defects"] = kept
            if not kept:
                layer_doc["defect_model"] = None
                layer_doc["defect_schema_version"] = None
            for identifier in sorted(removed):
                changes.append(
                    {
                        "path": f"layers.{lid}.bulk_defects.{identifier}",
                        "component_value": "removed",
                    }
                )
        if lid in cleared_legacy_traps:
            layer_doc["bulk_trap_distribution"] = None
            changes.append(
                {
                    "path": f"layers.{lid}.bulk_trap_distribution",
                    "component_value": None,
                }
            )
    if closure != "off":
        projection.setdefault("settings", {})["interface_charge_closure"] = "off"
        projection["settings"]["interface_charge_rebaseline_acknowledged"] = False
        changes.append(
            {"path": "settings.interface_charge_closure", "component_value": "off"}
        )

    ionic: CompiledIonicDevice1D | TwoSidedIonicComponent1D | None = None
    if ion_active and not use_two_sided:
        if ion_domain_layer is not None:
            for layer_doc in projection["layers"]:
                lid_cand = str(layer_doc["id"])
                if lid_cand == ion_domain_layer or layer_doc.get("role") == "substrate":
                    continue
                p_map = layer_doc.setdefault("parameters", {})
                if float(p_map.get("D_ion", 0.0) or 0.0) == 0.0 and float(
                    p_map.get("D_ion_neg", 0.0) or 0.0
                ) == 0.0:
                    for p_key in ("P0", "P0_neg"):
                        if float(p_map.get(p_key, 0.0) or 0.0) != 0.0:
                            p_map[p_key] = 0.0
                            changes.append(
                                {
                                    "path": f"layers.{lid_cand}.{p_key}",
                                    "component_value": 0.0,
                                }
                            )
        component_prepared = resolve_device(
            DeviceInput.model_validate(projection),
            prepared.defaults,
            prepared.resources,
            sources=prepared.sources,
        )
        ionic_dedup = compile_ionic_device_1d(
            component_prepared,
            intervals_per_layer=intervals_per_layer,
            ion_domain_layer=ion_domain_layer,
            ion_domain_endpoints=ion_domain_endpoints,
            te_softness=te_softness,
            te_regularization_mode=te_regularization_mode,
        )
        component: CompiledDevice1D = _rebind_contact_neutrality(
            ionic_dedup.electronic_component,
            last_layer_index=len(layers) - 1,
            layer_species=layer_species,
            layer_multivalent=layer_multivalent,
            energy_quadrature_order=quad_order,
        )
        if component is not ionic_dedup.electronic_component:
            ionic_dedup = replace(ionic_dedup, electronic_component=component)
        ionic = ionic_dedup
    else:
        ion_proj_changes = list(changes)
        if ion_active:
            for layer_doc in projection["layers"]:
                for name in ("P0", "P0_neg", "D_ion", "D_ion_neg"):
                    layer_doc.setdefault("parameters", {})[name] = 0.0
                    ion_proj_changes.append(
                        {
                            "path": f"layers.{layer_doc['id']}.{name}",
                            "component_value": 0.0,
                        }
                    )
            for name in ("ion_steric_diffusion_only", "ion_steric_shared_site"):
                projection.setdefault("settings", {})[name] = False
                ion_proj_changes.append(
                    {"path": f"settings.{name}", "component_value": False}
                )
        component_prepared = resolve_device(
            DeviceInput.model_validate(projection),
            prepared.defaults,
            prepared.resources,
            sources=prepared.sources,
        )
        base_component = compile_device_1d(
            component_prepared,
            intervals_per_layer=intervals_per_layer,
            interface_topology="two_sided_trace" if use_two_sided else interface_topology,
            te_softness=te_softness,
            te_regularization_mode=te_regularization_mode,
            interface_transmission=interface_transmission,
        )
        component = _rebind_contact_neutrality(
            base_component,
            last_layer_index=len(layers) - 1,
            layer_species=layer_species,
            layer_multivalent=layer_multivalent,
            energy_quadrature_order=quad_order,
        )
        if ion_active:
            ionic = _compile_two_sided_ionic_component(
                original_prepared=prepared,
                component_prepared=component_prepared,
                component=component,
                projection_changes=ion_proj_changes,
                layers=layers,
                ion_domain_layer=ion_domain_layer,
                ion_domain_endpoints=ion_domain_endpoints,
            )

    # ---- bulk defect regions ------------------------------------------------------
    node_count = component.node_count
    parameters = component.parameters
    bands = parameters.band_structure
    nodes = component.nodes_m
    thicknesses = [float(layer.thickness) for layer in layers]
    masks = layer_node_masks(nodes, thicknesses, RIGHT_LAYER_WINS)
    edges = np.concatenate(([0.0], np.cumsum(thicknesses)))

    mono_regions: list[MonovalentDefectRegion] = []
    boundary_mono_regions: list[MonovalentDefectRegion] = []
    interior_mask: np.ndarray = np.ones(node_count, dtype=bool)
    interior_mask[0] = False
    interior_mask[-1] = False

    for index, species in sorted(layer_species.items()):
        layer = layers[index]
        mask = np.asarray(masks[index], dtype=bool)
        gap = np.asarray(bands.physical_band_gap_eV, dtype=float)[mask]
        conduction = np.asarray(parameters.conduction_dos_m3, dtype=float)[mask]
        valence = np.asarray(parameters.valence_dos_m3, dtype=float)[mask]
        position = np.clip(
            (nodes[mask] - edges[index]) / (edges[index + 1] - edges[index]), 0.0, 1.0
        )
        document_digest = _canonical([layer.id, quad_order, [s.identifier for s in species]])
        try:
            region = MonovalentDefectRegion(
                identifier=f"layer[{index}]/{layer.id}",
                document_sha256=document_digest,
                active_nodes=mask,
                band_gap_eV=gap,
                effective_conduction_dos_m3=conduction,
                effective_valence_dos_m3=valence,
                temperature_K=float(parameters.temperature_K),
                species=tuple(species),
                energy_quadrature_order=quad_order,
                normalized_layer_positions=position,
            )
        except ValueError as error:
            raise CapabilityError(f"layers.{layer.id}: {error}") from error
        mono_regions.append(region)

        if dynamic_bulk_defects:
            bmask = mask & ~interior_mask
            if np.any(bmask):
                bpos = np.clip(
                    (nodes[bmask] - edges[index]) / (edges[index + 1] - edges[index]),
                    0.0,
                    1.0,
                )
                boundary_mono_regions.append(
                    MonovalentDefectRegion(
                        identifier=f"layer[{index}]/{layer.id}:contact_boundary",
                        document_sha256=document_digest,
                        active_nodes=bmask,
                        band_gap_eV=np.asarray(bands.physical_band_gap_eV, dtype=float)[
                            bmask
                        ],
                        effective_conduction_dos_m3=np.asarray(
                            parameters.conduction_dos_m3, dtype=float
                        )[bmask],
                        effective_valence_dos_m3=np.asarray(
                            parameters.valence_dos_m3, dtype=float
                        )[bmask],
                        temperature_K=float(parameters.temperature_K),
                        species=tuple(species),
                        energy_quadrature_order=quad_order,
                        normalized_layer_positions=bpos,
                    )
                )

    mv_regions: list[MultivalentDefectRegion] = []
    for index, mv_species in sorted(layer_multivalent.items()):
        layer = layers[index]
        mask = np.asarray(masks[index], dtype=bool)
        gap = np.asarray(bands.physical_band_gap_eV, dtype=float)[mask]
        conduction = np.asarray(parameters.conduction_dos_m3, dtype=float)[mask]
        valence = np.asarray(parameters.valence_dos_m3, dtype=float)[mask]
        doc_digest = _canonical([layer.id, [s.identifier for s in mv_species]])
        try:
            mv_regions.append(
                MultivalentDefectRegion(
                    identifier=f"layer[{index}]/{layer.id}:multivalent",
                    document_sha256=doc_digest,
                    active_nodes=mask,
                    band_gap_eV=gap,
                    effective_conduction_dos_m3=conduction,
                    effective_valence_dos_m3=valence,
                    temperature_K=float(parameters.temperature_K),
                    species=tuple(mv_species),
                )
            )
        except ValueError as error:
            raise CapabilityError(f"layers.{layer.id}: {error}") from error

    explicit_srh_mask: NDArray[np.bool_] = np.zeros(node_count, dtype=bool)
    for r_mono in mono_regions:
        explicit_srh_mask |= r_mono.active_nodes
    for r_mv in mv_regions:
        explicit_srh_mask |= r_mv.active_nodes
    if np.any(explicit_srh_mask):
        component = _mask_lifetime_srh(component, explicit_srh_mask)
        if ionic is not None:
            ionic = replace(ionic, electronic_component=component)

    # ---- validate interface defects before dark equilibrium solve -----------------
    elec_interfaces = [
        iface.to_mapping() for iface in prepared.interfaces if iface.electrical
    ]
    by_pair = {(str(e["left"]), str(e["right"])): e for e in elec_interfaces}
    ordered_iface_entries: list[dict[str, Any]] = []
    for k in range(len(layers) - 1):
        pair = (layers[k].id, layers[k + 1].id)
        if pair not in by_pair:
            raise CapabilityError(
                f"unsupported_dynamic_defect_1d: missing electrical interface {pair[0]}->{pair[1]}"
            )
        ordered_iface_entries.append(by_pair[pair])

    if dynamic_interface_defects or closure == "equilibrium_referenced":
        model_check = component.interface_model
        if (
            model_check is None
            or model_check.topology != "two_sided_trace"
            or not model_check.two_sided
        ):
            raise CapabilityError(
                "unsupported_dynamic_defect_1d: two-sided interfaces are required"
            )
        has_defect_iface = False
        for tspec in model_check.two_sided:
            entry = ordered_iface_entries[tspec.interface_index]
            defect = entry.get("defect")
            if defect is None:
                continue
            has_defect_iface = True
            cal = float(defect["calibration_factor"])
            state_cal = float(defect["iface_state_calibration_factor"])
            if (dynamic_interface_defects or closure == "equilibrium_referenced") and (
                cal != 1.0 or state_cal != 1.0
            ):
                raise CapabilityError(
                    f"interfaces.{entry['id']}: empirical calibration factors are refused for dynamic/charged interface closure"
                )
            raw_vn = entry.get("v_n")
            raw_vp = entry.get("v_p")
            micro_pair = tuple(
                float(v) for v in entry["microscopic_capture_velocities_m_s"]
            )
            if (
                (dynamic_interface_defects or closure == "equilibrium_referenced")
                and raw_vn is not None
                and raw_vp is not None
                and not (
                    math.isclose(float(raw_vn), micro_pair[0], rel_tol=1.0e-12, abs_tol=0.0)
                    and math.isclose(
                        float(raw_vp), micro_pair[1], rel_tol=1.0e-12, abs_tol=0.0
                    )
                )
            ):
                raise CapabilityError(
                    f"interfaces.{entry['id']}: compatibility SRV {(raw_vn, raw_vp)!r} disagrees with microscopic capture {micro_pair!r}"
                )
        if not has_defect_iface:
            raise CapabilityError(
                "unsupported_dynamic_defect_1d: at least one microscopic interface defect is required"
            )

    def build_device(
        comp_arg: CompiledDevice1D,
        ionic_arg: CompiledIonicDevice1D | TwoSidedIonicComponent1D | None,
        stationary_model: StationaryDefectModel1D | None,
        layout: Any,
        ref_logit: np.ndarray,
        traps: Sequence[DynamicInterfaceTrapBinding1D],
        caps: Sequence[str],
        dark_seed: np.ndarray | None = None,
    ) -> CompiledDynamicDefectDevice1D:
        manifest: dict[str, Any] = {
            "binding": BINDING_SCHEMA,
            "original_prepared_sha256": prepared.content_sha256,
            "electronic_component": {
                "prepared_sha256": component_prepared.content_sha256,
                "compiled_sha256": comp_arg.identity,
                "projection": changes,
            },
            "dynamic_bulk_defects": bool(dynamic_bulk_defects),
            "dynamic_interface_defects": bool(dynamic_interface_defects),
            "energy_quadrature_order": quad_order,
            "interface_charge_closure": closure,
            "ions": None if ionic_arg is None else ionic_arg.identity,
            "monovalent_regions": [r.identifier for r in mono_regions],
            "multivalent_regions": [r.identifier for r in mv_regions],
            "scope": "assembled residual, derivatives, steady, AC and BDF; scientific qualification remains open",
        }
        text = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
        )
        identity = _canonical(
            [
                text,
                None if stationary_model is None else stationary_model.identity_sha256,
                None if layout is None else layout.identity_sha256,
                [(t.interface_id, t.reference_logit) for t in traps],
            ]
        )
        return CompiledDynamicDefectDevice1D(
            device_id=str(data["id"]),
            sha256=identity,
            original_prepared_sha256=prepared.content_sha256,
            electronic_component=comp_arg,
            ionic_component=ionic_arg,
            stationary_defects=stationary_model,
            dynamic_bulk_traps=layout,
            bulk_trap_reference_logit=ref_logit,
            dynamic_interface_traps=tuple(traps),
            binding_manifest_json=text,
            capabilities=tuple(caps),
            dark_equilibrium_seed=(
                np.empty(0, dtype=np.float64) if dark_seed is None else dark_seed
            ),
        )

    # ---- unified dark V=0, G=0 reference solve (if needed) ------------------------
    need_dark_reference = bool(
        dynamic_bulk_defects
        or dynamic_interface_defects
        or closure == "equilibrium_referenced"
    )
    dynamic_layout = (
        compile_dynamic_bulk_trap_layout(mono_regions, node_count=node_count)
        if dynamic_bulk_defects
        else None
    )
    reference_logit: NDArray[np.float64] = np.empty(0, dtype=np.float64)
    eq_occ_by_iface: dict[int, float] = {}
    eq_logit_by_iface: dict[int, float] = {}
    probe_eq_u: NDArray[np.float64] | None = None

    if need_dark_reference:
        from solarlab.equations.assembly.dynamic_defects_1d import (
            solve_dynamic_defect_steady_1d,
        )

        probe_stationary: StationaryDefectModel1D | None = None
        if mono_regions or mv_regions:
            probe_stationary = StationaryDefectModel1D(
                node_count=node_count,
                identity_sha256=_canonical(
                    [
                        "dark_probe",
                        *(r.document_sha256 for r in mono_regions),
                        *(r.document_sha256 for r in mv_regions),
                    ]
                ),
                monovalent_regions=tuple(mono_regions),
                multivalent_regions=tuple(mv_regions),
            )
        probe = build_device(
            component,
            ionic,
            probe_stationary,
            None,
            np.empty(0, dtype=np.float64),
            (),
            ("unified_assembly", "dynamic_defect_1d_probe"),
        )
        equilibrium = solve_dynamic_defect_steady_1d(
            probe, voltage_V=0.0, generation_scale=0.0
        )
        if not equilibrium.converged:
            raise CapabilityError(
                "dynamic defect dark equilibrium reference failed to converge"
            )
        probe_eq_u = np.array(equilibrium.state.u, dtype=np.float64, copy=True)
        elec_eq = equilibrium.state.electronic_state
        n_eq = np.asarray(elec_eq.electron_density_m3, dtype=np.float64)
        p_eq = np.asarray(elec_eq.hole_density_m3, dtype=np.float64)
        phi_eq = np.asarray(elec_eq.potential_V, dtype=np.float64)
        zn_eq = np.asarray(elec_eq.u[node_count : 2 * node_count], dtype=np.float64)
        zp_eq = np.asarray(elec_eq.u[2 * node_count : 3 * node_count], dtype=np.float64)

        if dynamic_layout is not None:
            reference_logit = quasi_steady_bulk_trap_logit(n_eq, p_eq, dynamic_layout)

        if (
            component.interface_model is not None
            and component.interface_model.topology == "two_sided_trace"
        ):
            for tspec in component.interface_model.two_sided:
                entry = ordered_iface_entries[tspec.interface_index]
                if entry.get("defect") is None:
                    continue
                l_node = tspec.left_node
                r_node = tspec.right_node
                dark_qss = solve_two_sided_interface_qss(
                    tspec.trace_electrostatics,
                    tspec.physics,
                    TwoSidedBulkState(
                        phi_left_V=float(phi_eq[l_node]),
                        phi_right_V=float(phi_eq[r_node]),
                        n_left_m3=float(n_eq[l_node]),
                        p_left_m3=float(p_eq[l_node]),
                        n_right_m3=float(n_eq[r_node]),
                        p_right_m3=float(p_eq[r_node]),
                        electron_electrochemical_step=float(
                            zn_eq[r_node] - zn_eq[l_node]
                        ),
                        hole_electrochemical_step=float(
                            -(zp_eq[r_node] - zp_eq[l_node])
                        ),
                    ),
                )
                if dark_qss.occupancy is None:
                    raise CapabilityError(
                        f"interfaces.{entry['id']}: microscopic interface defect produced zero dark QSS occupancy"
                    )
                eq_occ_by_iface[tspec.interface_index] = float(dark_qss.occupancy)
                n_tl, p_tl, n_tr, p_tr = (float(v) for v in dark_qss.state_m3)
                vn_i = float(tspec.physics.surface_recombination_velocity_n_m_s)
                vp_i = float(tspec.physics.surface_recombination_velocity_p_m_s)
                filled_eq = vn_i * (n_tl + n_tr) + vp_i * (
                    float(tspec.physics.p1_left_m3) + float(tspec.physics.p1_right_m3)
                )
                empty_eq = vn_i * (
                    float(tspec.physics.n1_left_m3) + float(tspec.physics.n1_right_m3)
                ) + vp_i * (p_tl + p_tr)
                eq_logit_by_iface[tspec.interface_index] = (
                    math.log(filled_eq) - math.log(empty_eq)
                    if filled_eq > 0.0 and empty_eq > 0.0
                    else _logit(float(dark_qss.occupancy))
                )

    # ---- finalize stationary bulk model, interface bindings, and capabilities -----
    capabilities = ["unified_assembly", "dynamic_defect_1d"]
    stationary: StationaryDefectModel1D | None = None
    if dynamic_bulk_defects:
        capabilities.append("dynamic_bulk_traps")
        if boundary_mono_regions or mv_regions:
            stationary = StationaryDefectModel1D(
                node_count=node_count,
                identity_sha256=_canonical(
                    [
                        *(r.document_sha256 for r in boundary_mono_regions),
                        *(r.document_sha256 for r in mv_regions),
                    ]
                ),
                monovalent_regions=tuple(boundary_mono_regions),
                multivalent_regions=tuple(mv_regions),
            )
    elif mono_regions or mv_regions:
        stationary = StationaryDefectModel1D(
            node_count=node_count,
            identity_sha256=_canonical(
                [
                    *(r.document_sha256 for r in mono_regions),
                    *(r.document_sha256 for r in mv_regions),
                ]
            ),
            monovalent_regions=tuple(mono_regions),
            multivalent_regions=tuple(mv_regions),
        )
        if mono_regions:
            capabilities.append("stationary_charged_defects")
        if mv_regions:
            capabilities.append("stationary_multivalent_defects")

    interface_traps: list[DynamicInterfaceTrapBinding1D] = []
    if dynamic_interface_defects:
        assert component.interface_model is not None
        for tspec in component.interface_model.two_sided:
            entry = ordered_iface_entries[tspec.interface_index]
            defect = entry.get("defect")
            if defect is None:
                continue
            f_eq = eq_occ_by_iface[tspec.interface_index]
            trap_nt = float(defect["total_density_m2"])
            wl = wr = 0.0
            if closure == "equilibrium_referenced":
                wl, wr = sheet_charge_row_weights(
                    tspec.trace_electrostatics.eps_r_left,
                    tspec.trace_electrostatics.left_distance_m,
                    tspec.trace_electrostatics.eps_r_right,
                    tspec.trace_electrostatics.right_distance_m,
                )
            trap_spec = DynamicInterfaceTrapSpec(
                interface_id=str(entry["id"]),
                left_node=tspec.left_node,
                right_node=tspec.right_node,
                charge_type="acceptor",
                sheet_density_m2=trap_nt,
                v_n_m_s=float(tspec.physics.surface_recombination_velocity_n_m_s),
                v_p_m_s=float(tspec.physics.surface_recombination_velocity_p_m_s),
                n1_left_m3=float(tspec.physics.n1_left_m3),
                p1_left_m3=float(tspec.physics.p1_left_m3),
                n1_right_m3=float(tspec.physics.n1_right_m3),
                p1_right_m3=float(tspec.physics.p1_right_m3),
                equilibrium_occupancy=f_eq,
                equilibrium_referenced_charge=(closure == "equilibrium_referenced"),
            )
            interface_traps.append(
                DynamicInterfaceTrapBinding1D(
                    interface_index=tspec.interface_index,
                    interface_id=str(entry["id"]),
                    face=tspec.face,
                    left_node=tspec.left_node,
                    right_node=tspec.right_node,
                    trace_electrostatics=tspec.trace_electrostatics,
                    physics=tspec.physics,
                    trap_spec=trap_spec,
                    reference_logit=eq_logit_by_iface[tspec.interface_index],
                    weight_left=float(wl),
                    weight_right=float(wr),
                )
            )
        capabilities.append("dynamic_interface_traps")
    elif closure == "equilibrium_referenced":
        assert component.interface_model is not None
        updated_specs: list[TwoSidedInterfaceBindingSpec] = []
        for tspec in component.interface_model.two_sided:
            entry = ordered_iface_entries[tspec.interface_index]
            defect = entry.get("defect")
            if defect is None:
                updated_specs.append(tspec)
                continue
            f_eq = eq_occ_by_iface[tspec.interface_index]
            trap_nt = float(defect["total_density_m2"])
            wl, wr = sheet_charge_row_weights(
                tspec.trace_electrostatics.eps_r_left,
                tspec.trace_electrostatics.left_distance_m,
                tspec.trace_electrostatics.eps_r_right,
                tspec.trace_electrostatics.right_distance_m,
            )
            updated_specs.append(
                replace(
                    tspec,
                    charge=EquilibriumReferencedSheetCharge(
                        equilibrium_occupancy=f_eq,
                        trap_density_m2=trap_nt,
                    ),
                    weight_left=float(wl),
                    weight_right=float(wr),
                )
            )
        charged_model = CompiledInterfaceModel1D(
            topology="two_sided_trace",
            identity_sha256=_canonical(
                [
                    "charged_two_sided_v1",
                    component.interface_model.identity_sha256,
                    sorted(eq_occ_by_iface.items()),
                ]
            ),
            two_sided=tuple(updated_specs),
        )
        component = replace(
            component,
            sha256=_canonical([component.identity, charged_model.identity_sha256]),
            interface_model=charged_model,
        )
        if ionic is not None:
            ionic = replace(ionic, electronic_component=component)
        capabilities.append("equilibrium_referenced_interface_charge")

    if ionic is not None:
        capabilities.append("mobile_ions")

    dark_seed: NDArray[np.float64] | None = None
    if probe_eq_u is not None:
        extra_dofs = (0 if dynamic_layout is None else dynamic_layout.size) + 2 * len(
            interface_traps
        )
        dark_seed = np.r_[probe_eq_u, np.zeros(extra_dofs, dtype=np.float64)]

    return build_device(
        component,
        ionic,
        stationary,
        dynamic_layout,
        reference_logit,
        interface_traps,
        capabilities,
        dark_seed,
    )


def compile_metastable_device_1d(
    prepared: PreparedDevice,
    *,
    intervals_per_layer: int | Sequence[int] = 24,
    te_softness: float = 0.02,
    te_regularization_mode: TERegularizationName = "legacy_logistic",
    interface_transmission: float = 1.0,
) -> CompiledMetastableDevice1D:
    """Compile a single-layer metastable defect declaration into a freeze-ready model."""
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("compile_metastable_device_1d requires an existing PreparedDevice")
    layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    meta_layers = [
        (idx, layer)
        for idx, layer in enumerate(layers)
        if layer.metastable_document_json is not None
    ]
    if len(meta_layers) != 1:
        raise CapabilityError(
            "unsupported_metastable_1d: exactly one electrical layer must declare metastable_document"
        )
    for layer in layers:
        if layer.defects or layer.bulk_trap_distribution_json is not None:
            raise CapabilityError(
                f"layers.{layer.id}: metastable compilation refuses additional explicit bulk defect inventories"
            )
        params = dict(layer.parameters)
        if any(params[name] != 0.0 for name in ("P0", "D_ion", "P0_neg", "D_ion_neg")):
            raise CapabilityError(
                f"layers.{layer.id}: coupled mobile ions and metastable configurations are not bound"
            )

    meta_idx, meta_layer = meta_layers[0]
    eg_nominal = float(cast(float, dict(meta_layer.parameters)["Eg"]))
    doc_raw = json.loads(cast(bytes, meta_layer.metastable_document_json).decode())
    doc_resolved = MetastableDocumentInput.model_validate(doc_raw).resolved_data(
        eg_nominal
    )
    specs = tuple(
        _metastable_from_item(meta_layer.id, item)
        for item in doc_resolved["metastable_defects"]
    )
    preparation: MetastablePreparationInput | None = None
    if meta_layer.metastable_preparation_json is not None:
        prep_raw = json.loads(meta_layer.metastable_preparation_json.decode())
        preparation = MetastablePreparationInput.model_validate(prep_raw)

    intervals_stored: int | tuple[int, ...] = (
        intervals_per_layer
        if isinstance(intervals_per_layer, int)
        else tuple(int(c) for c in intervals_per_layer)
    )

    def _compile_electronic_at_temperature(temp_override: float | None) -> CompiledDevice1D:
        proj = prepared.to_input().editing_data()
        for ldoc in proj["layers"]:
            if ldoc["id"] == meta_layer.id:
                ldoc.pop("metastable_document", None)
                ldoc.pop("metastable_preparation", None)
                ldoc["defect_model"] = None
                ldoc["defect_schema_version"] = None
        if temp_override is not None:
            proj.setdefault("settings", {})["T"] = float(temp_override)
        proj_prepared = resolve_device(
            DeviceInput.model_validate(proj),
            prepared.defaults,
            prepared.resources,
            sources=prepared.sources,
        )
        return compile_device_1d(
            proj_prepared,
            intervals_per_layer=intervals_stored,
            te_softness=te_softness,
            te_regularization_mode=te_regularization_mode,
            interface_transmission=interface_transmission,
        )

    base_comp = _compile_electronic_at_temperature(None)
    nodes = base_comp.nodes_m
    thicknesses = [float(layer.thickness) for layer in layers]
    edges = np.concatenate(([0.0], np.cumsum(thicknesses)))
    left_edge, right_edge = float(edges[meta_idx]), float(edges[meta_idx + 1])
    mask = (nodes >= left_edge) & (nodes <= right_edge)
    # Never own outer contact nodes 0 / N-1 (matches metastable_preparation.py).
    mask[[0, -1]] = False
    if not np.any(mask):
        raise CapabilityError(
            f"layers.{meta_layer.id}: metastable layer has no interior grid nodes"
        )
    base_comp = _mask_lifetime_srh(base_comp, mask)

    def _extract_region_arrays(
        comp_obj: CompiledDevice1D,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        b = comp_obj.parameters.band_structure
        gap = np.asarray(b.physical_band_gap_eV, dtype=np.float64)[mask]
        nc = np.asarray(comp_obj.parameters.conduction_dos_m3, dtype=np.float64)[mask]
        nv = np.asarray(comp_obj.parameters.valence_dos_m3, dtype=np.float64)[mask]
        for label, vals in (("band gap", gap), ("conduction DOS", nc), ("valence DOS", nv)):
            if (
                vals.size == 0
                or not np.all(np.isfinite(vals))
                or np.any(vals <= 0.0)
                or not np.all(vals == vals[0])
            ):
                raise CapabilityError(
                    f"layers.{meta_layer.id}: metastable closure requires uniform positive {label}"
                )
        for sp in specs:
            try:
                sp.validate_band_gap(float(gap[0]))
            except ValueError as err:
                raise CapabilityError(f"layers.{meta_layer.id}: {err}") from err
        return gap, nc, nv, float(comp_obj.parameters.temperature_K)

    base_gap, base_nc, base_nv, base_temp = _extract_region_arrays(base_comp)
    active_count = int(np.count_nonzero(mask))
    data = prepared.to_mapping()
    doc_sha = _canonical([meta_layer.id, doc_resolved])

    def _freeze_builder(
        donor_fractions: Mapping[str, ArrayLike] | Sequence[ArrayLike],
        temperature_K: float | None = None,
    ) -> CompiledDynamicDefectDevice1D:
        comp_f = (
            base_comp
            if temperature_K is None
            or math.isclose(float(temperature_K), base_temp, rel_tol=0.0, abs_tol=1e-15)
            else _mask_lifetime_srh(
                _compile_electronic_at_temperature(float(temperature_K)), mask
            )
        )
        gap_f, nc_f, nv_f, temp_f = (
            (base_gap, base_nc, base_nv, base_temp)
            if comp_f is base_comp
            else _extract_region_arrays(comp_f)
        )
        y_mat: NDArray[np.float64] = np.empty((len(specs), active_count), dtype=np.float64)
        if isinstance(donor_fractions, Mapping):
            for s_idx, sp in enumerate(specs):
                short_id = sp.identifier.split("/", 1)[-1]
                if sp.identifier in donor_fractions:
                    raw_y = donor_fractions[sp.identifier]
                elif short_id in donor_fractions:
                    raw_y = donor_fractions[short_id]
                else:
                    raise ValueError(
                        f"missing donor_fraction entry for metastable species {sp.identifier!r}"
                    )
                y_mat[s_idx] = np.broadcast_to(
                    np.asarray(raw_y, dtype=np.float64), (active_count,)
                )
        else:
            if len(donor_fractions) != len(specs):
                raise ValueError(
                    "donor_fractions sequence length must match metastable_specs"
                )
            for s_idx, raw_y in enumerate(donor_fractions):
                y_mat[s_idx] = np.broadcast_to(
                    np.asarray(raw_y, dtype=np.float64), (active_count,)
                )

        f_region = FrozenMetastableRegion(
            identifier=f"layer[{meta_idx}]/{meta_layer.id}",
            document_sha256=doc_sha,
            active_nodes=mask,
            band_gap_eV=gap_f,
            effective_conduction_dos_m3=nc_f,
            effective_valence_dos_m3=nv_f,
            temperature_K=temp_f,
            species=specs,
            donor_fractions=y_mat,
        )
        y_digest = hashlib.sha256(y_mat.tobytes()).hexdigest()
        stat_model = StationaryDefectModel1D(
            node_count=comp_f.node_count,
            identity_sha256=_canonical(["frozen_metastable", doc_sha, temp_f, y_digest]),
            metastable_regions=(f_region,),
        )
        manifest_obj = {
            "binding": METASTABLE_BINDING_SCHEMA,
            "original_prepared_sha256": prepared.content_sha256,
            "electronic_compiled_sha256": comp_f.identity,
            "layer_id": meta_layer.id,
            "temperature_K": temp_f,
            "donor_fractions_sha256": y_digest,
        }
        manifest_txt = json.dumps(
            manifest_obj, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        dev_sha = _canonical([manifest_txt, stat_model.identity_sha256])
        return CompiledDynamicDefectDevice1D(
            device_id=str(data["id"]),
            sha256=dev_sha,
            original_prepared_sha256=prepared.content_sha256,
            electronic_component=comp_f,
            ionic_component=None,
            stationary_defects=stat_model,
            dynamic_bulk_traps=None,
            bulk_trap_reference_logit=np.empty(0, dtype=np.float64),
            dynamic_interface_traps=(),
            binding_manifest_json=manifest_txt,
            capabilities=(
                "unified_assembly",
                "dynamic_defect_1d",
                "frozen_metastable_defects",
            ),
        )

    meta_sha = _canonical(
        [
            METASTABLE_BINDING_SCHEMA,
            prepared.content_sha256,
            base_comp.identity,
            doc_sha,
            None if preparation is None else preparation.normalized_data(),
        ]
    )
    return CompiledMetastableDevice1D(
        device_id=str(data["id"]),
        sha256=meta_sha,
        prepared=prepared,
        metastable_layer_id=meta_layer.id,
        metastable_layer_index=meta_idx,
        metastable_specs=specs,
        preparation=preparation,
        intervals_per_layer=intervals_stored,
        base_electronic_device=base_comp,
        active_nodes=mask,
        band_gap_eV=base_gap,
        conduction_dos_m3=base_nc,
        valence_dos_m3=base_nv,
        temperature_K=base_temp,
        freeze_builder=_freeze_builder,
    )
