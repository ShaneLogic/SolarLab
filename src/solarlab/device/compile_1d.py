"""Bind an existing PreparedDevice to the first production new-core 1D DC slice.

Supported physics is explicit: Maxwell-Boltzmann carriers, fully ionized
dopants, static temperature/doping/band grading, DOS-aware SG transport,
semiconductor work-function contacts, local lifetime or explicit neutral
single-level SRH with literal above-valence-band energies and physical DOS
with radiative/Auger,
smooth Caughey-Thomas face transport without Poole-Frenkel enhancement,
and monochromatic Beer-Lambert, supplied coherent spectral TMM optics,
or a node-aligned fixed generation resource.
Historical mode labels do not select a legacy engine or enable its tier ceiling.
Unsupported active mechanisms fail before any nonlinear solve.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import hashlib
import json
import math
from typing import Any, Literal, TypedDict, cast

import numpy as np

from solarlab.config.identity import BuildSource, IdentityInputs, ModelInputs
from solarlab.config.model_parameters import device_topology, layer_parameter_source
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.device.resolved import PreparedDefect, PreparedDevice, PreparedLayer
from solarlab.device.neutral_defects import LITERAL_ABOVE_VB_DOS_REFERENCE_V1, bind_neutral_defects
from solarlab.device.optical_generation import bind_optical_generation_1d
from solarlab.discretization.face_average import DIELECTRIC_RULES
from solarlab.equations.dc_1d import (
    CapabilityError,
    CompiledDevice1D,
    CompiledInterfaceModel1D,
    DeduplicatedBareSRHSpec,
    DeduplicatedDefectSRHSpec,
    DeduplicatedThermionicFaceSpec,
    LocalRecombinationModel,
    TERegularizationName,
    TwoSidedInterfaceBindingSpec,
)
from solarlab.solvers.nonlinear.steady_1d import (
    POISSON_CONTACT_BALANCE_V1,
    SteadyOptions,
    SteadyStateError,
    solve_equilibrium,
)
from solarlab.equations.field_transport_1d import FieldTransport1D, bind_field_transport_1d
from solarlab.physics.generation.profile import GenerationProfile
from solarlab.mesh import (
    GEOMETRY_NAMES,
    HALF_ENDPOINT_V1,
    INTEGER_ENDPOINT_V1,
    PHYSICAL_BOUNDARY_TWO_SIDED_V1,
    PHYSICAL_HALF_ENDPOINT_V2,
    RIGHT_LAYER_WINS,
    TWO_SIDED_TRACE_V1,
    build_geometry,
    multilayer_grid,
)
from solarlab.mesh.geometry import ControlVolumeGeometry1D
from solarlab.mesh.grid import layer_node_masks
from solarlab.physics.band_structure.binding import (
    LayerParameterInput,
    ParameterBinding,
    TemperatureLawSpec,
    compile_layer_parameters,
)
from solarlab.physics.band_structure.doping_profile import DopingProfileSpec
from solarlab.physics.band_structure.grading import GradingSpec
from solarlab.physics.contacts import (
    CarrierExchangeChannel,
    ContactMaterial,
    doping_equilibrium_densities,
    semiconductor_work_function,
)
from solarlab.physics.contacts.exchange import resolve_exchange_velocities
from solarlab.physics.contacts.potential import (
    BUILT_IN_POTENTIAL_MODES,
    BuiltInPotentialSources,
    LegacyBandContact,
)
from solarlab.physics.contacts.reservoirs import metal_reservoir_floor
from solarlab.physics.electrostatics import PoissonOperator1D
from solarlab.physics.electrostatics.interface_traces import (
    TraceElectrostatics,
    sheet_charge_row_weights,
)
from solarlab.physics.generation.beer_lambert import BeerLambertBinding
from solarlab.physics.generation.beer_lambert_spectral import (
    SpectralBeerLambertBinding,
)
from solarlab.physics.generation.fixed import FixedGenerationProfile
from solarlab.physics.generation.optical_binding import (
    OpticalLayerSpec,
    OpticalStackBinding,
    constant_index_optical_constants,
)
from solarlab.physics.generation.profile import ControlVolumes
from solarlab.physics.generation.spectrum import SpectralPhotonFlux, sample_nk_table
from solarlab.physics.interface_transport import (
    EquilibriumReferencedSheetCharge,
    LEGACY_UNNORMALIZED_THERMIONIC_V1,
    TE_BARRIER_THRESHOLD_EV,
    THERMIONIC_MODEL_VERSIONS,
    ThermionicCombinationMode,
    ThermionicModelVersion,
    TwoSidedBulkState,
    TwoSidedInterfacePhysics,
    remove_shared_interface_nodes,
    resolve_layer_richardson,
    solve_two_sided_interface_qss,
)
from solarlab.physics.presets import (
    P08_MODEL_VERSION_MAPPINGS,
    P08_SOURCE_PATHS,
    P08CompilerOptions,
    build_p08_model_registry,
    build_p08_preset,
    select_preset,
)
from solarlab.physics.recombination.explicit_neutral import (
    NeutralDefectSpecies,
    NeutralExplicitDefectModel,
    compile_neutral_defect_species,
)
from solarlab.physics.recombination.interface import (
    INTERFACE_BAND_REFERENCE_MODES,
    INTERFACE_DEFECT_DISTRIBUTION_MODES,
    InterfaceBandReferenceMode,
    InterfaceDefectDistributionMode,
    InterfaceDefectSpec,
    InterfaceTrapQuadrature,
    REFERENCE_LAYER_BAND_ANCHOR,
    SCAPS_GAUSSIAN_INTERFACE_QUADRATURE,
    SINGLE_LEVEL_INTERFACE_DEFECT,
    build_plane_closure_params,
    build_scaps_gaussian_interface_quadrature,
    resolve_interface_defect_references,
)
from solarlab.physics.recombination.reaction import (
    BulkRecombinationCoefficients,
    BulkRecombinationReaction,
)
from solarlab.physics.registry import CapabilityContext, Scope
from solarlab.physics.semantic_projection import SemanticProjection
from solarlab.physics.transport import legacy_face_diffusivity
from solarlab.physics.transport.electrochemical_flux import (
    BAND_STEP_MODES,
    LEGACY_SUMMED_POTENTIAL_V1,
)
from solarlab.physics.transport.field_mobility_tangents import FieldMobilityCapabilityError

__all__ = [
    "CapabilityError",
    "CompiledDevice1D",
    "bind_spectral_beer_lambert_1d",
    "compile_device_1d",
    "p08_identity_inputs",
    "unsupported",
]

Q = ELEMENTARY_CHARGE_C
InterfaceTopologyName = Literal["deduplicated", "two_sided_trace"]
OpticalModelName = Literal["beer_lambert", "beer_lambert_spectral", "tmm"]
_SPECTRAL_BL_BINDING_1D_V1 = "prepared_spectral_beer_lambert_1d_v1"
_SPECTRAL_BL_BUDGET_RTOL = 1e-12
_LEGACY_BENCHMARK_DECK_OPT_IN: ContextVar[bool] = ContextVar(
    "_LEGACY_BENCHMARK_DECK_OPT_IN", default=False
)


@contextmanager
def legacy_benchmark_deck_scope() -> Iterator[None]:
    """Explicit opt-in scope for legacy benchmark/qualification decks (P09-01)."""
    token = _LEGACY_BENCHMARK_DECK_OPT_IN.set(True)
    try:
        yield
    finally:
        _LEGACY_BENCHMARK_DECK_OPT_IN.reset(token)


class _LayerScalars(TypedDict):
    """Resolved scalar fields consumed by this supported DC binding."""

    chi: float
    Eg: float
    Nc300: float | None
    Nv300: float | None
    ni: float
    n1: float
    p1: float
    mu_n: float
    mu_p: float
    B_rad: float
    v_th: float
    N_A: float
    N_D: float
    N_A_bulk: float | None
    N_D_bulk: float | None
    doping_profile_shape: str | None
    doping_decay_length: float | None
    doping_edge: str
    mu_T_gamma: float
    B_rad_T_gamma: float
    varshni_alpha: float
    varshni_beta: float
    Eg_back: float | None
    chi_back: float | None
    grading_profile: Literal["linear", "parabolic", "exponential"]
    grading_direction: Literal["front_to_back", "back_to_front"]
    grading_bowing: float
    grading_char_length: float | None


def unsupported(
    prepared: PreparedDevice,
    data: dict[str, Any],
    layers: tuple[PreparedLayer, ...],
    *,
    optical_model: OpticalModelName | None = None,
    allow_legacy_benchmark_deck: bool = False,
) -> None:
    if not isinstance(allow_legacy_benchmark_deck, bool):
        raise TypeError("allow_legacy_benchmark_deck must be a bool")
    settings = data["settings"]
    missing = []
    if not allow_legacy_benchmark_deck:
        if settings["built_in_potential_mode"] != "semiconductor_work_function":
            missing.append("settings.built_in_potential_mode: only explicit semiconductor_work_function contacts are bound")
        if not settings["dos_band_potentials"]:
            missing.append("settings.dos_band_potentials: physical DOS transport requires this enabled")
        for flag in (
            "te_physical_norm", "flat_band_metal_contacts",
            "interface_two_sided", "interface_shared_occupancy",
            "graded_optics", "interface_tunneling",
            "ion_steric_diffusion_only", "ion_steric_shared_site",
        ):
            if settings[flag]:
                missing.append("settings." + flag)
        if settings["het_recomb_despike"] != 0.0:
            missing.append("settings.het_recomb_despike")
    else:
        bip_mode = settings["built_in_potential_mode"]
        if bip_mode is not None and bip_mode not in BUILT_IN_POTENTIAL_MODES:
            missing.append("settings.built_in_potential_mode")
        if bip_mode == "metal_work_function" and (
            settings["work_function_left_eV"] is None or settings["work_function_right_eV"] is None
        ):
            missing.append("settings.built_in_potential_mode: metal_work_function requires both contact work functions")
        has_any_ions = any(
            dict(layer.parameters)[name] != 0.0
            for layer in layers
            for name in ("P0", "P0_neg", "D_ion", "D_ion_neg")
        )
        for flag in (
            "te_physical_norm",
            "interface_two_sided", "interface_shared_occupancy",
            "graded_optics", "interface_tunneling",
        ):
            if settings[flag]:
                missing.append("settings." + flag)
        if has_any_ions:
            for flag in ("ion_steric_diffusion_only", "ion_steric_shared_site"):
                if settings[flag]:
                    missing.append("settings." + flag)
        despike = float(settings["het_recomb_despike"])
        if not math.isfinite(despike) or not 0.0 <= despike <= 1.0:
            missing.append("settings.het_recomb_despike")
    if settings["interface_plane_generation"] and not settings["interface_plane_closure"]:
        missing.append("settings.interface_plane_generation: requires settings.interface_plane_closure")
    if settings["interface_plane_projection"] and settings["interface_plane_closure"]:
        missing.append("settings.interface_plane_projection and settings.interface_plane_closure are mutually exclusive")
    charge_closure = settings["interface_charge_closure"]
    if charge_closure not in ("off", "equilibrium_referenced"):
        missing.append("settings.interface_charge_closure")
    elif charge_closure == "equilibrium_referenced" and not settings["interface_charge_rebaseline_acknowledged"]:
        missing.append("settings.interface_charge_closure: equilibrium_referenced requires interface_charge_rebaseline_acknowledged=True")
    if settings["jv_solver_policy"] != "general":
        missing.append("settings.jv_solver_policy: a separately qualified solver route is required")
    if data["grain_boundaries"]:
        missing.append("grain_boundaries: lateral geometry is not bound")
    channels = data.get("tunnelling_channels")
    if channels is not None:
        missing.extend("tunnelling_channels." + name for name, channel in channels.items() if channel.get("enabled"))
    declaration = data["declaration"]
    spectral = declaration.get("spectrum") is not None
    scalar_override = optical_model == "beer_lambert"
    if spectral and declaration.get("fixed_generation") is not None:
        missing.append("spectrum and fixed_generation are mutually exclusive")
    if optical_model is not None and declaration.get("fixed_generation") is not None:
        missing.append(
            f"optical_model={optical_model!r} and fixed_generation are mutually exclusive"
        )
    if optical_model in ("beer_lambert_spectral", "tmm") and not spectral:
        missing.append(
            f"optical_model={optical_model!r}: requires an explicit spectrum resource"
        )
    for layer in prepared.layers:
        if layer.role == "substrate":
            parameters = dict(layer.parameters)
            if scalar_override:
                if parameters["alpha"] != 0.0:
                    missing.append(
                        "layers."
                        + layer.id
                        + ": scalar beer_lambert does not bind absorbing substrates"
                    )
            elif not spectral and (
                parameters["alpha"] != 0.0
                or layer.optical is not None
                or parameters["n_optical"] is not None
            ):
                missing.append(
                    "layers."
                    + layer.id
                    + ": absorbing/optical substrate generation is not bound"
                )
    for layer in layers:
        parameters = dict(layer.parameters)
        prefix = "layers." + layer.id + "."
        for name in ("P0", "P0_neg", "D_ion", "D_ion_neg"):
            if parameters[name] != 0.0:
                missing.append(prefix + name + ": mobile or frozen ion populations are not bound")
        if parameters["carrier_statistics"] != "maxwell_boltzmann":
            missing.append(prefix + "carrier_statistics")
        if parameters["dopant_ionization_model"] != "fully_ionized":
            missing.append(prefix + "dopant_ionization_model")
        if parameters["band_gap_narrowing_model"] != "off":
            missing.append(prefix + "band_gap_narrowing_model")
        if any(parameters[name] != 0.0 for name in ("pf_gamma_n", "pf_gamma_p")):
            missing.append(prefix + "field_mobility: Poole-Frenkel enhancement is not bound")
        if layer.bulk_trap_distribution_json is not None:
            missing.append(prefix + "bulk_trap_distribution: distributed defects are not bound")
        if layer.metastable_document_json is not None:
            missing.append(prefix + "metastable_document")
        has_spatial_trap_profile = (
            any(parameters[name] is not None for name in ("trap_N_t_interface", "trap_N_t_bulk", "trap_decay_length"))
            if not allow_legacy_benchmark_deck
            else any(parameters[name] is not None for name in ("trap_N_t_interface", "trap_decay_length"))
        )
        if has_spatial_trap_profile:
            missing.append(prefix + "trap_profile")
        if layer.cigs_graded_optics_json is not None or (
            not spectral
            and not scalar_override
            and (layer.optical is not None or parameters["n_optical"] is not None)
        ):
            missing.append(prefix + "optical_material: requires an explicit spectrum; graded optics is not bound")
    if missing:
        raise CapabilityError("unsupported_classical_1d_dc: " + "; ".join(missing))


def _solve_dark_uncharged_two_sided_equilibrium(
    *,
    device_id: str,
    geometry: ControlVolumeGeometry1D,
    binding: ParameterBinding,
    poisson: PoissonOperator1D,
    electron_diffusivity: np.ndarray,
    hole_diffusivity: np.ndarray,
    generation: GenerationProfile,
    recombination: BulkRecombinationReaction,
    channels: tuple[CarrierExchangeChannel, ...],
    phi_left_V: float,
    built_in_V: float,
    fermi_eV: float,
    reference_phi: np.ndarray,
    reference_n: np.ndarray,
    reference_p: np.ndarray,
    additional_recombination: LocalRecombinationModel | None,
    field_transport: FieldTransport1D | None,
    provisional_specs: Sequence[tuple[int, int, int, int, TraceElectrostatics, TwoSidedInterfacePhysics, float]],
    thermionic_mode: ThermionicModelVersion = LEGACY_UNNORMALIZED_THERMIONIC_V1,
    band_step_mode: str = LEGACY_SUMMED_POTENTIAL_V1,
) -> np.ndarray:
    """Certify dark (V=0, G=0) charge-off two_sided_trace steady state for equilibrium_referenced."""
    uncharged_specs = tuple(
        TwoSidedInterfaceBindingSpec(
            interface_index=k,
            face=face,
            left_node=l_node,
            right_node=r_node,
            trace_electrostatics=trace_elec,
            physics=phys,
            charge=None,
            weight_left=0.0,
            weight_right=0.0,
        )
        for k, face, l_node, r_node, trace_elec, phys, _ in provisional_specs
    )
    prov_model = CompiledInterfaceModel1D(
        topology="two_sided_trace",
        identity_sha256="0" * 64,
        two_sided=uncharged_specs,
        thermionic_mode=thermionic_mode,
    )
    prov_dev = CompiledDevice1D(
        device_id,
        "0" * 64,
        geometry,
        binding,
        poisson,
        electron_diffusivity,
        hole_diffusivity,
        generation,
        recombination,
        channels,
        phi_left_V,
        built_in_V,
        fermi_eV,
        reference_phi,
        reference_n,
        reference_p,
        additional_recombination,
        capabilities=("ion_free", "classical_1d_dc", "heterointerface_1d"),
        field_transport=field_transport,
        interface_model=prov_model,
        band_step_mode=band_step_mode,
    )
    try:
        return solve_equilibrium(
            prov_dev,
            options=SteadyOptions(
                continuity_tolerance_A_m2=5.0e-11,
                convergence_policy=POISSON_CONTACT_BALANCE_V1,
            ),
        ).state.u
    except SteadyStateError as error:
        raise CapabilityError(
            "equilibrium_referenced: dark charge-off two-sided reference failed to converge"
        ) from error


def _bind_interfaces_1d(
    *,
    device_id: str,
    prepared: PreparedDevice,
    settings: Mapping[str, Any],
    layers: tuple[PreparedLayer, ...],
    geometry: ControlVolumeGeometry1D,
    binding: ParameterBinding,
    poisson: PoissonOperator1D,
    electron_diffusivity: np.ndarray,
    hole_diffusivity: np.ndarray,
    generation: GenerationProfile,
    recombination: BulkRecombinationReaction,
    channels: tuple[CarrierExchangeChannel, ...],
    eps_r_node: np.ndarray,
    reference_phi: np.ndarray,
    reference_n: np.ndarray,
    reference_p: np.ndarray,
    phi_left_V: float,
    built_in_V: float,
    fermi_eV: float,
    additional_recombination: LocalRecombinationModel | None,
    field_transport: FieldTransport1D | None,
    topology: InterfaceTopologyName,
    te_softness: float,
    te_regularization_mode: TERegularizationName,
    interface_transmission: float,
    equilibrium_occupancies: Sequence[float] | None,
    thermionic_mode: ThermionicModelVersion = LEGACY_UNNORMALIZED_THERMIONIC_V1,
    combination_mode: ThermionicCombinationMode | None = None,
    band_step_mode: str = LEGACY_SUMMED_POTENTIAL_V1,
    interface_band_reference_mode: InterfaceBandReferenceMode = REFERENCE_LAYER_BAND_ANCHOR,
    interface_defect_distribution_mode: InterfaceDefectDistributionMode = SINGLE_LEVEL_INTERFACE_DEFECT,
) -> CompiledInterfaceModel1D | None:
    iface_count = len(layers) - 1
    charge_closure = str(settings["interface_charge_closure"])
    plane_proj = bool(settings["interface_plane_projection"])
    plane_clos = bool(settings["interface_plane_closure"])
    plane_gen = bool(settings["interface_plane_generation"])

    if iface_count == 0:
        if topology == "two_sided_trace":
            raise CapabilityError("unsupported_classical_1d_dc: two_sided_trace requires at least two electrical layers")
        if charge_closure != "off" or plane_proj or plane_clos or plane_gen:
            raise CapabilityError("unsupported_classical_1d_dc: heterointerface closure requires at least one electrical interface")
        return None

    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for iface in prepared.interfaces:
        if iface.electrical:
            entry = iface.to_mapping()
            by_pair[(str(entry["left"]), str(entry["right"]))] = entry

    ordered_entries: list[dict[str, Any]] = []
    for k in range(iface_count):
        pair = (layers[k].id, layers[k + 1].id)
        if pair not in by_pair:
            raise CapabilityError(f"unsupported_classical_1d_dc: missing electrical interface for {pair[0]}->{pair[1]}")
        ordered_entries.append(by_pair[pair])

    bands = binding.band_structure
    configured = binding.intrinsic_product_m6
    vt = binding.thermal_voltage_V
    temp = binding.temperature_K

    if topology == "deduplicated":
        if interface_defect_distribution_mode != SINGLE_LEVEL_INTERFACE_DEFECT:
            raise ValueError(
                f"interface_defect_distribution_mode={interface_defect_distribution_mode!r} requires two_sided_trace topology"
            )
        if charge_closure != "off":
            raise CapabilityError(
                "unsupported_classical_1d_dc: settings.interface_charge_closure requires two_sided_trace topology"
            )
        if equilibrium_occupancies is not None:
            raise ValueError("equilibrium_occupancies applies only to equilibrium_referenced two_sided_trace")

        bare_specs: list[DeduplicatedBareSRHSpec] = []
        defect_specs: list[DeduplicatedDefectSRHSpec] = []
        te_specs: list[DeduplicatedThermionicFaceSpec] = []

        for k, entry in enumerate(ordered_entries):
            plane = geometry.interfaces[k]
            if plane.shared_node is None or not (2 <= plane.shared_node <= geometry.node_count - 3):
                raise CapabilityError("unsupported_classical_1d_dc: deduplicated interface requires an interior shared node")
            idx = int(plane.shared_node)
            face = idx - 1

            left_p = dict(layers[k].parameters)
            right_p = dict(layers[k + 1].parameters)
            left_nc = 0.0 if left_p["Nc300"] is None else float(cast(float, left_p["Nc300"]))
            left_nv = 0.0 if left_p["Nv300"] is None else float(cast(float, left_p["Nv300"]))
            a_star_n = resolve_layer_richardson(
                float(cast(float, left_p["A_star_n"])),
                left_nc,
                temp,
            )
            a_star_p = resolve_layer_richardson(
                float(cast(float, left_p["A_star_p"])),
                left_nv,
                temp,
            )
            delta_ec = float(
                bands.physical_electron_affinity_eV[face]
                - bands.physical_electron_affinity_eV[face + 1]
            )
            delta_ev = float(
                (bands.physical_electron_affinity_eV[face] + bands.physical_band_gap_eV[face])
                - (bands.physical_electron_affinity_eV[face + 1] + bands.physical_band_gap_eV[face + 1])
            )
            if thermionic_mode == LEGACY_UNNORMALIZED_THERMIONIC_V1:
                cap_e = a_star_n > 0.0 and abs(delta_ec) > TE_BARRIER_THRESHOLD_EV
                cap_h = a_star_p > 0.0 and abs(delta_ev) > TE_BARRIER_THRESHOLD_EV
                if cap_e or cap_h:
                    te_specs.append(
                        DeduplicatedThermionicFaceSpec(
                            interface_index=k,
                            face=face,
                            delta_Ec_eV=delta_ec,
                            delta_Ev_eV=delta_ev,
                            a_star_n_A_m2_K2=a_star_n,
                            a_star_p_A_m2_K2=a_star_p,
                            cap_electron=cap_e,
                            cap_hole=cap_h,
                        )
                    )
            else:
                right_nc = 0.0 if right_p["Nc300"] is None else float(cast(float, right_p["Nc300"]))
                right_nv = 0.0 if right_p["Nv300"] is None else float(cast(float, right_p["Nv300"]))
                a_star_n_r = resolve_layer_richardson(
                    float(cast(float, right_p["A_star_n"])),
                    right_nc,
                    temp,
                )
                a_star_p_r = resolve_layer_richardson(
                    float(cast(float, right_p["A_star_p"])),
                    right_nv,
                    temp,
                )
                cap_e = min(a_star_n, a_star_n_r) > 0.0
                cap_h = min(a_star_p, a_star_p_r) > 0.0
                if cap_e or cap_h:
                    te_specs.append(
                        DeduplicatedThermionicFaceSpec(
                            interface_index=k,
                            face=face,
                            delta_Ec_eV=delta_ec,
                            delta_Ev_eV=delta_ev,
                            a_star_n_A_m2_K2=a_star_n,
                            a_star_p_A_m2_K2=a_star_p,
                            cap_electron=cap_e,
                            cap_hole=cap_h,
                            a_star_n_right_A_m2_K2=a_star_n_r,
                            a_star_p_right_A_m2_K2=a_star_p_r,
                            dos_n_left_m3=float(binding.conduction_dos_m3[face]),
                            dos_n_right_m3=float(binding.conduction_dos_m3[face + 1]),
                            dos_p_left_m3=float(binding.valence_dos_m3[face]),
                            dos_p_right_m3=float(binding.valence_dos_m3[face + 1]),
                            hole_band_offset_eV=-delta_ev,
                            transmission=interface_transmission,
                        )
                    )

            defect = entry.get("defect")
            raw_vn = entry.get("v_n")
            raw_vp = entry.get("v_p")
            if defect is None:
                vn = 0.0 if raw_vn is None else float(raw_vn)
                vp = 0.0 if raw_vp is None else float(raw_vp)
                if not (math.isfinite(vn) and math.isfinite(vp)) or vn < 0.0 or vp < 0.0:
                    raise CapabilityError(f"interfaces.{entry['id']}: velocities must be finite and non-negative")
                if vn > 0.0 or vp > 0.0:
                    bare_specs.append(
                        DeduplicatedBareSRHSpec(
                            interface_index=k,
                            node=idx,
                            v_n_m_s=vn,
                            v_p_m_s=vp,
                            ni_sq_m6=float(configured[idx]),
                            n1_m3=float(binding.srh_n1_m3[idx]),
                            p1_m3=float(binding.srh_p1_m3[idx]),
                        )
                    )
            else:
                kinetics = defect["kinetics"]
                d_spec = InterfaceDefectSpec(
                    areal_density_m2=float(defect["total_density_m2"]),
                    trap_depth_eV=float(defect["trap_depth_eV"]),
                    energy_reference=defect["energy_reference"],
                    sigma_n_m2=float(kinetics["sigma_n_m2"]),
                    sigma_p_m2=float(kinetics["sigma_p_m2"]),
                    v_th_n_m_s=float(kinetics["thermal_velocity_n_m_s"]),
                    v_th_p_m_s=float(kinetics["thermal_velocity_p_m_s"]),
                    calibration_factor=float(defect["calibration_factor"]),
                )
                vn, vp = d_spec.effective_capture_velocities_m_s()
                i_l, i_r = idx - 1, idx + 1
                refs = resolve_interface_defect_references(
                    left_role=layers[k].role,
                    right_role=layers[k + 1].role,
                    left_chi_phys_eV=float(bands.physical_electron_affinity_eV[i_l]),
                    left_Eg_phys_eV=float(bands.physical_band_gap_eV[i_l]),
                    left_Nc_m3=float(binding.conduction_dos_m3[i_l]),
                    left_Nv_m3=float(binding.valence_dos_m3[i_l]),
                    right_chi_phys_eV=float(bands.physical_electron_affinity_eV[i_r]),
                    right_Eg_phys_eV=float(bands.physical_band_gap_eV[i_r]),
                    right_Nc_m3=float(binding.conduction_dos_m3[i_r]),
                    right_Nv_m3=float(binding.valence_dos_m3[i_r]),
                    thermal_voltage_V=vt,
                    trap_depth_eV=d_spec.trap_depth_eV,
                    energy_reference=d_spec.energy_reference,
                    band_reference_mode=interface_band_reference_mode,
                )
                _, pl_eq = doping_equilibrium_densities(
                    float(binding.donor_density_m3[i_l]),
                    float(binding.acceptor_density_m3[i_l]),
                    float(configured[i_l]),
                )
                nr_eq, _ = doping_equilibrium_densities(
                    float(binding.donor_density_m3[i_r]),
                    float(binding.acceptor_density_m3[i_r]),
                    float(configured[i_r]),
                )
                plane_prm = None
                if plane_clos:
                    ref_node = i_l if refs.reference_side == "left" else i_r
                    depth_below_cb = (
                        d_spec.trap_depth_eV
                        if d_spec.energy_reference == "below_conduction_band"
                        else refs.reference_gap_eV - d_spec.trap_depth_eV
                    )
                    plane_prm = build_plane_closure_params(
                        chi_left_folded_eV=float(bands.transport_electron_affinity_eV[i_l]),
                        chi_right_folded_eV=float(bands.transport_electron_affinity_eV[i_r]),
                        valence_left_folded_eV=float(
                            bands.transport_electron_affinity_eV[i_l]
                            + bands.transport_band_gap_eV[i_l]
                        ),
                        valence_right_folded_eV=float(
                            bands.transport_electron_affinity_eV[i_r]
                            + bands.transport_band_gap_eV[i_r]
                        ),
                        Nc_ref_m3=float(binding.conduction_dos_m3[ref_node]),
                        Nv_ref_m3=float(binding.valence_dos_m3[ref_node]),
                        chi_ref_phys_eV=float(bands.physical_electron_affinity_eV[ref_node]),
                        trap_depth_below_ref_cb_eV=depth_below_cb,
                        thermal_voltage_V=vt,
                    )
                if vn > 0.0 or vp > 0.0 or plane_clos or plane_proj:
                    defect_specs.append(
                        DeduplicatedDefectSRHSpec(
                            interface_index=k,
                            node=idx,
                            v_n_m_s=vn,
                            v_p_m_s=vp,
                            ni_sq_eff_m6=float(nr_eq * pl_eq),
                            n1_m3=refs.n1_midgap_symmetric_m3,
                            p1_m3=refs.p1_midgap_symmetric_m3,
                            plane_params=plane_prm,
                        )
                    )

        if (plane_proj or plane_clos) and not defect_specs:
            raise CapabilityError(
                "unsupported_classical_1d_dc: interface_plane_projection/closure requires at least one interface defect"
            )

        if not bare_specs and not defect_specs and not te_specs:
            return None

        manifest: dict[str, Any] = {
            "schema": "compiled_interface_model_1d_v1",
            "topology": "deduplicated",
            "plane_projection": plane_proj,
            "plane_closure": plane_clos,
            "plane_generation": plane_gen,
            "te_softness": te_softness,
            "te_regularization_mode": te_regularization_mode,
            "bare_srh": [
                [s.interface_index, s.node, s.v_n_m_s, s.v_p_m_s, s.ni_sq_m6, s.n1_m3, s.p1_m3]
                for s in bare_specs
            ],
            "defect_srh": [
                [
                    s.interface_index,
                    s.node,
                    s.v_n_m_s,
                    s.v_p_m_s,
                    s.ni_sq_eff_m6,
                    s.n1_m3,
                    s.p1_m3,
                    None
                    if s.plane_params is None
                    else [
                        s.plane_params.bn_L,
                        s.plane_params.bn_R,
                        s.plane_params.bp_L,
                        s.plane_params.bp_R,
                        s.plane_params.ni_s_sq_m6,
                        s.plane_params.n1_s_m3,
                        s.plane_params.p1_s_m3,
                        s.plane_params.s_supply_m_s,
                    ],
                ]
                for s in defect_specs
            ],
            "thermionic_faces": [
                [
                    s.interface_index,
                    s.face,
                    s.delta_Ec_eV,
                    s.delta_Ev_eV,
                    s.a_star_n_A_m2_K2,
                    s.a_star_p_A_m2_K2,
                    s.cap_electron,
                    s.cap_hole,
                ]
                for s in te_specs
            ],
        }
        resolved_comb = (
            None
            if thermionic_mode == LEGACY_UNNORMALIZED_THERMIONIC_V1
            else ("harmonic_series" if combination_mode is None else combination_mode)
        )
        if thermionic_mode != LEGACY_UNNORMALIZED_THERMIONIC_V1:
            manifest["thermionic_mode"] = thermionic_mode
            manifest["combination_mode"] = resolved_comb
            manifest["interface_transmission"] = interface_transmission
            manifest["thermionic_faces_v2"] = [
                [
                    s.interface_index,
                    s.face,
                    s.a_star_n_right_A_m2_K2,
                    s.a_star_p_right_A_m2_K2,
                    s.dos_n_left_m3,
                    s.dos_n_right_m3,
                    s.dos_p_left_m3,
                    s.dos_p_right_m3,
                    s.hole_band_offset_eV,
                    s.transmission,
                ]
                for s in te_specs
            ]
        if interface_band_reference_mode != REFERENCE_LAYER_BAND_ANCHOR:
            manifest["interface_band_reference_mode"] = interface_band_reference_mode
        iface_sha = hashlib.sha256(
            json.dumps(manifest, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        return CompiledInterfaceModel1D(
            topology="deduplicated",
            identity_sha256=iface_sha,
            bare_srh=tuple(bare_specs),
            defect_srh=tuple(defect_specs),
            thermionic_faces=tuple(te_specs),
            plane_projection=plane_proj,
            plane_closure=plane_clos,
            plane_generation=plane_gen,
            te_softness=te_softness,
            te_regularization_mode=te_regularization_mode,
            thermionic_mode=thermionic_mode,
            combination_mode=resolved_comb,
        )

    # Two-sided trace topology
    if plane_proj or plane_clos or plane_gen:
        raise CapabilityError(
            "unsupported_classical_1d_dc: deduplicated plane projection/closure flags cannot be combined with two_sided_trace"
        )

    provisional_specs: list[tuple[int, int, int, int, TraceElectrostatics, TwoSidedInterfacePhysics, float]] = []
    has_any_defect = False
    for k, entry in enumerate(ordered_entries):
        plane = geometry.interfaces[k]
        l_node, r_node = int(plane.left_node), int(plane.right_node)
        face = l_node
        if r_node != l_node + 1 or l_node <= 0 or r_node >= geometry.node_count - 1:
            raise CapabilityError("unsupported_classical_1d_dc: two_sided_trace requires adjacent interior bulk nodes")

        left_p = dict(layers[k].parameters)
        right_p = dict(layers[k + 1].parameters)
        a_n_l = resolve_layer_richardson(float(cast(float, left_p["A_star_n"])), 0.0 if left_p["Nc300"] is None else float(cast(float, left_p["Nc300"])), temp)
        a_n_r = resolve_layer_richardson(float(cast(float, right_p["A_star_n"])), 0.0 if right_p["Nc300"] is None else float(cast(float, right_p["Nc300"])), temp)
        a_p_l = resolve_layer_richardson(float(cast(float, left_p["A_star_p"])), 0.0 if left_p["Nv300"] is None else float(cast(float, left_p["Nv300"])), temp)
        a_p_r = resolve_layer_richardson(float(cast(float, right_p["A_star_p"])), 0.0 if right_p["Nv300"] is None else float(cast(float, right_p["Nv300"])), temp)
        rich_n = min(a_n_l, a_n_r)
        rich_p = min(a_p_l, a_p_r)
        if rich_n <= 0.0 or rich_p <= 0.0:
            raise CapabilityError(
                f"interfaces.{entry['id']}: two_sided_trace requires positive A_star_n and A_star_p on both adjacent layers"
            )

        defect = entry.get("defect")
        raw_vn = entry.get("v_n")
        raw_vp = entry.get("v_p")
        trap_quadrature: InterfaceTrapQuadrature | None = None
        if defect is None:
            vn = 0.0 if raw_vn is None else float(raw_vn)
            vp = 0.0 if raw_vp is None else float(raw_vp)
            if charge_closure == "equilibrium_referenced":
                raise CapabilityError(
                    f"interfaces.{entry['id']}: equilibrium_referenced charge closure requires a canonical microscopic defect on every electrical interface"
                )
            if vn != 0.0 or vp != 0.0:
                raise CapabilityError(
                    f"interfaces.{entry['id']}: two_sided_trace recombination requires an explicit interface defect to define two-sided emission densities"
                )
            n1_l = p1_l = n1_r = p1_r = 0.0
            trap_density_m2 = 0.0
        else:
            has_any_defect = True
            kinetics = defect["kinetics"]
            cal = float(defect["calibration_factor"])
            state_cal = float(defect["iface_state_calibration_factor"])
            if charge_closure == "equilibrium_referenced":
                if cal != 1.0 or state_cal != 1.0:
                    raise CapabilityError(
                        f"interfaces.{entry['id']}: equilibrium_referenced charge closure forbids empirical interface calibration factors"
                    )
                micro_pair = tuple(float(v) for v in entry["microscopic_capture_velocities_m_s"])
                if raw_vn is not None and raw_vp is not None and not (
                    math.isclose(float(raw_vn), micro_pair[0], rel_tol=1.0e-12, abs_tol=0.0)
                    and math.isclose(float(raw_vp), micro_pair[1], rel_tol=1.0e-12, abs_tol=0.0)
                ):
                    raise CapabilityError(
                        f"interfaces.{entry['id']}: compatibility SRV {(raw_vn, raw_vp)!r} does not match microscopic sigma*v_th*N_t {micro_pair!r}"
                    )
            d_spec = InterfaceDefectSpec(
                areal_density_m2=float(defect["total_density_m2"]),
                trap_depth_eV=float(defect["trap_depth_eV"]),
                energy_reference=defect["energy_reference"],
                sigma_n_m2=float(kinetics["sigma_n_m2"]),
                sigma_p_m2=float(kinetics["sigma_p_m2"]),
                v_th_n_m_s=float(kinetics["thermal_velocity_n_m_s"]),
                v_th_p_m_s=float(kinetics["thermal_velocity_p_m_s"]),
                calibration_factor=cal,
            )
            vn, vp = d_spec.effective_capture_velocities_m_s()
            refs = resolve_interface_defect_references(
                left_role=layers[k].role,
                right_role=layers[k + 1].role,
                left_chi_phys_eV=float(bands.physical_electron_affinity_eV[l_node]),
                left_Eg_phys_eV=float(bands.physical_band_gap_eV[l_node]),
                left_Nc_m3=float(binding.conduction_dos_m3[l_node]),
                left_Nv_m3=float(binding.valence_dos_m3[l_node]),
                right_chi_phys_eV=float(bands.physical_electron_affinity_eV[r_node]),
                right_Eg_phys_eV=float(bands.physical_band_gap_eV[r_node]),
                right_Nc_m3=float(binding.conduction_dos_m3[r_node]),
                right_Nv_m3=float(binding.valence_dos_m3[r_node]),
                thermal_voltage_V=vt,
                trap_depth_eV=d_spec.trap_depth_eV,
                energy_reference=d_spec.energy_reference,
                require_in_gap=(charge_closure == "equilibrium_referenced"),
                band_reference_mode=interface_band_reference_mode,
            )
            n1_l, p1_l = refs.n1_left_m3, refs.p1_left_m3
            n1_r, p1_r = refs.n1_right_m3, refs.p1_right_m3
            trap_density_m2 = d_spec.areal_density_m2
            if interface_defect_distribution_mode == SCAPS_GAUSSIAN_INTERFACE_QUADRATURE:
                partner_meta = defect.get("partner_metadata")
                if (
                    isinstance(partner_meta, Mapping)
                    and partner_meta.get("distribution") == "gaussian"
                    and partner_meta.get("E_char_eV") is not None
                    and float(partner_meta["E_char_eV"]) > 0.0
                ):
                    trap_quadrature = build_scaps_gaussian_interface_quadrature(
                        left_role=layers[k].role,
                        right_role=layers[k + 1].role,
                        left_chi_phys_eV=float(bands.physical_electron_affinity_eV[l_node]),
                        left_Eg_phys_eV=float(bands.physical_band_gap_eV[l_node]),
                        left_Nc_m3=float(binding.conduction_dos_m3[l_node]),
                        left_Nv_m3=float(binding.valence_dos_m3[l_node]),
                        right_chi_phys_eV=float(bands.physical_electron_affinity_eV[r_node]),
                        right_Eg_phys_eV=float(bands.physical_band_gap_eV[r_node]),
                        right_Nc_m3=float(binding.conduction_dos_m3[r_node]),
                        right_Nv_m3=float(binding.valence_dos_m3[r_node]),
                        thermal_voltage_V=vt,
                        trap_depth_eV=d_spec.trap_depth_eV,
                        energy_reference=d_spec.energy_reference,
                        characteristic_energy_eV=float(partner_meta["E_char_eV"]),
                        band_reference_mode=interface_band_reference_mode,
                    )

        trace_elec = TraceElectrostatics.from_plane(
            plane,
            float(eps_r_node[l_node]),
            float(eps_r_node[r_node]),
        )
        phys = TwoSidedInterfacePhysics(
            thermal_voltage_V=vt,
            temperature_K=temp,
            D_n_left_m2_s=float(binding.electron_diffusivity_m2_s[l_node]),
            D_n_right_m2_s=float(binding.electron_diffusivity_m2_s[r_node]),
            D_p_left_m2_s=float(binding.hole_diffusivity_m2_s[l_node]),
            D_p_right_m2_s=float(binding.hole_diffusivity_m2_s[r_node]),
            N_C_left_m3=float(binding.conduction_dos_m3[l_node]),
            N_C_right_m3=float(binding.conduction_dos_m3[r_node]),
            N_V_left_m3=float(binding.valence_dos_m3[l_node]),
            N_V_right_m3=float(binding.valence_dos_m3[r_node]),
            richardson_n_A_m2_K2=rich_n,
            richardson_p_A_m2_K2=rich_p,
            conduction_band_step_eV=float(
                bands.physical_electron_affinity_eV[l_node]
                - bands.physical_electron_affinity_eV[r_node]
            ),
            hole_transport_step_eV=float(
                (bands.physical_electron_affinity_eV[r_node] + bands.physical_band_gap_eV[r_node])
                - (bands.physical_electron_affinity_eV[l_node] + bands.physical_band_gap_eV[l_node])
            ),
            transmission=interface_transmission,
            surface_recombination_velocity_n_m_s=vn,
            surface_recombination_velocity_p_m_s=vp,
            n1_left_m3=n1_l,
            n1_right_m3=n1_r,
            p1_left_m3=p1_l,
            p1_right_m3=p1_r,
            thermionic_mode=thermionic_mode,
            carrier_statistics=(
                "fermi_dirac"
                if thermionic_mode == LEGACY_UNNORMALIZED_THERMIONIC_V1
                else "maxwell_boltzmann"
            ),
            a_star_n_left_A_m2_K2=(
                None if thermionic_mode == LEGACY_UNNORMALIZED_THERMIONIC_V1 else a_n_l
            ),
            a_star_n_right_A_m2_K2=(
                None if thermionic_mode == LEGACY_UNNORMALIZED_THERMIONIC_V1 else a_n_r
            ),
            a_star_p_left_A_m2_K2=(
                None if thermionic_mode == LEGACY_UNNORMALIZED_THERMIONIC_V1 else a_p_l
            ),
            a_star_p_right_A_m2_K2=(
                None if thermionic_mode == LEGACY_UNNORMALIZED_THERMIONIC_V1 else a_p_r
            ),
            trap_quadrature=trap_quadrature,
        )
        provisional_specs.append((k, face, l_node, r_node, trace_elec, phys, trap_density_m2))

    eq_occ: list[float] | None = None
    if charge_closure == "equilibrium_referenced":
        if not has_any_defect:
            raise CapabilityError(
                "unsupported_classical_1d_dc: equilibrium_referenced charge closure requires interface defects"
            )
        if equilibrium_occupancies is not None:
            if len(equilibrium_occupancies) != iface_count:
                raise ValueError("equilibrium_occupancies must match the number of electrical interfaces")
            eq_occ = [float(v) for v in equilibrium_occupancies]
            if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in eq_occ):
                raise ValueError("equilibrium_occupancies must lie in [0, 1]")
        else:
            u_dark = _solve_dark_uncharged_two_sided_equilibrium(
                device_id=device_id,
                geometry=geometry,
                binding=binding,
                poisson=poisson,
                electron_diffusivity=electron_diffusivity,
                hole_diffusivity=hole_diffusivity,
                generation=generation,
                recombination=recombination,
                channels=channels,
                phi_left_V=phi_left_V,
                built_in_V=built_in_V,
                fermi_eV=fermi_eV,
                reference_phi=reference_phi,
                reference_n=reference_n,
                reference_p=reference_p,
                additional_recombination=additional_recombination,
                field_transport=field_transport,
                provisional_specs=provisional_specs,
                thermionic_mode=thermionic_mode,
                band_step_mode=band_step_mode,
            )
            count_nodes = geometry.node_count
            phi_eq = u_dark[:count_nodes]
            zn_eq = u_dark[count_nodes:2 * count_nodes]
            zp_eq = u_dark[2 * count_nodes:]
            shift_eq = (phi_eq - reference_phi) / vt
            n_eq = reference_n * np.exp(shift_eq + zn_eq)
            p_eq = reference_p * np.exp(-shift_eq - zp_eq)
            eq_occ = []
            for _, _, l_node, r_node, trace_elec, phys, _ in provisional_specs:
                dark_qss = solve_two_sided_interface_qss(
                    trace_elec,
                    phys,
                    TwoSidedBulkState(
                        phi_left_V=float(phi_eq[l_node]),
                        phi_right_V=float(phi_eq[r_node]),
                        n_left_m3=float(n_eq[l_node]),
                        p_left_m3=float(p_eq[l_node]),
                        n_right_m3=float(n_eq[r_node]),
                        p_right_m3=float(p_eq[r_node]),
                        electron_electrochemical_step=float(zn_eq[r_node] - zn_eq[l_node]),
                        hole_electrochemical_step=float(-(zp_eq[r_node] - zp_eq[l_node])),
                    ),
                )
                if dark_qss.occupancy is None:
                    eq_occ.append(0.0)
                else:
                    eq_occ.append(float(dark_qss.occupancy))
    elif equilibrium_occupancies is not None:
        raise ValueError("equilibrium_occupancies requires settings.interface_charge_closure='equilibrium_referenced'")

    two_sided_specs: list[TwoSidedInterfaceBindingSpec] = []
    manifest_two_sided: list[object] = []
    for idx_spec, (k, face, l_node, r_node, trace_elec, phys, trap_nt) in enumerate(provisional_specs):
        charge_obj: EquilibriumReferencedSheetCharge | None = None
        wl = wr = 0.0
        if eq_occ is not None:
            charge_obj = EquilibriumReferencedSheetCharge(
                equilibrium_occupancy=eq_occ[idx_spec],
                trap_density_m2=trap_nt,
            )
            wl, wr = sheet_charge_row_weights(
                trace_elec.eps_r_left,
                trace_elec.left_distance_m,
                trace_elec.eps_r_right,
                trace_elec.right_distance_m,
            )
        two_sided_specs.append(
            TwoSidedInterfaceBindingSpec(
                interface_index=k,
                face=face,
                left_node=l_node,
                right_node=r_node,
                trace_electrostatics=trace_elec,
                physics=phys,
                charge=charge_obj,
                weight_left=wl,
                weight_right=wr,
            )
        )
        manifest_two_sided.append(
            [
                k,
                face,
                l_node,
                r_node,
                trace_elec.left_distance_m,
                trace_elec.right_distance_m,
                trace_elec.eps_r_left,
                trace_elec.eps_r_right,
                phys.richardson_n_A_m2_K2,
                phys.richardson_p_A_m2_K2,
                phys.conduction_band_step_eV,
                phys.hole_transport_step_eV,
                phys.transmission,
                phys.surface_recombination_velocity_n_m_s,
                phys.surface_recombination_velocity_p_m_s,
                phys.n1_left_m3,
                phys.n1_right_m3,
                phys.p1_left_m3,
                phys.p1_right_m3,
                None if charge_obj is None else [charge_obj.equilibrium_occupancy, charge_obj.trap_density_m2, wl, wr],
            ]
        )

    manifest_ts: dict[str, Any] = {
        "schema": "compiled_interface_model_1d_v1",
        "topology": "two_sided_trace",
        "charge_closure": charge_closure,
        "two_sided": manifest_two_sided,
    }
    if thermionic_mode != LEGACY_UNNORMALIZED_THERMIONIC_V1:
        manifest_ts["thermionic_mode"] = thermionic_mode
        manifest_ts["carrier_statistics"] = "maxwell_boltzmann"
        manifest_ts["two_sided_richardson_sides"] = [
            [
                s.physics.a_star_n_left_A_m2_K2,
                s.physics.a_star_n_right_A_m2_K2,
                s.physics.a_star_p_left_A_m2_K2,
                s.physics.a_star_p_right_A_m2_K2,
            ]
            for s in two_sided_specs
        ]
    if interface_band_reference_mode != REFERENCE_LAYER_BAND_ANCHOR:
        manifest_ts["interface_band_reference_mode"] = interface_band_reference_mode
    if interface_defect_distribution_mode != SINGLE_LEVEL_INTERFACE_DEFECT:
        manifest_ts["interface_defect_distribution_mode"] = (
            interface_defect_distribution_mode
        )
        manifest_ts["two_sided_trap_quadrature"] = [
            None
            if s.physics.trap_quadrature is None
            else [
                s.physics.trap_quadrature.energies_eV.tolist(),
                s.physics.trap_quadrature.weights.tolist(),
                s.physics.trap_quadrature.n1_left_m3.tolist(),
                s.physics.trap_quadrature.p1_left_m3.tolist(),
                s.physics.trap_quadrature.n1_right_m3.tolist(),
                s.physics.trap_quadrature.p1_right_m3.tolist(),
            ]
            for s in two_sided_specs
        ]
    iface_sha = hashlib.sha256(
        json.dumps(manifest_ts, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return CompiledInterfaceModel1D(
        topology="two_sided_trace",
        identity_sha256=iface_sha,
        two_sided=tuple(two_sided_specs),
        thermionic_mode=thermionic_mode,
    )


def _layer_input(
    layer: PreparedLayer, *, allow_legacy_benchmark_deck: bool = False
) -> LayerParameterInput:
    parameters = dict(layer.parameters)
    if not allow_legacy_benchmark_deck:
        for name in ("chi", "Eg", "Nc300", "Nv300", "ni", "eps_r", "mu_n", "mu_p"):
            value = parameters[name]
            if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise CapabilityError(f"layers.{layer.id}.{name}: classical 1D DC requires a finite positive value")
    else:
        for name in ("ni", "eps_r", "mu_n", "mu_p"):
            value = parameters[name]
            if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise CapabilityError(f"layers.{layer.id}.{name}: classical 1D DC requires a finite positive value")
        for name in ("chi", "Eg"):
            value = parameters[name]
            if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
                raise CapabilityError(f"layers.{layer.id}.{name}: requires a finite non-negative value")
        for name in ("Nc300", "Nv300"):
            value = parameters[name]
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0.0):
                raise CapabilityError(f"layers.{layer.id}.{name}: requires None or a finite positive value")
    # Preparation has resolved the full scalar schema; the binding objects
    # below validate their model-specific domains. This cast changes no values.
    p = cast(_LayerScalars, parameters)
    return LayerParameterInput(
        name=layer.id, role=layer.role, thickness_m=layer.thickness,
        electron_affinity_eV=p["chi"], band_gap_eV=p["Eg"],
        intrinsic_density_m3=p["ni"], srh_n1_m3=p["n1"], srh_p1_m3=p["p1"],
        electron_mobility_m2_V_s=p["mu_n"], hole_mobility_m2_V_s=p["mu_p"],
        radiative_coefficient_m3_s=p["B_rad"], thermal_velocity_m_s=p["v_th"],
        doping=DopingProfileSpec(
            p["N_A"], p["N_D"], p["N_A_bulk"], p["N_D_bulk"],
            p["doping_profile_shape"], p["doping_decay_length"], p["doping_edge"],
        ),
        conduction_dos_300_m3=p["Nc300"], valence_dos_300_m3=p["Nv300"],
        temperature_laws=TemperatureLawSpec(
            mobility_exponent=p["mu_T_gamma"], radiative_exponent=p["B_rad_T_gamma"],
            varshni_alpha_eV_K=p["varshni_alpha"], varshni_beta_K=p["varshni_beta"],
        ),
        grading=GradingSpec(
            band_gap_back_eV=p["Eg_back"], electron_affinity_back_eV=p["chi_back"],
            profile=p["grading_profile"], direction=p["grading_direction"],
            bowing_eV=p["grading_bowing"], characteristic_length_m=p["grading_char_length"],
        ),
    )


def _bind_legacy_scaps_neutral_defects(
    layers: Sequence[PreparedLayer],
    masks: Sequence[np.ndarray],
    *,
    ni_sq_m6: np.ndarray,
    band_gap_eV: np.ndarray,
    thermal_voltage_V: float,
    conduction_dos_m3: np.ndarray,
    valence_dos_m3: np.ndarray,
) -> tuple[NeutralExplicitDefectModel | None, bool]:
    """Bind SCAPS unresolved-charge single-level bulk defects as parallel neutral SRH."""
    needs_scaps_fallback = any(
        layer.defects
        and (
            layer.defect_model == "effective_lifetime"
            or any(
                isinstance(item, PreparedDefect)
                and item.to_mapping()["charge_transition"] == "unresolved"
                and item.to_mapping()["neutral_reference"] == "unresolved"
                for item in layer.defects
            )
        )
        for layer in layers
    )
    if not needs_scaps_fallback:
        return (
            bind_neutral_defects(
                layers,
                masks,
                ni_sq_m6=ni_sq_m6,
                band_gap_eV=band_gap_eV,
                thermal_voltage_V=thermal_voltage_V,
                conduction_dos_m3=conduction_dos_m3,
                valence_dos_m3=valence_dos_m3,
            ),
            False,
        )
    species: list[NeutralDefectSpecies] = []
    documents: list[tuple[str, str]] = []
    for layer, mask in zip(layers, masks):
        if not layer.defects:
            continue
        where = f"layers.{layer.id}.bulk_defects"
        if any(not isinstance(item, PreparedDefect) for item in layer.defects):
            raise CapabilityError(f"{where}: multivalent defects are not bound")
        if any(
            value.shape != ni_sq_m6.shape
            or not np.all(np.isfinite(value[mask]))
            or np.any(value[mask] <= 0.0)
            for value in (conduction_dos_m3, valence_dos_m3)
        ):
            raise CapabilityError(
                f"{where}: literal defect energies require finite positive physical DOS on owned nodes"
            )
        resolved = [item.to_mapping() for item in layer.defects]
        document = {
            "schema_version": layer.defect_schema_version,
            "defect_model": layer.defect_model,
            "bulk_defects": resolved,
        }
        digest = hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        documents.append((layer.id, digest))
        for item in resolved:
            identifier = f"{layer.id}/{item['id']}"
            distribution, kinetics = item["distribution"], item["kinetics"]
            ct = item["charge_transition"]
            nr = item["neutral_reference"]
            if not (
                (ct == "neutral" and nr == "all_occupancies")
                or (ct == "unresolved" and nr == "unresolved")
            ):
                raise CapabilityError(f"{where}.{item['id']}: charged defects are not bound")
            if distribution["kind"] != "single_level" or item.get("spatial_profile") is not None:
                raise CapabilityError(f"{where}.{item['id']}: distributed defects are not bound")
            if item["degeneracy"] != 1.0:
                raise CapabilityError(f"{where}.{item['id']}: neutral DEF-1 requires unit degeneracy")
            total_nt = float(distribution["total_density_m3"])
            if total_nt == 0.0:
                continue
            center = float(distribution["center_eV_above_vb"])
            if center < 0.0 or np.any(center > band_gap_eV[mask]):
                raise CapabilityError(f"{where}.{item['id']}: resolved level lies outside the physical gap")
            try:
                compiled = compile_neutral_defect_species(
                    identifier=identifier,
                    document_sha256=digest,
                    active_nodes=mask,
                    ni_sq_m6=ni_sq_m6,
                    band_gap_eV=band_gap_eV,
                    thermal_voltage_V=thermal_voltage_V,
                    total_density_m3=total_nt,
                    center_eV_above_vb=center,
                    sigma_n_m2=kinetics["sigma_n_m2"],
                    sigma_p_m2=kinetics["sigma_p_m2"],
                    thermal_velocity_n_m_s=kinetics["thermal_velocity_n_m_s"],
                    thermal_velocity_p_m_s=kinetics["thermal_velocity_p_m_s"],
                    charge_transition="neutral",
                    distribution=distribution["kind"],
                    degeneracy=item["degeneracy"],
                )
            except ValueError as error:
                raise CapabilityError(f"{where}.{item['id']}: {error}") from error
            changed = compiled.active_nodes & (conduction_dos_m3 != valence_dos_m3)
            if np.any(changed):
                n1 = np.array(compiled.n1_m3, copy=True)
                p1 = np.array(compiled.p1_m3, copy=True)
                factor = np.sqrt(conduction_dos_m3[changed]) / np.sqrt(valence_dos_m3[changed])
                n1[changed] *= factor
                p1[changed] /= factor
                compiled = replace(compiled, n1_m3=n1, p1_m3=p1)
            species.append(compiled)
    if not species:
        return None, True
    owned = np.logical_or.reduce([item.active_nodes for item in species])
    return NeutralExplicitDefectModel(tuple(species), owned, tuple(documents)), True


def _bind_legacy_front_incoherent_tmm_1d(
    prepared: PreparedDevice, control_volumes: ControlVolumes
) -> GenerationProfile:
    """Bind spectral TMM with an incoherent front optical substrate (e.g. glass)."""
    data = prepared.to_mapping()
    name = data["declaration"].get("spectrum")
    if name is None:
        raise CapabilityError("optical_generation_1d requires an explicit spectrum resource")
    table = prepared.resources.get(name, "spectrum")
    if table.units != ("m", "m^-3/s"):
        raise CapabilityError("optical_generation_1d requires spectral photon flux per metre, not irradiance")
    table_wl_nm = table.array[:, 0] * 1e9
    needs_legacy_window = any(
        layer.optical is not None
        and (
            float(layer.optical.array[0, 0] * 1e9) > float(table_wl_nm[0])
            or float(layer.optical.array[-1, 0] * 1e9) < float(table_wl_nm[-1])
        )
        for layer in prepared.layers
    )
    if needs_legacy_window:
        wavelengths_nm = np.linspace(300.0, 1000.0, 200)
        flux_per_m = np.interp(wavelengths_nm, table_wl_nm, table.array[:, 1])
        spectrum = SpectralPhotonFlux(wavelengths_nm, flux_per_m, table.source.id)
    else:
        wavelengths_nm = table_wl_nm
        spectrum = SpectralPhotonFlux(wavelengths_nm, table.array[:, 1], table.source.id)
    dependencies: dict[str, str] = {
        "binding_convention": "prepared_front_incoherent_spectral_tmm_1d_v1",
        "prepared_device_sha256": prepared.content_sha256,
        "spectrum_resource_sha256": table.content_sha256,
        "temperature_K": str(data["settings"]["T"]),
        "monochromatic_Phi_policy": "not_used_absolute_spectrum",
    }
    specs: list[OpticalLayerSpec] = []
    for index, layer in enumerate(prepared.layers):
        parameters = dict(layer.parameters)
        incoherent = bool(parameters["incoherent"])
        if incoherent and (index > 0 or layer.role != "substrate"):
            raise CapabilityError(
                f"optical_generation_1d: {layer.id}: only a front substrate may be incoherent"
            )
        if layer.cigs_graded_optics_json is not None:
            raise CapabilityError(f"optical_generation_1d: {layer.id} has unsupported graded optics")
        optical = layer.optical
        if optical is not None:
            if parameters["n_optical"] is not None:
                raise CapabilityError(
                    f"optical_generation_1d: {layer.id} must select a table or constant index, not both"
                )
            native = optical.array
            try:
                n, k = sample_nk_table(
                    native[:, 0] * 1e9, native[:, 1], native[:, 2], wavelengths_nm, material=optical.name
                )
            except ValueError as error:
                raise CapabilityError(f"optical_generation_1d: {layer.id}: {error}") from error
            dependencies["nk_resource:" + layer.id] = optical.content_sha256
        else:
            if parameters["n_optical"] is None:
                raise CapabilityError(
                    f"optical_generation_1d: {layer.id} needs explicit n_optical or an n-k resource"
                )
            n, k = constant_index_optical_constants(
                wavelengths_nm,
                absorption_coefficient_m1=cast(float, parameters["alpha"]),
                refractive_index=cast(float, parameters["n_optical"]),
            )
        specs.append(
            OpticalLayerSpec(
                layer.id,
                layer.thickness,
                n,
                k,
                incoherent=incoherent,
                electrical=layer.role != "substrate",
            )
        )
    binding = OpticalStackBinding(
        tuple(specs),
        spectrum,
        ambient_index=1.0,
        exit_index=1.0,
        incidence="front",
        dependencies=tuple(dependencies.items()),
    )
    try:
        profile = binding.generation_profile(control_volumes)
        budget = binding.photon_budget()
    except ValueError as error:
        raise CapabilityError("optical_generation_1d: " + str(error)) from error
    for value in (
        profile.source_m3_s,
        profile.absorbed_m2_s,
        budget.reflectance,
        budget.transmittance,
        budget.layer_absorptance,
        budget.dropped_multipass,
    ):
        if not np.all(np.isfinite(value)) or np.any(value < 0.0):
            raise CapabilityError(
                "optical_generation_1d: generation and photon fractions must be finite and nonnegative"
            )
    if np.max(np.abs(budget.spectral_closure_residual())) > _SPECTRAL_BL_BUDGET_RTOL:
        raise CapabilityError("optical_generation_1d: spectral photon accounting does not close")
    tolerance = _SPECTRAL_BL_BUDGET_RTOL * budget.incident_m2_s
    weighted_source = float(np.sum(profile.source_m3_s * control_volumes.cell_measures_m))
    if (
        abs(profile.areal_generation_m2_s - budget.electrical_absorbed_m2_s) > tolerance
        or abs(weighted_source - budget.electrical_absorbed_m2_s) > tolerance
    ):
        raise CapabilityError(
            "optical_generation_1d: physical control volumes do not close the electrical photon budget"
        )
    return profile


def bind_spectral_beer_lambert_1d(
    prepared: PreparedDevice,
    control_volumes: ControlVolumes,
) -> GenerationProfile:
    """Bind single-pass spectral Beer-Lambert optics to 1D control volumes."""

    data = prepared.to_mapping()
    name = data["declaration"].get("spectrum")
    if name is None:
        raise CapabilityError(
            "beer_lambert_spectral_1d requires an explicit spectrum resource"
        )
    if data["declaration"].get("fixed_generation") is not None:
        raise CapabilityError(
            "beer_lambert_spectral_1d: spectrum and fixed_generation "
            "are mutually exclusive"
        )
    if data["settings"]["graded_optics"]:
        raise CapabilityError("beer_lambert_spectral_1d: graded optics is not bound")
    table = prepared.resources.get(name, "spectrum")
    if table.units != ("m", "m^-3/s"):
        raise CapabilityError(
            "beer_lambert_spectral_1d requires spectral photon flux per metre, "
            "not irradiance"
        )
    wavelengths_nm = table.array[:, 0] * 1e9
    spectrum = SpectralPhotonFlux(wavelengths_nm, table.array[:, 1], table.source.id)
    dependencies: dict[str, str] = {
        "binding_convention": _SPECTRAL_BL_BINDING_1D_V1,
        "prepared_device_sha256": prepared.content_sha256,
        "spectrum_resource_sha256": table.content_sha256,
        "temperature_K": str(data["settings"]["T"]),
        "monochromatic_Phi_policy": "not_used_absolute_spectrum",
    }
    specs: list[OpticalLayerSpec] = []
    for index, layer in enumerate(prepared.layers):
        parameters = dict(layer.parameters)
        incoherent = bool(parameters["incoherent"])
        if incoherent and index > 0:
            raise CapabilityError(
                f"beer_lambert_spectral_1d: {layer.id}: only the front optical "
                "layer may be incoherent"
            )
        if layer.cigs_graded_optics_json is not None:
            raise CapabilityError(
                f"beer_lambert_spectral_1d: {layer.id} has unsupported graded optics"
            )
        optical = layer.optical
        if optical is not None:
            if parameters["n_optical"] is not None:
                raise CapabilityError(
                    f"beer_lambert_spectral_1d: {layer.id} must select a table "
                    "or constant index, not both"
                )
            native = optical.array
            try:
                n, k = sample_nk_table(
                    native[:, 0] * 1e9,
                    native[:, 1],
                    native[:, 2],
                    wavelengths_nm,
                    material=optical.name,
                )
            except ValueError as error:
                raise CapabilityError(
                    f"beer_lambert_spectral_1d: {layer.id}: {error}"
                ) from error
            dependencies["nk_resource:" + layer.id] = optical.content_sha256
        else:
            if parameters["n_optical"] is None:
                raise CapabilityError(
                    f"beer_lambert_spectral_1d: {layer.id} needs explicit "
                    "n_optical or an n-k resource"
                )
            n, k = constant_index_optical_constants(
                wavelengths_nm,
                absorption_coefficient_m1=cast(float, parameters["alpha"]),
                refractive_index=cast(float, parameters["n_optical"]),
            )
        specs.append(
            OpticalLayerSpec(
                layer.id,
                layer.thickness,
                n,
                k,
                incoherent=incoherent,
                electrical=layer.role != "substrate",
            )
        )
    binding = SpectralBeerLambertBinding(
        tuple(specs),
        spectrum,
        incidence="front",
        dependencies=tuple(dependencies.items()),
    )
    try:
        profile = binding.generation_profile(control_volumes)
        budget = binding.photon_budget()
    except ValueError as error:
        raise CapabilityError("beer_lambert_spectral_1d: " + str(error)) from error
    for value in (
        profile.source_m3_s,
        profile.absorbed_m2_s,
        budget.reflectance,
        budget.transmittance,
        budget.layer_absorptance,
        budget.dropped_multipass,
    ):
        if not np.all(np.isfinite(value)) or np.any(value < 0.0):
            raise CapabilityError(
                "beer_lambert_spectral_1d: generation and photon fractions "
                "must be finite and nonnegative"
            )
    if np.max(np.abs(budget.spectral_closure_residual())) > _SPECTRAL_BL_BUDGET_RTOL:
        raise CapabilityError(
            "beer_lambert_spectral_1d: spectral photon accounting does not close"
        )
    tolerance = _SPECTRAL_BL_BUDGET_RTOL * budget.incident_m2_s
    weighted_source = float(
        np.sum(profile.source_m3_s * control_volumes.cell_measures_m)
    )
    if (
        abs(profile.areal_generation_m2_s - budget.electrical_absorbed_m2_s)
        > tolerance
        or abs(weighted_source - budget.electrical_absorbed_m2_s) > tolerance
    ):
        raise CapabilityError(
            "beer_lambert_spectral_1d: physical control volumes do not close "
            "the electrical photon budget"
        )
    return profile


def compile_device_1d(
    prepared: PreparedDevice,
    *,
    intervals_per_layer: int | Sequence[int] = 24,
    additional_recombination: LocalRecombinationModel | None = None,
    interface_topology: InterfaceTopologyName | None = None,
    te_softness: float = 0.02,
    te_regularization_mode: TERegularizationName = "legacy_logistic",
    interface_transmission: float = 1.0,
    equilibrium_occupancies: Sequence[float] | None = None,
    optical_model: OpticalModelName | None = None,
    dielectric_rule: str | None = None,
    geometry_name: str | None = None,
    band_step_mode: str = LEGACY_SUMMED_POTENTIAL_V1,
    thermionic_mode: ThermionicModelVersion | None = None,
    combination_mode: ThermionicCombinationMode | None = None,
    interface_band_reference_mode: InterfaceBandReferenceMode = REFERENCE_LAYER_BAND_ANCHOR,
    interface_defect_distribution_mode: InterfaceDefectDistributionMode = SINGLE_LEVEL_INTERFACE_DEFECT,
    allow_legacy_benchmark_deck: bool = False,
) -> CompiledDevice1D:
    """Cold compile genuine resolved configuration; no legacy/test execution.

    A scalar interval count is the mean per-layer budget, allocated by the
    prepared electrical_grid weights. A sequence explicitly supplies each
    electrical layer's interval count. Prepared grid alpha values are always
    used; without declarations the existing tanh-grid alpha is 3.
    """
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("compile_device_1d requires an existing PreparedDevice")
    if not isinstance(allow_legacy_benchmark_deck, bool):
        raise TypeError("allow_legacy_benchmark_deck must be a bool")
    legacy_opt_in = allow_legacy_benchmark_deck or _LEGACY_BENCHMARK_DECK_OPT_IN.get()
    if interface_topology is not None and interface_topology not in ("deduplicated", "two_sided_trace"):
        raise ValueError("interface_topology must be 'deduplicated', 'two_sided_trace', or None")
    if optical_model is not None and optical_model not in (
        "beer_lambert",
        "beer_lambert_spectral",
        "tmm",
    ):
        raise ValueError(
            "optical_model must be 'beer_lambert', 'beer_lambert_spectral', "
            "'tmm', or None"
        )
    if dielectric_rule is not None and dielectric_rule not in DIELECTRIC_RULES:
        raise ValueError(f"unknown dielectric_rule {dielectric_rule!r}")
    if geometry_name is not None and geometry_name not in GEOMETRY_NAMES:
        raise ValueError(f"unknown geometry_name {geometry_name!r}")
    if band_step_mode not in BAND_STEP_MODES:
        raise ValueError(f"unknown band_step_mode {band_step_mode!r}")
    if interface_band_reference_mode not in INTERFACE_BAND_REFERENCE_MODES:
        raise ValueError(
            f"unknown interface_band_reference_mode {interface_band_reference_mode!r}"
        )
    if interface_defect_distribution_mode not in INTERFACE_DEFECT_DISTRIBUTION_MODES:
        raise ValueError(
            f"unknown interface_defect_distribution_mode {interface_defect_distribution_mode!r}"
        )
    resolved_thermionic: ThermionicModelVersion = (
        LEGACY_UNNORMALIZED_THERMIONIC_V1 if thermionic_mode is None else thermionic_mode
    )
    if resolved_thermionic not in THERMIONIC_MODEL_VERSIONS:
        raise ValueError(f"unknown thermionic_mode {thermionic_mode!r}")
    if combination_mode is not None and combination_mode not in (
        "harmonic_series",
        "compact_support",
        "legacy_logistic",
        "hard",
    ):
        raise ValueError(f"unknown combination_mode {combination_mode!r}")
    if combination_mode is not None and resolved_thermionic == LEGACY_UNNORMALIZED_THERMIONIC_V1:
        raise ValueError(
            "combination_mode requires thermionic_mode='reciprocal_physical_thermionic_v2' "
            "or 'scaps_effective_velocity_thermionic_v1'"
        )
    if te_regularization_mode not in ("legacy_logistic", "compact_support"):
        raise ValueError("te_regularization_mode must be 'legacy_logistic' or 'compact_support'")
    if isinstance(te_softness, bool) or not math.isfinite(float(te_softness)) or float(te_softness) < 0.0:
        raise ValueError("te_softness must be a finite non-negative float")
    if (
        isinstance(interface_transmission, bool)
        or not math.isfinite(float(interface_transmission))
        or not 0.0 < float(interface_transmission) <= 1.0
    ):
        raise ValueError("interface_transmission must lie in (0, 1]")

    data = prepared.to_mapping()
    settings = data["settings"]
    layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    unsupported(
        prepared,
        data,
        layers,
        optical_model=optical_model,
        allow_legacy_benchmark_deck=legacy_opt_in,
    )
    resolved_topology: InterfaceTopologyName = (
        interface_topology
        if interface_topology is not None
        else ("two_sided_trace" if settings["interface_charge_closure"] == "equilibrium_referenced" else "deduplicated")
    )
    if resolved_topology == "two_sided_trace" and len(layers) < 2:
        raise CapabilityError("unsupported_classical_1d_dc: two_sided_trace requires at least two electrical layers")
    if (
        interface_defect_distribution_mode != SINGLE_LEVEL_INTERFACE_DEFECT
        and resolved_topology != "two_sided_trace"
    ):
        raise ValueError(
            f"interface_defect_distribution_mode={interface_defect_distribution_mode!r} requires two_sided_trace topology"
        )
    if geometry_name is not None:
        if resolved_topology == "deduplicated" and geometry_name in (
            TWO_SIDED_TRACE_V1,
            PHYSICAL_BOUNDARY_TWO_SIDED_V1,
        ):
            raise ValueError("deduplicated topology requires a shared-node geometry")
        if resolved_topology == "two_sided_trace" and geometry_name in (
            INTEGER_ENDPOINT_V1,
            HALF_ENDPOINT_V1,
        ):
            raise ValueError("two_sided_trace topology requires a two-sided split geometry")

    grid = {entry["layer"]: entry for entry in data["electrical_grid"]}
    if isinstance(intervals_per_layer, bool):
        raise ValueError("intervals_per_layer must be an integer or integer sequence")
    if isinstance(intervals_per_layer, int):
        if intervals_per_layer < 2:
            raise ValueError("intervals_per_layer must be at least 2")
        weights = np.asarray([grid.get(layer.id, {}).get("interval_weight", 1.0) for layer in layers])
        counts = np.maximum(2, np.rint(intervals_per_layer * weights / np.mean(weights))).astype(int).tolist()
    else:
        if isinstance(intervals_per_layer, (str, bytes)):
            raise ValueError("intervals_per_layer must be an integer or integer sequence")
        counts = list(intervals_per_layer)
        if len(counts) != len(layers) or any(type(count) is not int or count < 2 for count in counts):
            raise ValueError("interval counts must be integers >= 2, one per electrical layer")
    for index, layer in enumerate(layers):
        p = dict(layer.parameters)
        if settings["band_grading"] and (p["Eg_back"] is not None or p["chi_back"] is not None):
            counts[index] *= p["grading_N_mult"]
    alphas = [grid.get(layer.id, {}).get("alpha", 3.0) for layer in layers]
    thicknesses = [layer.thickness for layer in layers]
    interface_positions: np.ndarray = np.cumsum(thicknesses, dtype=float)[:-1]
    base_nodes = multilayer_grid(list(zip(thicknesses, counts)), alpha=alphas)
    if resolved_topology == "two_sided_trace":
        nodes = remove_shared_interface_nodes(base_nodes, interface_positions)
        geom_token = TWO_SIDED_TRACE_V1 if geometry_name is None else geometry_name
        geometry = build_geometry(geom_token, nodes, interface_positions)
    else:
        nodes = base_nodes
        geom_token = INTEGER_ENDPOINT_V1 if geometry_name is None else geometry_name
        geometry = build_geometry(geom_token, nodes, interface_positions)
    masks = layer_node_masks(nodes, thicknesses, RIGHT_LAYER_WINS)
    dos_potentials_enabled = True if not legacy_opt_in else bool(settings["dos_band_potentials"])
    binding = compile_layer_parameters(
        nodes,
        [_layer_input(layer, allow_legacy_benchmark_deck=legacy_opt_in) for layer in layers],
        temperature_K=settings["T"],
        temperature_scaling=True,
        band_grading=settings["band_grading"],
        dos_band_potentials=dos_potentials_enabled,
        dual_ions=False,
    )

    def paint(name: str) -> np.ndarray:
        result = np.empty(nodes.size)
        for layer, mask in zip(layers, masks):
            result[mask] = dict(layer.parameters)[name]
        return result

    electron_diffusivity = legacy_face_diffusivity(binding.electron_diffusivity_m2_s)
    hole_diffusivity = legacy_face_diffusivity(binding.hole_diffusivity_m2_s)
    try:
        field_transport = bind_field_transport_1d(
            nodes, electron_diffusivity, hole_diffusivity, binding.thermal_voltage_V,
            v_sat_n=paint("v_sat_n"), v_sat_p=paint("v_sat_p"),
            ct_beta_n=paint("ct_beta_n"), ct_beta_p=paint("ct_beta_p"),
            pf_gamma_n=paint("pf_gamma_n"), pf_gamma_p=paint("pf_gamma_p"),
        )
    except FieldMobilityCapabilityError as error:
        raise CapabilityError("unsupported_classical_1d_dc: " + str(error)) from error

    # The configured intrinsic product must agree with the physical DOS/bands
    # whenever explicit semiconductor_work_function contacts are active.
    bands = binding.band_structure
    configured = binding.intrinsic_product_m6
    if np.any(configured <= 0.0) or not np.all(np.isfinite(configured)):
        raise CapabilityError("inconsistent_intrinsic_product: configured ni and physical Nc/Nv/Eg do not share one Maxwell-Boltzmann equilibrium")
    strict_swf = (
        not legacy_opt_in
        or (
            settings["built_in_potential_mode"] == "semiconductor_work_function"
            and dos_potentials_enabled
            and not settings["flat_band_metal_contacts"]
        )
    )
    if strict_swf:
        log_product = np.log(binding.conduction_dos_m3) + np.log(binding.valence_dos_m3) - bands.physical_band_gap_eV / binding.thermal_voltage_V
        if not np.all(np.isfinite(log_product)) or np.max(np.abs(np.log(configured) - log_product)) > 1.0e-10:
            raise CapabilityError("inconsistent_intrinsic_product: configured ni and physical Nc/Nv/Eg do not share one Maxwell-Boltzmann equilibrium")
    eps_r_node = paint("eps_r")
    if dielectric_rule is None and geometry.name != PHYSICAL_HALF_ENDPOINT_V2:
        poisson = PoissonOperator1D.for_geometry(geometry, eps_r_node)
    else:
        layer_eps_r = [float(cast(float, dict(layer.parameters)["eps_r"])) for layer in layers]
        poisson = PoissonOperator1D.for_geometry(
            geometry,
            eps_r_node,
            dielectric_rule=dielectric_rule,
            layer_eps_r=layer_eps_r,
        )
    scaps_defect_fallback_used = False
    if not legacy_opt_in:
        tau_n_nodes = paint("tau_n")
        tau_p_nodes = paint("tau_p")
        explicit_defects = bind_neutral_defects(
            layers, masks, ni_sq_m6=configured, band_gap_eV=bands.physical_band_gap_eV,
            thermal_voltage_V=binding.thermal_voltage_V,
            conduction_dos_m3=binding.conduction_dos_m3, valence_dos_m3=binding.valence_dos_m3,
        )
    else:
        # When SCAPS layers carry explicit bulk_defects with defect_model=='effective_lifetime',
        # resolve_device already folded 1/tau = sum(1/tau_k) into tau_n/tau_p; suppress that
        # combined lifetime on owned defect layers so _bind_legacy_scaps_neutral_defects
        # evaluates each named defect species in parallel without double-counting SRH.
        explicit_defects, scaps_defect_fallback_used = _bind_legacy_scaps_neutral_defects(
            layers, masks, ni_sq_m6=configured, band_gap_eV=bands.physical_band_gap_eV,
            thermal_voltage_V=binding.thermal_voltage_V,
            conduction_dos_m3=binding.conduction_dos_m3, valence_dos_m3=binding.valence_dos_m3,
        )
        tau_n_nodes = paint("tau_n")
        tau_p_nodes = paint("tau_p")
        if scaps_defect_fallback_used:
            defect_free_tau = float(cast(float, dict(prepared.defaults.scaps)["tau_n"]))
            for layer, mask in zip(layers, masks):
                if layer.defects and layer.defect_model == "effective_lifetime":
                    tau_n_nodes[mask] = defect_free_tau
                    tau_p_nodes[mask] = defect_free_tau
    coefficients = BulkRecombinationCoefficients(
        nodes.size, configured, tau_n_nodes, tau_p_nodes,
        binding.srh_n1_m3, binding.srh_p1_m3, binding.radiative_coefficient_m3_s,
        paint("C_n"), paint("C_p"),
    )
    recombination = BulkRecombinationReaction(coefficients, explicit_defects)
    generation_name = data["declaration"].get("fixed_generation")
    has_front_incoherent_substrate = (
        legacy_opt_in
        and bool(prepared.layers)
        and prepared.layers[0].role == "substrate"
        and bool(dict(prepared.layers[0].parameters)["incoherent"])
    )
    if generation_name is not None:
        table = prepared.resources.get(generation_name, "fixed_generation")
        if table.array.shape != (nodes.size, 2) or not np.array_equal(table.array[:, 0], nodes):
            raise CapabilityError("fixed_generation: resource coordinates must exactly match the compiled nodes; interpolation is not bound")
        generation = FixedGenerationProfile(table.array[:, 1], nodes.size).generation_profile(geometry)
    elif optical_model == "beer_lambert":
        generation = BeerLambertBinding(nodes, paint("alpha"), settings["Phi"]).generation_profile(geometry)
    elif optical_model == "beer_lambert_spectral":
        generation = bind_spectral_beer_lambert_1d(prepared, geometry)
    elif optical_model == "tmm":
        generation = (
            _bind_legacy_front_incoherent_tmm_1d(prepared, geometry)
            if legacy_opt_in
            else bind_optical_generation_1d(prepared, geometry).profile
        )
    elif data["declaration"].get("spectrum") is not None:
        generation = (
            _bind_legacy_front_incoherent_tmm_1d(prepared, geometry)
            if legacy_opt_in
            else bind_optical_generation_1d(prepared, geometry).profile
        )
    else:
        generation = BeerLambertBinding(nodes, paint("alpha"), settings["Phi"]).generation_profile(geometry)

    def contact_material(node: int) -> ContactMaterial:
        return ContactMaterial(
            binding.temperature_K,
            float(bands.physical_electron_affinity_eV[node]),
            float(bands.physical_band_gap_eV[node]),
            float(binding.conduction_dos_m3[node]), float(binding.valence_dos_m3[node]),
            float(binding.donor_density_m3[node]), float(binding.acceptor_density_m3[node]),
        )

    phi_left = float(settings["phi_left"])
    reservoirs = [doping_equilibrium_densities(float(binding.donor_density_m3[node]), float(binding.acceptor_density_m3[node]), float(configured[node])) for node in (0, -1)]
    legacy_deck_active = False
    if strict_swf:
        left_material, right_material = contact_material(0), contact_material(-1)
        work_left = semiconductor_work_function(left_material)
        built_in = work_left - semiconductor_work_function(right_material)
        reference_phi = phi_left + built_in * (nodes - nodes[0]) / (nodes[-1] - nodes[0])
        fermi = -work_left - phi_left
        reference_n = binding.conduction_dos_m3 * np.exp((fermi + reference_phi + bands.physical_electron_affinity_eV) / binding.thermal_voltage_V)
        reference_p = binding.valence_dos_m3 * np.exp(-(fermi + reference_phi + bands.physical_electron_affinity_eV + bands.physical_band_gap_eV) / binding.thermal_voltage_V)
        for node, reservoir in zip((0, -1), reservoirs):
            reference_n[node], reference_p[node] = reservoir
        if scaps_defect_fallback_used or has_front_incoherent_substrate or float(settings["het_recomb_despike"]) != 0.0:
            legacy_deck_active = True
    else:
        legacy_deck_active = True
        if settings["flat_band_metal_contacts"]:
            barrier_eV = float(settings["contact_phi_B_eV"])
            for side_idx, node in enumerate((0, -1)):
                nc_val = float(binding.conduction_dos_m3[node])
                nv_val = float(binding.valence_dos_m3[node])
                reservoirs[side_idx] = metal_reservoir_floor(
                    reservoirs[side_idx][0],
                    reservoirs[side_idx][1],
                    intrinsic_density_squared_m6=float(configured[node]),
                    donor_density_m3=float(binding.donor_density_m3[node]),
                    acceptor_density_m3=float(binding.acceptor_density_m3[node]),
                    conduction_dos_m3=nc_val if math.isfinite(nc_val) and nc_val > 0.0 else None,
                    valence_dos_m3=nv_val if math.isfinite(nv_val) and nv_val > 0.0 else None,
                    barrier_eV=barrier_eV,
                    thermal_voltage_V=binding.thermal_voltage_V,
                )
        left_band = LegacyBandContact(
            float(bands.physical_electron_affinity_eV[0]),
            float(bands.physical_band_gap_eV[0]),
            float(binding.donor_density_m3[0]),
            float(binding.acceptor_density_m3[0]),
            math.sqrt(float(configured[0])),
        )
        right_band = LegacyBandContact(
            float(bands.physical_electron_affinity_eV[-1]),
            float(bands.physical_band_gap_eV[-1]),
            float(binding.donor_density_m3[-1]),
            float(binding.acceptor_density_m3[-1]),
            math.sqrt(float(configured[-1])),
        )
        elec_bands = tuple(
            (float(cast(float, dict(layer.parameters)["chi"])), float(cast(float, dict(layer.parameters)["Eg"])))
            for layer in layers
        )
        has_positive_contact_bands = all(
            math.isfinite(v) and v > 0.0
            for node in (0, -1)
            for v in (
                float(bands.physical_electron_affinity_eV[node]),
                float(bands.physical_band_gap_eV[node]),
                float(binding.conduction_dos_m3[node]),
                float(binding.valence_dos_m3[node]),
            )
        )
        sources = BuiltInPotentialSources(
            mode=settings["built_in_potential_mode"],
            manual_V_bi=float(settings["V_bi"]),
            flat_band_contacts=bool(settings["flat_band_contacts"]),
            left_band=left_band,
            right_band=right_band,
            electrical_layer_bands=elec_bands,
            left_material=contact_material(0) if has_positive_contact_bands else None,
            right_material=contact_material(-1) if has_positive_contact_bands else None,
            work_function_left_eV=settings["work_function_left_eV"],
            work_function_right_eV=settings["work_function_right_eV"],
        )
        built_in = sources.poisson_potential()
        vt = binding.thermal_voltage_V
        chi_t = bands.transport_electron_affinity_eV
        eg_t = bands.transport_band_gap_eV
        n_left_res, p_left_res = reservoirs[0]
        n_right_res, p_right_res = reservoirs[1]
        delta_phi_ref = vt * math.log(n_right_res / n_left_res) - float(chi_t[-1] - chi_t[0])
        reference_phi = phi_left + delta_phi_ref * (nodes - nodes[0]) / (nodes[-1] - nodes[0])
        fermi = -float(left_band.electron_affinity_eV + 0.5 * left_band.band_gap_eV) - phi_left
        d_psi_n = (reference_phi - reference_phi[0]) + (chi_t - chi_t[0])
        d_psi_p = (reference_phi - reference_phi[0]) + ((chi_t + eg_t) - (chi_t[0] + eg_t[0]))
        reference_n = n_left_res * np.exp(d_psi_n / vt)
        reference_p = p_left_res * np.exp(-d_psi_p / vt)
        reference_n[0], reference_p[0] = n_left_res, p_left_res
        reference_n[-1] = n_right_res
        if math.isclose(float(reference_p[-1]), p_right_res, rel_tol=1.0e-12, abs_tol=0.0):
            reference_p[-1] = p_right_res
    contacts = {entry["side"]: entry for entry in data["contacts"]}
    velocities = resolve_exchange_velocities(
        tuple(contacts[side][name] for side in ("left", "right") for name in ("S_n", "S_p")),
        flat_band_contacts=settings["flat_band_contacts"], selective_contacts_enabled=True,
    )
    channels = tuple(
        CarrierExchangeChannel(carrier, side, velocity, reservoirs[side_index][carrier_index])
        for velocity, (side_index, side, carrier_index, carrier) in zip(
            velocities, ((0, "left", 0, "n"), (0, "left", 1, "p"), (1, "right", 0, "n"), (1, "right", 1, "p")),
        )
    )
    if all(channel.kind == "blocking" for channel in channels):
        raise CapabilityError("fully_blocking_contacts: a closed carrier inventory constraint is not bound in this DC slice")

    interface_model = _bind_interfaces_1d(
        device_id=data["id"],
        prepared=prepared,
        settings=settings,
        layers=layers,
        geometry=geometry,
        binding=binding,
        poisson=poisson,
        electron_diffusivity=electron_diffusivity,
        hole_diffusivity=hole_diffusivity,
        generation=generation,
        recombination=recombination,
        channels=channels,
        eps_r_node=eps_r_node,
        reference_phi=reference_phi,
        reference_n=reference_n,
        reference_p=reference_p,
        phi_left_V=phi_left,
        built_in_V=built_in,
        fermi_eV=fermi,
        additional_recombination=additional_recombination,
        field_transport=field_transport,
        topology=resolved_topology,
        te_softness=float(te_softness),
        te_regularization_mode=te_regularization_mode,
        interface_transmission=float(interface_transmission),
        equilibrium_occupancies=equilibrium_occupancies,
        thermionic_mode=resolved_thermionic,
        combination_mode=combination_mode,
        band_step_mode=band_step_mode,
        interface_band_reference_mode=interface_band_reference_mode,
        interface_defect_distribution_mode=interface_defect_distribution_mode,
    )

    extra_id = None if additional_recombination is None else additional_recombination.model_id
    identity_parts: tuple[object, ...] = (
        "electrode_quasi_fermi_1d_v1", prepared.content_sha256, binding.sha256,
        generation.binding_sha256, geometry.name, counts, extra_id,
    )
    if explicit_defects is not None:
        identity_parts += (explicit_defects.identity_sha256,)
        owned = explicit_defects.owned_nodes
        if np.any(binding.conduction_dos_m3[owned] != binding.valence_dos_m3[owned]):
            identity_parts += (LITERAL_ABOVE_VB_DOS_REFERENCE_V1,)
    if field_transport is not None:
        identity_parts += (field_transport.identity_sha256,)
    if interface_model is not None:
        identity_parts += (interface_model.identity_sha256,)
    if dielectric_rule is not None:
        identity_parts += (("dielectric_rule", poisson.dielectric_rule),)
    if band_step_mode != LEGACY_SUMMED_POTENTIAL_V1:
        identity_parts += (("band_step_mode", band_step_mode),)
    if resolved_thermionic != LEGACY_UNNORMALIZED_THERMIONIC_V1 or combination_mode is not None:
        identity_parts += (
            (
                "thermionic_mode",
                resolved_thermionic,
                "harmonic_series" if combination_mode is None else combination_mode,
            ),
        )
    if legacy_deck_active:
        identity_parts += (
            (
                "legacy_benchmark_deck_v1",
                settings["built_in_potential_mode"],
                built_in,
                bool(settings["flat_band_metal_contacts"]),
                float(settings["contact_phi_B_eV"]),
                float(settings["het_recomb_despike"]),
                scaps_defect_fallback_used,
                has_front_incoherent_substrate,
            ),
        )
    identity = hashlib.sha256(json.dumps(
        identity_parts,
        separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    contact_capability = (
        "semiconductor_work_function_contacts"
        if not legacy_deck_active and settings["built_in_potential_mode"] == "semiconductor_work_function"
        else "legacy_benchmark_deck"
    )
    capabilities: tuple[str, ...] = (
        "ion_free",
        "classical_1d_dc",
        "maxwell_boltzmann",
        contact_capability,
        "bulk_srh_radiative_auger",
    )
    if interface_model is not None:
        capabilities = capabilities + ("heterointerface_1d",)
    return CompiledDevice1D(
        data["id"], identity, geometry, binding, poisson,
        electron_diffusivity, hole_diffusivity,
        generation, recombination, channels, phi_left, built_in, fermi,
        reference_phi, reference_n, reference_p, additional_recombination,
        capabilities=capabilities,
        field_transport=field_transport,
        interface_model=interface_model,
        band_step_mode=band_step_mode,
    )


def p08_identity_inputs(
    prepared: PreparedDevice,
    implementation: BuildSource,
    *,
    options: P08CompilerOptions | None = None,
    problem: str = "dc",
) -> IdentityInputs:
    """Bind a PreparedDevice and P08 model selection into source-bound IdentityInputs.

    Every P08 V2 option deterministically updates the selected ``ModelRef``
    version (``1 -> 2``) and its immutable source evidence, ensuring ``H_physics``
    (``physics_identity(inputs).sha256``) changes whenever any V2 model is active.
    """
    if type(prepared) is not PreparedDevice:
        raise ValueError("an actual explicit PreparedDevice is required")
    resolved_options = P08CompilerOptions() if options is None else options
    by_id = {doc.id: doc for doc in implementation.files}
    missing = [path for path in P08_SOURCE_PATHS if path not in by_id]
    if missing:
        raise ValueError("BuildSource is missing P08 source files: " + ", ".join(missing))
    sources = {path: by_id[path] for path in P08_SOURCE_PATHS}
    registry = build_p08_model_registry(sources)
    preset = build_p08_preset(resolved_options, sources)
    topology = device_topology(prepared)
    param_sources = {
        Scope("layer", (layer.id,)): layer_parameter_source(layer)
        for layer in prepared.layers
        if layer.id in topology.electrical_layer_ids
    }
    context = CapabilityContext(
        1,
        problem,
        "solarlab_assembled",
        "maxwell_boltzmann",
        temperature_K=prepared.to_mapping()["settings"]["T"],
    )
    selected = select_preset(preset, registry, topology, context, param_sources)
    projections = tuple(
        SemanticProjection(
            ref,
            "all_values_v1",
            (
                sources[
                    mapping.v1_source_path if ref == mapping.v1_model else mapping.v2_source_path
                ],
            ),
        )
        for mapping, ref in zip(P08_MODEL_VERSION_MAPPINGS, resolved_options.active_model_refs())
    )
    return IdentityInputs(
        prepared,
        selected.selection,
        registry,
        ModelInputs.from_mappings(defaults=dict(preset.defaults), overrides={}),
        projections,
    )
