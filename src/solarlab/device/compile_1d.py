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

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any, Literal, TypedDict, cast

import numpy as np

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.device.resolved import PreparedDevice, PreparedLayer
from solarlab.device.neutral_defects import LITERAL_ABOVE_VB_DOS_REFERENCE_V1, bind_neutral_defects
from solarlab.device.optical_generation import bind_optical_generation_1d
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
    INTEGER_ENDPOINT_V1,
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
    constant_index_optical_constants,
)
from solarlab.physics.generation.profile import ControlVolumes
from solarlab.physics.generation.spectrum import SpectralPhotonFlux, sample_nk_table
from solarlab.physics.interface_transport import (
    EquilibriumReferencedSheetCharge,
    TE_BARRIER_THRESHOLD_EV,
    TwoSidedBulkState,
    TwoSidedInterfacePhysics,
    remove_shared_interface_nodes,
    resolve_layer_richardson,
    solve_two_sided_interface_qss,
)
from solarlab.physics.recombination.interface import (
    InterfaceDefectSpec,
    build_plane_closure_params,
    resolve_interface_defect_references,
)
from solarlab.physics.recombination.reaction import (
    BulkRecombinationCoefficients,
    BulkRecombinationReaction,
)
from solarlab.physics.transport import legacy_face_diffusivity
from solarlab.physics.transport.field_mobility_tangents import FieldMobilityCapabilityError

__all__ = ["CapabilityError", "CompiledDevice1D", "compile_device_1d", "unsupported"]

Q = ELEMENTARY_CHARGE_C
InterfaceTopologyName = Literal["deduplicated", "two_sided_trace"]
OpticalModelName = Literal["beer_lambert", "beer_lambert_spectral", "tmm"]
_SPECTRAL_BL_BINDING_1D_V1 = "prepared_spectral_beer_lambert_1d_v1"
_SPECTRAL_BL_BUDGET_RTOL = 1e-12


class _LayerScalars(TypedDict):
    """Resolved scalar fields consumed by this supported DC binding."""

    chi: float
    Eg: float
    Nc300: float
    Nv300: float
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
) -> None:
    settings = data["settings"]
    missing = []
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
    if settings["interface_plane_generation"] and not settings["interface_plane_closure"]:
        missing.append("settings.interface_plane_generation: requires settings.interface_plane_closure")
    if settings["interface_plane_projection"] and settings["interface_plane_closure"]:
        missing.append("settings.interface_plane_projection and settings.interface_plane_closure are mutually exclusive")
    if settings["het_recomb_despike"] != 0.0:
        missing.append("settings.het_recomb_despike")
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
        if any(parameters[name] is not None for name in ("trap_N_t_interface", "trap_N_t_bulk", "trap_decay_length")):
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
            a_star_n = resolve_layer_richardson(
                float(cast(float, left_p["A_star_n"])),
                float(cast(float, left_p["Nc300"])),
                temp,
            )
            a_star_p = resolve_layer_richardson(
                float(cast(float, left_p["A_star_p"])),
                float(cast(float, left_p["Nv300"])),
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

        manifest = {
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
        a_n_l = resolve_layer_richardson(float(cast(float, left_p["A_star_n"])), float(cast(float, left_p["Nc300"])), temp)
        a_n_r = resolve_layer_richardson(float(cast(float, right_p["A_star_n"])), float(cast(float, right_p["Nc300"])), temp)
        a_p_l = resolve_layer_richardson(float(cast(float, left_p["A_star_p"])), float(cast(float, left_p["Nv300"])), temp)
        a_p_r = resolve_layer_richardson(float(cast(float, right_p["A_star_p"])), float(cast(float, right_p["Nv300"])), temp)
        rich_n = min(a_n_l, a_n_r)
        rich_p = min(a_p_l, a_p_r)
        if rich_n <= 0.0 or rich_p <= 0.0:
            raise CapabilityError(
                f"interfaces.{entry['id']}: two_sided_trace requires positive A_star_n and A_star_p on both adjacent layers"
            )

        defect = entry.get("defect")
        raw_vn = entry.get("v_n")
        raw_vp = entry.get("v_p")
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
            )
            n1_l, p1_l = refs.n1_left_m3, refs.p1_left_m3
            n1_r, p1_r = refs.n1_right_m3, refs.p1_right_m3
            trap_density_m2 = d_spec.areal_density_m2

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

    manifest_ts = {
        "schema": "compiled_interface_model_1d_v1",
        "topology": "two_sided_trace",
        "charge_closure": charge_closure,
        "two_sided": manifest_two_sided,
    }
    iface_sha = hashlib.sha256(
        json.dumps(manifest_ts, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return CompiledInterfaceModel1D(
        topology="two_sided_trace",
        identity_sha256=iface_sha,
        two_sided=tuple(two_sided_specs),
    )


def _layer_input(layer: PreparedLayer) -> LayerParameterInput:
    parameters = dict(layer.parameters)
    for name in ("chi", "Eg", "Nc300", "Nv300", "ni", "eps_r", "mu_n", "mu_p"):
        value = parameters[name]
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise CapabilityError(f"layers.{layer.id}.{name}: classical 1D DC requires a finite positive value")
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


def _bind_spectral_beer_lambert_1d(
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
) -> CompiledDevice1D:
    """Cold compile genuine resolved configuration; no legacy/test execution.

    A scalar interval count is the mean per-layer budget, allocated by the
    prepared electrical_grid weights. A sequence explicitly supplies each
    electrical layer's interval count. Prepared grid alpha values are always
    used; without declarations the existing tanh-grid alpha is 3.
    """
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("compile_device_1d requires an existing PreparedDevice")
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
    unsupported(prepared, data, layers, optical_model=optical_model)
    resolved_topology: InterfaceTopologyName = (
        interface_topology
        if interface_topology is not None
        else ("two_sided_trace" if settings["interface_charge_closure"] == "equilibrium_referenced" else "deduplicated")
    )
    if resolved_topology == "two_sided_trace" and len(layers) < 2:
        raise CapabilityError("unsupported_classical_1d_dc: two_sided_trace requires at least two electrical layers")

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
        geometry = build_geometry(TWO_SIDED_TRACE_V1, nodes, interface_positions)
    else:
        nodes = base_nodes
        geometry = build_geometry(INTEGER_ENDPOINT_V1, nodes, interface_positions)
    masks = layer_node_masks(nodes, thicknesses, RIGHT_LAYER_WINS)
    binding = compile_layer_parameters(
        nodes, [_layer_input(layer) for layer in layers],
        temperature_K=settings["T"], temperature_scaling=True,
        band_grading=settings["band_grading"], dos_band_potentials=True, dual_ions=False,
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

    # The configured intrinsic product must agree with the physical DOS/bands.
    # Changing it silently would replace the user's recombination population.
    bands = binding.band_structure
    log_product = np.log(binding.conduction_dos_m3) + np.log(binding.valence_dos_m3) - bands.physical_band_gap_eV / binding.thermal_voltage_V
    configured = binding.intrinsic_product_m6
    if np.any(configured <= 0.0) or not np.all(np.isfinite(configured)) or np.max(np.abs(np.log(configured) - log_product)) > 1.0e-10:
        raise CapabilityError("inconsistent_intrinsic_product: configured ni and physical Nc/Nv/Eg do not share one Maxwell-Boltzmann equilibrium")
    eps_r_node = paint("eps_r")
    poisson = PoissonOperator1D.for_geometry(geometry, eps_r_node)
    coefficients = BulkRecombinationCoefficients(
        nodes.size, configured, paint("tau_n"), paint("tau_p"),
        binding.srh_n1_m3, binding.srh_p1_m3, binding.radiative_coefficient_m3_s,
        paint("C_n"), paint("C_p"),
    )
    explicit_defects = bind_neutral_defects(
        layers, masks, ni_sq_m6=configured, band_gap_eV=bands.physical_band_gap_eV,
        thermal_voltage_V=binding.thermal_voltage_V,
        conduction_dos_m3=binding.conduction_dos_m3, valence_dos_m3=binding.valence_dos_m3,
    )
    recombination = BulkRecombinationReaction(coefficients, explicit_defects)
    generation_name = data["declaration"].get("fixed_generation")
    if generation_name is not None:
        table = prepared.resources.get(generation_name, "fixed_generation")
        if table.array.shape != (nodes.size, 2) or not np.array_equal(table.array[:, 0], nodes):
            raise CapabilityError("fixed_generation: resource coordinates must exactly match the compiled nodes; interpolation is not bound")
        generation = FixedGenerationProfile(table.array[:, 1], nodes.size).generation_profile(geometry)
    elif optical_model == "beer_lambert":
        generation = BeerLambertBinding(nodes, paint("alpha"), settings["Phi"]).generation_profile(geometry)
    elif optical_model == "beer_lambert_spectral":
        generation = _bind_spectral_beer_lambert_1d(prepared, geometry)
    elif optical_model == "tmm":
        generation = bind_optical_generation_1d(prepared, geometry).profile
    elif data["declaration"].get("spectrum") is not None:
        generation = bind_optical_generation_1d(prepared, geometry).profile
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

    left_material, right_material = contact_material(0), contact_material(-1)
    work_left = semiconductor_work_function(left_material)
    built_in = work_left - semiconductor_work_function(right_material)
    phi_left = float(settings["phi_left"])
    reference_phi = phi_left + built_in * (nodes - nodes[0]) / (nodes[-1] - nodes[0])
    fermi = -work_left - phi_left
    reference_n = binding.conduction_dos_m3 * np.exp((fermi + reference_phi + bands.physical_electron_affinity_eV) / binding.thermal_voltage_V)
    reference_p = binding.valence_dos_m3 * np.exp(-(fermi + reference_phi + bands.physical_electron_affinity_eV + bands.physical_band_gap_eV) / binding.thermal_voltage_V)
    reservoirs = [doping_equilibrium_densities(float(binding.donor_density_m3[node]), float(binding.acceptor_density_m3[node]), float(configured[node])) for node in (0, -1)]
    for node, reservoir in zip((0, -1), reservoirs):
        reference_n[node], reference_p[node] = reservoir
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
    identity = hashlib.sha256(json.dumps(
        identity_parts,
        separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    capabilities: tuple[str, ...] = (
        "ion_free",
        "classical_1d_dc",
        "maxwell_boltzmann",
        "semiconductor_work_function_contacts",
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
    )
