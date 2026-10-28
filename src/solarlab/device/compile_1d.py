"""Bind an existing PreparedDevice to the first production new-core 1D DC slice.

Supported physics is explicit: Maxwell-Boltzmann carriers, fully ionized
dopants, static temperature/doping/band grading, DOS-aware SG transport,
semiconductor work-function contacts, local lifetime SRH/radiative/Auger,
and monochromatic Beer-Lambert or a node-aligned fixed generation resource.
Historical mode labels do not select a legacy engine or enable its tier ceiling.
Unsupported active mechanisms fail before any nonlinear solve.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from typing import Any

import numpy as np

from solarlab.device.resolved import PreparedDevice, PreparedLayer
from solarlab.equations.dc_1d import (
    CapabilityError,
    CompiledDevice1D,
    LocalRecombinationModel,
)
from solarlab.mesh import INTEGER_ENDPOINT_V1, RIGHT_LAYER_WINS, build_geometry, multilayer_grid
from solarlab.mesh.grid import layer_node_masks
from solarlab.physics.band_structure.binding import (
    LayerParameterInput,
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
from solarlab.physics.generation.beer_lambert import BeerLambertBinding
from solarlab.physics.generation.fixed import FixedGenerationProfile
from solarlab.physics.recombination.reaction import (
    BulkRecombinationCoefficients,
    BulkRecombinationReaction,
)
from solarlab.physics.transport import legacy_face_diffusivity

__all__ = ["CapabilityError", "CompiledDevice1D", "compile_device_1d"]


def _unsupported(prepared: PreparedDevice, data: dict[str, Any], layers: tuple[PreparedLayer, ...]) -> None:
    settings = data["settings"]
    missing = []
    if settings["built_in_potential_mode"] != "semiconductor_work_function":
        missing.append("settings.built_in_potential_mode: only explicit semiconductor_work_function contacts are bound")
    if not settings["dos_band_potentials"]:
        missing.append("settings.dos_band_potentials: physical DOS transport requires this enabled")
    for flag in (
        "interface_plane_projection", "te_physical_norm", "flat_band_metal_contacts",
        "interface_two_sided", "interface_shared_occupancy", "interface_plane_closure",
        "interface_plane_generation", "graded_optics", "interface_tunneling",
        "ion_steric_diffusion_only", "ion_steric_shared_site",
    ):
        if settings[flag]:
            missing.append("settings." + flag)
    if settings["het_recomb_despike"] != 0.0:
        missing.append("settings.het_recomb_despike")
    if settings["interface_charge_closure"] != "off":
        missing.append("settings.interface_charge_closure")
    if settings["jv_solver_policy"] != "general":
        missing.append("settings.jv_solver_policy: a separately qualified solver route is required")
    if data["grain_boundaries"]:
        missing.append("grain_boundaries: lateral geometry is not bound")
    channels = data.get("tunnelling_channels")
    if channels is not None:
        missing.extend("tunnelling_channels." + name for name, channel in channels.items() if channel.get("enabled"))
    declaration = data["declaration"]
    if declaration.get("spectrum") is not None:
        missing.append("spectrum: spectral/TMM generation is not bound")
    for layer in prepared.layers:
        if layer.role == "substrate":
            parameters = dict(layer.parameters)
            if parameters["alpha"] != 0.0 or layer.optical is not None or parameters["n_optical"] is not None:
                missing.append("layers." + layer.id + ": absorbing/optical substrate generation is not bound")
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
        if any(parameters[name] != 0.0 for name in ("v_sat_n", "v_sat_p", "pf_gamma_n", "pf_gamma_p")):
            missing.append(prefix + "field_mobility")
        if layer.defects or layer.defect_model != "effective_lifetime" or layer.bulk_trap_distribution_json is not None:
            missing.append(prefix + "explicit_or_distributed_defects")
        if layer.metastable_document_json is not None:
            missing.append(prefix + "metastable_document")
        if any(parameters[name] is not None for name in ("trap_N_t_interface", "trap_N_t_bulk", "trap_decay_length")):
            missing.append(prefix + "trap_profile")
        if layer.optical is not None or parameters["n_optical"] is not None or layer.cigs_graded_optics_json is not None:
            missing.append(prefix + "optical_material: TMM/graded optics is not bound")
    for interface in prepared.interfaces:
        entry = interface.to_mapping()
        if interface.electrical and (
            entry.get("defect") is not None
            or any((entry.get(name) or 0.0) != 0.0 for name in ("v_n", "v_p"))
        ):
            missing.append("interfaces." + entry["id"] + ": interface recombination is not bound")
    if missing:
        raise CapabilityError("unsupported_classical_1d_dc: " + "; ".join(missing))


def _layer_input(layer: PreparedLayer) -> LayerParameterInput:
    p = dict(layer.parameters)
    for name in ("chi", "Eg", "Nc300", "Nv300", "ni", "eps_r", "mu_n", "mu_p"):
        if p[name] is None or not math.isfinite(float(p[name])) or float(p[name]) <= 0.0:
            raise CapabilityError(f"layers.{layer.id}.{name}: classical 1D DC requires a finite positive value")
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


def compile_device_1d(
    prepared: PreparedDevice,
    *,
    intervals_per_layer: int | Sequence[int] = 24,
    additional_recombination: LocalRecombinationModel | None = None,
) -> CompiledDevice1D:
    """Cold compile genuine resolved configuration; no legacy/test execution.

    A scalar interval count is the mean per-layer budget, allocated by the
    prepared electrical_grid weights. A sequence explicitly supplies each
    electrical layer's interval count. Prepared grid alpha values are always
    used; without declarations the existing tanh-grid alpha is 3.
    """
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("compile_device_1d requires an existing PreparedDevice")
    data = prepared.to_mapping()
    settings = data["settings"]
    layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    _unsupported(prepared, data, layers)
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
    nodes = multilayer_grid(list(zip(thicknesses, counts)), alpha=alphas)
    masks = layer_node_masks(nodes, thicknesses, RIGHT_LAYER_WINS)
    geometry = build_geometry(INTEGER_ENDPOINT_V1, nodes, np.cumsum(thicknesses)[:-1])
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

    # The configured intrinsic product must agree with the physical DOS/bands.
    # Changing it silently would replace the user's recombination population.
    bands = binding.band_structure
    log_product = np.log(binding.conduction_dos_m3) + np.log(binding.valence_dos_m3) - bands.physical_band_gap_eV / binding.thermal_voltage_V
    configured = binding.intrinsic_product_m6
    if np.any(configured <= 0.0) or not np.all(np.isfinite(configured)) or np.max(np.abs(np.log(configured) - log_product)) > 1.0e-10:
        raise CapabilityError("inconsistent_intrinsic_product: configured ni and physical Nc/Nv/Eg do not share one Maxwell-Boltzmann equilibrium")
    poisson = PoissonOperator1D.for_geometry(geometry, paint("eps_r"))
    coefficients = BulkRecombinationCoefficients(
        nodes.size, configured, paint("tau_n"), paint("tau_p"),
        binding.srh_n1_m3, binding.srh_p1_m3, binding.radiative_coefficient_m3_s,
        paint("C_n"), paint("C_p"),
    )
    recombination = BulkRecombinationReaction(coefficients)
    generation_name = data["declaration"].get("fixed_generation")
    if generation_name is not None:
        table = prepared.resources.get(generation_name, "fixed_generation")
        if table.array.shape != (nodes.size, 2) or not np.array_equal(table.array[:, 0], nodes):
            raise CapabilityError("fixed_generation: resource coordinates must exactly match the compiled nodes; interpolation is not bound")
        generation = FixedGenerationProfile(table.array[:, 1], nodes.size).generation_profile(geometry)
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
    reservoirs = [doping_equilibrium_densities(binding.donor_density_m3[node], binding.acceptor_density_m3[node], configured[node]) for node in (0, -1)]
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
    extra_id = None if additional_recombination is None else additional_recombination.model_id
    identity = hashlib.sha256(json.dumps(
        ("electrode_quasi_fermi_1d_v1", prepared.content_sha256, binding.sha256, generation.binding_sha256, geometry.name, counts, extra_id),
        separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    return CompiledDevice1D(
        data["id"], identity, geometry, binding, poisson,
        legacy_face_diffusivity(binding.electron_diffusivity_m2_s),
        legacy_face_diffusivity(binding.hole_diffusivity_m2_s),
        generation, recombination, channels, phi_left, built_in, fermi,
        reference_phi, reference_n, reference_p, additional_recombination,
    )
