"""Cold-compile a ``PreparedDevice`` to 1D nonlocal radiative reabsorption and WKB tunnelling.

Preserves strict separation from local-only ``compile_device_1d``:
* Disabled nonlocal configurations delegate directly to ``compile_device_1d`` and
  reproduce ``CompiledDevice1D`` bit for bit.
* Active photon recycling binds band-edge escape probabilities from supplied
  coherent spectral TMM optics (or explicit override probabilities) and builds
  rank-1 ``AbsorberReabsorptionSpec`` operators (or local ``B_rad * P_esc``
  scaling when ``photon_recycling_mode='local_escape_scaled'``).
* Active WKB tunnelling validates ``TunnellingChannelDocument`` pre-solve,
  requires ``driver='quasi_fermi_steady_state'``, and binds the single resolved
  material electron barrier to ``CompiledTunnellingChannels``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any, Literal, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import diags

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.neutral_defects import (
    LITERAL_ABOVE_VB_DOS_REFERENCE_V1,
    bind_neutral_defects,
)
from solarlab.device.optical_generation import OpticalGeneration1D, bind_optical_generation_1d
from solarlab.device.resolved import PreparedDevice, PreparedLayer
from solarlab.equations.assembly.nonlocal_1d import (
    AssembledNonlocalDC1DProblem,
    assembled_nonlocal_problem_factory,
)
from solarlab.equations.dc_1d import (
    CapabilityError,
    CompiledDevice1D,
    LocalRecombinationModel,
    State1D,
)
from solarlab.equations.field_transport_1d import bind_field_transport_1d
from solarlab.equations.implementation import use_assembled
from solarlab.equations.nonlocal_1d import (
    CompiledNonlocalDevice1D,
    NonlocalDC1DProblem,
    SparsitySupportChangedError,
    evaluate_nonlocal_couplings_1d,
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
    ContactEndpoint,
    ContactMaterial,
    assess_contact_thermodynamics,
    doping_equilibrium_densities,
    semiconductor_work_function,
)
from solarlab.physics.contacts.exchange import resolve_exchange_velocities
from solarlab.physics.electrostatics import PoissonOperator1D
from solarlab.physics.generation.beer_lambert import BeerLambertBinding
from solarlab.physics.generation.fixed import FixedGenerationProfile
from solarlab.physics.generation.photon_recycling import (
    ESCAPE_PROBABILITY_MODES,
    TMM_INTEGRATED_ABSORBANCE_CONTROL_VOLUME_V1,
    YABLONOVITCH_INTRINSIC_ALPHA_D_V1,
    AbsorberEscapeBinding,
    AbsorberEscapeInput,
    PhotonRecyclingCapabilityError,
    PhotonRecyclingError,
    bind_absorber_escape_from_optical_stack,
    effective_radiative_coefficient,
    wavelength_at_gap,
)
from solarlab.physics.generation.radiative_reabsorption import (
    CONTROL_VOLUME_CONSERVATIVE_V1,
    HISTORICAL_TRAPEZOID_UNIFORM_V1,
    LAGGED_BAKED_GENERATION_V1,
    AbsorberReabsorptionSpec,
)
from solarlab.physics.recombination.reaction import (
    BulkRecombinationCoefficients,
    BulkRecombinationReaction,
)
from solarlab.physics.transport import legacy_face_diffusivity
from solarlab.physics.transport.field_mobility_tangents import FieldMobilityCapabilityError
from solarlab.physics.transport.tunneling_channels import (
    PREALLOCATED_RESERVOIR_SUPERSET_V1,
    STRICT_WINDOW_REBIND_V1,
    WKB_SPARSITY_POLICIES,
    CompiledTunnellingChannels,
    TunnellingChannelCapabilityError,
    TunnellingChannelError,
    TunnellingChannelSchemaError,
    compile_tunnelling_channels,
    tunnelling_channel_document_from_mapping,
)
from solarlab.solvers.nonlinear.newton import (
    NewtonResult,
    NewtonSettings,
    factorize,
    gate_from,
    solve_newton,
)
from solarlab.solvers.nonlinear.problem import (
    CallableProblem,
    GateOutcome,
    PhysicalGate,
    ResidualScaling,
    StepControl,
)
from solarlab.solvers.nonlinear.production import (
    POISSON_CONTACT_BALANCE_V1,
    poisson_row_weights,
    production_convergence_norm,
    production_coordinates,
    production_residual_scaling,
)
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    SteadyState1D,
    SteadyStateError,
)

__all__ = [
    "PHOTON_RECYCLING_LOCAL_ESCAPE_SCALED",
    "PHOTON_RECYCLING_MODES",
    "PHOTON_RECYCLING_OFF",
    "CompiledNonlocalDevice1D",
    "compile_nonlocal_device_1d",
    "nonlocal_physical_gates",
    "solve_nonlocal_equilibrium_1d",
    "solve_nonlocal_steady_state_1d",
]

Q: float = ELEMENTARY_CHARGE_C

PHOTON_RECYCLING_OFF = "off"
PHOTON_RECYCLING_LOCAL_ESCAPE_SCALED = "local_escape_scaled"
PHOTON_RECYCLING_MODES: tuple[str, ...] = (
    PHOTON_RECYCLING_OFF,
    PHOTON_RECYCLING_LOCAL_ESCAPE_SCALED,
    CONTROL_VOLUME_CONSERVATIVE_V1,
    HISTORICAL_TRAPEZOID_UNIFORM_V1,
    LAGGED_BAKED_GENERATION_V1,
)

NonlocalProblem = AssembledNonlocalDC1DProblem | NonlocalDC1DProblem


class _LayerScalars(TypedDict):
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


def _has_uniform_dos(layers: tuple[PreparedLayer, ...]) -> bool:
    if not layers:
        return True
    nc_vals = [float(cast(float, dict(layer.parameters)["Nc300"])) for layer in layers]
    nv_vals = [float(cast(float, dict(layer.parameters)["Nv300"])) for layer in layers]
    return len(set(nc_vals)) == 1 and len(set(nv_vals)) == 1


def _validate_nonlocal_capabilities(
    prepared: PreparedDevice,
    data: dict[str, Any],
    layers: tuple[PreparedLayer, ...],
) -> None:
    settings = data["settings"]
    missing: list[str] = []
    if settings["built_in_potential_mode"] != "semiconductor_work_function":
        missing.append(
            "settings.built_in_potential_mode: only explicit semiconductor_work_function contacts are bound"
        )
    if not settings["dos_band_potentials"] and not _has_uniform_dos(layers):
        missing.append(
            "settings.dos_band_potentials: physical DOS transport requires this enabled when Nc300/Nv300 vary"
        )
    has_ions = any(
        dict(layer.parameters)[name] != 0.0
        for layer in layers
        for name in ("P0", "P0_neg", "D_ion", "D_ion_neg")
    )
    for flag in (
        "interface_plane_projection",
        "te_physical_norm",
        "flat_band_metal_contacts",
        "interface_two_sided",
        "interface_shared_occupancy",
        "interface_plane_closure",
        "interface_plane_generation",
        "graded_optics",
        "interface_tunneling",
    ):
        if settings[flag]:
            missing.append("settings." + flag)
    if has_ions:
        for flag in ("ion_steric_diffusion_only", "ion_steric_shared_site"):
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
    declaration = data["declaration"]
    spectral = declaration.get("spectrum") is not None
    if spectral and declaration.get("fixed_generation") is not None:
        missing.append("spectrum and fixed_generation are mutually exclusive")
    for layer in prepared.layers:
        if layer.role == "substrate":
            parameters = dict(layer.parameters)
            if not spectral and (
                parameters["alpha"] != 0.0
                or layer.optical is not None
                or parameters["n_optical"] is not None
            ):
                missing.append(
                    "layers." + layer.id + ": absorbing/optical substrate generation is not bound"
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
        if any(
            parameters[name] is not None
            for name in ("trap_N_t_interface", "trap_N_t_bulk", "trap_decay_length")
        ):
            missing.append(prefix + "trap_profile")
        if layer.cigs_graded_optics_json is not None or (
            not spectral and (layer.optical is not None or parameters["n_optical"] is not None)
        ):
            missing.append(
                prefix + "optical_material: requires an explicit spectrum; graded optics is not bound"
            )
    for interface in prepared.interfaces:
        entry = interface.to_mapping()
        if interface.electrical and (
            entry.get("defect") is not None
            or any((entry.get(name) or 0.0) != 0.0 for name in ("v_n", "v_p"))
        ):
            missing.append("interfaces." + entry["id"] + ": interface recombination is not bound")
    if missing:
        raise CapabilityError("unsupported_nonlocal_1d_dc: " + "; ".join(missing))


def _layer_input(layer: PreparedLayer) -> LayerParameterInput:
    parameters = dict(layer.parameters)
    for name in ("chi", "Eg", "Nc300", "Nv300", "ni", "eps_r", "mu_n", "mu_p"):
        value = parameters[name]
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise CapabilityError(
                f"layers.{layer.id}.{name}: nonlocal 1D DC requires a finite positive value"
            )
    p = cast(_LayerScalars, parameters)
    return LayerParameterInput(
        name=layer.id,
        role=layer.role,
        thickness_m=layer.thickness,
        electron_affinity_eV=p["chi"],
        band_gap_eV=p["Eg"],
        intrinsic_density_m3=p["ni"],
        srh_n1_m3=p["n1"],
        srh_p1_m3=p["p1"],
        electron_mobility_m2_V_s=p["mu_n"],
        hole_mobility_m2_V_s=p["mu_p"],
        radiative_coefficient_m3_s=p["B_rad"],
        thermal_velocity_m_s=p["v_th"],
        doping=DopingProfileSpec(
            p["N_A"],
            p["N_D"],
            p["N_A_bulk"],
            p["N_D_bulk"],
            p["doping_profile_shape"],
            p["doping_decay_length"],
            p["doping_edge"],
        ),
        conduction_dos_300_m3=p["Nc300"],
        valence_dos_300_m3=p["Nv300"],
        temperature_laws=TemperatureLawSpec(
            mobility_exponent=p["mu_T_gamma"],
            radiative_exponent=p["B_rad_T_gamma"],
            varshni_alpha_eV_K=p["varshni_alpha"],
            varshni_beta_K=p["varshni_beta"],
        ),
        grading=GradingSpec(
            band_gap_back_eV=p["Eg_back"],
            electron_affinity_back_eV=p["chi_back"],
            profile=p["grading_profile"],
            direction=p["grading_direction"],
            bowing_eV=p["grading_bowing"],
            characteristic_length_m=p["grading_char_length"],
        ),
    )


def _explicit_escape_bindings(
    absorber_layers: Sequence[PreparedLayer],
    explicit_escape_probabilities: Mapping[str, float],
    *,
    mode: str,
) -> tuple[AbsorberEscapeBinding, ...]:
    bindings: list[AbsorberEscapeBinding] = []
    for layer in absorber_layers:
        if layer.id not in explicit_escape_probabilities:
            raise CapabilityError(
                f"explicit_escape_probabilities is missing absorber layer {layer.id!r}"
            )
        raw_p = float(explicit_escape_probabilities[layer.id])
        if not math.isfinite(raw_p) or raw_p < 0.0 or raw_p > 1.0:
            raise CapabilityError(
                f"explicit escape probability for {layer.id!r} must lie in [0, 1], got {raw_p!r}"
            )
        params = dict(layer.parameters)
        eg_front = float(cast(float, params["Eg"]))
        eg_back = params.get("Eg_back")
        eg_eff = eg_front if eg_back is None else min(eg_front, float(eg_back))
        wl_gap = wavelength_at_gap(eg_eff)
        alpha_val = float(cast(float, params["alpha"]))
        n_opt_raw = params.get("n_optical")
        n_gap = 1.0 if n_opt_raw is None else float(n_opt_raw)
        k_gap = max(alpha_val, 0.0) * wl_gap / (4.0 * math.pi)
        od = max(alpha_val, 0.0) * layer.thickness
        bindings.append(
            AbsorberEscapeBinding(
                layer_id=layer.id,
                mode=mode,
                band_gap_eV=eg_eff,
                wavelength_gap_m=wl_gap,
                refractive_index_at_gap=n_gap,
                extinction_at_gap=k_gap,
                alpha_gap_m1=max(alpha_val, 0.0),
                thickness_m=layer.thickness,
                optical_depth=od,
                escape_probability=raw_p,
            )
        )
    return tuple(bindings)


def compile_nonlocal_device_1d(
    prepared: PreparedDevice,
    *,
    intervals_per_layer: int | Sequence[int] = 24,
    grid_alpha: float | Sequence[float] | None = None,
    additional_recombination: LocalRecombinationModel | None = None,
    photon_recycling_mode: str = PHOTON_RECYCLING_OFF,
    escape_probability_mode: str = YABLONOVITCH_INTRINSIC_ALPHA_D_V1,
    explicit_escape_probabilities: Mapping[str, float] | None = None,
    receiver_node_overrides: Mapping[str, ArrayLike] | None = None,
    lagged_carrier_densities_m3: tuple[ArrayLike, ArrayLike] | None = None,
    wkb_sparsity_policy: str = PREALLOCATED_RESERVOIR_SUPERSET_V1,
    driver: str = "quasi_fermi_steady_state",
) -> CompiledNonlocalDevice1D:
    """Cold-compile a ``PreparedDevice`` with optional photon recycling and WKB tunnelling."""
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("compile_nonlocal_device_1d requires a PreparedDevice")
    if photon_recycling_mode not in PHOTON_RECYCLING_MODES:
        raise CapabilityError(
            f"unknown photon_recycling_mode {photon_recycling_mode!r}; expected {PHOTON_RECYCLING_MODES}"
        )
    if escape_probability_mode not in ESCAPE_PROBABILITY_MODES:
        raise CapabilityError(
            f"unknown escape_probability_mode {escape_probability_mode!r}; expected {ESCAPE_PROBABILITY_MODES}"
        )
    if wkb_sparsity_policy not in WKB_SPARSITY_POLICIES:
        raise CapabilityError(
            f"unknown wkb_sparsity_policy {wkb_sparsity_policy!r}; expected {WKB_SPARSITY_POLICIES}"
        )

    data = prepared.to_mapping()
    settings = data["settings"]
    raw_tun = data.get("tunnelling_channels")
    try:
        tun_doc = None if raw_tun is None else tunnelling_channel_document_from_mapping(raw_tun)
    except (TunnellingChannelSchemaError, TunnellingChannelError) as exc:
        raise CapabilityError(f"invalid tunnelling_channels schema: {exc}") from exc

    has_active_tunnelling = tun_doc is not None and tun_doc.any_enabled
    if (
        photon_recycling_mode == PHOTON_RECYCLING_OFF
        and raw_tun is None
        and grid_alpha is None
        and settings["dos_band_potentials"]
    ):
        base = compile_device_1d(
            prepared,
            intervals_per_layer=intervals_per_layer,
            additional_recombination=additional_recombination,
        )
        return CompiledNonlocalDevice1D(base=base, wkb_sparsity_policy=wkb_sparsity_policy)

    layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    if not layers:
        raise CapabilityError("unsupported_nonlocal_1d_dc: device has no electrical layers")
    _validate_nonlocal_capabilities(prepared, data, layers)

    grid = {entry["layer"]: entry for entry in data["electrical_grid"]}
    if isinstance(intervals_per_layer, bool):
        raise ValueError("intervals_per_layer must be an integer or integer sequence")
    if isinstance(intervals_per_layer, int):
        if intervals_per_layer < 2:
            raise ValueError("intervals_per_layer must be at least 2")
        weights = np.asarray(
            [grid.get(layer.id, {}).get("interval_weight", 1.0) for layer in layers]
        )
        counts = (
            np.maximum(2, np.rint(intervals_per_layer * weights / np.mean(weights)))
            .astype(int)
            .tolist()
        )
    else:
        if isinstance(intervals_per_layer, (str, bytes)):
            raise ValueError("intervals_per_layer must be an integer or integer sequence")
        counts = list(intervals_per_layer)
        if len(counts) != len(layers) or any(
            type(count) is not int or count < 2 for count in counts
        ):
            raise ValueError("interval counts must be integers >= 2, one per electrical layer")

    for index, layer in enumerate(layers):
        p = dict(layer.parameters)
        if settings["band_grading"] and (p["Eg_back"] is not None or p["chi_back"] is not None):
            counts[index] *= p["grading_N_mult"]

    if grid_alpha is None:
        alphas: list[float] = [float(grid.get(layer.id, {}).get("alpha", 3.0)) for layer in layers]
    elif isinstance(grid_alpha, (int, float)) and not isinstance(grid_alpha, bool):
        alphas = [float(grid_alpha) for _ in layers]
    else:
        if isinstance(grid_alpha, (str, bytes, bool)):
            raise ValueError("grid_alpha must be a float, float sequence, or None")
        alphas = [float(v) for v in cast(Sequence[float], grid_alpha)]
        if len(alphas) != len(layers):
            raise ValueError("grid_alpha sequence must match electrical layers")

    thicknesses = [layer.thickness for layer in layers]
    interface_positions: np.ndarray = np.cumsum(thicknesses, dtype=float)[:-1]
    nodes = multilayer_grid(list(zip(thicknesses, counts)), alpha=alphas)
    masks = layer_node_masks(nodes, thicknesses, RIGHT_LAYER_WINS)
    geometry = build_geometry(INTEGER_ENDPOINT_V1, nodes, interface_positions)

    binding = compile_layer_parameters(
        nodes,
        [_layer_input(layer) for layer in layers],
        temperature_K=settings["T"],
        temperature_scaling=True,
        band_grading=settings["band_grading"],
        dos_band_potentials=True,
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
            nodes,
            electron_diffusivity,
            hole_diffusivity,
            binding.thermal_voltage_V,
            v_sat_n=paint("v_sat_n"),
            v_sat_p=paint("v_sat_p"),
            ct_beta_n=paint("ct_beta_n"),
            ct_beta_p=paint("ct_beta_p"),
            pf_gamma_n=paint("pf_gamma_n"),
            pf_gamma_p=paint("pf_gamma_p"),
        )
    except FieldMobilityCapabilityError as error:
        raise CapabilityError("unsupported_nonlocal_1d_dc: " + str(error)) from error

    bands = binding.band_structure
    log_product = (
        np.log(binding.conduction_dos_m3)
        + np.log(binding.valence_dos_m3)
        - bands.physical_band_gap_eV / binding.thermal_voltage_V
    )
    configured = binding.intrinsic_product_m6
    if (
        np.any(configured <= 0.0)
        or not np.all(np.isfinite(configured))
        or np.max(np.abs(np.log(configured) - log_product)) > 1.0e-10
    ):
        raise CapabilityError(
            "inconsistent_intrinsic_product: configured ni and physical Nc/Nv/Eg do not share one Maxwell-Boltzmann equilibrium"
        )
    poisson = PoissonOperator1D.for_geometry(geometry, paint("eps_r"))

    optical_gen: OpticalGeneration1D | None = None
    generation_name = data["declaration"].get("fixed_generation")
    if generation_name is not None:
        table = prepared.resources.get(generation_name, "fixed_generation")
        if table.array.shape != (nodes.size, 2) or not np.array_equal(table.array[:, 0], nodes):
            raise CapabilityError(
                "fixed_generation: resource coordinates must exactly match the compiled nodes; interpolation is not bound"
            )
        generation = FixedGenerationProfile(table.array[:, 1], nodes.size).generation_profile(
            geometry
        )
    elif data["declaration"].get("spectrum") is not None:
        optical_gen = bind_optical_generation_1d(prepared, geometry)
        generation = optical_gen.profile
    else:
        generation = BeerLambertBinding(nodes, paint("alpha"), settings["Phi"]).generation_profile(
            geometry
        )

    escape_bindings: tuple[AbsorberEscapeBinding, ...] = ()
    reabsorption_specs: list[AbsorberReabsorptionSpec] = []
    effective_b_rad = binding.radiative_coefficient_m3_s

    if photon_recycling_mode != PHOTON_RECYCLING_OFF:
        absorber_indices = [idx for idx, layer in enumerate(layers) if layer.role == "absorber"]
        if not absorber_indices:
            raise CapabilityError("photon_recycling requires at least one absorber layer")
        absorber_layers = [layers[idx] for idx in absorber_indices]
        absorber_masks = [masks[idx] for idx in absorber_indices]
        absorber_cv_weights = [geometry.cell_measures_m[masks[idx]] for idx in absorber_indices]

        if explicit_escape_probabilities is not None:
            try:
                escape_bindings = _explicit_escape_bindings(
                    absorber_layers,
                    explicit_escape_probabilities,
                    mode=escape_probability_mode,
                )
            except (PhotonRecyclingCapabilityError, PhotonRecyclingError) as exc:
                raise CapabilityError(str(exc)) from exc
        else:
            if optical_gen is None:
                raise CapabilityError(
                    "photon_recycling requires coherent spectral TMM optics or explicit_escape_probabilities"
                )
            absorber_inputs = [
                AbsorberEscapeInput(
                    layer_id=layer.id,
                    thickness_m=layer.thickness,
                    band_gap_front_eV=float(cast(float, dict(layer.parameters)["Eg"])),
                    band_gap_back_eV=cast(
                        float | None,
                        dict(layer.parameters).get("Eg_back")
                        if settings["band_grading"]
                        else None,
                    ),
                )
                for layer in absorber_layers
            ]
            try:
                escape_bindings = bind_absorber_escape_from_optical_stack(
                    optical_gen.binding,
                    absorber_inputs,
                    mode=escape_probability_mode,
                    electrical_nodes_m=nodes,
                    absorber_node_masks=absorber_masks,
                    absorber_weights_m=(
                        absorber_cv_weights
                        if escape_probability_mode == TMM_INTEGRATED_ABSORBANCE_CONTROL_VOLUME_V1
                        else None
                    ),
                )
            except (PhotonRecyclingCapabilityError, PhotonRecyclingError) as exc:
                raise CapabilityError(str(exc)) from exc

        if photon_recycling_mode == PHOTON_RECYCLING_LOCAL_ESCAPE_SCALED:
            effective_b_rad = effective_radiative_coefficient(
                binding.radiative_coefficient_m3_s,
                absorber_masks,
                [item.escape_probability for item in escape_bindings],
            )
        else:
            lagged_n: np.ndarray | None = None
            lagged_p: np.ndarray | None = None
            if photon_recycling_mode == LAGGED_BAKED_GENERATION_V1:
                if lagged_carrier_densities_m3 is None:
                    raise CapabilityError(
                        "lagged_baked_generation_v1 requires lagged_carrier_densities_m3=(n_lag, p_lag)"
                    )
                lagged_n = np.asarray(lagged_carrier_densities_m3[0], dtype=float)
                lagged_p = np.asarray(lagged_carrier_densities_m3[1], dtype=float)
                if lagged_n.shape != (nodes.size,) or lagged_p.shape != (nodes.size,):
                    raise CapabilityError(
                        "lagged_carrier_densities_m3 arrays must match compiled node_count"
                    )

            for layer, mask, esc in zip(absorber_layers, absorber_masks, escape_bindings):
                b_abs = binding.radiative_coefficient_m3_s[mask]
                if not np.any(b_abs > 0.0):
                    continue
                emit_nodes = np.flatnonzero(mask)
                ni_abs = configured[mask]
                try:
                    if photon_recycling_mode == CONTROL_VOLUME_CONSERVATIVE_V1:
                        recv_nodes = (
                            np.asarray(receiver_node_overrides[layer.id], dtype=np.intp)
                            if receiver_node_overrides is not None
                            and layer.id in receiver_node_overrides
                            else emit_nodes
                        )
                        reabsorption_specs.append(
                            AbsorberReabsorptionSpec.control_volume_conservative(
                                layer_id=layer.id,
                                node_count=nodes.size,
                                emitter_nodes=emit_nodes,
                                cell_measures_m=geometry.cell_measures_m,
                                escape_probability=esc.escape_probability,
                                intrinsic_radiative_B_m3_s=b_abs,
                                intrinsic_product_m6=ni_abs,
                                receiver_nodes=recv_nodes,
                            )
                        )
                    elif photon_recycling_mode == HISTORICAL_TRAPEZOID_UNIFORM_V1:
                        reabsorption_specs.append(
                            AbsorberReabsorptionSpec.historical_trapezoid_uniform(
                                layer_id=layer.id,
                                nodes_m=nodes,
                                absorber_mask=mask,
                                thickness_m=layer.thickness,
                                escape_probability=esc.escape_probability,
                                intrinsic_radiative_B_m3_s=b_abs,
                                intrinsic_product_m6=ni_abs,
                            )
                        )
                    elif photon_recycling_mode == LAGGED_BAKED_GENERATION_V1:
                        assert lagged_n is not None and lagged_p is not None
                        recv_nodes = (
                            np.asarray(receiver_node_overrides[layer.id], dtype=np.intp)
                            if receiver_node_overrides is not None
                            and layer.id in receiver_node_overrides
                            else emit_nodes
                        )
                        base_spec = AbsorberReabsorptionSpec.control_volume_conservative(
                            layer_id=layer.id,
                            node_count=nodes.size,
                            emitter_nodes=emit_nodes,
                            cell_measures_m=geometry.cell_measures_m,
                            escape_probability=esc.escape_probability,
                            intrinsic_radiative_B_m3_s=b_abs,
                            intrinsic_product_m6=ni_abs,
                            receiver_nodes=recv_nodes,
                        )
                        reabsorption_specs.append(
                            AbsorberReabsorptionSpec.lagged_baked(
                                base_spec,
                                lagged_n,
                                lagged_p,
                            )
                        )
                except (PhotonRecyclingCapabilityError, PhotonRecyclingError) as exc:
                    raise CapabilityError(str(exc)) from exc

    compiled_tunnelling: CompiledTunnellingChannels | None = None
    if has_active_tunnelling:
        interface_nodes = tuple(
            int(plane.shared_node)
            for plane in geometry.interfaces
            if plane.shared_node is not None
        )
        layer_names = tuple(layer.id for layer in layers)
        layer_boundaries_m = np.r_[0.0, np.cumsum(thicknesses, dtype=float)]
        try:
            compiled_tunnelling = compile_tunnelling_channels(
                tun_doc,
                node_count=nodes.size,
                interface_nodes=interface_nodes,
                positions_m=nodes,
                physical_affinity_eV=bands.physical_electron_affinity_eV,
                layer_names=layer_names,
                layer_boundaries_m=layer_boundaries_m,
                driver=driver,
            )
        except (TunnellingChannelCapabilityError, TunnellingChannelError) as exc:
            raise CapabilityError(str(exc)) from exc

    coefficients = BulkRecombinationCoefficients(
        nodes.size,
        configured,
        paint("tau_n"),
        paint("tau_p"),
        binding.srh_n1_m3,
        binding.srh_p1_m3,
        effective_b_rad,
        paint("C_n"),
        paint("C_p"),
    )
    explicit_defects = bind_neutral_defects(
        layers,
        masks,
        ni_sq_m6=configured,
        band_gap_eV=bands.physical_band_gap_eV,
        thermal_voltage_V=binding.thermal_voltage_V,
        conduction_dos_m3=binding.conduction_dos_m3,
        valence_dos_m3=binding.valence_dos_m3,
    )
    recombination = BulkRecombinationReaction(coefficients, explicit_defects)

    def contact_material(node: int) -> ContactMaterial:
        return ContactMaterial(
            binding.temperature_K,
            float(bands.physical_electron_affinity_eV[node]),
            float(bands.physical_band_gap_eV[node]),
            float(binding.conduction_dos_m3[node]),
            float(binding.valence_dos_m3[node]),
            float(binding.donor_density_m3[node]),
            float(binding.acceptor_density_m3[node]),
        )

    left_material, right_material = contact_material(0), contact_material(-1)
    work_left = semiconductor_work_function(left_material)
    built_in = work_left - semiconductor_work_function(right_material)
    phi_left = float(settings["phi_left"])
    reference_phi = phi_left + built_in * (nodes - nodes[0]) / (nodes[-1] - nodes[0])
    fermi = -work_left - phi_left
    reference_n = binding.conduction_dos_m3 * np.exp(
        (fermi + reference_phi + bands.physical_electron_affinity_eV)
        / binding.thermal_voltage_V
    )
    reference_p = binding.valence_dos_m3 * np.exp(
        -(
            fermi
            + reference_phi
            + bands.physical_electron_affinity_eV
            + bands.physical_band_gap_eV
        )
        / binding.thermal_voltage_V
    )
    reservoirs = [
        doping_equilibrium_densities(
            float(binding.donor_density_m3[node]),
            float(binding.acceptor_density_m3[node]),
            float(configured[node]),
        )
        for node in (0, -1)
    ]
    for node, reservoir in zip((0, -1), reservoirs):
        reference_n[node], reference_p[node] = reservoir

    contacts = {entry["side"]: entry for entry in data["contacts"]}
    velocities = resolve_exchange_velocities(
        tuple(contacts[side][name] for side in ("left", "right") for name in ("S_n", "S_p")),
        flat_band_contacts=settings["flat_band_contacts"],
        selective_contacts_enabled=True,
    )
    channels = tuple(
        CarrierExchangeChannel(carrier, side, velocity, reservoirs[side_index][carrier_index])
        for velocity, (side_index, side, carrier_index, carrier) in zip(
            velocities,
            ((0, "left", 0, "n"), (0, "left", 1, "p"), (1, "right", 0, "n"), (1, "right", 1, "p")),
        )
    )
    if all(channel.kind == "blocking" for channel in channels):
        raise CapabilityError(
            "fully_blocking_contacts: a closed carrier inventory constraint is not bound in this DC slice"
        )
    if compiled_tunnelling is not None and any(
        channel.kind != "fixed_reservoir" for channel in channels
    ):
        raise CapabilityError(
            "wkb_selective_contacts_unsupported: WKB tunnelling requires ohmic fixed_reservoir contacts"
        )

    extra_id = None if additional_recombination is None else additional_recombination.model_id
    identity_parts: tuple[object, ...] = (
        "electrode_quasi_fermi_1d_v1",
        prepared.content_sha256,
        binding.sha256,
        generation.binding_sha256,
        geometry.name,
        counts,
        extra_id,
    )
    if grid_alpha is not None:
        identity_parts += ("grid_alpha", alphas)
    if photon_recycling_mode == PHOTON_RECYCLING_LOCAL_ESCAPE_SCALED:
        identity_parts += (
            "local_escape_scaled",
            [item.sha256 for item in escape_bindings],
        )
    if explicit_defects is not None:
        identity_parts += (explicit_defects.identity_sha256,)
        owned = explicit_defects.owned_nodes
        if np.any(binding.conduction_dos_m3[owned] != binding.valence_dos_m3[owned]):
            identity_parts += (LITERAL_ABOVE_VB_DOS_REFERENCE_V1,)
    if field_transport is not None:
        identity_parts += (field_transport.identity_sha256,)
    identity = hashlib.sha256(
        json.dumps(identity_parts, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()

    base = CompiledDevice1D(
        data["id"],
        identity,
        geometry,
        binding,
        poisson,
        electron_diffusivity,
        hole_diffusivity,
        generation,
        recombination,
        channels,
        phi_left,
        built_in,
        fermi,
        reference_phi,
        reference_n,
        reference_p,
        additional_recombination,
        field_transport=field_transport,
    )
    return CompiledNonlocalDevice1D(
        base=base,
        escape_bindings=escape_bindings,
        radiative_reabsorption=tuple(reabsorption_specs),
        tunnelling_channels=compiled_tunnelling,
        wkb_sparsity_policy=wkb_sparsity_policy,
    )


def _continuity_current_bound(problem: NonlocalProblem, u: np.ndarray) -> float:
    count = problem.device.node_count
    residual = problem.residual(u)
    total = 0.0
    for carrier, offset in (("n", count), ("p", 2 * count)):
        rows = np.abs(residual[offset : offset + count]).copy()
        for side, node in (("left", 0), ("right", count - 1)):
            if problem.device.channel(carrier, side).kind == "fixed_reservoir":
                rows[node] = 0.0
        total += float(np.sum(rows))
    return Q * total


def nonlocal_physical_gates(
    problem: NonlocalProblem,
    options: SteadyOptions,
    *,
    equilibrium: bool,
) -> tuple[PhysicalGate, ...]:
    """Independent physical acceptance gates for nonlocal 1D steady states."""
    device = problem.device
    base = device.base
    count = device.node_count

    def all_currents(u: np.ndarray) -> np.ndarray:
        contact = problem.contact_current_density(u)
        return np.r_[contact[:1], problem.current_density(u), contact[1:]]

    gates: list[PhysicalGate] = [
        gate_from(
            "poisson_balance",
            lambda u: float(
                np.max(np.abs(problem.residual(u)[:count] * poisson_row_weights(base)))
            ),
            options.poisson_tolerance,
            "maximum normalized Gauss/contact residual",
        ),
        gate_from(
            "continuity_current_bound",
            lambda u: _continuity_current_bound(problem, u),
            options.continuity_tolerance_A_m2,
            "q * sum of absolute carrier balances, including nonlocal couplings",
        ),
        gate_from(
            "all_face_current_spread",
            lambda u: float(np.ptp(all_currents(u))),
            options.current_spread_tolerance_A_m2,
            "maximum minus minimum over every internal/external contact face",
        ),
    ]
    if equilibrium:

        def thermodynamics(u: np.ndarray) -> GateOutcome:
            state = device.state(u, voltage_V=problem.voltage_V)
            bands = device.parameters.band_structure

            def endpoint(node: int) -> ContactEndpoint:
                return ContactEndpoint(
                    float(state.electron_density_m3[node]),
                    float(state.hole_density_m3[node]),
                    float(device.parameters.conduction_dos_m3[node]),
                    float(device.parameters.valence_dos_m3[node]),
                    float(bands.physical_electron_affinity_eV[node]),
                    float(bands.physical_band_gap_eV[node]),
                )

            assessment = assess_contact_thermodynamics(
                built_in_potential_mode="semiconductor_work_function",
                left=endpoint(0),
                right=endpoint(-1),
                V_bi_bc=float(state.potential_V[-1] - state.potential_V[0]),
                thermal_voltage_V=device.thermal_voltage_V,
                has_band_reference=True,
            )
            span = assessment.fermi_level_span_eV
            return GateOutcome(
                "contact_thermodynamics",
                assessment.certified,
                math.inf if span is None else span,
                assessment.tolerance_eV,
                assessment.status,
            )

        def max_equilibrium_current(u: np.ndarray) -> float:
            fluxes = problem.fluxes(u)
            jn = fluxes.electron_current()
            if device.has_wkb_tunnelling:
                ext = evaluate_nonlocal_couplings_1d(
                    device, u, voltage_V=problem.voltage_V, derivatives=False
                )
                jn = jn + ext.electron_face_current_addition_A_m2
            return float(
                max(
                    np.max(np.abs(jn)),
                    np.max(np.abs(fluxes.hole_current())),
                    np.max(np.abs(problem.contact_current_density(u))),
                )
            )

        gates.extend(
            (
                gate_from(
                    "zero_equilibrium_carrier_current",
                    max_equilibrium_current,
                    options.equilibrium_current_tolerance_A_m2,
                    "each carrier (drift-diffusion + WKB) on every face and both contacts",
                ),
                thermodynamics,
            )
        )
    return tuple(gates)


def _density_basin_seed_nonlocal(problem: NonlocalProblem, u: np.ndarray) -> np.ndarray:
    count = problem.device.node_count
    _, n, p = problem.device.fields(u, voltage_V=problem.voltage_V)
    residual = problem.residual(u)[count:]
    density_inverse = np.r_[1.0 / n, -1.0 / p]
    matrix = problem.jacobian(u)[count:, count:] @ diags(density_inverse)
    increment = factorize(matrix).solve(-residual)
    trial_n, trial_p = n + increment[:count], p + increment[count:]
    if not all(
        np.all(np.isfinite(value)) and np.all(value > 0.0) for value in (trial_n, trial_p)
    ):
        raise ValueError("density preparation did not produce finite positive carriers")
    seed = np.array(u, copy=True)
    for offset, density, trial, delta, sign in (
        (count, n, trial_n, increment[:count], 1.0),
        (2 * count, p, trial_p, increment[count:], -1.0),
    ):
        with np.errstate(over="ignore", divide="ignore"):
            relative = delta / density
            change = np.log1p(relative)
        large = ~np.isfinite(change)
        change[large] = np.log(trial[large]) - np.log(density[large])
        seed[offset : offset + count] += sign * change
    return seed


def _build_nonlocal_problem(
    device: CompiledNonlocalDevice1D,
    voltage_V: float,
    generation_scale: float,
    *,
    assembled_factory: Any | None,
    use_assembled_system: bool,
) -> NonlocalProblem:
    if use_assembled_system:
        assert assembled_factory is not None
        return cast(AssembledNonlocalDC1DProblem, assembled_factory(device, voltage_V, generation_scale))
    return NonlocalDC1DProblem(device, voltage_V, generation_scale)


def _solve_nonlocal_point(
    device: CompiledNonlocalDevice1D,
    u: np.ndarray,
    voltage_V: float,
    generation_scale: float,
    options: SteadyOptions,
    *,
    equilibrium: bool = False,
    assembled_factory: Any | None = None,
    use_assembled_system: bool = True,
) -> tuple[NonlocalProblem, NewtonResult]:
    problem = _build_nonlocal_problem(
        device,
        voltage_V,
        generation_scale,
        assembled_factory=assembled_factory,
        use_assembled_system=use_assembled_system,
    )
    seed = np.asarray(u, dtype=float)
    max_rebinds = 8 if device.wkb_sparsity_policy == STRICT_WINDOW_REBIND_V1 else 0
    rebinds = 0
    while True:
        if device.wkb_sparsity_policy == STRICT_WINDOW_REBIND_V1 and device.has_wkb_tunnelling:
            problem = problem.rebind_for_state(seed)
        settings = NewtonSettings(
            production_residual_scaling(
                device.base,
                poisson_tolerance=options.poisson_tolerance,
                continuity_tolerance_A_m2=options.continuity_tolerance_A_m2,
                contact_log_tolerance=options.contact_log_tolerance,
            ),
            1.0,
            production_coordinates(
                device.base,
                potential_step_V=options.potential_step_V,
                quasi_fermi_step=options.quasi_fermi_step,
            ),
            options.budget(),
            nonlocal_physical_gates(problem, options, equilibrium=equilibrium),
            False,
        )
        try:
            result = solve_newton(
                problem,
                seed,
                settings,
                convergence_norm=production_convergence_norm(
                    device.base, convergence_policy=options.convergence_policy
                ),
            )
            break
        except SparsitySupportChangedError as exc:
            if rebinds >= max_rebinds:
                raise
            if exc.state is not None:
                seed = exc.state
            rebinds += 1
    if not result.accepted:
        raise SteadyStateError(
            f"Nonlocal DC V={voltage_V:g}, generation_scale={generation_scale:g}",
            result,
        )
    return problem, result


def _nonlocal_point_result(
    problem: NonlocalProblem,
    result: NewtonResult,
    preparation: tuple[NewtonResult, ...] = (),
    density_basin_stages: tuple[tuple[float, float], ...] = (),
) -> SteadyState1D:
    return SteadyState1D(
        problem.device.state(result.state, voltage_V=problem.voltage_V),
        problem.current_density(result.state),
        problem.contact_current_density(result.state),
        problem.voltage_V,
        problem.generation_scale,
        result,
        preparation,
        len(density_basin_stages),
        density_basin_stages,
    )


def _default_nonlocal_steady_options(
    device: CompiledNonlocalDevice1D,
    *,
    max_voltage_V: float = 0.0,
) -> SteadyOptions:
    """Default steady-state controls accounting for IEEE-754 face-flux roundoff on WKB barriers."""
    base = device.base
    donor = base.parameters.donor_density_m3
    acceptor = base.parameters.acceptor_density_m3
    unipolar = bool((donor[0] > acceptor[0]) == (donor[-1] > acceptor[-1]))
    if not device.has_wkb_tunnelling and not unipolar:
        return SteadyOptions()
    dx = np.diff(base.nodes_m)
    doping_face = np.maximum(
        np.maximum(donor[:-1], donor[1:]),
        np.maximum(acceptor[:-1], acceptor[1:]),
    )
    diff_face = np.maximum(
        base.electron_diffusivity_faces_m2_s, base.hole_diffusivity_faces_m2_s
    )
    max_face_current_scale = float(np.max(Q * diff_face * np.maximum(doping_face, 1.0) / dx))
    bias_span = 1.0 + abs(float(max_voltage_V)) / base.thermal_voltage_V
    roundoff_floor = 64.0 * float(np.finfo(float).eps) * max_face_current_scale * bias_span
    if not device.has_wkb_tunnelling and roundoff_floor <= 1.0e-8:
        return SteadyOptions()
    cont_tol = max(1.0e-8, roundoff_floor)
    return SteadyOptions(
        continuity_tolerance_A_m2=cont_tol,
        current_spread_tolerance_A_m2=cont_tol,
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )


def solve_nonlocal_equilibrium_1d(
    device: CompiledNonlocalDevice1D,
    *,
    options: SteadyOptions | None = None,
    use_assembled_system: bool | None = None,
) -> SteadyState1D:
    """Solve dark equilibrium (V=0, generation_scale=0) and certify nonlocal gates."""
    device.require_driver_capability("quasi_fermi_steady_state")
    controls = (
        _default_nonlocal_steady_options(device, max_voltage_V=0.0)
        if options is None
        else options
    )
    assembled = use_assembled() if use_assembled_system is None else bool(use_assembled_system)
    factory = assembled_nonlocal_problem_factory(device) if assembled else None
    base = device.base
    count = base.node_count

    def densities(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, n, p = base.fields(np.r_[phi, np.zeros(2 * count)])
        return n, p

    def residual(phi: np.ndarray) -> np.ndarray:
        n, p = densities(phi)
        if not np.all(np.isfinite(n)) or not np.all(np.isfinite(p)):
            return np.full(count, np.nan)
        rho = base.charge_ledger(n, p).space_charge_C_m3()
        return base.poisson.residual(phi, rho, *base.contact_potentials(0.0))

    def jacobian(phi: np.ndarray) -> object:
        n, p = densities(phi)
        slope = np.zeros(count)
        slope[1:-1] = (
            -Q
            * base.poisson.interior_widths_m
            * (n[1:-1] + p[1:-1])
            / base.thermal_voltage_V
        )
        return base.poisson.jacobian() + diags(slope)

    pb_settings = NewtonSettings(
        ResidualScaling(poisson_row_weights(base) / controls.poisson_tolerance),
        1.0,
        StepControl(np.full(count, controls.potential_step_V)),
        controls.budget(),
        (),
        False,
    )
    prepared = solve_newton(
        CallableProblem(residual, jacobian),
        base.reference_potential_V,
        pb_settings,
    )
    if not prepared.accepted:
        raise SteadyStateError("Poisson-Boltzmann equilibrium preparation", prepared)
    u = np.r_[prepared.state, np.zeros(2 * count)]
    problem, result = _solve_nonlocal_point(
        device,
        u,
        0.0,
        0.0,
        controls,
        equilibrium=True,
        assembled_factory=factory,
        use_assembled_system=assembled,
    )
    return _nonlocal_point_result(problem, result, (prepared,))


def solve_nonlocal_steady_state_1d(
    device: CompiledNonlocalDevice1D,
    *,
    initial_state: State1D | None = None,
    voltage_V: float = 0.0,
    generation_scale: float = 1.0,
    options: SteadyOptions | None = None,
    use_assembled_system: bool | None = None,
) -> SteadyState1D:
    """Solve a 1D steady-state operating point with nonlocal reabsorption / WKB tunnelling."""
    device.require_driver_capability("quasi_fermi_steady_state")
    controls = (
        _default_nonlocal_steady_options(device, max_voltage_V=voltage_V)
        if options is None
        else options
    )
    assembled = use_assembled() if use_assembled_system is None else bool(use_assembled_system)
    factory = assembled_nonlocal_problem_factory(device) if assembled else None
    target = _build_nonlocal_problem(
        device,
        voltage_V,
        generation_scale,
        assembled_factory=factory,
        use_assembled_system=assembled,
    )
    if initial_state is not None:
        if not isinstance(initial_state, State1D) or initial_state.device_sha256 != device.sha256:
            raise ValueError("initial_state must belong to the same compiled nonlocal device")
        dv = abs(target.voltage_V - initial_state.voltage_V)
        step_count = max(1, int(math.ceil(dv / controls.max_bias_step_V)))
        if step_count > controls.max_continuation_steps:
            raise ValueError("requested initialization exceeds the continuation-step budget")
        u_cur = initial_state.u
        prev_v = initial_state.voltage_V
        prep_steps: list[NewtonResult] = []
        for idx, v_step in enumerate(
            np.linspace(initial_state.voltage_V, target.voltage_V, step_count + 1)[1:]
        ):
            v_val = float(v_step)
            u_cur = device.rebase_coordinates(u_cur, prev_v, v_val)
            problem, result = _solve_nonlocal_point(
                device,
                u_cur,
                v_val,
                target.generation_scale,
                controls,
                equilibrium=(v_val == 0.0 and target.generation_scale == 0.0),
                assembled_factory=factory,
                use_assembled_system=assembled,
            )
            u_cur = result.state
            prev_v = v_val
            if idx != step_count - 1:
                prep_steps.append(result)
        return _nonlocal_point_result(problem, result, tuple(prep_steps))

    equilibrium = solve_nonlocal_equilibrium_1d(
        device, options=controls, use_assembled_system=assembled
    )
    if generation_scale == 0.0 and voltage_V == 0.0:
        return equilibrium

    u = equilibrium.state.u
    preparation = [*equilibrium.preparation, equilibrium.newton]
    if generation_scale > 0.0:
        fractions: tuple[float, ...] = (
            (1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0)
            if device.has_wkb_tunnelling
            else (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0)
        )
        light_steps = [fraction * generation_scale for fraction in fractions]
    else:
        light_steps = []

    bias_count = int(math.ceil(abs(voltage_V) / controls.max_bias_step_V))
    if len(light_steps) + bias_count > controls.max_continuation_steps:
        raise ValueError("requested initialization exceeds the continuation-step budget")

    points = [(0.0, value) for value in light_steps]
    points.extend(
        (float(value), generation_scale)
        for value in np.linspace(0.0, voltage_V, bias_count + 1)[1:]
    )
    previous_voltage = 0.0
    density_stages: list[tuple[float, float]] = []
    for index, (voltage, light) in enumerate(points):
        u = device.rebase_coordinates(u, previous_voltage, voltage)
        if index < len(light_steps) and light > 0.0 and not device.has_wkb_tunnelling:
            stage_problem = _build_nonlocal_problem(
                device,
                voltage,
                light,
                assembled_factory=factory,
                use_assembled_system=assembled,
            )
            u = _density_basin_seed_nonlocal(stage_problem, u)
            density_stages.append((voltage, light))
        problem, result = _solve_nonlocal_point(
            device,
            u,
            voltage,
            light,
            controls,
            assembled_factory=factory,
            use_assembled_system=assembled,
        )
        u = result.state
        previous_voltage = voltage
        if index != len(points) - 1:
            preparation.append(result)
    return _nonlocal_point_result(problem, result, tuple(preparation), tuple(density_stages))
