"""Compile one four-side-blocking 2D single-positive-ion finite-site device.

Dual-ion, shared-site, periodic-x, selective/Robin carrier contacts, and
dynamic-interface modes fail closed with ``CapabilityError``. Bounded
absorber radiative reabsorption and clamp-passive two-sided cross-node
interface SRH are compiled when explicitly enabled and validated.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.compile_ionic_1d import ION_INPUTS
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_2d import (
    CompiledIonicDevice2D, RadiativeReabsorptionSpec2D, TwoSidedInterfaceSRHCoupling2D,
)
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.mesh.geometry import (
    HALF_ENDPOINT_V1,
    PHYSICAL_HALF_ENDPOINT_V2,
    half_endpoint_geometry,
    physical_half_endpoint_geometry,
)
from solarlab.mesh.geometry_2d import RectangularGeometry2D
from solarlab.mesh.grid import LEFT_LAYER_WINS, node_owner
from solarlab.physics.band_structure.temperature import ion_diffusivity_at_temperature
from solarlab.physics.interface_transport import (
    ThermionicCombinationMode,
    ThermionicModelVersion,
)
from solarlab.physics.ions import DIFFUSION_ONLY_STERIC_V1, WHOLE_FLUX_STERIC_V1
from solarlab.physics.transport.electrochemical_flux import LEGACY_SUMMED_POTENTIAL_V1

__all__ = [
    "CompiledIonicDevice2D",
    "build_radiative_reabsorption_2d",
    "build_two_sided_interface_srh_2d",
    "compile_ionic_device_2d",
]


def _select_ion_layer(prepared: PreparedDevice, ion_domain_layer: str | None) -> tuple[Any, tuple[Any, ...]]:
    layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    if ion_domain_layer is not None:
        selected = [layer for layer in layers if layer.id == ion_domain_layer]
        if len(selected) != 1:
            raise CapabilityError("ionic_2d: ion_domain_layer must name one electrical layer")
        return selected[0], layers
    active = [
        layer for layer in layers
        if any(float(cast(float, dict(layer.parameters)[k])) != 0.0 for k in ("P0", "D_ion", "P0_neg", "D_ion_neg"))
    ]
    if len(active) == 1:
        return active[0], layers
    if len(layers) == 1:
        return layers[0], layers
    raise CapabilityError(
        "ionic_2d: declare exactly one active ion-bearing electrical layer or pass ion_domain_layer explicitly"
    )


def build_radiative_reabsorption_2d(
    prepared: PreparedDevice,
    electronic_2d: Any,
    *,
    enabled: bool | None = None,
) -> tuple[RadiativeReabsorptionSpec2D, ...]:
    """Build 2D absorber radiative-reabsorption specs when enabled (`mode=='full'` or explicit)."""
    data = prepared.to_mapping()
    settings = data["settings"]
    is_enabled = (settings.get("mode") == "full") if enabled is None else bool(enabled)
    if not is_enabled:
        return ()
    layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    owners = node_owner(
        electronic_2d.geometry.y_geometry.nodes_m,
        [layer.thickness for layer in layers],
        LEFT_LAYER_WINS,
    )
    b_rad_y = electronic_2d.column.recombination.coefficients.radiative_m3_s
    specs: list[RadiativeReabsorptionSpec2D] = []
    lateral_len = electronic_2d.geometry.lateral_length_m
    for idx, layer in enumerate(layers):
        if layer.role != "absorber":
            continue
        params = dict(layer.parameters)
        # Critical-angle internal reflection escape probability 1 / (4 n^2) when n_optical or eps_r given
        n_opt = params.get("n_optical")
        eps_val = float(cast(Any, params["eps_r"]))
        refractive = (
            float(cast(Any, n_opt))
            if isinstance(n_opt, (int, float)) and not isinstance(n_opt, bool) and float(n_opt) > 1.0
            else math.sqrt(max(eps_val, 1.0))
        )
        p_esc_raw = params.get("photon_escape_probability")
        p_esc = (
            float(cast(Any, p_esc_raw))
            if isinstance(p_esc_raw, (int, float)) and not isinstance(p_esc_raw, bool)
            else min(1.0 / (4.0 * refractive * refractive), 0.999999)
        )
        rows = np.flatnonzero(owners == idx)
        if rows.size < 2 or not np.any(b_rad_y[rows] > 0.0) or p_esc >= 1.0:
            continue
        y_lo, y_hi = int(rows[0]), int(rows[-1]) + 1
        area = float(layer.thickness) * lateral_len
        if area > 0.0:
            specs.append(RadiativeReabsorptionSpec2D(
                layer_id=layer.id,
                y_start=y_lo,
                y_stop=y_hi,
                escape_probability=p_esc,
                area_m2=area,
            ))
    return tuple(specs)


def build_two_sided_interface_srh_2d(
    prepared: PreparedDevice,
    electronic_2d: Any,
    *,
    interface_srh_mode: str = "off",
) -> tuple[TwoSidedInterfaceSRHCoupling2D, ...]:
    """Validate and build clamp-passive two-sided cross-node interface SRH sheets."""
    data = prepared.to_mapping()
    settings = data["settings"]
    unsupported_flags = [
        name for name in (
            "interface_plane_projection",
            "interface_shared_occupancy",
            "interface_plane_closure",
            "interface_plane_generation",
            "interface_tunneling",
        )
        if settings.get(name)
    ]
    if float(settings.get("het_recomb_despike", 0.0)) != 0.0:
        unsupported_flags.append("het_recomb_despike")
    if settings.get("interface_charge_closure", "off") != "off":
        unsupported_flags.append("interface_charge_closure")
    if unsupported_flags:
        raise CapabilityError(
            "dynamic_interface_2d_unsupported: 2D solver refuses dynamic/projected interface modes: "
            + ", ".join(unsupported_flags)
        )

    electrical_ifaces = [iface for iface in prepared.interfaces if iface.electrical]
    active_declared = [
        iface for iface in electrical_ifaces
        if iface.to_mapping().get("defect") is not None
        or any((iface.to_mapping().get(k) or 0.0) != 0.0 for k in ("v_n", "v_p"))
    ]
    if interface_srh_mode == "off":
        if active_declared or settings.get("interface_two_sided"):
            raise CapabilityError(
                "interface_srh_2d_not_enabled: active electrical interface recombination cannot be silently ignored"
            )
        return ()
    if interface_srh_mode != "two_sided_cross_node":
        raise CapabilityError(f"unsupported_interface_srh_2d: {interface_srh_mode}")
    if not settings.get("interface_two_sided"):
        raise CapabilityError(
            "two_sided_interface_srh_2d: requires explicit settings.interface_two_sided=True"
        )
    if electronic_2d.geometry.lateral_bc != "neumann":
        raise CapabilityError("two_sided_interface_srh_2d: requires Neumann lateral geometry")
    if any(ch.kind != "fixed_reservoir" for ch in electronic_2d.column.channels):
        raise CapabilityError("two_sided_interface_srh_2d: requires ohmic fixed-reservoir contacts")

    planes = electronic_2d.column.geometry.interfaces
    if len(electrical_ifaces) != len(planes):
        raise CapabilityError("two_sided_interface_srh_2d: electrical interface count disagrees with stack planes")

    elec_layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    ny = electronic_2d.geometry.ny
    thermal_V = electronic_2d.column.thermal_voltage_V

    def _doping_eq_n_p(params: Mapping[str, Any]) -> tuple[float, float]:
        n_d = float(cast(Any, params.get("N_D", 0.0)))
        n_a = float(cast(Any, params.get("N_A", 0.0)))
        ni_val = float(cast(Any, params["ni"]))
        net = 0.5 * (n_d - n_a)
        disc = math.sqrt(net * net + ni_val * ni_val)
        if net >= 0.0:
            n_eq = net + disc
            p_eq = (ni_val * ni_val) / n_eq
        else:
            p_eq = -net + disc
            n_eq = (ni_val * ni_val) / p_eq
        return n_eq, p_eq

    couplings: list[TwoSidedInterfaceSRHCoupling2D] = []
    for idx, (iface, plane) in enumerate(zip(electrical_ifaces, planes)):
        entry = iface.to_mapping()
        defect = entry.get("defect")
        v_n_decl = float(entry.get("v_n") or 0.0)
        v_p_decl = float(entry.get("v_p") or 0.0)
        mic_pair = entry.get("microscopic_capture_velocities_m_s")
        mic_vn = float(mic_pair[0]) if isinstance(mic_pair, (list, tuple)) and len(mic_pair) == 2 else 0.0
        mic_vp = float(mic_pair[1]) if isinstance(mic_pair, (list, tuple)) and len(mic_pair) == 2 else 0.0
        v_n_raw = v_n_decl if v_n_decl > 0.0 else mic_vn
        v_p_raw = v_p_decl if v_p_decl > 0.0 else mic_vp
        if defect is None and v_n_decl == 0.0 and v_p_decl == 0.0:
            continue
        if v_n_raw == 0.0 or v_p_raw == 0.0:
            if defect is None:
                raise CapabilityError(
                    "two_sided_interface_srh_2d: each active sheet requires a microscopic interface defect"
                )
            continue
        if defect is None:
            raise CapabilityError(
                "two_sided_interface_srh_2d: each active sheet requires a microscopic interface defect"
            )
        row = plane.shared_node
        if row is None or row <= 0 or row >= ny - 1:
            raise CapabilityError("two_sided_interface_srh_2d: requires interior shared interface row")
        calibration = float(cast(Any, defect.get("calibration_factor", 1.0)))
        if not math.isfinite(calibration) or calibration <= 0.0:
            raise CapabilityError("two_sided_interface_srh_2d: positive calibration_factor is required")
        v_n = v_n_raw * calibration
        v_p = v_p_raw * calibration

        left_layer, right_layer = elec_layers[idx], elec_layers[idx + 1]
        left_p, right_p = dict(left_layer.parameters), dict(right_layer.parameters)
        if left_layer.role == "absorber" and right_layer.role != "absorber":
            ref_p = left_p
        elif right_layer.role == "absorber" and left_layer.role != "absorber":
            ref_p = right_p
        else:
            ref_p = left_p if float(cast(Any, left_p["Eg"])) <= float(cast(Any, right_p["Eg"])) else right_p

        gap_eV = float(cast(Any, ref_p["Eg"]))
        ni_ref = float(cast(Any, ref_p["ni"]))
        depth_eV = float(cast(Any, defect["trap_depth_eV"]))
        energy_ref = str(defect.get("energy_reference", "below_conduction_band"))
        shift_eV = (0.5 * gap_eV - depth_eV) if energy_ref == "below_conduction_band" else (depth_eV - 0.5 * gap_eV)
        n1 = ni_ref * math.exp(shift_eV / thermal_V)
        p1 = ni_ref * math.exp(-shift_eV / thermal_V)

        n_l_eq, p_l_eq = _doping_eq_n_p(left_p)
        n_r_eq, p_r_eq = _doping_eq_n_p(right_p)
        pair_a_eq = float(n_r_eq * p_l_eq)
        pair_b_eq = float(n_l_eq * p_r_eq)
        couplings.append(TwoSidedInterfaceSRHCoupling2D(
            interface_index=idx,
            interface_row=int(row),
            left_sample_row=int(row - 1),
            right_sample_row=int(row + 1),
            electron_capture_velocity_m_s=v_n,
            hole_capture_velocity_m_s=v_p,
            n1_m3=n1,
            p1_m3=p1,
            pair_a_equilibrium_product_m6=pair_a_eq,
            pair_b_equilibrium_product_m6=pair_b_eq,
        ))
    if not couplings:
        raise CapabilityError("two_sided_interface_srh_2d: no active capture sheet found")
    return tuple(couplings)


def compile_ionic_device_2d(
    prepared: PreparedDevice,
    spatial: JV2DInput,
    *,
    ion_domain_layer: str | None = None,
    ion_domain_endpoints: tuple[str, str] | None = None,
    enable_radiative_reabsorption: bool | None = None,
    dielectric_rule: str | None = None,
    stack_geometry: str = HALF_ENDPOINT_V1,
    geometry_name: str | None = None,
    band_step_mode: str = LEGACY_SUMMED_POTENTIAL_V1,
    thermionic_mode: ThermionicModelVersion | None = None,
    combination_mode: ThermionicCombinationMode | None = None,
    optical_model: Literal["beer_lambert", "beer_lambert_spectral", "tmm"] | None = None,
) -> CompiledIonicDevice2D:
    """Compile a 2D single-positive-ion device with four-side blocking boundaries."""
    if not isinstance(prepared, PreparedDevice) or not isinstance(spatial, JV2DInput):
        raise TypeError("compile_ionic_device_2d requires PreparedDevice and JV2DInput")
    if spatial.lateral_length is None or spatial.Nx is None or spatial.Ny_per_layer is None:
        raise CapabilityError("ionic_2d: requires explicit lateral_length, Nx, and Ny_per_layer")
    if spatial.lateral_bc != "neumann":
        raise CapabilityError(
            "periodic_ionic_2d_unsupported: 2D mobile-ion requires explicit 'neumann' lateral boundaries"
        )
    if spatial.ion_dynamics not in (None, "single_mobile"):
        raise CapabilityError(
            f"ionic_2d_unsupported_dynamics: compile_ionic_device_2d requires single_mobile (got {spatial.ion_dynamics!r})"
        )

    data = prepared.to_mapping()
    settings = data["settings"]
    if settings.get("ion_steric_shared_site"):
        raise CapabilityError("dual_ion_2d_unsupported: shared-site dual ions are unsupported in 2D")
    for contact in data["contacts"]:
        if contact.get("S_n") is not None or contact.get("S_p") is not None:
            raise CapabilityError(
                "selective_robin_ionic_2d_unsupported: 2D mobile-ion requires ohmic carrier contacts"
            )

    domain_layer, layers = _select_ion_layer(prepared, ion_domain_layer)
    if ion_domain_endpoints is None:
        if len(layers) != 1 and ion_domain_layer is not None:
            raise CapabilityError("ionic_2d: multilayer ion_domain_layer requires explicit ('blocking', 'blocking') endpoints")
        endpoints = ("blocking", "blocking")
    else:
        if ion_domain_endpoints != ("blocking", "blocking"):
            raise CapabilityError("ionic_2d: ion interface exchange is unsupported; declare ('blocking', 'blocking')")
        endpoints = ion_domain_endpoints

    declarations = [
        {"layer": layer.id, **{name: dict(layer.parameters)[name] for name in ION_INPUTS}}
        for layer in prepared.layers
    ]
    for layer in prepared.layers:
        p = dict(layer.parameters)
        if p["P0_neg"] != 0.0 or p["D_ion_neg"] != 0.0:
            raise CapabilityError("dual_ion_2d_unsupported: 2D mobile-ion supports one positive species only")
        if layer.role == "substrate" and (p["P0"] != 0.0 or p["D_ion"] != 0.0):
            raise CapabilityError("ionic_2d: substrate ion populations are unsupported")
        if layer.id != domain_layer.id and (p["P0"] != 0.0 or p["D_ion"] != 0.0):
            raise CapabilityError("ionic_2d: multiple active ion layers are unsupported in 2D")

    dom_params = dict(domain_layer.parameters)
    reference = float(cast(float, dom_params["P0"]))
    capacity = float(cast(float, dom_params["P_lim"]))
    diffusion = float(cast(float, dom_params["D_ion"]))
    activation = float(cast(float, dom_params["E_a_ion"]))
    if not all(math.isfinite(v) for v in (reference, capacity, diffusion, activation)):
        raise CapabilityError("ionic_2d: finite ionic parameters are required")
    if not 0.0 < reference < capacity:
        raise CapabilityError("ionic_2d: strictly positive unsaturated finite-site reference 0 < P0 < P_lim is required")
    if diffusion < 0.0 or activation < 0.0:
        raise CapabilityError("ionic_2d: diffusivity and activation energy must be non-negative")
    if spatial.ion_dynamics == "single_mobile" and diffusion <= 0.0:
        raise CapabilityError("mobile_ion_2d_requires_positive_diffusivity: single_mobile requires D_ion > 0")

    interface_srh_mode = spatial.interface_srh or "off"
    # Validate dynamic interface flags BEFORE projecting away interface declarations
    unsupported_iface = [
        name for name in (
            "interface_plane_projection",
            "interface_shared_occupancy",
            "interface_plane_closure",
            "interface_plane_generation",
            "interface_tunneling",
        )
        if settings.get(name)
    ]
    if float(settings.get("het_recomb_despike", 0.0)) != 0.0:
        unsupported_iface.append("het_recomb_despike")
    if settings.get("interface_charge_closure", "off") != "off":
        unsupported_iface.append("interface_charge_closure")
    if unsupported_iface:
        raise CapabilityError(
            "dynamic_interface_2d_unsupported: " + ", ".join(unsupported_iface)
        )

    # Project electronic component cleanly (stripping only the bound ionic & optional two-sided SRH fields,
    # recording every projection change in the immutable manifest)
    projection = prepared.to_input().editing_data()
    changes: list[dict[str, object]] = []
    for layer_dict in projection["layers"]:
        for name in ("P0", "P0_neg", "D_ion", "D_ion_neg"):
            layer_dict.setdefault("parameters", {})[name] = 0.0
            changes.append({"path": f"layers.{layer_dict['id']}.{name}", "component_value": 0.0})
    for name in ("ion_steric_diffusion_only", "ion_steric_shared_site"):
        projection.setdefault("settings", {})[name] = False
        changes.append({"path": f"settings.{name}", "component_value": False})
    if interface_srh_mode == "two_sided_cross_node":
        projection.setdefault("settings", {})["interface_two_sided"] = False
        projection["interfaces"] = []
        changes.append({"path": "interfaces", "component_value": "bound_in_two_sided_cross_node_2d"})

    component_prepared = resolve_device(
        DeviceInput.model_validate(projection),
        prepared.defaults,
        prepared.resources,
        sources=prepared.sources,
    )
    geometry_spatial = JV2DInput(
        kind="jv_2d",
        lateral_length=spatial.lateral_length,
        Nx=spatial.Nx,
        Ny_per_layer=spatial.Ny_per_layer,
        lateral_bc=spatial.lateral_bc,
        microstructure=spatial.microstructure,
    )
    electronic_2d = compile_device_2d(
        component_prepared,
        geometry_spatial,
        dielectric_rule=dielectric_rule,
        stack_geometry=stack_geometry,
        geometry_name=geometry_name,
        band_step_mode=band_step_mode,
        thermionic_mode=thermionic_mode,
        combination_mode=combination_mode,
        optical_model=optical_model,
    )
    if any(ch.kind != "fixed_reservoir" for ch in electronic_2d.column.channels):
        raise CapabilityError(
            "selective_robin_ionic_2d_unsupported: 2D mobile-ion requires ohmic fixed-reservoir contacts"
        )

    iface_couplings = build_two_sided_interface_srh_2d(
        prepared, electronic_2d, interface_srh_mode=interface_srh_mode
    )
    reabsorption_specs = build_radiative_reabsorption_2d(
        prepared, electronic_2d, enabled=enable_radiative_reabsorption
    )

    domain_index = layers.index(domain_layer)
    planes = electronic_2d.column.geometry.interfaces
    left = 0 if domain_index == 0 else planes[domain_index - 1].shared_node
    right = electronic_2d.geometry.ny - 1 if domain_index == len(layers) - 1 else planes[domain_index].shared_node
    if left is None or right is None:
        raise CapabilityError("ionic_2d: domain endpoints require shared electrostatic stack nodes")
    row_indices: NDArray[np.intp] = np.arange(left, right + 1, dtype=np.intp)
    ion_y_geometry = (
        physical_half_endpoint_geometry(electronic_2d.column.nodes_m[row_indices])
        if electronic_2d.column.geometry.name == PHYSICAL_HALF_ENDPOINT_V2
        else half_endpoint_geometry(electronic_2d.column.nodes_m[row_indices])
    )
    ion_geometry = RectangularGeometry2D(electronic_2d.geometry.x_m, ion_y_geometry, lateral_bc="neumann")
    ion_count = ion_geometry.node_count
    ion_face_count = ion_geometry.face_spacing_m.size

    scaled_diffusion = ion_diffusivity_at_temperature(
        diffusion, electronic_2d.column.parameters.temperature_K, activation
    )
    if not math.isfinite(scaled_diffusion) or scaled_diffusion < 0.0:
        raise CapabilityError("ionic_2d: temperature-scaled ion diffusivity is outside the finite envelope")

    occupancy = reference / capacity
    logit = math.log(occupancy) - math.log1p(-occupancy)
    steric_model = (
        DIFFUSION_ONLY_STERIC_V1 if settings.get("ion_steric_diffusion_only") else WHOLE_FLUX_STERIC_V1
    )
    ref_arr: NDArray[np.float64] = np.full(ion_count, reference, dtype=float)
    lim_arr: NDArray[np.float64] = np.full(ion_count, capacity, dtype=float)
    logit_arr: NDArray[np.float64] = np.full(ion_count, logit, dtype=float)
    faces_arr: NDArray[np.float64] = np.full(ion_face_count, scaled_diffusion, dtype=float)
    ref_inventory = math.fsum((ion_geometry.contract_geometry.volumes * ref_arr).tolist())

    manifest: dict[str, Any] = {
        "binding": "single_positive_ion_classical_2d_residual_v1",
        "original_prepared_sha256": prepared.content_sha256,
        "original_ionic_inputs": declarations,
        "electronic_component": {
            "compiled_sha256": electronic_2d.identity,
            "projection_changes": changes,
        },
        "ion_domain": {
            "layer": domain_layer.id,
            "contacts": [endpoints[0], endpoints[1], "blocking_x_min", "blocking_x_max"],
            "row_indices": row_indices.tolist(),
            "x_m": ion_geometry.x_m.tolist(),
            "y_m": ion_geometry.y_geometry.nodes_m.tolist(),
            "cell_areas_m2": ion_geometry.contract_geometry.volumes.tolist(),
            "reference_inventory_m1": ref_inventory,
        },
        "ion_model": {
            "site_contract": "independent_finite_sites",
            "steric_model": steric_model,
            "scaled_nodal_diffusivity_m2_s": scaled_diffusion,
            "face_diffusivity_m2_s": faces_arr.tolist(),
        },
        "extensions_2d": {
            "interface_srh_mode": interface_srh_mode,
            "interface_srh_rows": [c.interface_row for c in iface_couplings],
            "radiative_reabsorption_layers": [s.layer_id for s in reabsorption_specs],
        },
        "spatial_declaration": json.loads(spatial.model_dump_json()),
    }
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    sha = hashlib.sha256(manifest_json.encode()).hexdigest()
    return CompiledIonicDevice2D(
        device_id=str(data.get("id") or "ionic_2d"),
        sha256=sha,
        original_prepared_sha256=prepared.content_sha256,
        electronic_component=electronic_2d,
        ion_geometry=ion_geometry,
        domain_layer_id=domain_layer.id,
        ion_row_indices=row_indices,
        positive_ion_reference_m3=ref_arr,
        positive_ion_site_limit_m3=lim_arr,
        positive_ion_logit_reference=logit_arr,
        positive_ion_diffusivity_faces_m2_s=faces_arr,
        steric_model=steric_model,
        binding_manifest_json=manifest_json,
        radiative_reabsorption=reabsorption_specs,
        interface_srh_couplings=iface_couplings,
    )
