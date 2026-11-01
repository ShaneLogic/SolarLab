"""Bind an explicit rectangular geometry to classical ion-free 2D DC.

Geometry and neutral grain bands of the existing typed JV2DInput are admitted;
this is an equation compiler, not a 2D J-V/time-protocol executor. The existing
1D compiler supplies physical coefficients and reservoirs without running a
solver. All residuals and tangents are assembled on the connected 2D graph.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.grain_2d import bind_grain_regions_2d, grain_declarations
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.device.optical_generation_2d import bind_optical_generation_2d
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.dc_2d import CompiledDevice2D
from solarlab.equations.field_transport_2d import bind_field_transport_2d
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.mesh.geometry import half_endpoint_geometry
from solarlab.mesh.geometry_2d import RectangularGeometry2D
from solarlab.physics.generation.beer_lambert import BeerLambertBinding
from solarlab.units import normalize_quantity

CLASSICAL_2D_DC_V1 = "classical_mb_fixed_reservoir_neumann_2d_dc_v1"
CLASSICAL_2D_CONTACTS_V1 = "classical_mb_carrier_exchange_neumann_2d_dc_v1"
_GEOMETRY_FIELDS = frozenset({"kind", "lateral_length", "Nx", "Ny_per_layer", "lateral_bc", "microstructure"})


def compile_device_2d(prepared: PreparedDevice, spatial: JV2DInput) -> CompiledDevice2D:
    """Compile without a nonlinear solve or inferred geometric defaults.

    Nx and Ny_per_layer are interval counts; all electrical layers receive
    exactly the requested vertical count and retain their declared grid alpha.
    Monochromatic Beer-Lambert and coherent normal/front spectral sources use
    physical half-endpoint volumes and uniform lateral extrusion. Fixed
    sources and optical-only front layers remain outside this compiler slice.
    """
    if not isinstance(prepared, PreparedDevice) or not isinstance(spatial, JV2DInput):
        raise TypeError("compile_device_2d requires PreparedDevice and typed JV2DInput geometry")
    for name in spatial.model_fields_set - _GEOMETRY_FIELDS:
        if getattr(spatial, name) is not None:
            raise CapabilityError("unsupported_classical_2d_dc: execution field " + name)
    if spatial.lateral_length is None or spatial.Nx is None or spatial.Ny_per_layer is None:
        raise CapabilityError("2D DC requires explicit lateral_length, Nx and Ny_per_layer")
    if spatial.lateral_bc != "neumann":
        raise CapabilityError("2D DC requires explicit zero-normal-flux neumann lateral boundaries")
    if spatial.Ny_per_layer < 2:
        raise ValueError("Ny_per_layer must be at least two intervals")
    data = prepared.to_mapping()
    spectral = data["declaration"].get("spectrum") is not None
    if data["declaration"].get("fixed_generation") is not None:
        raise CapabilityError("unsupported_classical_2d_dc: spectral/fixed optical generation is not bound")
    if data["settings"]["band_grading"]:
        raise CapabilityError("unsupported_classical_2d_dc: graded bands are outside this first slice")
    for layer in prepared.layers:
        p = dict(layer.parameters)
        if p["incoherent"]:
            raise CapabilityError("unsupported_classical_2d_dc: layers." + layer.id + ".incoherent optics is not bound")
        if layer.role == "substrate" or (not spectral and (layer.optical is not None or p["n_optical"] is not None)):
            raise CapabilityError("unsupported_classical_2d_dc: optical-only/index layers are not bound")
        if any(p[name] != 0.0 for name in ("pf_gamma_n", "pf_gamma_p")):
            raise CapabilityError("unsupported_classical_2d_dc: field-dependent carrier mobility is not bound")
    grain_declarations(prepared, spatial)  # Refuse invalid bands before coefficient compilation.
    column_prepared, projection = prepared, None
    if prepared.to_input().grain_boundaries:
        declaration_without_grains = prepared.to_input().editing_data()
        declaration_without_grains["grain_boundaries"] = []
        column_prepared = resolve_device(DeviceInput.model_validate(declaration_without_grains),
            prepared.defaults, prepared.resources, sources=prepared.sources)
        projection = json.dumps({"kind": "grain_free_coefficients_and_auxiliary_seed",
            "source_prepared_sha256": prepared.content_sha256,
            "coefficient_prepared_sha256": column_prepared.content_sha256,
            "removed_declaration": "grain_boundaries", "used_as_2d_result": False}, sort_keys=True).encode()
    column = compile_device_1d(column_prepared, intervals_per_layer=(spatial.Ny_per_layer,) * len(prepared.layers))
    if column.recombination.explicit_defects is not None:
        raise CapabilityError("unsupported_classical_2d_dc: explicit defect populations are not bound")
    length = normalize_quantity(spatial.lateral_length, "m")
    y_geometry = half_endpoint_geometry(column.nodes_m, [plane.position_m for plane in column.geometry.interfaces])
    geometry = RectangularGeometry2D(np.linspace(0.0, length, spatial.Nx + 1), y_geometry)
    alpha, eps = np.empty(column.node_count), np.empty(column.node_count)
    for layer, mask in zip(prepared.layers, column.parameters.layer_masks):
        values = dict(layer.parameters)
        alpha[mask], eps[mask] = values["alpha"], values["eps_r"]
    optical = bind_optical_generation_2d(prepared, geometry) if spectral else None
    generation = (BeerLambertBinding(column.nodes_m, alpha, data["settings"]["Phi"]).generation_profile(y_geometry)
                  if optical is None else optical.optical_y.profile)
    grains = bind_grain_regions_2d(prepared, spatial, geometry)
    faceD = []
    for carrier in ("electron", "hole"):
        faceD.append(np.r_[np.repeat(getattr(column.parameters, carrier + "_diffusivity_m2_s"), geometry.nx - 1),
                           np.repeat(getattr(column, carrier + "_diffusivity_faces_m2_s"), geometry.nx)])
    empirical: dict[str, np.ndarray] = {}
    for name in ("v_sat_n", "v_sat_p", "ct_beta_n", "ct_beta_p", "pf_gamma_n", "pf_gamma_p"):
        painted = np.empty(column.node_count)
        for layer, mask in zip(prepared.layers, column.parameters.layer_masks):
            painted[mask] = dict(layer.parameters)[name]
        empirical[name] = np.repeat(painted, geometry.nx)
    transport = bind_field_transport_2d(geometry, faceD[0], faceD[1], column.thermal_voltage_V,
        v_sat_n=empirical["v_sat_n"], v_sat_p=empirical["v_sat_p"],
        ct_beta_n=empirical["ct_beta_n"], ct_beta_p=empirical["ct_beta_p"],
        pf_gamma_n=empirical["pf_gamma_n"], pf_gamma_p=empirical["pf_gamma_p"])
    declaration = spatial.model_dump_json().encode()
    contact_model = (CLASSICAL_2D_DC_V1 if all(channel.kind == "fixed_reservoir" for channel in column.channels)
                     else CLASSICAL_2D_CONTACTS_V1)
    identity = hashlib.sha256(json.dumps((contact_model, prepared.content_sha256, column.identity,
                                         geometry.sha256, generation.binding_sha256, declaration.decode()),
                                        separators=(",", ":")).encode()).hexdigest()
    if grains is not None:
        identity = hashlib.sha256(json.dumps((identity, grains.model_id), separators=(",", ":")).encode()).hexdigest()
    if optical is not None:
        identity = hashlib.sha256(json.dumps((identity, optical.identity_sha256), separators=(",", ":")).encode()).hexdigest()
    if transport is not None:
        identity = hashlib.sha256(json.dumps((identity, transport.identity_sha256), separators=(",", ":")).encode()).hexdigest()
    return CompiledDevice2D(identity, column, geometry, generation, eps, declaration,
                            grain_recombination=grains, column_projection_json=projection, field_transport=transport)
