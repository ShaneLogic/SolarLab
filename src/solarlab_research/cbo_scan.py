"""Public 1D conduction/valence-band-offset (CBO) scan orchestration (P07-01).

Orchestrates 1D heterojunction band-offset scans strictly over the public
``solarlab`` configuration, compilation, and steady-state DC interfaces:
``delta_ec_eV = chi_absorber - chi_transport`` (positive values are electron
extraction spikes at an absorber/ETL boundary).

Per decision ``D20``, CBO scans pin pre-``P08`` model tokens by default so
prospective ``P08`` optical or transport corrections cannot silently alter
CBO onset benchmarks unless ``allow_prospective_p08_models=True`` is requested.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.config.device_inputs import DeviceInput
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import CapabilityError, CompiledDevice1D, compile_device_1d
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import State1D
from solarlab.experiments.jv.steady import SteadyJVMetrics, steady_jv_metrics
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import ModelRef
from solarlab.solvers.nonlinear.production import POISSON_CONTACT_BALANCE_V1
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    SteadyState1D,
    SteadyStateError,
    solve_steady_state,
)

__all__ = [
    "ADAPTIVE_JV_METRICS",
    "CBO_BOUNDARY_POLICIES",
    "FIXED_CONTACTS",
    "PRE_P08_CBO_MODEL_PINS",
    "PROSPECTIVE_P08_MODEL_IDS",
    "RECOMPUTED_BUILT_IN",
    "CBOCriticalInterval",
    "CBOGridConvergenceCertificate",
    "CBOScanConfig",
    "CBOScanPoint",
    "CBOScanResult",
    "CBOScanTermination",
    "CBOShortCircuitSample",
    "CBOVoltageGridConvergenceCertificate",
    "CBOVoltageGridConvergencePolicy",
    "CBOVoltageGridPointCertificate",
    "InterfaceCBOScanError",
    "certify_cbo_grid_convergence",
    "certify_cbo_voltage_grid_convergence",
    "default_cbo_prepared_device",
    "enforce_pre_p08_cbo_pinning",
    "prepare_cbo_point_device",
    "run_cbo_scan",
    "validate_cbo_boundary_policy",
]

FIXED_CONTACTS = "fixed_contacts"
RECOMPUTED_BUILT_IN = "recomputed_built_in"
CBO_BOUNDARY_POLICIES = (FIXED_CONTACTS, RECOMPUTED_BUILT_IN)
ADAPTIVE_JV_METRICS = ("Jsc", "Voc", "FF", "PCE")

PRE_P08_CBO_MODEL_PINS: tuple[ModelRef, ...] = (
    ModelRef("maxwell_boltzmann_statistics", 1),
    ModelRef("dos_aware_scharfetter_gummel_1d", 1),
    ModelRef("semiconductor_work_function_contacts", 1),
    ModelRef("bulk_srh_radiative_auger_1d", 1),
    ModelRef("monochromatic_beer_lambert_1d", 1),
)

PROSPECTIVE_P08_MODEL_IDS: frozenset[str] = frozenset({
    "spectral_control_volume_beer_lambert",
    "tmm_spectral_generation_p08",
    "prospective_interface_p08",
})


class InterfaceCBOScanError(RuntimeError):
    """A CBO scan could not preserve a certified continuation path or model contract."""


def validate_cbo_boundary_policy(policy: str) -> Literal["fixed_contacts", "recomputed_built_in"]:
    """Validate and normalize the contact/built-in-potential boundary policy."""
    normalized = str(policy).strip().lower()
    if normalized == FIXED_CONTACTS:
        return "fixed_contacts"
    if normalized == RECOMPUTED_BUILT_IN:
        return "recomputed_built_in"
    raise ValueError(f"boundary_policy must be one of {CBO_BOUNDARY_POLICIES}; got {policy!r}")


def enforce_pre_p08_cbo_pinning(
    prepared: PreparedDevice,
    *,
    requested_models: Sequence[ModelRef] = (),
    allow_prospective_p08_models: bool = False,
) -> tuple[ModelRef, ...]:
    """Enforce D20 pre-P08 model pinning on CBO scans unless explicitly opted in."""
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("prepared must be a PreparedDevice")
    models = tuple(requested_models) if requested_models else PRE_P08_CBO_MODEL_PINS
    for item in models:
        if not isinstance(item, ModelRef):
            raise TypeError("requested_models must contain ModelRef instances")
    if allow_prospective_p08_models:
        return models

    for item in models:
        if item.id in PROSPECTIVE_P08_MODEL_IDS or item.version >= 2:
            raise InterfaceCBOScanError(
                f"D20 model-pinning violation: prospective P08 model {item.id}@v{item.version} "
                "is disallowed unless allow_prospective_p08_models=True"
            )
    mapping = prepared.to_mapping()
    declaration = mapping.get("declaration", {})
    if declaration.get("spectrum") is not None:
        for layer in prepared.layers:
            params = dict(layer.parameters)
            if bool(params.get("incoherent", False)):
                raise InterfaceCBOScanError(
                    f"D20 model-pinning violation: layer {layer.id!r} enables spectral incoherent "
                    "optics (P08-04) without allow_prospective_p08_models=True"
                )
    return models


def default_cbo_prepared_device() -> PreparedDevice:
    """Construct a self-contained public 3-layer HTL/absorber/ETL PreparedDevice."""
    catalog_doc: dict[str, Any] = {
        "schema": "solarlab.default-catalog.v1",
        "material": {
            "eps_r": 25.0,
            "mu_n": 0.0019340863535916802,
            "mu_p": 0.0019340863535916802,
            "D_ion": 0.0,
            "P_lim": 1e28,
            "P0": 0.0,
            "ni": 363572901223.5858,
            "tau_n": 1e-6,
            "tau_p": 1e-6,
            "n1": 363572901223.5858,
            "p1": 363572901223.5858,
            "B_rad": 1e-18,
            "C_n": 1e-42,
            "C_p": 1e-42,
            "alpha": 2000000.0,
            "N_A": 0.0,
            "N_D": 0.0,
            "chi": 3.9,
            "Eg": 1.6,
            "A_star_n": 1.2e6,
            "A_star_p": 1.2e6,
            "v_th": 100000.0,
            "D_ion_neg": 0.0,
            "P0_neg": 0.0,
            "P_lim_neg": 1e28,
            "Nc300": 1e25,
            "Nv300": 1e25,
            "carrier_statistics": "maxwell_boltzmann",
            "dopant_ionization_model": "fully_ionized",
            "donor_binding_energy_eV": None,
            "acceptor_binding_energy_eV": None,
            "donor_degeneracy": 2.0,
            "acceptor_degeneracy": 4.0,
            "band_gap_narrowing_model": "off",
            "bgn_reference_energy_eV": 0.0,
            "bgn_reference_density_m3": 1e24,
            "bgn_log_shape": 0.5,
            "bgn_conduction_band_fraction": 0.5,
            "mu_T_gamma": -1.5,
            "E_a_ion": 0.0,
            "B_rad_T_gamma": 0.0,
            "varshni_alpha": 0.0,
            "varshni_beta": 0.0,
            "trap_N_t_interface": None,
            "trap_N_t_bulk": None,
            "trap_decay_length": None,
            "trap_profile_shape": "exponential",
            "trap_edge": "both",
            "optical_material": None,
            "n_optical": None,
            "incoherent": False,
            "v_sat_n": 0.0,
            "v_sat_p": 0.0,
            "ct_beta_n": 1.0,
            "ct_beta_p": 1.0,
            "pf_gamma_n": 0.0,
            "pf_gamma_p": 0.0,
            "Eg_back": None,
            "chi_back": None,
            "grading_profile": "linear",
            "grading_direction": "front_to_back",
            "grading_bowing": 0.0,
            "grading_char_length": None,
            "grading_N_mult": 1,
            "N_A_bulk": None,
            "N_D_bulk": None,
            "doping_profile_shape": None,
            "doping_decay_length": None,
            "doping_edge": "front",
        },
        "device": {
            "phi_left": 0.0,
            "V_bi": 0.0,
            "built_in_potential_mode": "semiconductor_work_function",
            "work_function_left_eV": None,
            "work_function_right_eV": None,
            "Phi": 2e21,
            "T": 300.0,
            "mode": "fast",
            "interface_plane_projection": False,
            "dos_band_potentials": True,
            "te_physical_norm": False,
            "ion_steric_diffusion_only": False,
            "ion_steric_shared_site": False,
            "flat_band_contacts": False,
            "flat_band_metal_contacts": False,
            "contact_phi_B_eV": 0.0,
            "interface_two_sided": False,
            "interface_shared_occupancy": False,
            "interface_plane_closure": False,
            "interface_plane_generation": False,
            "het_recomb_despike": 0.0,
            "band_grading": False,
            "graded_optics": False,
            "interface_tunneling": False,
            "tunnel_mass_eff": 0.2,
            "jv_solver_policy": "general",
            "interface_charge_closure": "off",
            "interface_charge_rebaseline_acknowledged": False,
        },
        "scaps": {
            "D_ion": 0.0,
            "P_lim": 1e28,
            "P0": 0.0,
            "B_rad": 1e-18,
            "C_n": 1e-42,
            "C_p": 1e-42,
            "alpha": 2000000.0,
            "tau_n": 1e-6,
            "tau_p": 1e-6,
        },
        "interface": {
            "calibration_factor": 1.0,
            "iface_state_calibration_factor": 1.0,
        },
        "constants": {
            "Q": 1.602176634e-19,
            "K_B": 1.380649e-23,
            "T": 300.0,
        },
        "contacts": {
            "S_n_left": None,
            "S_p_left": None,
            "S_n_right": None,
            "S_p_right": None,
        },
        "structural": {
            "grain_boundary_layer_role": "absorber",
            "tandem_light_direction": "top_first",
            "junction_incoherent": False,
        },
        "defect_model": "effective_lifetime",
        "defect_schema_version": None,
        "defect_degeneracy": 1.0,
        "distribution_width_convention": "not_applicable",
    }
    device_doc: dict[str, Any] = {
        "schema_version": "solarlab.device-preparation.v1",
        "id": "research_cbo_pin",
        "name": "Research 1D p-i-n heterojunction CBO baseline",
        "source_format": "canonical",
        "settings": {
            "built_in_potential_mode": "semiconductor_work_function",
            "dos_band_potentials": True,
            "T": 300.0,
            "Phi": 2e21,
        },
        "layers": [
            {
                "id": "htl",
                "name": "HTL",
                "role": "HTL",
                "thickness": 8e-8,
                "parameters": {
                    "chi": 3.6,
                    "Eg": 1.8,
                    "Nc300": 2e25,
                    "Nv300": 3e25,
                    "eps_r": 3.0,
                    "N_D": 0.0,
                    "N_A": 1e24,
                    "ni": 18609772721.447,
                    "n1": 18609772721.447,
                    "p1": 18609772721.447,
                    "mu_n": 3.8681727071833605e-5,
                    "mu_p": 3.8681727071833605e-5,
                    "alpha": 0.0,
                    "A_star_n": 1.2e6,
                    "A_star_p": 1.2e6,
                },
            },
            {
                "id": "absorber",
                "name": "absorber",
                "role": "absorber",
                "thickness": 3e-7,
                "parameters": {
                    "chi": 3.9,
                    "Eg": 1.6,
                    "Nc300": 1e25,
                    "Nv300": 1e25,
                    "eps_r": 25.0,
                    "N_D": 0.0,
                    "N_A": 5e22,
                    "ni": 363572901223.5858,
                    "n1": 363572901223.5858,
                    "p1": 363572901223.5858,
                    "tau_n": 5e-9,
                    "tau_p": 5e-9,
                    "mu_n": 0.0019340863535916802,
                    "mu_p": 0.0019340863535916802,
                    "alpha": 2000000.0,
                    "A_star_n": 1.2e6,
                    "A_star_p": 1.2e6,
                },
            },
            {
                "id": "etl",
                "name": "ETL",
                "role": "ETL",
                "thickness": 8e-8,
                "parameters": {
                    "chi": 3.9,
                    "Eg": 2.0,
                    "Nc300": 4e25,
                    "Nv300": 2e25,
                    "eps_r": 9.0,
                    "N_D": 5e21,
                    "N_A": 0.0,
                    "ni": 449039324.31171566,
                    "n1": 449039324.31171566,
                    "p1": 449039324.31171566,
                    "mu_n": 1.0e-5,
                    "mu_p": 1.0e-5,
                    "alpha": 0.0,
                    "A_star_n": 1.2e6,
                    "A_star_p": 1.2e6,
                },
            },
        ],
        "electrical_grid": [
            {"layer": "htl", "interval_weight": 12.0, "alpha": 3.0},
            {"layer": "absorber", "interval_weight": 24.0, "alpha": 3.0},
            {"layer": "etl", "interval_weight": 12.0, "alpha": 3.0},
        ],
    }
    raw_bytes = json.dumps(
        {"default_catalog": catalog_doc, "device": device_doc},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    source = SourceDocument("solarlab_research:cbo_baseline_v1", raw_bytes)
    catalog_with_evidence = {**catalog_doc, "evidence": [[source.id, source.sha256]]}
    return resolve_device(
        DeviceInput.model_validate(device_doc),
        DefaultCatalog.from_mapping(catalog_with_evidence),
        ResourceLibrary(),
        sources=(source,),
    )


def prepare_cbo_point_device(
    baseline: PreparedDevice,
    delta_ec_eV: float,
    *,
    boundary_policy: str = RECOMPUTED_BUILT_IN,
    transport_layer_id: str | None = None,
    absorber_layer_id: str | None = None,
) -> PreparedDevice:
    """Resolve a PreparedDevice at ``delta_ec_eV = chi_absorber - chi_transport``.

    Under ``recomputed_built_in``, the transport layer's ``chi`` is updated
    uniformly so outer-contact work functions adjust ``V_bi``.
    Under ``fixed_contacts``, if the swept transport layer touches an outer
    electrode, its contact-side affinity is anchored to the baseline contact
    affinity via linear grading (``grading_N_mult = 1``) so the absorber/transport
    heterointerface step equals ``delta_ec_eV`` while both contact work
    functions and ``V_bi`` remain strictly invariant.
    """
    if not math.isfinite(delta_ec_eV):
        raise ValueError("delta_ec_eV must be finite")
    policy = validate_cbo_boundary_policy(boundary_policy)
    electrical = [layer for layer in baseline.layers if layer.role != "substrate"]
    if len(electrical) < 2:
        raise ValueError("CBO scan requires at least two electrical layers")

    absorber = (
        next((layer for layer in electrical if layer.id == absorber_layer_id), None)
        if absorber_layer_id is not None
        else next((layer for layer in electrical if layer.role.lower() == "absorber"), None)
    )
    if absorber is None:
        raise ValueError("could not locate absorber layer in PreparedDevice")
    transport = (
        next((layer for layer in electrical if layer.id == transport_layer_id), None)
        if transport_layer_id is not None
        else next((layer for layer in electrical if layer.role.lower() == "etl"), None)
    )
    if transport is None:
        raise ValueError("could not locate transport layer in PreparedDevice")

    chi_abs = float(dict(absorber.parameters)["chi"])  # type: ignore[arg-type]
    chi_transport_base = float(dict(transport.parameters)["chi"])  # type: ignore[arg-type]
    target_chi = chi_abs - float(delta_ec_eV)
    if target_chi <= 0.0:
        raise ValueError(f"delta_ec_eV={delta_ec_eV:g} gives non-positive transport chi={target_chi:g} eV")

    data = baseline.to_input().editing_data()
    abs_idx = electrical.index(absorber)
    tr_idx = electrical.index(transport)
    layer_entry = next(item for item in data["layers"] if item["id"] == transport.id)
    params = layer_entry.setdefault("parameters", {})

    if policy == RECOMPUTED_BUILT_IN or (0 < tr_idx < len(electrical) - 1):
        params["chi"] = target_chi
        params.pop("chi_back", None)
    else:
        data.setdefault("settings", {})["band_grading"] = True
        params["grading_N_mult"] = 1
        params["grading_profile"] = "linear"
        params["grading_direction"] = "front_to_back"
        if tr_idx == len(electrical) - 1 and abs_idx < tr_idx:
            # Transport layer is on the right: front is absorber interface, back is right contact.
            params["chi"] = target_chi
            params["chi_back"] = chi_transport_base
        elif tr_idx == 0 and tr_idx < abs_idx:
            # Transport layer is on the left: front is left contact, back is absorber interface.
            params["chi"] = chi_transport_base
            params["chi_back"] = target_chi
        else:
            params["chi"] = target_chi

    return resolve_device(
        DeviceInput.model_validate(data),
        baseline.defaults,
        baseline.resources,
        sources=baseline.sources,
    )


@dataclass(frozen=True, slots=True)
class CBOVoltageGridConvergencePolicy:
    """Acceptance thresholds for nested J-V voltage-grid convergence."""

    minimum_voltage_grids: int = 2
    maximum_voc_change_V: float = 5.0e-3
    maximum_ff_change: float = 1.5e-2
    maximum_pce_change: float = 1.5e-2

    def __post_init__(self) -> None:
        if type(self.minimum_voltage_grids) is not int or self.minimum_voltage_grids < 2:
            raise ValueError("minimum_voltage_grids must be an integer >= 2")
        for name in ("maximum_voc_change_V", "maximum_ff_change", "maximum_pce_change"):
            val = getattr(self, name)
            if isinstance(val, bool) or not math.isfinite(float(val)) or float(val) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


def _default_cbo_steady_options() -> SteadyOptions:
    return SteadyOptions(
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
        continuity_tolerance_A_m2=1.0e-6,
        current_spread_tolerance_A_m2=1.0e-6,
    )


@dataclass(frozen=True, slots=True)
class CBOScanConfig:
    """Configuration for a public 1D CBO scan."""

    delta_ec_eV: tuple[float, ...]
    reference_delta_ec_eV: float = 0.0
    boundary_policy: str = RECOMPUTED_BUILT_IN
    interface_topology: Literal["deduplicated", "two_sided_trace"] = "deduplicated"
    interface_transmission: float = 1.0
    intervals_per_layer: int | tuple[int, ...] = 12
    voltages_V: tuple[float, ...] | None = None
    voltage_grids_V: tuple[tuple[float, ...], ...] | None = None
    incident_power_W_m2: float = 1000.0
    generation_scale: float = 1.0
    relative_drop_fraction: float = 0.05
    minimum_delta_step_eV: float = 5.0e-3
    maximum_delta_step_eV: float = 0.2
    max_bridge_points: int = 64
    calculate_jv_metrics: bool = True
    allow_prospective_p08_models: bool = False
    requested_models: tuple[ModelRef, ...] = ()
    steady_options: SteadyOptions = field(default_factory=_default_cbo_steady_options)
    voltage_grid_policy: CBOVoltageGridConvergencePolicy = field(
        default_factory=CBOVoltageGridConvergencePolicy
    )

    def __post_init__(self) -> None:
        validate_cbo_boundary_policy(self.boundary_policy)
        if not self.delta_ec_eV or any(not math.isfinite(v) for v in self.delta_ec_eV):
            raise ValueError("delta_ec_eV must be a non-empty sequence of finite floats")
        if not math.isfinite(self.reference_delta_ec_eV):
            raise ValueError("reference_delta_ec_eV must be finite")
        if not 0.0 < self.relative_drop_fraction < 1.0:
            raise ValueError("relative_drop_fraction must lie strictly in (0, 1)")
        if self.minimum_delta_step_eV <= 0.0 or self.maximum_delta_step_eV < self.minimum_delta_step_eV:
            raise ValueError("0 < minimum_delta_step_eV <= maximum_delta_step_eV is required")
        if self.voltages_V is not None and self.voltage_grids_V is not None:
            raise ValueError("specify either voltages_V or voltage_grids_V, not both")


@dataclass(frozen=True, slots=True)
class CBOShortCircuitSample:
    """One certified short-circuit point along the CBO continuation path."""

    delta_ec_eV: float
    output_current_A_m2: float
    requested: bool
    face_current_spread_A_m2: float
    applied_V_bi_V: float
    compiled_identity: str


@dataclass(frozen=True, slots=True)
class CBOScanPoint:
    """Certified operating results and J-V metrics at one CBO value."""

    delta_ec_eV: float
    requested: bool
    applied_V_bi_V: float
    compiled_identity: str
    short_circuit_output_A_m2: float
    face_current_spread_A_m2: float
    voltage_V: NDArray[np.float64]
    output_current_A_m2: NDArray[np.float64]
    metrics: SteadyJVMetrics | None
    certified: bool


@dataclass(frozen=True, slots=True)
class CBOCriticalInterval:
    """Sample-bounded onset interval for a relative metric drop on the spike branch."""

    metric: str
    reference_delta_ec_eV: float
    reference_value: float | None
    relative_drop_fraction: float
    threshold_value: float | None
    lower_delta_ec_eV: float | None
    upper_delta_ec_eV: float | None
    resolved: bool


@dataclass(frozen=True, slots=True)
class CBOScanTermination:
    """Bracket where continuation terminated along one CBO branch."""

    direction: str
    last_certified_delta_ec_eV: float
    first_failed_delta_ec_eV: float
    requested_delta_ec_eV: float
    reason: str


@dataclass(frozen=True, slots=True)
class CBOVoltageGridPointCertificate:
    """Nested voltage-grid convergence audit at a single CBO value."""

    delta_ec_eV: float
    voltage_point_counts: tuple[int, ...]
    voc_values_V: tuple[float, ...]
    ff_values: tuple[float, ...]
    pce_values: tuple[float, ...]
    final_voc_change_V: float | None
    final_ff_change: float | None
    final_pce_change: float | None
    certified: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CBOVoltageGridConvergenceCertificate:
    """Nested voltage-grid convergence certificate across all requested CBO points."""

    voltage_point_counts: tuple[int, ...]
    minimum_voltage_grids: int
    maximum_voc_change_V: float
    maximum_ff_change: float
    maximum_pce_change: float
    points: tuple[CBOVoltageGridPointCertificate, ...]
    certified: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CBOGridConvergenceCertificate:
    """Spatial mesh convergence certificate across multiple CBO scans."""

    metric: str
    grid_interval_counts: tuple[int, ...]
    critical_intervals_eV: tuple[tuple[float, float], ...]
    envelope_lower_eV: float | None
    envelope_upper_eV: float | None
    envelope_width_eV: float | None
    maximum_envelope_width_eV: float
    reference_values: tuple[float, ...]
    reference_relative_spread: float | None
    maximum_reference_relative_spread: float
    certified: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CBOScanResult:
    """Complete result of a public 1D CBO scan."""

    requested_delta_ec_eV: NDArray[np.float64]
    points: tuple[CBOScanPoint, ...]
    short_circuit_trace: tuple[CBOShortCircuitSample, ...]
    critical_intervals: tuple[CBOCriticalInterval, ...]
    terminations: tuple[CBOScanTermination, ...]
    reference_delta_ec_eV: float
    relative_drop_fraction: float
    boundary_policy: str
    interface_topology: str
    interface_transmission: float
    grid_node_count: int
    grid_interval_count: int
    bridge_point_count: int
    pinned_models: tuple[ModelRef, ...]
    voltage_grids_V: tuple[NDArray[np.float64], ...]
    voltage_grid_policy: CBOVoltageGridConvergencePolicy
    protocol_sha256: str

    @property
    def sync_vbi(self) -> bool:
        return self.boundary_policy == RECOMPUTED_BUILT_IN

    @property
    def complete(self) -> bool:
        req = [p for p in self.points if p.requested]
        return bool(
            not self.terminations
            and len(req) == self.requested_delta_ec_eV.size
            and all(
                any(math.isclose(p.delta_ec_eV, float(t), rel_tol=0.0, abs_tol=1.0e-12) for p in req)
                for t in self.requested_delta_ec_eV
            )
        )

    @property
    def certified(self) -> bool:
        return bool(self.points and all(p.certified for p in self.points))


def _validate_nested_voltage_grids(
    grids: Sequence[ArrayLike],
) -> tuple[NDArray[np.float64], ...]:
    if not grids:
        raise ValueError("at least one voltage grid is required")
    resolved: list[NDArray[np.float64]] = []
    for raw in grids:
        arr = np.array(raw, dtype=float, copy=True)
        if arr.ndim != 1 or arr.size < 2 or not np.all(np.isfinite(arr)):
            raise ValueError("each voltage grid must be a 1D finite array with at least 2 points")
        if not np.all(np.diff(arr) > 0.0):
            raise ValueError("each voltage grid must be strictly increasing")
        if not (arr[0] <= 0.0 <= arr[-1]):
            raise ValueError("each voltage grid must bracket 0 V")
        arr.setflags(write=False)
        resolved.append(arr)
    for coarse, fine in zip(resolved, resolved[1:]):
        if fine.size <= coarse.size:
            raise ValueError("nested voltage grids must strictly increase in point count")
        for v in coarse:
            if not np.any(np.isclose(fine, v, rtol=0.0, atol=1.0e-12)):
                raise ValueError(f"coarse voltage {v:g} V is missing from refined voltage grid")
    return tuple(resolved)


def _critical_interval_for_metric(
    samples: Sequence[tuple[float, float]],
    *,
    metric: str,
    reference_delta_ec_eV: float,
    relative_drop_fraction: float,
) -> CBOCriticalInterval:
    ordered = sorted(samples, key=lambda item: item[0])
    ref_val = next(
        (val for delta, val in ordered if math.isclose(delta, reference_delta_ec_eV, rel_tol=0.0, abs_tol=1.0e-9)),
        None,
    )
    if ref_val is None or not math.isfinite(ref_val) or ref_val <= 0.0:
        return CBOCriticalInterval(
            metric=metric,
            reference_delta_ec_eV=reference_delta_ec_eV,
            reference_value=ref_val,
            relative_drop_fraction=relative_drop_fraction,
            threshold_value=None,
            lower_delta_ec_eV=None,
            upper_delta_ec_eV=None,
            resolved=False,
        )
    threshold = ref_val * (1.0 - relative_drop_fraction)
    positive_branch = [
        (delta, val) for delta, val in ordered if delta >= reference_delta_ec_eV - 1.0e-12
    ]
    prev_delta: float | None = None
    for delta, val in positive_branch:
        if val <= threshold and prev_delta is not None:
            return CBOCriticalInterval(
                metric=metric,
                reference_delta_ec_eV=reference_delta_ec_eV,
                reference_value=ref_val,
                relative_drop_fraction=relative_drop_fraction,
                threshold_value=threshold,
                lower_delta_ec_eV=prev_delta,
                upper_delta_ec_eV=delta,
                resolved=True,
            )
        if val > threshold:
            prev_delta = delta
    return CBOCriticalInterval(
        metric=metric,
        reference_delta_ec_eV=reference_delta_ec_eV,
        reference_value=ref_val,
        relative_drop_fraction=relative_drop_fraction,
        threshold_value=threshold,
        lower_delta_ec_eV=prev_delta,
        upper_delta_ec_eV=None,
        resolved=False,
    )


def _seed_state_on_compiled(
    target_compiled: CompiledDevice1D,
    previous_steady: SteadyState1D,
) -> State1D | None:
    """Project an accepted steady state onto a neighboring CBO compiled device."""
    u_prev = previous_steady.state.u
    if u_prev.size != 3 * target_compiled.node_count:
        return None
    left_p, right_p = target_compiled.contact_potentials(previous_steady.voltage_V)
    u_trial = np.array(u_prev, dtype=float, copy=True)
    # Match exact Dirichlet boundary potentials for the new compiled device.
    old_left, old_right = float(u_prev[0]), float(u_prev[target_compiled.node_count - 1])
    span = target_compiled.nodes_m[-1] - target_compiled.nodes_m[0]
    s = (target_compiled.nodes_m - target_compiled.nodes_m[0]) / span
    u_trial[: target_compiled.node_count] += (left_p - old_left) * (1.0 - s) + (right_p - old_right) * s
    try:
        state: State1D = target_compiled.state(u_trial, voltage_V=previous_steady.voltage_V)
    except ValueError:
        return None
    return state


def _solve_jv_from_short_circuit(
    compiled: CompiledDevice1D,
    sc_steady: SteadyState1D,
    voltages_V: NDArray[np.float64],
    *,
    generation_scale: float,
    incident_power_W_m2: float,
    options: SteadyOptions,
) -> tuple[NDArray[np.float64], SteadyJVMetrics]:
    """Continue upward and downward from an already-certified V=0 state."""
    polarity = float(compiled.voltage_polarity)
    solutions_by_idx: dict[int, SteadyState1D] = {}

    zero_matches = np.flatnonzero(np.isclose(voltages_V, 0.0, rtol=0.0, atol=1.0e-12))
    if zero_matches.size:
        zero_idx = int(zero_matches[0])
        solutions_by_idx[zero_idx] = sc_steady
        up_indices = list(range(zero_idx + 1, voltages_V.size))
        down_indices = list(range(zero_idx - 1, -1, -1))
    else:
        pos = np.flatnonzero(voltages_V > 0.0)
        up_indices = pos.tolist()
        down_indices = np.flatnonzero(voltages_V < 0.0)[::-1].tolist()

    def _advance_bias(state_in: State1D, v_from: float, v_to: float) -> SteadyState1D:
        dv = v_to - v_from
        if abs(dv) <= 1.0e-15:
            return solve_steady_state(
                compiled,
                initial_state=state_in,
                voltage_V=v_to,
                generation_scale=generation_scale,
                options=options,
            )
        try:
            return solve_steady_state(
                compiled,
                initial_state=state_in,
                voltage_V=v_to,
                generation_scale=generation_scale,
                options=options,
            )
        except SteadyStateError:
            if abs(dv) <= 1.0e-3:
                raise
            v_mid = 0.5 * (v_from + v_to)
            mid_sol = _advance_bias(state_in, v_from, v_mid)
            return _advance_bias(mid_sol.state, v_mid, v_to)

    def march(indices: list[int], anchor: SteadyState1D) -> None:
        state = anchor.state
        prev_v = anchor.voltage_V
        for idx in indices:
            target_v = float(voltages_V[idx])
            dv = target_v - prev_v
            sub_count = max(1, int(math.ceil(abs(dv) / options.max_bias_step_V)))
            cur_v = prev_v
            for step_i in range(1, sub_count + 1):
                v_step = prev_v + dv * (step_i / sub_count)
                sol = _advance_bias(state, cur_v, v_step)
                state = sol.state
                cur_v = v_step
                if step_i == sub_count:
                    solutions_by_idx[idx] = sol
            prev_v = target_v

    march(up_indices, sc_steady)
    march(down_indices, sc_steady)

    ordered = [solutions_by_idx[i] for i in range(voltages_V.size)]
    currents = np.array([ -polarity * float(sol.current_density_A_m2[0]) for sol in ordered ], dtype=float)
    currents.setflags(write=False)
    metrics = steady_jv_metrics(voltages_V, currents, incident_power_W_m2=incident_power_W_m2)
    return currents, metrics


def run_cbo_scan(
    baseline: PreparedDevice,
    config: CBOScanConfig,
) -> CBOScanResult:
    """Run a certified 1D CBO scan with adaptive bridge continuation."""
    pinned = enforce_pre_p08_cbo_pinning(
        baseline,
        requested_models=config.requested_models,
        allow_prospective_p08_models=config.allow_prospective_p08_models,
    )
    policy = validate_cbo_boundary_policy(config.boundary_policy)

    if config.voltage_grids_V is not None:
        v_grids = _validate_nested_voltage_grids(config.voltage_grids_V)
    elif config.voltages_V is not None:
        v_grids = _validate_nested_voltage_grids((config.voltages_V,))
    else:
        v_grids = _validate_nested_voltage_grids((np.linspace(0.0, 1.125, 25),))
    finest_v = v_grids[-1]

    requested_arr = np.array(sorted(set(float(v) for v in config.delta_ec_eV)), dtype=float)
    requested_arr.setflags(write=False)
    ref_delta = float(config.reference_delta_ec_eV)

    all_targets = sorted(set([*requested_arr.tolist(), ref_delta]))
    pos_targets = [d for d in all_targets if d > ref_delta + 1.0e-12]
    neg_targets = sorted([d for d in all_targets if d < ref_delta - 1.0e-12], reverse=True)

    trace: list[CBOShortCircuitSample] = []
    sc_by_key: dict[float, tuple[CompiledDevice1D, SteadyState1D]] = {}
    terminations: list[CBOScanTermination] = []
    bridge_count = 0

    def compile_at(delta: float) -> CompiledDevice1D:
        prep = prepare_cbo_point_device(baseline, delta, boundary_policy=policy)
        return compile_device_1d(
            prep,
            intervals_per_layer=config.intervals_per_layer,
            interface_topology=config.interface_topology,
            interface_transmission=config.interface_transmission,
        )

    def solve_sc(
        delta: float,
        anchor: SteadyState1D | None,
        *,
        requested: bool,
    ) -> tuple[CompiledDevice1D, SteadyState1D]:
        compiled = compile_at(delta)
        seed = None if anchor is None else _seed_state_on_compiled(compiled, anchor)
        try:
            sol = solve_steady_state(
                compiled,
                initial_state=seed,
                voltage_V=0.0,
                generation_scale=config.generation_scale,
                options=config.steady_options,
            )
        except SteadyStateError:
            if seed is None:
                raise
            sol = solve_steady_state(
                compiled,
                initial_state=None,
                voltage_V=0.0,
                generation_scale=config.generation_scale,
                options=config.steady_options,
            )
        polarity = float(compiled.voltage_polarity)
        all_curr = np.r_[
            sol.contact_current_density_A_m2[:1],
            sol.current_density_A_m2,
            sol.contact_current_density_A_m2[1:],
        ]
        trace.append(
            CBOShortCircuitSample(
                delta_ec_eV=delta,
                output_current_A_m2=float(-polarity * sol.current_density_A_m2[0]),
                requested=requested,
                face_current_spread_A_m2=float(np.ptp(all_curr)),
                applied_V_bi_V=float(compiled.built_in_potential_V),
                compiled_identity=compiled.identity,
            )
        )
        return compiled, sol

    ref_requested = bool(np.any(np.isclose(requested_arr, ref_delta, rtol=0.0, atol=1.0e-12)))
    ref_compiled, ref_sc = solve_sc(ref_delta, None, requested=ref_requested)
    sc_by_key[ref_delta] = (ref_compiled, ref_sc)

    def continue_branch(targets: list[float], direction: str) -> None:
        nonlocal bridge_count
        current_delta = ref_delta
        current_sc = ref_sc
        for target in targets:
            while not math.isclose(current_delta, target, rel_tol=0.0, abs_tol=1.0e-12):
                remaining = target - current_delta
                step = math.copysign(min(abs(remaining), config.maximum_delta_step_eV), remaining)
                candidate = target if abs(remaining) <= config.maximum_delta_step_eV + 1.0e-15 else current_delta + step
                is_req = math.isclose(candidate, target, rel_tol=0.0, abs_tol=1.0e-12) and bool(
                    np.any(np.isclose(requested_arr, target, rtol=0.0, atol=1.0e-12))
                )
                while True:
                    try:
                        comp, sol = solve_sc(candidate, current_sc, requested=is_req)
                        if not is_req:
                            bridge_count += 1
                        current_delta = candidate
                        current_sc = sol
                        if is_req:
                            sc_by_key[target] = (comp, sol)
                        break
                    except (SteadyStateError, CapabilityError, ValueError) as exc:
                        half = 0.5 * (candidate - current_delta)
                        if abs(half) < config.minimum_delta_step_eV or bridge_count >= config.max_bridge_points:
                            terminations.append(
                                CBOScanTermination(
                                    direction=direction,
                                    last_certified_delta_ec_eV=current_delta,
                                    first_failed_delta_ec_eV=candidate,
                                    requested_delta_ec_eV=target,
                                    reason=str(exc),
                                )
                            )
                            return
                        candidate = current_delta + half
                        is_req = False

    continue_branch(pos_targets, "positive_spike")
    continue_branch(neg_targets, "negative_cliff")

    points: list[CBOScanPoint] = []
    for delta in sorted(sc_by_key):
        comp, sc_sol = sc_by_key[delta]
        is_req = bool(np.any(np.isclose(requested_arr, delta, rtol=0.0, atol=1.0e-12)))
        polarity = float(comp.voltage_polarity)
        sc_out = float(-polarity * sc_sol.current_density_A_m2[0])
        all_curr = np.r_[
            sc_sol.contact_current_density_A_m2[:1],
            sc_sol.current_density_A_m2,
            sc_sol.contact_current_density_A_m2[1:],
        ]
        spread = float(np.ptp(all_curr))
        if config.calculate_jv_metrics:
            j_out, metrics = _solve_jv_from_short_circuit(
                comp,
                sc_sol,
                finest_v,
                generation_scale=config.generation_scale,
                incident_power_W_m2=config.incident_power_W_m2,
                options=config.steady_options,
            )
            v_arr = finest_v
        else:
            v_arr = np.array([0.0], dtype=float)
            v_arr.setflags(write=False)
            j_out = np.array([sc_out], dtype=float)
            j_out.setflags(write=False)
            metrics = None
        points.append(
            CBOScanPoint(
                delta_ec_eV=delta,
                requested=is_req,
                applied_V_bi_V=float(comp.built_in_potential_V),
                compiled_identity=comp.identity,
                short_circuit_output_A_m2=sc_out,
                face_current_spread_A_m2=spread,
                voltage_V=v_arr,
                output_current_A_m2=j_out,
                metrics=metrics,
                certified=bool(sc_sol.converged and spread <= config.steady_options.current_spread_tolerance_A_m2),
            )
        )

    intervals: list[CBOCriticalInterval] = []
    jsc_samples = [
        (
            p.delta_ec_eV,
            p.short_circuit_output_A_m2
            if p.metrics is None or p.metrics.short_circuit_current_A_m2 is None
            else p.metrics.short_circuit_current_A_m2,
        )
        for p in points
    ]
    intervals.append(
        _critical_interval_for_metric(
            jsc_samples,
            metric="Jsc",
            reference_delta_ec_eV=ref_delta,
            relative_drop_fraction=config.relative_drop_fraction,
        )
    )
    if config.calculate_jv_metrics:
        for metric_name, attr in (
            ("Voc", "open_circuit_voltage_V"),
            ("FF", "fill_factor"),
            ("PCE", "efficiency"),
        ):
            m_samples = [
                (p.delta_ec_eV, float(getattr(p.metrics, attr)))
                for p in points
                if p.metrics is not None and getattr(p.metrics, attr) is not None
            ]
            intervals.append(
                _critical_interval_for_metric(
                    m_samples,
                    metric=metric_name,
                    reference_delta_ec_eV=ref_delta,
                    relative_drop_fraction=config.relative_drop_fraction,
                )
            )

    protocol_payload = {
        "schema": "solarlab_research.cbo_scan_protocol.v1",
        "baseline_sha256": baseline.content_sha256,
        "boundary_policy": policy,
        "interface_topology": config.interface_topology,
        "interface_transmission": config.interface_transmission,
        "reference_delta_ec_eV": ref_delta,
        "relative_drop_fraction": config.relative_drop_fraction,
        "pinned_models": [{"id": m.id, "version": m.version} for m in pinned],
    }
    protocol_sha = hashlib.sha256(
        json.dumps(protocol_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return CBOScanResult(
        requested_delta_ec_eV=requested_arr,
        points=tuple(p for p in points if p.requested),
        short_circuit_trace=tuple(trace),
        critical_intervals=tuple(intervals),
        terminations=tuple(terminations),
        reference_delta_ec_eV=ref_delta,
        relative_drop_fraction=config.relative_drop_fraction,
        boundary_policy=policy,
        interface_topology=config.interface_topology,
        interface_transmission=config.interface_transmission,
        grid_node_count=ref_compiled.node_count,
        grid_interval_count=ref_compiled.node_count - 1,
        bridge_point_count=bridge_count,
        pinned_models=pinned,
        voltage_grids_V=v_grids if config.calculate_jv_metrics else (),
        voltage_grid_policy=config.voltage_grid_policy,
        protocol_sha256=protocol_sha,
    )


def certify_cbo_voltage_grid_convergence(
    result: CBOScanResult,
    *,
    incident_power_W_m2: float = 1000.0,
) -> CBOVoltageGridConvergenceCertificate:
    """Certify nested voltage-grid convergence of Voc, FF, and PCE across CBO points."""
    policy = result.voltage_grid_policy
    reasons: list[str] = []
    counts = tuple(int(g.size) for g in result.voltage_grids_V)
    if len(result.voltage_grids_V) < policy.minimum_voltage_grids:
        reasons.append(
            f"requires at least {policy.minimum_voltage_grids} nested voltage grids; got {len(result.voltage_grids_V)}"
        )

    point_certs: list[CBOVoltageGridPointCertificate] = []
    finest = result.voltage_grids_V[-1] if result.voltage_grids_V else np.empty(0)
    for pt in result.points:
        pt_reasons: list[str] = []
        vocs: list[float] = []
        ffs: list[float] = []
        pces: list[float] = []
        for grid in result.voltage_grids_V:
            mask = np.isin(np.round(finest, 12), np.round(grid, 12))
            sub_v = pt.voltage_V[mask]
            sub_j = pt.output_current_A_m2[mask]
            m = steady_jv_metrics(sub_v, sub_j, incident_power_W_m2=incident_power_W_m2)
            if m.open_circuit_voltage_V is None or m.fill_factor is None or m.efficiency is None:
                pt_reasons.append(f"grid {grid.size} failed to bracket Voc/FF/PCE ({m.limitation})")
            else:
                vocs.append(m.open_circuit_voltage_V)
                ffs.append(m.fill_factor)
                pces.append(m.efficiency)

        d_voc = abs(vocs[-1] - vocs[-2]) if len(vocs) >= 2 else None
        d_ff = abs(ffs[-1] - ffs[-2]) if len(ffs) >= 2 else None
        d_pce = abs(pces[-1] - pces[-2]) if len(pces) >= 2 else None
        if d_voc is None or d_voc > policy.maximum_voc_change_V:
            pt_reasons.append(f"Voc change {d_voc} exceeds {policy.maximum_voc_change_V}")
        if d_ff is None or d_ff > policy.maximum_ff_change:
            pt_reasons.append(f"FF change {d_ff} exceeds {policy.maximum_ff_change}")
        if d_pce is None or d_pce > policy.maximum_pce_change:
            pt_reasons.append(f"PCE change {d_pce} exceeds {policy.maximum_pce_change}")

        pt_ok = not pt_reasons and len(vocs) == len(result.voltage_grids_V)
        if not pt_ok:
            reasons.append(f"CBO {pt.delta_ec_eV:+.4f} eV failed voltage-grid convergence")
        point_certs.append(
            CBOVoltageGridPointCertificate(
                delta_ec_eV=pt.delta_ec_eV,
                voltage_point_counts=counts,
                voc_values_V=tuple(vocs),
                ff_values=tuple(ffs),
                pce_values=tuple(pces),
                final_voc_change_V=d_voc,
                final_ff_change=d_ff,
                final_pce_change=d_pce,
                certified=pt_ok,
                reasons=tuple(pt_reasons),
            )
        )

    return CBOVoltageGridConvergenceCertificate(
        voltage_point_counts=counts,
        minimum_voltage_grids=policy.minimum_voltage_grids,
        maximum_voc_change_V=policy.maximum_voc_change_V,
        maximum_ff_change=policy.maximum_ff_change,
        maximum_pce_change=policy.maximum_pce_change,
        points=tuple(point_certs),
        certified=bool(point_certs and not reasons),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def certify_cbo_grid_convergence(
    scans: Sequence[CBOScanResult],
    *,
    metric: str = "FF",
    minimum_grids: int = 2,
    maximum_envelope_width_eV: float = 0.15,
    maximum_reference_relative_spread: float = 0.02,
) -> CBOGridConvergenceCertificate:
    """Certify spatial grid convergence of critical CBO onset across meshes."""
    ordered = sorted(scans, key=lambda s: s.grid_interval_count)
    reasons: list[str] = []
    if len(ordered) < minimum_grids:
        reasons.append(f"requires at least {minimum_grids} spatial grids; got {len(ordered)}")
    counts = tuple(s.grid_interval_count for s in ordered)
    if len(set(counts)) != len(counts):
        reasons.append("spatial grid interval counts must be unique")
    if ordered and len({s.protocol_sha256 for s in ordered}) != 1:
        reasons.append("spatial grid scans do not share identical physical protocol_sha256")
    for s in ordered:
        if not s.complete or not s.certified:
            reasons.append(f"grid {s.grid_interval_count} is incomplete or uncertified")

    intervals: list[tuple[float, float]] = []
    ref_vals: list[float] = []
    for s in ordered:
        ci = next((c for c in s.critical_intervals if c.metric == metric), None)
        if ci is None or not ci.resolved or ci.lower_delta_ec_eV is None or ci.upper_delta_ec_eV is None:
            reasons.append(f"grid {s.grid_interval_count} did not resolve {metric} critical interval")
        else:
            intervals.append((ci.lower_delta_ec_eV, ci.upper_delta_ec_eV))
        if ci is not None and ci.reference_value is not None:
            ref_vals.append(ci.reference_value)

    env_lo = min(lo for lo, _ in intervals) if intervals else None
    env_hi = max(hi for _, hi in intervals) if intervals else None
    env_w = (env_hi - env_lo) if (env_lo is not None and env_hi is not None) else None
    if env_w is None or env_w > maximum_envelope_width_eV:
        reasons.append(f"critical-interval envelope width {env_w} exceeds {maximum_envelope_width_eV}")

    ref_spread: float | None = None
    if ref_vals and min(ref_vals) > 0.0:
        ref_spread = (max(ref_vals) - min(ref_vals)) / max(ref_vals)
        if ref_spread > maximum_reference_relative_spread:
            reasons.append(f"reference {metric} spread {ref_spread} exceeds {maximum_reference_relative_spread}")
    else:
        reasons.append(f"missing positive reference {metric} values")

    return CBOGridConvergenceCertificate(
        metric=metric,
        grid_interval_counts=counts,
        critical_intervals_eV=tuple(intervals),
        envelope_lower_eV=env_lo,
        envelope_upper_eV=env_hi,
        envelope_width_eV=env_w,
        maximum_envelope_width_eV=maximum_envelope_width_eV,
        reference_values=tuple(ref_vals),
        reference_relative_spread=ref_spread,
        maximum_reference_relative_spread=maximum_reference_relative_spread,
        certified=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
    )
