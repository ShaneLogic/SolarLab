"""SolarScale MaterialRecord screening, sweep expansion, and native 1D evaluation.

Ports the SolarScale -> SolarLab material-screening policy onto ``solarlab``'s
canonical configuration, device compilation, and steady 1D solver stack with:
- strict provenance-backed DFT/MD property filtering (rejecting ``missing`` or
  ``swept`` provenance as fixed absorber inputs),
- deterministic import plans, generated YAML manifests, and multi-axis sweep
  manifest expansion with canonical SHA-256 digests,
- native 1D execution via :func:`import_standard_device`, :func:`resolve_device`,
  :func:`compile_device_1d` (with exact defect-free interface SRV local
  tangents), and :func:`solve_steady_state` (or :func:`compile_ionic_device_1d`
  + :func:`solve_ionic_steady_state` for single-domain ionic stacks),
- explicit per-candidate failure retention without silent physics fallbacks.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
import copy
import csv
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
import yaml

from solarlab.config.device_import import import_standard_device
from solarlab.config.resolve_device import resolve_device
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.device.compile_1d import CapabilityError, CompiledDevice1D, compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import State1D
from solarlab.equations.ionic_1d import IonicState1D
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument
from solarlab.physics.recombination.bulk import RecombinationRate
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions, solve_ionic_steady_state
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    SteadyState1D,
    SteadyStateError,
    solve_steady_state,
)

BOLTZMANN_CONSTANT_J_K = 1.380649e-23

__all__ = [
    "CandidatePlan",
    "DEFAULT_SWEEP_GRID",
    "DEFAULT_SWEEP_POLICY",
    "DEVICE_SWEEP_PROPERTY_MAP",
    "DefectFreeInterfaceSRVModel1D",
    "ImportPolicy",
    "MAPPED_PROPERTY_SPECS",
    "MaterialProperty",
    "MaterialRecord",
    "OPTIONAL_PHYSICAL_INPUTS",
    "PropertyProvenance",
    "REQUIRED_PHYSICAL_INPUTS",
    "SCORE_FIELDS",
    "SWEEP_DIMENSION_ORDER",
    "SWEEP_POLICIES",
    "SolarScaleJVMetrics",
    "SolarScaleJVResult",
    "SweepPolicyName",
    "advance_electronic_steady_1d",
    "compile_solarscale_electronic_1d",
    "compute_solarscale_jv_metrics",
    "expand_sweep_manifest",
    "generate_solarlab_inputs",
    "load_material_records",
    "manifest_sha256",
    "parse_material_records",
    "plan_solarlab_import",
    "prepare_solarscale_device",
    "run_smoke_device_results",
    "run_smoke_jv",
    "solarscale_default_catalog",
    "solve_solarscale_jv_1d",
    "write_device_results",
]

ImportPolicy = Literal["production", "exploratory"]
SweepPolicyName = Literal["quick", "exploratory", "production"]
SWEEP_DIMENSION_ORDER = (
    "absorber.thickness",
    "absorber.tau_n",
    "absorber.tau_p",
    "absorber.trap_N_t_bulk",
    "device.interfaces",
)

_J_ZERO_FRACTION_OF_JSC = 1.0e-3
_VOLATILE_MANIFEST_KEYS = frozenset({"timestamp", "git_commit"})


@dataclass(frozen=True)
class PropertyProvenance:
    kind: str = "missing"
    source: str = "not_available"
    notes: str = ""


@dataclass(frozen=True)
class MaterialProperty:
    value: Any
    unit: str = ""
    provenance: PropertyProvenance = field(default_factory=PropertyProvenance)


@dataclass(frozen=True)
class MaterialRecord:
    material_id: str
    formula: str | None = None
    schema_version: str | None = None
    properties: dict[str, MaterialProperty] = field(default_factory=dict)
    screening: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CandidatePlan:
    material_id: str
    rank: int
    selected: bool
    reason: str
    readiness: str
    ranking_score: float | None
    ranking_score_source: str | None
    ml_pv_score: float | None
    final_fom_score: float | None
    screening_evidence: dict[str, Any]
    mapped_parameters: dict[str, Any]
    material_metadata: dict[str, Any]
    sweep_parameters: dict[str, Any]
    missing_required: list[str]
    missing_optional: list[str]
    assumed_parameters: dict[str, Any]
    diagnostics: list[str]
    config: dict[str, Any] | None = None
    config_path: str | None = None


@dataclass(frozen=True)
class SolarScaleJVMetrics:
    """Photovoltaic metrics in the J_sc-positive 1D convention."""

    V_oc: float
    J_sc: float
    FF: float
    PCE: float
    voc_bracketed: bool


@dataclass(frozen=True)
class SolarScaleJVResult:
    """Forward/reverse 1D steady screening J-V result."""

    V_fwd: NDArray[np.float64]
    J_fwd: NDArray[np.float64]
    V_rev: NDArray[np.float64]
    J_rev: NDArray[np.float64]
    metrics_fwd: SolarScaleJVMetrics
    metrics_rev: SolarScaleJVMetrics
    hysteresis_index: float
    compiled_identity: str
    scientific_qualification: bool = False


SWEEP_POLICIES: dict[SweepPolicyName, dict[str, list[float]]] = {
    "quick": {
        "absorber.thickness": [300.0e-9],
        "absorber.tau_n": [1.0e-7],
        "absorber.tau_p": [1.0e-7],
        "absorber.trap_N_t_bulk": [1.0e22],
        "device.interfaces": [1.0],
    },
    "exploratory": {
        "absorber.thickness": [300.0e-9, 500.0e-9, 800.0e-9],
        "absorber.tau_n": [1.0e-9, 1.0e-7, 1.0e-6],
        "absorber.tau_p": [1.0e-9, 1.0e-7, 1.0e-6],
        "absorber.trap_N_t_bulk": [1.0e20, 1.0e22, 1.0e24],
        "device.interfaces": [0.0, 1.0, 10.0],
    },
    "production": {
        "absorber.thickness": [400.0e-9, 600.0e-9],
        "absorber.tau_n": [1.0e-8, 1.0e-7],
        "absorber.tau_p": [1.0e-8, 1.0e-7],
        "absorber.trap_N_t_bulk": [1.0e21, 1.0e22],
        "device.interfaces": [0.1, 1.0],
    },
}
DEFAULT_SWEEP_POLICY: SweepPolicyName = "quick"
DEFAULT_SWEEP_GRID = SWEEP_POLICIES[DEFAULT_SWEEP_POLICY]

DEVICE_SWEEP_PROPERTY_MAP: dict[str, str] = {
    "absorber_thickness_m": "absorber.thickness",
    "carrier_lifetime_s": "absorber.tau_n/tau_p",
    "trap_density_m3": "absorber.trap_N_t_bulk",
    "surface_recombination_velocity_m_s": "device.interfaces",
    "contact_work_function_ev": "device.contact_work_function",
}

MAPPED_PROPERTY_SPECS: dict[str, tuple[str, str | None, float]] = {
    "dielectric_static_avg": ("absorber.eps_r", "eps_r", 1.0),
    "electron_affinity_ev": ("absorber.chi", "chi", 1.0),
    "electron_mobility_cm2_v_s": ("absorber.mu_n", "mu_n", 1.0e-4),
    "hole_mobility_cm2_v_s": ("absorber.mu_p", "mu_p", 1.0e-4),
    "ion_diffusion_coefficient_m2_s": ("absorber.D_ion", "D_ion", 1.0),
    "ion_activation_energy_ev": ("absorber.E_a_ion", "E_a_ion", 1.0),
}

REQUIRED_PHYSICAL_INPUTS = (
    "band_gap_hse_ev",
    "dielectric_static_avg",
)

OPTIONAL_PHYSICAL_INPUTS = (
    "electron_affinity_ev",
    "electron_mobility_cm2_v_s",
    "hole_mobility_cm2_v_s",
    "ion_diffusion_coefficient_m2_s",
    "ion_activation_energy_ev",
)

SCORE_FIELDS = ("final_fom_score", "ml_pv_score")

_ALLOWED_READINESS: dict[ImportPolicy, set[str]] = {
    "production": {"promising"},
    "exploratory": {"phonon", "promising"},
}

_READY_FLAG_BY_POLICY: dict[ImportPolicy, str] = {
    "production": "solarlab_production_ready",
    "exploratory": "solarlab_provisional_ready",
}


@dataclass
class DefectFreeInterfaceSRVModel1D:
    """Exact node-local 1D recombination model for defect-free ``[v_n, v_p]`` interfaces."""

    interface_nodes: tuple[int, ...]
    electron_velocities_m_s: tuple[float, ...]
    hole_velocities_m_s: tuple[float, ...]
    ni_sq_m6: tuple[float, ...]
    n1_m3: tuple[float, ...]
    p1_m3: tuple[float, ...]
    cell_widths_m: tuple[float, ...]
    node_count: int
    model_id: str

    @classmethod
    def from_interfaces(
        cls,
        base_device: CompiledDevice1D,
        electrical_srv_pairs: Sequence[tuple[float, float]],
    ) -> DefectFreeInterfaceSRVModel1D | None:
        if not electrical_srv_pairs:
            return None
        thicknesses = [
            float(layer.thickness_m) for layer in base_device.parameters.layers
        ]
        if len(electrical_srv_pairs) != max(0, len(thicknesses) - 1):
            raise ValueError("electrical_srv_pairs must align with internal electrical interfaces")
        nodes = base_device.geometry.nodes_m
        boundaries: NDArray[np.float64] = np.cumsum(np.asarray(thicknesses, dtype=np.float64))[:-1]
        coeffs = base_device.recombination.coefficients
        active_nodes: list[int] = []
        v_n_list: list[float] = []
        v_p_list: list[float] = []
        ni_sq_list: list[float] = []
        n1_list: list[float] = []
        p1_list: list[float] = []
        dx_list: list[float] = []
        for (v_n, v_p), x_iface in zip(electrical_srv_pairs, boundaries, strict=True):
            if v_n < 0.0 or v_p < 0.0 or not (math.isfinite(v_n) and math.isfinite(v_p)):
                raise CapabilityError("interface SRV must be finite and non-negative")
            if v_n == 0.0 and v_p == 0.0:
                continue
            if v_n <= 0.0 or v_p <= 0.0:
                # One zero velocity yields infinite denominator => zero SRH rate
                continue
            idx = int(np.argmin(np.abs(nodes - float(x_iface))))
            active_nodes.append(idx)
            v_n_list.append(float(v_n))
            v_p_list.append(float(v_p))
            ni_sq_list.append(float(coeffs.ni_sq_m6[idx]))
            n1_list.append(float(coeffs.n1_m3[idx]))
            p1_list.append(float(coeffs.p1_m3[idx]))
            dx_list.append(float(base_device.geometry.cell_measures_m[idx]))
        if not active_nodes:
            return None
        payload = json.dumps(
            {
                "kind": "defect_free_interface_srv_1d_v1",
                "nodes": active_nodes,
                "v_n": v_n_list,
                "v_p": v_p_list,
                "ni_sq": ni_sq_list,
                "n1": n1_list,
                "p1": p1_list,
                "dx": dx_list,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return cls(
            interface_nodes=tuple(active_nodes),
            electron_velocities_m_s=tuple(v_n_list),
            hole_velocities_m_s=tuple(v_p_list),
            ni_sq_m6=tuple(ni_sq_list),
            n1_m3=tuple(n1_list),
            p1_m3=tuple(p1_list),
            cell_widths_m=tuple(dx_list),
            node_count=int(nodes.size),
            model_id=f"defect_free_interface_srv_1d_v1:{digest}",
        )

    def evaluate(self, n: NDArray[np.float64], p: NDArray[np.float64]) -> RecombinationRate:
        rate: NDArray[np.float64] = np.zeros(self.node_count, dtype=np.float64)
        dn: NDArray[np.float64] = np.zeros(self.node_count, dtype=np.float64)
        dp: NDArray[np.float64] = np.zeros(self.node_count, dtype=np.float64)
        for idx, v_n, v_p, ni_sq, n1, p1, dx in zip(
            self.interface_nodes,
            self.electron_velocities_m_s,
            self.hole_velocities_m_s,
            self.ni_sq_m6,
            self.n1_m3,
            self.p1_m3,
            self.cell_widths_m,
            strict=True,
        ):
            n_i = float(n[idx])
            p_i = float(p[idx])
            denom = (n_i + n1) / v_p + (p_i + p1) / v_n
            numer = n_i * p_i - ni_sq
            inv_denom = 1.0 / denom
            r_surf = numer * inv_denom
            dr_dn = (p_i * denom - numer / v_p) * inv_denom * inv_denom
            dr_dp = (n_i * denom - numer / v_n) * inv_denom * inv_denom
            inv_dx = 1.0 / dx
            rate[idx] += r_surf * inv_dx
            dn[idx] += dr_dn * inv_dx
            dp[idx] += dr_dp * inv_dx
        return RecombinationRate(rate=rate, d_rate_dn=dn, d_rate_dp=dp)


def solarscale_default_catalog() -> DefaultCatalog:
    """Return the explicit SI ``DefaultCatalog`` for SolarScale 1D screening."""
    vt_300 = BOLTZMANN_CONSTANT_J_K * 300.0 / ELEMENTARY_CHARGE_C
    ni_default = 1.0e25 * math.exp(-1.6 / (2.0 * vt_300))
    catalog_doc: dict[str, Any] = {
        "schema": "solarlab.default-catalog.v1",
        "material": {
            "eps_r": 25.0,
            "mu_n": 2.0e-4,
            "mu_p": 2.0e-4,
            "D_ion": 0.0,
            "P_lim": 1.0e28,
            "P0": 0.0,
            "ni": ni_default,
            "tau_n": 1.0e-6,
            "tau_p": 1.0e-6,
            "n1": ni_default,
            "p1": ni_default,
            "B_rad": 1.0e-18,
            "C_n": 1.0e-42,
            "C_p": 1.0e-42,
            "alpha": 2.0e6,
            "N_A": 0.0,
            "N_D": 0.0,
            "chi": 0.0,
            "Eg": 0.0,
            "A_star_n": 1.2017e6,
            "A_star_p": 1.2017e6,
            "v_th": 1.0e5,
            "D_ion_neg": 0.0,
            "P0_neg": 0.0,
            "P_lim_neg": 1.0e30,
            "Nc300": 1.0e25,
            "Nv300": 1.0e25,
            "carrier_statistics": "maxwell_boltzmann",
            "dopant_ionization_model": "fully_ionized",
            "donor_binding_energy_eV": None,
            "acceptor_binding_energy_eV": None,
            "donor_degeneracy": 2.0,
            "acceptor_degeneracy": 4.0,
            "band_gap_narrowing_model": "off",
            "bgn_reference_energy_eV": 0.009,
            "bgn_reference_density_m3": 1.0e23,
            "bgn_log_shape": 0.5,
            "bgn_conduction_band_fraction": 0.5,
            "mu_T_gamma": -1.5,
            "E_a_ion": 0.58,
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
            "ct_beta_n": 2.0,
            "ct_beta_p": 2.0,
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
            "Phi": 2.5e21,
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
            "P_lim": 1.0e28,
            "P0": 0.0,
            "B_rad": 1.0e-18,
            "C_n": 1.0e-42,
            "C_p": 1.0e-42,
            "alpha": 2.0e6,
            "tau_n": 1.0e-6,
            "tau_p": 1.0e-6,
        },
        "interface": {
            "calibration_factor": 1.0,
            "iface_state_calibration_factor": 1.0,
        },
        "constants": {
            "Q": ELEMENTARY_CHARGE_C,
            "K_B": BOLTZMANN_CONSTANT_J_K,
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
    evidence_sha = hashlib.sha256(
        json.dumps(catalog_doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    catalog_doc["evidence"] = [["solarscale_screening_defaults_v1", evidence_sha]]
    return DefaultCatalog.from_mapping(catalog_doc)


def manifest_sha256(manifest: Mapping[str, Any], *, ignore_volatile: bool = True) -> str:
    """Compute a canonical SHA-256 digest for a SolarScale plan, sweep, or result manifest."""

    def _strip(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return {
                str(k): _strip(v)
                for k, v in obj.items()
                if not (ignore_volatile and str(k) in _VOLATILE_MANIFEST_KEYS)
            }
        if isinstance(obj, list):
            return [_strip(item) for item in obj]
        if isinstance(obj, tuple):
            return [_strip(item) for item in obj]
        return obj

    cleaned = _strip(manifest)
    canonical = json.dumps(cleaned, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_solarscale_device(
    config_or_path: str | Path | Mapping[str, Any],
    *,
    defaults: DefaultCatalog | None = None,
    resources: ResourceLibrary | None = None,
) -> tuple[PreparedDevice, tuple[tuple[float, float], ...]]:
    """Resolve a SolarScale YAML config into a ``PreparedDevice`` and defect-free SRVs.

    Strips only SolarScale envelope metadata (``source``, ``notes``) and inactive
    sweep placeholders (``trap_N_t_bulk`` when ``trap_N_t_interface`` and
    ``trap_decay_length`` are both absent; ``D_ion`` when ``P0 == 0``) while
    binding the original YAML bytes as an immutable ``SourceDocument``.
    """
    catalog = solarscale_default_catalog() if defaults is None else defaults
    res_lib = ResourceLibrary() if resources is None else resources
    if isinstance(config_or_path, (str, Path)):
        cfg_path = Path(config_or_path)
        raw_bytes = cfg_path.read_bytes()
        raw_doc = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
        source_id = f"solarscale:{cfg_path.name}"
    elif isinstance(config_or_path, Mapping):
        raw_doc = copy.deepcopy(dict(config_or_path))
        raw_bytes = yaml.safe_dump(raw_doc, sort_keys=False).encode("utf-8")
        source_id = "solarscale:inline_config"
    else:
        raise TypeError("config_or_path must be a path or mapping")

    if not isinstance(raw_doc, Mapping) or "device" not in raw_doc or "layers" not in raw_doc:
        raise ValueError("SolarScale config must contain 'device' and 'layers'")

    doc = copy.deepcopy(dict(raw_doc))
    source_meta = doc.pop("source", None)
    doc.pop("notes", None)
    sim_hints = doc.get("simulation_hints")
    if isinstance(sim_hints, Mapping):
        cleaned_hints = {
            k: v
            for k, v in sim_hints.items()
            if k not in {"N_grid", "n_points", "v_rate", "V_max", "notes"}
        }
        if cleaned_hints:
            doc["simulation_hints"] = cleaned_hints
        else:
            doc.pop("simulation_hints", None)

    layers = doc.get("layers")
    if not isinstance(layers, list):
        raise ValueError("SolarScale config layers must be a list")

    elec_indices = [
        i for i, layer in enumerate(layers) if isinstance(layer, Mapping) and layer.get("role") != "substrate"
    ]
    dev_block = doc.get("device")
    optical_model = (
        str(dev_block.get("optical_model") or "beer_lambert")
        if isinstance(dev_block, Mapping)
        else "beer_lambert"
    )
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        # Under Beer-Lambert optics without supplied n/k tables, optical_material is
        # inactive provenance metadata (preserved verbatim in raw_source).
        if optical_model == "beer_lambert" and resources is None:
            layer.pop("optical_material", None)
        # When spatial trap profile is not configured (interface + decay length absent),
        # trap_N_t_bulk is inactive sweep metadata in SolarScale templates.
        if layer.get("trap_N_t_interface") is None and layer.get("trap_decay_length") is None:
            layer.pop("trap_N_t_bulk", None)
            layer.pop("trap_profile_shape", None)
        # On ion-free layers (P0 == 0), D_ion has zero physical population to transport.
        p0_val = float(layer.get("P0", 0.0) or 0.0)
        p0_neg_val = float(layer.get("P0_neg", 0.0) or 0.0)
        if p0_val == 0.0 and p0_neg_val == 0.0:
            if "D_ion" in layer and float(layer["D_ion"] or 0.0) != 0.0:
                layer["D_ion"] = 0.0

    electrical_srvs: list[tuple[float, float]] = []
    if isinstance(dev_block, dict):
        has_defects = bool(dev_block.get("interface_defects"))
        raw_ifaces = dev_block.get("interfaces")
        if not has_defects and isinstance(raw_ifaces, list) and len(elec_indices) >= 2:
            # Extract defect-free electrical interface SRVs aligned with full-stack slots
            first_elec = elec_indices[0]
            n_elec_ifaces = len(elec_indices) - 1
            all_pairs: list[tuple[float, float]] = []
            for k in range(max(0, len(layers) - 1)):
                if k < len(raw_ifaces) and isinstance(raw_ifaces[k], (list, tuple)) and len(raw_ifaces[k]) == 2:
                    all_pairs.append((float(raw_ifaces[k][0]), float(raw_ifaces[k][1])))
                else:
                    all_pairs.append((0.0, 0.0))
            electrical_srvs = all_pairs[first_elec : first_elec + n_elec_ifaces]
            dev_block["interfaces"] = [[0.0, 0.0] for _ in range(max(0, len(layers) - 1))]

    mat_id = (
        str(source_meta.get("material_id"))
        if isinstance(source_meta, Mapping) and source_meta.get("material_id")
        else "solarscale_device"
    )
    dev_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", mat_id).strip("_") or "solarscale_device"
    normalized_yaml = yaml.safe_dump(doc, sort_keys=False).encode("utf-8")
    norm_source = SourceDocument(f"{source_id}:standard", normalized_yaml)
    raw_source = SourceDocument(source_id, raw_bytes)
    device_input = import_standard_device(norm_source, id=dev_id, defaults=catalog)
    prepared = resolve_device(
        device_input,
        catalog,
        res_lib,
        sources=(norm_source, raw_source),
    )
    return prepared, tuple(electrical_srvs)


def compute_solarscale_jv_metrics(
    voltages_V: NDArray[np.float64],
    output_current_A_m2: NDArray[np.float64],
    *,
    incident_power_W_m2: float = 1000.0,
    voc_ceiling_V: float | None = None,
) -> SolarScaleJVMetrics:
    """Compute sampled ``(V_oc, J_sc, FF, PCE, voc_bracketed)`` metrics."""
    v = np.asarray(voltages_V, dtype=np.float64)
    j = np.asarray(output_current_A_m2, dtype=np.float64)
    if v.ndim != 1 or j.ndim != 1 or v.shape != j.shape or v.size < 2:
        raise ValueError("V and J must be 1D arrays of equal length >= 2")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(j))):
        raise ValueError("J-V metrics require finite voltage and current samples")
    order = np.argsort(v)
    v_s = v[order]
    j_s = j[order]
    j_sc = float(np.interp(0.0, v_s, j_s))
    j_tol = _J_ZERO_FRACTION_OF_JSC * abs(j_sc)
    crossings = np.where((j_s[:-1] > j_tol) & (j_s[1:] <= j_tol))[0]
    if crossings.size == 0:
        return SolarScaleJVMetrics(V_oc=0.0, J_sc=j_sc, FF=0.0, PCE=0.0, voc_bracketed=False)
    idx = int(crossings[0])
    dv = float(v_s[idx + 1] - v_s[idx])
    dj = float(j_s[idx + 1] - j_s[idx])
    v_oc = float(v_s[idx] - j_s[idx] * dv / dj) if dj != 0.0 else float(v_s[idx])
    v_oc = min(max(v_oc, float(v_s[idx])), float(v_s[idx + 1]))
    if voc_ceiling_V is not None and v_oc > voc_ceiling_V:
        return SolarScaleJVMetrics(V_oc=0.0, J_sc=j_sc, FF=0.0, PCE=0.0, voc_bracketed=False)
    mask = (v_s >= 0.0) & (v_s <= v_oc)
    op_power = v_s[mask] * j_s[mask]
    p_mpp = float(np.max(op_power)) if op_power.size else 0.0
    ff = p_mpp / (v_oc * j_sc) if (v_oc * j_sc) > 0.0 else 0.0
    pce = p_mpp / incident_power_W_m2 if incident_power_W_m2 > 0.0 else 0.0
    return SolarScaleJVMetrics(V_oc=v_oc, J_sc=j_sc, FF=ff, PCE=pce, voc_bracketed=True)


def _intervals_for_n_grid(prepared: PreparedDevice, n_grid: int) -> int | tuple[int, ...]:
    elec_layers = [layer for layer in prepared.layers if layer.role != "substrate"]
    n_elec = max(1, len(elec_layers))
    data = prepared.to_mapping()
    if data["electrical_grid"]:
        return max(2, int(round(float(n_grid) / n_elec)))
    per_layer = max(2, int(n_grid) // n_elec)
    return (per_layer,) * n_elec


def compile_solarscale_electronic_1d(
    config_or_path: str | Path | Mapping[str, Any],
    *,
    N_grid: int = 20,
    defaults: DefaultCatalog | None = None,
    resources: ResourceLibrary | None = None,
) -> tuple[PreparedDevice, CompiledDevice1D]:
    """Compile an ion-free 1D SolarScale config into a ``CompiledDevice1D``."""
    prepared, electrical_srvs = prepare_solarscale_device(
        config_or_path, defaults=defaults, resources=resources
    )
    elec_layers = [layer for layer in prepared.layers if layer.role != "substrate"]
    ion_layers = [
        layer.id
        for layer in elec_layers
        if float(dict(layer.parameters)["P0"] or 0.0) > 0.0
        or float(dict(layer.parameters)["P0_neg"] or 0.0) > 0.0
    ]
    if ion_layers:
        raise CapabilityError(
            "electronic_only_1d_refused_mobile_ions: config declares mobile ions and cannot use carrier-only 1D compile"
        )
    intervals = _intervals_for_n_grid(prepared, int(N_grid))
    base_dev = compile_device_1d(prepared, intervals_per_layer=intervals)
    srv_model = DefectFreeInterfaceSRVModel1D.from_interfaces(base_dev, electrical_srvs)
    device = (
        compile_device_1d(
            prepared,
            intervals_per_layer=intervals,
            additional_recombination=srv_model,
        )
        if srv_model is not None
        else base_dev
    )
    return prepared, device


def advance_electronic_steady_1d(
    device: CompiledDevice1D,
    seed: State1D,
    v_target: float,
    *,
    generation_scale: float,
    options: SteadyOptions,
    initial_step_V: float = 0.025,
    min_step_V: float = 0.0125,
) -> SteadyState1D:
    """Advance an electronic 1D steady state to ``v_target`` with adaptive bias halving."""
    cur_state = seed
    cur_v = float(seed.voltage_V)
    active_opts = options
    if cur_v == v_target:
        return solve_steady_state(
            device,
            initial_state=cur_state,
            voltage_V=v_target,
            generation_scale=generation_scale,
            options=active_opts,
        )
    step = min(initial_step_V, abs(v_target - cur_v))
    last_sol: SteadyState1D | None = None
    while True:
        rem = v_target - cur_v
        if abs(rem) <= 1.0e-14:
            assert last_sol is not None
            return last_sol
        d_try = math.copysign(min(step, abs(rem)), rem)
        v_try = v_target if abs(d_try) >= abs(rem) - 1.0e-14 else cur_v + d_try
        try:
            sol = solve_steady_state(
                device,
                initial_state=cur_state,
                voltage_V=v_try,
                generation_scale=generation_scale,
                options=active_opts,
            )
        except SteadyStateError:
            if step > min_step_V:
                step = max(min_step_V, 0.5 * step)
                continue
            if v_try > 1.0 and active_opts.continuity_tolerance_A_m2 < 1.0e-5:
                active_opts = SteadyOptions(
                    poisson_tolerance=max(active_opts.poisson_tolerance, 1.0e-9),
                    continuity_tolerance_A_m2=1.0e-5,
                    current_spread_tolerance_A_m2=1.0e-5,
                    max_iterations=active_opts.max_iterations,
                    max_bias_step_V=active_opts.max_bias_step_V,
                )
                sol = solve_steady_state(
                    device,
                    initial_state=cur_state,
                    voltage_V=v_try,
                    generation_scale=generation_scale,
                    options=active_opts,
                )
                step = initial_step_V
            else:
                raise
        cur_state = sol.state
        cur_v = v_try
        last_sol = sol
        if v_try == v_target:
            return sol
        step = min(initial_step_V, step * 1.5)


def solve_solarscale_jv_1d(
    config_or_path: str | Path | Mapping[str, Any],
    *,
    N_grid: int = 12,
    n_points: int = 4,
    v_rate: float = 5.0,
    V_max: float = 0.2,
    defaults: DefaultCatalog | None = None,
    resources: ResourceLibrary | None = None,
) -> SolarScaleJVResult:
    """Compile and solve a 1D SolarScale J-V sweep natively in ``solarlab``."""
    _ = v_rate
    if int(n_points) < 2:
        raise ValueError("n_points must be >= 2")
    if not math.isfinite(V_max) or V_max <= 0.0:
        raise ValueError("V_max must be finite and positive")

    prepared, electrical_srvs = prepare_solarscale_device(
        config_or_path, defaults=defaults, resources=resources
    )
    intervals = _intervals_for_n_grid(prepared, int(N_grid))
    voltages_fwd: NDArray[np.float64] = np.linspace(0.0, float(V_max), int(n_points), dtype=np.float64)

    elec_layers = [layer for layer in prepared.layers if layer.role != "substrate"]
    ion_layers = [
        layer.id
        for layer in elec_layers
        if float(dict(layer.parameters)["P0"] or 0.0) > 0.0
        or float(dict(layer.parameters)["P0_neg"] or 0.0) > 0.0
    ]

    if ion_layers:
        if any(v_n != 0.0 or v_p != 0.0 for v_n, v_p in electrical_srvs):
            raise CapabilityError(
                "ionic_interface_srv_1d_unbound: nonzero interface SRV with mobile ions is not bound in 1D steady screening"
            )
        if len(ion_layers) != 1:
            raise CapabilityError("multiple_ion_domains_1d_unbound: SolarScale 1D screening supports one ion domain")
        ionic_dev = compile_ionic_device_1d(
            prepared,
            intervals_per_layer=intervals,
            ion_domain_layer=ion_layers[0],
            ion_domain_endpoints=("blocking", "blocking"),
        )
        ion_opts = IonicSteadyOptions(
            SteadyOptions(),
            1.0e-11,
            1.0e6,
            1.0e6,
            1.0,
            ionic_dev.ion_node_count - 1,
            SolveBudget(65, 650, 65, 24),
            30.0,
        )
        polarity = float(ionic_dev.electronic_component.voltage_polarity)
        j_out_list: list[float] = []
        prev_ion: IonicState1D | None = None
        elec_dim = 3 * ionic_dev.node_count
        for idx, v_app in enumerate(voltages_fwd):
            v_target = float(v_app)
            if prev_ion is None:
                seed_ion = ionic_dev.initial_state(voltage_V=v_target)
            elif prev_ion.voltage_V == v_target:
                seed_ion = prev_ion
            else:
                u_rebased = prev_ion.u.copy()
                u_rebased[:elec_dim] = ionic_dev.electronic_component.rebase_coordinates(
                    u_rebased[:elec_dim], prev_ion.voltage_V, v_target
                )
                seed_ion = ionic_dev.state(u_rebased, voltage_V=v_target)
            res_i = solve_ionic_steady_state(
                ionic_dev,
                voltage_V=v_target,
                generation_scale=1.0,
                initial_state=seed_ion,
                options=ion_opts,
            )
            if not res_i.accepted or res_i.state is None or res_i.diagnostics is None:
                raise RuntimeError(
                    f"1D ionic steady J-V point {idx} at V={v_target:.4f} V failed: {res_i.error or res_i.termination}"
                )
            prev_ion = res_i.state
            tot_cur = np.asarray(res_i.diagnostics["total_current_density_A_m2"], dtype=np.float64)
            j_out_list.append(-polarity * float(tot_cur[0]))
        j_fwd: NDArray[np.float64] = np.asarray(j_out_list, dtype=np.float64)
        compiled_id = ionic_dev.identity
        eg_ceiling = float(np.min(ionic_dev.electronic_component.parameters.band_structure.physical_band_gap_eV))
    else:
        _, device = compile_solarscale_electronic_1d(
            config_or_path, N_grid=int(N_grid), defaults=defaults, resources=resources
        )
        polarity = float(device.voltage_polarity)
        steady_opts = SteadyOptions(max_iterations=250, max_bias_step_V=0.025)
        j_out_list = []
        seed_elec: State1D | None = None
        for idx, v_app in enumerate(voltages_fwd):
            v_target = float(v_app)
            if seed_elec is None:
                sol = solve_steady_state(
                    device,
                    voltage_V=v_target,
                    generation_scale=1.0,
                    options=steady_opts,
                )
            else:
                sol = advance_electronic_steady_1d(
                    device,
                    seed_elec,
                    v_target,
                    generation_scale=1.0,
                    options=steady_opts,
                )
            if not sol.converged:
                raise RuntimeError(f"1D steady J-V point {idx} at V={v_target:.4f} V failed to converge")
            seed_elec = sol.state
            j_out_list.append(-polarity * float(sol.current_density_A_m2[0]))
        j_fwd = np.asarray(j_out_list, dtype=np.float64)
        compiled_id = device.identity
        eg_ceiling = float(np.min(device.parameters.band_structure.physical_band_gap_eV))

    v_rev = voltages_fwd[::-1].copy()
    j_rev = j_fwd[::-1].copy()
    metrics_fwd = compute_solarscale_jv_metrics(voltages_fwd, j_fwd, voc_ceiling_V=eg_ceiling)
    metrics_rev = compute_solarscale_jv_metrics(v_rev, j_rev, voc_ceiling_V=eg_ceiling)
    return SolarScaleJVResult(
        V_fwd=voltages_fwd,
        J_fwd=j_fwd,
        V_rev=v_rev,
        J_rev=j_rev,
        metrics_fwd=metrics_fwd,
        metrics_rev=metrics_rev,
        hysteresis_index=0.0,
        compiled_identity=compiled_id,
        scientific_qualification=False,
    )


def load_material_records(path: str | Path) -> list[dict[str, Any]]:
    """Load raw SolarScale ``MaterialRecord`` dictionaries from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("MaterialRecord payload must contain a records list")
    return records


def parse_material_records(path: str | Path) -> list[MaterialRecord]:
    """Load and normalize SolarScale ``MaterialRecord`` JSON."""
    return [_parse_record(record) for record in load_material_records(path)]


def plan_solarlab_import(
    records_path: str | Path,
    *,
    template_path: str | Path,
    import_policy: ImportPolicy = "exploratory",
    limit: int | None = None,
    sweep_grid: Mapping[str, list[float]] | None = None,
    sweep_policy: SweepPolicyName = DEFAULT_SWEEP_POLICY,
    include_configs: bool = True,
    activate_bandgap: bool = False,
) -> dict[str, Any]:
    """Return a dry-run SolarScale import manifest without writing files."""
    if import_policy not in _ALLOWED_READINESS:
        raise ValueError(f"Unknown import_policy {import_policy!r}")

    records = parse_material_records(records_path)
    template = _load_yaml(template_path)
    if activate_bandgap:
        _validate_band_aligned_template(template)
    sweeps = _resolve_sweep_grid(sweep_policy=sweep_policy, sweep_grid=sweep_grid)

    candidates = sorted(
        (
            _candidate_plan(
                record,
                template,
                sweeps,
                import_policy,
                activate_bandgap=activate_bandgap,
                sweep_policy=sweep_policy,
            )
            for record in records
        ),
        key=_candidate_sort_key,
    )
    selected_count = 0
    limited_candidates: list[CandidatePlan] = []
    for rank, candidate in enumerate(candidates, start=1):
        candidate = _replace_candidate(candidate, rank=rank)
        if candidate.selected:
            if limit is not None and selected_count >= limit:
                candidate = _replace_candidate(
                    candidate,
                    selected=False,
                    reason="limit reached",
                    config=None,
                )
            else:
                selected_count += 1
        if not include_configs:
            candidate = _replace_candidate(candidate, config=None)
        limited_candidates.append(candidate)

    selected = [candidate for candidate in limited_candidates if candidate.selected]
    skipped = [candidate for candidate in limited_candidates if not candidate.selected]
    return {
        "schema": "solarlab.solarscale_import_plan",
        "schema_version": "0.4",
        "records_path": str(records_path),
        "template_path": str(template_path),
        "import_policy": import_policy,
        "allowed_readiness": sorted(_ALLOWED_READINESS[import_policy]),
        "activate_bandgap": activate_bandgap,
        "sweep_policy": sweep_policy,
        "sweep_grid": sweeps,
        "sweep_dimensions": _sweep_dimensions(sweeps),
        "dry_run": True,
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "summary": _screening_summary(limited_candidates),
        "selected": [_candidate_to_dict(candidate) for candidate in selected],
        "skipped": [_candidate_to_dict(candidate, include_config=False) for candidate in skipped],
    }


def generate_solarlab_inputs(
    records_path: str | Path,
    *,
    template_path: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    sweep_grid: Mapping[str, list[float]] | None = None,
    sweep_policy: SweepPolicyName = DEFAULT_SWEEP_POLICY,
    import_policy: ImportPolicy = "production",
    activate_bandgap: bool = False,
) -> dict[str, Any]:
    """Generate SolarLab YAML configs and an import manifest from SolarScale records."""
    plan = plan_solarlab_import(
        records_path,
        template_path=template_path,
        import_policy=import_policy,
        limit=limit,
        sweep_grid=sweep_grid,
        sweep_policy=sweep_policy,
        include_configs=True,
        activate_bandgap=activate_bandgap,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    for item in plan["selected"]:
        config = item.pop("config")
        config_path = out / f"{_safe_name(item['material_id'])}.yaml"
        _write_yaml(config_path, config)
        item["config_path"] = str(config_path)
        generated.append(item)

    manifest: dict[str, Any] = {
        "schema": "solarlab.solarscale_import_manifest",
        "schema_version": "0.4",
        "records_path": plan["records_path"],
        "template_path": plan["template_path"],
        "out_dir": str(out),
        "import_policy": import_policy,
        "allowed_readiness": plan["allowed_readiness"],
        "activate_bandgap": activate_bandgap,
        "sweep_policy": sweep_policy,
        "sweep_grid": plan["sweep_grid"],
        "sweep_dimensions": plan["sweep_dimensions"],
        "summary": plan["summary"],
        "generated": generated,
        "skipped": plan["skipped"],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def expand_sweep_manifest(
    manifest: Mapping[str, Any],
    *,
    out_dir: str | Path | None = None,
    max_points: int | None = None,
) -> dict[str, Any]:
    """Expand selected candidates into one YAML config per sweep-grid point."""
    if max_points is not None and max_points < 0:
        raise ValueError("max_points must be non-negative")
    sweeps = _manifest_sweep_grid(manifest)
    root = Path(out_dir or manifest.get("out_dir") or ".")
    root.mkdir(parents=True, exist_ok=True)
    config_dir = root / "sweep_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    stopped_by_limit = False
    for candidate in manifest.get("generated", []) or []:
        parent_config_path = candidate.get("config_path")
        if not parent_config_path:
            raise ValueError(
                f"Generated candidate {candidate.get('material_id')!r} is missing config_path"
            )
        parent_config = _load_yaml(parent_config_path)
        material_name = _safe_name(str(candidate.get("material_id", "material")))
        material_dir = config_dir / material_name
        material_dir.mkdir(parents=True, exist_ok=True)
        for candidate_index, sweep_values in enumerate(_iter_sweep_values(sweeps), start=1):
            if max_points is not None and len(generated) >= max_points:
                stopped_by_limit = True
                break
            sweep_point_id = f"{material_name}__sweep_{candidate_index:04d}"
            config = copy.deepcopy(parent_config)
            _apply_sweep_values(
                config,
                sweep_values,
                sweep_point_id=sweep_point_id,
                candidate_sweep_index=candidate_index,
                parent_config_path=str(parent_config_path),
            )
            config_path = material_dir / f"{sweep_point_id}.yaml"
            _write_yaml(config_path, config)
            sweep_item = copy.deepcopy(dict(candidate))
            sweep_item.update(
                {
                    "parent_config_path": str(parent_config_path),
                    "config_path": str(config_path),
                    "sweep_point_id": sweep_point_id,
                    "sweep_index": candidate_index,
                    "global_sweep_index": len(generated) + 1,
                    "sweep_values": dict(sweep_values),
                    "sweep_policy": manifest.get("sweep_policy"),
                }
            )
            generated.append(sweep_item)
        if stopped_by_limit:
            break

    per_candidate = _sweep_dimensions(sweeps)
    total_requested = per_candidate["total_points"] * len(manifest.get("generated", []) or [])
    sweep_manifest: dict[str, Any] = {
        "schema": "solarlab.solarscale_sweep_manifest",
        "schema_version": "0.1",
        "parent_schema": manifest.get("schema"),
        "parent_schema_version": manifest.get("schema_version"),
        "records_path": manifest.get("records_path"),
        "template_path": manifest.get("template_path"),
        "out_dir": str(root),
        "sweep_config_dir": str(config_dir),
        "import_policy": manifest.get("import_policy"),
        "allowed_readiness": manifest.get("allowed_readiness", []),
        "activate_bandgap": manifest.get("activate_bandgap"),
        "sweep_policy": manifest.get("sweep_policy"),
        "sweep_grid": sweeps,
        "sweep_dimensions": {
            **per_candidate,
            "generated_configs_per_candidate": per_candidate["total_points"],
            "candidate_count": len(manifest.get("generated", []) or []),
            "total_requested_points": total_requested,
            "total_generated_points": len(generated),
            "max_points": max_points,
            "truncated": stopped_by_limit,
        },
        "summary": manifest.get("summary", {}),
        "generated": generated,
        "skipped": manifest.get("skipped", []),
    }
    (root / "sweep_plan.json").write_text(
        json.dumps(sweep_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return sweep_manifest


def run_smoke_jv(
    config_path: str | Path,
    *,
    N_grid: int = 12,
    n_points: int = 4,
    v_rate: float = 5.0,
    V_max: float = 0.2,
    defaults: DefaultCatalog | None = None,
    resources: ResourceLibrary | None = None,
) -> dict[str, Any]:
    """Run a native 1D JV smoke check for one generated SolarScale config."""
    result = solve_solarscale_jv_1d(
        config_path,
        N_grid=N_grid,
        n_points=n_points,
        v_rate=v_rate,
        V_max=V_max,
        defaults=defaults,
        resources=resources,
    )
    return {
        "config_path": str(config_path),
        "N_grid": N_grid,
        "n_points": n_points,
        "v_rate": v_rate,
        "V_max": V_max,
        "metrics_fwd": {
            "V_oc": result.metrics_fwd.V_oc,
            "J_sc": result.metrics_fwd.J_sc,
            "FF": result.metrics_fwd.FF,
            "PCE": result.metrics_fwd.PCE,
            "voc_bracketed": result.metrics_fwd.voc_bracketed,
        },
        "metrics_rev": {
            "V_oc": result.metrics_rev.V_oc,
            "J_sc": result.metrics_rev.J_sc,
            "FF": result.metrics_rev.FF,
            "PCE": result.metrics_rev.PCE,
            "voc_bracketed": result.metrics_rev.voc_bracketed,
        },
        "hysteresis_index": result.hysteresis_index,
    }


def run_smoke_device_results(
    manifest: Mapping[str, Any],
    *,
    N_grid: int = 12,
    n_points: int = 4,
    v_rate: float = 5.0,
    V_max: float = 0.2,
    max_configs: int | None = 1,
    result_type: str = "smoke_jv",
    defaults: DefaultCatalog | None = None,
    resources: ResourceLibrary | None = None,
) -> dict[str, Any]:
    """Run native 1D JV smoke checks and return SolarScale-ingestible device results."""
    generated = list(manifest.get("generated", []) or [])
    if max_configs is not None:
        generated = generated[:max_configs]
    timestamp = _utc_timestamp()
    git_commit = _git_commit(Path(__file__).resolve())
    records = [
        _device_result_record(
            item,
            manifest,
            timestamp=timestamp,
            git_commit=git_commit,
            smoke_settings={
                "N_grid": N_grid,
                "n_points": n_points,
                "v_rate": v_rate,
                "V_max": V_max,
            },
            defaults=defaults,
            resources=resources,
        )
        for item in generated
    ]
    return {
        "schema": "solarlab.device_results",
        "schema_version": "0.2",
        "result_type": result_type,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "records_path": manifest.get("records_path"),
        "template_path": manifest.get("template_path"),
        "import_policy": manifest.get("import_policy"),
        "activate_bandgap": manifest.get("activate_bandgap"),
        "sweep_policy": manifest.get("sweep_policy"),
        "sweep_dimensions": manifest.get("sweep_dimensions", {}),
        "jv_settings": {
            "N_grid": N_grid,
            "n_points": n_points,
            "v_rate": v_rate,
            "V_max": V_max,
            "max_configs": max_configs,
        },
        "smoke_settings": {
            "N_grid": N_grid,
            "n_points": n_points,
            "v_rate": v_rate,
            "V_max": V_max,
            "max_configs": max_configs,
        },
        "records": records,
        "summary": _device_results_summary(records),
    }


def write_device_results(
    results: Mapping[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path | None = None,
) -> None:
    """Write device results as JSON plus an optional flat CSV table."""
    json_out = Path(json_path)
    json_out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    if csv_path is None:
        return
    csv_out = Path(csv_path)
    rows = [_device_result_csv_row(record) for record in results.get("records", []) or []]
    fieldnames = [
        "material_id",
        "config_path",
        "template_path",
        "import_policy",
        "activate_bandgap",
        "sweep_policy",
        "sweep_point_id",
        "sweep_index",
        "global_sweep_index",
        "sweep_values_json",
        "parent_config_path",
        "simulation_status",
        "Voc",
        "Jsc",
        "FF",
        "PCE",
        "hysteresis_index",
        "voc_bracketed_fwd",
        "voc_bracketed_rev",
        "error_type",
        "error_message",
        "git_commit",
        "timestamp",
    ]
    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _device_result_record(
    item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    timestamp: str,
    git_commit: str | None,
    smoke_settings: Mapping[str, Any],
    defaults: DefaultCatalog | None = None,
    resources: ResourceLibrary | None = None,
) -> dict[str, Any]:
    warnings = list(item.get("diagnostics", []) or [])
    config_path = item.get("config_path")
    if not config_path:
        return _failed_device_result_record(
            item,
            manifest,
            timestamp=timestamp,
            git_commit=git_commit,
            smoke_settings=smoke_settings,
            error=ValueError("generated manifest item is missing config_path"),
            warnings=warnings,
        )
    try:
        smoke = run_smoke_jv(
            config_path,
            N_grid=int(smoke_settings["N_grid"]),
            n_points=int(smoke_settings["n_points"]),
            v_rate=float(smoke_settings["v_rate"]),
            V_max=float(smoke_settings["V_max"]),
            defaults=defaults,
            resources=resources,
        )
    except Exception as exc:  # noqa: BLE001
        return _failed_device_result_record(
            item,
            manifest,
            timestamp=timestamp,
            git_commit=git_commit,
            smoke_settings=smoke_settings,
            error=exc,
            warnings=warnings,
        )

    metrics_fwd = dict(smoke.get("metrics_fwd", {}) or {})
    metrics_rev = dict(smoke.get("metrics_rev", {}) or {})
    if metrics_fwd.get("voc_bracketed") is False:
        warnings.append("forward JV did not bracket Voc; increase smoke V_max for physical metrics")
    if metrics_rev.get("voc_bracketed") is False:
        warnings.append("reverse JV did not bracket Voc; increase smoke V_max for physical metrics")
    return {
        **_device_result_base(
            item,
            manifest,
            timestamp=timestamp,
            git_commit=git_commit,
            smoke_settings=smoke_settings,
            warnings=warnings,
        ),
        "simulation_status": "completed",
        "JV_metrics": {
            "forward": metrics_fwd,
            "reverse": metrics_rev,
            "hysteresis_index": smoke.get("hysteresis_index"),
        },
        "smoke_result": smoke,
        "error": None,
    }


def _failed_device_result_record(
    item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    timestamp: str,
    git_commit: str | None,
    smoke_settings: Mapping[str, Any],
    error: Exception,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        **_device_result_base(
            item,
            manifest,
            timestamp=timestamp,
            git_commit=git_commit,
            smoke_settings=smoke_settings,
            warnings=warnings,
        ),
        "simulation_status": "failed",
        "JV_metrics": {
            "forward": None,
            "reverse": None,
            "hysteresis_index": None,
        },
        "smoke_result": None,
        "error": {
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    }


def _device_result_base(
    item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    timestamp: str,
    git_commit: str | None,
    smoke_settings: Mapping[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "material_id": item.get("material_id"),
        "config_path": item.get("config_path"),
        "template_path": manifest.get("template_path"),
        "records_path": manifest.get("records_path"),
        "import_policy": manifest.get("import_policy"),
        "activate_bandgap": manifest.get("activate_bandgap"),
        "sweep_policy": manifest.get("sweep_policy") or item.get("sweep_policy"),
        "sweep_point_id": item.get("sweep_point_id"),
        "sweep_index": item.get("sweep_index"),
        "global_sweep_index": item.get("global_sweep_index"),
        "parent_config_path": item.get("parent_config_path"),
        "sweep_values": item.get("sweep_values", {}),
        "mapped_parameters": item.get("mapped_parameters", {}),
        "material_metadata": item.get("material_metadata", {}),
        "screening_evidence": item.get("screening_evidence", {}),
        "ranking": {
            "ranking_score": item.get("ranking_score"),
            "ranking_score_source": item.get("ranking_score_source"),
            "ml_pv_score": item.get("ml_pv_score"),
            "final_fom_score": item.get("final_fom_score"),
        },
        "missing_required": item.get("missing_required", []),
        "missing_optional": item.get("missing_optional", []),
        "sweep_parameters": item.get("sweep_parameters", {}),
        "warnings": warnings,
        "jv_settings": dict(smoke_settings),
        "smoke_settings": dict(smoke_settings),
        "git_commit": git_commit,
        "timestamp": timestamp,
    }


def _device_results_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(record.get("simulation_status", "unknown")) for record in records)
    completed = [record for record in records if record.get("simulation_status") == "completed"]
    failed = [record for record in records if record.get("simulation_status") == "failed"]
    best = sorted(
        completed,
        key=lambda record: _metric_value(record, "PCE") or float("-inf"),
        reverse=True,
    )
    return {
        "total_records": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "failed_materials": [record.get("material_id") for record in failed],
        "best_by_pce": [
            {
                "material_id": record.get("material_id"),
                "sweep_point_id": record.get("sweep_point_id"),
                "sweep_index": record.get("sweep_index"),
                "PCE": _metric_value(record, "PCE"),
                "Voc": _metric_value(record, "V_oc"),
                "Jsc": _metric_value(record, "J_sc"),
                "FF": _metric_value(record, "FF"),
            }
            for record in best[:10]
        ],
    }


def _metric_value(record: Mapping[str, Any], field_name: str) -> float | None:
    metrics = record.get("JV_metrics", {})
    if not isinstance(metrics, Mapping):
        return None
    forward = metrics.get("forward")
    if not isinstance(forward, Mapping):
        return None
    value = forward.get(field_name)
    return None if value is None else float(value)


def _device_result_csv_row(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record.get("JV_metrics", {})
    forward = metrics.get("forward") if isinstance(metrics, Mapping) else None
    reverse = metrics.get("reverse") if isinstance(metrics, Mapping) else None
    error = record.get("error") or {}
    return {
        "material_id": record.get("material_id"),
        "config_path": record.get("config_path"),
        "template_path": record.get("template_path"),
        "import_policy": record.get("import_policy"),
        "activate_bandgap": record.get("activate_bandgap"),
        "sweep_policy": record.get("sweep_policy"),
        "sweep_point_id": record.get("sweep_point_id"),
        "sweep_index": record.get("sweep_index"),
        "global_sweep_index": record.get("global_sweep_index"),
        "sweep_values_json": json.dumps(record.get("sweep_values", {}) or {}, sort_keys=True),
        "parent_config_path": record.get("parent_config_path"),
        "simulation_status": record.get("simulation_status"),
        "Voc": _dict_get(forward, "V_oc"),
        "Jsc": _dict_get(forward, "J_sc"),
        "FF": _dict_get(forward, "FF"),
        "PCE": _dict_get(forward, "PCE"),
        "hysteresis_index": _dict_get(metrics, "hysteresis_index"),
        "voc_bracketed_fwd": _dict_get(forward, "voc_bracketed"),
        "voc_bracketed_rev": _dict_get(reverse, "voc_bracketed"),
        "error_type": error.get("error_type"),
        "error_message": error.get("error_message"),
        "git_commit": record.get("git_commit"),
        "timestamp": record.get("timestamp"),
    }


def _dict_get(value: Any, key: str) -> Any:
    if not isinstance(value, Mapping):
        return None
    return value.get(key)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_commit(anchor: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=anchor if anchor.is_dir() else anchor.parent,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:  # noqa: BLE001
        return None
    return result.stdout.strip() or None


def _parse_record(record: Mapping[str, Any]) -> MaterialRecord:
    material_id = str(record.get("material_id", "")).strip()
    properties_raw = record.get("properties", {})
    properties: dict[str, MaterialProperty] = {}
    if isinstance(properties_raw, Mapping):
        for name, prop in properties_raw.items():
            properties[str(name)] = _parse_property(prop)
    notes = record.get("notes", [])
    return MaterialRecord(
        material_id=material_id,
        formula=record.get("formula"),
        schema_version=record.get("schema_version"),
        properties=properties,
        screening=dict(record.get("screening", {}) or {}),
        stages=dict(record.get("stages", {}) or {}),
        source=dict(record.get("source", {}) or {}),
        notes=[str(note) for note in notes] if isinstance(notes, list) else [str(notes)],
    )


def _parse_property(prop: Any) -> MaterialProperty:
    if not isinstance(prop, Mapping):
        return MaterialProperty(value=prop)
    provenance = prop.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    return MaterialProperty(
        value=prop.get("value"),
        unit=str(prop.get("unit", "") or ""),
        provenance=PropertyProvenance(
            kind=str(provenance.get("kind", "missing") or "missing"),
            source=str(provenance.get("source", "not_available") or "not_available"),
            notes=str(provenance.get("notes", "") or ""),
        ),
    )


def _resolve_sweep_grid(
    *,
    sweep_policy: SweepPolicyName,
    sweep_grid: Mapping[str, list[float]] | None,
) -> dict[str, list[float]]:
    if sweep_policy not in SWEEP_POLICIES:
        raise ValueError(
            f"Unknown sweep_policy {sweep_policy!r}; expected one of "
            + ", ".join(sorted(SWEEP_POLICIES))
        )
    grid = {name: list(values) for name, values in SWEEP_POLICIES[sweep_policy].items()}
    if sweep_grid:
        for name, values in sweep_grid.items():
            grid[str(name)] = list(values)
    _validate_sweep_grid(grid)
    return grid


def _validate_sweep_grid(grid: Mapping[str, list[float]]) -> None:
    required = {
        "absorber.thickness",
        "absorber.tau_n",
        "absorber.tau_p",
        "absorber.trap_N_t_bulk",
        "device.interfaces",
    }
    missing = sorted(required - set(grid))
    if missing:
        raise ValueError("sweep grid is missing required dimensions: " + ", ".join(missing))
    empty = [name for name, values in grid.items() if not values]
    if empty:
        raise ValueError("sweep grid dimensions must not be empty: " + ", ".join(sorted(empty)))


def _sweep_dimensions(sweeps: Mapping[str, list[float]]) -> dict[str, Any]:
    sizes = {name: len(values) for name, values in sweeps.items()}
    total = 1
    for size in sizes.values():
        total *= size
    return {
        "dimensions": sizes,
        "total_points": total,
        "generated_configs_per_candidate": 1,
        "note": (
            "Sweep-grid size per selected candidate. Baseline imports generate one config; "
            "expand_sweep_manifest generates the full matrix or a capped subset."
        ),
    }


def _candidate_plan(
    record: MaterialRecord,
    template: Mapping[str, Any],
    sweeps: Mapping[str, list[float]],
    import_policy: ImportPolicy,
    *,
    activate_bandgap: bool,
    sweep_policy: SweepPolicyName,
) -> CandidatePlan:
    material_id = record.material_id
    if not material_id:
        return CandidatePlan(
            material_id="",
            rank=0,
            selected=False,
            reason="missing material_id",
            readiness="",
            ranking_score=None,
            ranking_score_source=None,
            ml_pv_score=None,
            final_fom_score=None,
            screening_evidence={
                "readiness": None,
                "resolved_readiness": "",
                "gates": {},
                "thresholds": {},
                "raw_screening": {},
            },
            mapped_parameters={},
            material_metadata={},
            sweep_parameters={},
            missing_required=list(REQUIRED_PHYSICAL_INPUTS),
            missing_optional=list(OPTIONAL_PHYSICAL_INPUTS),
            assumed_parameters={},
            diagnostics=[],
        )

    readiness = str(
        record.screening.get("readiness", "") or _property_value(record, "screening_readiness") or ""
    )
    ready, reason = _is_device_ready(record, import_policy=import_policy)
    mapped, missing_required, missing_optional, diagnostics = _mapped_parameters(
        record,
        activate_bandgap=activate_bandgap,
    )
    material_metadata = _material_metadata(record)
    sweep_parameters = _sweep_parameters(record, sweeps)
    assumed = _template_assumptions(template, mapped, sweeps)
    ranking_score, ranking_source = _ranking_score(record)
    ml_score = _numeric_property(record, "ml_pv_score")
    fom_score = _numeric_property(record, "final_fom_score")
    screening_evidence = _screening_evidence(record, resolved_readiness=readiness)
    selected = ready and not missing_required
    if ready and missing_required:
        reason = "missing required SolarLab physical inputs: " + ", ".join(missing_required)

    config = None
    if selected:
        config = _build_config(
            template,
            record,
            mapped,
            sweeps,
            missing_optional,
            diagnostics,
            import_policy=import_policy,
            activate_bandgap=activate_bandgap,
            sweep_policy=sweep_policy,
        )

    return CandidatePlan(
        material_id=material_id,
        rank=0,
        selected=selected,
        reason=reason,
        readiness=readiness,
        ranking_score=ranking_score,
        ranking_score_source=ranking_source,
        ml_pv_score=ml_score,
        final_fom_score=fom_score,
        screening_evidence=screening_evidence,
        mapped_parameters=mapped,
        material_metadata=material_metadata,
        sweep_parameters=sweep_parameters,
        missing_required=missing_required,
        missing_optional=missing_optional,
        assumed_parameters=assumed,
        diagnostics=diagnostics,
        config=config,
    )


def _is_device_ready(record: MaterialRecord, *, import_policy: ImportPolicy) -> tuple[bool, str]:
    if import_policy not in _ALLOWED_READINESS:
        raise ValueError(f"Unknown import_policy {import_policy!r}")

    flag_name = _READY_FLAG_BY_POLICY[import_policy]
    flag_value = _property_value(record, flag_name)
    if flag_value is None:
        flag_value = record.screening.get(
            "production_solarlab_ready"
            if import_policy == "production"
            else "provisional_solarlab_ready"
        )
    if flag_value is True:
        return True, f"{flag_name}=true"
    if import_policy == "production":
        return False, f"{flag_name} is not true"

    readiness = str(
        record.screening.get("readiness", "") or _property_value(record, "screening_readiness") or ""
    )
    if readiness:
        if readiness in _ALLOWED_READINESS[import_policy]:
            return True, f"screening.readiness={readiness!r}"
        return False, f"screening.readiness={readiness!r} not allowed by {import_policy} policy"

    dft_flag = _property_value(record, "dft_result_available")
    if import_policy == "exploratory" and dft_flag is True:
        return True, "dft_result_available=true"
    return False, f"{flag_name} is not true and screening.readiness is unavailable"


def _mapped_parameters(
    record: MaterialRecord,
    *,
    activate_bandgap: bool,
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    mapped: dict[str, Any] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []
    diagnostics: list[str] = []

    band_gap = record.properties.get("band_gap_hse_ev")
    if band_gap is None or band_gap.value is None or band_gap.provenance.kind == "missing":
        missing_required.append("band_gap_hse_ev")
    elif band_gap.provenance.kind not in {"computed", "derived"}:
        missing_required.append("band_gap_hse_ev")
        diagnostics.append(
            f"band_gap_hse_ev provenance kind {band_gap.provenance.kind!r} is not accepted as a fixed input"
        )
    elif activate_bandgap:
        mapped["absorber.Eg"] = _coerce_numeric(band_gap.value)
        diagnostics.append("band_gap_hse_ev activated as absorber Eg")
    else:
        diagnostics.append("band_gap_hse_ev kept as metadata; absorber Eg follows the template")

    for prop_name, (target, _field, scale) in MAPPED_PROPERTY_SPECS.items():
        prop = record.properties.get(prop_name)
        if prop is None or prop.value is None or prop.provenance.kind == "missing":
            if prop_name in REQUIRED_PHYSICAL_INPUTS:
                missing_required.append(prop_name)
            else:
                missing_optional.append(prop_name)
            continue
        if prop.provenance.kind == "swept":
            missing_optional.append(prop_name)
            diagnostics.append(f"{prop_name} is swept and was not mapped as a fixed absorber input")
            continue
        if prop.provenance.kind not in {"computed", "derived"}:
            if prop_name in REQUIRED_PHYSICAL_INPUTS:
                missing_required.append(prop_name)
            else:
                missing_optional.append(prop_name)
            diagnostics.append(
                f"{prop_name} provenance kind {prop.provenance.kind!r} is not accepted as a fixed input"
            )
            continue
        mapped[target] = _coerce_numeric(prop.value) * scale

    for score_field in SCORE_FIELDS:
        if score_field in record.properties:
            diagnostics.append(f"{score_field} imported as ranking metadata only")
    if _property_value(record, "slme_0p5um") is not None or _property_value(record, "slme_2um") is not None:
        diagnostics.append("SLME is preserved as ranking metadata; it is not converted to absorber alpha")
    if _property_value(record, "absorption_edge_ev") is not None:
        diagnostics.append("absorption_edge_ev is not converted to a scalar absorption coefficient")
    return mapped, missing_required, missing_optional, diagnostics


def _sweep_parameters(
    record: MaterialRecord,
    sweeps: Mapping[str, list[float]],
) -> dict[str, Any]:
    sweep_parameters: dict[str, Any] = {name: list(values) for name, values in sweeps.items()}
    for prop_name, target_name in DEVICE_SWEEP_PROPERTY_MAP.items():
        prop = record.properties.get(prop_name)
        if prop is None:
            continue
        value = prop.value
        kind = prop.provenance.kind
        if kind == "swept":
            if target_name in sweeps:
                sweep_parameters.setdefault(target_name, list(sweeps[target_name]))
        elif value is not None:
            sweep_parameters[target_name] = value
    return sweep_parameters


def _template_assumptions(
    template: Mapping[str, Any],
    mapped: Mapping[str, Any],
    sweeps: Mapping[str, list[float]],
) -> dict[str, Any]:
    layers = template.get("layers")
    if not isinstance(layers, list):
        return {}
    absorber = _absorber_layer(copy.deepcopy(layers))
    assumptions: dict[str, Any] = {}
    if "absorber.Eg" not in mapped:
        assumptions["absorber.Eg"] = absorber.get("Eg", 0.0)
    for _prop_name, (target, field_name, _scale) in MAPPED_PROPERTY_SPECS.items():
        if field_name is None or target in mapped:
            continue
        if field_name in absorber:
            assumptions[f"absorber.{field_name}"] = absorber[field_name]
    assumptions["absorber.thickness"] = sweeps["absorber.thickness"][0]
    assumptions["absorber.tau_n"] = sweeps["absorber.tau_n"][0]
    assumptions["absorber.tau_p"] = sweeps["absorber.tau_p"][0]
    assumptions["absorber.trap_N_t_bulk"] = sweeps["absorber.trap_N_t_bulk"][0]
    assumptions["device.interfaces"] = sweeps["device.interfaces"][0]
    return assumptions


def _build_config(
    template: Mapping[str, Any],
    record: MaterialRecord,
    mapped: Mapping[str, Any],
    sweeps: Mapping[str, list[float]],
    missing_optional: list[str],
    diagnostics: list[str],
    *,
    import_policy: ImportPolicy,
    activate_bandgap: bool,
    sweep_policy: SweepPolicyName,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(template))
    layers = config.get("layers")
    if not isinstance(layers, list):
        raise ValueError("Template layers must be a list")

    absorber = _absorber_layer(layers)
    material_id = record.material_id
    absorber["name"] = f"SolarScale_{material_id}"
    activate_material_specific_device_evidence = bool(
        activate_bandgap and "absorber.Eg" in mapped and "absorber.chi" in mapped
    )
    for target, value in mapped.items():
        if target == "absorber.chi" and not activate_material_specific_device_evidence:
            continue
        field_name = target.split(".", 1)[1]
        absorber[field_name] = value
    optical_material = _text_property(record, "optical_material")
    if activate_material_specific_device_evidence and optical_material and optical_material != "MAPbI3":
        absorber["optical_material"] = optical_material

    absorber["thickness"] = sweeps["absorber.thickness"][0]
    absorber["tau_n"] = sweeps["absorber.tau_n"][0]
    absorber["tau_p"] = sweeps["absorber.tau_p"][0]
    absorber["trap_N_t_bulk"] = sweeps["absorber.trap_N_t_bulk"][0]
    absorber["trap_profile_shape"] = "exponential"

    device = config.setdefault("device", {})
    interface_s = sweeps["device.interfaces"][0]
    device["interfaces"] = [[interface_s, interface_s] for _ in range(max(0, len(layers) - 1))]
    device.setdefault("mode", "fast")

    notes = list(config.get("notes", []) or [])
    notes.append(
        "Generated from SolarScale MaterialRecord. Only computed/derived DFT/MD absorber inputs are fixed; device-only unknowns remain sweep dimensions."
    )
    if import_policy == "exploratory":
        notes.append(
            "Exploratory import may include records whose MD/ion gate is missing; do not treat those as publication-grade candidates."
        )
    if "absorber.Eg" in mapped:
        notes.append(
            "HSE band gap is explicitly activated as absorber Eg. Use only with a fully band-aligned template."
        )
    else:
        notes.append(
            "HSE band gap is preserved as metadata; absorber Eg follows the template to avoid partial band-alignment activation."
        )
    material_metadata = _material_metadata(record)
    screening_evidence = _screening_evidence(
        record,
        resolved_readiness=str(
            record.screening.get("readiness", "") or _property_value(record, "screening_readiness") or ""
        ),
    )
    config["notes"] = notes
    config["source"] = {
        "schema": "solarlab.solarscale_import_config",
        "schema_version": "0.4",
        "material_id": material_id,
        "import_policy": import_policy,
        "activate_bandgap": activate_bandgap,
        "sweep_policy": sweep_policy,
        "sweep_grid": {name: list(values) for name, values in sweeps.items()},
        "sweep_dimensions": _sweep_dimensions(sweeps),
        "screening": record.screening,
        "screening_evidence": screening_evidence,
        "ranking": {
            "ranking_score": _ranking_score(record)[0],
            "ranking_score_source": _ranking_score(record)[1],
            "ml_pv_score": _numeric_property(record, "ml_pv_score"),
            "final_fom_score": _numeric_property(record, "final_fom_score"),
        },
        "material_metadata": material_metadata,
        "mapped_parameters": dict(mapped),
        "optical_material": optical_material,
        "nk_provenance": _text_property(record, "nk_provenance"),
        "nk_csv_path": _text_property(record, "nk_csv_path"),
        "nk_source_path": _text_property(record, "nk_source_path"),
        "band_alignment_provenance": _text_property(record, "band_alignment_provenance"),
        "band_alignment_source_path": _text_property(record, "band_alignment_source_path"),
        "missing_optional": list(missing_optional),
        "sweep_parameters": _sweep_parameters(record, sweeps),
        "diagnostics": list(diagnostics),
    }
    config["simulation_hints"] = {
        "N_grid": 30,
        "n_points": 8,
        "v_rate": 5.0,
        "V_max": 1.4,
        "notes": (
            "First-pass SolarScale imports should run low-resolution screening "
            "before production sweeps."
        ),
    }
    return config


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "device" not in data or "layers" not in data:
        raise ValueError(f"Template is not a SolarLab device config: {path}")
    return dict(data)


def _write_yaml(path: Path, config: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(config), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _manifest_sweep_grid(manifest: Mapping[str, Any]) -> dict[str, list[float]]:
    raw = manifest.get("sweep_grid")
    if not isinstance(raw, Mapping):
        raise ValueError("Manifest is missing sweep_grid")
    sweeps = {str(name): [float(value) for value in values] for name, values in raw.items()}
    _validate_sweep_grid(sweeps)
    return {name: sweeps[name] for name in _sweep_dimension_names(sweeps)}


def _sweep_dimension_names(sweeps: Mapping[str, list[float]]) -> list[str]:
    ordered = [name for name in SWEEP_DIMENSION_ORDER if name in sweeps]
    ordered.extend(sorted(name for name in sweeps if name not in set(ordered)))
    return ordered


def _iter_sweep_values(sweeps: Mapping[str, list[float]]) -> Iterator[dict[str, float]]:
    names = _sweep_dimension_names(sweeps)
    for values in itertools.product(*(sweeps[name] for name in names)):
        yield dict(zip(names, values, strict=True))


def _apply_sweep_values(
    config: dict[str, Any],
    sweep_values: Mapping[str, float],
    *,
    sweep_point_id: str,
    candidate_sweep_index: int,
    parent_config_path: str,
) -> None:
    layers = config.get("layers")
    if not isinstance(layers, list):
        raise ValueError("Sweep config layers must be a list")
    absorber = _absorber_layer(layers)
    for target, value in sweep_values.items():
        if target.startswith("absorber."):
            absorber[target.split(".", 1)[1]] = float(value)
        elif target == "device.interfaces":
            device = config.setdefault("device", {})
            interface_s = float(value)
            device["interfaces"] = [[interface_s, interface_s] for _ in range(max(0, len(layers) - 1))]
        else:
            raise ValueError(f"Unsupported sweep target {target!r}")
    source = config.setdefault("source", {})
    source["sweep_point_id"] = sweep_point_id
    source["sweep_index"] = candidate_sweep_index
    source["parent_config_path"] = parent_config_path
    source["sweep_values"] = {name: float(value) for name, value in sweep_values.items()}


def _absorber_layer(layers: list[dict[str, Any]]) -> dict[str, Any]:
    absorbers = [layer for layer in layers if layer.get("role") == "absorber"]
    if len(absorbers) != 1:
        raise ValueError("Template must contain exactly one absorber layer")
    return absorbers[0]


def _ranking_score(record: MaterialRecord) -> tuple[float | None, str | None]:
    for score_name in SCORE_FIELDS:
        score = _numeric_property(record, score_name)
        if score is not None:
            return score, score_name
    return None, None


def _candidate_sort_key(candidate: CandidatePlan) -> tuple[int, float, str]:
    selected_bucket = 0 if candidate.selected else 1
    score = candidate.ranking_score if candidate.ranking_score is not None else float("-inf")
    return selected_bucket, -score, candidate.material_id


def _candidate_to_dict(candidate: CandidatePlan, *, include_config: bool = True) -> dict[str, Any]:
    data: dict[str, Any] = {
        "material_id": candidate.material_id,
        "rank": candidate.rank,
        "selected": candidate.selected,
        "reason": candidate.reason,
        "readiness": candidate.readiness,
        "ranking_score": candidate.ranking_score,
        "ranking_score_source": candidate.ranking_score_source,
        "ml_pv_score": candidate.ml_pv_score,
        "final_fom_score": candidate.final_fom_score,
        "screening_evidence": candidate.screening_evidence,
        "mapped_parameters": candidate.mapped_parameters,
        "material_metadata": candidate.material_metadata,
        "dft_properties": _legacy_dft_properties(candidate.mapped_parameters, candidate.material_metadata),
        "md_properties": _legacy_md_properties(candidate.mapped_parameters),
        "sweep_parameters": candidate.sweep_parameters,
        "missing_required": candidate.missing_required,
        "missing_optional": candidate.missing_optional,
        "assumed_parameters": candidate.assumed_parameters,
        "diagnostics": candidate.diagnostics,
    }
    if candidate.config_path is not None:
        data["config_path"] = candidate.config_path
    if include_config and candidate.config is not None:
        data["config"] = candidate.config
    return data


def _replace_candidate(candidate: CandidatePlan, **changes: Any) -> CandidatePlan:
    data: dict[str, Any] = {
        "material_id": candidate.material_id,
        "rank": candidate.rank,
        "selected": candidate.selected,
        "reason": candidate.reason,
        "readiness": candidate.readiness,
        "ranking_score": candidate.ranking_score,
        "ranking_score_source": candidate.ranking_score_source,
        "ml_pv_score": candidate.ml_pv_score,
        "final_fom_score": candidate.final_fom_score,
        "screening_evidence": candidate.screening_evidence,
        "mapped_parameters": candidate.mapped_parameters,
        "material_metadata": candidate.material_metadata,
        "sweep_parameters": candidate.sweep_parameters,
        "missing_required": candidate.missing_required,
        "missing_optional": candidate.missing_optional,
        "assumed_parameters": candidate.assumed_parameters,
        "diagnostics": candidate.diagnostics,
        "config": candidate.config,
        "config_path": candidate.config_path,
    }
    data.update(changes)
    return CandidatePlan(**data)


def _screening_summary(candidates: list[CandidatePlan]) -> dict[str, Any]:
    readiness_counts = Counter(candidate.readiness or "<missing>" for candidate in candidates)
    skipped_reason_counts = Counter(candidate.reason for candidate in candidates if not candidate.selected)
    selected_candidates = [candidate for candidate in candidates if candidate.selected]
    return {
        "readiness_distribution": dict(sorted(readiness_counts.items())),
        "gate_summary": _gate_summary(candidates),
        "skipped_reason_counts": dict(sorted(skipped_reason_counts.items())),
        "top_selected_candidates": [
            {
                "material_id": candidate.material_id,
                "rank": candidate.rank,
                "readiness": candidate.readiness,
                "ranking_score": candidate.ranking_score,
                "ranking_score_source": candidate.ranking_score_source,
            }
            for candidate in selected_candidates[:10]
        ],
    }


def _gate_summary(candidates: list[CandidatePlan]) -> dict[str, Any]:
    statuses = ("pass", "fail", "missing", "unknown")
    totals: Counter[str] = Counter()
    by_gate: dict[str, Counter[str]] = {}
    records_with_gate_data = 0
    for candidate in candidates:
        gates = candidate.screening_evidence.get("gates", {})
        if not isinstance(gates, Mapping) or not gates:
            continue
        records_with_gate_data += 1
        for gate_name, gate_value in gates.items():
            status = _gate_status(gate_value)
            gate_counts = by_gate.setdefault(str(gate_name), Counter())
            gate_counts[status] += 1
            totals[status] += 1
    return {
        "records_with_gate_data": records_with_gate_data,
        "totals": {status: totals.get(status, 0) for status in statuses},
        "by_gate": {
            gate_name: {status: counts.get(status, 0) for status in statuses}
            for gate_name, counts in sorted(by_gate.items())
        },
    }


def _gate_status(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("status", "state", "outcome", "result"):
            if key in value:
                return _gate_status(value[key])
        for key in ("passed", "pass", "is_passed"):
            if key in value:
                return "pass" if bool(value[key]) else "fail"
        if value.get("missing") is True or value.get("available") is False:
            return "missing"
        return "unknown"
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    if value is None:
        return "missing"
    text = str(value).strip().lower().replace("-", "_")
    if text in {"pass", "passed", "ok", "true", "yes", "accepted"}:
        return "pass"
    if text in {"fail", "failed", "false", "no", "blocked", "rejected", "reject"}:
        return "fail"
    if text in {"missing", "not_run", "not_available", "none", ""}:
        return "missing"
    if text in {"unknown", "pending", "inconclusive"}:
        return "unknown"
    return "unknown"


def _screening_evidence(record: MaterialRecord, *, resolved_readiness: str) -> dict[str, Any]:
    raw_screening = copy.deepcopy(record.screening)
    return {
        "readiness": raw_screening.get("readiness"),
        "resolved_readiness": resolved_readiness,
        "gates": copy.deepcopy(raw_screening.get("gates", {})),
        "thresholds": copy.deepcopy(raw_screening.get("thresholds", {})),
        "raw_screening": raw_screening,
    }


def _property_value(record: MaterialRecord, name: str) -> Any:
    prop = record.properties.get(name)
    if prop is None:
        return None
    return prop.value


def _material_metadata(record: MaterialRecord) -> dict[str, Any]:
    return {
        "band_gap_hse_ev": _numeric_property(record, "band_gap_hse_ev"),
        "electron_effective_mass_m0": _numeric_property(record, "electron_effective_mass_m0"),
        "hole_effective_mass_m0": _numeric_property(record, "hole_effective_mass_m0"),
        "electron_affinity_ev": _numeric_property(record, "electron_affinity_ev"),
        "cbm_vs_vacuum_ev": _numeric_property(record, "cbm_vs_vacuum_ev"),
        "vbm_vs_vacuum_ev": _numeric_property(record, "vbm_vs_vacuum_ev"),
        "optical_material": _text_property(record, "optical_material"),
        "nk_csv_path": _text_property(record, "nk_csv_path"),
        "nk_provenance": _text_property(record, "nk_provenance"),
        "nk_source_path": _text_property(record, "nk_source_path"),
        "nk_wavelength_min_nm": _numeric_property(record, "nk_wavelength_min_nm"),
        "nk_wavelength_max_nm": _numeric_property(record, "nk_wavelength_max_nm"),
        "band_alignment_provenance": _text_property(record, "band_alignment_provenance"),
        "band_alignment_source_path": _text_property(record, "band_alignment_source_path"),
        "slme_0p5um": _numeric_property(record, "slme_0p5um"),
        "slme_2um": _numeric_property(record, "slme_2um"),
        "absorption_edge_ev": _numeric_property(record, "absorption_edge_ev"),
    }


def _validate_band_aligned_template(template: Mapping[str, Any]) -> None:
    layers = template.get("layers")
    if not isinstance(layers, list):
        raise ValueError("Template layers must be a list")
    invalid: list[str] = []
    for layer in layers:
        if layer.get("role") == "substrate":
            continue
        name = str(layer.get("name", "<unnamed>"))
        chi = _optional_float(layer.get("chi"))
        eg = _optional_float(layer.get("Eg"))
        if chi is None or eg is None or eg <= 0.0:
            invalid.append(name)
    if invalid:
        raise ValueError(
            "--activate-bandgap requires a fully band-aligned template with "
            "chi and positive Eg on every electrical layer; invalid layers: "
            + ", ".join(invalid)
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _legacy_dft_properties(
    mapped: Mapping[str, Any],
    material_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "band_gap_hse_ev": mapped.get("absorber.Eg", material_metadata.get("band_gap_hse_ev")),
        "dielectric_static_avg": mapped.get("absorber.eps_r"),
        "electron_mobility_cm2_v_s": _inverse_scaled(mapped.get("absorber.mu_n"), 1.0e-4),
        "hole_mobility_cm2_v_s": _inverse_scaled(mapped.get("absorber.mu_p"), 1.0e-4),
    }


def _legacy_md_properties(mapped: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ion_diffusion_coefficient_m2_s": mapped.get("absorber.D_ion"),
        "ion_activation_energy_ev": mapped.get("absorber.E_a_ion"),
    }


def _inverse_scaled(value: Any, scale: float) -> float | None:
    if value is None:
        return None
    return float(value) / scale


def _numeric_property(record: MaterialRecord, name: str) -> float | None:
    prop = record.properties.get(name)
    if prop is None or prop.value is None or prop.provenance.kind == "missing":
        return None
    try:
        return float(prop.value)
    except (TypeError, ValueError):
        return None


def _text_property(record: MaterialRecord, name: str) -> str | None:
    prop = record.properties.get(name)
    if prop is None or prop.value is None or prop.provenance.kind == "missing":
        return None
    text = str(prop.value).strip()
    return text or None


def _coerce_numeric(value: Any) -> float:
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def _safe_name(material_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", material_id)
