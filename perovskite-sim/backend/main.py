import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, Optional

# BLAS thread pinning — defensive fix. The Radau solver hits ~300x300 dense LU
# factors thousands of times per J-V sweep. On a multi-core machine, OpenBLAS
# and MKL try to parallelise each call across every core; at this matrix size
# thread-creation + contention overhead can dominate and turn a ~3 s sweep into
# several minutes. The slow test suite (tests/conftest.py) pins BLAS for the
# same reason, but the backend previously inherited no such guard, so the first
# TMM J-V sweep from the UI could intermittently stall.
#
# Set the env vars BEFORE importing numpy so BLAS reads them on library load.
# Opt out with PEROVSKITE_BLAS_PIN=0 if running on a dedicated box and you want
# the solver to use all cores for something larger (e.g. parallel sweeps).
if os.environ.get("PEROVSKITE_BLAS_PIN", "1") != "0":
    for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_var, "1")

import numpy as np
import traceback
import yaml
from dataclasses import asdict, is_dataclass, replace
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt

from perovskite_sim.experiments import degradation, impedance, jv_sweep
from perovskite_sim.experiments import (
    dynamic_defect_transient as dynamic_defect_transient_exp,
)
from perovskite_sim.experiments import interface_charge_jv as interface_charge_jv_exp
from perovskite_sim.experiments import external_circuit as external_circuit_exp
from perovskite_sim.experiments import electrothermal as electrothermal_exp
from perovskite_sim.experiments import identifiability as identifiability_exp
from perovskite_sim.experiments import thermal_balance as thermal_balance_exp
from perovskite_sim.experiments import dark_jv as dark_jv_exp
from perovskite_sim.experiments import suns_voc as suns_voc_exp
from perovskite_sim.experiments import eqe as eqe_exp
from perovskite_sim.experiments import mott_schottky as ms_exp
from perovskite_sim.experiments import tpv as tpv_exp
from perovskite_sim.experiments.protocol import (
    ExperimentProtocol,
    ExperimentProtocolError,
    ImplicitProtocolError,
    ProtocolMode,
    resolve_experiment_protocol,
)
from perovskite_sim.experiments.steady_state import run_jv_sweep_ss
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiSteadyStateError,
    build_equilibrium_referenced_interface_charge_dark_reference,
    build_two_sided_trace_grid,
    solve_equilibrium_referenced_interface_charge_steady_state,
    solve_quasi_fermi_jv_sweep,
)
from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import (
    GridResolutionError,
    require_thick_layer_interface_resolution,
)
from perovskite_sim.models.config_loader import (
    built_in_potential_fields_from_device_dict,
    electrical_grid_from_config_dict,
    interface_charge_fields_from_device_dict,
    interfaces_from_device_dict,
    load_device_from_yaml,
    material_params_from_dict,
)
from perovskite_sim.models.device import (
    DeviceStack,
    InterfaceChargeClosureParkedError,
    LayerSpec,
    MicroscopicInterfaceDefectContractError,
    require_uncalibrated_microscopic_interface_defects,
)
from perovskite_sim.models.mode import resolve_mode
from perovskite_sim.models.tunneling_channels import (
    tunnelling_channel_document_from_mapping,
)
from perovskite_sim.physics.contacts import (
    ContactThermodynamicError,
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.physics.defect_closure import (
    MONOVALENT_BULK_DEFECT_MODEL_VERSION,
)
from perovskite_sim.solver.mol import build_material_arrays
from backend.jobs import JobRegistry, JobStatus, _DRAIN_TIMEOUT
from backend.progress import ProgressReporter
from backend.user_configs import (
    is_shipped_name,
    validate_user_filename,
    write_user_config,
)

def _describe_active_physics(stack) -> str:
    """Return a short human-readable description of the active physics tier.

    Used by the SSE result payload so the frontend solver console can
    show which Phase 1–3 upgrades ran without re-deriving the flags.
    """
    mode_name = str(getattr(stack, "mode", "full")).lower()
    mode = resolve_mode(mode_name)
    # Drive every label fragment off the mode flags so the indicator can't
    # silently drift from the physics that actually ran. Missing labels
    # (e.g. PR when use_photon_recycling is False) are left out rather than
    # rendered as "no PR" to keep the string short.
    parts: list[str] = [
        "band offsets · TE" if mode.use_thermionic_emission else "flat bands",
        "TMM" if mode.use_tmm_optics else "Beer-Lambert",
        "dual ions" if mode.use_dual_ions else "single ion",
        "trap profile" if mode.use_trap_profile else "uniform τ",
        "T-scaling" if mode.use_temperature_scaling else "T=300K",
    ]
    # Phase 3.x extras appended only when on, so the label stays short for
    # LEGACY / FAST and expands only for FULL or custom modes that opt in.
    if mode.use_photon_recycling:
        parts.append("photon recycling")
    if mode.use_radiative_reabsorption:
        parts.append("PR reabsorption")
    if mode.use_field_dependent_mobility:
        parts.append("μ(E)")
    if mode.use_selective_contacts:
        parts.append("Robin contacts")
    if getattr(stack, "interface_charge_closure", "off") == (
        "equilibrium_referenced"
    ):
        parts.append("equilibrium-referenced interface charge")
    return f"{mode.name.upper()}  " + " · ".join(parts)


_JOB_REGISTRY = JobRegistry()


CONFIGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs")
)


def _coerce_numbers(obj: Any) -> Any:
    """Recursively convert strings that look like numbers into floats.

    PyYAML's 1.1 resolver leaves scientific-notation literals without a decimal
    point (e.g. ``1e-9``) as strings; the frontend numeric editor then fails.
    """
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        try:
            return float(obj)
        except (ValueError, TypeError):
            return obj
    return obj


def resolve_config_path(config_path: str) -> str:
    """Resolve config_path to an absolute path inside perovskite-sim/configs if needed."""
    if os.path.isabs(config_path):
        return config_path
    backend_dir = os.path.dirname(__file__)
    candidate1 = os.path.abspath(os.path.join(backend_dir, config_path))
    if os.path.exists(candidate1):
        return candidate1
    candidate2 = os.path.join(CONFIGS_DIR, os.path.basename(config_path))
    if os.path.exists(candidate2):
        return candidate2
    return config_path


def _opt_S(v) -> Optional[float]:
    """Parse a Robin contact S value. ``None`` and missing → None
    (= ohmic Dirichlet, the documented "absent / disabled" sentinel);
    every other value is coerced to float, including 0.0 (= Neumann
    blocking — distinct from absent)."""
    if v is None:
        return None
    return float(v)


def _flag(v) -> bool:
    """Parse a boolean device flag with the same string-truthiness rule as
    config_loader (so ``true``/``1``/``yes``/``on`` and real booleans all
    read as True). Absent / None → False."""
    return str(v).strip().lower() in ("true", "1", "yes", "on")


def stack_from_dict(cfg: dict) -> DeviceStack:
    """Build a DeviceStack from a dict with the same schema as the YAML files."""
    dev = cfg.get("device", {}) or {}
    layers: list[LayerSpec] = []
    for layer_cfg in cfg.get("layers", []) or []:
        # Delegate to the shared loader parser so the inline-device path (the
        # frontend's only path) carries EVERY layer field the YAML loader does.
        # It previously hand-rolled a subset and silently dropped 17 (TE A_star,
        # effective DOS Nc300/Nv300, trap profiles, dual-ion, temperature
        # scaling), disabling that physics for UI-built devices.
        p = material_params_from_dict(layer_cfg)
        layers.append(
            LayerSpec(
                name=str(layer_cfg["name"]),
                thickness=float(layer_cfg["thickness"]),
                params=p,
                role=str(layer_cfg["role"]),
            )
        )
    grid_interval_weights, grid_alphas = electrical_grid_from_config_dict(
        cfg, layers
    )
    # Legacy ``interfaces`` schema: list of (v_n, v_p) m/s SRV pairs aligned
    # with the heterointerfaces. Phase E1.5 / E1.8 SCAPS-style schema:
    # ``interface_defects`` list of per-slot ``{sigma_n_cm2, sigma_p_cm2,
    # N_t_cm2, v_th_cm_s, E_t_eV_below_cb} | None``. When both schemas
    # supply data on the same index k, ``interface_defects`` takes
    # precedence (computes SRV via σ·v_th·N_t kinetic identity); legacy
    # ``interfaces`` survives for slots where ``interface_defects[k]`` is
    # None or absent.
    # Parsed by the SAME helper the YAML loader uses (Phase E1.6's optional
    # per-slot ``calibration_factor`` from the live editor round-trips through
    # it). These two paths drifted apart repeatedly before they shared a
    # parser — do not re-inline this.
    interfaces, interface_defects = interfaces_from_device_dict(dev, len(layers))
    mode_name = str(dev.get("mode", "full"))
    # Validate early so an unknown mode fails the HTTP request rather than
    # blowing up inside the worker thread where the error is harder to surface.
    resolve_mode(mode_name)
    # Stage B(a) microstructure — mirror load_device_from_yaml's behaviour
    # so the inline-device path round-trips the ``microstructure:`` block
    # the same way the YAML loader does. Without this, loading a preset
    # like configs/twod/nip_MAPbI3_singleGB.yaml in the workstation and
    # submitting via ``device:`` would silently drop the GB block at the
    # backend boundary (the frontend's startJob always sends device: and
    # never config_path:, so the load_device_from_yaml microstructure
    # path is never used at runtime). Lazy import keeps FastAPI startup
    # cost unchanged when no jv_2d run is dispatched.
    ms_block = cfg.get("microstructure")
    if ms_block:
        from perovskite_sim.twod.microstructure import (
            load_microstructure_from_yaml_block,
        )
        microstructure = load_microstructure_from_yaml_block(ms_block)
    else:
        from perovskite_sim.twod.microstructure import Microstructure
        microstructure = Microstructure()
    # Robin contacts: accept the nested ``contacts: {left/right: {S_n, S_p}}``
    # block as well as flat top-level S_* keys, mirroring config_loader. Flat
    # keys win when both are present. Absent → None (ohmic Dirichlet).
    contacts_cfg = dev.get("contacts", {}) or {}
    left_cfg = contacts_cfg.get("left", {}) or {}
    right_cfg = contacts_cfg.get("right", {}) or {}
    return DeviceStack(
        layers=tuple(layers),
        **built_in_potential_fields_from_device_dict(dev),
        Phi=float(dev.get("Phi", 2.5e21)),
        grid_interval_weights=grid_interval_weights,
        grid_alphas=grid_alphas,
        jv_solver_policy=str(dev.get("jv_solver_policy", "general")),
        **interface_charge_fields_from_device_dict(dev),
        interfaces=interfaces,
        interface_defects=interface_defects,
        T=float(dev.get("T", 300.0)),
        mode=mode_name,
        # SCAPS-validation physics flags — mirror load_device_from_yaml so the
        # inline-device path (the frontend's only path) no longer silently
        # drops them. Same string-truthiness parsing as the YAML loader.
        # dos_band_potentials defaults ON (absent → True), matching the loader
        # default; the rest default off / 0.0. DOS is a no-op without per-layer
        # Nc300/Nv300, so non-DOS device dicts are bit-identical.
        interface_plane_projection=_flag(dev.get("interface_plane_projection")),
        dos_band_potentials=_flag(dev.get("dos_band_potentials", True)),
        te_physical_norm=_flag(dev.get("te_physical_norm")),
        ion_steric_diffusion_only=_flag(
            dev.get("ion_steric_diffusion_only", True)
        ),
        ion_steric_shared_site=_flag(dev.get("ion_steric_shared_site", True)),
        flat_band_contacts=_flag(dev.get("flat_band_contacts")),
        flat_band_metal_contacts=_flag(dev.get("flat_band_metal_contacts")),
        contact_phi_B_eV=float(dev.get("contact_phi_B_eV", 0.0)),
        interface_two_sided=_flag(dev.get("interface_two_sided")),
        interface_shared_occupancy=_flag(
            dev.get("interface_shared_occupancy")
        ),
        interface_plane_closure=_flag(dev.get("interface_plane_closure")),
        interface_plane_generation=_flag(
            dev.get("interface_plane_generation")
        ),
        het_recomb_despike=float(dev.get("het_recomb_despike", 0.0)),
        band_grading=_flag(dev.get("band_grading")),
        graded_optics=_flag(dev.get("graded_optics")),
        interface_tunneling=_flag(dev.get("interface_tunneling")),
        tunnel_mass_eff=float(dev.get("tunnel_mass_eff", 0.2)),
        # D8 WKB tunnelling family. Shares one parser with the YAML loader so
        # the file and inline paths cannot drift — the failure mode is silent
        # (an inline run just loses the channel), which is why the semantic
        # hash of every shipped config is compared across both paths.
        tunnelling_channels=tunnelling_channel_document_from_mapping(dev),
        # Stage B(c.1) Robin / selective contacts. None = ohmic Dirichlet
        # (the pre-3.3 default); 0 = Neumann blocking; positive finite =
        # Robin. The frontend distinguishes these three states via
        # parseNumOrNull in config-editor.ts.
        S_n_left=_opt_S(dev.get("S_n_left", left_cfg.get("S_n"))),
        S_p_left=_opt_S(dev.get("S_p_left", left_cfg.get("S_p"))),
        S_n_right=_opt_S(dev.get("S_n_right", right_cfg.get("S_n"))),
        S_p_right=_opt_S(dev.get("S_p_right", right_cfg.get("S_p"))),
        autoloop_generated_lever=_flag(dev.get("autoloop_generated_lever")),
        microstructure=microstructure,
    )


def _is_scaps_schema(cfg: dict) -> bool:
    """True if the config uses the SCAPS layer schema (``mu_n_cm2`` / ``N_C_cm3``
    / ``thickness_nm``, cm/eV units, ``ni`` computed from the DOS) rather than
    the standard schema. Only ``scaps_compat.load_scaps_yaml`` can parse it; the
    rest of the backend / frontend assumes the standard schema."""
    for layer in cfg.get("layers", []) or []:
        if "mu_n_cm2" in layer or "thickness_nm" in layer or "N_C_cm3" in layer:
            return True
    return False


def _stack_to_config_dict(stack: DeviceStack) -> dict:
    """Serialize a DeviceStack to the standard config dict the frontend edits and
    ``stack_from_dict`` rebuilds, so SCAPS-schema presets (parsed only by
    scaps_compat) flow through the standard UI / inline-device path.

    Layer params come straight from ``dataclasses.asdict(MaterialParams)`` — flat
    standard fields, including the Nc300/Nv300 the SCAPS loader computes — so the
    round-trip is exact for every field ``stack_from_dict`` reads. Interface
    recombination is emitted as both resolved (v_n, v_p) SRV pairs and an
    equivalent standard-schema defect block. The original sigma/v_th split is
    not identifiable after loading, so the serializer chooses v_th=1e7 cm/s
    and derives sigma to preserve SRV exactly; N_t, trap depth, and both
    calibration factors retain their original solver semantics."""
    layers = []
    for ls in stack.layers:
        # Drop None-valued fields: stack_from_dict treats several optional
        # params (Eg_back / chi_back / grading_char_length / n_optical …) as
        # "absent" via key-presence and does float(value) when the key exists,
        # so a serialized None would crash with float(None). None == absent.
        d = {k: v for k, v in asdict(ls.params).items() if v is not None}
        defect_document = ls.params.defect_document
        if defect_document is None:
            d.pop("defect_model", None)
            d.pop("bulk_defects", None)
        else:
            defect_payload = defect_document.to_dict()
            d["defect_schema_version"] = defect_payload["schema_version"]
            d["defect_model"] = defect_payload["defect_model"]
            d["bulk_defects"] = defect_payload["bulk_defects"]
        d["name"] = ls.name
        d["role"] = ls.role
        d["thickness"] = ls.thickness
        layers.append(d)
    interface_defects = []
    for pair, defect in zip(stack.interfaces, stack.interface_defects):
        if defect is None:
            interface_defects.append(None)
            continue
        document = defect.microscopic_document
        if document is None:
            N_t_cm2 = defect.N_t_cm2 if defect.N_t_cm2 > 0.0 else 1.0
            v_th_cm_s = 1.0e7
            microscopic_fields = {
                "sigma_n_cm2": pair[0] / (v_th_cm_s * N_t_cm2 * 1.0e-2),
                "sigma_p_cm2": pair[1] / (v_th_cm_s * N_t_cm2 * 1.0e-2),
                "N_t_cm2": N_t_cm2,
                "v_th_cm_s": v_th_cm_s,
                "E_t_eV_below_cb": defect.E_t_eV,
            }
        else:
            microscopic_fields = document.to_scaps_cgs_fields()
        interface_defects.append({
            **microscopic_fields,
            "calibration_factor": defect.calibration_factor,
            "iface_state_calibration_factor": (
                defect.iface_state_calibration_factor
            ),
        })
    device = {
        "mode": str(stack.mode),
        "Phi": stack.Phi,
        "T": stack.T,
        "interfaces": [list(p) for p in stack.interfaces],
        "interface_defects": interface_defects,
        "dos_band_potentials": stack.dos_band_potentials,
        "te_physical_norm": stack.te_physical_norm,
        "ion_steric_diffusion_only": stack.ion_steric_diffusion_only,
        "ion_steric_shared_site": stack.ion_steric_shared_site,
        "autoloop_generated_lever": stack.autoloop_generated_lever,
        "flat_band_contacts": stack.flat_band_contacts,
        "flat_band_metal_contacts": stack.flat_band_metal_contacts,
        "contact_phi_B_eV": stack.contact_phi_B_eV,
        "interface_two_sided": stack.interface_two_sided,
        "interface_shared_occupancy": stack.interface_shared_occupancy,
        "interface_plane_closure": stack.interface_plane_closure,
        "interface_plane_generation": stack.interface_plane_generation,
        "interface_plane_projection": stack.interface_plane_projection,
        "het_recomb_despike": stack.het_recomb_despike,
        "band_grading": stack.band_grading,
        "graded_optics": stack.graded_optics,
        "interface_tunneling": stack.interface_tunneling,
        "tunnel_mass_eff": stack.tunnel_mass_eff,
        "tunnelling_channels": (
            stack.tunnelling_channels.to_dict()
            if stack.tunnelling_channels is not None
            else None
        ),
        "jv_solver_policy": stack.jv_solver_policy,
        "interface_charge_closure": stack.interface_charge_closure,
        "interface_charge_rebaseline_acknowledged": (
            stack.interface_charge_rebaseline_acknowledged
        ),
        "S_n_left": stack.S_n_left,
        "S_p_left": stack.S_p_left,
        "S_n_right": stack.S_n_right,
        "S_p_right": stack.S_p_right,
    }
    if stack.built_in_potential_mode is None:
        # Preserve the shape and exact semantics of shipped compatibility
        # presets when they round-trip through the frontend.
        device["V_bi"] = stack.V_bi
    else:
        device["built_in_potential_mode"] = stack.built_in_potential_mode
        if stack.built_in_potential_mode == "legacy_manual":
            device["V_bi_override"] = stack.V_bi
        elif stack.built_in_potential_mode == "metal_work_function":
            device["work_function_left_eV"] = stack.work_function_left_eV
            device["work_function_right_eV"] = stack.work_function_right_eV
    config = {"device": device, "layers": layers}
    if stack.grid_interval_weights or stack.grid_alphas:
        electrical = tuple(
            layer for layer in stack.layers if layer.role != "substrate"
        )
        if (
            len(stack.grid_interval_weights) != len(electrical)
            or len(stack.grid_alphas) != len(electrical)
        ):
            raise ValueError(
                "DeviceStack electrical-grid tuples must align with its "
                "electrical layers"
            )
        config["electrical_grid"] = {
            "interval_weights": {
                layer.name: weight
                for layer, weight in zip(electrical, stack.grid_interval_weights)
            },
            "alphas": {
                layer.name: alpha
                for layer, alpha in zip(electrical, stack.grid_alphas)
            },
        }
        # Apply the same strict contract used on input before exposing a config
        # that the frontend may later submit through stack_from_dict.
        electrical_grid_from_config_dict(config, stack.layers)
    return config


def _config_dict_from_path(path: str) -> dict:
    """Load a config file as a STANDARD-schema dict. SCAPS-schema files are
    converted via scaps_compat; standard files pass through unchanged."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if _is_scaps_schema(cfg):
        from perovskite_sim.scaps_compat import load_scaps_yaml
        return _stack_to_config_dict(load_scaps_yaml(path))
    return cfg


def _load_stack(config_path: Optional[str], device: Optional[dict]) -> DeviceStack:
    """Load a stack without applying an experiment-specific capability gate."""
    if device is not None:
        return stack_from_dict(device)
    else:
        if not config_path:
            raise HTTPException(
                status_code=400,
                detail="Either 'device' or 'config_path' must be provided",
            )
        resolved = resolve_config_path(config_path)
        # SCAPS-schema files need the scaps_compat parser; load_device_from_yaml
        # assumes the standard schema and raises KeyError 'mu_n' on them.
        try:
            with open(resolved) as f:
                is_scaps = _is_scaps_schema(yaml.safe_load(f))
        except (OSError, yaml.YAMLError):
            is_scaps = False
        if is_scaps:
            from perovskite_sim.scaps_compat import load_scaps_yaml

            return load_scaps_yaml(resolved)
        return load_device_from_yaml(resolved)


def build_stack(config_path: Optional[str], device: Optional[dict]) -> DeviceStack:
    """Load a stack and enforce the production interface-charge gate."""
    stack = _load_stack(config_path, device)
    try:
        stack.require_interface_charge_off(consumer="backend experiment routes")
    except InterfaceChargeClosureParkedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return stack


def _require_interface_charge_research_stack(
    stack: DeviceStack,
    *,
    consumer: str,
) -> DeviceStack:
    """Apply the shared fail-closed contract for charged research stacks."""

    violations: list[str] = []
    if stack.interface_charge_closure != "equilibrium_referenced":
        violations.append(
            "interface_charge_closure must be 'equilibrium_referenced'"
        )
    if not stack.interface_charge_rebaseline_acknowledged:
        violations.append("interface-charge rebaseline must be acknowledged")
    if stack.het_recomb_despike != 0.0:
        violations.append("recombination de-spiking must be disabled")
    if stack.flat_band_contacts or stack.flat_band_metal_contacts:
        violations.append("calibrated flat-band contact floors must be disabled")
    if stack.contact_phi_B_eV != 0.0:
        violations.append("the calibrated contact barrier must be zero")
    if stack.autoloop_generated_lever:
        violations.append("autoloop-generated calibration levers are not accepted")
    try:
        require_uncalibrated_microscopic_interface_defects(
            stack,
            consumer=consumer,
        )
    except MicroscopicInterfaceDefectContractError as exc:
        violations.append(str(exc))
    if violations:
        raise HTTPException(status_code=422, detail="; ".join(violations))
    return stack


def build_jv_stack(
    config_path: Optional[str],
    device: Optional[dict],
) -> DeviceStack:
    """Load a J-V stack, admitting only the certified charged DC slice."""

    stack = _load_stack(config_path, device)
    if stack.interface_charge_closure == "off":
        return stack
    if (config_path is None) == (device is None):
        raise HTTPException(
            status_code=422,
            detail=(
                "charged interface J-V requires exactly one of 'device' or "
                "'config_path'"
            ),
        )
    return _require_interface_charge_research_stack(
        stack,
        consumer="charged interface J-V endpoint",
    )


def build_interface_charge_research_stack(
    config_path: Optional[str],
    device: Optional[dict],
) -> DeviceStack:
    """Load only stacks eligible for the charged steady-state research lane."""
    if (config_path is None) == (device is None):
        raise HTTPException(
            status_code=422,
            detail="provide exactly one of 'device' or 'config_path'",
        )
    return _require_interface_charge_research_stack(
        _load_stack(config_path, device),
        consumer="interface-charge research endpoint",
    )


def build_dynamic_defect_transient_stack(
    config_path: Optional[str],
    device: Optional[dict],
) -> DeviceStack:
    """Load only charged stacks eligible for the certified D6 transient."""

    if (config_path is None) == (device is None):
        raise HTTPException(
            status_code=422,
            detail=(
                "dynamic-defect transient requires exactly one of 'device' "
                "or 'config_path'"
            ),
        )
    return _require_interface_charge_research_stack(
        _load_stack(config_path, device),
        consumer="production dynamic-defect transient",
    )


def to_serializable(obj):
    """Recursively convert dataclasses and numpy arrays to JSON-serializable types."""
    if is_dataclass(obj):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            # Preserve multidimensional face/frequency diagnostics. Flattening
            # here used to discard the shape of complex Y_faces arrays.
            return to_serializable(obj.tolist())
        return obj.tolist()
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    # NOTE: bool MUST be checked before int — Python ``bool`` subclasses
    # ``int``, so ``isinstance(True, int) == True`` and the int branch
    # below would silently coerce ``True``/``False`` to ``1``/``0``,
    # breaking strict-equality checks (e.g. ``=== true``) on the frontend.
    # ``np.bool_`` is included so numpy scalar booleans round-trip too.
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    else:
        return obj


app = FastAPI(title="Perovskite Solar Cell Simulator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/configs")
def list_configs():
    """List YAML configs available to the frontend.

    Each entry carries a ``namespace`` tag so the frontend can render the
    dropdown as two ``<optgroup>``s — shipped (top-level configs/) and user
    (configs/user/). Returning a list of dicts is a deliberate breaking
    change vs the Phase 2a flat-list shape; the frontend api wrapper updates
    in lockstep.
    """
    def _peek_metadata(path: str) -> tuple[str, list[str]]:
        # Cheap YAML peek — returns (device_type, tier_compat).
        #
        # device_type: tandem presets have no stack.layers and must be routed
        # to the Tandem pane instead of the single-cell Device editor.
        #
        # tier_compat: list of physics tiers this preset runs correctly under.
        # Every preset supports 'legacy' and 'fast' — both tiers no-op safely
        # when layers leave the opt-in Phase 1/2/3.1 parameters unset: TE
        # needs non-zero band offsets between neighbouring layers (chi/Eg),
        # TMM needs `optical_material` on a layer, PR needs TMM to be active,
        # dual-ion and trap-profile and T-scaling each need their own opt-in
        # config keys. 'full' is advertised when the preset explicitly selects
        # it (needed by full-only field-mobility/Robin demos), or when every
        # electrical layer has positive band alignment. Both standard
        # ``chi/Eg`` and SCAPS ``chi_eV/E_g_eV`` spellings are recognized.
        # Tandem presets are single-cell-only for now and advertise legacy/fast.
        legacy_tiers = ["legacy", "fast"]
        try:
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            return "single", legacy_tiers
        device_type = str(data.get("device_type", "single"))
        if device_type != "single":
            return device_type, legacy_tiers
        layers = data.get("layers") or []
        electrical = [
            layer for layer in layers if layer.get("role") != "substrate"
        ]
        device = data.get("device") or {}
        explicit_full = str(device.get("mode", "")).lower() == "full"

        def _positive(layer: dict, *keys: str) -> bool:
            return any(float(layer.get(key, 0.0) or 0.0) > 0.0 for key in keys)

        has_band_alignment = electrical and all(
            _positive(layer, "chi", "chi_eV")
            and _positive(layer, "Eg", "E_g_eV")
            for layer in electrical
        )
        if explicit_full or has_band_alignment:
            return device_type, [*legacy_tiers, "full"]
        return device_type, legacy_tiers

    try:
        entries: list[dict] = []
        seen_names: set[str] = set()
        for f in sorted(os.listdir(CONFIGS_DIR)):
            if f.endswith((".yaml", ".yml")):
                full = os.path.join(CONFIGS_DIR, f)
                if os.path.isfile(full):
                    device_type, tier_compat = _peek_metadata(full)
                    entries.append({
                        "name": f,
                        "namespace": "shipped",
                        "device_type": device_type,
                        "tier_compat": tier_compat,
                    })
                    seen_names.add(f)
        user_dir = os.path.join(CONFIGS_DIR, "user")
        if os.path.isdir(user_dir):
            for f in sorted(os.listdir(user_dir)):
                if f.endswith((".yaml", ".yml")):
                    full = os.path.join(user_dir, f)
                    device_type, tier_compat = _peek_metadata(full)
                    entries.append({
                        "name": f,
                        "namespace": "user",
                        "device_type": device_type,
                        "tier_compat": tier_compat,
                    })
                    seen_names.add(f)
        # Phase 6 shipped 2D presets live under ``configs/twod/`` (Stage A
        # baseline, Stage B(a) microstructure, T7 B(c.x) demo). They are
        # listed in the same ``shipped`` namespace as top-level presets so
        # the existing dropdown surfaces them without a UI redesign.
        # Collision policy: top-level precedence is preserved — a basename
        # already listed (top-level or user/) shadows the twod entry, and
        # the shadowing is reported on stderr so a maintainer can dedupe.
        twod_dir = os.path.join(CONFIGS_DIR, "twod")
        if os.path.isdir(twod_dir):
            for f in sorted(os.listdir(twod_dir)):
                if not f.endswith((".yaml", ".yml")):
                    continue
                if f in seen_names:
                    print(
                        f"[list_configs] basename collision: 'configs/twod/{f}' "
                        f"shadowed by an earlier entry; skipping",
                        file=sys.stderr,
                    )
                    continue
                full = os.path.join(twod_dir, f)
                if os.path.isfile(full):
                    device_type, tier_compat = _peek_metadata(full)
                    entries.append({
                        "name": f,
                        "namespace": "shipped",
                        "device_type": device_type,
                        "tier_compat": tier_compat,
                    })
                    seen_names.add(f)
        return {"status": "ok", "configs": entries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/optical-materials")
def list_optical_materials() -> dict:
    """Auto-scan ``perovskite_sim/data/nk/`` and return the sorted material list.

    The frontend optical-material picker calls this to populate its dropdown,
    so dropping a new ``<name>.csv`` in the nk directory makes it visible with
    no code change (same convention as ``/api/configs``).
    """
    nk_dir = Path(__file__).resolve().parent.parent / "perovskite_sim" / "data" / "nk"
    return {"materials": sorted(p.stem for p in nk_dir.glob("*.csv"))}


@app.get("/api/layer-templates")
def list_layer_templates() -> dict:
    """Return the parsed layer templates library used by the Add Layer dialog.

    The library lives in ``perovskite_sim/data/layer_templates.yaml`` so the
    frontend can populate the dialog without re-deriving material defaults.
    """
    path = (
        Path(__file__).resolve().parent.parent
        / "perovskite_sim"
        / "data"
        / "layer_templates.yaml"
    )
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail="layer_templates.yaml missing — Phase 2b data file not installed",
        )
    with path.open() as f:
        templates = yaml.safe_load(f) or {}
    return {"status": "ok", "templates": _coerce_numbers(templates)}


@app.get("/api/configs/{name}")
def get_config(name: str):
    """Return the parsed YAML device config so the frontend can edit it.

    Search order — top-level ``configs/`` → ``configs/user/`` →
    ``configs/twod/``. Top-level precedence is preserved on basename
    collision so a user-renamed top-level preset always wins, matching
    the listing order in :func:`list_configs`. ``os.path.basename``
    strips any leading path components in case a caller URL-encodes a
    slash.
    """
    safe_name = os.path.basename(name)
    candidates = [
        os.path.join(CONFIGS_DIR, safe_name),
        os.path.join(CONFIGS_DIR, "user", safe_name),
        os.path.join(CONFIGS_DIR, "twod", safe_name),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Config '{safe_name}' not found")
    try:
        # SCAPS-schema presets (mu_n_cm2 / N_C_cm3 / …) are converted to the
        # standard schema so the frontend, which assumes the standard fields,
        # gets real editable values instead of an all-zero device.
        cfg = _config_dict_from_path(path)
        return {"status": "ok", "name": safe_name, "config": _coerce_numbers(cfg)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UserConfigPayload(BaseModel):
    name: str
    config: dict
    overwrite: bool = False


@app.post("/api/configs/user")
def save_user_config(payload: UserConfigPayload):
    """Write a user-edited DeviceConfig to ``configs/user/<name>.yaml``.

    The frontend Save-As dialog calls this. ``user_configs`` owns filename
    validation, shipped-name reservation, and atomic writes.
    """
    try:
        validate_user_filename(payload.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if is_shipped_name(payload.name):
        raise HTTPException(
            status_code=409,
            detail=f"{payload.name!r} is reserved by a shipped preset",
        )
    try:
        write_user_config(payload.name, payload.config, overwrite=payload.overwrite)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "saved": payload.name}


class InterfaceChargeSteadyStateResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_path: Optional[str] = None
    device: Optional[dict] = None
    N_grid: StrictInt = 60
    V_app: float = 0.0
    illuminated: StrictBool = True
    research_acknowledged: StrictBool = False


class InterfaceChargeSolverControlsEvidence(BaseModel):
    finite_difference_step: float
    newton_residual_tolerance: float
    max_newton_iterations: int
    poisson_tolerance_V: float
    poisson_max_iterations: int
    continuity_tolerance_A_m2: float
    current_spread_tolerance_A_m2: float
    poisson_residual_tolerance: float
    illumination_steps: tuple[float, ...]


class InterfaceChargeContactEvidence(BaseModel):
    status: str
    built_in_potential_mode: str
    tolerance_eV: float
    fermi_level_span_eV: Optional[float]
    potential_mismatch_V: Optional[float]
    metal_work_function_mismatch_eV: Optional[float]
    contact_quasi_fermi_levels_eV: tuple[float, ...]
    message: str


class InterfaceChargeDarkReferenceEvidence(BaseModel):
    certified: bool
    charge_on_off_bit_identical: bool
    grid_sha256: str
    stack_sha256: str
    dark_state_sha256: str
    interface_defect_document_sha256: tuple[str, ...]
    capture_velocities_m_s: tuple[tuple[float, float], ...]
    equilibrium_occupancy: tuple[float, ...]
    trap_density_m2: tuple[float, ...]
    incremental_sheet_charge_C_m2: tuple[float, ...]
    trace_potential_shift_V: tuple[tuple[float, float], ...]


class InterfaceChargeOperatingPointEvidence(BaseModel):
    V_app_V: float
    illuminated: bool
    certified: bool
    current_density_A_m2: float
    electron_continuity_bound_A_m2: float
    hole_continuity_bound_A_m2: float
    face_current_spread_A_m2: float
    max_normalized_cell_residual: float
    poisson_residual: float
    poisson_residual_C_m2: float
    interface_local_residual: float
    numerical_residual_limit: float
    newton_iterations: int
    residual_evaluations: int
    operating_state_sha256: str


class InterfaceChargePerInterfaceEvidence(BaseModel):
    interface_index: int
    equilibrium_occupancy: float
    occupancy: float
    trap_density_m2: float
    incremental_sheet_charge_C_m2: float
    trace_potential_shift_left_V: float
    trace_potential_shift_right_V: float
    normalized_gauss_residual: float
    scaled_local_jacobian_condition: float


class InterfaceChargeResearchProvenance(BaseModel):
    requested_grid_intervals: int
    actual_grid_nodes: int
    interface_count: int
    interface_topology: str
    interface_charge_closure: str
    research_acknowledged: bool


class InterfaceChargeSteadyStateResearchResult(BaseModel):
    evidence_status: Literal["internal_numerical_research"]
    capability_scope: Literal[
        "steady_state_equilibrium_referenced_interface_charge"
    ]
    production_unlocked: Literal[False]
    provenance: InterfaceChargeResearchProvenance
    solver_controls: InterfaceChargeSolverControlsEvidence
    contact_thermodynamics: InterfaceChargeContactEvidence
    dark_reference: InterfaceChargeDarkReferenceEvidence
    operating_point: InterfaceChargeOperatingPointEvidence
    interfaces: tuple[InterfaceChargePerInterfaceEvidence, ...]
    limitations: tuple[str, ...]


class InterfaceChargeSteadyStateResearchResponse(BaseModel):
    status: Literal["ok"]
    result: InterfaceChargeSteadyStateResearchResult


class JVRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_path: Optional[str] = None
    device: Optional[dict] = None
    N_grid: int = 80
    n_points: int = 40
    v_rate: float = 1.0
    V_max: Optional[float] = None
    # "transient" (default) = legacy Radau forward/reverse sweep;
    # "steady_state" = ion-free Newton driver (run_jv_sweep_ss);
    # "quasi_fermi" = cancellation-safe, certificate-bearing QF driver.
    solver: str = "transient"
    iface_states: bool = False  # SS driver only: interface-plane carrier states
    # QF only: reciprocal physical interface plane.
    interface_boundary: bool = False
    interface_transport_model: str = "fermi_richardson"
    protocol_mode: ProtocolMode = "compatibility"
    experiment_protocol: Optional[dict[str, Any]] = None
    interface_charge_jv_protocol: Optional[dict[str, Any]] = None


class ExternalCircuitJVRequest(JVRequest):
    """Intrinsic J-V request plus a strict, area-normalized DC circuit."""

    model_config = ConfigDict(extra="forbid")

    external_circuit_protocol: dict[str, Any]
    incident_power_W_m2: float = 1000.0


class ElectrothermalOperatingPointRequest(BaseModel):
    """Strict protocols for a fresh-state electrothermal MPP root."""

    model_config = ConfigDict(extra="forbid")

    config_path: Optional[str] = None
    device: Optional[dict] = None
    thermal_protocol: dict[str, Any]
    external_circuit_protocol: dict[str, Any]
    electrical_protocol: dict[str, Any]
    operating_protocol: dict[str, Any]


class InterfaceSRHIdentifiabilityRequest(BaseModel):
    """Strict synthetic interface-SRH identifiability protocol."""

    model_config = ConfigDict(extra="forbid")

    protocol: dict[str, Any]


def _interface_charge_research_solver_controls() -> dict[str, float | int]:
    """Return the frozen, certificate-compatible controls for the API lane."""
    return {
        "finite_difference_step": 1.0e-5,
        "newton_residual_tolerance": 4.0e-7,
        "max_newton_iterations": 60,
        "poisson_tolerance_V": 1.0e-12,
        "poisson_max_iterations": 100,
        "continuity_tolerance_A_m2": 1.0e-4,
        "current_spread_tolerance_A_m2": 1.0e-4,
        "poisson_residual_tolerance": 1.0e-8,
    }


def _require_research_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _backend_research_array_sha256(label: str, *arrays: object) -> str:
    digest = hashlib.sha256(label.encode("ascii"))
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _build_interface_charge_research_result(
    req: InterfaceChargeSteadyStateResearchRequest,
    grid: np.ndarray,
    charge_off_material: object,
    contact_certificate: object,
    dark_reference: object,
    charged_dark: object,
    result: object,
    solver_controls: dict[str, float | int],
) -> InterfaceChargeSteadyStateResearchResult:
    """Validate and serialize the charged operating-point evidence."""
    if not bool(getattr(contact_certificate, "certified", False)):
        raise ValueError("contact thermodynamic certificate is not certified")
    if not bool(getattr(dark_reference.dark_state, "certified", False)):
        raise ValueError("dark reference is not certified")
    if (
        not bool(getattr(charged_dark, "certified", False))
        or charged_dark.interface_charge_closure != "equilibrium_referenced"
    ):
        raise ValueError("charged dark-reference validation is not certified")
    if not bool(getattr(result, "certified", False)):
        raise ValueError("charged operating point is not certified")
    if result.interface_charge_closure != "equilibrium_referenced":
        raise ValueError("charged result has the wrong interface-charge closure")
    if result.interface_topology != "two_sided_trace":
        raise ValueError("charged result has the wrong interface topology")
    if not bool(result.interface_boundary):
        raise ValueError("charged result lacks the two-sided interface boundary")
    if float(result.V_app) != float(req.V_app):
        raise ValueError("charged result voltage does not match the request")
    if bool(result.illuminated) is not bool(req.illuminated):
        raise ValueError("charged result illumination does not match the request")

    equilibrium = np.asarray(dark_reference.equilibrium_occupancy, dtype=float)
    trap_density = np.asarray(dark_reference.trap_density_m2, dtype=float)
    document_hashes = tuple(dark_reference.interface_defect_document_sha256)
    capture_velocities = np.asarray(
        dark_reference.capture_velocities_m_s,
        dtype=float,
    )
    result_equilibrium = np.asarray(
        result.interface_equilibrium_occupancy,
        dtype=float,
    )
    occupancy = np.asarray(result.interface_occupancy, dtype=float)
    sheet_charge = np.asarray(
        result.interface_incremental_sheet_charge_C_m2,
        dtype=float,
    )
    trace_shift = np.asarray(
        result.interface_trace_potential_shift_V,
        dtype=float,
    )
    gauss = np.asarray(result.interface_normalized_gauss_residual, dtype=float)
    condition = np.asarray(
        result.interface_scaled_local_jacobian_condition,
        dtype=float,
    )
    interface_count = equilibrium.size
    if interface_count == 0:
        raise ValueError("interface evidence must not be empty")
    vectors = (
        trap_density,
        result_equilibrium,
        occupancy,
        sheet_charge,
        gauss,
        condition,
    )
    if equilibrium.shape != (interface_count,) or any(
        vector.shape != (interface_count,) for vector in vectors
    ):
        raise ValueError("interface evidence arrays are misaligned")
    if trace_shift.shape != (interface_count, 2):
        raise ValueError("interface trace-shift evidence is misaligned")
    if len(document_hashes) != interface_count or capture_velocities.shape != (
        interface_count,
        2,
    ):
        raise ValueError("microscopic interface-defect evidence is misaligned")
    if any(
        not np.all(np.isfinite(value))
        for value in (equilibrium, *vectors, trace_shift, capture_velocities)
    ):
        raise ValueError("interface evidence contains non-finite values")
    if np.any(capture_velocities < 0.0):
        raise ValueError(
            "microscopic interface capture velocities must be non-negative"
        )
    if np.any(trap_density <= 0.0):
        raise ValueError("interface trap densities must be positive")
    if np.any((equilibrium < 0.0) | (equilibrium > 1.0)) or np.any(
        (occupancy < 0.0) | (occupancy > 1.0)
    ):
        raise ValueError("interface occupancies must lie in [0, 1]")
    if not np.array_equal(result_equilibrium, equilibrium):
        raise ValueError("operating-point f_eq differs from the dark reference")
    expected_charge = -Q * trap_density * (occupancy - equilibrium)
    if not np.allclose(
        sheet_charge,
        expected_charge,
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise ValueError("interface sheet charge violates -q*Nt*(f-f_eq)")
    if np.any(np.abs(sheet_charge) > Q * trap_density * (1.0 + 1.0e-12)):
        raise ValueError("interface sheet charge exceeds one electron per trap")
    if np.any(np.abs(gauss) > 1.0e-10):
        raise ValueError("interface Gauss residual exceeds the research gate")
    if np.any((condition < 0.0) | (condition > 1.0e8)):
        raise ValueError("interface Jacobian condition exceeds the research gate")

    dark_charge = np.asarray(
        charged_dark.interface_incremental_sheet_charge_C_m2,
        dtype=float,
    )
    dark_trace_shift = np.asarray(
        charged_dark.interface_trace_potential_shift_V,
        dtype=float,
    )
    if dark_charge.shape != (interface_count,) or dark_trace_shift.shape != (
        interface_count,
        2,
    ):
        raise ValueError("charged dark-reference evidence is misaligned")
    if np.any(dark_charge != 0.0) or np.any(dark_trace_shift != 0.0):
        raise ValueError("charged dark reference must have exact zero increment")
    if not np.array_equal(
        np.asarray(charged_dark.interface_equilibrium_occupancy, dtype=float),
        equilibrium,
    ) or not np.array_equal(
        np.asarray(charged_dark.interface_occupancy, dtype=float),
        equilibrium,
    ):
        raise ValueError("charged dark-reference occupancy differs from f_eq")
    dark_identity_fields = (
        "y",
        "phi",
        "electron_quasi_fermi_potential_V",
        "hole_quasi_fermi_potential_V",
        "electron_face_current_A_m2",
        "hole_face_current_A_m2",
        "total_face_current_A_m2",
        "electron_rate_per_s",
        "hole_rate_per_s",
    )
    dark_arrays_identical = all(
        np.array_equal(
            np.asarray(getattr(dark_reference.dark_state, name)),
            np.asarray(getattr(charged_dark, name)),
        )
        for name in dark_identity_fields
    )
    if not dark_arrays_identical:
        raise ValueError("charged and charge-off dark arrays are not bit-identical")

    grid_array = np.asarray(grid, dtype=float)
    if (
        grid_array.ndim != 1
        or grid_array.size < 3
        or not np.all(np.isfinite(grid_array))
        or np.any(np.diff(grid_array) <= 0.0)
    ):
        raise ValueError("research grid is not finite and strictly increasing")
    scalar_evidence = (
        result.current_A_m2,
        result.electron_continuity_bound_A_m2,
        result.hole_continuity_bound_A_m2,
        result.face_current_spread_A_m2,
        result.max_normalized_cell_residual,
        result.poisson_residual,
        result.poisson_residual_C_m2,
        result.interface_local_residual,
        result.numerical_residual_limit,
    )
    if not np.all(np.isfinite(np.asarray(scalar_evidence, dtype=float))):
        raise ValueError("operating-point certificate contains non-finite values")
    if any(value < 0.0 for value in scalar_evidence[1:]):
        raise ValueError("operating-point residual evidence must be non-negative")
    scalar_gates = (
        (
            result.electron_continuity_bound_A_m2,
            solver_controls["continuity_tolerance_A_m2"],
        ),
        (
            result.hole_continuity_bound_A_m2,
            solver_controls["continuity_tolerance_A_m2"],
        ),
        (
            result.face_current_spread_A_m2,
            solver_controls["current_spread_tolerance_A_m2"],
        ),
        (
            result.max_normalized_cell_residual,
            solver_controls["newton_residual_tolerance"],
        ),
        (
            result.poisson_residual,
            solver_controls["poisson_residual_tolerance"],
        ),
        (result.interface_local_residual, 1.0e-7),
        (
            result.numerical_residual_limit,
            solver_controls["newton_residual_tolerance"],
        ),
    )
    if any(value > limit for value, limit in scalar_gates):
        raise ValueError("operating-point certificate exceeds a research gate")
    if result.newton_iterations < 0 or result.residual_evaluations <= 0:
        raise ValueError("operating-point iteration evidence is invalid")
    state_arrays = (
        np.asarray(result.y, dtype=float),
        np.asarray(result.phi, dtype=float),
        np.asarray(result.electron_quasi_fermi_potential_V, dtype=float),
        np.asarray(result.hole_quasi_fermi_potential_V, dtype=float),
    )
    ion_blocks = 2 if bool(charge_off_material.has_dual_ions) else 1
    expected_state_size = (
        (2 + ion_blocks) * grid_array.size
        + 4 * int(charge_off_material.N_iface_state)
    )
    if (
        state_arrays[0].shape != (expected_state_size,)
        or any(array.shape != (grid_array.size,) for array in state_arrays[1:])
        or any(not np.all(np.isfinite(array)) for array in state_arrays)
    ):
        raise ValueError("operating-point state arrays are non-finite or misaligned")
    operating_state_sha256 = _backend_research_array_sha256(
        "interface-charge-api-operating-state-v1",
        grid_array,
        *state_arrays,
        equilibrium,
        occupancy,
        sheet_charge,
        trace_shift,
    )

    contact_payload = asdict(contact_certificate)
    return InterfaceChargeSteadyStateResearchResult(
        evidence_status="internal_numerical_research",
        capability_scope=(
            "steady_state_equilibrium_referenced_interface_charge"
        ),
        production_unlocked=False,
        provenance=InterfaceChargeResearchProvenance(
            requested_grid_intervals=int(req.N_grid),
            actual_grid_nodes=int(grid_array.size),
            interface_count=int(interface_count),
            interface_topology=result.interface_topology,
            interface_charge_closure=result.interface_charge_closure,
            research_acknowledged=bool(req.research_acknowledged),
        ),
        solver_controls=InterfaceChargeSolverControlsEvidence(
            **solver_controls,
            illumination_steps=tuple(float(v) for v in result.illumination_steps),
        ),
        contact_thermodynamics=InterfaceChargeContactEvidence(**contact_payload),
        dark_reference=InterfaceChargeDarkReferenceEvidence(
            certified=True,
            charge_on_off_bit_identical=dark_arrays_identical,
            grid_sha256=_require_research_sha256(
                "grid_sha256", dark_reference.grid_sha256
            ),
            stack_sha256=_require_research_sha256(
                "stack_sha256", dark_reference.stack_sha256
            ),
            dark_state_sha256=_require_research_sha256(
                "dark_state_sha256", dark_reference.dark_state_sha256
            ),
            interface_defect_document_sha256=tuple(
                _require_research_sha256(
                    f"interface_defect_document_sha256[{index}]",
                    value,
                )
                for index, value in enumerate(document_hashes)
            ),
            capture_velocities_m_s=tuple(
                (float(values[0]), float(values[1]))
                for values in capture_velocities
            ),
            equilibrium_occupancy=tuple(float(v) for v in equilibrium),
            trap_density_m2=tuple(float(v) for v in trap_density),
            incremental_sheet_charge_C_m2=tuple(float(v) for v in dark_charge),
            trace_potential_shift_V=tuple(
                (float(values[0]), float(values[1]))
                for values in dark_trace_shift
            ),
        ),
        operating_point=InterfaceChargeOperatingPointEvidence(
            V_app_V=float(result.V_app),
            illuminated=bool(result.illuminated),
            certified=True,
            current_density_A_m2=float(result.current_A_m2),
            electron_continuity_bound_A_m2=float(
                result.electron_continuity_bound_A_m2
            ),
            hole_continuity_bound_A_m2=float(
                result.hole_continuity_bound_A_m2
            ),
            face_current_spread_A_m2=float(result.face_current_spread_A_m2),
            max_normalized_cell_residual=float(
                result.max_normalized_cell_residual
            ),
            poisson_residual=float(result.poisson_residual),
            poisson_residual_C_m2=float(result.poisson_residual_C_m2),
            interface_local_residual=float(result.interface_local_residual),
            numerical_residual_limit=float(result.numerical_residual_limit),
            newton_iterations=int(result.newton_iterations),
            residual_evaluations=int(result.residual_evaluations),
            operating_state_sha256=operating_state_sha256,
        ),
        interfaces=tuple(
            InterfaceChargePerInterfaceEvidence(
                interface_index=index,
                equilibrium_occupancy=float(equilibrium[index]),
                occupancy=float(occupancy[index]),
                trap_density_m2=float(trap_density[index]),
                incremental_sheet_charge_C_m2=float(sheet_charge[index]),
                trace_potential_shift_left_V=float(trace_shift[index, 0]),
                trace_potential_shift_right_V=float(trace_shift[index, 1]),
                normalized_gauss_residual=float(gauss[index]),
                scaled_local_jacobian_condition=float(condition[index]),
            )
            for index in range(interface_count)
        ),
        limitations=(
            "internal numerical evidence only; no external solver validation",
            "equilibrium-referenced incremental charge is not absolute trap charge",
            "production J-V, transient, impedance, and 2D charge coupling remain parked",
        ),
    )


def _parse_experiment_protocol(
    payload: object | None,
) -> ExperimentProtocol | None:
    """Parse an API protocol payload without accepting partial schemas."""

    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ExperimentProtocolError(
            "experiment_protocol must be a JSON object"
        )
    try:
        return ExperimentProtocol.from_dict(payload)
    except ExperimentProtocolError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentProtocolError(
            f"invalid experiment_protocol payload: {exc}"
        ) from exc


def _parse_protocol_inputs(
    payload: object | None,
    mode: object,
) -> tuple[ExperimentProtocol | None, ProtocolMode]:
    """Validate protocol request fields before starting expensive work."""

    if mode not in ("compatibility", "research_strict"):
        raise ExperimentProtocolError(
            "protocol_mode must be 'compatibility' or 'research_strict'"
        )
    protocol = _parse_experiment_protocol(payload)
    if mode == "research_strict" and (
        protocol is None or protocol.implicit_legacy_protocol
    ):
        raise ImplicitProtocolError(
            "research_strict protocol mode requires an explicit experiment "
            "history with implicit_legacy_protocol=False"
        )
    return protocol, mode


_CHARGED_JV_JOB_PARAM_KEYS = frozenset({
    "N_grid",
    "n_points",
    "v_rate",
    "V_max",
    "illuminated",
    "solver",
    "iface_states",
    "interface_boundary",
    "interface_transport_model",
    "experiment_protocol",
    "protocol_mode",
    "interface_charge_jv_protocol",
})


def _resolve_interface_charge_jv_protocol(
    stack: DeviceStack,
    *,
    N_grid: object,
    n_points: object,
    v_rate: object,
    V_max: object | None,
    illuminated: object,
    solver: object,
    iface_states: object,
    interface_boundary: object,
    interface_transport_model: object,
    experiment_protocol: ExperimentProtocol | None,
    protocol_mode: ProtocolMode,
    supplied_protocol: object | None,
    request_param_keys: set[str] | None = None,
) -> interface_charge_jv_exp.InterfaceChargeJVProtocol | None:
    """Resolve the sole public protocol admitted for charged interface J-V."""

    if stack.interface_charge_closure == "off":
        if supplied_protocol is not None:
            raise interface_charge_jv_exp.InterfaceChargeJVProtocolError(
                "interface_charge_jv_protocol requires "
                "interface_charge_closure='equilibrium_referenced'"
            )
        return None

    if stack.interface_charge_closure != "equilibrium_referenced":
        raise interface_charge_jv_exp.InterfaceChargeJVProtocolError(
            "unsupported interface-charge closure for J-V execution"
        )

    violations: list[str] = []
    if request_param_keys is not None:
        extra = request_param_keys - _CHARGED_JV_JOB_PARAM_KEYS
        if extra:
            violations.append(
                "unknown charged J-V params: " + ", ".join(sorted(extra))
            )
    if isinstance(N_grid, (bool, np.bool_)) or not isinstance(
        N_grid, (int, np.integer)
    ) or int(N_grid) < 3:
        violations.append("N_grid must be an integer >= 3")
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(
        n_points, (int, np.integer)
    ) or int(n_points) < 2:
        violations.append("n_points must be an integer >= 2")
    try:
        rate = float(v_rate)
    except (TypeError, ValueError):
        rate = float("nan")
    if (
        isinstance(v_rate, (bool, np.bool_))
        or not np.isfinite(rate)
        or rate != 0.0
    ):
        violations.append("charged interface J-V requires v_rate=0")
    if illuminated is not True:
        violations.append("charged interface J-V requires illuminated=true")
    if solver != "quasi_fermi":
        violations.append("charged interface J-V requires solver='quasi_fermi'")
    if iface_states is not False:
        violations.append("charged interface J-V requires iface_states=false")
    if interface_boundary is not True:
        violations.append("charged interface J-V requires interface_boundary=true")
    if interface_transport_model != "fermi_dirac_richardson":
        violations.append(
            "charged interface J-V requires "
            "interface_transport_model='fermi_dirac_richardson'"
        )
    if experiment_protocol is not None or protocol_mode != "compatibility":
        violations.append(
            "charged interface J-V uses interface_charge_jv_protocol, not "
            "the transient ExperimentProtocol"
        )

    active_ions: list[str] = []
    explicit_bulk_defects: list[str] = []
    for layer in stack.layers:
        params = layer.params
        if params is None:
            continue
        if any(
            float(getattr(params, name)) != 0.0
            for name in ("D_ion", "P0", "D_ion_neg", "P0_neg")
        ):
            active_ions.append(layer.name)
        if (
            params.defect_model != "effective_lifetime"
            or bool(params.bulk_defects)
            or params.bulk_trap_distribution is not None
        ):
            explicit_bulk_defects.append(layer.name)
    if active_ions:
        violations.append(
            "charged interface J-V v1 is ion-free; active ion fields in "
            + ", ".join(active_ions)
        )
    if explicit_bulk_defects:
        violations.append(
            "charged interface J-V v1 excludes explicit bulk-defect "
            "composition; active layers: " + ", ".join(explicit_bulk_defects)
        )

    try:
        voltage_max = 1.25 if V_max is None else float(V_max)
    except (TypeError, ValueError):
        voltage_max = float("nan")
    if (
        isinstance(V_max, (bool, np.bool_))
        or not np.isfinite(voltage_max)
        or voltage_max <= 0.0
    ):
        violations.append("charged interface J-V requires finite V_max > 0")
    if violations:
        raise interface_charge_jv_exp.InterfaceChargeJVProtocolError(
            "; ".join(violations)
        )

    expected = interface_charge_jv_exp.build_interface_charge_jv_protocol(
        stack,
        np.linspace(0.0, voltage_max, int(n_points)),
    )
    if supplied_protocol is None:
        return expected
    if not isinstance(supplied_protocol, dict):
        raise interface_charge_jv_exp.InterfaceChargeJVProtocolError(
            "interface_charge_jv_protocol must be a JSON object"
        )
    resolved = interface_charge_jv_exp.InterfaceChargeJVProtocol.from_dict(
        supplied_protocol
    )
    if resolved != expected:
        raise interface_charge_jv_exp.InterfaceChargeJVProtocolError(
            "interface_charge_jv_protocol does not match the requested stack "
            "temperature or voltage sampling"
        )
    return resolved


def _preflight_job_experiment_protocol(
    kind: str,
    params: dict[str, Any],
    stack: DeviceStack,
    supplied: ExperimentProtocol | None,
    mode: ProtocolMode,
) -> None:
    """Reject protocol/execution mismatches before a worker is submitted."""

    if kind == "jv":
        solver = str(params.get("solver", "transient"))
        if solver != "transient":
            if supplied is not None or mode != "compatibility":
                raise ExperimentProtocolError(
                    "experiment protocols are supported only by "
                    "solver='transient'; "
                    f"solver={solver!r} has different steady-state semantics"
                )
            return
        raw_illuminated = params.get("illuminated", True)
        illuminated = (
            bool(raw_illuminated)
            if not isinstance(raw_illuminated, str)
            else raw_illuminated.lower() != "false"
        )
        expected = jv_sweep.build_jv_experiment_protocol(
            stack,
            n_points=int(params.get("n_points", 30)),
            v_rate=float(params.get("v_rate", 1.0)),
            V_max=(
                float(params["V_max"])
                if params.get("V_max") is not None
                else None
            ),
            illuminated=illuminated,
            implicit_legacy_protocol=True,
        )
    elif kind == "impedance":
        raw_illuminated = params.get("illuminated", True)
        illuminated = (
            bool(raw_illuminated)
            if not isinstance(raw_illuminated, str)
            else raw_illuminated.lower() != "false"
        )
        frequencies = np.logspace(
            np.log10(float(params.get("f_min", 10.0))),
            np.log10(float(params.get("f_max", 1e5))),
            int(params.get("n_freq", 15)),
        )
        expected = impedance.build_impedance_experiment_protocol(
            stack,
            frequencies,
            V_dc=float(params.get("V_dc", 0.9)),
            delta_V=float(params.get("delta_V", 0.01)),
            n_cycles=int(params.get("n_cycles", 5)),
            n_extract=int(params.get("n_extract", 2)),
            points_per_cycle=int(params.get("points_per_cycle", 40)),
            illuminated=illuminated,
            method=str(params.get("method", "transient_ion_aware")),
            dc_settle_time=float(params.get("dc_settle_time", 1e-3)),
            implicit_legacy_protocol=True,
        )
    elif kind == "tpv":
        expected = tpv_exp.build_tpv_experiment_protocol(
            stack,
            delta_G_frac=float(params.get("delta_G_frac", 0.05)),
            t_pulse=float(params.get("t_pulse", 1e-6)),
            t_decay=float(params.get("t_decay", 50e-6)),
            n_points=int(params.get("n_points", 200)),
            implicit_legacy_protocol=True,
        )
    elif kind == "suns_voc":
        raw_suns = params.get("suns_levels")
        suns_levels = (
            suns_voc_exp.DEFAULT_SUNS
            if raw_suns is None
            else tuple(float(value) for value in raw_suns)
        )
        expected = suns_voc_exp.build_suns_voc_experiment_protocol(
            stack,
            suns_levels,
            t_settle=float(params.get("t_settle", 1e-3)),
            implicit_legacy_protocol=True,
        )
    elif kind == "eqe":
        lambda_min_nm = float(params.get("lambda_min_nm", 300.0))
        lambda_max_nm = float(params.get("lambda_max_nm", 1000.0))
        n_lambda = int(params.get("n_lambda", 80))
        if n_lambda < 2 or lambda_max_nm <= lambda_min_nm:
            raise ValueError(
                "EQE sweep needs n_lambda >= 2 and lambda_max > lambda_min"
            )
        wavelengths_nm = np.linspace(lambda_min_nm, lambda_max_nm, n_lambda)
        expected = eqe_exp.build_eqe_experiment_protocol(
            stack,
            wavelengths_nm,
            Phi_incident=float(params.get("Phi_incident", 1e22)),
            t_settle=float(params.get("t_settle", 1e-1)),
            implicit_legacy_protocol=True,
        )
    else:  # pragma: no cover - guarded by the caller's protocol-kind set
        raise ValueError(f"unsupported protocol-bearing job kind {kind!r}")

    resolve_experiment_protocol(supplied, expected, mode=mode)


def _resolve_dynamic_defect_impedance_protocol(
    stack: DeviceStack,
    frequencies: np.ndarray,
    *,
    method: str,
    N_grid: int,
    V_dc: float,
    delta_V: float,
    illuminated: bool,
    defect_energy_quadrature_order: int,
    state_step: float,
    voltage_step: float,
    supplied: object | None,
) -> impedance.DynamicDefectImpedanceProtocol | None:
    dynamic_methods = {
        "dynamic_defect_frequency",
        impedance.DYNAMIC_DEFECT_IMPEDANCE_METHOD,
    }
    if method not in dynamic_methods:
        if supplied is not None:
            raise impedance.DynamicDefectImpedanceProtocolError(
                "dynamic_defect_protocol is valid only with the certified "
                "dynamic-defect frequency method"
            )
        return None
    parsed = (
        None
        if supplied is None
        else impedance.DynamicDefectImpedanceProtocol.from_dict(supplied)
    )
    grid = jv_sweep.build_electrical_grid(stack, N_grid)
    capability = impedance.classify_dynamic_defect_capability(stack)
    if "interface" in capability:
        grid = build_two_sided_trace_grid(grid, stack)
    expected = impedance.build_dynamic_defect_impedance_protocol(
        stack,
        grid,
        frequencies,
        requested_grid_intervals=N_grid,
        V_dc=V_dc,
        delta_V=delta_V,
        illuminated=illuminated,
        defect_energy_quadrature_order=defect_energy_quadrature_order,
        state_step=state_step,
        voltage_step=voltage_step,
    )
    return impedance.resolve_dynamic_defect_impedance_protocol(parsed, expected)


_DYNAMIC_DEFECT_TRANSIENT_JOB_PARAM_KEYS = frozenset(
    {
        "N_grid",
        "times_s",
        "voltage_V",
        "illuminated",
        "method",
        "dynamic_defect_transient_protocol",
    }
)


def _resolve_dynamic_defect_transient_protocol(
    stack: DeviceStack,
    *,
    method: object,
    N_grid: object,
    times_s: object,
    voltage_V: object,
    illuminated: object,
    supplied: object | None,
) -> tuple[np.ndarray, dynamic_defect_transient_exp.DynamicDefectTransientProtocol]:
    if method != dynamic_defect_transient_exp.DYNAMIC_DEFECT_TRANSIENT_METHOD:
        raise dynamic_defect_transient_exp.DynamicDefectTransientProtocolError(
            "dynamic_defect_transient_protocol requires method="
            f"{dynamic_defect_transient_exp.DYNAMIC_DEFECT_TRANSIENT_METHOD!r}"
        )
    if isinstance(N_grid, bool) or not isinstance(N_grid, int):
        raise TypeError("N_grid must be an integer")
    if N_grid < 4:
        raise ValueError("N_grid must be >= 4")
    if not isinstance(illuminated, bool):
        raise TypeError("illuminated must be boolean")
    parsed = (
        None
        if supplied is None
        else dynamic_defect_transient_exp.DynamicDefectTransientProtocol.from_dict(
            supplied
        )
    )
    grid = jv_sweep.build_electrical_grid(stack, N_grid)
    grid = build_two_sided_trace_grid(grid, stack)
    expected = dynamic_defect_transient_exp.build_dynamic_defect_transient_protocol(
        stack,
        grid,
        times_s,
        voltage_V,
        requested_grid_intervals=N_grid,
        illuminated=illuminated,
    )
    return (
        grid,
        dynamic_defect_transient_exp.resolve_dynamic_defect_transient_protocol(
            parsed,
            expected,
        ),
    )


def _summarize_qf_bulk_defect_evidence(points):
    """Collapse pointwise constitutive diagnostics without losing identity."""

    diagnostics = [getattr(point, "bulk_defect_diagnostics", None) for point in points]
    present = [item is not None for item in diagnostics]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "QF J-V bulk-defect diagnostics are incomplete across voltage points"
        )
    first = diagnostics[0]
    identity = (
        first.model_identity_sha256,
        tuple(first.species_identifiers),
        tuple(first.charge_transitions),
        tuple(getattr(first, "distribution_kinds", ())),
        tuple(getattr(first, "source_energy_orders", ())),
        tuple(getattr(first, "spatial_profile_sha256s", ())),
        tuple(getattr(first, "minimum_density_multipliers", ())),
        tuple(getattr(first, "maximum_density_multipliers", ())),
    )
    for item in diagnostics[1:]:
        candidate = (
            item.model_identity_sha256,
            tuple(item.species_identifiers),
            tuple(item.charge_transitions),
            tuple(getattr(item, "distribution_kinds", ())),
            tuple(getattr(item, "source_energy_orders", ())),
            tuple(getattr(item, "spatial_profile_sha256s", ())),
            tuple(getattr(item, "minimum_density_multipliers", ())),
            tuple(getattr(item, "maximum_density_multipliers", ())),
        )
        if candidate != identity:
            raise ValueError(
                "QF J-V bulk-defect identity changed across voltage points"
            )

    charge_maxima = []
    recombination_maxima = []
    for item in diagnostics:
        charge = np.asarray(item.total_charge_density_C_m3, dtype=float)
        recombination = np.asarray(
            item.total_recombination_rate_m3_s,
            dtype=float,
        )
        if (
            charge.size == 0
            or recombination.size == 0
            or not np.all(np.isfinite(charge))
            or not np.all(np.isfinite(recombination))
        ):
            raise ValueError("QF J-V bulk-defect diagnostics must be finite and non-empty")
        charge_maxima.append(float(np.max(np.abs(charge))))
        recombination_maxima.append(float(np.max(np.abs(recombination))))

    return jv_sweep.JVBulkDefectEvidence(
        model=MONOVALENT_BULK_DEFECT_MODEL_VERSION,
        model_identity_sha256=identity[0],
        species_identifiers=identity[1],
        charge_transitions=identity[2],
        points_completed=len(diagnostics),
        minimum_occupancy=min(float(item.minimum_occupancy) for item in diagnostics),
        maximum_occupancy=max(float(item.maximum_occupancy) for item in diagnostics),
        minimum_kinetic_denominator_s1=min(
            float(item.minimum_kinetic_denominator_s1) for item in diagnostics
        ),
        maximum_absolute_charge_density_C_m3=max(charge_maxima),
        maximum_absolute_recombination_rate_m3_s=max(recombination_maxima),
        distribution_kinds=identity[3],
        source_energy_orders=identity[4],
        spatial_closure=(
            "layer-density-profile-v1" if identity[5] else None
        ),
        spatial_profile_sha256s=identity[5],
        minimum_density_multipliers=identity[6],
        maximum_density_multipliers=identity[7],
    )


def _run_jv_dispatch(
    stack,
    *,
    N_grid: int,
    n_points: int,
    v_rate: float,
    V_max: Optional[float],
    illuminated: bool,
    solver: str = "transient",
    iface_states: bool = False,
    interface_boundary: bool = False,
    interface_transport_model: str = "fermi_richardson",
    experiment_protocol: ExperimentProtocol | None = None,
    protocol_mode: ProtocolMode = "compatibility",
    interface_charge_jv_protocol: (
        interface_charge_jv_exp.InterfaceChargeJVProtocol | None
    ) = None,
    progress=None,
):
    """Route a J-V sweep to the requested solver.

    ``solver="transient"`` (default) runs the legacy Radau forward/reverse
    sweep. ``solver="steady_state"`` and ``solver="quasi_fermi"`` produce one
    zero-scan-rate curve, so they are wrapped with forward == reverse and zero
    hysteresis. Stack policy is enforced inside each driver; no implicit
    solver substitution is permitted.
    """
    interface_charge_closure = getattr(stack, "interface_charge_closure", "off")
    if interface_charge_closure == "equilibrium_referenced":
        if interface_charge_jv_protocol is None:
            raise interface_charge_jv_exp.InterfaceChargeJVProtocolError(
                "charged interface J-V requires a resolved protocol"
            )
        shared_grid = jv_sweep.build_electrical_grid(stack, N_grid)
        require_thick_layer_interface_resolution(
            shared_grid,
            stack,
            N_grid=N_grid,
            allow_underresolved_grid=False,
        )
        grid = build_two_sided_trace_grid(shared_grid, stack)
        execution = interface_charge_jv_exp.solve_interface_charge_jv(
            grid,
            stack,
            interface_charge_jv_protocol,
            progress=progress,
        )
        sweep = execution.sweep

        def _charged_statuses(branch: str):
            return tuple(
                jv_sweep.JVPointStatus(
                    branch=branch,
                    index=index,
                    voltage=float(point.V_app),
                    valid=bool(point.certified),
                    attempted_currents=(float(point.current_A_m2),),
                    reason_code="certified_interface_charge_qf",
                    message=(
                        "equilibrium-referenced interface-charge QF/DC "
                        "certificate"
                    ),
                    candidate_current=float(point.current_A_m2),
                    solver="quasi_fermi",
                    max_normalized_residual=float(
                        point.max_normalized_cell_residual
                    ),
                    electron_continuity_bound_A_m2=float(
                        point.electron_continuity_bound_A_m2
                    ),
                    hole_continuity_bound_A_m2=float(
                        point.hole_continuity_bound_A_m2
                    ),
                    face_current_spread_A_m2=float(
                        point.face_current_spread_A_m2
                    ),
                    poisson_residual=float(point.poisson_residual),
                )
                for index, point in enumerate(sweep.points)
            )

        return jv_sweep.JVResult(
            V_fwd=sweep.voltages_V,
            J_fwd=sweep.currents_A_m2,
            V_rev=sweep.voltages_V,
            J_rev=sweep.currents_A_m2,
            metrics_fwd=sweep.metrics,
            metrics_rev=sweep.metrics,
            hysteresis_index=0.0,
            status_fwd=_charged_statuses("jv_forward"),
            status_rev=_charged_statuses("jv_reverse"),
            interface_charge_evidence=execution.evidence,
        )
    if interface_charge_jv_protocol is not None:
        raise interface_charge_jv_exp.InterfaceChargeJVProtocolError(
            "interface_charge_jv_protocol cannot be used with charge-off J-V"
        )
    if solver != "transient" and (
        experiment_protocol is not None or protocol_mode != "compatibility"
    ):
        raise ExperimentProtocolError(
            "experiment protocols are supported only by solver='transient'; "
            f"solver={solver!r} has different steady-state semantics"
        )
    if interface_boundary and solver != "quasi_fermi":
        raise ValueError(
            "interface_boundary requires solver='quasi_fermi'"
        )
    if (
        not interface_boundary
        and interface_transport_model != "fermi_richardson"
    ):
        raise ValueError(
            "interface_transport_model requires interface_boundary=true"
        )
    if solver == "steady_state":
        ss = run_jv_sweep_ss(
            stack,
            N_grid=N_grid,
            n_points=n_points,
            V_max=V_max if V_max is not None else 1.25,
            illuminated=illuminated,
            iface_states=iface_states,
            progress=progress,
        )

        point_acceptance = getattr(ss, "point_acceptance", ())
        if point_acceptance is None:
            point_acceptance = ()
        point_residual = getattr(ss, "point_residual", None)
        point_current_bound = getattr(
            ss,
            "point_continuity_current_bound",
            getattr(ss, "point_current_bound", None),
        )

        def _has_point_count(values) -> bool:
            try:
                return len(values) == len(ss.V)
            except TypeError:
                return False

        metadata_aligned = all(
            _has_point_count(values)
            for values in (
                point_acceptance,
                point_residual,
                point_current_bound,
            )
        )

        def _point_value(values, index: int) -> float | None:
            if values is None:
                return None
            try:
                value = float(values[index])
            except (IndexError, KeyError, TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        def _steady_state_statuses(branch: str):
            statuses = []
            upstream_valid = True
            for index, (voltage, current) in enumerate(zip(ss.V, ss.J)):
                try:
                    acceptance = str(point_acceptance[index])
                except (IndexError, TypeError):
                    acceptance = "not_reported"
                residual = _point_value(point_residual, index)
                current_bound = _point_value(point_current_bound, index)

                if acceptance == "residual_converged":
                    evidence_complete = (
                        metadata_aligned
                        and residual is not None
                        and current_bound is not None
                    )
                    message = "strict steady-state residual gate satisfied"
                elif acceptance == "current_bounded_stall":
                    evidence_complete = (
                        metadata_aligned
                        and residual is not None
                        and current_bound is not None
                    )
                    message = (
                        "steady-state stall accepted under the continuity-"
                        "current bound"
                    )
                elif acceptance == "transient_assisted":
                    evidence_complete = False
                    message = (
                        "transient fallback point; no steady-state Newton "
                        "certificate"
                    )
                elif acceptance == "not_reported":
                    evidence_complete = False
                    message = "steady-state point acceptance was not reported"
                else:
                    evidence_complete = False
                    message = (
                        f"unknown steady-state point acceptance {acceptance!r}"
                    )
                if not metadata_aligned:
                    message += "; point metadata is incomplete or misaligned"

                local_valid = (
                    acceptance in {
                        "residual_converged",
                        "current_bounded_stall",
                    }
                    and evidence_complete
                )
                valid = upstream_valid and local_valid
                statuses.append(jv_sweep.JVPointStatus(
                    branch=branch,
                    index=index,
                    voltage=float(voltage),
                    valid=valid,
                    upstream_valid=upstream_valid,
                    attempted_currents=(float(current),),
                    reason_code=acceptance,
                    message=message,
                    candidate_current=float(current),
                    solver="steady_state",
                    max_normalized_residual=residual,
                    electron_continuity_bound_A_m2=current_bound,
                    hole_continuity_bound_A_m2=current_bound,
                ))
                upstream_valid = valid
            return tuple(statuses)

        return jv_sweep.JVResult(
            V_fwd=ss.V, J_fwd=ss.J, V_rev=ss.V, J_rev=ss.J,
            metrics_fwd=ss.metrics, metrics_rev=ss.metrics,
            hysteresis_index=0.0,
            status_fwd=_steady_state_statuses("jv_forward"),
            status_rev=_steady_state_statuses("jv_reverse"),
        )
    if solver == "quasi_fermi":
        if not illuminated:
            raise ValueError(
                "solver='quasi_fermi' currently certifies illuminated J-V "
                "only; use the transient solver for dark J-V"
            )
        x = jv_sweep.build_electrical_grid(stack, N_grid)
        require_thick_layer_interface_resolution(
            x,
            stack,
            N_grid=N_grid,
            allow_underresolved_grid=False,
        )
        jv_sweep.require_jv_driver_capability(
            stack,
            requested_driver="quasi_fermi",
        )
        qf = solve_quasi_fermi_jv_sweep(
            x,
            stack,
            np.linspace(
                0.0,
                V_max if V_max is not None else 1.25,
                n_points,
            ),
            interface_boundary=interface_boundary,
            interface_transport_model=interface_transport_model,
            stop_after_voc=True,
        )

        def _statuses(branch: str):
            return tuple(
                jv_sweep.JVPointStatus(
                    branch=branch,
                    index=index,
                    voltage=float(point.V_app),
                    valid=bool(point.certified),
                    attempted_currents=(float(point.current_A_m2),),
                    reason_code="certified_qf",
                    message="cancellation-safe quasi-Fermi certificate",
                    candidate_current=float(point.current_A_m2),
                    solver="quasi_fermi",
                    max_normalized_residual=float(
                        point.max_normalized_cell_residual
                    ),
                    electron_continuity_bound_A_m2=float(
                        point.electron_continuity_bound_A_m2
                    ),
                    hole_continuity_bound_A_m2=float(
                        point.hole_continuity_bound_A_m2
                    ),
                    face_current_spread_A_m2=float(
                        point.face_current_spread_A_m2
                    ),
                    poisson_residual=float(point.poisson_residual),
                )
                for index, point in enumerate(qf.points)
            )

        return jv_sweep.JVResult(
            V_fwd=qf.voltages_V,
            J_fwd=qf.currents_A_m2,
            V_rev=qf.voltages_V,
            J_rev=qf.currents_A_m2,
            metrics_fwd=qf.metrics,
            metrics_rev=qf.metrics,
            hysteresis_index=0.0,
            status_fwd=_statuses("jv_forward"),
            status_rev=_statuses("jv_reverse"),
            bulk_defect_evidence=_summarize_qf_bulk_defect_evidence(qf.points),
        )
    if solver != "transient":
        raise ValueError(f"unknown solver {solver!r}")
    return jv_sweep.run_jv_sweep(
        stack, N_grid=N_grid, n_points=n_points, v_rate=v_rate,
        V_max=V_max, illuminated=illuminated, progress=progress,
        experiment_protocol=experiment_protocol, protocol_mode=protocol_mode,
    )


@app.post("/api/jv")
def run_jv(req: JVRequest):
    try:
        experiment_protocol, protocol_mode = _parse_protocol_inputs(
            req.experiment_protocol,
            req.protocol_mode,
        )
        stack = build_jv_stack(req.config_path, req.device)
        charged_protocol = _resolve_interface_charge_jv_protocol(
            stack,
            N_grid=req.N_grid,
            n_points=req.n_points,
            v_rate=req.v_rate,
            V_max=req.V_max,
            illuminated=True,
            solver=req.solver,
            iface_states=req.iface_states,
            interface_boundary=req.interface_boundary,
            interface_transport_model=req.interface_transport_model,
            experiment_protocol=experiment_protocol,
            protocol_mode=protocol_mode,
            supplied_protocol=req.interface_charge_jv_protocol,
        )
        result = _run_jv_dispatch(
            stack, N_grid=req.N_grid, n_points=req.n_points, v_rate=req.v_rate,
            V_max=req.V_max, illuminated=True, solver=req.solver,
            iface_states=req.iface_states,
            interface_boundary=req.interface_boundary,
            interface_transport_model=req.interface_transport_model,
            experiment_protocol=experiment_protocol,
            protocol_mode=protocol_mode,
            interface_charge_jv_protocol=charged_protocol,
        )
        return {"status": "ok", "result": to_serializable(result)}
    except HTTPException:
        raise
    except (
        ExperimentProtocolError,
        GridResolutionError,
        jv_sweep.JVDriverCapabilityError,
        interface_charge_jv_exp.InterfaceChargeJVProtocolError,
        interface_charge_jv_exp.InterfaceChargeJVCertificationError,
        QuasiFermiSteadyStateError,
        TypeError,
        ValueError,
    ) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print("[JV API Exception]", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/jv/external-circuit")
def run_external_circuit_jv(req: ExternalCircuitJVRequest):
    """Run an intrinsic J-V experiment, then map it to terminal coordinates."""

    try:
        circuit = external_circuit_exp.ExternalCircuitProtocol.from_dict(
            req.external_circuit_protocol
        )
        if (
            not np.isfinite(req.incident_power_W_m2)
            or req.incident_power_W_m2 <= 0.0
        ):
            raise ValueError("incident_power_W_m2 must be positive and finite")
        experiment_protocol, protocol_mode = _parse_protocol_inputs(
            req.experiment_protocol,
            req.protocol_mode,
        )
    except (ExperimentProtocolError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        stack = build_stack(req.config_path, req.device)
        intrinsic = _run_jv_dispatch(
            stack,
            N_grid=req.N_grid,
            n_points=req.n_points,
            v_rate=req.v_rate,
            V_max=req.V_max,
            illuminated=True,
            solver=req.solver,
            iface_states=req.iface_states,
            interface_boundary=req.interface_boundary,
            interface_transport_model=req.interface_transport_model,
            experiment_protocol=experiment_protocol,
            protocol_mode=protocol_mode,
        )
        result = external_circuit_exp.apply_external_circuit(
            intrinsic,
            circuit,
            incident_power_W_m2=req.incident_power_W_m2,
        )
        return {"status": "ok", "result": to_serializable(result)}
    except HTTPException:
        raise
    except (
        ExperimentProtocolError,
        GridResolutionError,
        external_circuit_exp.ExternalCircuitError,
        jv_sweep.JVDriverCapabilityError,
    ) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        print("[External Circuit J-V API Exception]", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/jv/electrothermal-operating-point")
def run_electrothermal_operating_point(req: ElectrothermalOperatingPointRequest):
    """Solve one protocol-conditioned terminal-MPP electrothermal root."""

    try:
        thermal = thermal_balance_exp.LumpedThermalProtocol.from_dict(
            req.thermal_protocol
        )
        circuit = external_circuit_exp.ExternalCircuitProtocol.from_dict(
            req.external_circuit_protocol
        )
        electrical = electrothermal_exp.ElectrothermalJVProtocol.from_dict(
            req.electrical_protocol
        )
        operating = (
            electrothermal_exp.ElectrothermalOperatingPointProtocol.from_dict(
                req.operating_protocol
            )
        )
    except (
        electrothermal_exp.ElectrothermalError,
        external_circuit_exp.ExternalCircuitError,
        thermal_balance_exp.ThermalBalanceError,
        TypeError,
        ValueError,
    ) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        stack = build_stack(req.config_path, req.device)
        result = electrothermal_exp.solve_electrothermal_operating_point(
            stack,
            thermal,
            circuit,
            electrical,
            operating,
        )
        return {"status": "ok", "result": to_serializable(result)}
    except HTTPException:
        raise
    except (
        ExperimentProtocolError,
        GridResolutionError,
        electrothermal_exp.ElectrothermalError,
        external_circuit_exp.ExternalCircuitError,
        thermal_balance_exp.ThermalBalanceError,
        jv_sweep.JVDriverCapabilityError,
    ) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        print("[Electrothermal Operating Point API Exception]", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/identifiability/interface-srh-synthetic")
def run_interface_srh_identifiability(req: InterfaceSRHIdentifiabilityRequest):
    """Run synthetic recovery/rank evidence without claiming material values."""

    try:
        protocol = identifiability_exp.InterfaceSRHIdentifiabilityProtocol.from_dict(
            req.protocol
        )
    except (identifiability_exp.IdentifiabilityError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = identifiability_exp.run_interface_srh_identifiability(protocol)
        return {"status": "ok", "result": result.to_dict()}
    except identifiability_exp.IdentifiabilityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        print("[Interface SRH Identifiability API Exception]", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/api/research/interface-charge/steady-state",
    response_model=InterfaceChargeSteadyStateResearchResponse,
)
def run_interface_charge_steady_state_research(
    req: InterfaceChargeSteadyStateResearchRequest,
):
    """Run the only backend-exposed equilibrium-referenced charge workflow."""
    try:
        if not req.research_acknowledged:
            raise HTTPException(
                status_code=422,
                detail=(
                    "research_acknowledged=true is required; this endpoint "
                    "does not unlock production interface charge"
                ),
            )
        if req.N_grid < 12 or req.N_grid > 240:
            raise HTTPException(
                status_code=422,
                detail="N_grid must lie in the closed interval [12, 240]",
            )
        if not np.isfinite(req.V_app) or abs(req.V_app) > 2.0:
            raise HTTPException(
                status_code=422,
                detail="V_app must be finite and lie in [-2, 2] V",
            )

        stack = build_interface_charge_research_stack(
            req.config_path,
            req.device,
        )
        shared_grid = jv_sweep.build_electrical_grid(stack, req.N_grid)
        grid = build_two_sided_trace_grid(shared_grid, stack)
        charge_off_stack = replace(stack, interface_charge_closure="off")
        charge_off_material = build_material_arrays(grid, charge_off_stack)
        if charge_off_material.iface_state_charge != 0.0:
            raise ValueError("legacy shared-node interface charge must remain zero")
        contact_certificate = require_contact_thermodynamic_certificate(
            charge_off_stack,
            charge_off_material,
        )
        solver_controls = _interface_charge_research_solver_controls()
        dark_reference = (
            build_equilibrium_referenced_interface_charge_dark_reference(
                grid,
                stack,
                interface_transmission=1.0,
                **solver_controls,
            )
        )
        charged_dark = (
            solve_equilibrium_referenced_interface_charge_steady_state(
                grid,
                stack,
                0.0,
                dark_reference=dark_reference,
                illuminated=False,
                **solver_controls,
            )
        )
        result = solve_equilibrium_referenced_interface_charge_steady_state(
            grid,
            stack,
            float(req.V_app),
            dark_reference=dark_reference,
            illuminated=bool(req.illuminated),
            **solver_controls,
        )
        evidence = _build_interface_charge_research_result(
            req,
            grid,
            charge_off_material,
            contact_certificate,
            dark_reference,
            charged_dark,
            result,
            solver_controls,
        )
        return InterfaceChargeSteadyStateResearchResponse(
            status="ok",
            result=evidence,
        )
    except HTTPException:
        raise
    except (
        ContactThermodynamicError,
        GridResolutionError,
        QuasiFermiSteadyStateError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        print("[Interface Charge Research API Exception]", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Tandem endpoint
# ---------------------------------------------------------------------------

class TandemRequest(BaseModel):
    config_path: str
    N_grid: int = 40
    n_points: int = 15


@app.post("/api/tandem")
def run_tandem(req: TandemRequest):
    """Run a series-connected 2T tandem J-V sweep from a tandem YAML config.

    Loads the tandem config, builds the AM1.5G wavelength grid using the
    same parameters as ``_compute_tmm_generation`` (300–1000 nm, 200 points),
    calls ``run_tandem_jv``, and returns the series-matched J-V together with
    per-sub-cell voltages and four tandem metrics.
    """
    import dataclasses

    import numpy as np
    from perovskite_sim.data import load_am15g
    from perovskite_sim.experiments.tandem_jv import run_tandem_jv
    from perovskite_sim.models.tandem_config import load_tandem_from_yaml

    # --- load config -------------------------------------------------------
    try:
        cfg = load_tandem_from_yaml(resolve_config_path(req.config_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # --- build wavelength grid (same defaults as _compute_tmm_generation) --
    try:
        lam_min, lam_max, n_wl = 300.0, 1000.0, 200
        wavelengths_nm = np.linspace(lam_min, lam_max, n_wl)
        wavelengths_m = wavelengths_nm * 1e-9
        _, spectral_flux = load_am15g(wavelengths_nm)

        result = run_tandem_jv(
            cfg,
            wavelengths_m,
            spectral_flux,
            wavelengths_nm,
            N_grid=req.N_grid,
            n_points=req.n_points,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        # series_match_jv raises when sub-cell J ranges do not overlap, which
        # happens when the stub FA_Cs_1p77 / SnPb_1p22 n,k CSVs mis-match the
        # real Lin 2019 spectral response. Surface a 400 with a clear pointer.
        msg = str(exc)
        if "Sub-cell J ranges do not overlap" in msg:
            detail = (
                f"{msg} — this tandem preset ships with stub n,k data "
                "(rigid bandgap shifts of MAPbI3). Replace "
                "perovskite_sim/data/nk/FA_Cs_1p77.csv and SnPb_1p22.csv "
                "with real Lin 2019 SI data before expecting physical results."
            )
            raise HTTPException(status_code=400, detail=detail)
        raise HTTPException(status_code=400, detail=msg)
    except Exception as exc:
        print("[Tandem API Exception]", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

    # --- serialise metrics (frozen dataclass → dict) -----------------------
    metrics_dict = dataclasses.asdict(result.metrics)

    return {
        "V": result.V.tolist(),
        "J": result.J.tolist(),
        "V_top": result.V_top.tolist(),
        "V_bot": result.V_bot.tolist(),
        "metrics": metrics_dict,
        "benchmark": cfg.benchmark,
    }


class ISRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_path: Optional[str] = None
    device: Optional[dict] = None
    N_grid: int = 40
    V_dc: float = 0.9
    n_freq: int = 15
    f_min: float = 10.0
    f_max: float = 1e5
    delta_V: float = 0.01
    n_cycles: int = 5
    n_extract: int = 2
    points_per_cycle: int = 40
    dc_settle_time: float = 1e-3
    illuminated: bool = True
    method: Literal[
        "transient",
        "transient_ion_aware",
        "quasi_fermi_frequency",
        "qf_frequency_ion_free",
        "ion_aware_frequency",
        "ion_aware_frequency_certified",
        "dynamic_defect_frequency",
        "dynamic_defect_frequency_certified",
    ] = "transient_ion_aware"
    require_operating_point_certificate: bool = False
    require_frequency_window_certificate: bool = False
    protocol_mode: ProtocolMode = "compatibility"
    experiment_protocol: Optional[dict[str, Any]] = None
    defect_energy_quadrature_order: int = 32
    dynamic_defect_state_step: float = 1.0e-5
    dynamic_defect_voltage_step: float = 1.0e-5
    dynamic_defect_protocol: Optional[dict[str, Any]] = None


@app.post("/api/impedance")
def run_impedance_api(req: ISRequest):
    try:
        experiment_protocol, protocol_mode = _parse_protocol_inputs(
            req.experiment_protocol,
            req.protocol_mode,
        )
        stack = build_stack(req.config_path, req.device)
        frequencies = np.logspace(np.log10(req.f_min), np.log10(req.f_max), req.n_freq)
        dynamic_defect_protocol = _resolve_dynamic_defect_impedance_protocol(
            stack,
            frequencies,
            method=req.method,
            N_grid=req.N_grid,
            V_dc=req.V_dc,
            delta_V=req.delta_V,
            illuminated=req.illuminated,
            defect_energy_quadrature_order=req.defect_energy_quadrature_order,
            state_step=req.dynamic_defect_state_step,
            voltage_step=req.dynamic_defect_voltage_step,
            supplied=req.dynamic_defect_protocol,
        )
        dynamic_kwargs = (
            {}
            if dynamic_defect_protocol is None
            else {
                "dynamic_defect_protocol": dynamic_defect_protocol,
                "defect_energy_quadrature_order": (
                    req.defect_energy_quadrature_order
                ),
                "dynamic_defect_state_step": req.dynamic_defect_state_step,
                "dynamic_defect_voltage_step": req.dynamic_defect_voltage_step,
            }
        )
        result = impedance.run_impedance(
            stack,
            frequencies,
            V_dc=req.V_dc,
            delta_V=req.delta_V,
            N_grid=req.N_grid,
            n_cycles=req.n_cycles,
            n_extract=req.n_extract,
            points_per_cycle=req.points_per_cycle,
            illuminated=req.illuminated,
            method=req.method,
            dc_settle_time=req.dc_settle_time,
            require_operating_point_certificate=(
                req.require_operating_point_certificate
            ),
            require_frequency_window_certificate=(
                req.require_frequency_window_certificate
            ),
            experiment_protocol=experiment_protocol,
            protocol_mode=protocol_mode,
            **dynamic_kwargs,
        )
        out = to_serializable(result)
        if "Z" in out:
            Z = np.array(result.Z)
            out["Z_real"] = Z.real.tolist()
            out["Z_imag"] = Z.imag.tolist()
            del out["Z"]
        return {"status": "ok", "result": out}
    except HTTPException:
        raise
    except (
        ExperimentProtocolError,
        GridResolutionError,
        impedance.DynamicDefectImpedanceCapabilityError,
        impedance.DynamicDefectImpedanceProtocolError,
        impedance.ImpedanceCapabilityError,
        impedance.ImpedanceCertificationError,
        QuasiFermiSteadyStateError,
        TypeError,
        ValueError,
    ) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        print("[Impedance API Exception]", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class DynamicDefectTransientRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_path: Optional[str] = None
    device: Optional[dict] = None
    N_grid: StrictInt = 4
    times_s: tuple[float, ...] = (0.0, 1.0e-8, 1.0e-6, 1.0e-4)
    voltage_V: tuple[float, ...] = (0.0, 0.05, 0.05, 0.05)
    illuminated: StrictBool = False
    method: Literal["dynamic_defect_transient_certified"] = (
        "dynamic_defect_transient_certified"
    )
    dynamic_defect_transient_protocol: Optional[dict[str, Any]] = None


@app.post("/api/dynamic-defect-transient")
def run_dynamic_defect_transient_api(req: DynamicDefectTransientRequest):
    try:
        stack = build_dynamic_defect_transient_stack(req.config_path, req.device)
        grid, protocol = _resolve_dynamic_defect_transient_protocol(
            stack,
            method=req.method,
            N_grid=req.N_grid,
            times_s=req.times_s,
            voltage_V=req.voltage_V,
            illuminated=req.illuminated,
            supplied=req.dynamic_defect_transient_protocol,
        )
        result = dynamic_defect_transient_exp.run_dynamic_defect_transient(
            grid,
            stack,
            protocol,
        )
        out = to_serializable(result)
        out["active_physics"] = _describe_active_physics(stack)
        return {"status": "ok", "result": out}
    except HTTPException:
        raise
    except (
        dynamic_defect_transient_exp.DynamicDefectTransientCapabilityError,
        dynamic_defect_transient_exp.DynamicDefectTransientCertificationError,
        dynamic_defect_transient_exp.DynamicDefectTransientProtocolError,
        GridResolutionError,
        TypeError,
        ValueError,
    ) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class DegRequest(BaseModel):
    config_path: Optional[str] = None
    device: Optional[dict] = None
    t_end: float = 100.0
    n_snapshots: int = 10
    N_grid: int = 40
    V_bias: float = 0.9
    metric_V_max: Optional[float] = None
    metric_settle_time: float = 1e-3


class JobRequest(BaseModel):
    kind: str  # "jv" | "impedance" | "degradation" | "tpv" | "current_decomp" | "spatial"
               # | "dark_jv" | "suns_voc" | "voc_t" | "eqe" | "el" | "mott_schottky"
               # | "tandem" | "dynamic_defect_transient"
    config_path: Optional[str] = None
    device: Optional[dict] = None
    params: dict = {}


@app.post("/api/jobs")
def start_job(req: JobRequest):
    """Start an experiment on a worker thread and return a job ID.

    The caller then opens GET /api/jobs/{id}/events to receive
    Server-Sent-Events with incremental progress and the final result.
    """
    kind = req.kind
    p = req.params

    experiment_protocol: ExperimentProtocol | None = None
    protocol_mode: ProtocolMode = "compatibility"
    interface_charge_jv_protocol: (
        interface_charge_jv_exp.InterfaceChargeJVProtocol | None
    ) = None
    dynamic_defect_protocol: (
        impedance.DynamicDefectImpedanceProtocol | None
    ) = None
    dynamic_defect_transient_grid: np.ndarray | None = None
    dynamic_defect_transient_protocol: (
        dynamic_defect_transient_exp.DynamicDefectTransientProtocol | None
    ) = None
    if kind in {"jv", "impedance", "tpv", "suns_voc", "eqe"}:
        try:
            experiment_protocol, protocol_mode = _parse_protocol_inputs(
                p.get("experiment_protocol"),
                p.get("protocol_mode", "compatibility"),
            )
        except ExperimentProtocolError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Tandem is config-only (no single DeviceStack), so it skips build_stack.
    if kind != "tandem":
        try:
            if kind == "jv":
                stack = build_jv_stack(req.config_path, req.device)
            elif kind == "dynamic_defect_transient":
                stack = build_dynamic_defect_transient_stack(
                    req.config_path,
                    req.device,
                )
            else:
                stack = build_stack(req.config_path, req.device)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"stack build failed: {e}")

    if kind in {"jv", "impedance", "tpv", "suns_voc", "eqe"}:
        try:
            if kind == "jv":
                interface_charge_jv_protocol = (
                    _resolve_interface_charge_jv_protocol(
                        stack,
                        N_grid=p.get("N_grid", 60),
                        n_points=p.get("n_points", 30),
                        v_rate=p.get("v_rate", 1.0),
                        V_max=p.get("V_max"),
                        illuminated=p.get("illuminated", True),
                        solver=p.get("solver", "transient"),
                        iface_states=p.get("iface_states", False),
                        interface_boundary=p.get(
                            "interface_boundary", False
                        ),
                        interface_transport_model=p.get(
                            "interface_transport_model",
                            "fermi_richardson",
                        ),
                        experiment_protocol=experiment_protocol,
                        protocol_mode=protocol_mode,
                        supplied_protocol=p.get(
                            "interface_charge_jv_protocol"
                        ),
                        request_param_keys=set(p),
                    )
                )
            if interface_charge_jv_protocol is None:
                _preflight_job_experiment_protocol(
                    kind,
                    p,
                    stack,
                    experiment_protocol,
                    protocol_mode,
                )
            if kind == "impedance":
                raw_illuminated = p.get("illuminated", True)
                illuminated = (
                    bool(raw_illuminated)
                    if not isinstance(raw_illuminated, str)
                    else raw_illuminated.lower() != "false"
                )
                frequencies = np.logspace(
                    np.log10(float(p.get("f_min", 10.0))),
                    np.log10(float(p.get("f_max", 1e5))),
                    int(p.get("n_freq", 15)),
                )
                dynamic_defect_protocol = (
                    _resolve_dynamic_defect_impedance_protocol(
                        stack,
                        frequencies,
                        method=str(
                            p.get("method", "transient_ion_aware")
                        ),
                        N_grid=int(p.get("N_grid", 40)),
                        V_dc=float(p.get("V_dc", 0.9)),
                        delta_V=float(p.get("delta_V", 0.01)),
                        illuminated=illuminated,
                        defect_energy_quadrature_order=int(
                            p.get("defect_energy_quadrature_order", 32)
                        ),
                        state_step=float(
                            p.get("dynamic_defect_state_step", 1.0e-5)
                        ),
                        voltage_step=float(
                            p.get("dynamic_defect_voltage_step", 1.0e-5)
                        ),
                        supplied=p.get("dynamic_defect_protocol"),
                    )
                )
        except (
            ExperimentProtocolError,
            interface_charge_jv_exp.InterfaceChargeJVProtocolError,
            impedance.DynamicDefectImpedanceCapabilityError,
            impedance.DynamicDefectImpedanceProtocolError,
            TypeError,
            ValueError,
        ) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    if kind == "dynamic_defect_transient":
        try:
            extra = set(p) - _DYNAMIC_DEFECT_TRANSIENT_JOB_PARAM_KEYS
            if extra:
                raise dynamic_defect_transient_exp.DynamicDefectTransientProtocolError(
                    "dynamic_defect_transient job parameters do not match schema; "
                    f"extra={sorted(extra)}"
                )
            dynamic_defect_transient_grid, dynamic_defect_transient_protocol = (
                _resolve_dynamic_defect_transient_protocol(
                    stack,
                    method=p.get(
                        "method",
                        dynamic_defect_transient_exp.DYNAMIC_DEFECT_TRANSIENT_METHOD,
                    ),
                    N_grid=p.get("N_grid", 4),
                    times_s=p.get("times_s", (0.0, 1.0e-8, 1.0e-6, 1.0e-4)),
                    voltage_V=p.get("voltage_V", (0.0, 0.05, 0.05, 0.05)),
                    illuminated=p.get("illuminated", False),
                    supplied=p.get("dynamic_defect_transient_protocol"),
                )
            )
        except (
            dynamic_defect_transient_exp.DynamicDefectTransientCapabilityError,
            dynamic_defect_transient_exp.DynamicDefectTransientProtocolError,
            GridResolutionError,
            TypeError,
            ValueError,
        ) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    if kind == "jv":
        def _run(reporter: ProgressReporter) -> dict:
            # illuminated defaults to True; frontend sends False for dark J-V
            _illum = p.get("illuminated", True)
            illuminated = bool(_illum) if not isinstance(_illum, str) else _illum.lower() != "false"
            result = _run_jv_dispatch(
                stack,
                N_grid=int(p.get("N_grid", 60)),
                n_points=int(p.get("n_points", 30)),
                v_rate=float(p.get("v_rate", 1.0)),
                V_max=float(p["V_max"]) if p.get("V_max") is not None else None,
                illuminated=illuminated,
                solver=str(p.get("solver", "transient")),
                iface_states=bool(p.get("iface_states", False)),
                interface_boundary=bool(p.get("interface_boundary", False)),
                interface_transport_model=str(
                    p.get("interface_transport_model", "fermi_richardson")
                ),
                experiment_protocol=experiment_protocol,
                protocol_mode=protocol_mode,
                interface_charge_jv_protocol=interface_charge_jv_protocol,
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "impedance":
        def _run(reporter: ProgressReporter) -> dict:
            _illum = p.get("illuminated", True)
            illuminated = (
                bool(_illum)
                if not isinstance(_illum, str)
                else _illum.lower() != "false"
            )
            _strict = p.get("require_operating_point_certificate", False)
            require_certificate = (
                bool(_strict)
                if not isinstance(_strict, str)
                else _strict.lower() == "true"
            )
            _window_strict = p.get(
                "require_frequency_window_certificate", False
            )
            require_frequency_window_certificate = (
                bool(_window_strict)
                if not isinstance(_window_strict, str)
                else _window_strict.lower() == "true"
            )
            freqs = np.logspace(
                np.log10(float(p.get("f_min", 10.0))),
                np.log10(float(p.get("f_max", 1e5))),
                int(p.get("n_freq", 15)),
            )
            dynamic_kwargs = (
                {}
                if dynamic_defect_protocol is None
                else {
                    "dynamic_defect_protocol": dynamic_defect_protocol,
                    "defect_energy_quadrature_order": int(
                        p.get("defect_energy_quadrature_order", 32)
                    ),
                    "dynamic_defect_state_step": float(
                        p.get("dynamic_defect_state_step", 1.0e-5)
                    ),
                    "dynamic_defect_voltage_step": float(
                        p.get("dynamic_defect_voltage_step", 1.0e-5)
                    ),
                }
            )
            result = impedance.run_impedance(
                stack, frequencies=freqs,
                V_dc=float(p.get("V_dc", 0.9)),
                delta_V=float(p.get("delta_V", 0.01)),
                N_grid=int(p.get("N_grid", 40)),
                n_cycles=int(p.get("n_cycles", 5)),
                n_extract=int(p.get("n_extract", 2)),
                points_per_cycle=int(p.get("points_per_cycle", 40)),
                illuminated=illuminated,
                method=str(p.get("method", "transient_ion_aware")),
                dc_settle_time=float(p.get("dc_settle_time", 1e-3)),
                require_operating_point_certificate=require_certificate,
                require_frequency_window_certificate=(
                    require_frequency_window_certificate
                ),
                experiment_protocol=experiment_protocol,
                protocol_mode=protocol_mode,
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
                **dynamic_kwargs,
            )
            out = to_serializable(result)
            if "Z" in out:
                Z = np.array(result.Z)
                out["Z_real"] = Z.real.tolist()
                out["Z_imag"] = Z.imag.tolist()
                del out["Z"]
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "dynamic_defect_transient":
        if (
            dynamic_defect_transient_grid is None
            or dynamic_defect_transient_protocol is None
        ):  # pragma: no cover - guarded by preflight above
            raise HTTPException(
                status_code=500,
                detail="dynamic-defect transient preflight state is missing",
            )

        def _run(reporter: ProgressReporter) -> dict:
            result = dynamic_defect_transient_exp.run_dynamic_defect_transient(
                dynamic_defect_transient_grid,
                stack,
                dynamic_defect_transient_protocol,
                progress=lambda stage, cur, tot, msg: reporter.report(
                    stage,
                    cur,
                    tot,
                    msg,
                ),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "degradation":
        def _run(reporter: ProgressReporter) -> dict:
            result = degradation.run_degradation(
                stack,
                t_end=float(p.get("t_end", 100.0)),
                n_snapshots=int(p.get("n_snapshots", 10)),
                V_bias=float(p.get("V_bias", 0.9)),
                N_grid=int(p.get("N_grid", 40)),
                metric_V_max=float(p["metric_V_max"]) if p.get("metric_V_max") is not None else None,
                metric_settle_time=float(p.get("metric_settle_time", 1e-3)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            if "t" in out:
                out["times"] = out.pop("t")
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "current_decomp":
        def _run(reporter: ProgressReporter) -> dict:
            _illum = p.get("illuminated", True)
            illuminated = bool(_illum) if not isinstance(_illum, str) else _illum.lower() != "false"
            result = jv_sweep.run_jv_sweep(
                stack,
                N_grid=int(p.get("N_grid", 60)),
                n_points=int(p.get("n_points", 30)),
                v_rate=float(p.get("v_rate", 1.0)),
                V_max=float(p["V_max"]) if p.get("V_max") is not None else None,
                illuminated=illuminated,
                decompose_currents=True,
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = {}
            out["V_fwd"] = result.V_fwd.tolist()
            out["V_rev"] = result.V_rev.tolist()
            if result.decomp_fwd:
                out["Jn_fwd"] = result.decomp_fwd.J_n.tolist()
                out["Jp_fwd"] = result.decomp_fwd.J_p.tolist()
                out["Jion_fwd"] = result.decomp_fwd.J_ion.tolist()
                out["Jdisp_fwd"] = result.decomp_fwd.J_disp.tolist()
                out["Jtotal_fwd"] = result.decomp_fwd.J_total.tolist()
            if result.decomp_rev:
                out["Jn_rev"] = result.decomp_rev.J_n.tolist()
                out["Jp_rev"] = result.decomp_rev.J_p.tolist()
                out["Jion_rev"] = result.decomp_rev.J_ion.tolist()
                out["Jdisp_rev"] = result.decomp_rev.J_disp.tolist()
                out["Jtotal_rev"] = result.decomp_rev.J_total.tolist()
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "spatial":
        def _run(reporter: ProgressReporter) -> dict:
            _illum = p.get("illuminated", True)
            illuminated = bool(_illum) if not isinstance(_illum, str) else _illum.lower() != "false"
            result = jv_sweep.run_jv_sweep(
                stack,
                N_grid=int(p.get("N_grid", 60)),
                n_points=int(p.get("n_points", 15)),
                v_rate=float(p.get("v_rate", 1.0)),
                V_max=float(p["V_max"]) if p.get("V_max") is not None else None,
                illuminated=illuminated,
                save_snapshots=True,
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            # Convert snapshots to serialisable dicts with x in nm for readability
            def snap_to_dict(s):
                return {
                    "x": (s.x * 1e9).tolist(),        # nm
                    "phi": s.phi.tolist(),              # V
                    "E": s.E.tolist(),                  # V/m
                    "n": s.n.tolist(),                  # m^-3
                    "p": s.p.tolist(),                  # m^-3
                    "P": s.P.tolist(),                  # m^-3
                    "rho": s.rho.tolist(),              # C/m^3 (charge density * q)
                    "V_app": s.V_app,
                }
            out = {
                "V_fwd": result.V_fwd.tolist(),
                "V_rev": result.V_rev.tolist(),
                "snapshots_fwd": [snap_to_dict(s) for s in (result.snapshots_fwd or [])],
                "snapshots_rev": [snap_to_dict(s) for s in (result.snapshots_rev or [])],
            }
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "tpv":
        from perovskite_sim.experiments.tpv import run_tpv

        def _run(reporter: ProgressReporter) -> dict:
            result = run_tpv(
                stack,
                N_grid=int(p.get("N_grid", 80)),
                delta_G_frac=float(p.get("delta_G_frac", 0.05)),
                t_pulse=float(p.get("t_pulse", 1e-6)),
                t_decay=float(p.get("t_decay", 50e-6)),
                n_points=int(p.get("n_points", 200)),
                experiment_protocol=experiment_protocol,
                protocol_mode=protocol_mode,
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "dark_jv":
        def _run(reporter: ProgressReporter) -> dict:
            result = dark_jv_exp.run_dark_jv(
                stack,
                V_max=float(p.get("V_max", 1.2)),
                n_points=int(p.get("n_points", 60)),
                N_grid=int(p.get("N_grid", 60)),
                v_rate=float(p.get("v_rate", 1.0)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "suns_voc":
        def _run(reporter: ProgressReporter) -> dict:
            suns_raw = p.get("suns_levels")
            if suns_raw is None:
                suns_levels = suns_voc_exp.DEFAULT_SUNS
            else:
                suns_levels = tuple(float(x) for x in suns_raw)
            result = suns_voc_exp.run_suns_voc(
                stack,
                suns_levels=suns_levels,
                N_grid=int(p.get("N_grid", 60)),
                t_settle=float(p.get("t_settle", 1e-3)),
                experiment_protocol=experiment_protocol,
                protocol_mode=protocol_mode,
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "voc_t":
        from perovskite_sim.experiments.voc_t import run_voc_t

        def _run(reporter: ProgressReporter) -> dict:
            result = run_voc_t(
                stack,
                T_min=float(p.get("T_min", 250.0)),
                T_max=float(p.get("T_max", 350.0)),
                n_points=int(p.get("n_points", 6)),
                N_grid=int(p.get("N_grid", 60)),
                jv_n_points=int(p.get("jv_n_points", 30)),
                v_rate=float(p.get("v_rate", 1.0)),
                V_max=float(p["V_max"]) if p.get("V_max") is not None else None,
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "eqe":
        def _run(reporter: ProgressReporter) -> dict:
            lam_min = float(p.get("lambda_min_nm", 300.0))
            lam_max = float(p.get("lambda_max_nm", 1000.0))
            # Dense grid + full settle + 1-sun flux match scripts/plot_eqe.py so
            # the UI reproduces the publication figure: a coarse grid joined by
            # straight segments looks jagged, a short settle leaves the electronic
            # transient undamped, and a low probe flux lets the small photo-signal
            # sit in residual noise (EQE > 1). The frontend smooths the remaining
            # per-point numerical noise for display.
            n_lam = int(p.get("n_lambda", 80))
            if n_lam < 2 or lam_max <= lam_min:
                raise ValueError(
                    "EQE sweep needs n_lambda >= 2 and lambda_max > lambda_min"
                )
            wavelengths_nm = np.linspace(lam_min, lam_max, n_lam)
            result = eqe_exp.compute_eqe(
                stack,
                wavelengths_nm=wavelengths_nm,
                Phi_incident=float(p.get("Phi_incident", 1e22)),
                N_grid=int(p.get("N_grid", 60)),
                t_settle=float(p.get("t_settle", 1e-1)),
                experiment_protocol=experiment_protocol,
                protocol_mode=protocol_mode,
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "el":
        from perovskite_sim.experiments.el_spectrum import run_el_spectrum

        def _run(reporter: ProgressReporter) -> dict:
            lam_min = float(p.get("lambda_min_nm", 400.0))
            lam_max = float(p.get("lambda_max_nm", 1000.0))
            n_lam = int(p.get("n_lambda", 25))
            if n_lam < 2 or lam_max <= lam_min:
                raise ValueError(
                    "EL sweep needs n_lambda >= 2 and lambda_max > lambda_min"
                )
            wavelengths_nm = np.linspace(lam_min, lam_max, n_lam)
            result = run_el_spectrum(
                stack,
                V_inj=float(p.get("V_inj", 1.0)),
                wavelengths_nm=wavelengths_nm,
                N_grid=int(p.get("N_grid", 60)),
                n_points_dark=int(p.get("n_points_dark", 30)),
                v_rate=float(p.get("v_rate", 1.0)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "mott_schottky":
        def _run(reporter: ProgressReporter) -> dict:
            V_lo = float(p.get("V_lo", -0.3))
            V_hi = float(p.get("V_hi", 0.4))
            n_pts = int(p.get("n_points", 8))
            if n_pts < 3 or V_hi <= V_lo:
                raise ValueError(
                    "Mott-Schottky needs n_points >= 3 and V_hi > V_lo"
                )
            V_range = np.linspace(V_lo, V_hi, n_pts)
            result = ms_exp.run_mott_schottky(
                stack,
                V_range=V_range,
                frequency=float(p.get("frequency", 1e6)),
                delta_V=float(p.get("delta_V", 0.01)),
                N_grid=int(p.get("N_grid", 40)),
                n_cycles=int(p.get("n_cycles", 5)),
                n_extract=int(p.get("n_extract", 2)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
    elif kind == "tandem":
        from perovskite_sim.data import load_am15g
        from perovskite_sim.experiments.tandem_jv import run_tandem_jv
        from perovskite_sim.models.tandem_config import load_tandem_from_yaml

        if not req.config_path:
            raise HTTPException(status_code=400, detail="tandem kind requires config_path")
        try:
            cfg = load_tandem_from_yaml(resolve_config_path(req.config_path))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        def _run(reporter: ProgressReporter) -> dict:
            wavelengths_nm = np.linspace(300.0, 1000.0, 200)
            wavelengths_m = wavelengths_nm * 1e-9
            _, spectral_flux = load_am15g(wavelengths_nm)
            try:
                result = run_tandem_jv(
                    cfg,
                    wavelengths_m,
                    spectral_flux,
                    wavelengths_nm,
                    N_grid=int(p.get("N_grid", 40)),
                    n_points=int(p.get("n_points", 15)),
                    progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
                )
            except ValueError as exc:
                msg = str(exc)
                if "Sub-cell J ranges do not overlap" in msg:
                    raise RuntimeError(
                        f"{msg} — this tandem preset ships with stub n,k data "
                        "(rigid bandgap shifts of MAPbI3). Replace "
                        "perovskite_sim/data/nk/FA_Cs_1p77.csv and "
                        "SnPb_1p22.csv with real Lin 2019 SI data before "
                        "expecting physical results."
                    )
                raise

            return {
                "V": result.V.tolist(),
                "J": result.J.tolist(),
                "V_top": result.V_top.tolist(),
                "V_bot": result.V_bot.tolist(),
                # ``main.py`` imports ``asdict`` at module top; bare
                # ``dataclasses`` is in scope only inside the legacy
                # blocking ``run_tandem`` (which has its own local
                # ``import dataclasses``), not this streaming ``_run``.
                # A module-qualified call would crash the worker thread
                # with ``NameError`` after a successful tandem sweep —
                # same shape as the L1008 jv_2d regression fixed in
                # f29b190.
                "metrics": asdict(result.metrics),
                "benchmark": cfg.benchmark,
            }
    elif kind == "jv_2d":
        from perovskite_sim.solver.tolerances import ComponentwiseAtol
        from perovskite_sim.twod.experiments.jv_protocol_2d import (
            JV2DProtocol,
            resolve_jv_2d_protocol,
        )
        from perovskite_sim.twod.experiments.jv_sweep_2d import (
            build_jv_2d_execution_protocol,
            run_jv_sweep_2d,
        )
        from perovskite_sim.twod.microstructure import (
            Microstructure, load_microstructure_from_yaml_block,
        )

        # Resolve and validate geometry before submitting an asynchronous job.
        # A bad microstructure/BC pair is a request-contract error, not a
        # numerical worker failure.
        try:
            ms_block = p.get("microstructure")
            if ms_block:
                ms = load_microstructure_from_yaml_block(ms_block)
            else:
                ms = getattr(stack, "microstructure", None) or Microstructure()
            requested_lateral_bc = p.get("lateral_bc")
            if requested_lateral_bc is None:
                lateral_bc = "neumann" if ms.grain_boundaries else "periodic"
            else:
                lateral_bc = str(requested_lateral_bc)
            if lateral_bc not in {"periodic", "neumann"}:
                raise ValueError(
                    "jv_2d lateral_bc must be 'periodic' or 'neumann'"
                )
            if ms.grain_boundaries and lateral_bc != "neumann":
                raise ValueError(
                    "finite-width jv_2d grain boundaries require "
                    "lateral_bc='neumann'; periodic-x is not area-certified"
                )
            ion_dynamics = str(p.get("ion_dynamics", "frozen"))
            if ion_dynamics not in {"frozen", "single_mobile"}:
                raise ValueError(
                    "jv_2d ion_dynamics must be 'frozen' or 'single_mobile'"
                )
            interface_srh = str(p.get("interface_srh", "off"))
            if interface_srh not in {"off", "two_sided_cross_node"}:
                raise ValueError(
                    "jv_2d interface_srh must be 'off' or "
                    "'two_sided_cross_node'"
                )
            extended_topology = (
                ion_dynamics != "frozen" or interface_srh != "off"
            )

            _illum = p.get("illuminated", True)
            illuminated = (
                bool(_illum)
                if not isinstance(_illum, str)
                else _illum.lower() != "false"
            )
            _save = p.get("save_snapshots", True)
            save_snapshots = (
                bool(_save)
                if not isinstance(_save, str)
                else _save.lower() != "false"
            )
            lateral_length = float(p.get("lateral_length", 500e-9))
            nx_intervals = int(p.get("Nx", 10))
            voltage_maximum = float(p.get("V_max", 1.2))
            voltage_step = float(p.get("V_step", 0.05))
            ny_per_layer = int(p.get("Ny_per_layer", 20))
            settle_time = float(p.get("settle_t", 1e-7))
            solver_rtol = float(p.get("rtol", 1.0e-6))
            max_nfev_per_solve = int(p.get("max_nfev_per_solve", 200_000))
            max_bisect = int(p.get("max_bisect", 6))
            ion_inventory_rtol = float(p.get("ion_inventory_rtol", 1.0e-9))
            initial_state_settle_s = float(
                p.get("initial_state_settle_s", 1.0e-3)
            )
            raw_atol = p.get("componentwise_atol")
            if raw_atol is None:
                solver_atol = (
                    float(p["atol"])
                    if "atol" in p
                    else (ComponentwiseAtol() if extended_topology else 1.0e-8)
                )
            else:
                if "atol" in p:
                    raise ValueError(
                        "jv_2d accepts either atol or componentwise_atol, not both"
                    )
                if not isinstance(raw_atol, dict):
                    raise TypeError("componentwise_atol must be a JSON object")
                expected_atol_keys = {
                    "carrier_fraction",
                    "ion_fraction",
                    "interface_fraction",
                    "minimum_atol",
                    "refinement_factor",
                }
                if set(raw_atol) != expected_atol_keys:
                    raise ValueError(
                        "componentwise_atol keys do not match schema; "
                        f"missing={sorted(expected_atol_keys - set(raw_atol))}, "
                        f"extra={sorted(set(raw_atol) - expected_atol_keys)}"
                    )
                solver_atol = ComponentwiseAtol(**raw_atol)

            jv_protocol_mode = str(p.get("protocol_mode", "compatibility"))
            if jv_protocol_mode not in {"compatibility", "research_strict"}:
                raise ValueError(
                    "jv_2d protocol_mode must be 'compatibility' or "
                    "'research_strict'"
                )
            supplied_jv_protocol = None
            raw_jv_protocol = p.get("jv_2d_protocol")
            if raw_jv_protocol is not None:
                if not isinstance(raw_jv_protocol, dict):
                    raise TypeError("jv_2d_protocol must be a JSON object")
                supplied_jv_protocol = JV2DProtocol.from_dict(raw_jv_protocol)

            # Legacy requests do not need an outer duplicate preflight. Any
            # protocol-bearing or extended request is validated before job
            # submission so a mismatch is an HTTP 422 rather than a worker
            # failure after a job id has been returned.
            resolved_jv_protocol = supplied_jv_protocol
            if (
                extended_topology
                or supplied_jv_protocol is not None
                or jv_protocol_mode != "compatibility"
            ):
                expected_protocol = build_jv_2d_execution_protocol(
                    stack,
                    ms,
                    lateral_length=lateral_length,
                    Nx=nx_intervals,
                    V_max=voltage_maximum,
                    V_step=voltage_step,
                    illuminated=illuminated,
                    lateral_bc=lateral_bc,
                    Ny_per_layer=ny_per_layer,
                    settle_t=settle_time,
                    save_snapshots=save_snapshots,
                    ion_dynamics=ion_dynamics,
                    interface_srh=interface_srh,
                    rtol=solver_rtol,
                    atol=solver_atol,
                    max_nfev_per_solve=max_nfev_per_solve,
                    max_bisect=max_bisect,
                    ion_inventory_rtol=ion_inventory_rtol,
                    initial_state_settle_s=initial_state_settle_s,
                    implicit_legacy_protocol=True,
                )
                if extended_topology and jv_protocol_mode != "research_strict":
                    raise ImplicitProtocolError(
                        "mobile-ion or interface-SRH jv_2d requires an explicit "
                        "research_strict protocol"
                    )
                resolved_jv_protocol = resolve_jv_2d_protocol(
                    supplied_jv_protocol,
                    expected_protocol,
                    mode=jv_protocol_mode,
                )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        def _run(reporter: ProgressReporter) -> dict:
            result = run_jv_sweep_2d(
                stack=stack,
                microstructure=ms,
                lateral_length=lateral_length,
                Nx=nx_intervals,
                V_max=voltage_maximum,
                V_step=voltage_step,
                illuminated=illuminated,
                lateral_bc=lateral_bc,
                Ny_per_layer=ny_per_layer,
                settle_t=settle_time,
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
                save_snapshots=save_snapshots,
                ion_dynamics=ion_dynamics,
                interface_srh=interface_srh,
                rtol=solver_rtol,
                atol=solver_atol,
                max_nfev_per_solve=max_nfev_per_solve,
                max_bisect=max_bisect,
                ion_inventory_rtol=ion_inventory_rtol,
                initial_state_settle_s=initial_state_settle_s,
                jv_2d_protocol=resolved_jv_protocol,
                protocol_mode=jv_protocol_mode,
            )

            def snap2d_to_dict(s):
                return {
                    "V": float(s.V),
                    "x": (s.x * 1e9).tolist(),
                    "y": (s.y * 1e9).tolist(),
                    "phi": s.phi.tolist(),
                    "n": s.n.tolist(),
                    "p": s.p.tolist(),
                    "Jx_n": s.Jx_n.tolist(),
                    "Jy_n": s.Jy_n.tolist(),
                    "Jx_p": s.Jx_p.tolist(),
                    "Jy_p": s.Jy_p.tolist(),
                    "P_ion": None if s.P_ion is None else s.P_ion.tolist(),
                }

            out = {
                "V": result.V.tolist(),
                "J": result.J.tolist(),
                "grid_x": (result.grid_x * 1e9).tolist(),
                "grid_y": (result.grid_y * 1e9).tolist(),
                "lateral_bc": result.lateral_bc,
                "snapshots": [snap2d_to_dict(s) for s in result.snapshots],
                # Centralised V_oc / J_sc / FF / PCE extraction (Layer 1+2
                # of the Phase 6 acceptance follow-up). Carries the
                # ``voc_bracketed`` flag so the frontend can warn the
                # user when V_max stopped short of V_oc; raw V/J above
                # are unchanged. Use the module-top ``asdict`` import; this
                # handler does NOT have a local ``import dataclasses`` (the
                # tandem block at L909 does), so a module-qualified call
                # would crash the worker thread with a NameError.
                "metrics": asdict(result.metrics),
                "protocol": (
                    None if result.protocol is None else result.protocol.to_dict()
                ),
                "protocol_hash": (
                    None if result.protocol is None else result.protocol.protocol_hash
                ),
                "current_diagnostics": [
                    {
                        "terminal_electron_A_m2": item.terminal_electron_A_m2,
                        "terminal_hole_A_m2": item.terminal_hole_A_m2,
                        "terminal_positive_ion_A_m2": (
                            item.terminal_positive_ion_A_m2
                        ),
                        "terminal_displacement_A_m2": (
                            item.terminal_displacement_A_m2
                        ),
                        "terminal_total_A_m2": item.terminal_total_A_m2,
                        "max_face_spread_A_m2": item.max_face_spread_A_m2,
                        "max_relative_face_spread": item.max_relative_face_spread,
                    }
                    for item in result.current_components
                ],
                "ion_diagnostics": to_serializable(result.ion_diagnostics),
                "interface_srh_diagnostics": [
                    {
                        "interface_rows": list(item.interface_rows),
                        "max_total_surface_rate_m2_s": float(
                            np.max(item.total_surface_rate_m2_s)
                        ),
                        "pair_a_clamped_count": int(np.sum(item.pair_a_clamped)),
                        "pair_b_clamped_count": int(np.sum(item.pair_b_clamped)),
                    }
                    for item in result.interface_srh_diagnostics
                ],
            }
            active_physics = _describe_active_physics(stack)
            if ion_dynamics == "single_mobile":
                active_physics += " · 2D single positive mobile ion"
            if interface_srh == "two_sided_cross_node":
                active_physics += " · 2D two-sided interface SRH"
            out["active_physics"] = active_physics
            return out
    elif kind == "voc_grain_sweep":
        from perovskite_sim.twod.experiments.voc_grain_sweep import run_voc_grain_sweep

        def _run(reporter: ProgressReporter) -> dict:
            raw_sizes = p.get("grain_sizes_nm") or p.get("grain_sizes")
            if not raw_sizes:
                raise HTTPException(
                    status_code=400,
                    detail="voc_grain_sweep requires grain_sizes_nm (list of nm)",
                )
            grain_sizes_m = [float(s) * 1e-9 for s in raw_sizes]
            tau_n_gb = float(p.get("tau_gb_n", 1e-9))
            tau_p_gb = float(p.get("tau_gb_p", 1e-9))
            _illum = p.get("illuminated", True)
            illuminated = bool(_illum) if not isinstance(_illum, str) else _illum.lower() != "false"

            result = run_voc_grain_sweep(
                stack=stack,
                grain_sizes=grain_sizes_m,
                tau_gb=(tau_n_gb, tau_p_gb),
                gb_width=float(p.get("gb_width", 10e-9)),
                Nx=int(p.get("Nx", 10)),
                Ny_per_layer=int(p.get("Ny_per_layer", 10)),
                V_max=float(p.get("V_max", 1.2)),
                V_step=float(p.get("V_step", 0.05)),
                illuminated=illuminated,
                settle_t=float(p.get("settle_t", 1e-3)),
                progress=lambda stage, cur, tot, msg: reporter.report(stage, cur, tot, msg),
            )
            return {
                "grain_sizes_nm": (result.grain_sizes_m * 1e9).tolist(),
                "V_oc_V": result.V_oc_V.tolist(),
                "J_sc_Am2": result.J_sc_Am2.tolist(),
                "FF": result.FF.tolist(),
                "active_physics": _describe_active_physics(stack),
            }
    else:
        raise HTTPException(status_code=400, detail=f"unknown kind: {kind}")

    job_id = _JOB_REGISTRY.submit(_run)
    return {"status": "ok", "job_id": job_id}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    """Stream progress events, the final result, and a done marker."""
    try:
        _JOB_REGISTRY.status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")

    async def _gen():
        loop = asyncio.get_event_loop()
        while True:
            ev = await loop.run_in_executor(
                None, lambda: _JOB_REGISTRY.next_event(job_id, timeout=0.5)
            )
            if ev is _DRAIN_TIMEOUT:
                yield ": keepalive\n\n"
                status, _, _ = _JOB_REGISTRY.status(job_id)
                if status == JobStatus.RUNNING:
                    continue
                # Fallthrough: worker finished between drain and status
                # check — loop once more to pick up the done sentinel.
                continue
            if ev is None:
                status, result, error = _JOB_REGISTRY.status(job_id)
                if status == JobStatus.DONE:
                    yield f"event: result\ndata: {json.dumps(result)}\n\n"
                elif status == JobStatus.ERROR:
                    yield f"event: error\ndata: {json.dumps({'message': error})}\n\n"
                yield "event: done\ndata: {}\n\n"
                return
            payload = {
                "stage": ev.stage,
                "current": ev.current,
                "total": ev.total,
                "eta_s": ev.eta_s,
                "message": ev.message,
            }
            yield f"event: progress\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.post("/api/degradation")
def run_degradation_api(req: DegRequest):
    try:
        stack = build_stack(req.config_path, req.device)
        result = degradation.run_degradation(
            stack, t_end=req.t_end, n_snapshots=req.n_snapshots,
            N_grid=req.N_grid, V_bias=req.V_bias,
            metric_V_max=req.metric_V_max,
            metric_settle_time=req.metric_settle_time,
        )
        out = to_serializable(result)
        if "t" in out:
            out["times"] = out.pop("t")
        return {"status": "ok", "result": out}
    except HTTPException:
        raise
    except Exception as e:
        print("[Degradation API Exception]", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
