"""Declarative YAML/JSON schema, preset loader, and execution runners for 1D iodine aging.

Provides Pydantic-validated declarative decks (``solarlab.iodine-deck.v1``) that
researchers, CLI commands, and HTTP routes can load from packaged presets or
user YAML/JSON files without modifying SolarLab's frozen 20-DTO V2 frontend
configuration schema. Every execution payload explicitly carries
``scientific_qualification = False`` and ``scientific_acceptance = "not_evaluated"``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from solarlab.config.resolve_device import resolve_device
from solarlab.contracts import ContractError
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.device.compile_iodine_1d import (
    IodineBoundaryDeclaration,
    IodineChemistryDeclaration,
    IodineLayerSpec,
    IodineRateParameterization,
    compile_iodine_device_1d,
)
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.iodine_1d import (
    CompiledIodineDevice1D,
    ContactReactionMode,
    IodineDomainMode,
    IodineState1D,
)
from solarlab.experiments.iodine_aging import (
    IodineAgingProtocol,
    IodineProbeMode,
    IodineProbeSpec,
    StressRecoveryStage,
    measure_frozen_chemistry_jv,
    record_iodine_aging_in_run_store,
    run_iodine_aging_protocol,
    save_iodine_aging_bundle,
)
from solarlab.io.run_store import RunStore
from solarlab.materials import (
    PACKAGED_PRESET_NAMES,
    load_packaged_default_catalog,
    load_packaged_preset_device_mapping,
    load_packaged_preset_sources,
    load_packaged_resource_library,
)
from solarlab.materials.iodine_presets import (
    PACKAGED_IODINE_PRESET_NAMES,
    load_packaged_iodine_preset_mapping,
    load_packaged_iodine_preset_source,
)
from solarlab.materials.packaged import _StrictYamlMappingLoader
from solarlab.materials.source import SourceDocument
from solarlab.physics.chemistry import IodineThermodynamicMode, ReactionEOrder
from solarlab.solvers.linear.iodine_impedance import (
    ChemResponseMode,
    IodineImpedanceResult,
    solve_iodine_small_signal_1d,
)
from solarlab.solvers.nonlinear.iodine_1d import (
    solve_frozen_chemistry_electronic_1d,
    solve_iodine_bdf_transient,
    solve_iodine_steady_state,
)
from solarlab.solvers.time.bdf import BDFOptions

__all__ = [
    "IODINE_DECK_SCHEMA_VERSION",
    "IodineAgingSpecInput",
    "IodineBDFOptionsInput",
    "IodineBoundaryInput",
    "IodineChemistryInput",
    "IodineDeckInput",
    "IodineElectronicStackInput",
    "IodineImpedanceSpecInput",
    "IodineLayerInput",
    "IodineOverridesInput",
    "IodineProbeInput",
    "IodineStageInput",
    "IodineSteadySpecInput",
    "apply_iodine_deck_overrides",
    "compile_iodine_deck",
    "execute_iodine_aging_deck",
    "execute_iodine_impedance_deck",
    "execute_iodine_steady_deck",
    "export_iodine_deck_schema",
    "load_iodine_deck",
    "resolve_iodine_operating_state",
]

IODINE_DECK_SCHEMA_VERSION = "solarlab.iodine-deck.v1"
_EPS0_F_M = 8.8541878128e-12


class IodineLayerInput(BaseModel):
    """Declarative per-layer iodine transport, reaction, partition, and outgassing spec."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    layer_id: str = Field(min_length=1, max_length=128)
    units: Literal["cgs", "si"] = "cgs"
    D_minus: float = Field(ge=0.0)
    D_atom: float = Field(ge=0.0)
    D_mol: float = Field(ge=0.0)
    initial_iodide: float = Field(default=0.0, ge=0.0)
    initial_atom: float = Field(default=0.0, ge=0.0)
    initial_mol: float = Field(default=0.0, ge=0.0)
    compensating_donor: float | None = Field(default=None, ge=0.0)
    E_a_minus_eV: float = Field(default=0.0, ge=0.0)
    E_a_atom_eV: float = Field(default=0.0, ge=0.0)
    E_a_mol_eV: float = Field(default=0.0, ge=0.0)
    rate_mode: IodineRateParameterization = "explicit_rates"
    k_a: float = Field(default=0.0, ge=0.0)
    k_b: float = Field(default=0.0, ge=0.0)
    k_c: float = Field(default=0.0, ge=0.0)
    k_d: float = Field(default=0.0, ge=0.0)
    k_e: float = Field(default=0.0, ge=0.0)
    k_e_3body: float = Field(default=0.0, ge=0.0)
    capture_radius_nm: float = Field(default=0.2, ge=0.0)
    K_diss: float = Field(default=1.0e8, gt=0.0)
    n_e_star: float = Field(default=1.0e8, ge=0.0)
    partition_minus: float = Field(default=1.0, gt=0.0)
    partition_atom: float = Field(default=1.0, gt=0.0)
    partition_mol: float = Field(default=1.0, gt=0.0)
    iodide_site_limit: float | None = Field(default=None, gt=0.0)
    volumetric_mol_escape_s_1: float = Field(default=0.0, ge=0.0)

    def to_layer_spec(self) -> IodineLayerSpec:
        if self.units == "cgs":
            spec = IodineLayerSpec.from_cgs(
                self.layer_id,
                D_minus_cm2_s=self.D_minus,
                D_atom_cm2_s=self.D_atom,
                D_mol_cm2_s=self.D_mol,
                initial_iodide_cm3=self.initial_iodide,
                initial_atom_cm3=self.initial_atom,
                initial_mol_cm3=self.initial_mol,
                compensating_donor_cm3=self.compensating_donor,
                rate_mode=self.rate_mode,
                k_a_cm3_s=self.k_a,
                k_b_s_1=self.k_b,
                k_c_cm3_s=self.k_c,
                k_d_cm3_s=self.k_d,
                k_e_cm3_s=self.k_e,
                k_e_3body_cm6_s=self.k_e_3body,
                capture_radius_nm=self.capture_radius_nm,
                K_diss_cm3=self.K_diss,
                n_e_star_cm3=self.n_e_star,
                partition_minus=self.partition_minus,
                partition_atom=self.partition_atom,
                partition_mol=self.partition_mol,
                iodide_site_limit_cm3=self.iodide_site_limit,
                volumetric_mol_escape_s_1=self.volumetric_mol_escape_s_1,
            )
            if (
                self.E_a_minus_eV != 0.0
                or self.E_a_atom_eV != 0.0
                or self.E_a_mol_eV != 0.0
            ):
                return IodineLayerSpec(
                    **{
                        **asdict(spec),
                        "E_a_minus_eV": self.E_a_minus_eV,
                        "E_a_atom_eV": self.E_a_atom_eV,
                        "E_a_mol_eV": self.E_a_mol_eV,
                    }
                )
            return spec
        return IodineLayerSpec(
            layer_id=self.layer_id,
            D_minus_m2_s=self.D_minus,
            D_atom_m2_s=self.D_atom,
            D_mol_m2_s=self.D_mol,
            initial_iodide_m3=self.initial_iodide,
            initial_atom_m3=self.initial_atom,
            initial_mol_m3=self.initial_mol,
            compensating_donor_m3=self.compensating_donor,
            E_a_minus_eV=self.E_a_minus_eV,
            E_a_atom_eV=self.E_a_atom_eV,
            E_a_mol_eV=self.E_a_mol_eV,
            rate_mode=self.rate_mode,
            k_a_m3_s=self.k_a,
            k_b_s_1=self.k_b,
            k_c_m3_s=self.k_c,
            k_d_m3_s=self.k_d,
            k_e_m3_s=self.k_e,
            k_e_3body_m6_s=self.k_e_3body,
            capture_radius_m=self.capture_radius_nm * 1.0e-9,
            K_diss_m3=self.K_diss,
            n_e_star_m3=self.n_e_star,
            partition_minus=self.partition_minus,
            partition_atom=self.partition_atom,
            partition_mol=self.partition_mol,
            iodide_site_limit_m3=self.iodide_site_limit,
            volumetric_mol_escape_s_1=self.volumetric_mol_escape_s_1,
        )


class IodineBoundaryInput(BaseModel):
    """Outer contact boundary reaction and volatile I2 escape specification."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    mol_escape_velocity_m_s: float = Field(default=0.0, ge=0.0)
    contact_reaction_mode: ContactReactionMode = "none"
    contact_oxidation_rate_m4_s: float = Field(default=0.0, ge=0.0)
    contact_corrosion_velocity_m_s: float = Field(default=0.0, ge=0.0)

    def to_declaration(self) -> IodineBoundaryDeclaration:
        return IodineBoundaryDeclaration(
            mol_escape_velocity_m_s=self.mol_escape_velocity_m_s,
            contact_reaction_mode=self.contact_reaction_mode,
            contact_oxidation_rate_m4_s=self.contact_oxidation_rate_m4_s,
            contact_corrosion_velocity_m_s=self.contact_corrosion_velocity_m_s,
        )


class IodineChemistryInput(BaseModel):
    """Complete declarative 5-reaction I-/I0/I2 chemistry configuration."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    domain_mode: IodineDomainMode = "absorber_blocking"
    thermodynamic_mode: IodineThermodynamicMode = "wegscheider_detailed_balance"
    reaction_e_order: ReactionEOrder = "bimolecular_n"
    initialize_equilibrium_split: bool = False
    enforce_spatial_wegscheider: bool = True
    steric_diffusion_only: bool = True
    left_boundary: IodineBoundaryInput = Field(default_factory=IodineBoundaryInput)
    right_boundary: IodineBoundaryInput = Field(default_factory=IodineBoundaryInput)
    layers: list[IodineLayerInput] = Field(min_length=1, max_length=16)

    def to_declaration(self) -> IodineChemistryDeclaration:
        return IodineChemistryDeclaration(
            layers=tuple(layer.to_layer_spec() for layer in self.layers),
            domain_mode=self.domain_mode,
            thermodynamic_mode=self.thermodynamic_mode,
            reaction_e_order=self.reaction_e_order,
            left_boundary=self.left_boundary.to_declaration(),
            right_boundary=self.right_boundary.to_declaration(),
            initialize_equilibrium_split=self.initialize_equilibrium_split,
            enforce_spatial_wegscheider=self.enforce_spatial_wegscheider,
            steric_diffusion_only=self.steric_diffusion_only,
        )


class IodineElectronicStackInput(BaseModel):
    """Electronic stack reference: packaged preset name or inline DeviceInput mapping."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    preset: str | None = None
    device: dict[str, Any] | None = None
    allow_legacy_benchmark_deck: bool = False
    intervals_per_layer: int | list[int] = 18

    @model_validator(mode="after")
    def _validate_source_and_intervals(self) -> IodineElectronicStackInput:
        if (self.preset is None) == (self.device is None):
            raise ValueError(
                "electronic_stack must specify exactly one of 'preset' or 'device'"
            )
        if self.preset is not None and self.preset not in PACKAGED_PRESET_NAMES:
            raise ValueError(
                f"unknown electronic device preset {self.preset!r}; "
                f"expected one of {list(PACKAGED_PRESET_NAMES)}"
            )
        if isinstance(self.intervals_per_layer, int):
            if not 2 <= self.intervals_per_layer <= 256:
                raise ValueError("intervals_per_layer must lie in [2, 256]")
        else:
            if not self.intervals_per_layer or any(
                not 2 <= int(n) <= 256 for n in self.intervals_per_layer
            ):
                raise ValueError("each intervals_per_layer entry must lie in [2, 256]")
        return self


class IodineSteadySpecInput(BaseModel):
    """Default operating point and optional frozen-chemistry J-V sweep for steady runs."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    voltage_V: float = Field(default=0.9, ge=-2.0, le=3.0)
    generation_scale: float = Field(default=1.0, ge=0.0, le=100.0)
    open_loss_precondition_s: float = Field(default=20.0, gt=0.0, le=3600.0)
    compute_frozen_jv: bool = True
    jv_v_min_V: float = Field(default=0.0, ge=-2.0, le=3.0)
    jv_v_max_V: float = Field(default=1.25, ge=-2.0, le=3.0)
    jv_n_points: int = Field(default=26, ge=2, le=201)


class IodineImpedanceSpecInput(BaseModel):
    """Default small-signal AC impedance and redox capacitance sweep parameters."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    voltage_V: float = Field(default=0.9, ge=-2.0, le=3.0)
    generation_scale: float = Field(default=1.0, ge=0.0, le=100.0)
    f_min_Hz: float = Field(default=1.0e-2, gt=0.0, le=1.0e9)
    f_max_Hz: float = Field(default=1.0e6, gt=0.0, le=1.0e10)
    n_frequencies: int = Field(default=25, ge=2, le=201)
    chem_mode: Literal["mobile", "frozen", "both"] = "both"

    @model_validator(mode="after")
    def _check_freq_order(self) -> IodineImpedanceSpecInput:
        if self.f_max_Hz <= self.f_min_Hz:
            raise ValueError("f_max_Hz must exceed f_min_Hz")
        return self


class IodineStageInput(BaseModel):
    """One constant-bias, constant-illumination stress or recovery stage."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1, max_length=128)
    duration_s: float = Field(gt=0.0, le=1.0e9)
    voltage_V: float = Field(ge=-2.0, le=3.0)
    generation_scale: float = Field(ge=0.0, le=100.0)
    max_step_s: float = Field(default=200.0, gt=0.0, le=1.0e8)

    def to_stage(self) -> StressRecoveryStage:
        return StressRecoveryStage(
            name=self.name,
            duration_s=self.duration_s,
            voltage_V=self.voltage_V,
            generation_scale=self.generation_scale,
            max_step_s=self.max_step_s,
        )


class IodineProbeInput(BaseModel):
    """Non-destructive J-V characterization probe at aging checkpoints."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    mode: IodineProbeMode = "frozen_chemistry_jv"
    v_min_V: float = Field(default=0.0, ge=-2.0, le=3.0)
    v_max_V: float = Field(default=1.25, ge=-2.0, le=3.0)
    n_points: int = Field(default=26, ge=2, le=201)
    scan_rate_V_s: float = Field(default=1.0, gt=0.0, le=1.0e4)
    generation_scale: float = Field(default=1.0, ge=0.0, le=100.0)
    incident_power_W_m2: float = Field(default=1000.0, gt=0.0, le=1.0e6)

    def to_probe_spec(self) -> IodineProbeSpec:
        voltages = tuple(
            float(v) for v in np.linspace(self.v_min_V, self.v_max_V, self.n_points)
        )
        return IodineProbeSpec(
            mode=self.mode,
            voltages_V=voltages,
            scan_rate_V_s=self.scan_rate_V_s,
            generation_scale=self.generation_scale,
            incident_power_W_m2=self.incident_power_W_m2,
        )


class IodineBDFOptionsInput(BaseModel):
    """Variable-step BDF-1/BDF-2 time-integration tolerances and limits."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    initial_step_s: float = Field(default=1.0e-5, gt=0.0)
    min_step_s: float = Field(default=1.0e-14, gt=0.0)
    max_step_s: float = Field(default=200.0, gt=0.0)
    relative_tolerance: float = Field(default=1.0e-2, gt=0.0, le=0.1)
    absolute_tolerance: float = Field(default=1.0e15, gt=0.0)
    max_order: Literal[1, 2] = 1
    max_steps: int = Field(default=800, ge=4, le=10000)

    def to_bdf_options(self) -> BDFOptions:
        return BDFOptions(
            initial_step_s=self.initial_step_s,
            min_step_s=self.min_step_s,
            max_step_s=self.max_step_s,
            relative_tolerance=self.relative_tolerance,
            absolute_tolerance=self.absolute_tolerance,
            max_order=self.max_order,
            max_steps=self.max_steps,
        )


class IodineAgingSpecInput(BaseModel):
    """Complete multi-stage aging protocol specification."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    stages: list[IodineStageInput] = Field(min_length=1, max_length=32)
    checkpoint_times_s: list[float] = Field(default_factory=list)
    probe: IodineProbeInput = Field(default_factory=IodineProbeInput)
    bdf: IodineBDFOptionsInput = Field(default_factory=IodineBDFOptionsInput)

    def to_protocol(self) -> IodineAgingProtocol:
        stages = tuple(s.to_stage() for s in self.stages)
        max_stage_step = max(s.max_step_s for s in stages)
        bdf_opts = self.bdf.to_bdf_options()
        if max_stage_step > bdf_opts.max_step_s:
            bdf_opts = BDFOptions(
                initial_step_s=bdf_opts.initial_step_s,
                min_step_s=bdf_opts.min_step_s,
                max_step_s=max_stage_step,
                relative_tolerance=bdf_opts.relative_tolerance,
                absolute_tolerance=bdf_opts.absolute_tolerance,
                max_order=bdf_opts.max_order,
                max_steps=bdf_opts.max_steps,
            )
        return IodineAgingProtocol(
            stages=stages,
            checkpoint_times_s=tuple(self.checkpoint_times_s),
            probe=self.probe.to_probe_spec(),
            bdf_options=bdf_opts,
        )


class IodineDeckInput(BaseModel):
    """Top-level ``solarlab.iodine-deck.v1`` declarative configuration."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["solarlab.iodine-deck.v1"] = "solarlab.iodine-deck.v1"
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(default="Iodine Chemistry & Aging Deck", max_length=256)
    description: str = Field(default="", max_length=2048)
    reference: str = Field(default="", max_length=512)
    electronic_stack: IodineElectronicStackInput
    chemistry: IodineChemistryInput
    steady: IodineSteadySpecInput = Field(default_factory=IodineSteadySpecInput)
    impedance: IodineImpedanceSpecInput = Field(
        default_factory=IodineImpedanceSpecInput
    )
    aging: IodineAgingSpecInput | None = None


class IodineOverridesInput(BaseModel):
    """Bounded interactive parameter overrides for CLI and HTTP workstation runs."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    intervals_per_layer: int | list[int] | None = None
    k_a_cm3_s: float | None = Field(default=None, ge=0.0, le=1.0e-5)
    mu_bl_cm2_V_s: float | None = Field(default=None, gt=0.0, le=100.0)
    ebl_partition: float | None = Field(default=None, gt=0.0, le=1.0e3)
    volumetric_escape_s_1: float | None = Field(default=None, ge=0.0, le=1.0e3)
    contact_corrosion_m_s: float | None = Field(default=None, ge=0.0, le=1.0)
    voltage_V: float | None = Field(default=None, ge=-2.0, le=3.0)
    generation_scale: float | None = Field(default=None, ge=0.0, le=100.0)
    compute_frozen_jv: bool | None = None
    f_min_Hz: float | None = Field(default=None, gt=0.0, le=1.0e9)
    f_max_Hz: float | None = Field(default=None, gt=0.0, le=1.0e10)
    n_frequencies: int | None = Field(default=None, ge=2, le=101)
    chem_mode: Literal["mobile", "frozen", "both"] | None = None
    probe_mode: IodineProbeMode | None = None
    stages: list[IodineStageInput] | None = Field(default=None, max_length=12)
    checkpoint_times_s: list[float] | None = None


def export_iodine_deck_schema() -> dict[str, Any]:
    """Export deterministic JSON Schema and SHA-256 digest for ``IodineDeckInput``."""
    schema = IodineDeckInput.model_json_schema()
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "solarlab.iodine-deck-schema.v1",
        "deck_schema_version": IODINE_DECK_SCHEMA_VERSION,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "packaged_presets": list(PACKAGED_IODINE_PRESET_NAMES),
        "json_schema": schema,
        "scientific_qualification": False,
    }


def load_iodine_deck(
    source: str | Path | Mapping[str, Any],
) -> tuple[IodineDeckInput, SourceDocument]:
    """Load and validate an ``IodineDeckInput`` from preset name, YAML/JSON path, or dict."""
    if isinstance(source, Mapping):
        raw_bytes = json.dumps(dict(source), sort_keys=True).encode("utf-8")
        doc = SourceDocument("inline-iodine-deck.json", raw_bytes)
        return IodineDeckInput.model_validate(dict(source)), doc
    if isinstance(source, str) and not Path(source).is_file():
        preset_doc = load_packaged_iodine_preset_source(source)
        mapping = load_packaged_iodine_preset_mapping(source)
        return IodineDeckInput.model_validate(mapping), preset_doc
    path = Path(source)
    raw = path.read_bytes()
    src_doc = SourceDocument(path.name, raw)
    text = raw.decode("utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        parsed = yaml.load(text, Loader=_StrictYamlMappingLoader)
    else:
        parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"iodine deck file {path} must contain a mapping/object")
    return IodineDeckInput.model_validate(parsed), src_doc


def apply_iodine_deck_overrides(
    deck: IodineDeckInput,
    overrides: IodineOverridesInput | None,
) -> IodineDeckInput:
    """Return a validated copy of ``deck`` with optional interactive overrides applied."""
    if overrides is None:
        return deck
    data = deck.model_dump(mode="python")

    if overrides.intervals_per_layer is not None:
        data["electronic_stack"]["intervals_per_layer"] = overrides.intervals_per_layer

    if overrides.mu_bl_cm2_V_s is not None and data["electronic_stack"].get("device"):
        mu_si = float(overrides.mu_bl_cm2_V_s) * 1.0e-4
        for layer in data["electronic_stack"]["device"].get("layers", []):
            if layer.get("role") in {"ETL", "HTL"}:
                params = layer.setdefault("parameters", {})
                eps_r = float(params.get("eps_r", 3.0))
                params["mu_n"] = mu_si
                params["mu_p"] = mu_si
                params["B_rad"] = float(Q * (2.0 * mu_si) / (_EPS0_F_M * eps_r))

    for layer_spec in data["chemistry"]["layers"]:
        lid = layer_spec["layer_id"]
        is_cgs = layer_spec.get("units", "cgs") == "cgs"
        if overrides.k_a_cm3_s is not None and lid == "absorber":
            layer_spec["rate_mode"] = "explicit_rates"
            layer_spec["k_a"] = (
                float(overrides.k_a_cm3_s)
                if is_cgs
                else float(overrides.k_a_cm3_s) * 1.0e-6
            )
        if overrides.ebl_partition is not None and lid in {"ebl", "htl"}:
            part = float(overrides.ebl_partition)
            layer_spec["partition_minus"] = part
            layer_spec["partition_atom"] = part
            layer_spec["partition_mol"] = part
        if overrides.volumetric_escape_s_1 is not None:
            target = (
                "ebl"
                if data["chemistry"]["domain_mode"] == "full_stack_permeable"
                else "absorber"
            )
            if lid == target:
                layer_spec["volumetric_mol_escape_s_1"] = float(
                    overrides.volumetric_escape_s_1
                )

    if overrides.contact_corrosion_m_s is not None:
        v_corr = float(overrides.contact_corrosion_m_s)
        if v_corr > 0.0:
            data["chemistry"]["left_boundary"]["contact_reaction_mode"] = (
                "metal_faradaic_corrosion"
            )
            data["chemistry"]["left_boundary"]["contact_corrosion_velocity_m_s"] = (
                v_corr
            )
        else:
            data["chemistry"]["left_boundary"]["contact_reaction_mode"] = "none"
            data["chemistry"]["left_boundary"]["contact_corrosion_velocity_m_s"] = 0.0

    if overrides.voltage_V is not None:
        data["steady"]["voltage_V"] = float(overrides.voltage_V)
        data["impedance"]["voltage_V"] = float(overrides.voltage_V)
    if overrides.generation_scale is not None:
        data["steady"]["generation_scale"] = float(overrides.generation_scale)
        data["impedance"]["generation_scale"] = float(overrides.generation_scale)
    if overrides.compute_frozen_jv is not None:
        data["steady"]["compute_frozen_jv"] = bool(overrides.compute_frozen_jv)
    if overrides.f_min_Hz is not None:
        data["impedance"]["f_min_Hz"] = float(overrides.f_min_Hz)
    if overrides.f_max_Hz is not None:
        data["impedance"]["f_max_Hz"] = float(overrides.f_max_Hz)
    if overrides.n_frequencies is not None:
        data["impedance"]["n_frequencies"] = int(overrides.n_frequencies)
    if overrides.chem_mode is not None:
        data["impedance"]["chem_mode"] = overrides.chem_mode

    if data.get("aging") is not None:
        if overrides.probe_mode is not None:
            data["aging"]["probe"]["mode"] = overrides.probe_mode
        if overrides.stages is not None:
            data["aging"]["stages"] = [s.model_dump(mode="python") for s in overrides.stages]
            total_s = sum(float(s.duration_s) for s in overrides.stages)
            if overrides.checkpoint_times_s is None:
                data["aging"]["checkpoint_times_s"] = [0.0, total_s]
        if overrides.checkpoint_times_s is not None:
            data["aging"]["checkpoint_times_s"] = list(overrides.checkpoint_times_s)

    return IodineDeckInput.model_validate(data)


def compile_iodine_deck(
    deck: IodineDeckInput,
    *,
    source: SourceDocument | None = None,
    intervals_override: int | Sequence[int] | None = None,
) -> tuple[PreparedDevice, CompiledIodineDevice1D]:
    """Resolve the electronic stack and compile ``CompiledIodineDevice1D``."""
    defaults: DefaultCatalog = load_packaged_default_catalog()
    resources = load_packaged_resource_library()
    src_tuple = (
        (*load_packaged_preset_sources(),)
        if source is None
        else (source, *load_packaged_preset_sources())
    )
    if deck.electronic_stack.preset is not None:
        dev_map = load_packaged_preset_device_mapping(deck.electronic_stack.preset)
    else:
        assert deck.electronic_stack.device is not None
        dev_map = json.loads(json.dumps(deck.electronic_stack.device))
        settings_map = dev_map.setdefault("settings", {})
        if settings_map.get("built_in_potential_mode") == "metal_work_function":
            settings_map.setdefault("V_bi", 0.0)
        for layer_entry in dev_map.get("layers", []):
            if isinstance(layer_entry, dict) and isinstance(
                layer_entry.get("parameters"), dict
            ):
                p = layer_entry["parameters"]
                p.setdefault("A_star_n", 0.0)
                p.setdefault("A_star_p", 0.0)
                p.setdefault("D_ion", 0.0)
                p.setdefault("P0", 0.0)
                p.setdefault("P_lim", 1.0e28)

    prepared = resolve_device(
        DeviceInput.model_validate(dev_map),
        defaults,
        resources,
        sources=src_tuple,
    )
    chem_decl = deck.chemistry.to_declaration()
    intervals = (
        deck.electronic_stack.intervals_per_layer
        if intervals_override is None
        else intervals_override
    )
    compiled = compile_iodine_device_1d(
        prepared,
        chem_decl,
        intervals_per_layer=intervals,
        allow_legacy_benchmark_deck=deck.electronic_stack.allow_legacy_benchmark_deck,
    )
    return prepared, compiled


def resolve_iodine_operating_state(
    device: CompiledIodineDevice1D,
    *,
    voltage_V: float,
    generation_scale: float,
    open_loss_precondition_s: float = 20.0,
) -> tuple[IodineState1D, str]:
    """Resolve closed-inventory steady state or open-loss quasi-steady operating state."""
    if not device.has_open_iodine_loss:
        ss = solve_iodine_steady_state(
            device,
            voltage_V=voltage_V,
            generation_scale=generation_scale,
        )
        if ss.converged and ss.state is not None:
            return ss.state, "closed_inventory_steady"

    u0, _ = solve_frozen_chemistry_electronic_1d(
        device,
        device.initial_coordinates(voltage_V=0.0),
        voltage_V=voltage_V,
        generation_scale=generation_scale,
        prev_voltage_V=0.0,
    )
    horizon_s = open_loss_precondition_s if device.has_open_iodine_loss else 600.0
    bdf_res, _ = solve_iodine_bdf_transient(
        device,
        u0,
        (0.0, horizon_s),
        voltage_fn=lambda _t: voltage_V,
        generation_fn=lambda _t: generation_scale,
        options=BDFOptions(
            initial_step_s=1.0e-5,
            min_step_s=1.0e-14,
            max_step_s=max(horizon_s * 0.25, 1.0),
            relative_tolerance=1.0e-2,
            absolute_tolerance=1.0e15,
            max_order=1,
            max_steps=400,
        ),
    )
    if len(bdf_res.accepted_coordinates) == 0:
        raise ContractError("failed_to_resolve_iodine_operating_state")
    state = device.state(bdf_res.accepted_coordinates[-1], voltage_V=voltage_V)
    mode = (
        f"open_loss_quasi_steady_{horizon_s:g}s"
        if device.has_open_iodine_loss
        else "bdf_relaxed_steady"
    )
    return state, mode


def _serialize_state_profiles(
    device: CompiledIodineDevice1D,
    state: IodineState1D,
) -> dict[str, Any]:
    e_nodes_nm = [float(x * 1.0e9) for x in device.electronic_component.nodes_m]
    c_nodes_nm = [float(x * 1.0e9) for x in device.chem_geometry.nodes_m]
    e_state = state.electronic_state
    return {
        "electrical_nodes_nm": e_nodes_nm,
        "chem_nodes_nm": c_nodes_nm,
        "potential_V": [float(v) for v in e_state.potential_V],
        "electron_density_m3": [float(v) for v in e_state.electron_density_m3],
        "hole_density_m3": [float(v) for v in e_state.hole_density_m3],
        "iodide_m3": [float(v) for v in state.iodide_density_m3],
        "iodine_atom_m3": [float(v) for v in state.iodine_atom_density_m3],
        "iodine_molecule_m3": [float(v) for v in state.iodine_molecule_density_m3],
        "species_inventories_m2": {
            k: float(v) for k, v in state.species_inventories_m2.items()
        },
        "total_iodine_atoms_m2": float(state.total_iodine_inventory_m2),
    }


def execute_iodine_steady_deck(
    deck: IodineDeckInput,
    *,
    source: SourceDocument | None = None,
) -> dict[str, Any]:
    """Run coupled steady/quasi-steady state and optional frozen-chemistry J-V probe."""
    t0 = time.perf_counter()
    _, compiled = compile_iodine_deck(deck, source=source)
    spec = deck.steady
    state, state_mode = resolve_iodine_operating_state(
        compiled,
        voltage_V=spec.voltage_V,
        generation_scale=spec.generation_scale,
        open_loss_precondition_s=spec.open_loss_precondition_s,
    )
    frozen_jv_payload: dict[str, Any] | None = None
    if spec.compute_frozen_jv:
        v_grid = np.linspace(spec.jv_v_min_V, spec.jv_v_max_V, spec.jv_n_points)
        probe = measure_frozen_chemistry_jv(
            compiled,
            state,
            v_grid,
            generation_scale=spec.generation_scale,
        )
        frozen_jv_payload = {
            "voltages_V": [float(v) for v in probe.voltages_V],
            "output_current_A_m2": [float(j) for j in probe.output_current_A_m2],
            "metrics": asdict(probe.metrics),
        }

    return {
        "schema": "solarlab.iodine-steady-result.v1",
        "deck_id": deck.id,
        "deck_name": deck.name,
        "device_sha256": compiled.identity,
        "domain_mode": compiled.domain_mode,
        "operating_state_mode": state_mode,
        "voltage_V": spec.voltage_V,
        "generation_scale": spec.generation_scale,
        "profiles": _serialize_state_profiles(compiled, state),
        "frozen_jv": frozen_jv_payload,
        "wall_s": float(time.perf_counter() - t0),
        "scientific_acceptance": "not_evaluated",
        "scientific_qualification": False,
    }


def _serialize_impedance_sweep(res: IodineImpedanceResult) -> dict[str, Any]:
    return {
        "chem_mode": res.chem_mode,
        "frequencies_Hz": [float(f) for f in res.frequencies_Hz],
        "admittance_real_S_m2": [float(y.real) for y in res.admittance_S_m2],
        "admittance_imag_S_m2": [float(y.imag) for y in res.admittance_S_m2],
        "impedance_real_Ohm_m2": [float(z.real) for z in res.impedance_Ohm_m2],
        "impedance_imag_Ohm_m2": [float(z.imag) for z in res.impedance_Ohm_m2],
        "apparent_capacitance_F_m2": [float(c) for c in res.apparent_capacitance_F_m2],
        "max_maxwell_spread_A_m2_V": float(res.max_maxwell_spread_A_m2_V),
        "max_backward_error": float(res.max_backward_error),
    }


def execute_iodine_impedance_deck(
    deck: IodineDeckInput,
    *,
    source: SourceDocument | None = None,
) -> dict[str, Any]:
    """Run coupled operating-point solve and small-signal AC impedance / capacitance sweep."""
    t0 = time.perf_counter()
    _, compiled = compile_iodine_deck(deck, source=source)
    spec = deck.impedance
    state, state_mode = resolve_iodine_operating_state(
        compiled,
        voltage_V=spec.voltage_V,
        generation_scale=spec.generation_scale,
        open_loss_precondition_s=deck.steady.open_loss_precondition_s,
    )
    freqs: NDArray[np.float64] = np.geomspace(
        spec.f_min_Hz, spec.f_max_Hz, spec.n_frequencies
    )
    modes: tuple[ChemResponseMode, ...] = (
        ("mobile", "frozen") if spec.chem_mode == "both" else (spec.chem_mode,)
    )
    sweeps: dict[str, Any] = {}
    for m in modes:
        res = solve_iodine_small_signal_1d(
            compiled,
            state,
            freqs,
            generation_scale=spec.generation_scale,
            chem_mode=m,
        )
        sweeps[m] = _serialize_impedance_sweep(res)

    redox_excess_cap: list[float] | None = None
    if "mobile" in sweeps and "frozen" in sweeps:
        c_mob = sweeps["mobile"]["apparent_capacitance_F_m2"]
        c_frz = sweeps["frozen"]["apparent_capacitance_F_m2"]
        redox_excess_cap = [float(cm - cf) for cm, cf in zip(c_mob, c_frz, strict=True)]

    return {
        "schema": "solarlab.iodine-impedance-result.v1",
        "deck_id": deck.id,
        "deck_name": deck.name,
        "device_sha256": compiled.identity,
        "domain_mode": compiled.domain_mode,
        "operating_state_mode": state_mode,
        "voltage_V": spec.voltage_V,
        "generation_scale": spec.generation_scale,
        "frequencies_Hz": [float(f) for f in freqs],
        "sweeps": sweeps,
        "redox_excess_capacitance_F_m2": redox_excess_cap,
        "profiles": _serialize_state_profiles(compiled, state),
        "wall_s": float(time.perf_counter() - t0),
        "scientific_acceptance": "not_evaluated",
        "scientific_qualification": False,
    }


def execute_iodine_aging_deck(
    deck: IodineDeckInput,
    *,
    source: SourceDocument | None = None,
    store_dir: Path | None = None,
    bundle_dir: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run multi-stage iodine stress/recovery aging protocol and optional persistence."""
    if deck.aging is None:
        raise ValueError(f"iodine deck {deck.id!r} does not define an 'aging' section")
    _, compiled = compile_iodine_deck(deck, source=source)
    protocol = deck.aging.to_protocol()
    result = run_iodine_aging_protocol(compiled, protocol)

    persisted_store: dict[str, str] | None = None
    if store_dir is not None:
        store_root = Path(store_dir).resolve()
        actual_run_id = run_id or f"iodine-aging-{deck.id}-{int(time.time() * 1000)}"
        with RunStore(store_root) as store:
            run_rec = record_iodine_aging_in_run_store(
                store,
                actual_run_id,
                compiled,
                protocol,
                result,
            )
            persisted_store = {
                "store_id": store.store_id,
                "run_id": str(run_rec["run_id"]),
                "attempt_id": str(run_rec["current_attempt_id"]),
            }

    if bundle_dir is not None:
        save_iodine_aging_bundle(bundle_dir, result)

    checkpoints_out: list[dict[str, Any]] = []
    for ckpt in result.checkpoints:
        frozen_dict: dict[str, Any] | None = None
        if ckpt.frozen_jv is not None:
            frozen_dict = {
                "voltages_V": [float(v) for v in ckpt.frozen_jv.voltages_V],
                "output_current_A_m2": [
                    float(j) for j in ckpt.frozen_jv.output_current_A_m2
                ],
                "metrics": asdict(ckpt.frozen_jv.metrics),
            }
        dyn_dict: dict[str, Any] | None = None
        if ckpt.dynamic_jv is not None:
            dyn_dict = {
                "voltages_V": [float(v) for v in ckpt.dynamic_jv.voltages_V],
                "scan_rate_V_s": float(ckpt.dynamic_jv.scan_rate_V_s),
                "reverse_output_current_A_m2": [
                    float(j) for j in ckpt.dynamic_jv.reverse_output_current_A_m2
                ],
                "forward_output_current_A_m2": [
                    float(j) for j in ckpt.dynamic_jv.forward_output_current_A_m2
                ],
                "reverse_metrics": asdict(ckpt.dynamic_jv.reverse_metrics),
                "forward_metrics": asdict(ckpt.dynamic_jv.forward_metrics),
                "hysteresis_index": ckpt.dynamic_jv.hysteresis_index,
            }
        checkpoints_out.append(
            {
                "time_s": float(ckpt.time_s),
                "stage_name": ckpt.stage_name,
                "in_device_iodine_m2": float(ckpt.in_device_iodine_m2),
                "cumulative_escaped_iodine_atoms_m2": float(
                    ckpt.cumulative_escaped_iodine_atoms_m2
                ),
                "cumulative_immobilized_iodine_atoms_m2": float(
                    ckpt.cumulative_immobilized_iodine_atoms_m2
                ),
                "conservation_residual_m2": float(ckpt.conservation_residual_m2),
                "profiles": _serialize_state_profiles(compiled, ckpt.state),
                "frozen_jv": frozen_dict,
                "dynamic_jv": dyn_dict,
            }
        )

    step_times = [0.0] + [float(step.end_s) for step in result.step_ledger]
    step_in_device = [float(result.reference_iodine_inventory_m2)] + [
        float(step.in_device_iodine_m2) for step in result.step_ledger
    ]
    cum_esc = 0.0
    cum_imm = 0.0
    step_escaped = [0.0]
    step_immobilized = [0.0]
    for step in result.step_ledger:
        cum_esc += float(step.step_escaped_iodine_atoms_m2)
        cum_imm += float(step.step_immobilized_iodine_atoms_m2)
        step_escaped.append(cum_esc)
        step_immobilized.append(cum_imm)

    return {
        "schema": "solarlab.iodine-aging-result.v1",
        "deck_id": deck.id,
        "deck_name": deck.name,
        "protocol_sha256": result.protocol_sha256,
        "device_sha256": result.device_sha256,
        "domain_mode": compiled.domain_mode,
        "reference_iodine_inventory_m2": float(result.reference_iodine_inventory_m2),
        "max_relative_conservation_error": float(
            result.max_relative_conservation_error
        ),
        "accepted_bdf_steps": len(result.step_ledger),
        "completed": bool(result.completed),
        "wall_s": float(result.wall_s),
        "trajectory": {
            "times_s": step_times,
            "in_device_iodine_m2": step_in_device,
            "cumulative_escaped_iodine_atoms_m2": step_escaped,
            "cumulative_immobilized_iodine_atoms_m2": step_immobilized,
        },
        "checkpoints": checkpoints_out,
        "persisted_run_store": persisted_store,
        "scientific_acceptance": "not_evaluated",
        "scientific_qualification": False,
    }
