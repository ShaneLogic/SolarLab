"""Named starting selections, without the old mode-ceiling semantics.

Migration supplies effective choices and historical disables with their source
evidence. This module neither imports old drivers nor guesses effective physics
from mode flags. Unresolved driver/environment behavior remains explicit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from solarlab.discretization.face_average import (
    DIELECTRIC_RULES,
    LAYER_SERIES_CAPACITANCE_V2,
    NODAL_HARMONIC_DIELECTRIC_V1,
)
from solarlab.materials.source import SourceDocument
from solarlab.mesh.geometry import (
    GEOMETRY_NAMES,
    HALF_ENDPOINT_V1,
    INTEGER_ENDPOINT_V1,
    PHYSICAL_HALF_ENDPOINT_V2,
)
from solarlab.physics.interface_transport.thermionic import (
    LEGACY_UNNORMALIZED_THERMIONIC_V1,
    RECIPROCAL_PHYSICAL_THERMIONIC_V2,
    THERMIONIC_MODEL_VERSIONS,
    ThermionicCombinationMode,
    ThermionicModelVersion,
)
from solarlab.physics.registry import (
    CapabilityContext, CapabilityRule, EvidenceRef, ModelDefinition, ModelRef, ModelRegistry, Scope,
    _strings, _token, _verify_evidence_sources, metadata_digest, metadata_value,
)
from solarlab.physics.selection import (
    ModelChoice, ParameterSource, Selection, Topology, _choices, select_models,
)
from solarlab.physics.transport.electrochemical_flux import (
    BAND_STEP_MODES,
    HOLE_SEPARATE_INCREMENT_V1,
    LEGACY_SUMMED_POTENTIAL_V1,
)

__all__ = [
    "HistoricalDisable",
    "NamedPreset",
    "P08CompilerOptions",
    "P08ModelVersionMapping",
    "P08_MODEL_VERSION_MAPPINGS",
    "P08_SOURCE_PATHS",
    "P08_V1_MODEL_REFS",
    "P08_V2_MODEL_REFS",
    "PresetSelection",
    "build_p08_model_registry",
    "build_p08_preset",
    "select_preset",
]


@dataclass(frozen=True, slots=True)
class HistoricalDisable:
    """One source-observed suppression, with explicit model-version mapping."""

    id: str
    models: tuple[ModelRef, ...]
    evidence: EvidenceRef

    def __post_init__(self) -> None:
        _token(self.id, "historical disabled mechanism")
        models = tuple(self.models)
        if not models or any(not isinstance(model, ModelRef) for model in models):
            raise ValueError("historical disable requires explicit model versions")
        if len(set(models)) != len(models) or not isinstance(self.evidence, EvidenceRef):
            raise ValueError("invalid historical disable mapping/evidence")
        object.__setattr__(self, "models", tuple(sorted(models)))


@dataclass(frozen=True, slots=True)
class NamedPreset:
    id: str
    version: int
    label: str
    defaults: tuple[tuple[str, tuple[ModelChoice, ...]], ...]
    evidence: tuple[EvidenceRef, ...]
    historical_disables: tuple[HistoricalDisable, ...] = ()
    unresolved_behavior: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ModelRef(self.id, self.version)
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("preset requires a display label")
        defaults = tuple(self.defaults)
        if any(not isinstance(item, tuple) or len(item) != 2 for item in defaults):
            raise ValueError("preset defaults require (scope kind, choices) pairs")
        if len({kind for kind, _ in defaults}) != len(defaults):
            raise ValueError("duplicate preset scope kind")
        if any(kind not in {"device", "layer", "interface", "contact"} for kind, _ in defaults):
            raise ValueError("unknown preset scope kind")
        object.__setattr__(self, "defaults", tuple(sorted(
            (kind, tuple(sorted(_choices(choices), key=lambda item: item.id)))
            for kind, choices in defaults
        )))
        evidence = tuple(self.evidence)
        if not evidence or any(not isinstance(item, EvidenceRef) for item in evidence):
            raise ValueError("preset requires evidence for its supplied defaults")
        object.__setattr__(self, "evidence", tuple(sorted(set(evidence))))
        disabled = tuple(self.historical_disables)
        if any(not isinstance(item, HistoricalDisable) for item in disabled):
            raise ValueError("invalid historical disable record")
        if len({item.id for item in disabled}) != len(disabled):
            raise ValueError("duplicate historical mechanism ID")
        active = {choice.model for _, choices in self.defaults for choice in choices}
        if any(active.intersection(item.models) for item in disabled):
            raise ValueError("starting preset enables a historically disabled mechanism")
        object.__setattr__(self, "historical_disables", tuple(sorted(disabled, key=lambda item: item.id)))
        object.__setattr__(self, "unresolved_behavior", _strings(self.unresolved_behavior, "unresolved legacy behavior"))

    @property
    def content_sha256(self) -> str:
        return metadata_digest(metadata_value(self))

    def verify_sources(
        self, sources: Iterable[tuple[str, bytes]], *, expected_sha256: str,
    ) -> None:
        """Bind default/disable records to bytes without changing their policy."""
        if expected_sha256 != self.content_sha256:
            raise ValueError("preset changed from the expected frozen identity")
        _verify_evidence_sources(
            (*self.evidence, *(item.evidence for item in self.historical_disables)), sources,
        )


@dataclass(frozen=True, slots=True)
class PresetSelection:
    preset: NamedPreset
    selection: Selection

    def __post_init__(self) -> None:
        if not isinstance(self.preset, NamedPreset) or not isinstance(self.selection, Selection):
            raise ValueError("preset selection requires validated metadata")

    @property
    def reenabled_historical_mechanisms(self) -> tuple[str, ...]:
        active = {item.model for item in self.selection.instances}
        return tuple(item.id for item in self.preset.historical_disables if active.intersection(item.models))

    @property
    def content_sha256(self) -> str:
        return metadata_digest(metadata_value(self))


def select_preset(
    preset: NamedPreset, registry: ModelRegistry, topology: Topology,
    context: CapabilityContext, parameter_sources: Mapping[Scope, ParameterSource],
    overrides: Mapping[Scope, Mapping[str, Sequence[ModelChoice]]] | None = None,
    *, instance_parameter_sources: Mapping[str, ParameterSource] | None = None,
) -> PresetSelection:
    """Explicit overrides may depart from history; that departure is retained."""
    if not isinstance(preset, NamedPreset):
        raise ValueError("a named preset is required")
    return PresetSelection(preset, select_models(
        registry, topology, context, dict(preset.defaults), parameter_sources, overrides,
        instance_parameter_sources=instance_parameter_sources,
    ))


OpticalModelChoice = Literal["beer_lambert", "beer_lambert_spectral", "tmm"]


@dataclass(frozen=True, slots=True)
class P08ModelVersionMapping:
    """Explicit pre-P08 (v1) to prospective P08 (v2) model-version correspondence."""

    item_id: str
    family: str
    local_instance_id: str
    v1_model: ModelRef
    v2_model: ModelRef
    v1_token: str
    v2_token: str
    v1_source_path: str
    v2_source_path: str
    layer_parameters: tuple[str, ...]
    v1_label: str
    v2_label: str
    semantic_delta: str


P08_MODEL_VERSION_MAPPINGS: tuple[P08ModelVersionMapping, ...] = (
    P08ModelVersionMapping(
        item_id="P08-02a",
        family="electrostatics_face_permittivity",
        local_instance_id="face_permittivity",
        v1_model=ModelRef("electrostatics.face_permittivity", 1),
        v2_model=ModelRef("electrostatics.face_permittivity", 2),
        v1_token=NODAL_HARMONIC_DIELECTRIC_V1,
        v2_token=LAYER_SERIES_CAPACITANCE_V2,
        v1_source_path="solarlab/discretization/face_average.py",
        v2_source_path="solarlab/physics/electrostatics/poisson.py",
        layer_parameters=("eps_r",),
        v1_label="Unweighted nodal harmonic face permittivity (v1)",
        v2_label="Sub-face material-extent series dielectric capacitance (v2)",
        semantic_delta=(
            "Replaces unweighted nodal harmonic face means and shared-node right-layer "
            "permittivity bleed with exact sub-face material-extent series capacitance "
            "C_f = eps_0 / (d_L/eps_L + d_R/eps_R) and 2D cell-parallel horizontal permittivity."
        ),
    ),
    P08ModelVersionMapping(
        item_id="P08-02b",
        family="control_volume_geometry",
        local_instance_id="control_volume",
        v1_model=ModelRef("mesh.endpoint_control_volume", 1),
        v2_model=ModelRef("mesh.endpoint_control_volume", 2),
        v1_token=INTEGER_ENDPOINT_V1,
        v2_token=PHYSICAL_HALF_ENDPOINT_V2,
        v1_source_path="solarlab/mesh/geometry.py",
        v2_source_path="solarlab/mesh/geometry.py",
        layer_parameters=("N_A", "N_D"),
        v1_label="Historical integer-endpoint 1D control volumes (v1)",
        v2_label="Coupled physical half-endpoint control volumes sum(w)==L (v2)",
        semantic_delta=(
            "Couples contact half-cells (w_0 = dx_0/2, w_{N-1} = dx_{N-1}/2, sum(w) == L) "
            "simultaneously across carrier storage, mobile ions, optical generation, Poisson "
            "contact displacement, and 1D/2D column ports."
        ),
    ),
    P08ModelVersionMapping(
        item_id="P08-02c",
        family="band_step_differencing",
        local_instance_id="band_step",
        v1_model=ModelRef("transport.band_step_differencing", 1),
        v2_model=ModelRef("transport.band_step_differencing", 2),
        v1_token=LEGACY_SUMMED_POTENTIAL_V1,
        v2_token=HOLE_SEPARATE_INCREMENT_V1,
        v1_source_path="solarlab/physics/transport/electrochemical_flux.py",
        v2_source_path="solarlab/physics/transport/electrochemical_flux.py",
        layer_parameters=("mu_n", "mu_p"),
        v1_label="Summed nodal transport-potential differencing (v1)",
        v2_label="Separate electrostatic and band-offset face increments (v2)",
        semantic_delta=(
            "Forms (d_phi + d_chi_t) + d_Eg_t per face before summation so ~5.5 eV valence-band "
            "offsets cannot quantize sub-millivolt intra-layer electrostatic steps in binary64."
        ),
    ),
    P08ModelVersionMapping(
        item_id="P08-03",
        family="interface_thermionic",
        local_instance_id="thermionic",
        v1_model=ModelRef("interface_transport.thermionic_emission", 1),
        v2_model=ModelRef("prospective_interface_p08", 2),
        v1_token=LEGACY_UNNORMALIZED_THERMIONIC_V1,
        v2_token=RECIPROCAL_PHYSICAL_THERMIONIC_V2,
        v1_source_path="solarlab/physics/interface_transport/thermionic.py",
        v2_source_path="solarlab/physics/interface_transport/thermionic.py",
        layer_parameters=("A_star_n", "A_star_p", "chi", "Eg", "Nc300", "Nv300"),
        v1_label="Legacy density-weighted unnormalized thermionic cap (v1)",
        v2_label="Reciprocal unequal-DOS physical Bethe/Richardson emission (v2)",
        semantic_delta=(
            "Separates unfolded physical band-edge steps (chi_phys, Eg_phys), per-side DOS "
            "activities (c_L/N_L, c_R/N_R), reciprocal min(A*_L, A*_R), transmission T, and "
            "cancellation-free expm1 electrochemical drive with C^infinity flatband corners."
        ),
    ),
    P08ModelVersionMapping(
        item_id="P08-04",
        family="optical_generation",
        local_instance_id="optics",
        v1_model=ModelRef("monochromatic_beer_lambert_1d", 1),
        v2_model=ModelRef("spectral_control_volume_beer_lambert", 2),
        v1_token="beer_lambert",
        v2_token="beer_lambert_spectral",
        v1_source_path="solarlab/physics/generation/beer_lambert.py",
        v2_source_path="solarlab/physics/generation/beer_lambert_spectral.py",
        layer_parameters=("alpha", "n_optical", "incoherent"),
        v1_label="Pre-P08 monochromatic Beer-Lambert / coherent TMM optics (v1)",
        v2_label="Wavelength-resolved control-volume single-pass Beer-Lambert (v2)",
        semantic_delta=(
            "Integrates spectral alpha(lambda) = 4*pi*k(lambda)/lambda over exact physical "
            "control-volume optical depths with closed front-substrate + electrical + transmitted "
            "photon budget."
        ),
    ),
)

P08_SOURCE_PATHS: tuple[str, ...] = tuple(
    sorted(
        {m.v1_source_path for m in P08_MODEL_VERSION_MAPPINGS}
        | {m.v2_source_path for m in P08_MODEL_VERSION_MAPPINGS}
    )
)
P08_V1_MODEL_REFS: tuple[ModelRef, ...] = tuple(m.v1_model for m in P08_MODEL_VERSION_MAPPINGS)
P08_V2_MODEL_REFS: tuple[ModelRef, ...] = tuple(m.v2_model for m in P08_MODEL_VERSION_MAPPINGS)


@dataclass(frozen=True, slots=True)
class P08CompilerOptions:
    """Explicit opt-in selector bundle for Stage P08 prospective physics channels.

    Default ``P08CompilerOptions()`` leaves every field at its pre-P08 ``v1``
    setting, preserving 0-ULP / 0-byte manifest and SHA-256 parity with the
    ``refactor/wave2-n-baseline`` baseline.
    """

    dielectric_rule: str | None = None
    geometry_name: str | None = None
    band_step_mode: str = LEGACY_SUMMED_POTENTIAL_V1
    thermionic_mode: ThermionicModelVersion | None = None
    combination_mode: ThermionicCombinationMode | None = None
    optical_model: OpticalModelChoice | None = None

    def __post_init__(self) -> None:
        if self.dielectric_rule is not None and self.dielectric_rule not in DIELECTRIC_RULES:
            raise ValueError(f"unknown dielectric_rule {self.dielectric_rule!r}")
        if self.geometry_name is not None and self.geometry_name not in GEOMETRY_NAMES:
            raise ValueError(f"unknown geometry_name {self.geometry_name!r}")
        if self.band_step_mode not in BAND_STEP_MODES:
            raise ValueError(f"unknown band_step_mode {self.band_step_mode!r}")
        if self.thermionic_mode is not None and self.thermionic_mode not in THERMIONIC_MODEL_VERSIONS:
            raise ValueError(f"unknown thermionic_mode {self.thermionic_mode!r}")
        if self.combination_mode is not None and self.combination_mode not in (
            "harmonic_series",
            "compact_support",
            "legacy_logistic",
            "hard",
        ):
            raise ValueError(f"unknown combination_mode {self.combination_mode!r}")
        if (
            self.combination_mode is not None
            and (self.thermionic_mode is None or self.thermionic_mode == LEGACY_UNNORMALIZED_THERMIONIC_V1)
        ):
            raise ValueError(
                "combination_mode requires thermionic_mode='reciprocal_physical_thermionic_v2' "
                "or 'scaps_effective_velocity_thermionic_v1'"
            )
        if self.optical_model is not None and self.optical_model not in (
            "beer_lambert",
            "beer_lambert_spectral",
            "tmm",
        ):
            raise ValueError(f"unknown optical_model {self.optical_model!r}")

    @classmethod
    def all_prospective_v2(
        cls,
        *,
        combination_mode: ThermionicCombinationMode = "harmonic_series",
        optical_model: OpticalModelChoice | None = "beer_lambert_spectral",
    ) -> P08CompilerOptions:
        """Construct the full Stage P08 prospective V2 physics bundle."""
        return cls(
            dielectric_rule=LAYER_SERIES_CAPACITANCE_V2,
            geometry_name=PHYSICAL_HALF_ENDPOINT_V2,
            band_step_mode=HOLE_SEPARATE_INCREMENT_V1,
            thermionic_mode=RECIPROCAL_PHYSICAL_THERMIONIC_V2,
            combination_mode=combination_mode,
            optical_model=optical_model,
        )

    @property
    def is_v1_default(self) -> bool:
        return (
            self.dielectric_rule is None
            and self.geometry_name is None
            and self.band_step_mode == LEGACY_SUMMED_POTENTIAL_V1
            and (self.thermionic_mode is None or self.thermionic_mode == LEGACY_UNNORMALIZED_THERMIONIC_V1)
            and self.combination_mode is None
            and self.optical_model is None
        )

    def active_model_refs(self) -> tuple[ModelRef, ...]:
        """Return the resolved (v1 or v2) ModelRef tuple for the five P08 families."""
        dielectric_v2 = (
            self.dielectric_rule == LAYER_SERIES_CAPACITANCE_V2
            or (self.dielectric_rule is None and self.geometry_name == PHYSICAL_HALF_ENDPOINT_V2)
        )
        geometry_v2 = self.geometry_name == PHYSICAL_HALF_ENDPOINT_V2
        band_v2 = self.band_step_mode == HOLE_SEPARATE_INCREMENT_V1
        dielectric_ref = (
            P08_MODEL_VERSION_MAPPINGS[0].v2_model
            if dielectric_v2
            else P08_MODEL_VERSION_MAPPINGS[0].v1_model
        )
        geometry_ref = (
            P08_MODEL_VERSION_MAPPINGS[1].v2_model
            if geometry_v2
            else P08_MODEL_VERSION_MAPPINGS[1].v1_model
        )
        band_ref = (
            P08_MODEL_VERSION_MAPPINGS[2].v2_model
            if band_v2
            else P08_MODEL_VERSION_MAPPINGS[2].v1_model
        )
        if self.thermionic_mode == RECIPROCAL_PHYSICAL_THERMIONIC_V2:
            if self.combination_mode == "compact_support":
                te_ref = ModelRef("prospective_interface_p08_compact_support", 2)
            elif self.combination_mode == "legacy_logistic":
                te_ref = ModelRef("prospective_interface_p08_logistic", 2)
            elif self.combination_mode == "hard":
                te_ref = ModelRef("prospective_interface_p08_hard", 2)
            else:
                te_ref = P08_MODEL_VERSION_MAPPINGS[3].v2_model
        elif self.thermionic_mode == "scaps_effective_velocity_thermionic_v1":
            te_ref = ModelRef("prospective_interface_p08_scaps_velocity", 2)
        else:
            te_ref = P08_MODEL_VERSION_MAPPINGS[3].v1_model

        if self.optical_model == "beer_lambert_spectral":
            optics_ref = P08_MODEL_VERSION_MAPPINGS[4].v2_model
        elif self.optical_model == "tmm":
            optics_ref = ModelRef("tmm_spectral_generation_p08", 2)
        else:
            optics_ref = P08_MODEL_VERSION_MAPPINGS[4].v1_model
        return (dielectric_ref, geometry_ref, band_ref, te_ref, optics_ref)

    def compiler_kwargs_1d(self) -> dict[str, Any]:
        """Keyword arguments for ``compile_device_1d`` / ``compile_ionic_device_1d``."""
        kwargs: dict[str, Any] = {}
        if self.dielectric_rule is not None:
            kwargs["dielectric_rule"] = self.dielectric_rule
        if self.geometry_name is not None:
            kwargs["geometry_name"] = self.geometry_name
        if self.band_step_mode != LEGACY_SUMMED_POTENTIAL_V1:
            kwargs["band_step_mode"] = self.band_step_mode
        if self.thermionic_mode is not None:
            kwargs["thermionic_mode"] = self.thermionic_mode
        if self.combination_mode is not None:
            kwargs["combination_mode"] = self.combination_mode
        if self.optical_model is not None:
            kwargs["optical_model"] = self.optical_model
        return kwargs

    def compiler_kwargs_2d(self) -> dict[str, Any]:
        """Keyword arguments for ``compile_device_2d`` / ``compile_ionic_device_2d``."""
        kwargs = self.compiler_kwargs_1d()
        if "geometry_name" in kwargs and kwargs["geometry_name"] in {
            HALF_ENDPOINT_V1,
            PHYSICAL_HALF_ENDPOINT_V2,
        }:
            kwargs["stack_geometry"] = kwargs.pop("geometry_name")
        return kwargs


def build_p08_model_registry(sources: Mapping[str, SourceDocument]) -> ModelRegistry:
    """Build a source-bound ModelRegistry covering both v1 and v2 P08 model versions."""
    missing = [path for path in P08_SOURCE_PATHS if path not in sources]
    if missing:
        raise ValueError("missing P08 source documents: " + ", ".join(missing))
    definitions: list[ModelDefinition] = []
    for mapping in P08_MODEL_VERSION_MAPPINGS:
        variants: list[tuple[ModelRef, str, str]] = [
            (mapping.v1_model, mapping.v1_label, mapping.v1_source_path),
            (mapping.v2_model, mapping.v2_label, mapping.v2_source_path),
        ]
        if mapping.item_id == "P08-03":
            variants.extend([
                (
                    ModelRef("prospective_interface_p08_compact_support", 2),
                    "Reciprocal unequal-DOS physical Bethe/Richardson emission (compact_support, v2)",
                    mapping.v2_source_path,
                ),
                (
                    ModelRef("prospective_interface_p08_logistic", 2),
                    "Reciprocal unequal-DOS physical Bethe/Richardson emission (legacy_logistic, v2)",
                    mapping.v2_source_path,
                ),
                (
                    ModelRef("prospective_interface_p08_hard", 2),
                    "Reciprocal unequal-DOS physical Bethe/Richardson emission (hard cap, v2)",
                    mapping.v2_source_path,
                ),
                (
                    ModelRef("prospective_interface_p08_scaps_velocity", 2),
                    "SCAPS effective-velocity interface thermionic emission (v2)",
                    mapping.v2_source_path,
                ),
            ])
        elif mapping.item_id == "P08-04":
            variants.append(
                (
                    ModelRef("tmm_spectral_generation_p08", 2),
                    "Explicit P08 coherent TMM spectral optical generation (v2)",
                    mapping.v2_source_path,
                )
            )
        for ref, label, src_path in variants:
            doc = sources[src_path]
            if type(doc) is not SourceDocument or doc.id != src_path:
                raise ValueError(f"source document for {src_path!r} must have matching id")
            cap = CapabilityRule(
                "source_bound_p08_" + mapping.local_instance_id,
                (1, 2),
                ("dc", "transient", "ac"),
                (EvidenceRef(doc.id, doc.sha256, "source"),),
                backends=("solarlab_assembled", "metadata_only"),
                statistics=("maxwell_boltzmann", "fermi_dirac"),
            )
            definitions.append(
                ModelDefinition(
                    ref=ref,
                    label=label,
                    family=mapping.family,
                    scopes=("layer",),
                    support="node",
                    parameters=mapping.layer_parameters,
                    reads=(),
                    owns=(),
                    capability=cap,
                    parameter_schema="full_layer",
                )
            )
    return ModelRegistry(tuple(definitions))


def build_p08_preset(
    options: P08CompilerOptions,
    sources: Mapping[str, SourceDocument],
    *,
    preset_id: str | None = None,
    version: int | None = None,
) -> NamedPreset:
    """Build an immutable NamedPreset for either pre-P08 v1 or prospective P08 v2 options."""
    missing = [path for path in P08_SOURCE_PATHS if path not in sources]
    if missing:
        raise ValueError("missing P08 source documents: " + ", ".join(missing))
    active_refs = options.active_model_refs()
    choices = tuple(
        ModelChoice(mapping.local_instance_id, ref, parameter_schema="full_layer")
        for mapping, ref in zip(P08_MODEL_VERSION_MAPPINGS, active_refs)
    )
    evidence = tuple(
        EvidenceRef(sources[path].id, sources[path].sha256, "source")
        for path in P08_SOURCE_PATHS
    )
    is_v1 = all(ref.version == 1 for ref in active_refs)
    resolved_id = preset_id or ("solarlab_baseline_physics" if is_v1 else "solarlab_prospective_p08")
    resolved_ver = version if version is not None else (1 if is_v1 else 2)
    label = (
        "SolarLab pre-P08 v1 baseline physics preset"
        if is_v1
        else "SolarLab Stage P08 prospective v2 physics preset"
    )
    disables: tuple[HistoricalDisable, ...] = ()
    if is_v1:
        disables = tuple(
            HistoricalDisable(
                "prospective_" + mapping.local_instance_id + "_v2",
                (mapping.v2_model,),
                EvidenceRef(sources[mapping.v2_source_path].id, sources[mapping.v2_source_path].sha256, "source"),
            )
            for mapping in P08_MODEL_VERSION_MAPPINGS
        )
    return NamedPreset(
        id=resolved_id,
        version=resolved_ver,
        label=label,
        defaults=(("layer", choices),),
        evidence=evidence,
        historical_disables=disables,
    )
