"""P03 metadata against real scalar inputs and explicitly scoped source contracts.

These tests execute no physics or legacy runner. Context facts are declarations
awaiting the gated physical validators; even compatible metadata stays unable
to run. Source evidence below is read from the checkout, never imported.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path

import pytest

from solarlab.config.inputs import LayerInput
from solarlab.config.resolve import load_template_library, resolve_layers
from solarlab.experiments.registry import ExperimentDefinition, ExperimentRegistry
from solarlab.materials.parameters import LayerValuesInput
from solarlab.physics.presets import HistoricalDisable, NamedPreset, select_preset
from solarlab.physics.registry import (
    CapabilityContext, CapabilityRule, EvidenceRef, ModelDefinition, ModelRef,
    ModelRegistry, OwnedVariable, Requirement, Scope, metadata_value,
)
from solarlab.physics.selection import (
    ModelChoice, ParameterSource, Topology, select_models, validate_selection,
)

ROOT = Path(__file__).resolve().parents[2]
LEGACY = "perovskite-sim/perovskite_sim/"
TRANSPORT = ModelRef("transport.constant", 1)
SRH = ModelRef("recombination.srh", 1)
INTERFACE = ModelRef("interface.two_sided", 1)
LEFT = Scope("layer", ("left",))
RIGHT = Scope("layer", ("right",))
JOIN = Scope("interface", ("left", "right"))


def evidence(path: str, symbol: str = "") -> EvidenceRef:
    return EvidenceRef(path + ("#" + symbol if symbol else ""),
                       hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), "source")


def declared_rule(**changes: object) -> CapabilityRule:
    rule = CapabilityRule(
        "scalar_metadata", (1, 2), ("dc", "transient"),
        (evidence(LEGACY + "models/parameters.py"),),
        statistics=("maxwell_boltzmann",),
    )
    return replace(rule, **changes)


@pytest.fixture
def inputs():
    document = (ROOT / (LEGACY + "data/layer_templates.yaml")).read_bytes()
    library = load_template_library(document, source_id="shipped.layer_templates")
    # Deliberately repeated display names; stable IDs define topology.
    layers = resolve_layers(tuple(LayerInput.model_validate({
        "id": id, "name": "absorber", "template": "MAPbI3_absorber",
    }) for id in ("left", "right")), library)
    sources = {Scope("layer", (layer.id,)): ParameterSource(
        layer.id, (*layer.material.parameters, ("thickness", layer.thickness),
                   ("N_A", layer.N_A), ("N_D", layer.N_D)),
        EvidenceRef(layer.source_id, layer.source_sha256, "source"),
    ) for layer in layers}
    return Topology(("left", "right"), (("anode", "left"), ("cathode", "right"))), sources


@pytest.fixture
def registry() -> ModelRegistry:
    transport = ModelDefinition(
        TRANSPORT, "Constant carrier mobility", "transport", ("layer",), "face",
        ("mu_n", "mu_p"), ("phi",),
        (OwnedVariable("n", "m^-3", "node", ("node",)), OwnedVariable("p", "m^-3", "node", ("node",))),
        declared_rule(),
    )
    srh = ModelDefinition(
        SRH, "Effective lifetime SRH", "recombination", ("layer",), "node",
        ("tau_n", "tau_p", "n1", "p1", "ni"), ("n", "p"), (),
        declared_rule(evidence=(evidence(LEGACY + "physics/recombination.py"),)),
        composition="additive", requires=(Requirement(TRANSPORT),),
    )
    interface = ModelDefinition(
        INTERFACE, "Directed two-sided trace metadata", "interface", ("interface",),
        "interface_left", (), ("left_n", "right_n"),
        (OwnedVariable("trace_n", "m^-3", "interface_left", ()),),
        declared_rule(evidence=(evidence(LEGACY + "physics/two_sided_interface.py"),)),
        requires=(Requirement(TRANSPORT, "left"), Requirement(TRANSPORT, "right")),
    )
    return ModelRegistry((transport, srh, interface))


def context(**changes: object) -> CapabilityContext:
    return replace(CapabilityContext(1, "dc", "metadata_only", "maxwell_boltzmann",
                                     available_variables=("phi",)), **changes)


def srh_choice(id: str = "srh", layer: str = "left") -> ModelChoice:
    return ModelChoice(id, SRH, field_bindings=(
        ("n", f"layer/{layer}/carrier:n"), ("p", f"layer/{layer}/carrier:p"),
    ))


def selection(registry, inputs, **changes):
    topology, sources = inputs
    kwargs = dict(registry=registry, topology=topology, context=context(),
                  defaults={"layer": (ModelChoice("carrier", TRANSPORT),)},
                  parameter_sources=sources)
    kwargs.update(changes)
    return select_models(**kwargs)


def jv_definition() -> ExperimentDefinition:
    return ExperimentDefinition(
        "jv", 1, "J–V source contract awaiting runner migration",
        evidence("perovskite-sim/backend/main.py", "JVRequest"),
        evidence("perovskite-sim/backend/api_schema.py", "JVResultOut"),
        evidence(LEGACY + "experiments/protocol.py", "resolve_experiment_protocol"),
        evidence(LEGACY + "experiments/jv_sweep.py", "run_jv_sweep"),
        "jv", evidence("perovskite-sim/frontend/src/panels/jv.ts", "renderJVResults"),
        declared_rule(), (TRANSPORT,),
    )


def test_real_templates_select_scoped_models_and_preflight_without_executor(registry, inputs):
    selected = selection(registry, inputs, overrides={
        LEFT: {"recombination": (srh_choice(),)},
        JOIN: {"interface": (ModelChoice("trace", INTERFACE, field_bindings=(
            ("left_n", "layer/left/carrier:n"), ("right_n", "layer/right/carrier:n"),
        )),)},
    })
    assert {item.id for item in selected.instances} == {
        "layer/left/carrier", "layer/right/carrier", "layer/left/srh", "interface/left/right/trace",
    }
    assert selected.evaluation_order.index("layer/left/carrier") < selected.evaluation_order.index("layer/left/srh")
    assert selected.evaluation_order.index("layer/right/carrier") < selected.evaluation_order.index("interface/left/right/trace")
    srh = next(item for item in selected.instances if item.model == SRH)
    assert dict(srh.parameters)["tau_n"] == dict(inputs[1][LEFT].values)["tau_n"]
    assert srh.parameter_source_sha256 == inputs[1][LEFT].content_sha256
    assert srh.read_variables == ("layer/left/carrier:n", "layer/left/carrier:p")
    experiments = ExperimentRegistry((jv_definition(),))
    result = experiments.preflight("jv", 1, selected, registry)
    assert result.declared_compatible
    assert not result.can_execute
    assert result.availability_problems == ("qualified_executor_not_registered",)
    assert experiments.export()["registered_executors"] == []
    assert not hasattr(experiments, "run")


def test_selection_replaces_families_and_explicit_empty_clears(registry, inputs):
    transport = replace(registry.get(TRANSPORT), ref=ModelRef("transport.other", 1))
    expanded = ModelRegistry((*registry.definitions, transport))
    selected = selection(expanded, inputs, overrides={
        LEFT: {"transport": (ModelChoice("replacement", transport.ref, (("mu_n", 0.0),)),)},
        RIGHT: {"transport": ()},
    })
    assert len(selected.instances) == 1
    instance = selected.instances[0]
    assert instance.id == "layer/left/replacement"
    assert dict(instance.parameters)["mu_n"] == 0
    assert instance.overridden_parameters == ("mu_n",)
    assert len(instance.owned_variables) == 2  # Zero does not erase support.
    positive = selection(expanded, inputs, overrides={
        LEFT: {"transport": (ModelChoice("replacement", transport.ref, (("mu_n", 1e-320),)),)},
        RIGHT: {"transport": ()},
    })
    assert selected.content_sha256 != positive.content_sha256
    assert selected.instances[0].owned_variables == positive.instances[0].owned_variables


def test_additive_instances_and_deterministic_declaration_order(registry, inputs):
    overrides = {LEFT: {"recombination": (srh_choice("second"), srh_choice("first"))}}
    selected = selection(registry, inputs, overrides=overrides)
    reordered = ModelRegistry(tuple(reversed(registry.definitions)))
    other = selection(reordered, inputs, overrides={LEFT: {
        "recombination": tuple(reversed(overrides[LEFT]["recombination"])),
    }})
    assert len([x for x in selected.instances if x.model == SRH]) == 2
    assert reordered.content_sha256 == registry.content_sha256
    assert selected.content_sha256 == other.content_sha256
    jv = jv_definition()
    second = replace(jv, id="jv_other")
    assert ExperimentRegistry((jv, second)).content_sha256 == ExperimentRegistry((second, jv)).content_sha256
    assert replace(jv, label="Another display name").id == jv.id
    exported = registry.export()
    exported["models"].clear()
    assert registry.export()["models"]
    with pytest.raises(FrozenInstanceError):
        selected.instances[0].local_id = "changed"


@pytest.mark.parametrize("overrides,match", [
    ({LEFT: {"transport": (ModelChoice("dup", TRANSPORT), ModelChoice("dup", TRANSPORT))}}, "duplicate model instance"),
    ({LEFT: {"transport": (ModelChoice("one", TRANSPORT), ModelChoice("two", TRANSPORT))}}, "exclusive"),
    ({LEFT: {"transport": (ModelChoice("wrong", SRH),)}}, "does not belong"),
    ({LEFT: {"unknown_family": ()}}, "unknown model family"),
    ({Scope("interface", ("right", "left")): {}}, "directed electrical topology"),
    ({Scope("layer", ("absorber",)): {}}, "directed electrical topology"),
    ({LEFT: {"transport": None}}, None),
])
def test_invalid_scoped_overrides_reject(registry, inputs, overrides, match):
    with pytest.raises((ValueError, TypeError), match=match):
        selection(registry, inputs, overrides=overrides)


def test_unknown_unconsumed_missing_and_invalid_parameter_values(registry, inputs):
    with pytest.raises(ValueError, match="unconsumed"):
        selection(registry, inputs, overrides={LEFT: {"transport": (
            ModelChoice("carrier", TRANSPORT, (("D_ion", 0),)),
        )}})
    with pytest.raises(ValueError, match="missing consumed"):
        selection(registry, inputs, parameter_sources={})
    with pytest.raises(ValueError, match="parameter schema"):
        replace(registry.get(TRANSPORT), parameters=("Nc300",))
    for value in (True, None, float("nan"), float("inf"), -1.0, "1 kg"):
        with pytest.raises(ValueError):
            ModelChoice("carrier", TRANSPORT, (("mu_n", value),))
    with pytest.raises(ValueError):
        ModelChoice("carrier", TRANSPORT, (("unknown", 1.0),))


def test_missing_dependency_and_cycle_reject(registry, inputs):
    with pytest.raises(ValueError, match="missing dependency"):
        selection(registry, inputs, overrides={LEFT: {
            "transport": (), "recombination": (ModelChoice("srh", SRH),),
        }}, context=context(available_variables=("phi", "n", "p")))
    base = replace(registry.get(TRANSPORT), owns=(), reads=(), parameters=())
    a = replace(base, ref=ModelRef("a", 1), family="a", requires=(Requirement(ModelRef("b", 1)),))
    b = replace(base, ref=ModelRef("b", 1), family="b", requires=(Requirement(ModelRef("a", 1)),))
    cyclic = ModelRegistry((a, b))
    with pytest.raises(ValueError, match="cyclic"):
        selection(cyclic, inputs, defaults={"layer": (ModelChoice("a", a.ref), ModelChoice("b", b.ref))})


def test_conflicting_models_ownership_and_unbound_fields_reject(registry, inputs):
    conflicting = ModelRegistry(tuple(
        replace(item, conflicts=(TRANSPORT,)) if item.ref == SRH else item
        for item in registry.definitions
    ))
    with pytest.raises(ValueError, match="conflicting model"):
        selection(conflicting, inputs, overrides={LEFT: {"recombination": (srh_choice(),)}})
    with pytest.raises(ValueError, match="duplicate ownership"):
        selection(registry, inputs, defaults={"layer": (ModelChoice("carrier", TRANSPORT,
                  field_bindings=(("n", "shared_n"),)),)})
    with pytest.raises(ValueError, match="missing required physical variable"):
        selection(registry, inputs, context=context(available_variables=()))
    with pytest.raises(ValueError, match="unused field binding"):
        selection(registry, inputs, overrides={LEFT: {"transport": (
            ModelChoice("carrier", TRANSPORT, field_bindings=(("typo", "phi"),)),
        )}})


def test_registry_duplicate_unknown_versions_and_schema_drift(registry, inputs, monkeypatch):
    with pytest.raises(ValueError, match="duplicate model ID/version"):
        ModelRegistry((*registry.definitions, registry.get(TRANSPORT)))
    with pytest.raises(ValueError, match="unknown model ID/version"):
        registry.get(ModelRef(TRANSPORT.id, 2))
    with pytest.raises(ValueError, match="unknown dependency"):
        ModelRegistry((registry.get(SRH),))
    with pytest.raises(ValueError, match="unknown conflict"):
        ModelRegistry((replace(registry.get(TRANSPORT), conflicts=(ModelRef("absent", 1),)),))
    original = LayerValuesInput.model_json_schema()
    monkeypatch.setattr(LayerValuesInput, "model_json_schema", classmethod(lambda cls: {**original, "changed": True}))
    with pytest.raises(ValueError, match="schema changed"):
        selection(registry, inputs)


@pytest.mark.parametrize("build", [
    lambda: ModelRef("x", True), lambda: ModelRef("x", 0), lambda: ModelRef("a/b", 1),
    lambda: Scope("interface", ("same", "same")), lambda: Scope("layer", ("x", "y")),
    lambda: Scope("layer", "x"), lambda: EvidenceRef("source", "bad"),
    lambda: OwnedVariable("n", "m^-3", "node", ()),
    lambda: OwnedVariable("n", "m^-3", "node", ("x", "x")),
    lambda: OwnedVariable("n", "kilogram", "node", ("x",)),
    lambda: Topology(("same", "same")),
    lambda: Topology(("a", "b", "c"), (("contact", "b"),)),
    lambda: ModelChoice("carrier", TRANSPORT, field_bindings=(("n", "x"), ("n", "y"))),
    lambda: ModelChoice("carrier", TRANSPORT, field_bindings=("np",)),
    lambda: context(dimension=True), lambda: context(temperature_K=0),
    lambda: declared_rule(required_facts=("dark",), forbidden_facts=("dark",)),
])
def test_malformed_metadata_rejects(build):
    with pytest.raises(ValueError):
        build()


def test_one_pass_input_sequences_are_snapshotted_once():
    topology = Topology(("a", "b"), (("c", x) for x in ("a",)))
    assert topology.contacts == (("c", "a"),)
    choice = ModelChoice("carrier", TRANSPORT, field_bindings=(("n", x) for x in ("target",)))
    assert choice.field_bindings == (("n", "target"),)


def historical_disables() -> tuple[HistoricalDisable, ...]:
    # Derive the actual flag IDs; no duplicate default table in new core.
    path = LEGACY + "models/mode.py"
    tree = ast.parse((ROOT / path).read_text())
    call = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "LEGACY" for target in node.targets))
    flags = [item.arg for item in call.keywords if isinstance(item.value, ast.Constant) and item.value.value is False]
    result = tuple(HistoricalDisable(flag, (ModelRef("legacy." + flag, 1),), evidence(path, "LEGACY")) for flag in flags)
    # The MOL policy adds suppressions not represented by those nine flags.
    policy = evidence("reproducibility/RefactorCoverageV1.json", "driver:mol")
    return (*result, *(HistoricalDisable(name, (ModelRef("legacy." + name, 1),), policy)
                      for name in ("dos", "physical_te", "steric_diffusion")))


def test_named_preset_preserves_historical_disables_without_a_ceiling(registry, inputs):
    disabled = historical_disables()
    assert len(disabled) == 12
    preset = NamedPreset("legacy_observed", 1, "Observed legacy metadata", (
        ("layer", (ModelChoice("carrier", TRANSPORT),)),
    ), (evidence(LEGACY + "models/mode.py", "LEGACY"),), disabled,
        ("driver_environment_resolution_pending_P03_04",))
    result = select_preset(preset, registry, inputs[0], context(), inputs[1])
    assert result.reenabled_historical_mechanisms == ()
    assert result.preset.unresolved_behavior
    ref = next(item.models[0] for item in disabled if item.id == "use_field_dependent_mobility")
    planned = replace(registry.get(TRANSPORT), ref=ref)
    expanded = ModelRegistry((*registry.definitions, planned))
    changed = select_preset(preset, expanded, inputs[0], context(), inputs[1], {
        LEFT: {"transport": (ModelChoice("explicit", ref),)},
    })
    assert changed.reenabled_historical_mechanisms == ("use_field_dependent_mobility",)
    assert len(changed.preset.historical_disables) == 12
    with pytest.raises(ValueError, match="historically disabled"):
        replace(preset, defaults=(("layer", (ModelChoice("bad", ref),)),))


def narrow_rules() -> dict[str, CapabilityRule]:
    docs = "perovskite-sim/docs/"
    return {
        "fd": CapabilityRule("fd_equilibrium", (1,), ("equilibrium",),
            (evidence(docs + "DegenerateSemiconductorClosure.md"),),
            statistics=("fermi_dirac",), required_facts=("dark", "uniform_DOS", "homojunction", "ohmic", "recombination_off", "positive_Eg_Nc300_Nv300"),
            forbidden_facts=("illumination", "spatial_doping", "heterojunction", "interface_defects", "mobile_ions", "selective_contacts", "field_mobility", "advanced_interfaces", "environment_override"),
            required_resources=("semiconductor_work_function_contacts",)),
        "wkb": CapabilityRule("wkb_electron_dc", (1,), ("dc",),
            (evidence(docs + "ResolvedIntrabandTunnellingContractV2.md"),),
            backends=("qf",), statistics=("maxwell_boltzmann",), temperature_K=300,
            required_facts=("dark", "one_isolated_material_barrier", "adjacent_reservoirs", "electron_intraband"),
            forbidden_facts=("mobile_ions", "illumination", "multiple_barriers", "interface_plane", "hole_channel", "contact_channel", "band_to_band", "defect_assisted")),
        "dynamic_interface": CapabilityRule("interface_only_ac", (1,), ("ac",),
            (evidence(docs + "DynamicInterfaceDefectDeviceAc.md"),),
            required_facts=("two_sided",), forbidden_facts=("bulk_explicit_defects", "mobile_ions", "selective_contacts", "field_mobility", "photon_recycling")),
        "twod_combined": CapabilityRule("twod_mobile_interface", (2,), ("transient",),
            (evidence(docs + "TwodTransportContract.md"),),
            required_facts=("neumann_x", "two_sided", "single_positive_ion"),
            forbidden_facts=("field_mobility", "photon_recycling", "periodic_x", "dual_ions", "dynamic_interface_occupancy")),
    }


@pytest.mark.parametrize(("name", "change", "issue"), [
    ("fd", {"mode": "dc"}, "mode"), ("fd", {"facts": ("illumination",)}, "forbidden_fact:illumination"),
    ("fd", {"resources": ()}, "missing_resource:semiconductor_work_function_contacts"),
    ("fd", {"facts": ("mobile_ions",)}, "forbidden_fact:mobile_ions"),
    ("wkb", {"mode": "ac"}, "mode"), ("wkb", {"dimension": 2}, "dimension"),
    ("wkb", {"temperature_K": 301}, "temperature"), ("wkb", {"backend": "mol"}, "backend"),
    ("wkb", {"facts": ("hole_channel",)}, "forbidden_fact:hole_channel"),
    ("wkb", {"facts": ("mobile_ions",)}, "forbidden_fact:mobile_ions"),
    ("wkb", {"facts": ("multiple_barriers",)}, "forbidden_fact:multiple_barriers"),
    ("dynamic_interface", {"facts": ("bulk_explicit_defects",)}, "forbidden_fact:bulk_explicit_defects"),
    ("dynamic_interface", {"facts": ("mobile_ions",)}, "forbidden_fact:mobile_ions"),
    ("dynamic_interface", {"dimension": 2}, "dimension"),
    ("twod_combined", {"facts": ("periodic_x",)}, "forbidden_fact:periodic_x"),
    ("twod_combined", {"facts": ("dynamic_interface_occupancy",)}, "forbidden_fact:dynamic_interface_occupancy"),
    ("twod_combined", {"facts": ("dual_ions",)}, "forbidden_fact:dual_ions"),
])
def test_narrow_scope_conditions_reject_without_promoting_metadata(name, change, issue, registry, inputs):
    rule = narrow_rules()[name]
    baseline = CapabilityContext(rule.dimensions[0], rule.modes[0],
                                rule.backends[0] if rule.backends else "metadata_only",
                                rule.statistics[0] if rule.statistics else "maxwell_boltzmann",
                                facts=rule.required_facts, resources=rule.required_resources,
                                temperature_K=rule.temperature_K)
    assert rule.problems(baseline, ()) == ()  # Necessary declarations only.
    assert issue in rule.problems(replace(baseline, **change), ())
    experiment = replace(jv_definition(), capability=rule, required_models=())
    metadata_only = select_models(registry, inputs[0], baseline, {}, {})
    result = ExperimentRegistry((experiment,)).preflight("jv", 1, metadata_only, registry)
    assert result.declared_compatible
    assert not result.can_execute


def test_experiment_source_names_exist_without_importing_them():
    definition = jv_definition()
    for item in (definition.parameter_schema, definition.result_schema,
                 definition.protocol_builder, definition.executor_source):
        path, symbol = item.id.split("#")
        names = {node.name for node in ast.parse((ROOT / path).read_text()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        assert symbol in names
    path, symbol = definition.renderer_source.id.split("#")
    assert f"export function {symbol}(" in (ROOT / path).read_text()


def test_experiment_duplicate_missing_models_and_identity_mismatch(registry, inputs):
    experiment = jv_definition()
    with pytest.raises(ValueError, match="duplicate experiment"):
        ExperimentRegistry((experiment, experiment))
    experiments = ExperimentRegistry((experiment,))
    with pytest.raises(ValueError, match="unknown experiment"):
        experiments.get("jv", 2)
    empty = select_models(registry, inputs[0], context(), {}, {})
    assert experiments.preflight("jv", 1, empty, registry).declared_problems == (
        "experiment:missing_model:transport.constant/1",
    )
    changed = ModelRegistry(tuple(replace(item, label="changed") for item in registry.definitions))
    with pytest.raises(ValueError, match="identity mismatch"):
        experiments.preflight("jv", 1, empty, changed)
    with pytest.raises(TypeError):
        replace(experiment, executor="pretend.available.runner")
    assert metadata_value(experiment)["executor_source"]["sha256"]


@pytest.mark.parametrize("change", [
    "removed_dependency", "reversed_order", "missing_parameters", "duplicate_ownership",
    "wrong_owned_spec", "wrong_scope", "unconsumed_parameters", "missing_read",
    "exclusive_family", "stale_diagnostics",
])
def test_preflight_revalidates_edited_selection(registry, inputs, change):
    selected = selection(registry, inputs, overrides={LEFT: {"recombination": (srh_choice(),)}})
    items = list(selected.instances)
    left = next(i for i, item in enumerate(items) if item.id == "layer/left/carrier")
    right = next(i for i, item in enumerate(items) if item.id == "layer/right/carrier")
    if change == "removed_dependency":
        removed = items.pop(left)
        edited = replace(selected, instances=tuple(items), evaluation_order=tuple(
            id for id in selected.evaluation_order if id != removed.id
        ))
    elif change == "reversed_order":
        edited = replace(selected, evaluation_order=tuple(reversed(selected.evaluation_order)))
    elif change == "stale_diagnostics":
        edited = replace(selected, context=replace(selected.context, statistics="fermi_dirac"))
    else:
        if change == "missing_parameters":
            items[left] = replace(items[left], parameters=())
        elif change == "duplicate_ownership":
            items[right] = replace(items[right], owned_variables=items[left].owned_variables)
        elif change == "wrong_owned_spec":
            owned = list(items[left].owned_variables)
            owned[0] = (owned[0][0], replace(owned[0][1], unit="V"))
            items[left] = replace(items[left], owned_variables=tuple(owned))
        elif change == "wrong_scope":
            items[left] = replace(items[left], scope=Scope("device"))
        elif change == "unconsumed_parameters":
            items[left] = replace(items[left], parameters=(*items[left].parameters, ("D_ion", 0.0)))
        elif change == "missing_read":
            items[left] = replace(items[left], read_variables=())
        elif change == "exclusive_family":
            extra = replace(items[left], local_id="another", owned_variables=tuple(
                (target + ":other", variable) for target, variable in items[left].owned_variables
            ))
            items.append(extra)
        # Preserve the original relative order where possible; the validators
        # must independently check the edited declaration, not merely its IDs.
        order = tuple(item.id for item in items)
        edited = replace(selected, instances=tuple(items), evaluation_order=order)
    with pytest.raises(ValueError):
        validate_selection(edited, registry)
    with pytest.raises(ValueError):
        ExperimentRegistry((jv_definition(),)).preflight("jv", 1, edited, registry)


def test_simultaneous_state_reads_do_not_create_implicit_dependency_cycles(registry, inputs):
    base = replace(registry.get(TRANSPORT), parameters=(), owns=(), reads=(), requires=())
    a = replace(base, ref=ModelRef("coupled_a", 1), family="a", reads=("b",),
                owns=(OwnedVariable("a", "1", "node", ("node",)),))
    b = replace(base, ref=ModelRef("coupled_b", 1), family="b", reads=("a",),
                owns=(OwnedVariable("b", "1", "node", ("node",)),))
    coupled = ModelRegistry((a, b))
    choices = (ModelChoice("a", a.ref, field_bindings=(("a", "state_a"), ("b", "state_b"))),
               ModelChoice("b", b.ref, field_bindings=(("a", "state_a"), ("b", "state_b"))))
    result = select_models(coupled, Topology(("left",)), context(), {"layer": choices}, {})
    assert result.evaluation_order == ("layer/left/a", "layer/left/b")
    assert validate_selection(result, coupled) == ()
