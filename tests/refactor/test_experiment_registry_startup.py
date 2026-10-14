"""Real registered data preparation; no numerical executor or fake success."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from solarlab.config.model_parameters import device_topology
from solarlab.config.resolve_device import resolve_device
from solarlab.experiments.registry import ExperimentDefinition, ExperimentRegistry
from solarlab.experiments.jv.inputs import JVExperimentInput
from solarlab.experiments.two_dimensional.inputs import SpatialExperimentInput
from solarlab.physics.registry import CapabilityContext, CapabilityRule, EvidenceRef, ModelRegistry
from solarlab.physics.selection import select_models
from test_device_configuration_preparation import resources as resources
import test_jv_protocol_preparation as jv
import test_spatial_protocol_preparation as spatial

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=("jv", "dark_jv", "jv_2d", "voc_grain_sweep"))
def bound_case(request, resources):
    kind = request.param
    one_dimensional = kind in ("jv", "dark_jv")
    helper = jv if one_dimensional else spatial
    defaults = helper.defaults.__wrapped__()
    raw, input_sources = helper.request(defaults, V_max=1.2)
    raw["experiment"]["kind"] = kind
    if kind == "voc_grain_sweep":
        raw["experiment"] = {"kind": kind, "grain_sizes_nm": [500, 1000]}
    expected = helper.prepare(raw, defaults, resources, input_sources)
    dto = JVExperimentInput if one_dimensional else SpatialExperimentInput
    relative = "src/solarlab/experiments/" + ("jv" if one_dimensional else "two_dimensional") + "/preparation.py"
    builder = "prepare_jv_experiment" if one_dimensional else "prepare_spatial_experiment"
    result_name = "PreparedJVExperiment" if one_dimensional else "PreparedSpatialExperiment"
    sources = {}

    def ref(path, symbol=""):
        sources[path] = (ROOT / path).read_bytes()
        return EvidenceRef(path + ("#" + symbol if symbol else ""), hashlib.sha256(sources[path]).hexdigest(), "source")

    schema_id = "schema:" + kind + ".validation.v1"
    schema_bytes = json.dumps(dto.model_json_schema(mode="validation"), sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()
    sources[schema_id] = schema_bytes
    parameter_schema = EvidenceRef(schema_id, hashlib.sha256(schema_bytes).hexdigest(), "source")
    builder_ref, result_ref = ref(relative, builder), ref(relative, result_name)
    executor_path, executor_name, renderer_name = {
        "jv": ("experiments/jv_sweep.py", "run_jv_sweep", "renderJV"),
        "dark_jv": ("experiments/dark_jv.py", "run_dark_jv", "renderDarkJV"),
        "jv_2d": ("twod/experiments/jv_sweep_2d.py", "run_jv_sweep_2d", "renderJV2D"),
        "voc_grain_sweep": ("twod/experiments/voc_grain_sweep.py", "run_voc_grain_sweep", "renderVocGrainSweep"),
    }[kind]
    executor = "perovskite-sim/perovskite_sim/" + executor_path
    renderer = ref("web/src/workstation/panes/main-plot-pane.ts", renderer_name)
    dimension = 1 if one_dimensional else 2
    rule = CapabilityRule(kind + ".preparation", (dimension,), ("transient",),
                          (builder_ref,), backends=("metadata_only",))
    definition = ExperimentDefinition(kind, 1, kind, parameter_schema, result_ref, builder_ref,
        ref(executor, executor_name),
        kind, renderer, rule)
    registry = ExperimentRegistry((definition,))
    models = ModelRegistry(())
    device = resolve_device(dto.model_validate(raw).device, defaults, resources, sources=input_sources)
    selection = select_models(models, device_topology(device),
        CapabilityContext(dimension, "transient", "metadata_only", "maxwell_boltzmann"), {}, {})
    kwargs = dict(models=models, selection=selection, startup_sources=tuple(sources.items()),
                  expected_sha256=registry.content_sha256, expected_model_sha256=models.content_sha256,
                  sources=input_sources)
    return registry, raw, defaults, resources, kwargs, expected


def invoke(case, *, registry=None, raw=None, **changes):
    old, request, defaults, resources, kwargs, _ = case
    return (registry or old).prepare(request["experiment"]["kind"], 1, request if raw is None else raw,
                                    defaults, resources, **{**kwargs, **changes})


def test_actual_registered_preparation_matches_complete_public_result(bound_case):
    result = invoke(bound_case)
    expected = bound_case[-1]
    assert type(result) is type(expected)
    assert result.input_json == expected.input_json
    assert result._resolved_json == expected._resolved_json
    assert result.content_sha256 == expected.content_sha256
    assert not result.can_execute
    assert result.to_mapping()["identity"]["execution_identity"] is None


@pytest.mark.parametrize("field", ("expected_sha256", "expected_model_sha256"))
def test_startup_rejects_changed_registry_identities(bound_case, field):
    with pytest.raises(ValueError, match="frozen identity"):
        invoke(bound_case, **{field: "0" * 64})


@pytest.mark.parametrize("role", ("parameter_schema", "result_schema", "executor_source", "renderer_source"))
def test_startup_consumes_every_actual_source_role(bound_case, role):
    registry, _, _, _, kwargs, _ = bound_case
    id = getattr(registry.definitions[0], role).id.partition("#")[0]
    missing = tuple(row for row in kwargs["startup_sources"] if row[0] != id)
    with pytest.raises(ValueError, match="missing evidence"):
        invoke(bound_case, startup_sources=missing)
    tampered = tuple((key, value + b"\n" if key == id else value) for key, value in kwargs["startup_sources"])
    with pytest.raises(ValueError, match="content mismatch"):
        invoke(bound_case, startup_sources=tampered)


def test_frozen_but_incorrect_schema_or_result_contract_cannot_reach_preparer(bound_case):
    old, _, _, _, kwargs, _ = bound_case
    definition = old.definitions[0]
    invalid_schema = b'{"type":"object","additionalProperties":true}'
    reference = replace(definition.parameter_schema, sha256=hashlib.sha256(invalid_schema).hexdigest())
    registry = ExperimentRegistry((replace(definition, parameter_schema=reference),))
    documents = tuple((key, invalid_schema if key == reference.id else value)
                      for key, value in kwargs["startup_sources"])
    with pytest.raises(ValueError, match="actual preparation input schema"):
        invoke(bound_case, registry=registry, expected_sha256=registry.content_sha256, startup_sources=documents)
    reference = replace(definition.result_schema, id=definition.result_schema.id.partition("#")[0] + "#WrongResult")
    registry = ExperimentRegistry((replace(definition, result_schema=reference),))
    with pytest.raises(ValueError, match="entry symbol"):
        invoke(bound_case, registry=registry, expected_sha256=registry.content_sha256)


@pytest.mark.parametrize("role", ("executor_source", "renderer_source"))
def test_frozen_source_must_contain_the_declared_entry(bound_case, role):
    old = bound_case[0]
    definition = old.definitions[0]
    reference = getattr(definition, role)
    changed = replace(reference, id=reference.id.partition("#")[0] + "#missing_entry")
    registry = ExperimentRegistry((replace(definition, **{role: changed}),))
    with pytest.raises(ValueError, match="symbol is absent"):
        invoke(bound_case, registry=registry, expected_sha256=registry.content_sha256)


def test_legacy_executor_is_never_a_preparation_fallback(bound_case):
    old = bound_case[0]
    definition = old.definitions[0]
    registry = ExperimentRegistry((replace(definition, protocol_builder=definition.executor_source),))
    with pytest.raises(ValueError, match="supported public preparation entry"):
        invoke(bound_case, registry=registry, expected_sha256=registry.content_sha256)


def test_real_capability_and_topology_checks_still_apply(bound_case):
    old, raw, _, _, kwargs, _ = bound_case
    selection = kwargs["selection"]
    context = replace(selection.context, mode="ac")
    selected = select_models(kwargs["models"], selection.topology, context, {}, {})
    with pytest.raises(ValueError, match="capability refused"):
        invoke(bound_case, selection=selected)
    changed = deepcopy(raw)
    changed["device"]["id"] = "another_device"
    with pytest.raises(ValueError, match="actual prepared device"):
        invoke(bound_case, raw=changed)
    context = replace(selection.context, temperature_K=301)
    selected = select_models(kwargs["models"], selection.topology, context, {}, {})
    with pytest.raises(ValueError, match="temperature context"):
        invoke(bound_case, selection=selected)
    context = replace(selection.context, statistics="fermi_dirac")
    selected = select_models(kwargs["models"], selection.topology, context, {}, {})
    with pytest.raises(ValueError, match="statistics context"):
        invoke(bound_case, selection=selected)


def test_registered_preparation_imports_no_old_engine(bound_case, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(("perovskite_sim", "backend", "scipy.integrate")), name
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert not invoke(bound_case).can_execute
