"""Actual public Layout and per-point preparation; all numerical reuse disabled."""
from dataclasses import FrozenInstanceError, replace
import json
import random
import struct

import pytest

from solarlab.config.identity import (
    CACHE_LAYERS, CacheLayerIdentity, ExecutionEngine, cold_point_binding, execution_manifest,
)
from solarlab.config.resolve_device import resolve_device
from solarlab.contracts.layout import EquationSpec, Layout, Support, VariableSpec
from solarlab.contracts.units import ONE
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument
from solarlab.sweeps.inputs import SweepInput
from solarlab.sweeps.preparation import prepare_point, prepare_sweep
from test_device_configuration_preparation import resources as resources
from test_execution_identity_preparation import (
    changed, changed_resource, device as device, execution as execution, identity_inputs,
)
from test_execution_manifest import binding, prepared as prepared
from test_jv_protocol_preparation import defaults as defaults
from test_sweep_preparation import single


@pytest.fixture
def layout():
    # This is a real checked Layout, but a labelled two-word metadata fixture,
    # not a claim that the actual device's equations have been compiled.
    return Layout((Support("fixture", "global", (2,)),),
                  (VariableSpec("fixture_state", "fixture_owner", "fixture", (2,), ONE),),
                  (EquationSpec("fixture_equation", "fixture_owner", "fixture", (2,), ONE,
                                derivative_support=("fixture_state",)),))


@pytest.fixture
def plan():
    return SourceDocument("fixture:uncompiled_plan", b'{"scope":"metadata_only","connectivity":[[0,1]]}')


def bound(inputs, execution, layout, plan, *, point="fixture_point", matrix=None):
    engine = ExecutionEngine("solarlab")
    manifest = execution_manifest(inputs, execution, engine=engine)
    return cold_point_binding(inputs, execution, manifest, point_id=point, layout=layout, topology_plan=plan,
        engine=engine, environment=execution.environment, expected_sha256=manifest.sha256, numeric_matrix=matrix)


def test_four_domains_are_separate_immutable_and_never_enable_lu_or_rebind(device, execution, layout, plan):
    inputs = identity_inputs(device)
    matrix = SourceDocument("fixture:matrix_words", struct.pack("<4d", 1.0, -0.0, 0.0, 2.0))
    first, second = (bound(inputs, execution, layout, plan, matrix=matrix) for _ in range(2))
    assert first == second and first is not second
    assert tuple(item.layer for item in first.layers) == CACHE_LAYERS
    assert len({item.sha256 for item in first.layers}) == 4
    assert all(left is not right and not left.cache_eligible for left, right in zip(first.layers, second.layers))
    assert not first.can_execute and not first.numerical_reuse
    with pytest.raises(FrozenInstanceError):
        first.point_id = "other"
    detached = first.export()
    detached["layers"].clear()
    assert len(first.export()["layers"]) == 4
    with pytest.raises(ValueError, match="four distinct"):
        replace(first, layers=first.layers[::-1])


def test_unconstructed_factorization_has_no_key(device, execution, layout, plan):
    value = bound(identity_inputs(device), execution, layout, plan)
    factor = value.layer("numeric_factorization")
    assert factor.sha256 is None and factor.canonical_json is None and not factor.cache_eligible
    with pytest.raises(ValueError, match="unconstructed"):
        CacheLayerIdentity("topology_plan", None)
    with pytest.raises(ValueError, match="unknown"):
        value.layer("result_cache_by_physics")


def test_layer_labels_cannot_be_interchanged(device, execution, layout, plan):
    value = bound(identity_inputs(device), execution, layout, plan)
    with pytest.raises(ValueError, match="canonical domain"):
        CacheLayerIdentity("numeric_parameter_binding", value.layer("topology_plan").canonical_json)
    with pytest.raises(TypeError):
        CacheLayerIdentity("numeric_factorization", None, cache_eligible=True)


@pytest.mark.parametrize("field,value", [
    ("thickness", "650 nm"), ("T", "301 K"), ("eps_r", 31.0),
    ("mu_n", "3 cm^2/Vs"), ("trap_N_t_bulk", "1e19 m^-3"), ("D_ion", 0.0),
])
def test_changed_physical_parameters_get_new_numeric_and_state_keys(device, execution, layout, plan, field, value):
    def edit(raw):
        if field == "T":
            raw["settings"][field] = value
        elif field == "thickness":
            raw["layers"][1][field] = value
        else:
            raw["layers"][1]["parameters"][field] = value
    other = changed(device, edit)
    original_bytes = other.input_json
    first = bound(identity_inputs(device), execution, layout, plan)
    second = bound(identity_inputs(other), execution, layout, plan)
    assert first.layer("topology_plan").sha256 == second.layer("topology_plan").sha256
    assert first.layer("numeric_parameter_binding").sha256 != second.layer("numeric_parameter_binding").sha256
    assert first.layer("working_point_initial_state").sha256 != second.layer("working_point_initial_state").sha256
    assert other.input_json == original_bytes and not second.numerical_reuse


def test_equivalent_mobility_units_keep_parameter_keys_but_still_cold_build(device, execution, layout, plan):
    other = changed(device, lambda raw: raw["layers"][1]["parameters"].update(mu_n="2 cm^2/Vs"))
    assert dict(device.layers[1].parameters)["mu_n"] == dict(other.layers[1].parameters)["mu_n"]
    first, second = (bound(identity_inputs(item), execution, layout, plan) for item in (device, other))
    assert first.layer("numeric_parameter_binding").sha256 == second.layer("numeric_parameter_binding").sha256
    assert first is not second and not first.numerical_reuse and not second.numerical_reuse


def test_changed_used_optical_table_cannot_reuse_numeric_binding(device, execution, layout, plan):
    resource = device.resources.get("MAPbI3", "nk")
    other = resolve_device(device.to_input(), device.defaults, ResourceLibrary((changed_resource(resource),)), sources=device.sources)
    first, second = (bound(identity_inputs(item), execution, layout, plan) for item in (device, other))
    assert first.input_sha256 == second.input_sha256 and first.physics_sha256 != second.physics_sha256
    assert first.layer("numeric_parameter_binding").sha256 != second.layer("numeric_parameter_binding").sha256
    assert resource.source.content == device.resources.get("MAPbI3", "nk").source.content


@pytest.mark.parametrize("role", ["protocol", "preparation", "initial_state", "numeric_controls"])
def test_history_state_and_controls_do_not_alias_parameter_or_result_identity(device, execution, layout, plan, role):
    source = getattr(execution, role)
    other = replace(execution, **{role: replace(source, content=source.content+b"\n")})
    first, second = (bound(identity_inputs(device), description, layout, plan) for description in (execution, other))
    assert first.physics_sha256 == second.physics_sha256
    assert first.layer("topology_plan").sha256 == second.layer("topology_plan").sha256
    assert first.layer("numeric_parameter_binding").sha256 == second.layer("numeric_parameter_binding").sha256
    assert first.layer("working_point_initial_state").sha256 != second.layer("working_point_initial_state").sha256
    assert first.execution_sha256 != second.execution_sha256


def test_actual_grid_is_separate_from_structural_layout(device, execution, layout, plan):
    other = replace(execution, grid=replace(execution.grid, content=execution.grid.content+b"\n"))
    first, second = (bound(identity_inputs(device), item, layout, plan) for item in (execution, other))
    assert first.layer("topology_plan").sha256 == second.layer("topology_plan").sha256
    assert first.layer("numeric_parameter_binding").sha256 != second.layer("numeric_parameter_binding").sha256


def test_support_plan_and_matrix_words_are_independently_bound(device, execution, layout, plan):
    inputs = identity_inputs(device)
    positive = SourceDocument("fixture:matrix", struct.pack("<d", 0.0))
    negative = SourceDocument("fixture:matrix", struct.pack("<d", -0.0))
    first, second = (bound(inputs, execution, layout, plan, matrix=value) for value in (positive, negative))
    assert first.layers[:3] == second.layers[:3]
    assert first.layer("numeric_factorization").sha256 != second.layer("numeric_factorization").sha256
    # A different compiler support plan (including future WKB support changes)
    # cannot silently reuse a structure key. This grants no WKB qualification.
    changed_plan = replace(plan, content=b'{"scope":"metadata_only","nonlocal_support":[[0,1]]}')
    other = bound(inputs, execution, layout, changed_plan)
    assert first.layer("topology_plan").sha256 != other.layer("topology_plan").sha256
    broken = replace(layout)
    object.__setattr__(broken, "identity", "0"*64)
    with pytest.raises(ValueError, match="actual declarations"):
        bound(inputs, execution, broken, plan)
    with pytest.raises(ValueError, match="Layout"):
        bound(inputs, execution, object(), plan)


def test_real_random_order_sweep_fresh_binding_keeps_all_numerical_words(prepared, execution, layout, plan, record_property):
    raw = single(prepared.to_input(), family="jv", owner_id=None, parameter=("v_rate",), values=("20 mV/s", "40 mV/s", "60 mV/s"))
    sweep = prepare_sweep(SweepInput.model_validate(raw), prepared.defaults, prepared.resources, sources=prepared.sources)
    before = sweep._resolved_json
    records = {}
    for point in sweep.to_mapping()["points"]:
        current = prepare_point(point["applied_input"], prepared.defaults, prepared.resources, prepared.sources)
        records[point["id"]] = current, binding(current, execution)
    random_order = list(records)
    random.Random(2901).shuffle(random_order)
    snapshots, payloads, instances = {}, [], []
    for point_id in [*records, *random_order]:
        direct, (inputs, description, engine, manifest) = records[point_id]
        current, actual = sweep.prepare_point_binding(point_id, manifest, inputs, description, layout=layout,
            topology_plan=plan, engine=engine, environment=description.environment, expected_sha256=manifest.sha256)
        assert current is not direct and current.input_json == direct.input_json and current._resolved_json == direct._resolved_json
        assert actual is not snapshots.get(point_id)
        if point_id in snapshots:
            assert actual == snapshots[point_id]
        snapshots[point_id] = actual
        instances.append(actual)
        payloads.append(dict(point=point_id, prepared_sha256=current.content_sha256, binding_sha256=actual.content_sha256,
                             exact_bytes_equal=True))
    assert len({id(value) for value in instances}) == 6
    assert len({value.layer("topology_plan").sha256 for value in instances}) == 1
    assert len({value.layer("numeric_parameter_binding").sha256 for value in instances}) == 1
    assert len({value.layer("working_point_initial_state").sha256 for value in instances}) == 3
    assert all(value.layer("numeric_factorization").sha256 is None and not value.numerical_reuse for value in instances)
    assert sweep._resolved_json == before
    record_property("binding_payloads", json.dumps(payloads))
