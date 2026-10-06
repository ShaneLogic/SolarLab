"""Identity preparation through actual typed device/selection/resource APIs.

Only configuration data and source bytes are read. The execution documents are
explicit non-executed provenance fixtures, not a solver or a claimed DAE seed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import struct
import subprocess
import sys

import pytest

from solarlab.config.device_import import import_standard_device
from solarlab.config.identity import (
    BuildSource, CACHE_LAYERS, ExecutionDescription, IdentityInputs, ModelInputs,
    RuntimeEnvironment, execution_identity, input_identity, physics_identity,
)
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.model_parameters import device_topology, layer_parameter_source, structured_parameter_source
from solarlab.config.resolve_device import resolve_device
from solarlab.device.inputs import DeviceInput
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import CapabilityContext, CapabilityRule, EvidenceRef, ModelDefinition, ModelRef, ModelRegistry, Scope
from solarlab.physics.selection import ModelChoice, select_models
from solarlab.physics.semantic_projection import SemanticProjection

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "perovskite-sim/perovskite_sim"
CATALOG_PATHS = ("models/parameters.py", "models/device.py", "models/defects.py", "physics/statistics.py",
                 "physics/band_gap_narrowing.py", "scaps_compat/loader.py", "constants.py",
                 "twod/microstructure.py", "models/tandem_config.py")


@pytest.fixture(scope="module")
def device():
    defaults = read_legacy_default_catalog(tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in CATALOG_PATHS))
    original = SourceDocument("fixture:nip_MAPbI3", (ROOT / "perovskite-sim/tests/fixtures/configs/nip_MAPbI3.yaml").read_bytes())
    table = ResourceTable("MAPbI3", "nk", SourceDocument("nk:MAPbI3", (LEGACY / "data/nk/MAPbI3.csv").read_bytes()))
    data = import_standard_device(original, id="identity_device", defaults=defaults).editing_data()
    data["settings"].update(mode="full", T="300 K")
    # Explicit data-only fixture edit binds one real optical table. No optical
    # calculation or claim of equality to the old Beer-Lambert run is made.
    data["layers"][1]["parameters"].update(optical_material=table.name, n_optical=None)
    return resolve_device(DeviceInput.model_validate(data), defaults, ResourceLibrary((table,)), sources=(original,))


def changed(device, edit):
    data = deepcopy(device.to_input().editing_data())
    edit(data)
    return resolve_device(DeviceInput.model_validate(data), device.defaults, device.resources, sources=device.sources)


def identity_inputs(device, *, version=1, label="Temperature-dependent mobility", rule="all_values_v1", overrides=None,
                    local_id="mobility", parameters=("mu_n", "mu_p", "mu_T_gamma")):
    source = SourceDocument("physics/temperature.py", (LEGACY / "physics/temperature.py").read_bytes())
    ref = ModelRef("transport.temperature_mobility", version)
    capability = CapabilityRule("source_data_only", (1,), ("dc", "transient"),
                               (EvidenceRef(source.id + "#mu_at_T", source.sha256, "source"),),
                               backends=("metadata_only",))
    definition = ModelDefinition(ref, label, "transport", ("layer",), "node", parameters, (), (), capability,
                                 parameter_schema="full_layer")
    registry = ModelRegistry((definition,))
    defaults = {"layer": (ModelChoice(local_id, ref, parameter_schema="full_layer"),)}
    overrides = {} if overrides is None else overrides
    sources = {Scope("layer", (layer.id,)): layer_parameter_source(layer) for layer in device.layers}
    context = CapabilityContext(1, "transient", "metadata_only", "maxwell_boltzmann", temperature_K=device.values["settings"]["T"])
    selection = select_models(registry, device_topology(device), context, defaults, sources, overrides)
    evidence = ()
    if rule == "without_display_label_v1":
        evidence = tuple(SourceDocument(name, (ROOT / name).read_bytes()) for name in
                         ("src/solarlab/physics/registry.py", "src/solarlab/physics/selection.py"))
    projection = SemanticProjection(ref, rule, (source,), evidence)
    return IdentityInputs(device, selection, registry, ModelInputs.from_mappings(defaults=defaults, overrides=overrides), (projection,))


@pytest.fixture(scope="module")
def execution():
    commit = subprocess.run(["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True, timeout=2).stdout.strip()
    # The declared implementation snapshot comes from immutable Git objects,
    # not an uncommitted producer file relabelled as that commit's content.
    name = "perovskite-sim/perovskite_sim/physics/temperature.py"
    source = subprocess.run(["git", "show", commit + ":" + name], cwd=ROOT,
                            capture_output=True, check=True, timeout=2).stdout
    implementation = BuildSource(commit, (SourceDocument(name, source),))
    environment = RuntimeEnvironment(platform.python_implementation(), platform.python_version(), platform.system(), platform.machine(),
                                     tuple((name, importlib.metadata.version(name)) for name in ("numpy", "pydantic")),
                                     (("scope", "nonexecuted_identity_fixture"),))
    def document(name, value):
        return SourceDocument(name, json.dumps(value, sort_keys=True).encode())
    return ExecutionDescription(
        document("protocol.fixture.v1", {"schema": "identity-protocol-fixture.v1", "segments": [{"start_s": 0, "end_s": 1, "voltage_V": 0.2}], "events": []}),
        document("preparation.fixture.v1", {"schema": "identity-preparation-fixture.v1", "method": "provided_data_only", "history": []}),
        document("initial_state.fixture.v1", {"schema": "identity-state-fixture.v1", "qualification": "not_a_DAE_seed", "dtype": "<f8", "shape": [2], "fields": ["example_n", "example_p"], "units": ["m^-3", "m^-3"], "words": struct.pack("<2d", 1.0, -0.0).hex()}),
        document("grid.fixture.v1", {"schema": "identity-grid-fixture.v1", "coordinate_unit": "m", "nodes": [0, 1e-7, 7e-7]}),
        document("controls.fixture.v1", {"schema": "identity-controls-fixture.v1", "absolute": 1e-8, "relative": 1e-6}),
        implementation, environment, commit,
    )


def hashes(inputs, execution):
    return input_identity(inputs).sha256, physics_identity(inputs).sha256, execution_identity(inputs, execution).sha256


def test_real_prepared_device_selection_resource_and_execution_producers(device, execution):
    inputs = identity_inputs(device)
    result = [input_identity(inputs), physics_identity(inputs), execution_identity(inputs, execution)]
    assert len(device.layers) == 3 and len(inputs.selection.instances) == 3
    physical = result[1].export()["document"]["payload"]
    assert physical["used_resources"][0]["name"] == "MAPbI3"
    assert physical["used_resources"][0]["content_sha256"] == device.resources.get("MAPbI3", "nk").content_sha256
    for item in result:
        assert hashlib.sha256(item.canonical_json).hexdigest() == item.sha256
        assert item.status == "prepared_pending_dependencies" and not item.cache_eligible
    assert len(set(item.sha256 for item in result)) == 3
    assert CACHE_LAYERS == ("topology_plan", "numeric_parameter_binding", "working_point_initial_state", "numeric_factorization")


def test_key_order_unit_equivalence_and_normalized_signed_zero(device, execution):
    a = changed(device, lambda data: data["settings"].update(phi_left="-0 V"))
    def equivalent(data):
        data["settings"]["phi_left"] = 0.0
        data["layers"][1]["thickness"] = "400 nm"
        data["layers"][1]["parameters"]["mu_n"] = "2 cm^2/Vs"
        data["layers"][1]["parameters"] = dict(reversed(list(data["layers"][1]["parameters"].items())))
    b = changed(a, equivalent)
    assert hashes(identity_inputs(a), execution) == hashes(identity_inputs(b), execution)


def test_omitted_versus_explicit_default_and_null_preserve_input_presence(device):
    a = changed(device, lambda data: data["settings"].pop("phi_left", None))
    b = changed(a, lambda data: data["settings"].update(phi_left=device.values["settings"]["phi_left"]))
    ia, ib = identity_inputs(a), identity_inputs(b)
    assert input_identity(ia).sha256 != input_identity(ib).sha256
    assert physics_identity(ia).sha256 == physics_identity(ib).sha256
    c = changed(device, lambda data: data["layers"][0]["parameters"].pop("n_optical", None))
    d = changed(c, lambda data: data["layers"][0]["parameters"].update(n_optical=None))
    assert input_identity(identity_inputs(c)).sha256 != input_identity(identity_inputs(d)).sha256
    assert physics_identity(identity_inputs(c)).sha256 == physics_identity(identity_inputs(d)).sha256


def test_original_family_override_and_clear_are_not_inferred_from_selection(device):
    implicit = identity_inputs(device)
    scope = Scope("layer", (device.layers[0].id,))
    choice = implicit.model_inputs.defaults[0][1][0]
    explicit = identity_inputs(device, overrides={scope: {"transport": (choice,)}})
    cleared = identity_inputs(device, overrides={scope: {"transport": ()}})
    assert physics_identity(implicit).sha256 == physics_identity(explicit).sha256
    assert len({input_identity(item).sha256 for item in (implicit, explicit, cleared)}) == 3
    assert physics_identity(cleared).sha256 != physics_identity(implicit).sha256
    # Identical selections cannot attest historical input presence. The caller
    # must supply that provenance; replay rejects only contradictory bindings.
    with pytest.raises(ValueError, match="original model inputs"):
        replace(explicit, model_inputs=cleared.model_inputs)


def test_documentary_default_projection_is_source_supported_over_the_whole_model_domain(device):
    for temperature in (300.0, 315.0):
        prepared = changed(device, lambda data: data["settings"].update(T=temperature))
        old = identity_inputs(prepared, label="Old default display label", rule="without_display_label_v1")
        new = identity_inputs(prepared, label="New inert display label", rule="without_display_label_v1")
        assert old.selection.instances == new.selection.instances
        assert input_identity(old).sha256 == input_identity(new).sha256
        assert physics_identity(old).sha256 == physics_identity(new).sha256
    # The full-value rule is intentionally conservative, without this waiver.
    assert physics_identity(identity_inputs(device, label="one")).sha256 != physics_identity(identity_inputs(device, label="two")).sha256
    proof = old.projections[0]
    with pytest.raises(ValueError, match="exact audited"):
        replace(proof, projection_evidence=())
    with pytest.raises(ValueError, match="exact audited"):
        replace(proof, projection_evidence=(replace(proof.projection_evidence[0], content=proof.projection_evidence[0].content + b"\n"), proof.projection_evidence[1]))


def test_reference_temperature_does_not_erase_the_mobility_law_or_temperature_derivative(device):
    base = identity_inputs(device)
    modified = replace(device.defaults, material=tuple((name, -2.5 if name == "mu_T_gamma" else value) for name, value in device.defaults.material))
    other = resolve_device(device.to_input(), modified, device.resources, sources=device.sources)
    comparison = identity_inputs(other)
    assert device.values["settings"]["T"] == other.values["settings"]["T"] == 300.0
    assert input_identity(base).sha256 == input_identity(comparison).sha256
    assert physics_identity(base).sha256 != physics_identity(comparison).sha256
    # Independent exact derivative of mu0*(T/Tref)^gamma at Tref. No old
    # function is imported and equal point values cannot erase this difference.
    mu0 = Fraction.from_float(dict(device.layers[1].parameters)["mu_n"])
    assert mu0 * Fraction(-3, 2) / 300 != mu0 * Fraction(-5, 2) / 300
    with pytest.raises(ValueError, match="unknown model-owned"):
        replace(base.projections[0], rule="mu_at_reference_v1")


def test_changed_effective_default_is_bound_even_when_original_input_is_identical(device):
    other_defaults = replace(device.defaults, material=tuple((name, 0.1 if name == "mu_T_gamma" else value) for name, value in device.defaults.material))
    other = resolve_device(device.to_input(), other_defaults, device.resources, sources=device.sources)
    assert input_identity(identity_inputs(device)).sha256 == input_identity(identity_inputs(other)).sha256
    assert physics_identity(identity_inputs(device)).sha256 != physics_identity(identity_inputs(other)).sha256


def test_model_semantic_version_instance_id_and_other_effective_parameters_are_bound(device):
    base = identity_inputs(device)
    assert physics_identity(identity_inputs(device, version=2)).sha256 != physics_identity(base).sha256
    assert physics_identity(identity_inputs(device, local_id="different_instance")).sha256 != physics_identity(base).sha256
    changed_doping = changed(device, lambda data: data["layers"][1]["parameters"].update(N_D="1e15 cm^-3"))
    assert physics_identity(identity_inputs(changed_doping)).sha256 != physics_identity(base).sha256
    # A parameter not consumed by this selected model remains in the device
    # portion instead of being dropped by a selection-only payload hash.
    body = physics_identity(identity_inputs(changed_doping)).export()["document"]["payload"]
    assert body["layers"][1]["remaining_effective_parameters"]["N_D"] == 1e21


def changed_resource(table):
    rows = table.source.content.decode().splitlines()
    for index, row in enumerate(rows):
        if row.strip() and not row.startswith("#") and not row.startswith("wavelength"):
            values = row.split(",")
            values[1] = str(Decimal(values[1]) + Decimal("0.125"))
            rows[index] = ",".join(values)
            break
    return ResourceTable(table.name, table.kind, replace(table.source, content=("\n".join(rows) + "\n").encode()))


def test_same_name_used_resource_replacement_changes_physics_but_unused_table_does_not(device):
    table = device.resources.get("MAPbI3", "nk")
    replaced = resolve_device(device.to_input(), device.defaults, ResourceLibrary((changed_resource(table),)), sources=device.sources)
    assert input_identity(identity_inputs(replaced)).sha256 == input_identity(identity_inputs(device)).sha256
    assert physics_identity(identity_inputs(replaced)).sha256 != physics_identity(identity_inputs(device)).sha256
    unused = ResourceTable("not_selected", "nk", table.source)
    extra = resolve_device(device.to_input(), device.defaults, ResourceLibrary((table, changed_resource(unused))), sources=device.sources)
    assert physics_identity(identity_inputs(extra)).sha256 == physics_identity(identity_inputs(device)).sha256


def test_actual_structured_contact_parameters_are_bound_without_raw_device_hash_leakage(device):
    source = structured_parameter_source(device, "contact", device.to_mapping()["contacts"][0]["id"])
    schema = source.structured.data.schema
    ref = ModelRef("parameters.contact", schema.version)
    definition = ModelDefinition(ref, "Contact data", "contact", ("contact",), "port", schema.parameter_names, (), (),
                                 CapabilityRule("contact_data_only", (1,), ("dc",), (schema.source,)),
                                 parameter_schema=schema.id, structured_schema=schema)
    registry = ModelRegistry((definition,))
    scope = source.structured.data.scope
    overrides = {scope: {"contact": (ModelChoice(source.structured.data.local_id, ref, parameter_schema=schema.id),)}}
    context = CapabilityContext(1, "dc", "metadata_only", "maxwell_boltzmann")
    selection = select_models(registry, device_topology(device), context, {}, {}, overrides,
                              instance_parameter_sources={source.structured.data.instance_id: source})
    proof = SourceDocument(schema.source.id, schema.input_json.encode())
    inputs = IdentityInputs(device, selection, registry, ModelInputs.from_mappings(defaults={}, overrides=overrides),
                            (SemanticProjection(ref, "all_values_v1", (proof,)),))
    values = physics_identity(inputs).export()["document"]["payload"]["models"][0]["effective_values"]
    assert values["resolved"]["id"] == device.to_mapping()["contacts"][0]["id"]
    assert "device_sha256" not in values


def test_selection_source_and_device_association_are_revalidated(device):
    original = identity_inputs(device)
    other = changed(device, lambda data: data["layers"][0]["parameters"].update(mu_n="2e-10 m^2/Vs"))
    with pytest.raises(ValueError, match="selection does not match"):
        replace(original, prepared=other)
    different_id = changed(device, lambda data: data.update(id="another_device"))
    with pytest.raises(ValueError, match="topology/device"):
        replace(original, prepared=different_id)
    with pytest.raises(ValueError, match="exactly the selected"):
        replace(original, projections=())
    projection = original.projections[0]
    with pytest.raises(ValueError, match="source bytes"):
        replace(original, projections=(replace(projection, sources=(replace(projection.sources[0], content=projection.sources[0].content + b"\n"),)),))


@pytest.mark.parametrize("role", ["protocol", "preparation", "initial_state", "grid", "numeric_controls"])
def test_each_required_execution_role_affects_only_the_new_execution_record(device, execution, role):
    inputs = identity_inputs(device)
    original = execution_identity(inputs, execution)
    document = getattr(execution, role)
    different = replace(execution, **{role: replace(document, content=document.content + b"\n")})
    assert execution_identity(inputs, different).sha256 != original.sha256
    with pytest.raises(ValueError, match="actual immutable SourceDocument"):
        replace(execution, **{role: None})


def test_build_source_content_commit_environment_and_current_head_have_distinct_roles(device, execution):
    inputs = identity_inputs(device)
    base = execution_identity(inputs, execution)
    commit = "a" * 40 if execution.implementation.build_source_commit != "a" * 40 else "b" * 40
    current = replace(execution, current_repository_head=commit)
    assert execution_identity(inputs, current).sha256 == base.sha256
    assert execution_identity(inputs, current).current_repository_head == commit
    new_build = replace(execution.implementation, build_source_commit=commit)
    assert execution_identity(inputs, replace(execution, implementation=new_build)).sha256 != base.sha256
    file = execution.implementation.files[0]
    new_build = replace(execution.implementation, files=(replace(file, content=file.content + b"\n"),))
    assert execution_identity(inputs, replace(execution, implementation=new_build)).sha256 != base.sha256
    environment = replace(execution.environment, settings=(*execution.environment.settings, ("threads", "2")))
    assert execution_identity(inputs, replace(execution, environment=environment)).sha256 != base.sha256
    with pytest.raises(ValueError, match="build-source commit"):
        replace(execution.implementation, build_source_commit="devel")
    with pytest.raises(ValueError, match="source-content"):
        replace(execution.implementation, files=())


def test_raw_initial_state_words_remain_distinct_from_configuration_signed_zero(device, execution):
    inputs = identity_inputs(device)
    positive = replace(execution, initial_state=SourceDocument("state.raw.f64", struct.pack("<d", 0.0)))
    negative = replace(execution, initial_state=SourceDocument("state.raw.f64", struct.pack("<d", -0.0)))
    assert execution_identity(inputs, positive).sha256 != execution_identity(inputs, negative).sha256


def test_identity_results_and_model_input_snapshots_are_immutable(device, execution):
    inputs = identity_inputs(device)
    result = execution_identity(inputs, execution)
    before = result.canonical_json
    exported = result.export()
    exported["document"]["payload"]["H_physics"] = "changed"
    inputs.model_inputs.default_mapping().clear()
    assert result.canonical_json == before and execution_identity(inputs, execution) == result
    with pytest.raises(FrozenInstanceError):
        result.sha256 = "forged"
    with pytest.raises(FrozenInstanceError):
        result.cache_eligible = True
    with pytest.raises(ValueError, match="actual prepared"):
        input_identity({"some": "payload"})


def test_new_identity_imports_do_not_activate_legacy_native_or_companion_packages(tmp_path):
    script = """
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'perovskite_sim','scipy','sksundae','flint','backend','solarlab_server','solarlab_research'}:
            raise AssertionError('unexpected runtime import: '+fullname)
sys.meta_path.insert(0, Boundary())
import solarlab.config.identity
import solarlab.physics.semantic_projection
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(ROOT / "src")], cwd=tmp_path,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
