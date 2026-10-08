"""Real V1 records, detached source bundles and public preparation boundaries."""
from __future__ import annotations

import builtins
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from solarlab.config.resolve_device import resolve_device, resolve_tandem
from solarlab.materials.source import SourceDocument
from solarlab.reproducibility.preparation import prepare_configuration, read_preparation_context
from solarlab.reproducibility.registry import RegistryDocument, config_id, load_registry, mapping, validate_registry_pair
from solarlab.reproducibility.sources import SourceRef, SourceSet, read_sources

ROOT = Path(__file__).resolve().parents[2]
V1_MATRIX = ROOT / 'perovskite-sim/reproducibility/ConfigBenchmarkMatrix.yaml'
V1_REFINEMENT = ROOT / 'perovskite-sim/reproducibility/NumericalRefinementRegistry.yaml'
MATRIX = ROOT / 'reproducibility/ConfigBenchmarkMatrixV2.yaml'
REFINEMENT = ROOT / 'reproducibility/NumericalRefinementRegistryV2.yaml'


def document(path):
    return SourceDocument(str(path.relative_to(ROOT)), path.read_bytes())


@pytest.fixture(scope='module')
def registries():
    matrix_source, refinement_source = document(MATRIX), document(REFINEMENT)
    matrix_spec = RegistryDocument.model_validate(mapping(matrix_source))
    refinement_spec = RegistryDocument.model_validate(mapping(refinement_source))
    matrix = load_registry(matrix_source, read_sources(ROOT, matrix_spec.sources).documents)
    refinement = load_registry(refinement_source, read_sources(ROOT, refinement_spec.sources).documents)
    validate_registry_pair(matrix, refinement)
    return matrix, refinement


def changed(registry, edit):
    raw = registry.export()
    edit(raw)
    return load_registry(SourceDocument(registry.source.id, yaml.safe_dump(raw).encode()), registry.sources.documents)


def test_complete_original_inventory_preserves_current_and_historical_records(registries):
    matrix, refinement = registries
    old_matrix = mapping(document(V1_MATRIX))
    old_refinement = mapping(document(V1_REFINEMENT))
    assert Counter(entry.section for entry in matrix.spec.entries) == {'configs': 55, 'benchmarks': 44}
    assert len(refinement.spec.entries) == 47
    assert len(old_matrix['resources']) == 21
    assert matrix.original_bytes() == V1_MATRIX.read_bytes()
    assert refinement.original_bytes() == V1_REFINEMENT.read_bytes()
    assert {entry.original_key for entry in refinement.spec.entries} == set(old_refinement['lanes'])
    for row in old_matrix['configs']:
        entry = matrix.entry(config_id(row['path']))
        assert matrix.original(entry.id) == row
        assert entry.legacy_semantic_sha256 == row['semantic_sha256']
        assert entry.legacy_status == row['status']
        assert entry.new_executor is None and entry.identities.H_physics is None
        assert entry.identities.protocol_sha256 is None
        assert entry.identities.dependencies
    for id, row in old_matrix['benchmarks'].items():
        assert matrix.original('benchmark:' + id) == row
    for id, row in old_refinement['lanes'].items():
        assert refinement.original('lane:' + id) == row
    certified = matrix.original('benchmark:interface-charge-jv-api-integration')
    assert any('b08bd323e50c8efb9ee7f51428c35a480960c0c39edc65b8500831c4eee5a52d' in value and '3fa71d6' in value for value in certified['limitations'])
    historical = refinement.original('lane:ionmonger-ion-aware-dc-resolved-v2')
    assert any('v1 partial certificate' in value for value in historical['limitations'])
    historical['limitations'].clear()
    assert refinement.original('lane:ionmonger-ion-aware-dc-resolved-v2')['limitations']
    for registry in registries:
        assert not registry.can_execute and registry.spec.activation == 'pending_migration'
        assert registry.sources.reference(registry.spec.original_registry).source_commit == 'e3eee454b1cf61ff43e2cb668035be0527dacdd3'


def test_all_supported_configurations_reach_actual_public_preparation_without_numerical_imports(registries, monkeypatch):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(('perovskite_sim', 'backend', 'scipy.integrate')), name
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    matrix = registries[0]
    defaults, resources = read_preparation_context(matrix)
    prepared, blocked = {}, {}
    for entry in matrix.spec.entries:
        if entry.section != 'configs':
            continue
        result = prepare_configuration(matrix, entry.id, defaults=defaults, resources=resources)
        assert not result.can_execute
        if result.prepared is None:
            blocked[entry.original_key] = [issue.model_dump(mode='json') for issue in result.declaration.issues]
            continue
        value = result.prepared
        prepared[entry.original_key] = value
        assert not value.can_execute
        assert value.content_sha256 == entry.preparation.content_sha256
        input = value.to_input()
        resolver = resolve_tandem if input.schema_version == 'solarlab.tandem-preparation.v1' else resolve_device
        assert resolver(input, defaults, resources, sources=value.sources).content_sha256 == value.content_sha256
    assert len(prepared) == 51
    assert set(blocked) == {'tests/fixtures/configs/tpv_physical_reference.yaml',
        'tests/fixtures/configs/driftfusion_benchmark_tmm.yaml', 'tests/fixtures/configs/ionmonger_benchmark_tmm.yaml', 'configs/calado2016_ion_sweep.yaml'}
    assert 'temperature' in blocked['tests/fixtures/configs/tpv_physical_reference.yaml'][0]['message']
    assert blocked['configs/calado2016_ion_sweep.yaml'][0]['path'] == ['simulation_hints', 'jv_sweep']
    assert 'full-layer-aligned' in blocked['tests/fixtures/configs/ionmonger_benchmark_tmm.yaml'][0]['message']
    scaps = prepared['configs/scaps_mirror_v2.yaml']
    assert scaps.layers[2].material.values['chi'] == 3.94
    assert scaps.to_mapping()['diagnostics']  # Historical effective behavior is retained, not repaired.
    assert prepared['tests/fixtures/configs/tandem_lin2019.yaml'].to_input().top_cell_reference == 'nip_wideGap_FACs_1p77.yaml'


@pytest.mark.parametrize('target', ['registry', 'configuration', 'executor'])
def test_missing_or_tampered_source_is_rejected(registries, target):
    matrix, refinement = registries
    selected = refinement if target == 'executor' else matrix
    if target == 'registry':
        id = selected.spec.original_registry
    elif target == 'configuration':
        id = 'perovskite-sim/configs/scaps_mirror_v2.yaml'
    else:
        id = selected.spec.entries[0].executor.source_id
    original = selected.sources.document(id)
    missing = tuple(source for source in selected.sources.documents if source.id != id)
    with pytest.raises(ValueError, match='missing'):
        load_registry(selected.source, missing)
    changed_sources = tuple(replace(source, content=original.content + b'\n# tampered bytes\n') if source.id == id else source for source in selected.sources.documents)
    with pytest.raises(ValueError, match='source SHA-256 mismatch'):
        load_registry(selected.source, changed_sources)


@pytest.mark.parametrize('edit,match', [
    (lambda raw: raw['entries'].pop(), 'inventory'),
    (lambda raw: raw['entries'].append(deepcopy(raw['entries'][0])), 'inventory'),
    (lambda raw: raw['entries'][0].update(id='config:wrong.yaml'), 'stable identity'),
    (lambda raw: raw['entries'][0].update(original_record_sha256='0'*64), 'record digest'),
    (lambda raw: raw['entries'][0].update(legacy_semantic_sha256='0'*64), 'evidence/semantic'),
    (lambda raw: raw['entries'][0].update(configurations=['config:configs/scaps_mirror_v2.yaml']), 'configuration link'),
    (lambda raw: raw['entries'][55].update(test_nodes=['tests/reproducibility/test_matrix.py::other']), 'test/node'),
    (lambda raw: raw['entries'][0]['preparation']['resolver'].update(symbol='resolve_tandem'), 'reader binding'),
    (lambda raw: raw['entries'][0]['preparation'].update(input_id='registry.retargeted'), 'configuration identity'),
])
def test_broken_old_to_new_and_test_reader_bindings_fail_closed(registries, edit, match):
    with pytest.raises(ValueError, match=match):
        changed(registries[0], edit)


def test_executor_location_version_and_cross_registry_bindings_are_exact(registries):
    matrix, refinement = registries
    with pytest.raises(ValueError, match='binding'):
        changed(refinement, lambda raw: raw['entries'][0]['executor'].update(symbol='run_mobile_ion_transient_jv'))
    with pytest.raises(ValueError, match='executor binding'):
        changed(refinement, lambda raw: raw['entries'][0].update(executor_version='new'))
    with pytest.raises(ValueError, match='configuration link'):
        changed(refinement, lambda raw: raw['entries'][0].update(configurations=['config:missing.yaml']))
    with pytest.raises(ValueError, match='required'):
        validate_registry_pair(refinement, matrix)


@pytest.mark.parametrize('edit', [
    lambda raw: raw.update(activation='active'),
    lambda raw: raw['entries'][0].update(new_executor='solarlab.fake:run'),
    lambda raw: raw['entries'][0]['identities'].update(H_physics='1'*64),
    lambda raw: raw['entries'][0]['identities'].update(H_execution='2'*64),
    lambda raw: raw['entries'][0]['identities'].update(protocol_sha256='3'*64),
    lambda raw: raw['entries'][0]['identities'].update(dependencies=[]),
    lambda raw: raw['entries'][55].update(new_test_nodes=['tests/migrated.py::test_pass']),
])
def test_no_premature_activation_executor_certificate_or_identity_claim(registries, edit):
    with pytest.raises(ValueError):
        changed(registries[0], edit)


def test_forged_prepared_content_or_default_context_cannot_be_advertised(registries):
    original = registries[0]
    defaults, resources = read_preparation_context(original)
    altered = changed(original, lambda raw: raw['entries'][0]['preparation'].update(content_sha256='0'*64))
    with pytest.raises(ValueError, match='actual public reader'):
        prepare_configuration(altered, altered.spec.entries[0].id, defaults=defaults, resources=resources)
    altered = changed(original, lambda raw: raw['preparation_context'].update(default_catalog_sha256='0'*64))
    with pytest.raises(ValueError, match='context identity'):
        read_preparation_context(altered)


def test_duplicate_keys_ambiguous_source_paths_and_path_escape_rejected(registries, tmp_path):
    matrix = registries[0]
    with pytest.raises(ValueError, match='unique strings'):
        load_registry(SourceDocument('duplicate', matrix.source.content + b'\nactivation: active\n'), matrix.sources.documents)
    ref = matrix.spec.sources[0]
    raw = ref.model_dump()
    for path in ('../ConfigBenchmarkMatrix.yaml', '/absolute.yaml', 'a/../b.yaml', 'a\\b', 'a//b'):
        with pytest.raises(ValueError, match='bundle-relative'):
            SourceRef.model_validate(dict(raw, path=path))
    duplicate = SourceRef.model_validate(dict(raw, id='second', source_commit='0'*40))
    one = matrix.sources.document(ref.id)
    ambiguous = SourceSet((ref,duplicate),(one,SourceDocument('second',one.content)))
    with pytest.raises(ValueError, match='ambiguous'):
        ambiguous.at_path(ref.path)
    outside = tmp_path/'outside'
    outside.write_bytes(one.content)
    base=tmp_path/'bundle'
    base.mkdir()
    link=base/'ref'
    link.symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        read_sources(base,(SourceRef.model_validate(dict(raw,path='ref')),))


def test_relocated_bundle_and_detached_sources_need_no_checkout_or_working_directory(registries, tmp_path, monkeypatch):
    matrix = registries[0]
    for ref in matrix.spec.sources:
        path=tmp_path/ref.path
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(matrix.sources.document(ref.id).content)
    sources = read_sources(tmp_path,matrix.spec.sources)
    monkeypatch.chdir(tmp_path)
    detached = load_registry(SourceDocument('caller-supplied-v2',matrix.source.content),sources.documents)
    def forbidden(*args, **kwargs):
        raise AssertionError('Detached preparation must use supplied bytes')
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    defaults, resources = read_preparation_context(detached)
    result = prepare_configuration(detached,'config:configs/scaps_mirror_v2.yaml',defaults=defaults,resources=resources)
    assert result.prepared is not None and not result.can_execute
    assert detached.original_bytes() == matrix.original_bytes()


def test_unchanged_legitimate_readers_and_preparation_consumers_import(tmp_path):
    code = '''
import json
from pathlib import Path
import perovskite_sim.reproducibility as legacy
from perovskite_sim.validation.numerical_certificate import load_refinement_registry
from solarlab.config.resolve_device import resolve_device
from solarlab.experiments.jv.preparation import prepare_jv_experiment
from solarlab.experiments.two_dimensional.preparation import prepare_spatial_experiment
from solarlab.sweeps.preparation import prepare_sweep
from solarlab.reproducibility.registry import load_registry
root=Path(__import__('sys').argv[1])
value=load_refinement_registry(root/'perovskite-sim/reproducibility/NumericalRefinementRegistry.yaml',project_root=root/'perovskite-sim')
assert len(value.lanes)==47
print(json.dumps(dict(lanes=len(value.lanes),legacy_reader=legacy.__file__,new_reader=__import__('inspect').getfile(load_registry),scientific_calls=0)))
'''
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT/'perovskite-sim'),PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT)],cwd=tmp_path,env=env,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    receipt=json.loads(result.stdout)
    assert receipt['lanes']==47 and receipt['scientific_calls']==0
    assert Path(receipt['new_reader']).resolve()==ROOT/'src/solarlab/reproducibility/registry.py'
    assert Path(receipt['legacy_reader']).resolve()==ROOT/'perovskite-sim/perovskite_sim/reproducibility.py'


def test_export_and_input_source_byte_identities_are_independent(registries):
    matrix=registries[0]
    source=SourceDocument('reopened',json.dumps(matrix.export()).encode())
    reopened=load_registry(source,matrix.sources.documents)
    assert reopened.content_sha256==matrix.content_sha256
    assert reopened.original_bytes()==matrix.original_bytes()
    assert hashlib.sha256(reopened.original_bytes()).hexdigest()==matrix.sources.reference(matrix.spec.original_registry).sha256
    assert source.sha256!=matrix.source.sha256


def test_blocked_mappings_preserve_the_actual_legacy_behavior_and_future_obligation(registries):
    matrix = registries[0]
    tpv = matrix.entry('config:tests/fixtures/configs/tpv_physical_reference.yaml')
    assert tpv.migration_status == 'blocked_input'
    assert {item.source.symbol for item in tpv.migration_notes} == {'load_device_from_yaml', 'stack_from_dict'}
    assert all('temperature is ignored, not an established alias' in item.note for item in tpv.migration_notes)
    for name in ('driftfusion_benchmark_tmm', 'ionmonger_benchmark_tmm'):
        entry = matrix.entry(f'config:tests/fixtures/configs/{name}.yaml')
        note, = entry.migration_notes
        assert note.source.symbol == 'interfaces_from_device_dict'
        assert 'raw index' in note.note and 'pads remaining slots with (0,0)' in note.note
        assert 'no automatic pair shift or permanent exclusion' in note.note
    hints = matrix.entry('config:configs/calado2016_ion_sweep.yaml')
    note, = hints.migration_notes
    assert note.source.symbol == 'load_simulation_hints'
    assert 'lossless typed migration' in note.note and 'deleting them' in note.note
    with pytest.raises(ValueError, match='missing or ambiguous symbol'):
        changed(matrix, lambda raw: next(entry for entry in raw['entries'] if entry['id'] == tpv.id)['migration_notes'][0]['source'].update(symbol='invented_alias_converter'))
