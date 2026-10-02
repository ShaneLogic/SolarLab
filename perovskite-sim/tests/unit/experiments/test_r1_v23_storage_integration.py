"""Result-only streaming and storage observations preserve bounded evidence."""
from copy import deepcopy
import io
import json
import weakref
import zipfile

import numpy as np
import pytest

from scripts import run_r1_v9_prototype as runner
from scripts import run_r1_v12_cases as cases
from scripts import r1_v23_storage_observation as storage
from scripts.r1_v16_memory import NativeMemoryObserver
from perovskite_sim.experiments import one_dimensional_mechanism_r1_pair_codec as codec
from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
from tests.unit.experiments.test_r1_v16_runner_observation import fixture_api, scalar_report
from tests.unit.experiments.test_r1_v19_tail_observation import synthetic_case_shell


ROOTS = {'prepared', 'raw_result', 'result', 'rows', 'persisted', 'replay'}
JSON_PHASES = ['result_json_validate', 'result_json_stream', 'result_json_sync', 'result_json_replace']
NUMERIC_PHASES = ['numeric_payload', 'numeric_array_collection', 'numeric_archive_write']
SCIENCE_FILES = ['PreparedV1.json', 'ResultV1.json', 'AcceptedStepsV1.jsonl',
                 'ReplayRowsV1.jsonl', 'PhysicsReplayV1.json', 'ReplayLedgerV1.json']


def write_result_json(*args, **kwargs):
    # The implementation has its own independent writer/atomicity test suite.
    from scripts.r1_v23_json_writer import write_result_json as write
    return write(*args, **kwargs)


def original_numeric(path, result):
    if protocol.nonfinite_numeric_paths(result):
        with path.open('wb') as stream:
            np.savez_compressed(stream, **codec.numeric_arrays(result, allow_nonfinite=True))
    else:
        codec.write_numeric_sidecar(path, result)


def npz_members(path):
    """Compare the member format including memory order, not ZIP timestamps."""
    result = []
    with zipfile.ZipFile(path) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for name in archive.namelist():
            raw = archive.read(name)
            stream = io.BytesIO(raw)
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            result.append((name, version, dtype.str, shape, fortran, stream.read()))
    return result


def storage_fixture(*, candidate=False, failure=False, nonfinite=False):
    case, state, api = fixture_api(fail='integrate' if failure else None)
    original_run = api['run']

    def integrate(prepared, observer):
        def observe(row):
            row['dt_s'] = 0. if row['time_s'] == 0. else 1.
            row['state']['pair'] = {'hi': 1., 'lo': 2.**-80}
            row['state']['negative_zero_pair'] = {'hi': -0., 'lo': -0.}
            observer(row)
        try:
            result = original_run(prepared, observe)
        except RuntimeError as error:
            result = error.result
            if nonfinite:
                result['failure'] = {'numerical_evidence': {
                    'attempted': np.array([np.nan, np.inf, -np.inf, -0.]),
                    'pair': {'hi': 1., 'lo': 2.**-80}}}
            raise
        result['ordered_arrays'] = {
            'fortran': np.asfortranarray(np.arange(12.).reshape(3, 4)),
            'strided': np.arange(24.).reshape(4, 6)[:, ::2]}
        return result

    def persist(path, result):
        assert result is state['raw_reference']()
        assert result['accepted_steps'][0]['state']['native_words'].dtype == np.dtype('float64')
        state['calls'].append('numeric_write')
        return original_numeric(path, result)

    def observed_persist(path, result, *, phase_observer):
        assert result is state['raw_reference']()
        state['calls'].append('numeric_write')
        return storage.persist_numeric_observed(path, result, phase_observer=phase_observer)

    def verify(path, result):
        state['calls'].append('numeric_readback')
        if not failure:
            assert state['raw_reference']() is None and state['array_reference']() is None
        method = codec.verify_failed_numeric_sidecar if nonfinite else codec.verify_numeric_sidecar
        return method(path, result)

    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import json_data
    api.update(run=integrate, json_data=lambda value: runner.tag_nonfinite(json_data(value)),
               persist_numeric=persist, verify_numeric=verify)
    if candidate:
        api.update(persist_result=write_result_json, persist_result_observed=write_result_json,
                   persist_numeric_observed=observed_persist)
    return case, state, api


def execute(directory, *, candidate=False, observer=None, failure=False, nonfinite=False):
    directory.mkdir()
    case, state, api = storage_fixture(candidate=candidate, failure=failure, nonfinite=nonfinite)
    report = runner.run_trajectory(directory, case, 'baseline', api, lambda: None,
                                   memory_observer=observer, extent_check=cases.extent)
    return report, state


@pytest.mark.parametrize('observed', [False, True])
def test_result_only_candidate_preserves_scientific_bytes_and_npz_member_order(tmp_path, observed):
    reference, reference_state = execute(tmp_path / 'old')
    boundaries = []

    def observe(phase, event, roots):
        assert set(roots) == ROOTS
        if phase in JSON_PHASES:
            assert roots['result'] is not roots['raw_result']
            assert isinstance(roots['result'], dict) and roots['raw_result'] is not None
        if phase in NUMERIC_PHASES:
            assert isinstance(roots['raw_result']['ordered_arrays']['fortran'], np.ndarray)
        boundaries.append((phase, event))

    current, state = execute(tmp_path / 'new', candidate=True, observer=observe if observed else None)
    assert current['four_predicates_passed'] and scalar_report(reference) == scalar_report(current)
    assert state['calls'] == reference_state['calls']
    assert 'result_write_observation' not in reference
    assert current['result_write_observation'] and all(
        type(value) in (str, int, float, bool, type(None))
        for value in current['result_write_observation'].values())
    for name in SCIENCE_FILES:
        assert (tmp_path / 'old' / name).read_bytes() == (tmp_path / 'new' / name).read_bytes(), name
        assert runner.sha(tmp_path / 'old' / name) == runner.sha(tmp_path / 'new' / name)
    assert npz_members(tmp_path / 'old/StateArraysV1.npz') == npz_members(tmp_path / 'new/StateArraysV1.npz')
    assert state['raw_reference']() is None and state['array_reference']() is None
    if observed:
        assert [(p, e) for p, e in boundaries if p in JSON_PHASES + NUMERIC_PHASES] == [
            (phase, event) for phase in JSON_PHASES + NUMERIC_PHASES for event in ('begin', 'end')]
        assert boundaries.index(('numeric_archive_write', 'end')) < boundaries.index(('raw_release', 'begin'))


@pytest.mark.parametrize('nonfinite', [False, True])
def test_finite_and_tagged_nonfinite_failure_prefix_remains_lossless(tmp_path, nonfinite):
    old, _ = execute(tmp_path / 'old', failure=True, nonfinite=nonfinite)
    new, _ = execute(tmp_path / 'new', candidate=True, observer=lambda *args: None,
                     failure=True, nonfinite=nonfinite)
    assert old['failure'] == new['failure'] == {
        'type': 'RuntimeError', 'message': 'bounded integration failure'}
    assert not new['four_predicates_passed'] and new['extent']['accepted_rows'] == 2
    assert new['numeric_sidecar_exact'] and new['replay_completed']
    for name in SCIENCE_FILES + ['FailureWitnessV1.json']:
        assert (tmp_path / 'old' / name).read_bytes() == (tmp_path / 'new' / name).read_bytes(), name
    assert npz_members(tmp_path / 'old/StateArraysV1.npz') == npz_members(tmp_path / 'new/StateArraysV1.npz')
    result = json.loads((tmp_path / 'new/ResultV1.json').read_text())
    if nonfinite:
        assert result['failure']['numerical_evidence']['attempted'] == [
            {'nonfinite': 'nan'}, {'nonfinite': 'inf'}, {'nonfinite': '-inf'}, -0.]
        assert result['certificate']['certified'] is False and 'sha256' not in result


@pytest.mark.parametrize('phase', ['result_json_stream', 'numeric_array_collection'])
@pytest.mark.parametrize('failure', [False, True])
def test_sticky_storage_callback_failure_preserves_science_and_original_failure(tmp_path, phase, failure):
    reference, _ = execute(tmp_path / 'old', failure=failure)
    events = []

    def broken(name, event, roots):
        events.append((name, event))
        if name == phase and event == 'begin':
            raise RuntimeError('bounded storage diagnostic failure')

    observed, _ = execute(tmp_path / 'new', candidate=True, observer=broken, failure=failure)
    assert scalar_report(reference) == scalar_report(observed)
    assert not observed['memory_observation']['passed']
    assert observed['memory_observation']['error']['message'] == 'bounded storage diagnostic failure'
    assert events[-1] == (phase, 'begin')
    assert observed['memory_observation']['call_count'] == len(events)
    for name in SCIENCE_FILES:
        assert (tmp_path / 'old' / name).read_bytes() == (tmp_path / 'new' / name).read_bytes()


def test_real_native_storage_events_balance_and_join_one_receipt(tmp_path):
    native_path = tmp_path / 'NativeMemoryPhasesV1.jsonl'
    observer = NativeMemoryObserver(native_path)
    try:
        report, _ = execute(tmp_path / 'case', candidate=True, observer=observer)
    finally:
        observer.close()
    records = [json.loads(line) for line in native_path.read_text().splitlines()]
    assert report['memory_observation']['passed'] and observer.summary()['passed']
    assert report['memory_observation']['call_count'] == observer.summary()['event_count'] == len(records)
    assert all(set(row['roots']) == ROOTS for row in records)
    assert [(row['phase'], row['event']) for row in records if row['phase'] in JSON_PHASES + NUMERIC_PHASES] == [
        (phase, event) for phase in JSON_PHASES + NUMERIC_PHASES for event in ('begin', 'end')]


def test_storage_callback_time_stays_in_original_main_cost(tmp_path, monkeypatch):
    clock = {'now': 100.}
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock['now'])
    events = []

    def observe(phase, event, roots):
        events.append((phase, event))
        clock['now'] += .125

    report, _ = execute(tmp_path / 'case', candidate=True, observer=observe)
    measurement = report['memory_observation']
    assert measurement['elapsed_s'] == len(events) * .125
    assert measurement['main_elapsed_s'] == sum(
        phase not in {'rows_release', 'after_main'} for phase, event in events) * .125
    assert report['cost']['elapsed_s'] >= measurement['main_elapsed_s']
    stats = report['result_write_observation']
    assert stats['total_elapsed_s'] == len(JSON_PHASES) * 2 * .125
    assert stats['other_elapsed_s'] == stats['total_elapsed_s']
    assert report['phase_timing']['exclusive_sum_s'] <= report['cost']['elapsed_s']


def test_disabled_diagnostics_ignore_observed_only_hooks(tmp_path):
    directory = tmp_path / 'case'
    directory.mkdir()
    case, _, api = storage_fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError('observed storage hook used when diagnostics were disabled')

    api.update(persist_result_observed=forbidden, persist_numeric_observed=forbidden)
    report = runner.run_trajectory(directory, case, 'baseline', api, lambda: None,
                                   extent_check=cases.extent)
    assert report['four_predicates_passed']
    assert 'result_write_observation' not in report and 'memory_observation' not in report


@pytest.mark.parametrize('metadata', [{'unexpected': []}, {'result': {}}, {'nan': float('nan')}])
def test_writer_cannot_attach_retained_objects_or_nonfinite_stats(tmp_path, metadata):
    directory = tmp_path / 'case'
    directory.mkdir()
    case, _, api = storage_fixture()

    def bad_writer(path, value):
        runner.write(path, value)
        return metadata

    api['persist_result'] = bad_writer
    report = runner.run_trajectory(directory, case, 'baseline', api, lambda: None,
                                   extent_check=cases.extent)
    assert report['execution_status'] == 'failed' and not report['four_predicates_passed']
    assert 'result_write_observation' not in report
    assert report['failure']['message'] == 'result writer observation must contain only finite scalar metadata'


@pytest.mark.parametrize('dtype,layout', [
    ('float64', 'C'), ('float64', 'F'), ('float64', 'strided'), ('float32', 'F'), ('int32', 'strided'),
])
def test_numeric_staging_preserves_dtype_shape_layout_and_all_member_bytes(tmp_path, dtype, layout):
    source = np.arange(24, dtype=dtype).reshape(4, 6)
    array = np.asfortranarray(source) if layout == 'F' else source[:, ::2] if layout == 'strided' else source
    result = {'array': array, 'scalar_pair': {'hi': 1., 'lo': -0.},
              'small': [2.**-80, -0.], 'integer_array': np.array([1, 2], dtype='int32')}
    original_numeric(tmp_path / 'old.npz', result)
    boundaries = []
    storage.persist_numeric_observed(tmp_path / 'new.npz', result,
        phase_observer=lambda p, e, r: boundaries.append((p, e)))
    assert boundaries == [(phase, event) for phase in NUMERIC_PHASES for event in ('begin', 'end')]
    assert npz_members(tmp_path / 'old.npz') == npz_members(tmp_path / 'new.npz')
    assert codec.verify_numeric_sidecar(tmp_path / 'new.npz', result)['exact_key_coverage']


def test_numeric_payload_mapping_is_not_retained_into_archive_write(tmp_path, monkeypatch):
    class Payload(dict):
        pass

    references = {}

    def payload(result):
        value = Payload(words=np.array([1., -0., 2.**-80]))
        references['payload'] = weakref.ref(value)
        return value

    monkeypatch.setattr(storage.codec, 'state_sidecar_payload', payload)
    original_archive = storage.np.savez_compressed

    def archive(stream, **arrays):
        assert references['payload']() is None
        return original_archive(stream, **arrays)

    monkeypatch.setattr(storage.np, 'savez_compressed', archive)
    storage.persist_numeric_observed(tmp_path / 'new.npz', {'synthetic': True},
        phase_observer=lambda p, e, r: None)
    assert references['payload']() is None


@pytest.mark.parametrize('phase,operation', [
    ('numeric_payload', 'state_sidecar_payload'), ('numeric_array_collection', 'numeric_arrays'),
    ('numeric_archive_write', 'savez_compressed'),
])
def test_numeric_phase_error_preserves_original_error_and_direct_write_behavior(tmp_path, monkeypatch, phase, operation):
    events = []

    def fail(*args, **kwargs):
        raise OSError('bounded numeric storage failure')

    owner = storage.np if operation == 'savez_compressed' else storage.codec
    monkeypatch.setattr(owner, operation, fail)
    with pytest.raises(OSError, match='bounded numeric storage failure'):
        storage.persist_numeric_observed(tmp_path / 'new.npz', {'array': np.array([1.])},
            phase_observer=lambda p, e, r: events.append((p, e)))
    assert events[-1] == (phase, 'error')
    assert (tmp_path / 'new.npz').is_file()  # Original wb route is deliberately unchanged.
    assert not list(tmp_path.glob('.pending-*'))


def test_v12_explicitly_wires_only_the_result_writer_and_observed_numeric_path(tmp_path, monkeypatch):
    from scripts.r1_v23_json_writer import write_result_json as actual_writer
    args, _ = synthetic_case_shell(tmp_path, monkeypatch)
    calls = []

    def trajectory(directory, case, mode, api, source_guard, **kwargs):
        assert api['persist_result'] is actual_writer
        assert api['persist_result_observed'] is actual_writer
        assert api['persist_numeric_observed'] is storage.persist_numeric_observed
        little_case, _, little_api = storage_fixture()
        little_api.update({name: api[name] for name in (
            'persist_result', 'persist_result_observed', 'persist_numeric', 'persist_numeric_observed', 'verify_numeric')})
        report = runner.run_trajectory(directory, little_case, mode, little_api, source_guard,
            memory_observer=kwargs.get('memory_observer'), extent_check=cases.extent)
        report['result_request_binding_passed'] = True
        calls.append(report['result_write_observation'])
        return report

    monkeypatch.setattr(cases, 'run_trajectory', trajectory)
    assert cases.run(args) == 0
    assert len(calls) == 1
    summary = json.loads((args.output / 'SummaryV1.json').read_text())
    assert summary['result_write_observation'] == calls[0]
    assert summary['memory_observation']['passed'] and summary['memory_collector']['closed']
    manifest = json.loads((args.output / 'ManifestV1.json').read_text())
    assert manifest['ResultV1.json']['sha256'] == runner.sha(args.output / 'ResultV1.json')
