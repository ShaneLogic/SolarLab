"""Bounded analysis observation controls, not a physical qualification run."""
from copy import deepcopy
import inspect
import json
import weakref

import pytest

from scripts import analyze_r1_v9_prototype as analysis
from scripts.r1_v16_memory import NativeMemoryObserver


ROOT_NAMES = {'prepared', 'raw_result', 'result', 'rows', 'persisted', 'replay'}
CONTEXT_PHASES = ['analysis_context_rows_match', 'analysis_prepared_seal',
                  'analysis_result_seal', 'analysis_source_digest', 'analysis_rows_digest']
BUDGET = {
    'local_state_arithmetic': {'potential_qf_trace_absolute_error_V': 1e-26,
                               'population_occupancy_relative_error': 1e-27},
    'independent_side_operator': {'symmetric_relative_target': 1e-10,
                                  'floor': 1., 'precision_stability_target': 1e-20},
    'comparison': {'original_gate': 1e-6},
}


class WeakDict(dict):
    """Track whether the diagnostic helpers retain supplied containers."""


def sealed(value):
    value = WeakDict(value)
    value['sha256'] = analysis.digest(value)
    return value


def sample(monkeypatch, *, failure=None):
    """Replace physical calculations only; exercise actual seals and files."""
    fields = {name: {'hi': [0.], 'lo': [0.]} for name in (
        'phi_V', 'dqfn_V', 'dqfp_V', 'n_m3', 'p_m3', 'positive_m3', 'occupancy',
        'trace_potential_V', 'trace_state_m3', 'sheet_charge_C_m2',
        'positive_flux_m2_s', 'positive_rate_m3_s', 'boundary_flux_m2_s')}
    rows = []
    for index, time_s in enumerate((0., 1.)):
        direct = deepcopy(fields)
        direct['phi_V']['hi'] = [float(index)]
        rows.append({'time_s': time_s, 'substeps': 1, 'state': {'fields': direct},
                     'physics_reconstruction': {
                         'eliminated_precision': {'fields': deepcopy(direct)},
                         'eliminated_operator': {
                             name: {'normalization_floor': 1., 'unit': unit}
                             for name, unit in (('positive_ion_flux', 'm^-2 s^-1'),
                                                ('positive_ion_rate', 'm^-3 s^-1'))}}})
    prepared = sealed({'schema': 'SyntheticPrepared', 'state': {'fields': fields}})
    result = sealed({'schema': 'SyntheticResult', 'source': {'synthetic': True},
                     'accepted_steps': rows})
    context = {'source_digest': analysis.digest(result['source']),
               'prepared_sha256': prepared['sha256'], 'result_sha256': result['sha256'],
               'saved_rows_digest': analysis.digest(rows)}
    calls = []
    receipt = object()

    def initial(frozen, actual_prepared, actual_result):
        calls.append('initial_checks')
        assert actual_prepared is prepared and actual_result is result
        if failure == 'initial':
            raise ValueError('bounded initial-check failure')
        return {'qualified': True, 'zero_plus_fields': fields}

    def verify(frozen, row, **kwargs):
        calls.append(('verify', kwargs['row_index']))
        assert kwargs['context'] is context and kwargs['replay_receipt'] is receipt
        assert kwargs['recomputation'] is None and not kwargs['fixed_v8_case']
        if kwargs['row_index'] == 1:
            if failure == 'row_error':
                raise ValueError('bounded row-analysis failure')
            if failure == 'interrupt':
                raise KeyboardInterrupt('bounded row-analysis interruption')
        return {'constitutive_checks_passed': True, 'independent_path_verified': True,
                'qualified': True, 'role_binding': {'synthetic': True},
                'fixed_V8_exact_copy_detected': False,
                'arithmetic': {side: {'oracle': row['state']['fields']}
                               for side in ('direct', 'eliminated')}}

    monkeypatch.setattr(analysis, 'initial_checks', initial)
    monkeypatch.setattr(analysis, 'verify_two_sides', verify)
    monkeypatch.setattr(analysis, 'original_gate', lambda row, limit: {'passed': True})
    monkeypatch.setattr(analysis, 'compare_state_arithmetic',
                        lambda *args: {'passed': True, 'synthetic': True})
    return rows, prepared, result, context, receipt, calls


def execute(directory, inputs, *, observer=None):
    rows, prepared, result, context, receipt, _ = inputs
    return analysis.analyze_records(rows, {}, prepared, result, context=context,
        output=directory, budget=BUDGET, replay_receipt=receipt, phase_observer=observer)


def expected_events():
    result = [('analysis_context_validation', 'begin')]
    result.extend((phase, event) for phase in CONTEXT_PHASES for event in ('begin', 'end'))
    result.append(('analysis_context_validation', 'end'))
    result.extend((phase, event) for phase in (
        'analysis_initial_checks', 'analysis_row_analysis', 'analysis_report_writes')
        for event in ('begin', 'end'))
    return result


@pytest.mark.parametrize('function', [analysis.validate_analysis_context, analysis.analyze_records])
def test_observer_is_optional_and_keyword_only(function):
    parameter = inspect.signature(function).parameters['phase_observer']
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_native_metadata_does_not_change_science_files(tmp_path, monkeypatch):
    inputs = sample(monkeypatch)
    # Freeze the sole nondeterministic scientific field for exact file checking.
    monkeypatch.setattr(analysis.time, 'monotonic', lambda: 100.)
    reference = execute(tmp_path / 'reference', inputs)
    reference_calls = list(inputs[-1])
    inputs[-1].clear()
    observer = NativeMemoryObserver(tmp_path / 'NativeMemoryPhasesV1.jsonl')
    try:
        observed = execute(tmp_path / 'observed', inputs, observer=observer)
    finally:
        observer.close()
    assert reference == observed and observed['passed']
    assert inputs[-1] == reference_calls
    assert observer.summary()['passed'] and observer.summary()['roots_retained'] is False
    assert observer.summary()['event_count'] == len(expected_events())
    assert {p.name for p in (tmp_path / 'observed').iterdir()} == {
        'RowsV1.jsonl', 'FailuresV1.json', 'ResultV1.json', 'ManifestV1.json'}
    for path in (tmp_path / 'reference').iterdir():
        assert path.read_bytes() == (tmp_path / 'observed' / path.name).read_bytes(), path.name


def test_phase_order_roots_and_separate_timing_are_explicit(tmp_path, monkeypatch):
    inputs = sample(monkeypatch)
    clock = {'now': 0.}
    monkeypatch.setattr(analysis.time, 'monotonic', lambda: clock['now'])
    reference = execute(tmp_path / 'reference', inputs)
    events = []

    def observe(phase, event, roots):
        assert set(roots) == ROOT_NAMES
        assert roots['prepared'] is inputs[1] and roots['result'] is inputs[2]
        assert roots['rows'] is inputs[0]
        assert roots['raw_result'] is None and roots['persisted'] is None
        assert roots['replay'] is (None if phase in CONTEXT_PHASES else inputs[4])
        events.append((phase, event))
        clock['now'] += .25

    observed = execute(tmp_path / 'observed', inputs, observer=observe)
    assert events == expected_events()
    # Original wall_seconds starts after initial_checks and ends before writes.
    # The row-analysis observer cost is included; no time is subtracted.
    assert observed['wall_seconds'] - reference['wall_seconds'] == .5
    assert {k: v for k, v in observed.items() if k != 'wall_seconds'} == {
        k: v for k, v in reference.items() if k != 'wall_seconds'}
    for name in ('RowsV1.jsonl', 'FailuresV1.json'):
        assert (tmp_path / 'reference' / name).read_bytes() == (tmp_path / 'observed' / name).read_bytes()
    assert analysis.sha(tmp_path / 'observed' / 'ResultV1.json') == json.loads(
        (tmp_path / 'observed' / 'ManifestV1.json').read_text())['ResultV1.json']['sha256']


@pytest.mark.parametrize('mutation,phase,message', [
    ('rows', 'analysis_context_rows_match', 'analysis rows differ from the supplied result'),
    ('prepared', 'analysis_prepared_seal', 'analysis prepared content seal is invalid'),
    ('result', 'analysis_result_seal', 'analysis result content seal is invalid'),
    ('context', 'analysis_context_validation', 'analysis context does not describe the supplied source/preparation/result/rows'),
])
def test_invalid_inputs_keep_exact_exception_and_balanced_error_boundary(
        tmp_path, monkeypatch, mutation, phase, message):
    inputs = sample(monkeypatch)
    if mutation == 'rows':
        inputs = (inputs[0][:-1], *inputs[1:])
    elif mutation == 'prepared':
        inputs[1]['schema'] = 'tampered'
    elif mutation == 'result':
        inputs[2]['source']['synthetic'] = False
    else:
        inputs[3]['saved_rows_digest'] = 'tampered'
    with pytest.raises(ValueError, match=message) as original:
        execute(tmp_path / 'disabled', inputs)
    events = []
    with pytest.raises(ValueError, match=message) as observed:
        execute(tmp_path / 'enabled', inputs, observer=lambda p, e, r: events.append((p, e)))
    assert str(observed.value) == str(original.value)
    assert (phase, 'error') in events and (phase, 'end') not in events
    assert events[-1] == ('analysis_context_validation', 'error')
    stack = []
    for name, event in events:
        if event == 'begin':
            stack.append(name)
        else:
            assert stack.pop() == name
    assert not stack and not inputs[-1]


def test_historical_mode_still_recomputes_and_validates_content_seals(monkeypatch):
    rows, prepared, result, context, _, _ = sample(monkeypatch)
    context.clear()
    events = []
    expected = analysis.validate_analysis_context(rows, prepared, result, context,
        historical=True, phase_observer=lambda p, e, r: events.append((p, e)))
    assert expected['saved_rows_digest'] == analysis.digest(rows)
    assert events == [(phase, event) for phase in CONTEXT_PHASES for event in ('begin', 'end')]
    prepared['schema'] = 'tampered'
    with pytest.raises(ValueError, match='analysis prepared content seal is invalid'):
        analysis.validate_analysis_context(rows, prepared, result, context, historical=True)


@pytest.mark.parametrize('failure,phase,kind,message', [
    ('initial', 'analysis_initial_checks', ValueError, 'bounded initial-check failure'),
    ('interrupt', 'analysis_row_analysis', KeyboardInterrupt, 'bounded row-analysis interruption'),
])
def test_scientific_exception_or_interrupt_propagates_with_error_boundary(
        tmp_path, monkeypatch, failure, phase, kind, message):
    inputs = sample(monkeypatch, failure=failure)
    events = []
    with pytest.raises(kind, match=message):
        execute(tmp_path / 'enabled', inputs, observer=lambda p, e, r: events.append((p, e)))
    assert events[-2:] == [(phase, 'begin'), (phase, 'error')]
    assert not (tmp_path / 'enabled' / 'ResultV1.json').exists()
    if failure == 'interrupt':
        assert len((tmp_path / 'enabled' / 'RowsV1.jsonl').read_text().splitlines()) == 1


def test_caught_row_error_remains_a_failed_report_with_retained_row_evidence(tmp_path, monkeypatch):
    inputs = sample(monkeypatch, failure='row_error')
    events = []
    report = execute(tmp_path / 'enabled', inputs,
                     observer=lambda p, e, r: events.append((p, e)))
    assert events == expected_events()
    assert not report['passed'] and report['failed_rows'] == 1
    assert report['first_failure']['error'] == {
        'type': 'ValueError', 'message': 'bounded row-analysis failure'}
    assert len((tmp_path / 'enabled' / 'RowsV1.jsonl').read_text().splitlines()) == 2


def test_report_write_error_retains_original_exception(tmp_path, monkeypatch):
    inputs = sample(monkeypatch)
    original_write = analysis.write

    def broken_write(path, value):
        if path.name == 'ResultV1.json':
            raise OSError('bounded report-write failure')
        return original_write(path, value)

    monkeypatch.setattr(analysis, 'write', broken_write)
    events = []
    with pytest.raises(OSError, match='bounded report-write failure'):
        execute(tmp_path / 'enabled', inputs, observer=lambda p, e, r: events.append((p, e)))
    assert events[-2:] == [('analysis_report_writes', 'begin'), ('analysis_report_writes', 'error')]
    assert (tmp_path / 'enabled' / 'RowsV1.jsonl').exists()
    assert (tmp_path / 'enabled' / 'FailuresV1.json').exists()
    assert not (tmp_path / 'enabled' / 'ManifestV1.json').exists()


@pytest.mark.parametrize('event', ['begin', 'end'])
def test_observer_failure_propagates_to_the_callers_diagnostic_sink(tmp_path, monkeypatch, event):
    inputs = sample(monkeypatch)

    def broken(phase, actual_event, roots):
        if phase == 'analysis_initial_checks' and actual_event == event:
            raise RuntimeError('bounded observation failure')

    with pytest.raises(RuntimeError, match='bounded observation failure'):
        execute(tmp_path / 'enabled', inputs, observer=broken)


@pytest.mark.parametrize('observer_error', [RuntimeError, KeyboardInterrupt])
def test_error_observation_never_overwrites_original_failure(tmp_path, monkeypatch, observer_error):
    inputs = sample(monkeypatch, failure='initial')

    def broken(phase, event, roots):
        if event == 'error':
            raise observer_error('secondary observation failure')

    with pytest.raises(ValueError, match='bounded initial-check failure'):
        execute(tmp_path / 'enabled', inputs, observer=broken)


def test_closed_phase_does_not_retain_root_mapping_or_input_containers():
    prepared, result, rows = WeakDict(), WeakDict(), WeakDict()
    references = [weakref.ref(value) for value in (prepared, result, rows)]
    events = []
    with analysis.observed_analysis_phase(lambda p, e, r: events.append((p, e)), 'bounded',
                                          prepared=prepared, result=result, rows=rows):
        pass
    del prepared, result, rows
    assert all(reference() is None for reference in references)
    assert events == [('bounded', 'begin'), ('bounded', 'end')]
