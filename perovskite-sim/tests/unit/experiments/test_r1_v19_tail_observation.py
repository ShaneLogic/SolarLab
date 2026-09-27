"""Bounded end-to-end observer/final-seal controls; no physical solve is used."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import run_r1_v9_prototype as runner
from scripts import run_r1_v12_cases as cases
from scripts import r1_v19_observation as tail_module
from scripts.r1_v16_memory import NativeMemoryObserver
from scripts.r1_v19_observation import TailObservation
from tests.unit.experiments.test_r1_v16_runner_observation import (
    ROOT_NAMES, SCIENCE_FILES, fixture_api as v16_fixture_api, scalar_report,
)
from tests.unit.experiments.test_r1_v19_analysis_observation import BUDGET


LIMITS = {'elapsed_s': 1800., 'peak_rss_bytes': 4 * 1024**3,
          'row_payload_bytes': 256 * 1024**2, 'case_artifact_bytes': 1024**3}


def fixture_api():
    """Give the old two-event fixture the explicit V12 duration contract."""
    case, state, api = v16_fixture_api()
    original_run = api['run']

    def run(prepared, observer):
        def observe(row):
            row['dt_s'] = 0. if row['time_s'] == 0. else 1.
            observer(row)
        return original_run(prepared, observe)

    api['run'] = run
    return case, state, api


def qualifying(report):
    report.update(source_unchanged=True, module_function_identity={'unchanged': True},
                  result_request_binding_passed=True, memory_profile_enabled=True)
    return report


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def exact_manifest(directory):
    manifest = json.loads((directory / 'ManifestV1.json').read_text())
    actual = {path.relative_to(directory).as_posix(): {
        'sha256': cases.sha(path), 'bytes': path.stat().st_size}
        for path in directory.rglob('*')
        if path.is_file() and path != directory / 'ManifestV1.json'}
    assert manifest == actual
    summary = json.loads((directory / 'SummaryV1.json').read_text())
    assert summary['cost']['case_artifact_bytes'] == sum(
        path.stat().st_size for path in directory.rglob('*') if path.is_file())
    return summary


def observed_small_run(directory, *, observer=None):
    directory.mkdir()
    case, state, api = fixture_api()
    report = runner.run_trajectory(directory, case, 'baseline', api, lambda: None,
                                   memory_observer=observer)
    return qualifying(report), state


def test_nested_analysis_callbacks_join_one_receipt_and_include_their_time(tmp_path, monkeypatch):
    directory = tmp_path / 'observed'
    directory.mkdir()
    case, state, api = fixture_api()
    native = NativeMemoryObserver(directory / 'NativeMemoryPhasesV1.jsonl')
    clock = {'now': 100.}
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock['now'])
    original_analysis = api['after_main']

    def analyze(prepared, result, replay, *, phase_observer):
        roots = {'prepared': prepared, 'raw_result': None, 'result': result,
                 'rows': result['accepted_steps'], 'persisted': None, 'replay': replay}
        phase_observer('analysis_outer', 'begin', roots)
        phase_observer('analysis_inner', 'begin', roots)
        report = original_analysis(prepared, result, replay)
        phase_observer('analysis_inner', 'end', roots)
        phase_observer('analysis_outer', 'end', roots)
        return report

    api['after_main_observed'] = analyze

    def timed_observer(phase, event, roots):
        assert set(roots) == ROOT_NAMES
        native(phase, event, roots)
        clock['now'] += .125

    try:
        report = runner.run_trajectory(directory, case, 'baseline', api, lambda: None,
                                       memory_observer=timed_observer)
    finally:
        native.close()
    measurement = report['memory_observation']
    rows = records(directory / 'NativeMemoryPhasesV1.jsonl')
    assert measurement['passed'] and native.summary()['passed']
    assert measurement['call_count'] == native.summary()['event_count'] == len(rows)
    assert measurement['elapsed_s'] == len(rows) * .125
    assert measurement['main_elapsed_s'] == sum(
        row['phase'] not in {'rows_release', 'after_main', 'analysis_outer', 'analysis_inner'}
        for row in rows) * .125
    nested = [(row['phase'], row['event']) for row in rows
              if row['phase'] in {'after_main', 'analysis_outer', 'analysis_inner'}]
    assert nested == [('after_main', 'begin'), ('analysis_outer', 'begin'),
                      ('analysis_inner', 'begin'), ('analysis_inner', 'end'),
                      ('analysis_outer', 'end'), ('after_main', 'end')]
    assert report['independent_analysis_elapsed_s'] >= .75
    assert state['calls'].count('after_main') == 1


def test_unobserved_runner_uses_the_original_callback_and_science_files(tmp_path):
    reference, _ = observed_small_run(tmp_path / 'reference')
    directory = tmp_path / 'disabled'
    directory.mkdir()
    case, state, api = fixture_api()

    def forbidden(*args, **kwargs):
        raise AssertionError('disabled diagnostics invoked the observed callback')

    api['after_main_observed'] = forbidden
    report = runner.run_trajectory(directory, case, 'baseline', api, lambda: None)
    assert scalar_report(report) == scalar_report(reference)
    assert 'memory_observation' not in report
    assert state['calls'].count('after_main') == 1
    for name in SCIENCE_FILES:
        assert (directory / name).read_bytes() == (tmp_path / 'reference' / name).read_bytes()


def test_tail_qualification_requires_close_and_final_seal_counts_every_event(tmp_path):
    directory = tmp_path / 'case'
    directory.mkdir()
    observer = NativeMemoryObserver(directory / 'NativeMemoryPhasesV1.jsonl')
    case, _, api = fixture_api()
    report = qualifying(runner.run_trajectory(directory, case, 'baseline', api,
        lambda: None, memory_observer=observer, extent_check=cases.extent))
    cases.update_qualification(report, 'baseline', None, memory_profile=True)
    assert not report['integrity_passed'] and not report['numerical_passed']
    tail = TailObservation(observer, report)
    with tail.phase('post_analysis_source_guard'):
        pass
    tail.event('case_finalization', 'begin')
    callback_calls = []

    def close_before_seal():
        callback_calls.append(True)
        assert not tail.closed
        tail.event('case_finalization', 'end')
        tail.finish()
        cases.update_qualification(report, 'baseline', None, memory_profile=True)
        report['final_receipt_refresh'] = 'closed-before-final-seal'

    cases.finalize_manifest(directory, report, LIMITS, before_final_seal=close_before_seal)
    assert callback_calls == [True]
    saved = exact_manifest(directory)
    assert saved == report
    assert report['integrity_passed'] and report['numerical_passed'] and report['baseline_usable']
    assert report['absolute_engineering_qualified']
    phase_rows = records(directory / 'NativeMemoryPhasesV1.jsonl')
    assert report['memory_observation']['call_count'] == report['memory_collector']['event_count'] == len(phase_rows)
    assert phase_rows[-1]['phase'] == 'case_finalization' and phase_rows[-1]['event'] == 'end'
    assert report['memory_collector']['closed'] and not report['memory_collector']['open_phases']
    assert set(report['memory_observation']['tail_phase_elapsed_s']) == {
        'post_analysis_source_guard', 'case_finalization'}
    snapshot = deepcopy(report)
    file_bytes = (directory / 'NativeMemoryPhasesV1.jsonl').read_bytes()
    tail.finish()
    tail.event('late', 'begin')
    assert report == snapshot and (directory / 'NativeMemoryPhasesV1.jsonl').read_bytes() == file_bytes


def test_finalize_none_keeps_the_original_unobserved_seal(tmp_path):
    initial = {'cost': {'elapsed_s': 1., 'peak_rss_bytes': 100,
                        'accepted_row_payload_bytes': 100}}
    for name in ('original', 'explicit_none'):
        directory = tmp_path / name
        directory.mkdir()
        (directory / 'Data.json').write_text('{"synthetic":true}\n')
        report = deepcopy(initial)
        if name == 'original':
            cases.finalize_manifest(directory, report, LIMITS)
        else:
            cases.finalize_manifest(directory, report, LIMITS, before_final_seal=None)
        exact_manifest(directory)
    for path in (tmp_path / 'original').iterdir():
        assert path.read_bytes() == (tmp_path / 'explicit_none' / path.name).read_bytes()


@pytest.mark.parametrize('failure_type', [ValueError, KeyboardInterrupt])
def test_tail_phase_preserves_primary_error_and_balances_native_events(tmp_path, failure_type):
    observer = NativeMemoryObserver(tmp_path / 'Phases.jsonl')
    summary = {'memory_observation': {'enabled': True, 'passed': True,
                                     'call_count': 0, 'elapsed_s': 0., 'error': None}}
    tail = TailObservation(observer, summary)
    try:
        with pytest.raises(failure_type, match='original source-guard failure'):
            with tail.phase('source_guard'):
                raise failure_type('original source-guard failure')
    finally:
        tail.finish()
    assert [(row['phase'], row['event']) for row in records(tmp_path / 'Phases.jsonl')] == [
        ('source_guard', 'begin'), ('source_guard', 'error')]
    assert summary['memory_collector']['closed'] and not summary['memory_collector']['open_phases']


@pytest.mark.parametrize('failure_type', [RuntimeError, KeyboardInterrupt])
def test_failed_error_observation_cannot_replace_primary_source_error(tmp_path, failure_type):
    native = NativeMemoryObserver(tmp_path / 'Phases.jsonl')
    original_rss = native._rss
    calls = {'rss': 0}

    def bad_rss():
        calls['rss'] += 1
        if calls['rss'] == 2:
            raise failure_type('secondary collector error')
        return original_rss()

    native._rss = bad_rss
    summary = {'memory_observation': {'passed': True, 'call_count': 0, 'elapsed_s': 0.}}
    tail = TailObservation(native, summary)
    try:
        with pytest.raises(ValueError, match='original source-guard failure'):
            with tail.phase('source_guard'):
                raise ValueError('original source-guard failure')
    finally:
        tail.finish()
    assert not summary['memory_observation']['passed']
    assert summary['memory_observation']['error'] is not None
    assert not cases.memory_observation_passed(summary, enabled=True)
    calls_before = calls['rss']
    tail.event('later', 'begin')
    assert calls['rss'] == calls_before


def test_existing_diagnostic_failure_survives_tail_close(tmp_path):
    native = NativeMemoryObserver(tmp_path / 'Phases.jsonl')
    primary = {'type': 'ValueError', 'message': 'original runner observation failure'}
    summary = {'memory_observation': {'passed': False, 'call_count': 7,
                                     'elapsed_s': .4, 'main_elapsed_s': .3, 'error': primary}}
    tail = TailObservation(native, summary)
    with tail.phase('tail'):
        pass
    tail.finish()
    assert summary['memory_observation']['error'] == primary
    assert not summary['memory_observation']['passed']
    assert summary['memory_observation']['call_count'] == 9
    assert summary['memory_observation']['main_elapsed_s'] == .3


def synthetic_case_shell(tmp_path, monkeypatch, *, guard_failure=None, observer_error=None,
                         observer_event=('case_finalization', 'error'), close_failure=None):
    """Exercise the real outer run/finalization, replacing only physical work."""
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_checkout as checkout
    from perovskite_sim.experiments import one_dimensional_mechanism_r1 as physical
    from perovskite_sim.models import config_loader
    from scripts import verify_r1_v7_precision as freeze
    from scripts import r1_v16_memory as memory

    for key in cases.THREAD_KEYS:
        monkeypatch.setenv(key, '1')
    material = {'fixture': b'synthetic fixture\n', 'reference': b'{}\n'}
    specifications = {name: {'path': name, 'sha256': hashlib.sha256(raw).hexdigest()}
                      for name, raw in material.items()}
    monkeypatch.setattr(cases, 'INPUTS', specifications)
    monkeypatch.setattr(checkout, 'require_r1_checkout', lambda **kwargs:
                        SimpleNamespace(read_bytes=lambda path: material[path]))
    monkeypatch.setattr(config_loader, 'load_device_from_yaml', lambda path: object())
    monkeypatch.setattr(physical, 'build_r1_material', lambda *args: (object(), object()))
    monkeypatch.setattr(freeze, 'freeze_healthy_material', lambda *args, **kwargs: {'synthetic': True})
    monkeypatch.setattr(cases, 'function_snapshot', lambda: {})
    monkeypatch.setattr(cases, 'check_functions', lambda saved: {'unchanged': True})
    state = {'source_calls': 0, 'native': None, 'tail': None}

    def tracked_tail(observer, summary):
        state['tail'] = TailObservation(observer, summary)
        return state['tail']

    monkeypatch.setattr(tail_module, 'TailObservation', tracked_tail)

    def source_snapshot(commit):
        state['source_calls'] += 1
        if guard_failure is not None and state['source_calls'] >= 4:
            raise guard_failure('bounded final source-guard failure')
        return {'synthetic_source': True}

    monkeypatch.setattr(cases, 'source_snapshot', source_snapshot)
    original_trajectory = runner.run_trajectory

    def small_trajectory(directory, case, mode, api, source_guard, **kwargs):
        little_case, _, little_api = fixture_api()
        result = original_trajectory(directory, little_case, 'baseline', little_api,
            source_guard, memory_observer=kwargs.get('memory_observer'), extent_check=cases.extent)
        result['result_request_binding_passed'] = True
        return result

    monkeypatch.setattr(cases, 'run_trajectory', small_trajectory)
    class FaultObserver(NativeMemoryObserver):
        def __init__(self, path):
            super().__init__(path)
            state['native'] = self

        def __call__(self, phase, event, roots):
            if observer_error is not None and (phase, event) == observer_event:
                original_rss = self._rss
                try:
                    self._rss = lambda: (_ for _ in ()).throw(observer_error('secondary final observer failure'))
                    return super().__call__(phase, event, roots)
                finally:
                    self._rss = original_rss
            return super().__call__(phase, event, roots)

        def close(self):
            was_closed = self._closed
            try:
                super().close()
            finally:
                if close_failure is not None and not was_closed:
                    raise close_failure('bounded close failure')

    monkeypatch.setattr(memory, 'NativeMemoryObserver', FaultObserver)
    budget = deepcopy(BUDGET)
    budget.update(schema='R1V12PrecisionBudgetV1', engineering={
        'relative_limits': cases.RELATIVE_LIMITS, 'absolute_limits': None})
    budget_path = tmp_path / 'Budget.json'
    cases.write(budget_path, budget)
    request = cases.make_request('D_N32_F0p01_T2', cases.sha(budget_path), LIMITS)
    request_path = tmp_path / 'Request.json'
    cases.write(request_path, request)
    args = SimpleNamespace(output=tmp_path / 'Case', request_file=request_path,
        request_sha256=cases.sha(request_path), budget_file=budget_path,
        budget_sha256=cases.sha(budget_path), expected_commit='a' * 40,
        source_sha256='b' * 64, mode='baseline', baseline_summary=None,
        baseline_sha256=None, memory_profile=True)
    return args, state


def test_complete_outer_shell_seals_closed_observation_and_qualification(tmp_path, monkeypatch):
    args, _ = synthetic_case_shell(tmp_path, monkeypatch)
    assert cases.run(args) == 0
    report = exact_manifest(args.output)
    phase_rows = records(args.output / 'NativeMemoryPhasesV1.jsonl')
    assert report['integrity_passed'] and report['numerical_passed']
    assert report['memory_observation']['call_count'] == report['memory_collector']['event_count'] == len(phase_rows)
    assert phase_rows[-1]['phase'] == 'case_finalization' and phase_rows[-1]['event'] == 'end'


def test_unobserved_outer_shell_has_no_observer_requirements(tmp_path, monkeypatch):
    args, _ = synthetic_case_shell(tmp_path, monkeypatch)
    args.memory_profile = False
    assert cases.run(args) == 0
    report = exact_manifest(args.output)
    assert report['integrity_passed'] and report['numerical_passed']
    assert 'memory_observation' not in report and 'memory_collector' not in report
    assert not (args.output / 'NativeMemoryPhasesV1.jsonl').exists()


@pytest.mark.parametrize('failure_type', [ValueError, KeyboardInterrupt])
def test_final_source_guard_failure_retains_evidence_and_fails_closed(tmp_path, monkeypatch, failure_type):
    args, state = synthetic_case_shell(tmp_path, monkeypatch, guard_failure=failure_type)
    assert cases.run(args) == 1
    report = exact_manifest(args.output)
    assert not report['integrity_passed'] and not report['numerical_passed']
    assert report['runner_error']['type'] == failure_type.__name__
    assert report['runner_error']['message'] == 'bounded final source-guard failure'
    assert report['memory_collector']['closed']
    assert state['source_calls'] >= 5
    assert (args.output / 'ResultV1.json').is_file()


@pytest.mark.parametrize('failure_type', [OSError, KeyboardInterrupt])
def test_finalization_failure_closes_collector_and_propagates_primary(tmp_path, monkeypatch, failure_type):
    args, _ = synthetic_case_shell(tmp_path, monkeypatch)
    original_write = cases.write

    def broken(path, value):
        if path.name == 'ManifestV1.json':
            raise failure_type('original finalization failure')
        return original_write(path, value)

    monkeypatch.setattr(cases, 'write', broken)
    with pytest.raises(failure_type, match='original finalization failure'):
        cases.run(args)
    phase_rows = records(args.output / 'NativeMemoryPhasesV1.jsonl')
    assert phase_rows[-1]['phase'] == 'case_finalization' and phase_rows[-1]['event'] == 'error'
    assert (args.output / 'ResultV1.json').is_file()
    assert not (args.output / 'ManifestV1.json').exists()


@pytest.mark.parametrize('observer_error', [RuntimeError, KeyboardInterrupt])
def test_finalization_primary_error_survives_failed_terminal_observation(tmp_path, monkeypatch, observer_error):
    args, state = synthetic_case_shell(tmp_path, monkeypatch, observer_error=observer_error)
    original_write = cases.write

    def broken(path, value):
        if path.name == 'ManifestV1.json':
            raise OSError('original finalization failure')
        return original_write(path, value)

    monkeypatch.setattr(cases, 'write', broken)
    with pytest.raises(OSError, match='original finalization failure'):
        cases.run(args)
    collector = state['native'].summary()
    assert collector['closed'] and not collector['passed']
    assert collector['failure'] is not None
    assert not (args.output / 'ManifestV1.json').exists()


def test_normal_tail_observer_failure_keeps_science_but_cannot_qualify(tmp_path, monkeypatch):
    args, state = synthetic_case_shell(tmp_path, monkeypatch, observer_error=RuntimeError,
        observer_event=('post_analysis_source_guard', 'begin'))
    assert cases.run(args) == 1
    report = exact_manifest(args.output)
    assert report['execution_status'] == 'completed' and report['four_predicates_passed']
    assert not report['integrity_passed'] and not report['numerical_passed']
    assert not report['memory_observation']['passed'] and not report['memory_collector']['passed']
    assert report['memory_observation']['error']['type'] == 'MemoryObservationError'
    assert state['tail'].closed
    rows = records(args.output / 'NativeMemoryPhasesV1.jsonl')
    assert len(rows) == report['memory_collector']['event_count']
    assert report['memory_observation']['call_count'] == len(rows) + 1


@pytest.mark.parametrize('failure_type', [RuntimeError, KeyboardInterrupt])
def test_close_failure_is_recorded_and_cannot_grant_qualification(tmp_path, monkeypatch, failure_type):
    args, state = synthetic_case_shell(tmp_path, monkeypatch, close_failure=failure_type)
    if failure_type is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt, match='bounded close failure'):
            cases.run(args)
        report = state['tail'].summary
    else:
        assert cases.run(args) == 1
        report = exact_manifest(args.output)
    assert state['tail'].closed and state['native'].summary()['closed']
    assert not report['integrity_passed'] and not report['numerical_passed']
    assert not report['memory_observation']['passed']
    assert report['memory_observation']['error'] == {
        'type': failure_type.__name__, 'message': 'bounded close failure'}
    assert not cases.memory_observation_passed(report, enabled=True)


def test_primary_finalization_failure_survives_close_interruption(tmp_path, monkeypatch):
    args, state = synthetic_case_shell(tmp_path, monkeypatch, close_failure=KeyboardInterrupt)
    original_write = cases.write

    def broken(path, value):
        if path.name == 'ManifestV1.json':
            raise OSError('original finalization failure')
        return original_write(path, value)

    monkeypatch.setattr(cases, 'write', broken)
    with pytest.raises(OSError, match='original finalization failure'):
        cases.run(args)
    assert state['tail'].closed
    assert state['tail'].summary['memory_observation']['error'] == {
        'type': 'KeyboardInterrupt', 'message': 'bounded close failure'}
    assert not state['tail'].summary['memory_observation']['passed']
