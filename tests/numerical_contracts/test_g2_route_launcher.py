"""Real process/resource tests with synthetic children only; no scientific imports."""
from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from scripts.benchmarks import g2_route_launcher as launcher


CHILD = r'''
import json, os, signal, subprocess, sys, time
from pathlib import Path
from scripts.benchmarks.g2_route_launcher import inherited_channel, THREAD_VARIABLES
mode, directory, slots, channel = sys.argv[1:]
root = Path(directory)
with inherited_channel(slots, int(channel)):
    root.mkdir()
    (root/'Started.json').write_text(json.dumps({
        'pid': os.getpid(), 'pgid': os.getpgrp(), 'cwd': str(Path.cwd()),
        'thread_environment': {key: os.environ.get(key) for key in THREAD_VARIABLES},
        'dont_write_bytecode': os.environ.get('PYTHONDONTWRITEBYTECODE')}))
    if mode == 'normal':
        (root/'payload').write_text('synthetic payload')
        time.sleep(.08)
    elif mode == 'peer':
        for i in range(30):
            (root/'Progress.json').write_text(json.dumps({'step': i}))
            time.sleep(.05)
    elif mode == 'artifact':
        (root/'large.bin').write_bytes(b'a'*(2*1024*1024))
        time.sleep(60)
    elif mode == 'rss':
        code = 'import time; data=bytearray(80*1024*1024); time.sleep(60)'
        children = [subprocess.Popen([sys.executable, '-c', code]) for _ in range(2)]
        (root/'Children.json').write_text(json.dumps([p.pid for p in children]))
        time.sleep(60)
    elif mode == 'grandchild':
        grand = "import os,signal,time,json; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(%r).write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp()})); time.sleep(60)" % str(root/'Grandchild.json')
        middle = 'import os,subprocess,sys,time,json; from pathlib import Path; p=subprocess.Popen([sys.executable,"-c",%r]); Path(%r).write_text(json.dumps({"pid":os.getpid(),"pgid":os.getpgrp(),"child":p.pid})); time.sleep(60)' % (grand, str(root/'Middle.json'))
        subprocess.Popen([sys.executable, '-c', middle])
        time.sleep(60)
    elif mode == 'ignoreterm':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (root/'IgnoresTerm').touch()
        time.sleep(60)
    elif mode == 'wall' or mode == 'cancel':
        time.sleep(60)
    elif mode == 'nonzero':
        sys.exit(7)
    else:
        raise RuntimeError(mode)
'''


def make_plan(root, modes, *, same_slot=False, wall_s=4.0, rss_bytes=512*1024**2,
              artifact_bytes=8*1024**2):
    root.mkdir(exist_ok=True)
    slots = root/'slots'
    slots.mkdir(exist_ok=True)
    cases = []
    for index, mode in enumerate(modes):
        case_id = f'synthetic_{index}_{mode}'
        channel = 0 if same_slot else index
        directory = root/case_id
        supervision = {
            'poll_s': .02, 'term_grace_s': .15, 'kill_wait_s': .6, 'receipt_reserve_s': .25,
            'rss_bytes': rss_bytes, 'artifact_bytes': artifact_bytes,
            'closure_path': str(root/(case_id+'.closure.json')),
            'cancel_path': str(root/(case_id+'.cancel')),
        }
        cases.append({'id': case_id, 'execution': {
            'directory': str(directory), 'channel': channel, 'wall_s': wall_s,
            'command': [sys.executable, '-c', CHILD, mode, str(directory), str(slots), str(channel)],
            'supervision': supervision}})
    plan = {'schema': 'solarlab.g2-route.run-plan.v1', 'status': 'admitted',
            'purpose': 'synthetic launcher CI only; no scientific admission',
            'authority': {'scientific_execution_authorized': False},
            'resources': {'WS1_channels': 4, 'threads_per_case': 1,
                          'rss_bytes_per_case': rss_bytes, 'combined_persisted_case_bytes': artifact_bytes},
            'artifacts': {'channel_directory': str(slots)}, 'cases': cases}
    path = root/'RunPlan.synthetic.json'
    path.write_text(json.dumps(plan))
    return path, cases


def start(path, case):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(launcher.REPOSITORY)
    return subprocess.Popen(
        [sys.executable, '-m', 'scripts.benchmarks.g2_route_launcher', 'run',
         '--plan', str(path), '--case', case['id']],
        cwd=launcher.REPOSITORY, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )


def finish(process, case):
    stdout, stderr = process.communicate(timeout=12)
    receipt = Path(case['execution']['supervision']['closure_path'])
    assert receipt.exists(), (process.returncode, stdout, stderr)
    result = json.loads(receipt.read_bytes())
    assert result['case_id'] == case['id']
    assert result['pgid'] != os.getpgrp()
    assert result['closure']['all_named_owned_processes_and_groups_absent'], result
    assert result['closure']['process_group_absent']
    assert all(row['pid_absent'] and row['same_id_process_group_absent'] for row in result['closure']['checks'])
    assert all(launcher._absent(pid) for pid in result['observed_pids'])
    assert result['scientific_status'] == 'not_evaluated' and result['scientific_pass'] is None
    assert result['wall_s'] < case['execution']['wall_s'], result
    returned = json.loads(stdout)
    assert returned['wall_through_receipt_s'] < case['execution']['wall_s']
    assert result['receipt_bytes'] == receipt.stat().st_size
    actual_bytes = launcher._artifact_size((Path(case['execution']['directory']), receipt,
                                           receipt.with_suffix('.log')), time.monotonic()+2)
    assert result['artifact_bytes'] == actual_bytes
    assert result['peak_artifact_bytes'] >= actual_bytes
    assert process.returncode == (0 if result['process_outcome'] == 'completed' else 1)
    return result


def wait_for(path, timeout=4):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        if path.exists():
            return
        time.sleep(.01)
    raise AssertionError('missing readiness: '+str(path))


@contextmanager
def running(path, cases):
    processes = [start(path, case) for case in cases]
    try:
        yield processes
    finally:
        # Every test failure still asks the owned supervisors to clean up.
        for process, case in zip(processes, cases):
            if process.poll() is None:
                Path(case['execution']['supervision']['cancel_path']).touch()
                process.send_signal(signal.SIGTERM)
        for process in processes:
            if process.poll() is None:
                process.communicate(timeout=12)


def test_normal_exit_and_inherited_slot_environment(tmp_path):
    path, cases = make_plan(tmp_path, ['normal'])
    with running(path, cases) as processes:
        result = finish(processes[0], cases[0])
    assert result['process_outcome'] == 'completed' and result['exit_code'] == 0
    assert result['signals_sent'] == []
    started = json.loads((Path(cases[0]['execution']['directory'])/'Started.json').read_bytes())
    assert started['pid'] == started['pgid'] == result['pid']
    assert started['cwd'] == str(launcher.REPOSITORY)
    assert set(started['thread_environment'].values()) == {'1'}
    assert started['dont_write_bytecode'] == '1'


def test_outer_wall_stops_only_the_owned_group(tmp_path):
    path, cases = make_plan(tmp_path, ['wall'], wall_s=1.8)
    with running(path, cases) as processes:
        result = finish(processes[0], cases[0])
    assert result['process_outcome'] == 'wall_limit'
    assert result['signals_sent'][0]['signal'] == 'SIGTERM'
    assert result['exit_signal'] == signal.SIGTERM


def test_group_rss_is_sum_of_children_not_only_leader(tmp_path):
    cap = 128*1024**2
    path, cases = make_plan(tmp_path, ['rss'], rss_bytes=cap)
    with running(path, cases) as processes:
        result = finish(processes[0], cases[0])
    assert result['process_outcome'] == 'rss_limit', result
    assert result['peak_group_rss_bytes'] > cap
    witness = result['rss_peak_sample']
    assert len(witness['members']) >= 3
    assert max(row['rss_bytes'] for row in witness['members']) < cap


def test_artifact_limit_preserves_output_and_closes_group(tmp_path):
    path, cases = make_plan(tmp_path, ['artifact'], artifact_bytes=1024**2)
    with running(path, cases) as processes:
        result = finish(processes[0], cases[0])
    assert result['process_outcome'] == 'artifact_limit'
    assert result['peak_artifact_bytes'] > 1024**2
    assert (Path(cases[0]['execution']['directory'])/'large.bin').stat().st_size == 2*1024**2


@pytest.mark.parametrize('via_signal', [False, True])
def test_cancel_one_case_leaves_peer_running_to_normal_completion(tmp_path, via_signal):
    path, cases = make_plan(tmp_path, ['cancel', 'peer'])
    with running(path, cases) as processes:
        for case in cases:
            wait_for(Path(case['execution']['directory'])/'Started.json')
        if via_signal:
            processes[0].send_signal(signal.SIGTERM)
        else:
            Path(cases[0]['execution']['supervision']['cancel_path']).touch()
        stopped = finish(processes[0], cases[0])
        assert processes[1].poll() is None
        peer = finish(processes[1], cases[1])
    assert stopped['process_outcome'] == 'cancelled'
    assert peer['process_outcome'] == 'completed' and peer['signals_sent'] == []
    assert stopped['pgid'] != peer['pgid']
    assert json.loads((Path(cases[1]['execution']['directory'])/'Progress.json').read_bytes())['step'] == 29


def test_long_blocking_call_ignoring_term_requires_sigkill(tmp_path):
    path, cases = make_plan(tmp_path, ['ignoreterm'], wall_s=1.8)
    with running(path, cases) as processes:
        result = finish(processes[0], cases[0])
    assert result['process_outcome'] == 'wall_limit'
    assert [row['signal'] for row in result['signals_sent']] == ['SIGTERM', 'SIGKILL']
    assert result['exit_signal'] == signal.SIGKILL


def test_grandchild_in_same_group_is_gone_after_closeout(tmp_path):
    path, cases = make_plan(tmp_path, ['grandchild'], wall_s=1.8)
    with running(path, cases) as processes:
        wait_for(Path(cases[0]['execution']['directory'])/'Grandchild.json')
        result = finish(processes[0], cases[0])
    directory = Path(cases[0]['execution']['directory'])
    middle = json.loads((directory/'Middle.json').read_bytes())
    grandchild = json.loads((directory/'Grandchild.json').read_bytes())
    assert middle['pgid'] == grandchild['pgid'] == result['pgid']
    assert {middle['pid'], grandchild['pid']} <= set(result['observed_pids'])
    assert 'SIGKILL' in [row['signal'] for row in result['signals_sent']]


def test_slot_conflict_refuses_without_starting_second_case(tmp_path):
    path, cases = make_plan(tmp_path, ['cancel', 'normal'], same_slot=True)
    with running(path, cases[:1]) as processes:
        wait_for(Path(cases[0]['execution']['directory'])/'Started.json')
        conflict = start(path, cases[1])
        stdout, stderr = conflict.communicate(timeout=4)
        assert conflict.returncode == 2 and 'channel occupied' in stderr
        assert not Path(cases[1]['execution']['directory']).exists()
        assert not Path(cases[1]['execution']['supervision']['closure_path']).exists()
        Path(cases[0]['execution']['supervision']['cancel_path']).touch()
        result = finish(processes[0], cases[0])
    assert result['process_outcome'] == 'cancelled'


def test_child_nonzero_is_not_normal_completion(tmp_path):
    path, cases = make_plan(tmp_path, ['nonzero'])
    with running(path, cases) as processes:
        result = finish(processes[0], cases[0])
    assert result['process_outcome'] == 'child_failed' and result['exit_code'] == 7


@pytest.mark.parametrize('field,bad', [('rss_bytes', True), ('artifact_bytes', 0), ('poll_s', -1), ('term_grace_s', float('nan'))])
def test_all_resource_values_must_be_explicit_valid_plan_fields(tmp_path, field, bad):
    path, cases = make_plan(tmp_path, ['normal'])
    execution = cases[0]['execution']
    execution['supervision'][field] = bad
    with pytest.raises(launcher.LauncherError):
        launcher.Limits.from_execution(execution)
    del execution['supervision'][field]
    with pytest.raises(KeyError):
        launcher.Limits.from_execution(execution)


def test_inherited_channel_never_reopens_or_unlocks_slot(tmp_path, monkeypatch):
    with launcher._owned_channel(tmp_path, 0) as (fd, quarantine):
        with monkeypatch.context() as child:
            child.setenv(launcher.CHANNEL_FD_ENV, str(fd))
            def forbidden(*args, **kwargs):
                raise AssertionError('consumer reopened or relocked the inherited slot')
            child.setattr(Path, 'open', forbidden)
            child.setattr(launcher.fcntl, 'flock', forbidden)
            with launcher.inherited_channel(tmp_path, 0) as inherited:
                assert inherited == fd


def test_inherited_channel_rejects_wrong_inode_and_missing_fd(tmp_path, monkeypatch):
    with launcher._owned_channel(tmp_path, 0) as (fd, quarantine):
        (tmp_path/'1.lock').touch()
        monkeypatch.setenv(launcher.CHANNEL_FD_ENV, str(fd))
        with pytest.raises(launcher.LauncherError, match='inode'):
            with launcher.inherited_channel(tmp_path, 1):
                pass
        monkeypatch.delenv(launcher.CHANNEL_FD_ENV)
        with pytest.raises(launcher.LauncherError, match='missing'):
            with launcher.inherited_channel(tmp_path, 0):
                pass


def test_quarantined_slot_never_starts_a_new_case(tmp_path):
    path, cases = make_plan(tmp_path, ['normal'])
    (tmp_path/'slots/0.quarantine.json').write_text('{"reason":"manufactured unresolved closure"}')
    process = start(path, cases[0])
    stdout, stderr = process.communicate(timeout=4)
    assert process.returncode == 2 and 'quarantined' in stderr
    assert not Path(cases[0]['execution']['directory']).exists()


def test_supervision_outputs_are_never_overwritten(tmp_path):
    path, cases = make_plan(tmp_path, ['normal'])
    receipt = Path(cases[0]['execution']['supervision']['closure_path'])
    original = b'preserved prior failure'
    receipt.write_bytes(original)
    process = start(path, cases[0])
    stdout, stderr = process.communicate(timeout=4)
    assert process.returncode == 2 and 'already exists' in stderr
    assert receipt.read_bytes() == original
    assert not Path(cases[0]['execution']['directory']).exists()


def test_case_cancel_and_artifact_paths_are_independent(tmp_path):
    path, cases = make_plan(tmp_path, ['normal', 'normal'])
    plan = json.loads(path.read_bytes())
    plan['cases'][1]['execution']['supervision']['cancel_path'] = cases[0]['execution']['supervision']['cancel_path']
    path.write_text(json.dumps(plan))
    with pytest.raises(launcher.LauncherError, match='paths overlap'):
        launcher.load_case(path, cases[0]['id'], 'run')


def test_empty_inventory_alone_cannot_certify_process_closure(monkeypatch):
    monkeypatch.setattr(launcher, '_absent', lambda *args, **kwargs: False)
    evidence = launcher._closure(123456, {123456}, 0, {})
    assert evidence['all_named_owned_processes_and_groups_absent'] is False
    assert evidence['process_group_absent'] is False


def test_zero_returncode_with_a_remaining_member_is_not_closed(monkeypatch):
    monkeypatch.setattr(launcher, '_absent', lambda *args, **kwargs: True)
    evidence = launcher._closure(123456, {123456, 123457}, 0,
        {123457: {'pid': 123457, 'pgid': 123456, 'ppid': 1, 'rss_bytes': 0, 'state': 'Z'}})
    assert evidence['all_named_owned_processes_and_groups_absent'] is False
    assert evidence['remaining_group_members'][0]['state'] == 'Z'


def make_quarantine(path, case, process):
    plan = json.loads(path.read_bytes())
    marker = Path(plan['artifacts']['channel_directory'])/f"{case['execution']['channel']}.quarantine.json"
    marker.write_text(json.dumps({
        'schema': 'solarlab.g2-route.channel-quarantine.v1', 'case_id': case['id'],
        'channel': case['execution']['channel'], 'at': datetime.now(timezone.utc).isoformat(),
        'pgid': process.pid, 'observed_pids': [process.pid],
        'plan_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'closure_path': case['execution']['supervision']['closure_path'],
        'reason': 'synthetic quarantine fixture; not a scientific execution',
    }))
    return marker


def write_release_record(path, verification):
    path.write_text(json.dumps({
        'schema': 'solarlab.g2-route.quarantine-release-record.v1',
        'actor': 'coordinator', 'purpose': 'synthetic release-guard test only',
        'at': datetime.now(timezone.utc).isoformat(),
        'verification_command': verification['command'], 'verification_output': verification,
    }))


def test_quarantine_verification_is_read_only_and_reports_live_command(tmp_path):
    path, cases = make_plan(tmp_path, ['normal'])
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    try:
        marker = make_quarantine(path, cases[0], process)
        original = marker.read_bytes()
        result = launcher.verify_quarantine(path, cases[0]['id'])
        assert result['all_named_owned_processes_and_groups_absent'] is False
        assert marker.read_bytes() == original
        details = result['remaining_commands']['processes']
        assert any(row['pid'] == process.pid and row['pgid'] == process.pid and row['command_line'] for row in details)
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=4)


def test_quarantine_release_rechecks_live_pids_instead_of_trusting_record(tmp_path):
    path, cases = make_plan(tmp_path, ['normal'])
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    try:
        marker = make_quarantine(path, cases[0], process)
        claimed = copy.deepcopy(launcher.verify_quarantine(path, cases[0]['id']))
        claimed['all_named_owned_processes_and_groups_absent'] = True
        record = tmp_path/'Release.synthetic.json'
        write_release_record(record, claimed)
        with pytest.raises(launcher.LauncherError, match='remain'):
            launcher.release_quarantine(path, cases[0]['id'], record)
        assert marker.exists() and process.poll() is None
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=4)


def test_quarantine_release_requires_record_and_preserves_failed_receipt(tmp_path):
    path, cases = make_plan(tmp_path, ['normal'])
    process = subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
    marker = make_quarantine(path, cases[0], process)
    process.wait(timeout=4)
    failure = Path(cases[0]['execution']['supervision']['closure_path'])
    original_failure = b'{"process_outcome":"closure_failure","synthetic_only":true}'
    failure.write_bytes(original_failure)
    with pytest.raises(launcher.LauncherError, match='record is required'):
        launcher.release_quarantine(path, cases[0]['id'], None)
    verified = launcher.verify_quarantine(path, cases[0]['id'])
    assert verified['all_named_owned_processes_and_groups_absent']
    record = tmp_path/'Release.synthetic.json'
    write_release_record(record, verified)
    original_record = record.read_bytes()
    result = launcher.release_quarantine(path, cases[0]['id'], record)
    assert not marker.exists() and record.read_bytes() == original_record
    assert failure.read_bytes() == original_failure
    assert result['original_case_outcome'] == 'closure_failure'
    assert result['fresh_verification']['all_named_owned_processes_and_groups_absent']


def test_changed_quarantine_cannot_be_released_using_old_record(tmp_path):
    path, cases = make_plan(tmp_path, ['normal'])
    process = subprocess.Popen([sys.executable, '-c', 'pass'], start_new_session=True)
    marker = make_quarantine(path, cases[0], process)
    process.wait(timeout=4)
    verified = launcher.verify_quarantine(path, cases[0]['id'])
    record = tmp_path/'Release.synthetic.json'
    write_release_record(record, verified)
    marker.write_bytes(marker.read_bytes()+b'\n')
    with pytest.raises(launcher.LauncherError, match='changed since'):
        launcher.release_quarantine(path, cases[0]['id'], record)
    assert marker.exists()
