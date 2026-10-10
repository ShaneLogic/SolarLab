"""Real loopback fetch and supervised attempts; RunStore infrastructure only.

The production stored panel is exercised in jsdom with the actual HTTP client.
No model, trajectory, scientific result, browser or production route is run.
The existing HTTP fixture owns and positively closes its reader and threads.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import struct
import subprocess
import time

import pytest

import execution_workers as workers
from solarlab.experiments.execution import ProcessSupervisor
from solarlab.experiments.worker import WorkerBinding
from solarlab.io.artifacts import bytes_artifact
from solarlab.io.run_store import RunStore
from test_event_service import _HTTPServer, _receipt
from test_execution import REGISTRY, SCHEMA, assert_reaped, limits, ready, release, request


REPOSITORY = Path(__file__).resolve().parents[2]
IDENTITY = {"H_input": "run-client:infrastructure-input",
            "H_physics": "run-client:no-physical-model",
            "H_execution": "run-client:infrastructure-only"}


def _node(script: str, directory: Path) -> dict:
    node = shutil.which("node")
    assert node is not None, "the locked web toolchain needs Node"
    started = time.perf_counter()
    result = subprocess.run(
        [node, "--input-type=module", "-"], input=script, text=True,
        cwd=directory, capture_output=True, timeout=20, check=False,
    )
    elapsed = time.perf_counter() - started
    _receipt({"kind": "run_client_node", "command": [node, "--input-type=module", "-"],
              "wall_seconds": elapsed, "exit_code": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr,
              "real_browser": False, "scientific_fixture": False})
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def _claim(store: RunStore, run_id: str):
    run = store.get_run(run_id)
    return store.claim(run_id, expected_attempt_id=run["current_attempt_id"],
                       expected_generation=run["generation"], owner_id="client-fixture")


def test_typescript_client_reads_actual_local_stored_run(tmp_path):
    # Both the database and the transpiled test module stay in this new temp dir.
    module = tmp_path / "run-client.mjs"
    compiler = REPOSITORY / "web/node_modules/typescript/lib/typescript.js"
    assert compiler.is_file(), "use the installed locked TypeScript toolchain"
    prepare = f"""
import ts from {json.dumps(compiler.as_uri())};
import {{ readFile, writeFile }} from 'node:fs/promises';
const source = await readFile({json.dumps(str(REPOSITORY / 'web/src/run-client.ts'))}, 'utf8');
const result = ts.transpileModule(source, {{ compilerOptions: {{ target: ts.ScriptTarget.ES2023, module: ts.ModuleKind.ES2022 }} }});
await writeFile({json.dumps(str(module))}, result.outputText);
"""
    with RunStore((tmp_path / "store").resolve()) as store:
        run_id = "run-client-fixture"
        metadata = {"zero": 0, "negative_zero": -0.0, "null_value": None,
                    "large_integer": 18446744073709551617,
                    "qualification": "infrastructure_only", "enabled": False}
        store.enqueue(run_id, opaque_identity=IDENTITY, input_metadata=metadata)
        old = _claim(store, run_id)
        store.complete(old, state="failed", result={"reason": "old infrastructure failure"})
        store.retry(run_id, expected_attempt_id=old.attempt_id, expected_generation=old.generation)
        token = _claim(store, run_id)
        store.enqueue("foreign", opaque_identity=IDENTITY, input_metadata={})
        store.append_event(token, event_id="zero", payload={"value": 0, "null_value": None})
        payload = b"\x00\x80\xffexact-words\x00"
        artifact = store.publish(token, "bytes", bytes_artifact(payload, metadata={"zero": 0, "missing_reason": None}))
        result = {"zero": 0, "tau": None, "reason": None, "qualification": "not_evaluated"}
        store.complete(token, state="succeeded", result=result, artifacts=[artifact["artifact_id"]])
        store.enqueue("still-running", opaque_identity=IDENTITY, input_metadata={})
        running = _claim(store, "still-running")
        expected = store.events(run_id=run_id)
        selection = {
            "storeId": store.store_id, "target": {"runId": run_id, "attemptId": token.attempt_id},
            "oldAttemptId": old.attempt_id,
            "running": {"runId": running.run_id, "attemptId": running.attempt_id},
            "artifact": {"artifact_id": artifact["artifact_id"], "manifest_sha256": artifact["manifest_sha256"]},
            "payload": list(payload), "identity": IDENTITY, "result": result,
            "sequences": [event["sequence"] for event in expected],
            "attemptSequences": [event["sequence"] for event in expected if event["attempt_id"] == token.attempt_id],
        }
        with _HTTPServer(store.root) as server:
            selection["endpoint"] = f"http://127.0.0.1:{server.port}/"
            script = prepare + f"""
import assert from 'node:assert/strict';
const {{ StoredRunClient }} = await import({json.dumps(module.as_uri())});
const fixture = {json.dumps(selection)};
const client = new StoredRunClient(fixture.endpoint, fixture.storeId);
const run = await client.getRun(fixture.target.runId);
assert.deepEqual(run.value.request.identity, fixture.identity);
const input = run.value.request.input_metadata;
assert.equal(input.large_integer, 18446744073709551617n);
assert(Object.is(input.negative_zero, -0));
assert.equal(input.zero, 0); assert.equal(input.null_value, null);
assert.equal(Object.hasOwn(input, 'omitted'), false);
assert.equal(typeof run.value.created_ns, 'bigint');
const attempt = await client.getAttempt(fixture.target);
assert.deepEqual(attempt.value.result, fixture.result);
const old = await client.getAttempt({{ runId: fixture.target.runId, attemptId: fixture.oldAttemptId }});
assert.equal(old.value.state, 'failed');
assert.equal(old.value.result.reason, 'old infrastructure failure');
const obtained = [];
let after = '0';
while (true) {{
  const page = await client.getEvents(fixture.target.runId, after, 2);
  if (!page.value.events.length) break;
  obtained.push(...page.value.events);
  after = page.value.next_cursor.toString();
}}
assert.deepEqual(obtained.map(event => event.sequence), fixture.sequences);
const downloaded = await client.getArtifact(fixture.target, fixture.artifact);
assert.deepEqual([...downloaded.bytes], fixture.payload);
assert.deepEqual(downloaded.record.value.metadata, {{ zero: 0, missing_reason: null }});
const received = [];
let first;
first = client.subscribe(fixture.target, record => {{ received.push(record.value); first.unsubscribe(); }});
assert.equal((await first.done).reason, 'unsubscribed');
const resumed = client.subscribe(fixture.target, record => {{ received.push(record.value); }}, first.cursor);
assert.equal((await resumed.done).reason, 'terminal');
assert.deepEqual(received.map(event => event.sequence), fixture.attemptSequences);
assert.deepEqual(received.at(-1).payload.result, fixture.result);
const ack = client.subscribe(fixture.target, () => assert.fail('204 must not synthesize an event'), resumed.cursor);
assert.equal((await ack.done).reason, 'terminal_acknowledged');
let active;
active = client.subscribe(fixture.running, () => active.unsubscribe());
assert.equal((await active.done).reason, 'unsubscribed');
assert.equal((await client.getAttempt(fixture.running)).value.state, 'running');
await assert.rejects(new StoredRunClient(fixture.endpoint, 'foreign-store').getRun(fixture.target.runId),
  error => error.code === 'identity_mismatch');
console.log(JSON.stringify({{ real_http: true, native_fetch_stream: true, real_browser: false,
  pages_and_attempts: true, exact_metadata_and_bytes: true, unsubscribe_did_not_cancel: true,
  terminal_acknowledgement: true, event_count: received.length, cursor: resumed.cursor.sequence }}));
"""
            receipt = _node(script, tmp_path)
            assert receipt["native_fetch_stream"] and receipt["unsubscribe_did_not_cancel"]
            assert store.get_run("still-running")["state"] == "running"
            assert store.attempts(run_id)[-1]["result"] == result
            # Deliberately corrupt only this newly created infrastructure object.
            (store.root / artifact["relative_path"]).write_bytes(b"corrupt fixture")
            receipt = _node(f"""
import assert from 'node:assert/strict';
import {{ StoredRunClient }} from {json.dumps(module.as_uri())};
const fixture = {json.dumps(selection)};
const client = new StoredRunClient(fixture.endpoint, fixture.storeId);
await assert.rejects(client.getArtifact(fixture.target, fixture.artifact),
  error => error.code === 'http' && error.status === 409 && error.detail.detail.code === 'artifact_integrity');
assert.equal((await client.getRun(fixture.target.runId)).value.state, 'succeeded');
console.log(JSON.stringify({{ corrupt_artifact_rejected: true, historical_terminal_unchanged: true }}));
""", tmp_path)
            assert receipt["corrupt_artifact_rejected"]
        assert server.stopped


def _supervised_modules(directory):
    """Use the same locked transpiler and DOM implementation as existing tests."""
    compiler = REPOSITORY / "web/node_modules/typescript/lib/typescript.js"
    panel_source = REPOSITORY / "web/src/panels/stored-run.ts"
    client_source = REPOSITORY / "web/src/run-client.ts"
    client = directory / "run-client.mjs"
    panel = directory / "stored-run.mjs"
    _node(f"""
import ts from {json.dumps(compiler.as_uri())};
import {{ readFile, writeFile }} from 'node:fs/promises';
for (const [source, destination, panel] of {json.dumps([[str(client_source), str(client), False], [str(panel_source), str(panel), True]])}) {{
  let code = await readFile(source, 'utf8');
  if (panel) code = code.replace("'../run-client'", "'./run-client.mjs'");
  const result = ts.transpileModule(code, {{ compilerOptions: {{ target: ts.ScriptTarget.ES2023, module: ts.ModuleKind.ES2022 }} }});
  await writeFile(destination, result.outputText);
}}
console.log(JSON.stringify({{ transpiled: 2, production_sources_changed: false }}));
""", directory)
    return client, panel


def _supervised_terminal_read(directory, modules, fixture):
    """Read real committed output using the client and the unmocked panel."""
    client_module, panel_module = modules
    jsdom = REPOSITORY / "web/node_modules/jsdom/lib/api.js"
    return _node(rf"""
import assert from 'node:assert/strict';
import {{ JSDOM }} from {json.dumps(jsdom.as_uri())};
import {{ StoredRunClient }} from {json.dumps(client_module.as_uri())};
import {{ mountStoredRunPanel }} from {json.dumps(panel_module.as_uri())};
const fixture = {json.dumps(fixture)};
const client = new StoredRunClient(fixture.endpoint, fixture.storeId);
const run = await client.getRun(fixture.target.runId);
const attempt = await client.getAttempt(fixture.target);
assert.equal(run.value.current_attempt_id, fixture.target.attemptId);
assert.equal(run.value.generation, fixture.generation);
assert.equal(attempt.value.generation, fixture.generation);
assert.equal(attempt.value.state, fixture.state);
assert.equal(attempt.value.result.status, 'prepared_pending_dependencies');
assert.equal(attempt.value.result.process.process_joined, true);
assert.equal(attempt.value.result.process.process_exit_observed, true);
assert.equal(attempt.value.result.resources.calls, null);
assert.equal(attempt.value.result.resources.accepted_steps, null);
assert.equal(attempt.value.result.counter_observation_kind, 'worker_reported_or_null_unobserved');
assert.equal(attempt.value.result.process.token.attempt_id, fixture.target.attemptId);
assert.equal(attempt.value.result.process.token.generation, fixture.generation);
assert.equal(run.value.request.input_metadata.transport.H_execution, fixture.H_execution);
if (fixture.state === 'succeeded') {{
  assert.equal(attempt.value.result.worker_result.large_integer, 18446744073709551617n);
  assert(Object.is(attempt.value.result.worker_result.negative_zero, -0));
  assert.equal(attempt.value.result.worker_result.tau, null);
  assert.equal(attempt.value.result.worker_result.scientific_qualification, false);
}} else {{
  assert.equal(attempt.value.result.worker_result.reason, fixture.reason);
}}
const received = [];
const subscription = client.subscribe(fixture.target, record => received.push(record.value), fixture.cursor ?? undefined);
assert.equal((await subscription.done).reason, 'terminal');
assert.deepEqual([...(fixture.seen ?? []), ...received.map(event => event.sequence.toString())], fixture.sequences);
assert(received.every(event => event.attempt_id === fixture.target.attemptId &&
  event.generation === fixture.eventGenerations[event.sequence.toString()]));
assert.equal(received.filter(event => event.kind === 'terminal').length, 1);
assert.equal(received.at(-1).payload.state, fixture.state);
assert.equal((await client.subscribe(fixture.target, () => assert.fail('terminal replay synthesized a duplicate'), subscription.cursor).done).reason, 'terminal_acknowledged');

const dom = new JSDOM('<main id="panel"></main>', {{ url: fixture.endpoint }});
globalThis.document = dom.window.document;
const root = document.getElementById('panel');
const downloads = [];
// Capture the panel's generated downloads, without replacing its client or HTTP.
dom.window.HTMLAnchorElement.prototype.click = function () {{ downloads.push({{ href: this.href, name: this.download }}); }};
const field = role => root.querySelector(`[data-role="${{role}}"]`);
const until = async predicate => {{
  const end = Date.now() + 3000;
  while (!predicate()) {{ assert(Date.now() < end, 'panel did not reach its observed boundary'); await new Promise(resolve => setTimeout(resolve, 5)); }}
}};
let panel;
try {{
  panel = mountStoredRunPanel(root, {{ endpoint: fixture.endpoint, storeId: fixture.storeId, ...fixture.target }});
  await panel.ready;
  await until(() => field('transport').textContent === 'Terminal event received; stream closed');
  assert.equal(field('attempt-state').textContent, fixture.state);
  assert(root.textContent.includes(fixture.target.attemptId));
  assert(root.textContent.includes('prepared_pending_dependencies'));
  const original = [...field('documents').querySelectorAll('pre')].map(node => node.textContent);
  assert(original.includes(run.json));
  assert(original.includes(attempt.json));
  assert(original.some(text => /"calls"\s*:\s*null/.test(text)));
  assert(field('events').textContent.includes('scientific_qualification'));
  assert(field('events').textContent.includes('18446744073709551617'));
  const buttons = () => [...root.querySelectorAll('button')];
  buttons().find(button => button.textContent === 'Download original JSON').click();
  await until(() => downloads.length === 1);
  assert.equal(await (await fetch(downloads[0].href)).text(), run.json);
  if (fixture.state === 'succeeded') {{
    const direct = await client.getArtifact(fixture.target, fixture.artifact);
    assert.deepEqual([...direct.bytes], fixture.payload);
    assert.equal(direct.record.value.metadata.scientific_qualification, false);
    const download = buttons().find(button => button.textContent === 'Verify and download artifact');
    assert(download, 'committed artifact download is unavailable');
    download.click();
    await until(() => downloads.length === 2);
    assert.deepEqual([...new Uint8Array(await (await fetch(downloads[1].href)).arrayBuffer())], fixture.payload);
  }} else {{
    assert(field('artifacts').textContent.includes('No committed artifact references'));
    assert(root.textContent.includes(fixture.reason));
  }}
  const cursor = panel.cursor.sequence;
  panel.reconnect();
  await until(() => field('transport').textContent === 'Terminal already acknowledged (HTTP 204)');
  assert.equal(panel.cursor.sequence, cursor);
  assert.equal(field('attempt-state').textContent, fixture.state);
  if (fixture.oldAttemptId) {{
    await panel.select({{ endpoint: fixture.endpoint, storeId: fixture.storeId, runId: fixture.target.runId, attemptId: fixture.oldAttemptId }});
    await until(() => field('transport').textContent === 'Terminal event received; stream closed');
    assert.equal(field('run-state').textContent, `succeeded; current attempt ${{fixture.target.attemptId}}`);
    assert.equal(field('attempt-state').textContent, 'cancelled');
    assert(root.textContent.includes(fixture.oldAttemptId));
  }}
  panel.dispose();
  assert.equal(root.childNodes.length, 0);
  panel = mountStoredRunPanel(root, {{ endpoint: fixture.endpoint, storeId: fixture.storeId, ...fixture.target }});
  await panel.ready;
  await until(() => field('transport').textContent === 'Terminal event received; stream closed');
  assert.equal(field('attempt-state').textContent, fixture.state);
  console.log(JSON.stringify({{ real_http: true, real_client: true, real_panel: true, DOM: 'jsdom', real_browser: false,
    state: fixture.state, generation: fixture.generation, calls: null, scientific_qualification: false,
    original_json_preserved: true, positive_process_exit_in_store: true, replay_and_panel_reopen: true,
    artifact_download: fixture.state === 'succeeded', terminal_events: 1 }}));
}} finally {{ panel?.dispose(); dom.window.close(); }}
""", directory)


@pytest.mark.parametrize("outcome", ["succeeded", "failed", "cancelled"])
def test_supervised_attempt_reaches_durable_client_and_panel(tmp_path, outcome):
    """Compose actual process/store/service/TS consumers; no terminal injection."""
    tmp_path = tmp_path.resolve()
    modules = _supervised_modules(tmp_path)
    payload = request(outcome=outcome, scope="supervised-client-infrastructure", scientific_qualification=False)
    binding = WorkerBinding.from_function("supervised_client", workers.supervised_client,
        registry_sha256=REGISTRY.sha256, schema_sha256=SCHEMA.sha256)
    cap = limits(wall_seconds=20.0, output_bytes=1024**2, accepted_steps=None, calls=None)
    old_attempt = None
    old_generation = None
    with RunStore(tmp_path / "db") as store, ProcessSupervisor(store, tmp_path / "work", (binding,),
        max_active=1, max_pending=1, shared_rss_bytes=768*1024**2, shared_output_bytes=16*1024**2,
        soft_grace_seconds=0.1, terminate_grace_seconds=0.1, poll_seconds=0.01) as pool:
        run_id = "supervised-" + outcome
        if outcome == "succeeded":
            previous = pool.submit(run_id, binding.id, payload, cap)
            ready(store, previous)
            assert pool.cancel(previous, reason="earlier_infrastructure_attempt")
            assert assert_reaped(pool.wait(previous, 3))["store_state"] == "cancelled"
            prior = store.get_run(run_id)
            old_generation = prior["generation"]
            store.retry(run_id, expected_attempt_id=previous.attempt_id, expected_generation=prior["generation"])
            old_attempt = previous.attempt_id
        handle = pool.submit(run_id, binding.id, payload, cap)
        pid = ready(store, handle)
        run = store.get_run(run_id)
        assert handle.attempt_id != old_attempt
        assert isinstance(run["generation"], int) and run["generation"] > 0
        if old_generation is not None:
            assert run["generation"] > old_generation
        fixture = {"storeId": store.store_id, "target": {"runId": run_id, "attemptId": handle.attempt_id},
                   "generation": run["generation"], "state": outcome, "oldAttemptId": old_attempt,
                   "H_execution": payload.execution_identity.sha256,
                   "eventGenerations": {str(event["sequence"]): event["generation"]
                       for event in store.events(run_id=run_id, limit=1000)
                       if event["attempt_id"] == handle.attempt_id}}
        with _HTTPServer(store.root) as server:
            fixture["endpoint"] = f"http://127.0.0.1:{server.port}/"
            observed = _node(f"""
import assert from 'node:assert/strict';
import {{ StoredRunClient }} from {json.dumps(modules[0].as_uri())};
const fixture = {json.dumps(fixture)};
const client = new StoredRunClient(fixture.endpoint, fixture.storeId);
const seen = [];
let stream;
stream = client.subscribe(fixture.target, record => {{
  const event = record.value;
  assert.equal(event.attempt_id, fixture.target.attemptId);
  assert.equal(event.generation, fixture.eventGenerations[event.sequence.toString()]);
  seen.push(event.sequence.toString());
  if (event.payload.ui_progress?.stage === 'ready') {{
    const progress = event.payload.ui_progress;
    assert.equal(progress.large_integer, 18446744073709551617n);
    assert(Object.is(progress.negative_zero, -0));
    assert.equal(progress.unobserved, null);
    assert.equal(progress.scientific_qualification, false);
    assert.deepEqual(progress.forbidden_modules_loaded, []);
    stream.unsubscribe();
  }}
}});
assert.equal((await stream.done).reason, 'unsubscribed');
assert.equal((await client.getAttempt(fixture.target)).value.state, 'running');
console.log(JSON.stringify({{ cursor: stream.cursor, seen, running_after_unsubscribe: true }}));
""", tmp_path)
            assert observed["running_after_unsubscribe"]
            assert store.get_run(run_id)["state"] == "running" and pool.accounting()["active"] == 1
            if outcome == "cancelled":
                assert pool.cancel(handle, reason="supervised_fixture_cancelled")
            else:
                release(tmp_path, store, handle)
            receipt = assert_reaped(pool.wait(handle, 4))
            assert receipt["pid"] == pid and receipt["terminal_committed"]
            assert receipt["store_state"] == outcome
            assert receipt["resources"]["accepted_steps"] is None and receipt["resources"]["calls"] is None
            if outcome != "cancelled":
                assert receipt["exitcode"] == 0
            else:
                assert "soft_stop" in receipt["signals"]
            events = [event for event in store.events(run_id=run_id, limit=1000) if event["attempt_id"] == handle.attempt_id]
            # Enqueue/retry events precede claim and retain their own generation.
            assert events[-1]["generation"] == run["generation"]
            assert [event["generation"] for event in events] == sorted(event["generation"] for event in events)
            assert sum(event["kind"] == "terminal" for event in events) == 1 and events[-1]["kind"] == "terminal"
            fixture.update(cursor=observed["cursor"], seen=observed["seen"], sequences=[str(event["sequence"]) for event in events],
                           eventGenerations={str(event["sequence"]): event["generation"] for event in events})
            if outcome == "succeeded":
                artifact = store.artifact(receipt["terminal_artifacts"][0])
                fixture.update(artifact={"artifact_id": artifact["artifact_id"], "manifest_sha256": artifact["manifest_sha256"]},
                               payload=list(struct.pack("<2d", 0.0, -0.0) + b"\x00supervised"))
            else:
                fixture["reason"] = "worker_exception" if outcome == "failed" else "supervised_fixture_cancelled"
                assert receipt["worker_result"]["reason"] == fixture["reason"]
                assert receipt["terminal_artifacts"] == []
            observed = _supervised_terminal_read(tmp_path, modules, fixture)
            assert observed["real_client"] and observed["real_panel"] and observed["calls"] is None
            _receipt({"kind": "supervised_client_process", "outcome": outcome, "generation": run["generation"],
                      "receipt": receipt, "scientific_qualification": False})
        assert server.stopped
        expected_attempts = store.attempts(run_id)
        assert pool.accounting()["active"] == 0
    # A new store connection and new real service must read the same committed
    # output. Its explicit new endpoint starts its own source-bound cursor.
    with RunStore(tmp_path / "db") as reopened, _HTTPServer(tmp_path / "db") as server:
        assert reopened.store_id == fixture["storeId"]
        assert reopened.attempts(run_id) == expected_attempts
        fixture.update(endpoint=f"http://127.0.0.1:{server.port}/", cursor=None, seen=[])
        assert _supervised_terminal_read(tmp_path, modules, fixture)["replay_and_panel_reopen"]
    assert server.stopped
