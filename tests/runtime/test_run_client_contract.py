"""Real loopback fetch from the TypeScript client; RunStore infrastructure only.

No model, trajectory, scientific result, browser or production UI is exercised.
The existing HTTP fixture owns and positively closes its reader and threads.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time

from solarlab.io.artifacts import bytes_artifact
from solarlab.io.run_store import RunStore
from test_event_service import _HTTPServer, _receipt


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
