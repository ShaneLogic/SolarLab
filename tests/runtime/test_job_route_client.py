"""Live V2 TypeScript job client against the shared loopback solarlab_server."""
import json
import subprocess

import pytest

from solarlab.experiments.execution import JobHandle
from solarlab.io.run_store import RunStore
from solarlab_server.job_service import JobService
from job_route_fixtures import ROOT
from test_job_routes import _HTTPServer, registered as registered, supervisor

pytestmark = pytest.mark.frontend


def test_actual_v2_typescript_client_executes_solarlab_job_and_retired_legacy_route_refuses(tmp_path, registered):
    registration, _ = registered
    with RunStore(tmp_path / "store") as store, supervisor(store, tmp_path / "work", registration) as pool:
        with _HTTPServer(store.root, job_service=JobService(registration, pool)) as server:
            endpoint = f"http://127.0.0.1:{server.port}/"
            code = """
import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';

const ts = (await import(pathToFileURL(process.argv[2]))).default;
const srcRoot = process.argv[3];
const outDir = process.argv[4];
const endpoint = process.argv[5];

async function transpile(relName, replacements = []) {
  let code = await readFile(`${srcRoot}/${relName}.ts`, 'utf8');
  for (const [from, to] of replacements) code = code.replaceAll(from, to);
  const out = ts.transpileModule(code, {
    compilerOptions: { target: ts.ScriptTarget.ES2023, module: ts.ModuleKind.ES2022 },
  }).outputText;
  await writeFile(`${outDir}/${relName}.mjs`, out);
}

await transpile('run-client');
await transpile('steady-jv-client', [["'./run-client'", "'./run-client.mjs'"]]);
await transpile('v2-workstation-client', [["import.meta.env.VITE_SOLARLAB_SERVICE_BASE", "undefined"]]);
await transpile('v2-job-client', [
  ["'./run-client'", "'./run-client.mjs'"],
  ["'./steady-jv-client'", "'./steady-jv-client.mjs'"],
  ["'./v2-workstation-client'", "'./v2-workstation-client.mjs'"],
]);
await writeFile(`${outDir}/base.mjs`, `export const BASE = ${JSON.stringify(endpoint.replace(/\\/$/, ''))};`);
await transpile('job-stream', [["'./api'", "'./base.mjs'"]]);

// 1. Retained legacy job-stream.ts is explicitly refused with HTTP 410 and zero fallback.
const legacyMod = await import(pathToFileURL(`${outDir}/job-stream.mjs`));
await assert.rejects(
  () => legacyMod.startJob('jv', null, {}, 'fixture:scaps'),
  err => String(err.message).includes('410'),
);

// 2. Production V2 job client submits engine='solarlab' via POST /jobs and streams durable RunStore events.
const { startJob, streamJobEvents } = await import(pathToFileURL(`${outDir}/v2-job-client.mjs`));
const handle = await startJob('jv', null, { n_points: 2, V_max: 0.05, N_grid: 18 }, null, endpoint);
const decoded = JSON.parse(Buffer.from(handle.slice(3), 'base64').toString('utf8'));
let progress = 0;
let result;
let timer;
await new Promise((resolve, reject) => {
  timer = setTimeout(() => reject(new Error('bounded client wait expired')), 15000);
  streamJobEvents(handle, {
    onProgress: () => { progress += 1; },
    onResult: value => { result = value; },
    onError: msg => reject(new Error(msg)),
    onDone: () => resolve(undefined),
  });
}).finally(() => clearTimeout(timer));

assert(progress > 0);
assert.equal(result.scientific_qualification, false);
assert.equal(result.execution_manifest.engine, 'solarlab');
assert.deepEqual(result.V_fwd, [0, 0.05]);
console.log(JSON.stringify({
  runId: decoded.runId,
  attemptId: decoded.attemptId,
  progress,
  scientific_qualification: result.scientific_qualification,
}));
"""
            program = tmp_path / "v2-client-check.mjs"
            program.write_text(code)
            command = [
                "node",
                str(program),
                str(ROOT / "web/node_modules/typescript/lib/typescript.js"),
                str(ROOT / "web/src"),
                str(tmp_path),
                endpoint,
            ]
            actual = subprocess.run(command, capture_output=True, text=True, timeout=25)
            assert actual.returncode == 0, actual.stdout + actual.stderr
            record = json.loads(actual.stdout)
            assert store.get_run(record["runId"])["state"] == "succeeded"
            assert pool.wait(JobHandle(store.store_id, record["runId"], record["attemptId"]), 5).data["process_joined"]
