"""Existing TypeScript client against the shared loopback job service."""
import json
import subprocess

import pytest

from solarlab.experiments.execution import JobHandle
from solarlab.io.run_store import RunStore
from solarlab_server.job_service import JobService, legacy_job_target
from job_route_fixtures import ROOT
from test_job_routes import _HTTPServer, registered as registered, supervisor

pytestmark = pytest.mark.frontend


def test_actual_old_typescript_client_consumes_the_live_new_adapter(tmp_path, registered):
    registration, _ = registered
    with RunStore(tmp_path / "store") as store, supervisor(store, tmp_path / "work", registration) as pool:
        with _HTTPServer(store.root, job_service=JobService(registration, pool, allow_legacy_routes=True)) as server:
            # Only the test module's BASE import is bound to the actual loopback
            # socket. The production TS client source is transpiled unchanged.
            code = '''
import {readFile, writeFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
const ts = (await import(pathToFileURL(process.argv[2]))).default;
const source = await readFile(process.argv[3], 'utf8');
const output = ts.transpileModule(source, {compilerOptions:{target:ts.ScriptTarget.ES2023,module:ts.ModuleKind.ES2022}}).outputText;
await writeFile(process.argv[4], output.replace("from './api'", "from './base.mjs'"));
await writeFile(new URL('./base.mjs',pathToFileURL(process.argv[4])), `export const BASE=${JSON.stringify(process.argv[5])};`);
globalThis.EventSource = class {
  constructor(url) { this.events={}; this.abort=new AbortController(); this.run(url); }
  addEventListener(name, fn) { (this.events[name]??=[]).push(fn); }
  emit(name, value) { for (const fn of this.events[name]??[]) fn(value); }
  close() { this.abort.abort(); }
  async run(url) {
    try {
      const response=await fetch(url,{signal:this.abort.signal}); assert.equal(response.status,200);
      const reader=response.body.getReader(), decoder=new TextDecoder(); let buffer='';
      for (;;) {
        const {done,value}=await reader.read(); if(done) break; buffer+=decoder.decode(value,{stream:true});
        let at; while((at=buffer.indexOf('\\n\\n'))>=0) {
          const block=buffer.slice(0,at); buffer=buffer.slice(at+2);
          const fields=Object.fromEntries(block.split('\\n').filter(x=>!x.startsWith(':')).map(x=>{const p=x.indexOf(':');return [x.slice(0,p),x.slice(p+1).trimStart()]}));
          if(fields.event) this.emit(fields.event,{data:fields.data});
        }
      }
    } catch(error) { if(error.name!=='AbortError') this.emit('error',{}); }
  }
};
const {startJob,streamJobEvents}=await import(pathToFileURL(process.argv[4]));
const job=await startJob('jv',null,{},'fixture:scaps');
let progress=0,result; let timer;
await new Promise((resolve,reject)=>{
  timer=setTimeout(()=>reject(Error('bounded client wait expired')),15000);
  streamJobEvents(job,{onProgress:()=>progress++,onResult:value=>result=value,onError:reject,onDone:resolve});
}).finally(()=>clearTimeout(timer));
assert(progress>0); assert(result.artifacts.length>0); assert.equal(result.execution.worker_result.scientific_qualification,false);
console.log(JSON.stringify({job,progress,result_received:true,done_received:true}));
'''
            program = tmp_path / "client-check.mjs"
            program.write_text(code)
            command = ["node", str(program), str(ROOT / "web/node_modules/typescript/lib/typescript.js"),
                str(ROOT / "web/src/job-stream.ts"), str(tmp_path / "job-stream.mjs"), f"http://127.0.0.1:{server.port}"]
            actual = subprocess.run(command, capture_output=True, text=True, timeout=25)
            assert actual.returncode == 0, actual.stdout + actual.stderr
            record = json.loads(actual.stdout)
            run_id, attempt_id = legacy_job_target(record["job"])
            assert store.get_run(run_id)["state"] == "succeeded"
            assert pool.wait(JobHandle(store.store_id, run_id, attempt_id), 3).data["process_joined"]
