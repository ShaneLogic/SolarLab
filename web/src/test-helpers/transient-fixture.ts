/** Synthetic wire/display fixtures only; actual numerical evidence is separate. */
import type { TransientCatalog, TransientProtocol, TransientResult, TransientArrayName } from '../transient-client'
import { TRANSIENT_SCOPE, TRANSIENT_UNITS } from '../transient-client'
import type { FloatArray } from '../steady-jv-client'

export const registry = 'a'.repeat(64), schema = 'b'.repeat(64), prepared = 'c'.repeat(64)
export const protocol: TransientProtocol = { times_s: [0, 1e-7, 2e-7], voltage_V: [0, 0, 0], generation_scale: [1, 1.0025, 1.01],
  intervals_per_layer: 12, controls: { max_steps: 32, wall_s: 30,
    newton_budget: { max_iterations: 50, max_residual_evaluations: 1000, max_jacobian_evaluations: 50, max_backtracks: 24 } } }
export const catalog: TransientCatalog = { registry, schema, ceiling: { wall_seconds: 60, rss_bytes: 1610612736, output_bytes: 100663296 },
  routes: [{ id: 'transient_1d', version: 1, devices: [{ id: 'research_1d', preparedSha256: prepared }], bounds: {
    samples: { min: 2, max: 33 }, times_s: { min: 0, max: 1 }, voltage_V: { min: -2, max: 2 }, generation_scale: { min: 0, max: 10 },
    intervals_per_layer: { min: 2, max: 256 }, max_steps: { min: 1, max: 32 }, wall_s: { min: 0, max: 30 },
    max_iterations: { min: 1, max: 50 }, max_residual_evaluations: { min: 1, max: 1000 },
    max_jacobian_evaluations: { min: 1, max: 50 }, max_backtracks: { min: 0, max: 24 },
  } }] }
const numeric = (min: number, max: number) => ({ type: 'number', minimum: min, maximum: max })
export function catalogDocument() {
  const keys = ['device_id', 'times_s', 'voltage_V', 'generation_scale', 'intervals_per_layer', 'controls']
  const array = (min: number, max: number) => ({ type: 'array', minItems: 2, maxItems: 33, items: numeric(min, max) })
  return { schema: 'solarlab.execution-catalog.v1', job_schema_sha256: schema,
    resource_ceiling: { ...catalog.ceiling, accepted_steps: null, calls: null }, bindings: [{
      experiment_id: 'transient_1d', version: 1, engine: 'solarlab', scope: TRANSIENT_SCOPE, driver: 'transient_1d',
      input_schema: { additionalProperties: false, required: keys, properties: {
        device_id: { type: 'string' }, times_s: array(0, 1), voltage_V: array(-2, 2), generation_scale: array(0, 10),
        intervals_per_layer: numeric(2, 256), controls: { $ref: '#/$defs/_TimeControls' },
      }, $defs: { _TimeControls: { additionalProperties: false, properties: {
        max_steps: numeric(1, 32), wall_s: { type: 'number', exclusiveMinimum: 0, maximum: 30 }, newton_budget: { $ref: '#/$defs/_NewtonBudgetInput' },
      } }, _NewtonBudgetInput: { additionalProperties: false, properties: {
        max_iterations: numeric(1, 50), max_residual_evaluations: numeric(1, 1000), max_jacobian_evaluations: numeric(1, 50), max_backtracks: numeric(0, 24),
      } } } }, preparation_context: { devices: [{ device_id: 'research_1d', prepared_sha256: prepared }] },
    }] }
}

export function resultFixture(failed = false, initialOnly = false): TransientResult {
  const count = initialOnly ? 1 : failed ? 2 : 3, requested = failed ? 4 : 3, n = 4
  const times = [0, 1e-7, 2e-7, 3e-7].slice(0, requested)
  const vector = (values: number[]): FloatArray => ({ shape: [values.length], values })
  const arrays = {
    nodes_m: vector([0, 1e-7, 2e-7, 3e-7]), accepted_times_s: vector(times.slice(0, count)), requested_times_s: vector(times),
    requested_voltage_V: vector(times.map(() => 0)), requested_generation_scale: vector(failed ? [1, 1, 1.5, 1] : protocol.generation_scale),
    potential_V: { shape: [count, n], values: Array.from({ length: count * n }, (_, i) => .2 + i * .01) },
    electron_density_m3: { shape: [count, n], values: Array.from({ length: count * n }, (_, i) => 1e16 + i * 1e14) },
    hole_density_m3: { shape: [count, n], values: Array.from({ length: count * n }, (_, i) => 2e16 + i * 1e14) },
  } as Record<TransientArrayName, FloatArray>
  for (const name of Object.keys(TRANSIENT_UNITS) as TransientArrayName[]) if (name.includes('current_density')) {
    const width = name.startsWith('terminal_') ? 2 : n - 1
    arrays[name] = { shape: [count - 1, width], values: Array.from({ length: (count - 1) * width }, (_, i) => -40 - i) }
  }
  const attempts = Array.from({ length: count - 1 + (failed ? 1 : 0) }, (_, i) => ({ accepted: i < count - 1,
    start_s: times[i], end_s: times[i + 1], termination: i < count - 1 ? 'converged' : 'budget_exhausted', error: null }))
  const summary = { schema: 'solarlab.transient-result.v1', engine: 'solarlab', physical_certificate_status: 'not_evaluated',
    compiled_identity: 'd'.repeat(64), outcome: failed ? 'partial' : 'completed', reason: failed ? 'budget_exhausted' : null,
    requested_count: requested, accepted_count: count, attempt_count: attempts.length }
  return { state: failed ? 'failed' : 'succeeded', arrays, summary, attempts, originalJson: '{"synthetic_fixture":true}', manifestJson: '',
    diagnosticsJson: JSON.stringify({ attempts: { $tuple: attempts }, raw_nonfinite: { $float: { encoded: 'not interpreted as zero' } } }),
    worker: { schema: 'solarlab.transient.worker-result.v1', engine: 'solarlab', scientific_qualification: false,
      prepared_sha256: prepared, registered_device_id: 'research_1d', execution_catalog_sha256: registry,
      H_input: '1'.repeat(64), H_physics: '2'.repeat(64), H_execution: '3'.repeat(64), reason: summary.reason },
  }
}

const jsonBytes = (value: unknown) => new TextEncoder().encode(JSON.stringify(value))
const hex = (value: Uint8Array) => Array.from(value, n => n.toString(16).padStart(2, '0')).join('')
const digest = async (bytes: Uint8Array<ArrayBuffer>) => hex(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)))
export function npy(array: FloatArray): Uint8Array<ArrayBuffer> {
  const header = `{'descr': '<f8', 'fortran_order': False, 'shape': (${array.shape.join(', ')},), }\n`
  const bytes = new Uint8Array(10 + header.length + array.values.length * 8), view = new DataView(bytes.buffer)
  bytes.set([147, 78, 85, 77, 80, 89, 1, 0]); view.setUint16(8, header.length, true); bytes.set(new TextEncoder().encode(header), 10)
  array.values.forEach((value, i) => view.setFloat64(10 + header.length + i * 8, value, true)); return bytes
}

/** Real SHA-256/NPY envelopes for testing the unchanged StoredRunClient, without a solver. */
export async function storedFixture(data = resultFixture()) {
  const artifactMap = new Map<string, { record: Record<string, unknown>; bytes: Uint8Array<ArrayBuffer> }>()
  const references: { artifact_id: string; manifest_sha256: string }[] = []
  const payloads: Record<string, unknown> = {}, identity = { H_input: data.worker.H_input, H_physics: data.worker.H_physics, H_execution: data.worker.H_execution }
  const role = (value: unknown) => ({ content_hex: hex(jsonBytes(value)) })
  const manifest = jsonBytes({ schema: 'solarlab.execution-manifest.v1', execution_identity: { payload: { ...identity, engine: { name: 'solarlab' } } },
    content_roles: { protocol: role({ times_s: data.arrays.requested_times_s.values, voltage_V: data.arrays.requested_voltage_V.values,
      generation_scale: data.arrays.requested_generation_scale.values }), preparation: role({ prepared_sha256: prepared }) } })
  const manifestSha = await digest(manifest)
  const worker = { ...data.worker, manifest_sha256: manifestSha, transient_result: data.summary, payloads }
  async function add(name: string, bytes: Uint8Array<ArrayBuffer>, format: string, metadata: Record<string, unknown>) {
    const sha256 = await digest(bytes), ref = { artifact_id: name, manifest_sha256: 'e'.repeat(64) }
    references.push(ref)
    artifactMap.set(name, { bytes, record: { ...ref, name, run_id: 'run', attempt_id: 'attempt', generation: 1,
      phase: 'committed', sha256, size_bytes: bytes.length, format, metadata } })
    if (name !== 'execution_manifest') payloads[name] = { sha256, format, metadata }
  }
  for (const [name, array] of Object.entries(data.arrays)) await add(name, npy(array), 'npy', {
    schema: data.summary.schema, compiled_identity: data.summary.compiled_identity, unit: TRANSIENT_UNITS[name as TransientArrayName],
    dtype: '<f8', order: 'C', shape: array.shape,
  })
  await add('transient_diagnostics', new TextEncoder().encode(data.diagnosticsJson), 'bytes', { schema: data.summary.schema,
    compiled_identity: data.summary.compiled_identity, role: 'raw_diagnostics' })
  await add('execution_manifest', manifest, 'bytes', { manifest_sha256: manifestSha })
  const result = { worker_result: worker }, terminal = { state: data.state, result, artifacts: references }
  const attempt = { run_id: 'run', attempt_id: 'attempt', state: data.state, generation: 1, created_ns: 0,
    attempt_number: 1, owner_id: 'owner', started_ns: 1, finished_ns: 2, result, terminal }
  const run = { run_id: 'run', current_attempt_id: 'attempt', state: data.state, generation: 1, created_ns: 0,
    request: { identity_kind: 'opaque_upstream', identity, input_metadata: {
      execution_manifest: { manifest_sha256: manifestSha, H_execution: identity.H_execution },
      transport: { documents: { registry: { sha256: registry } } },
    } } }
  const response = (value: unknown) => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json', 'X-RunStore-ID': 'store' } })
  function fetchStored(input: URL | string) {
    const path = new URL(input).pathname
    if (path.endsWith('/runs/run')) return Promise.resolve(response(run))
    if (path.endsWith('/attempts/attempt')) return Promise.resolve(response(attempt))
    const parts = path.split('/'), content = parts.at(-1) === 'content', name = parts.at(content ? -2 : -1)!
    const artifact = artifactMap.get(name)
    if (!artifact) throw Error('Unexpected fixture request: ' + path)
    return Promise.resolve(content ? new Response(artifact.bytes, { headers: { 'Content-Type': 'application/octet-stream',
      'X-RunStore-ID': 'store', ETag: `"${artifact.record.sha256}"` } }) : response(artifact.record))
  }
  return { data, run, attempt, artifactMap, fetchStored }
}
