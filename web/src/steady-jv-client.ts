/** The explicitly installed classical DC route; stored reads use RunStore's client. */
import { StoredRunClient, RunClientError } from './run-client'
import type { ArtifactRecord, AttemptIdentity, StoredObject } from './run-client'

export const STEADY_SCOPE = 'static_classical_ion_free_1d_dc'
export const ARRAY_UNITS = {
  nodes_m: 'm', voltage_V: 'V', output_current_A_m2: 'A/m^2',
  current_density_A_m2: 'A/m^2', contact_current_density_A_m2: 'A/m^2',
  potential_V: 'V', electron_density_m3: 'm^-3', hole_density_m3: 'm^-3',
} as const
export type ArrayName = keyof typeof ARRAY_UNITS
export interface FloatArray { shape: number[]; values: number[] }
export interface SteadyProtocol {
  voltage_V: number[]; intervals_per_layer: number; generation_scale: number; incident_power_W_m2: number | null
}
export interface SteadyDevice { id: string; preparedSha256: string }
export interface SteadyRoute {
  id: string; version: number; devices: readonly SteadyDevice[]
  minPoints: number; maxPoints: number; minIntervals: number; maxIntervals: number
}
export interface SteadyCatalog {
  registry: string; schema: string; routes: readonly SteadyRoute[]
  ceiling: { wall_seconds: number; rss_bytes: number; output_bytes: number }
}
export interface SteadyJob {
  reader: StoredRunClient; target: Readonly<AttemptIdentity>; protocol: Readonly<SteadyProtocol>
  device: SteadyDevice; catalog: SteadyCatalog
}
export interface SteadyResult {
  arrays: Record<ArrayName, FloatArray>; metrics: StoredObject; evidence: unknown[]
  worker: StoredObject; originalJson: string
}

function check(value: unknown, message: string): asserts value {
  if (!value) throw new RunClientError('protocol', message)
}
function object(value: unknown): StoredObject {
  check(value !== null && typeof value === 'object' && !Array.isArray(value), 'Expected an object')
  return value as StoredObject
}
function id(value: unknown): string {
  check(typeof value === 'string' && /^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$/.test(value), 'Invalid registered identifier')
  return value
}
function sha(value: unknown): string {
  check(typeof value === 'string' && /^[a-f0-9]{64}$/.test(value), 'Missing source identity header or digest')
  return value
}
function positive(value: unknown, integral = false): number {
  check(typeof value === 'number' && Number.isFinite(value) && value > 0
    && (!integral || Number.isSafeInteger(value)), 'Invalid service bound')
  return value
}

export function voltageGrid(start: number, end: number, count: number): number[] {
  check(Number.isFinite(start) && Number.isFinite(end) && start !== end
    && Number.isSafeInteger(count) && count >= 2 && count <= 101, 'Use 2–101 distinct, finite voltage samples')
  return Array.from({ length: count }, (_, i) => i === 0 ? start : i === count - 1 ? end : start + (end - start) * i / (count - 1))
}

function protocol(value: SteadyProtocol, route: SteadyRoute): void {
  const v = value.voltage_V
  check(v.length >= route.minPoints && v.length <= route.maxPoints && v.every(Number.isFinite)
    && (v.every((x, i) => i === 0 || x > v[i - 1]) || v.every((x, i) => i === 0 || x < v[i - 1])),
  'Voltage samples must be finite and strictly monotonic within the installed schema')
  check(Number.isSafeInteger(value.intervals_per_layer) && value.intervals_per_layer >= route.minIntervals
    && value.intervals_per_layer <= route.maxIntervals, 'Grid intervals are outside the installed schema')
  check(Number.isFinite(value.generation_scale) && value.generation_scale >= 0, 'Generation scale must be finite and nonnegative')
  check(value.incident_power_W_m2 === null || (Number.isFinite(value.incident_power_W_m2) && value.incident_power_W_m2 > 0),
    'Incident power must be positive, or blank when unknown')
}

/** Narrow decoder for this worker's plain C-order float64 NPY payloads.
 * getArtifact verifies the original bytes first. No pickle/eval/general dtype loader. */
export function readSteadyArray(bytes: Uint8Array, record: ArtifactRecord, allowEmpty = false): FloatArray {
  check(record.format === 'npy' && bytes.length >= 10
    && Array.from(bytes.slice(0, 6)).join(',') === '147,78,85,77,80,89', 'Invalid steady NPY artifact')
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const version = bytes[6]
  check((version === 1 || version === 2) && bytes[7] === 0 && bytes.length >= (version === 1 ? 10 : 12), 'Unsupported NPY version')
  const prefix = version === 1 ? 10 : 12
  const length = version === 1 ? view.getUint16(8, true) : view.getUint32(8, true)
  check(length <= 4096 && prefix + length <= bytes.length, 'Invalid NPY header size')
  const header = new TextDecoder('utf-8', { fatal: true }).decode(bytes.slice(prefix, prefix + length))
  const match = /^\{'descr': '([<>]f8)', 'fortran_order': False, 'shape': \((\d+(?:,\s*\d+)*,?)\),?\s*\}\s*$/.exec(header)
  check(match, 'Only plain C-order float64 steady arrays are supported')
  const shape = match[2].split(',').filter(x => x.trim()).map(Number)
  check(shape.length >= 1 && shape.length <= 2 && shape.every(n => Number.isSafeInteger(n) && (allowEmpty ? n >= 0 : n > 0)), 'Invalid steady array shape')
  const count = shape.reduce((a, b) => a * b, 1)
  check(Number.isSafeInteger(count) && prefix + length + count * 8 === bytes.length, 'NPY byte count mismatch')
  check(record.metadata.dtype === match[1] && record.metadata.order === 'C'
    && JSON.stringify(record.metadata.shape) === JSON.stringify(shape), 'NPY differs from committed metadata')
  const values = Array.from({ length: count }, (_, i) => view.getFloat64(prefix + length + i * 8, match[1][0] === '<'))
  check(values.every(Number.isFinite), 'Non-finite steady field cannot be plotted as a valid result')
  return { shape, values }
}

/** Shared bounded JSON transport for the installed research job routes. */
export async function requestJobService(endpoint: string, path: string, init: RequestInit = {}): Promise<{ value: StoredObject; response: Response }> {
  const response = await fetch(new URL(path, endpoint), { ...init, redirect: 'error', cache: 'no-store' })
  const reader = response.body?.getReader()
  check(reader, 'Missing service response')
  const chunks: Uint8Array[] = []; let size = 0
  try {
    while (true) {
      const part = await reader.read(); if (part.done) break
      size += part.value.length; check(size <= 1024 ** 2, 'Service response exceeds 1 MiB')
      chunks.push(part.value)
    }
  } finally { await reader.cancel().catch(() => undefined); reader.releaseLock() }
  const bytes = new Uint8Array(size); let offset = 0
  for (const part of chunks) { bytes.set(part, offset); offset += part.length }
  const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  if (!response.ok) throw new RunClientError('http', `Service HTTP ${response.status}: ${text.slice(0, 2048)}`, { status: response.status })
  check(response.headers.get('content-type')?.split(';')[0] === 'application/json', 'Expected JSON service response')
  return { value: object(JSON.parse(text)), response }
}

export class SteadyJVClient {
  readonly endpoint: string
  constructor(endpoint: string | URL) {
    const url = new URL(endpoint)
    check(['http:', 'https:'].includes(url.protocol) && !url.username && !url.password && !url.search && !url.hash,
      'Use an explicit HTTP service endpoint')
    this.endpoint = url.href.replace(/\/$/, '') + '/'
  }

  private request(path: string, init: RequestInit = {}): Promise<{ value: StoredObject; response: Response }> {
    return requestJobService(this.endpoint, path, init)
  }

  async discover(signal?: AbortSignal): Promise<SteadyCatalog> {
    const { value, response } = await this.request('experiments', { signal })
    const registry = sha(response.headers.get('X-Solarlab-Registry')), schema = sha(response.headers.get('X-Solarlab-Job-Schema'))
    check(value.schema === 'solarlab.execution-catalog.v1' && value.job_schema_sha256 === schema
      && response.headers.get('etag') === `"${registry}"` && Array.isArray(value.bindings), 'Inconsistent execution catalog')
    const limits = object(value.resource_ceiling)
    check(limits.accepted_steps === null && limits.calls === null, 'Steady counter budgets are unsupported')
    const ceiling = Object.freeze({ wall_seconds: positive(limits.wall_seconds), rss_bytes: positive(limits.rss_bytes, true), output_bytes: positive(limits.output_bytes, true) })
    const routes: SteadyRoute[] = []
    for (const raw of value.bindings) {
      const entry = object(raw)
      if (entry.engine !== 'solarlab' || entry.scope !== STEADY_SCOPE || entry.driver !== 'steady_jv') continue
      const schema = object(entry.input_schema), properties = object(schema.properties)
      const keys = ['device_id', 'voltage_V', 'intervals_per_layer', 'generation_scale', 'incident_power_W_m2']
      const required = schema.required
      if (schema.additionalProperties !== false || !Array.isArray(required)
        || required.length !== keys.length || keys.some(key => !required.includes(key))
        || Object.keys(properties).some(key => !keys.includes(key))) continue
      const voltage = object(properties.voltage_V), grid = object(properties.intervals_per_layer)
      const installed = object(entry.preparation_context)
      check(Array.isArray(installed.devices), 'Missing installed device list')
      const devices = installed.devices.map(raw => { const d = object(raw); return Object.freeze({ id: id(d.device_id), preparedSha256: sha(d.prepared_sha256) }) })
      check(new Set(devices.map(d => d.id)).size === devices.length, 'Duplicate installed device IDs')
      routes.push(Object.freeze({ id: id(entry.experiment_id), version: positive(entry.version, true), devices: Object.freeze(devices),
        minPoints: positive(voltage.minItems, true), maxPoints: Math.min(101, positive(voltage.maxItems, true)),
        minIntervals: positive(grid.minimum, true), maxIntervals: positive(grid.maximum, true) }))
    }
    return Object.freeze({ registry, schema, ceiling, routes: Object.freeze(routes) })
  }

  async submit(catalog: SteadyCatalog, route: SteadyRoute, deviceId: string, input: SteadyProtocol, wallSeconds: number,
    runId = `steady-${crypto.randomUUID()}`): Promise<SteadyJob> {
    check(catalog.routes.includes(route), 'Select an installed route')
    const device = route.devices.find(d => d.id === deviceId); check(device, 'Select an installed device')
    protocol(input, route)
    check(Number.isFinite(wallSeconds) && wallSeconds > 0 && wallSeconds <= catalog.ceiling.wall_seconds, 'Wall budget exceeds the service ceiling')
    // Snapshot exactly the serialized input; further form edits cannot relabel this job.
    const frozen = JSON.parse(JSON.stringify({ voltage_V: [...input.voltage_V], intervals_per_layer: input.intervals_per_layer,
      generation_scale: input.generation_scale, incident_power_W_m2: input.incident_power_W_m2 })) as SteadyProtocol
    const { value, response } = await this.request('jobs', { method: 'POST', headers: {
      'Content-Type': 'application/json', 'X-Solarlab-Registry': catalog.registry, 'X-Solarlab-Job-Schema': catalog.schema,
    }, body: JSON.stringify({ schema_version: 'solarlab.job-submission.v1', run_id: id(runId),
      experiment_id: route.id, version: route.version, engine: 'solarlab', input: { device_id: deviceId, ...frozen },
      limits: { ...catalog.ceiling, wall_seconds: wallSeconds, accepted_steps: null, calls: null } }) })
    check(response.status === 202 && value.status === 'accepted' && value.run_id === runId
      && value.execution_catalog_sha256 === catalog.registry && value.job_schema_sha256 === catalog.schema, 'Job acceptance identity mismatch')
    return { reader: new StoredRunClient(this.endpoint, id(value.store_id)), target: Object.freeze({ runId, attemptId: id(value.attempt_id) }),
      protocol: Object.freeze(frozen), device, catalog }
  }

  async cancel(job: SteadyJob, reason = 'User requested steady DC cancellation'): Promise<void> {
    const run = (await job.reader.getRun(job.target.runId)).value
    check(run.current_attempt_id === job.target.attemptId, 'The selected attempt is stale; cancellation refused')
    check(run.state === 'queued' || run.state === 'running', 'The selected job is already terminal')
    const { value } = await this.request(`jobs/${encodeURIComponent(job.target.runId)}/attempts/${encodeURIComponent(job.target.attemptId)}/cancel`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: `{"expected_generation":${run.generation.toString()},"reason":${JSON.stringify(reason)}}`,
    })
    check(value.run_id === job.target.runId && value.attempt_id === job.target.attemptId, 'Cancellation identity mismatch')
  }

  async readResult(job: SteadyJob, signal?: AbortSignal): Promise<SteadyResult> {
    const [snapshot, run] = await Promise.all([job.reader.getAttempt(job.target, signal), job.reader.getRun(job.target.runId, signal)])
    const attempt = snapshot.value
    check(attempt.state === 'succeeded' && attempt.terminal, `Attempt is ${attempt.state}; no successful steady result`)
    const worker = object(attempt.result?.worker_result), identity = object(run.value.request.identity)
    check(worker.schema === 'solarlab.steady-jv.worker-result.v1' && worker.engine === 'solarlab'
      && worker.prepared_sha256 === job.device.preparedSha256 && worker.execution_catalog_sha256 === job.catalog.registry,
    'Stored result differs from the selected steady job')
    for (const key of ['H_input', 'H_physics', 'H_execution']) check(sha(worker[key]) === identity[key], 'Stored execution identity mismatch')
    const expected = object(worker.array_payloads), document = object(worker.steady_result)
    check(document.schema === 'solarlab.steady-jv.v1' && document.engine === 'solarlab' && document.outcome === 'completed'
      && typeof document.compiled_identity === 'string' && document.compiled_identity.length > 0, 'Unsupported steady result schema')
    check(Object.keys(expected).length === 8 && Object.keys(ARRAY_UNITS).every(name => name in expected)
      && attempt.terminal.artifacts.length === 9, 'Missing or unexpected steady artifacts')
    const arrays = {} as Record<ArrayName, FloatArray>; const names = new Set<string>()
    for (const reference of attempt.terminal.artifacts) {
      const artifact = await job.reader.getArtifact(job.target, reference, signal)
      const record = artifact.record.value, name = String(record.name)
      check(!names.has(name) && String(record.generation) === String(attempt.generation), 'Duplicate or foreign-generation artifact')
      names.add(name)
      if (name === 'execution_manifest') { check(record.sha256 === sha(worker.manifest_sha256), 'Execution manifest differs'); continue }
      check(Object.hasOwn(ARRAY_UNITS, name) && record.sha256 === sha(expected[name]), 'Array content differs from the worker result')
      const key = name as ArrayName
      check(record.metadata.unit === ARRAY_UNITS[key] && record.metadata.compiled_identity === document.compiled_identity,
        'Unexpected steady array unit or compiled identity')
      arrays[key] = readSteadyArray(artifact.bytes, record)
    }
    check(names.has('execution_manifest') && Object.keys(arrays).length === 8, 'Incomplete steady result')
    const p = job.protocol.voltage_V.length, n = arrays.nodes_m.values.length
    for (const [name, array] of Object.entries(arrays)) {
      const shape = name === 'nodes_m' ? [n] : ['voltage_V', 'output_current_A_m2'].includes(name) ? [p]
        : [p, name === 'current_density_A_m2' ? n - 1 : name === 'contact_current_density_A_m2' ? 2 : n]
      check(JSON.stringify(array.shape) === JSON.stringify(shape), 'Steady array dimensions disagree')
    }
    check(arrays.voltage_V.values.every((v, i) => v === job.protocol.voltage_V[i]) && n >= 3, 'Result voltage/grid differs from the submitted protocol')
    check(Array.isArray(document.solver_evidence) && document.solver_evidence.length === p, 'Missing solver evidence')
    return { arrays, metrics: object(document.metrics), evidence: document.solver_evidence, worker, originalJson: snapshot.json }
  }
}
