/** The installed ion-free transient route and its committed, possibly partial history. */
import { StoredRunClient, RunClientError } from './run-client'
import type { AttemptIdentity, StoredObject } from './run-client'
import { readSteadyArray, requestJobService } from './steady-jv-client'
import type { FloatArray } from './steady-jv-client'

export const TRANSIENT_SCOPE = 'classical_ion_free_1d_transient'
export const TRANSIENT_UNITS = {
  requested_times_s: 's', requested_voltage_V: 'V', requested_generation_scale: '1',
  accepted_times_s: 's', nodes_m: 'm', potential_V: 'V', electron_density_m3: 'm^-3', hole_density_m3: 'm^-3',
  conduction_current_density_A_m2: 'A/m^2', displacement_current_density_A_m2: 'A/m^2', total_current_density_A_m2: 'A/m^2',
  terminal_conduction_current_density_A_m2: 'A/m^2', terminal_displacement_current_density_A_m2: 'A/m^2',
  terminal_total_current_density_A_m2: 'A/m^2',
} as const
export type TransientArrayName = keyof typeof TRANSIENT_UNITS
export interface TransientProtocol {
  times_s: number[]; voltage_V: number[]; generation_scale: number[]; intervals_per_layer: number
  controls: { max_steps: number; wall_s: number; newton_budget: {
    max_iterations: number; max_residual_evaluations: number; max_jacobian_evaluations: number; max_backtracks: number
  } }
}
export interface Bound { min: number; max: number }
export interface TransientRoute {
  id: string; version: number; devices: readonly { id: string; preparedSha256: string }[]
  bounds: Record<'samples' | 'times_s' | 'voltage_V' | 'generation_scale' | 'intervals_per_layer'
    | 'max_steps' | 'wall_s' | 'max_iterations' | 'max_residual_evaluations' | 'max_jacobian_evaluations' | 'max_backtracks', Bound>
}
export interface TransientCatalog {
  registry: string; schema: string; routes: readonly TransientRoute[]
  ceiling: { wall_seconds: number; rss_bytes: number; output_bytes: number }
}
export interface TransientJob {
  reader: StoredRunClient; target: Readonly<AttemptIdentity>
  submitted?: { catalog: TransientCatalog; device: TransientRoute['devices'][number]; input: TransientProtocol }
}
export interface TransientResult {
  state: 'succeeded' | 'failed'; arrays: Record<TransientArrayName, FloatArray>
  summary: StoredObject; worker: StoredObject; attempts: StoredObject[]
  diagnosticsJson: string; originalJson: string; manifestJson: string
}

function check(value: unknown, message: string): asserts value {
  if (!value) throw new RunClientError('protocol', message)
}
function object(value: unknown): StoredObject {
  check(value !== null && typeof value === 'object' && !Array.isArray(value), 'Expected a transient record')
  return value as StoredObject
}
function id(value: unknown): string {
  check(typeof value === 'string' && /^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$/.test(value), 'Invalid registered or stored identifier')
  return value
}
function sha(value: unknown): string {
  check(typeof value === 'string' && /^[a-f0-9]{64}$/.test(value), 'Missing source identity digest')
  return value
}
function numeric(value: unknown): number {
  check(typeof value === 'number' && Number.isFinite(value), 'Expected a finite service bound')
  return value
}
function same(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object' || Array.isArray(a) !== Array.isArray(b)) return false
  const x = a as StoredObject, y = b as StoredObject, keys = Object.keys(x)
  return keys.length === Object.keys(y).length && keys.every(k => Object.hasOwn(y, k) && same(x[k], y[k]))
}
function within(value: number, bound: Bound, name: string, integer = false) {
  check(Number.isFinite(value) && value >= bound.min && value <= bound.max && (!integer || Number.isSafeInteger(value)),
    `${name} must be within ${bound.min}–${bound.max}${integer ? ' (integer)' : ''}`)
}

export function parseTransientSamples(text: string): Pick<TransientProtocol, 'times_s' | 'voltage_V' | 'generation_scale'> {
  const rows = text.trim().split(/\r?\n/).map(line => line.trim()).filter(Boolean)
  check(rows.length >= 2 && rows.length <= 33, 'Enter 2–33 rows: time (s), voltage (V), generation scale')
  const values = rows.map((line, i) => {
    const tokens = line.includes(',') ? line.split(',').map(value => value.trim()) : line.split(/\s+/)
    check(tokens.length === 3 && tokens.every(token => token.length > 0 && Number.isFinite(Number(token))), `Invalid sample row ${i + 1}`)
    return tokens.map(Number)
  })
  return { times_s: values.map(row => row[0]), voltage_V: values.map(row => row[1]), generation_scale: values.map(row => row[2]) }
}

export class TransientClient {
  readonly endpoint: string
  constructor(endpoint: string | URL) {
    const url = new URL(endpoint)
    check(['http:', 'https:'].includes(url.protocol) && !url.username && !url.password && !url.search && !url.hash,
      'Use an explicit HTTP service endpoint')
    this.endpoint = url.href.replace(/\/$/, '') + '/'
  }

  async discover(signal?: AbortSignal): Promise<TransientCatalog> {
    const { value, response } = await requestJobService(this.endpoint, 'experiments', { signal })
    const registry = sha(response.headers.get('X-Solarlab-Registry')), schema = sha(response.headers.get('X-Solarlab-Job-Schema'))
    check(value.schema === 'solarlab.execution-catalog.v1' && value.job_schema_sha256 === schema
      && response.headers.get('etag') === `"${registry}"` && Array.isArray(value.bindings), 'Inconsistent execution catalog')
    const cap = object(value.resource_ceiling)
    check(cap.accepted_steps === null && cap.calls === null, 'Whole-job transient counter caps are unsupported')
    const ceiling = { wall_seconds: numeric(cap.wall_seconds), rss_bytes: numeric(cap.rss_bytes), output_bytes: numeric(cap.output_bytes) }
    check(Object.values(ceiling).every(n => n > 0), 'Invalid service resource ceiling')
    const routes: TransientRoute[] = []
    for (const raw of value.bindings) {
      const entry = object(raw)
      if (entry.engine !== 'solarlab' || entry.scope !== TRANSIENT_SCOPE || entry.driver !== 'transient_1d') continue
      const schema = object(entry.input_schema), properties = object(schema.properties)
      const keys = ['device_id', 'times_s', 'voltage_V', 'generation_scale', 'intervals_per_layer', 'controls']
      if (schema.additionalProperties !== false || !same([...Object.keys(properties)].sort(), [...keys].sort())
        || !Array.isArray(schema.required) || !same([...schema.required].sort(), [...keys].sort())) continue
      const definition = (value: unknown) => {
        const data = object(value)
        if (data.$ref === undefined) return data
        check(typeof data.$ref === 'string' && /^#\/\$defs\/[A-Za-z0-9_]+$/.test(data.$ref), 'Unsupported installed control schema')
        return object(object(schema.$defs)[data.$ref.slice('#/$defs/'.length)])
      }
      const controls = definition(properties.controls), budget = definition(object(controls.properties).newton_budget)
      check(controls.additionalProperties === false && budget.additionalProperties === false, 'Unbounded installed controls')
      const bound = (value: unknown): Bound => {
        const data = object(value), min = numeric(data.minimum ?? data.exclusiveMinimum), max = numeric(data.maximum)
        check(min <= max, 'Invalid installed bounds'); return { min, max }
      }
      const time = object(properties.times_s), installed = object(entry.preparation_context)
      check(Array.isArray(installed.devices), 'Missing installed device list')
      const devices = installed.devices.map(raw => { const d = object(raw); return { id: id(d.device_id), preparedSha256: sha(d.prepared_sha256) } })
      check(new Set(devices.map(d => d.id)).size === devices.length, 'Duplicate installed device IDs')
      const c = object(controls.properties), b = object(budget.properties)
      const bounds = { samples: { min: numeric(time.minItems), max: Math.min(33, numeric(time.maxItems)) },
        times_s: bound(time.items), voltage_V: bound(object(properties.voltage_V).items),
        generation_scale: bound(object(properties.generation_scale).items), intervals_per_layer: bound(properties.intervals_per_layer),
        max_steps: bound(c.max_steps), wall_s: bound(c.wall_s), max_iterations: bound(b.max_iterations),
        max_residual_evaluations: bound(b.max_residual_evaluations), max_jacobian_evaluations: bound(b.max_jacobian_evaluations),
        max_backtracks: bound(b.max_backtracks) }
      check(bounds.samples.min >= 2 && bounds.samples.min <= bounds.samples.max, 'Unsupported transient sample count')
      routes.push({ id: id(entry.experiment_id), version: numeric(entry.version), devices, bounds })
    }
    return { registry, schema, routes, ceiling }
  }

  async submit(catalog: TransientCatalog, route: TransientRoute, deviceId: string, input: TransientProtocol,
    wallSeconds: number, runId = `transient-${crypto.randomUUID()}`): Promise<TransientJob> {
    check(catalog.routes.includes(route), 'Select an installed transient route')
    const device = route.devices.find(d => d.id === deviceId); check(device, 'Select an installed device')
    const b = route.bounds
    within(input.times_s.length, b.samples, 'Sample count', true)
    for (const name of ['times_s', 'voltage_V', 'generation_scale'] as const) {
      check(input[name].length === input.times_s.length, 'Time, voltage and generation sample lengths must match')
      for (const value of input[name]) within(value, b[name], name)
    }
    check(input.times_s.every((v, i) => !i || v > input.times_s[i - 1]), 'Time samples must be strictly increasing')
    within(input.intervals_per_layer, b.intervals_per_layer, 'Intervals per layer', true)
    within(input.controls.max_steps, b.max_steps, 'Step budget', true)
    within(input.controls.wall_s, b.wall_s, 'Transient wall budget')
    check(input.controls.wall_s > 0 && input.times_s.length - 1 <= input.controls.max_steps, 'Protocol exceeds its step or time budget')
    for (const name of ['max_iterations', 'max_residual_evaluations', 'max_jacobian_evaluations', 'max_backtracks'] as const)
      within(input.controls.newton_budget[name], b[name], name, true)
    check(Number.isFinite(wallSeconds) && wallSeconds > 0 && wallSeconds <= catalog.ceiling.wall_seconds, 'Wall budget exceeds service ceiling')
    const frozen: TransientProtocol = JSON.parse(JSON.stringify(input))
    const { value, response } = await requestJobService(this.endpoint, 'jobs', { method: 'POST', headers: {
      'Content-Type': 'application/json', 'X-Solarlab-Registry': catalog.registry, 'X-Solarlab-Job-Schema': catalog.schema,
    }, body: JSON.stringify({ schema_version: 'solarlab.job-submission.v1', run_id: id(runId), experiment_id: route.id,
      version: route.version, engine: 'solarlab', input: { device_id: deviceId, ...frozen },
      limits: { ...catalog.ceiling, wall_seconds: wallSeconds, accepted_steps: null, calls: null } }) })
    check(response.status === 202 && value.status === 'accepted' && value.run_id === runId
      && value.execution_catalog_sha256 === catalog.registry && value.job_schema_sha256 === catalog.schema,
    'Transient acceptance identity mismatch')
    return { ...this.open(id(value.store_id), runId, id(value.attempt_id)), submitted: { catalog, device, input: frozen } }
  }

  /** Only reads the explicitly selected attempt; it cannot submit or replay a solve. */
  open(storeId: string, runId: string, attemptId: string): TransientJob {
    return { reader: new StoredRunClient(this.endpoint, id(storeId)), target: Object.freeze({ runId: id(runId), attemptId: id(attemptId) }) }
  }

  async cancel(job: TransientJob): Promise<void> {
    const run = (await job.reader.getRun(job.target.runId)).value
    check(run.current_attempt_id === job.target.attemptId, 'The selected attempt is stale; cancellation refused')
    check(run.state === 'queued' || run.state === 'running', 'The selected attempt is already terminal')
    const { value } = await requestJobService(this.endpoint, `jobs/${encodeURIComponent(job.target.runId)}/attempts/${encodeURIComponent(job.target.attemptId)}/cancel`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: `{"expected_generation":${run.generation.toString()},"reason":"User requested transient cancellation"}`,
    })
    check(value.run_id === job.target.runId && value.attempt_id === job.target.attemptId, 'Cancellation identity mismatch')
  }

  async readResult(job: TransientJob, signal?: AbortSignal): Promise<TransientResult | undefined> {
    const [snapshot, run] = await Promise.all([job.reader.getAttempt(job.target, signal), job.reader.getRun(job.target.runId, signal)])
    const attempt = snapshot.value
    check(attempt.terminal, 'Attempt has no committed terminal result')
    if (attempt.state === 'cancelled') {
      check(attempt.terminal.artifacts.length === 0, 'Unexpected history attached to cancelled transient'); return undefined
    }
    check(attempt.state === 'succeeded' || attempt.state === 'failed', 'Unsupported terminal state')
    if (!attempt.terminal.artifacts.length && attempt.state === 'failed') return undefined
    const worker = object(attempt.result?.worker_result), identity = object(run.value.request.identity)
    const metadata = object(run.value.request.input_metadata), execution = object(metadata.execution_manifest)
    const transport = object(metadata.transport), registry = object(object(transport.documents).registry)
    check(worker.schema === 'solarlab.transient.worker-result.v1' && worker.engine === 'solarlab'
      && worker.scientific_qualification === false && worker.execution_catalog_sha256 === sha(registry.sha256), 'Not the bound transient worker result')
    for (const key of ['H_input', 'H_physics', 'H_execution']) check(sha(worker[key]) === identity[key], 'Transient execution identity mismatch')
    check(sha(worker.manifest_sha256) === execution.manifest_sha256 && worker.H_execution === execution.H_execution, 'Execution manifest identity mismatch')
    if (job.submitted) check(worker.prepared_sha256 === job.submitted.device.preparedSha256
      && worker.execution_catalog_sha256 === job.submitted.catalog.registry, 'Result differs from the submitted device/catalog')
    const expected = object(worker.payloads), summary = object(worker.transient_result)
    check(summary.schema === 'solarlab.transient-result.v1' && summary.engine === 'solarlab'
      && summary.physical_certificate_status === 'not_evaluated' && sha(summary.compiled_identity), 'Unsupported transient history')
    const keys = [...Object.keys(TRANSIENT_UNITS), 'transient_diagnostics']
    check(same(Object.keys(expected).sort(), [...keys].sort()) && attempt.terminal.artifacts.length === keys.length + 1,
      'Missing or unexpected transient artifacts')
    const arrays = {} as Record<TransientArrayName, FloatArray>, names = new Set<string>()
    let diagnosticsJson = '', manifestJson = ''
    for (const reference of attempt.terminal.artifacts) {
      const artifact = await job.reader.getArtifact(job.target, reference, signal), record = artifact.record.value, name = String(record.name)
      check(!names.has(name) && String(record.generation) === String(attempt.generation), 'Duplicate or foreign-generation artifact'); names.add(name)
      if (name === 'execution_manifest') {
        check(record.sha256 === worker.manifest_sha256, 'Execution manifest differs from stored identity')
        manifestJson = new TextDecoder('utf-8', { fatal: true }).decode(artifact.bytes); continue
      }
      check(Object.hasOwn(expected, name), 'Unexpected transient payload')
      const binding = object(expected[name])
      check(record.sha256 === sha(binding.sha256) && record.format === binding.format && same(record.metadata, binding.metadata)
        && record.metadata.compiled_identity === summary.compiled_identity && record.metadata.schema === summary.schema,
      'Transient payload metadata or compiled identity mismatch')
      if (name === 'transient_diagnostics') {
        check(record.format === 'bytes' && record.metadata.role === 'raw_diagnostics', 'Unexpected diagnostic payload')
        diagnosticsJson = new TextDecoder('utf-8', { fatal: true }).decode(artifact.bytes)
      } else {
        check(Object.hasOwn(TRANSIENT_UNITS, name), 'Unexpected transient array')
        const key = name as TransientArrayName
        check(record.metadata.unit === TRANSIENT_UNITS[key], 'Unexpected transient array unit')
        arrays[key] = readSteadyArray(artifact.bytes, record, true)
      }
    }
    check(names.size === 16 && manifestJson && diagnosticsJson && Object.keys(arrays).length === 14, 'Incomplete transient history')
    const n = arrays.nodes_m.values.length, times = arrays.requested_times_s.values, accepted = arrays.accepted_times_s.values
    check(n >= 3 && times.length >= 2 && accepted.length >= 1 && accepted.length <= times.length
      && accepted.every((v, i) => v === times[i]) && times.every((v, i) => !i || v > times[i - 1]), 'Accepted history is not the requested time prefix')
    check(arrays.nodes_m.values.every((v, i, a) => !i || v > a[i - 1]), 'Node coordinates are not increasing')
    for (const [name, array] of Object.entries(arrays)) {
      const shape = name === 'nodes_m' ? [n] : name.startsWith('requested_') ? [times.length]
        : name === 'accepted_times_s' ? [accepted.length] : name.includes('current_density')
          ? [accepted.length - 1, name.startsWith('terminal_') ? 2 : n - 1] : [accepted.length, n]
      check(same(array.shape, shape), 'Transient array shape disagrees with the accepted history')
    }
    const manifest = object(JSON.parse(manifestJson)), bound = object(object(manifest.execution_identity).payload)
    check(bound.H_input === worker.H_input && bound.H_physics === worker.H_physics && object(bound.engine).name === 'solarlab', 'Manifest execution binding differs')
    const roles = object(manifest.content_roles)
    const role = (name: string): StoredObject => {
      const hex = object(roles[name]).content_hex
      check(typeof hex === 'string' && hex.length % 2 === 0 && /^[a-f0-9]+$/.test(hex), 'Invalid manifest content role')
      return object(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Uint8Array.from(hex.match(/../g)!, v => parseInt(v, 16)))))
    }
    const protocol = { times_s: times, voltage_V: arrays.requested_voltage_V.values, generation_scale: arrays.requested_generation_scale.values }
    check(same(protocol, role('protocol')) && role('preparation').prepared_sha256 === worker.prepared_sha256, 'History differs from the manifest protocol/device')
    if (job.submitted) for (const name of ['times_s', 'voltage_V', 'generation_scale'] as const)
      check(same(protocol[name], job.submitted.input[name]), 'History differs from the submitted protocol')
    const raw = object(JSON.parse(diagnosticsJson)), encoded = raw.attempts
    const values = Array.isArray(encoded) ? encoded : object(encoded).$tuple
    check(Array.isArray(values), 'Missing raw step attempts')
    const attempts = values.map(object), complete = attempt.state === 'succeeded'
    check(summary.outcome === (complete ? 'completed' : 'partial') && summary.reason === worker.reason
      && summary.accepted_count === accepted.length && summary.requested_count === times.length && summary.attempt_count === attempts.length,
    'Terminal state differs from the preserved transient outcome')
    check(complete ? accepted.length === times.length : accepted.length < times.length, 'Incorrect complete/partial history')
    check(attempts.length === accepted.length - 1 + (complete ? 0 : 1)
      && attempts.every((a, i) => a.accepted === (i < accepted.length - 1) && a.start_s === times[i] && a.end_s === times[i + 1]),
    'Diagnostics lost the first failed attempt or accepted prefix')
    return { state: attempt.state, arrays, summary, worker, attempts, diagnosticsJson, manifestJson, originalJson: snapshot.json }
  }
}
