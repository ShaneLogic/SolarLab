/** The explicitly installed ionic route and its immutable completed/failed histories. */
import { StoredRunClient, RunClientError } from './run-client'
import type { ArtifactRecord, AttemptIdentity, StoredObject } from './run-client'
import { readSteadyArray, requestJobService } from './steady-jv-client'
import type { FloatArray } from './steady-jv-client'
import { parseTransientSamples } from './transient-client'
import type { Bound } from './transient-client'

export const IONIC_SCOPE = 'classical_1d_dark_positive_diffusivity_ionic_transient'
export const IONIC_EXPERIMENT = 'ionic_transient_1d'
export const IONIC_STORED_UNITS: Record<string, string> = {
  requested_times_s: 's', requested_voltage_V: 'V', requested_generation_scale: '1', accepted_times_s: 's',
  nodes_m: 'm', ion_node_indices: '1', ion_cell_measures_m: 'm', potential_V: 'V',
  carrier_density_m3: 'm^-3', ion_density_m3: 'm^-3', ion_inventory_m2: 'm^-2',
  all_face_current_density_A_m2: 'A/m^2', terminal_current_density_A_m2: 'A/m^2',
}
const currents = ['carrier_conduction', 'ion_conduction', 'conduction', 'displacement', 'total'] as const
const terminalCurrents = ['conduction', 'displacement', 'total'] as const
type Budget = { max_iterations: number; max_residual_evaluations: number; max_jacobian_evaluations: number; max_backtracks: number }
export interface IonicProtocol {
  times_s: number[]; voltage_V: number[]; generation_scale: number[]; intervals_per_layer: number[]
  controls: { initial_steady: { budget: Budget; inventory_row_node: number; wall_s: number }
    transient: { newton_budget: Budget; max_steps: number; wall_s: number } }
}
export interface IonicRoute {
  id: string; version: number; devices: readonly { id: string; preparedSha256: string }[]
  bounds: Record<string, Bound>; domain: string; policy: StoredObject
}
export interface IonicCatalog {
  registry: string; schema: string; routes: readonly IonicRoute[]
  ceiling: { wall_seconds: number; rss_bytes: number; output_bytes: number }
}
export interface IonicJob {
  reader: StoredRunClient; target: Readonly<AttemptIdentity>
  submitted?: { catalog: IonicCatalog; device: IonicRoute['devices'][number]; input: IonicProtocol }
}
export interface IonicResult {
  state: 'succeeded' | 'failed'; arrays: Record<string, FloatArray>; storedArrays: Record<string, FloatArray>
  units: Record<string, string>; species: string[]; summary: StoredObject; worker: StoredObject; attempts: StoredObject[]
  diagnosticsJson: string; initialJson: string; originalJson: string; manifestJson: string
}

function check(value: unknown, message: string): asserts value { if (!value) throw new RunClientError('protocol', message) }
function object(value: unknown): StoredObject {
  check(value !== null && typeof value === 'object' && !Array.isArray(value), 'Expected an ionic record'); return value as StoredObject
}
function id(value: unknown): string {
  check(typeof value === 'string' && /^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$/.test(value), 'Invalid installed/stored identifier'); return value
}
function sha(value: unknown): string { check(typeof value === 'string' && /^[a-f0-9]{64}$/.test(value), 'Missing identity digest'); return value }
function numeric(value: unknown): number { check(typeof value === 'number' && Number.isFinite(value), 'Expected a finite number'); return value }
function same(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object' || Array.isArray(a) !== Array.isArray(b)) return false
  const x = a as StoredObject, y = b as StoredObject, keys = Object.keys(x)
  return keys.length === Object.keys(y).length && keys.every(k => Object.hasOwn(y, k) && same(x[k], y[k]))
}
function canonical(value: unknown): string {
  if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']'
  if (value && typeof value === 'object') return '{' + Object.keys(value).sort().map(k => JSON.stringify(k) + ':' + canonical((value as StoredObject)[k])).join(',') + '}'
  return JSON.stringify(value)
}
async function digest(bytes: Uint8Array): Promise<string> {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', Uint8Array.from(bytes))), b => b.toString(16).padStart(2, '0')).join('')
}
const text = (bytes: Uint8Array) => new TextDecoder('utf-8', { fatal: true }).decode(bytes)
function hexBytes(value: unknown): Uint8Array {
  check(typeof value === 'string' && value.length % 2 === 0 && /^[a-f0-9]+$/.test(value), 'Invalid bound source bytes')
  return Uint8Array.from(value.match(/../g)!, v => parseInt(v, 16))
}
async function document(value: unknown): Promise<{ data: StoredObject; sha256: string }> {
  const doc = object(value); check(typeof doc.base64 === 'string', 'Missing original request document')
  const bytes = Uint8Array.from(atob(doc.base64), c => c.charCodeAt(0)), hash = sha(doc.sha256)
  check(await digest(bytes) === hash, 'Stored request/catalog/schema bytes changed')
  return { data: object(JSON.parse(text(bytes))), sha256: hash }
}
function within(value: number, bound: Bound, name: string, integer = false) {
  check(Number.isFinite(value) && value >= bound.min && value <= bound.max && (!integer || Number.isSafeInteger(value)), `${name} must be within ${bound.min}–${bound.max}`)
}
function tuple(value: unknown): unknown[] {
  const values = Array.isArray(value) ? value : object(value).$tuple
  check(Array.isArray(values), 'Missing retained ionic sequence'); return values
}
function rawArray(value: unknown, dtype = '<f8'): FloatArray {
  const data = object(object(value).$array), real = object(data.real)
  check(data.schema === 'solarlab.numeric-array.v1' && data.dtype === dtype && data.imag === null
    && Array.isArray(data.shape) && Array.isArray(real.values) && Array.isArray(real.mask)
    && real.mask.length === real.values.length && real.mask.every(v => v === false)
    && Array.isArray(real.nonfinite_words) && real.nonfinite_words.length === real.values.length && real.nonfinite_words.every(v => v === null),
  'Accepted raw ionic array is nonfinite or unsupported')
  const shape = data.shape.map(numeric), values = real.values.map(numeric)
  check(shape.every(n => Number.isSafeInteger(n) && n >= 0) && shape.reduce((a, b) => a * b, 1) === values.length, 'Invalid raw array shape')
  return { shape, values }
}

/** Existing float reader remains strict; only ionic rank-three packs and int64 indices add decoding here. */
export function readIonicArray(bytes: Uint8Array, record: ArtifactRecord): FloatArray {
  const shape = record.metadata.shape
  if (record.metadata.dtype === '<f8' && Array.isArray(shape) && shape.length <= 2) return readSteadyArray(bytes, record, true)
  check(record.format === 'npy' && bytes.length >= 10 && Array.from(bytes.slice(0, 6)).join(',') === '147,78,85,77,80,89', 'Invalid ionic NPY')
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), version = bytes[6], prefix = version === 1 ? 10 : 12
  check((version === 1 || version === 2) && bytes[7] === 0 && bytes.length >= prefix, 'Unsupported ionic NPY version')
  const length = version === 1 ? view.getUint16(8, true) : view.getUint32(8, true)
  check(length <= 4096 && prefix + length <= bytes.length, 'Invalid ionic NPY header')
  const match = /^\{'descr': '([<>][fi]8)', 'fortran_order': False, 'shape': \((\d+(?:,\s*\d+)*,?)\),?\s*\}\s*$/.exec(text(bytes.slice(prefix, prefix + length)))
  check(match, 'Only plain numeric ionic packs are supported')
  const dimensions = match[2].split(',').filter(v => v.trim()).map(Number), integer = match[1].includes('i')
  check(dimensions.length === (integer ? 1 : 3) && dimensions.every(n => Number.isSafeInteger(n) && n >= 0), 'Unexpected ionic pack rank')
  const count = dimensions.reduce((a, b) => a * b, 1)
  check(Number.isSafeInteger(count) && prefix + length + count * 8 === bytes.length && same(shape, dimensions)
    && record.metadata.dtype === match[1] && record.metadata.order === 'C', 'Ionic NPY metadata/size mismatch')
  const values = Array.from({ length: count }, (_, i) => integer ? Number(view.getBigInt64(prefix + length + i * 8, match[1][0] === '<'))
    : view.getFloat64(prefix + length + i * 8, match[1][0] === '<'))
  check(values.every(v => Number.isFinite(v) && (!integer || Number.isSafeInteger(v))), 'Invalid accepted ionic array values')
  return { shape: dimensions, values }
}

export const parseIonicSamples = parseTransientSamples

export class IonicTransientClient {
  readonly endpoint: string
  constructor(endpoint: string | URL) {
    const url = new URL(endpoint)
    check(['http:', 'https:'].includes(url.protocol) && !url.username && !url.password && !url.search && !url.hash, 'Use an explicit HTTP endpoint')
    this.endpoint = url.href.replace(/\/$/, '') + '/'
  }
  async discover(signal?: AbortSignal): Promise<IonicCatalog> {
    const { value, response } = await requestJobService(this.endpoint, 'experiments', { signal })
    const registry = sha(response.headers.get('X-Solarlab-Registry')), schema = sha(response.headers.get('X-Solarlab-Job-Schema'))
    check(value.schema === 'solarlab.execution-catalog.v1' && value.job_schema_sha256 === schema
      && response.headers.get('etag') === `"${registry}"` && Array.isArray(value.bindings), 'Inconsistent installed catalog')
    const cap = object(value.resource_ceiling)
    check(cap.calls === null && cap.accepted_steps === null, 'Ionic counter caps are unsupported')
    const ceiling = { wall_seconds: Math.min(30, numeric(cap.wall_seconds)), rss_bytes: numeric(cap.rss_bytes), output_bytes: numeric(cap.output_bytes) }
    check(Object.values(ceiling).every(v => v > 0), 'Invalid resource ceiling')
    const routes: IonicRoute[] = []
    for (const raw of value.bindings) {
      const entry = object(raw)
      if (entry.engine !== 'solarlab' || entry.driver !== IONIC_EXPERIMENT || entry.experiment_id !== IONIC_EXPERIMENT
        || entry.version !== 1 || entry.scope !== IONIC_SCOPE) continue
      const schema = object(entry.input_schema), properties = object(schema.properties), installed = object(entry.preparation_context)
      const keys = ['device_id', 'times_s', 'voltage_V', 'generation_scale', 'intervals_per_layer', 'controls'].sort()
      check(schema.additionalProperties === false && same(Object.keys(properties).sort(), keys)
        && Array.isArray(schema.required) && same([...schema.required].sort(), keys), 'Unsupported ionic request schema')
      const definition = (raw: unknown) => {
        const data = object(raw)
        if (!data.$ref) return data
        check(typeof data.$ref === 'string' && /^#\/\$defs\/[A-Za-z0-9_]+$/.test(data.$ref), 'Unsupported control reference')
        return object(object(schema.$defs)[data.$ref.slice(8)])
      }
      const bound = (raw: unknown): Bound => {
        const data = object(raw), min = numeric(data.minimum ?? data.exclusiveMinimum), max = numeric(data.maximum)
        check(min <= max, 'Invalid installed control bounds'); return { min, max }
      }
      const controls = definition(properties.controls), c = object(controls.properties)
      const initial = definition(c.initial_steady), ip = object(initial.properties), time = definition(c.transient), tp = object(time.properties)
      const ib = definition(ip.budget), tb = definition(tp.newton_budget)
      check([controls, initial, time, ib, tb].every(v => v.additionalProperties === false), 'Unbounded ionic work controls')
      const samples = object(properties.times_s), intervals = object(properties.intervals_per_layer)
      const bounds: Record<string, Bound> = { samples: { min: numeric(samples.minItems), max: Math.min(17, numeric(samples.maxItems)) },
        layers: { min: numeric(intervals.minItems), max: numeric(intervals.maxItems) }, intervals_per_layer: bound(intervals.items),
        times_s: bound(samples.items), voltage_V: bound(object(properties.voltage_V).items), generation_scale: bound(object(properties.generation_scale).items),
        max_steps: bound(tp.max_steps), wall_s: bound(tp.wall_s), initial_wall_s: bound(ip.wall_s), inventory_row_node: bound(ip.inventory_row_node) }
      for (const key of ['max_iterations', 'max_residual_evaluations', 'max_jacobian_evaluations', 'max_backtracks']) {
        bounds[key] = bound(object(tb.properties)[key]); bounds['initial_' + key] = bound(object(ib.properties)[key])
      }
      // The installed v1 time validator is stricter than its shared budget definition.
      for (const [key, max] of Object.entries({ max_iterations: 50, max_residual_evaluations: 500, max_jacobian_evaluations: 50 })) bounds[key].max = Math.min(bounds[key].max, max)
      check(bounds.generation_scale.min === 0 && bounds.generation_scale.max === 0 && same(installed.ion_domain_endpoints, ['blocking', 'blocking'])
        && Array.isArray(installed.devices), 'Only installed dark blocking ionic devices are admitted')
      const devices = installed.devices.map(raw => { const d = object(raw); return { id: id(d.device_id), preparedSha256: sha(d.prepared_sha256) } })
      check(new Set(devices.map(d => d.id)).size === devices.length && new Set(devices.map(d => d.preparedSha256)).size === devices.length, 'Ambiguous installed devices')
      routes.push({ id: IONIC_EXPERIMENT, version: 1, devices, bounds, domain: id(installed.ion_domain_layer), policy: object(installed.fixed_numerical_policy) })
    }
    return { registry, schema, routes, ceiling }
  }
  async submit(catalog: IonicCatalog, route: IonicRoute, deviceId: string, input: IonicProtocol, wallSeconds: number,
    runId = `ionic-transient-${crypto.randomUUID()}`): Promise<IonicJob> {
    check(catalog.routes.includes(route), 'Select the installed ionic route')
    const device = route.devices.find(d => d.id === deviceId); check(device, 'Select an installed ionic device')
    const b = route.bounds; within(input.times_s.length, b.samples, 'Sample count', true)
    for (const key of ['times_s', 'voltage_V', 'generation_scale'] as const) {
      check(input[key].length === input.times_s.length, 'Sample lengths differ')
      input[key].forEach(v => within(v, b[key], key))
    }
    check(input.times_s.every((v, i) => !i || v > input.times_s[i - 1]), 'Time must be strictly increasing')
    within(input.intervals_per_layer.length, b.layers, 'Layer count', true)
    input.intervals_per_layer.forEach(v => within(v, b.intervals_per_layer, 'Layer intervals', true))
    const initial = input.controls.initial_steady, time = input.controls.transient
    within(initial.inventory_row_node, b.inventory_row_node, 'Inventory constraint node', true)
    within(initial.wall_s, b.initial_wall_s, 'Initial wall budget'); within(time.wall_s, b.wall_s, 'Transient wall budget')
    within(time.max_steps, b.max_steps, 'Step budget', true)
    check(initial.wall_s > 0 && time.wall_s > 0 && input.times_s.length - 1 <= time.max_steps, 'Invalid step/time budget')
    for (const key of Object.keys(time.newton_budget) as (keyof Budget)[]) {
      within(time.newton_budget[key], b[key], key, true); within(initial.budget[key], b['initial_' + key], 'Initial ' + key, true)
    }
    check(wallSeconds > 0 && wallSeconds <= catalog.ceiling.wall_seconds, 'Invalid whole-job wall budget')
    const frozen = JSON.parse(JSON.stringify(input)) as IonicProtocol
    const { value, response } = await requestJobService(this.endpoint, 'jobs', { method: 'POST', headers: {
      'Content-Type': 'application/json', 'X-Solarlab-Registry': catalog.registry, 'X-Solarlab-Job-Schema': catalog.schema,
    }, body: JSON.stringify({ schema_version: 'solarlab.job-submission.v1', run_id: id(runId), experiment_id: route.id, version: 1,
      engine: 'solarlab', input: { device_id: deviceId, ...frozen }, limits: { ...catalog.ceiling, wall_seconds: wallSeconds, calls: null, accepted_steps: null } }) })
    check(response.status === 202 && value.status === 'accepted' && value.run_id === runId && value.execution_catalog_sha256 === catalog.registry
      && value.job_schema_sha256 === catalog.schema, 'Ionic acceptance identity mismatch')
    return { ...this.open(id(value.store_id), runId, id(value.attempt_id)), submitted: { catalog, device, input: frozen } }
  }
  open(storeId: string, runId: string, attemptId: string): IonicJob {
    return { reader: new StoredRunClient(this.endpoint, id(storeId)), target: Object.freeze({ runId: id(runId), attemptId: id(attemptId) }) }
  }
  async cancel(job: IonicJob): Promise<void> {
    const run = (await job.reader.getRun(job.target.runId)).value
    check(run.current_attempt_id === job.target.attemptId && ['queued', 'running'].includes(run.state), 'Cancellation target is stale or terminal')
    const { value } = await requestJobService(this.endpoint, `jobs/${encodeURIComponent(job.target.runId)}/attempts/${encodeURIComponent(job.target.attemptId)}/cancel`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: `{"expected_generation":${run.generation.toString()},"reason":"User requested ionic transient cancellation"}`,
    })
    check(value.run_id === job.target.runId && value.attempt_id === job.target.attemptId, 'Cancellation identity mismatch')
  }
  async readResult(job: IonicJob, signal?: AbortSignal): Promise<IonicResult | undefined> {
    const [snapshot, run] = await Promise.all([job.reader.getAttempt(job.target, signal), job.reader.getRun(job.target.runId, signal)])
    const attempt = snapshot.value; check(attempt.terminal, 'Attempt is not committed')
    if (attempt.state === 'cancelled') { check(!attempt.terminal.artifacts.length, 'Cancelled attempt cannot invent ionic history'); return undefined }
    check(attempt.state === 'succeeded' || attempt.state === 'failed', 'Unsupported terminal state')
    if (!attempt.terminal.artifacts.length && attempt.state === 'failed') return undefined
    const worker = object(attempt.result?.worker_result), metadata = object(run.value.request.input_metadata), identity = object(run.value.request.identity)
    const documents = object(object(metadata.transport).documents)
    const [registry, submitted, schema] = await Promise.all([document(documents.registry), document(documents.request), document(documents.schema)])
    const catalog = registry.data, execution = object(metadata.execution_manifest), capture = object(worker.capture)
    check(worker.schema === 'solarlab.ionic-transient.worker-result.v1' && worker.engine === 'solarlab' && worker.scientific_qualification === false
      && worker.execution_catalog_sha256 === registry.sha256 && catalog.job_schema_sha256 === schema.sha256, 'Not the bound ionic worker')
    for (const key of ['H_input', 'H_physics', 'H_execution']) check(sha(worker[key]) === identity[key], 'Ionic execution identity mismatch')
    check(worker.manifest_sha256 === execution.manifest_sha256 && worker.H_execution === execution.H_execution
      && capture.request_sha256 === submitted.sha256, 'Ionic manifest/request capture mismatch')
    check(Array.isArray(catalog.bindings), 'Missing recorded catalog')
    const entries = catalog.bindings.map(object).filter(e => e.driver === IONIC_EXPERIMENT)
    check(entries.length === 1, 'Ambiguous ionic registration')
    const entry = entries[0], installed = object(entry.preparation_context)
    check(entry.experiment_id === IONIC_EXPERIMENT && entry.version === 1 && entry.engine === 'solarlab' && entry.scope === IONIC_SCOPE
      && worker.experiment_id === IONIC_EXPERIMENT && worker.experiment_version === 1 && Array.isArray(installed.devices), 'Wrong ionic route or version')
    const devices = installed.devices.map(object).filter(d => d.prepared_sha256 === worker.prepared_sha256)
    check(devices.length === 1 && devices[0].device_id === worker.registered_device_id && entry.model_registry_sha256 === worker.registered_model_registry_sha256
      && same(installed.server_sources, worker.server_sources) && same(installed.server_runtime, worker.server_runtime)
      && same(installed.fixed_numerical_policy, worker.registered_numerical_policy), 'Ionic registered device/source/policy mismatch')
    const definitions = object(catalog.registry).experiments
    check(Array.isArray(definitions), 'Missing recorded experiment schema')
    const definition = definitions.map(object).find(d => d.id === IONIC_EXPERIMENT && d.version === 1)
    check(definition && object(definition.parameter_schema).sha256 === worker.registered_input_schema_sha256, 'Ionic input schema identity mismatch')
    const summary = object(worker.ionic_result), expected = object(worker.payloads), species = tuple(summary.ion_species).map(String)
    check(summary.schema === 'solarlab.ionic-transient-result.v1' && summary.engine === 'solarlab' && summary.physical_certificate_status === 'not_evaluated'
      && (same(species, ['positive']) || same(species, ['positive', 'negative'])), 'Unsupported ionic history scope')
    const components: Record<string, string[]> = {
      carrier_density_m3: ['electron_density_m3', 'hole_density_m3'], ion_density_m3: species.map(s => s + '_ion_density_m3'),
      ion_inventory_m2: species.map(s => s + '_ion_inventory_m2'), all_face_current_density_A_m2: currents.map(s => s + '_current_density_A_m2'),
      terminal_current_density_A_m2: terminalCurrents.map(s => 'terminal_' + s + '_current_density_A_m2'),
    }
    const keys = [...Object.keys(IONIC_STORED_UNITS), 'ionic_initial_steady', 'ionic_transient_diagnostics']
    check(same(Object.keys(expected).sort(), keys.sort()) && attempt.terminal.artifacts.length === 16, 'Incomplete ionic artifact roles')
    const storedArrays: Record<string, FloatArray> = {}, names = new Set<string>(), hashes: Record<string, string> = {}
    let manifestJson = '', diagnosticsJson = '', initialJson = ''
    for (const reference of attempt.terminal.artifacts) {
      const artifact = await job.reader.getArtifact(job.target, reference, signal), record = artifact.record.value, name = String(record.name)
      check(!names.has(name) && String(record.generation) === String(attempt.generation), 'Duplicate or foreign ionic artifact'); names.add(name)
      if (name === 'execution_manifest') { check(record.sha256 === worker.manifest_sha256, 'Wrong execution manifest'); manifestJson = text(artifact.bytes); continue }
      const binding = object(expected[name]); hashes[name] = String(record.sha256)
      check(binding.sha256 === record.sha256 && binding.format === record.format && same(binding.metadata, record.metadata)
        && record.metadata.schema === summary.schema && record.metadata.compiled_identity === summary.compiled_identity, 'Ionic payload binding mismatch')
      if (name in IONIC_STORED_UNITS) {
        check(record.metadata.unit === IONIC_STORED_UNITS[name] && record.metadata.dtype === (name === 'ion_node_indices' ? '<i8' : '<f8')
          && same(record.metadata.components, components[name] ?? []), 'Ionic unit/dtype/component order mismatch')
        storedArrays[name] = readIonicArray(artifact.bytes, record)
      } else if (name === 'ionic_initial_steady') { check(record.metadata.role === 'actual_cold_initial', 'Wrong initial evidence'); initialJson = text(artifact.bytes) }
      else { check(name === 'ionic_transient_diagnostics' && record.metadata.role === 'raw_ionic_history', 'Wrong ionic diagnostics'); diagnosticsJson = text(artifact.bytes) }
    }
    check(names.size === 16 && Object.keys(storedArrays).length === 13 && manifestJson && diagnosticsJson && initialJson, 'Incomplete ionic history')
    const manifest = object(JSON.parse(manifestJson)), bound = object(object(manifest.execution_identity).payload), roles = object(manifest.content_roles)
    const role = async (name: string) => { const doc = object(roles[name]), bytes = hexBytes(doc.content_hex); check(await digest(bytes) === sha(doc.sha256), 'Manifest role bytes differ'); return object(JSON.parse(text(bytes))) }
    const [protocol, preparation, grid, controls] = await Promise.all(['protocol', 'preparation', 'grid', 'numeric_controls'].map(role))
    check(bound.H_input === worker.H_input && bound.H_physics === worker.H_physics && object(bound.engine).name === 'solarlab'
      && capture.manifest_sha256 === worker.manifest_sha256 && same(capture.runtime, bound.environment), 'Manifest/runtime identity mismatch')
    const sourceBindings: StoredObject = {}
    check(Array.isArray(manifest.implementation_content) && Array.isArray(preparation.adapter_sources), 'Missing computation source capture')
    for (const raw of [...manifest.implementation_content, ...preparation.adapter_sources]) { const s = object(raw); sourceBindings[String(s.id)] = sha(s.sha256) }
    check(same(capture.source_sha256, sourceBindings) && capture.build_source_commit === object(bound.implementation).build_source_commit
      && capture.compiled_identity === summary.compiled_identity && preparation.prepared_sha256 === worker.prepared_sha256, 'Actual source/compiled/configuration binding differs')
    check(submitted.data.schema === 'solarlab.ionic-transient.worker-request.v1' && submitted.data.engine === 'solarlab'
      && submitted.data.prepared_sha256 === worker.prepared_sha256 && same(submitted.data.protocol, protocol)
      && same(submitted.data.grid, grid) && same(submitted.data.controls, controls), 'Original ionic request differs from manifest')
    const arrays: Record<string, FloatArray> = {}, units: Record<string, string> = {}
    for (const [name, pack] of Object.entries(storedArrays)) {
      if (!components[name]) { arrays[name] = pack; units[name] = IONIC_STORED_UNITS[name]; continue }
      const labels = components[name]; check(pack.shape[0] === labels.length, 'Wrong packed component count')
      const shape = pack.shape.slice(1), count = shape.reduce((a, b) => a * b, 1)
      labels.forEach((key, i) => { arrays[key] = { shape, values: pack.values.slice(i * count, (i + 1) * count) }; units[key] = IONIC_STORED_UNITS[name] })
    }
    const t = arrays.accepted_times_s.values, requested = arrays.requested_times_s.values, n = arrays.nodes_m.values.length, m = arrays.ion_node_indices.values.length, count = t.length
    check(requested.length >= 2 && count <= requested.length && t.every((v, i) => Object.is(v, requested[i]))
      && requested.every((v, i) => !i || v > requested[i - 1]) && n >= 3 && m >= 2, 'Invalid accepted ionic prefix/grid')
    check(arrays.nodes_m.values.every((v, i, a) => !i || v > a[i - 1]) && arrays.ion_node_indices.values.every((v, i, a) =>
      Number.isSafeInteger(v) && v >= 0 && v < n && (!i || v === a[i - 1] + 1)), 'Invalid ionic node mapping')
    arrays.ion_nodes_m = { shape: [m], values: arrays.ion_node_indices.values.map(i => arrays.nodes_m.values[i]) }; units.ion_nodes_m = 'm'
    const raw = object(JSON.parse(diagnosticsJson)), history = raw.history === null ? undefined : object(raw.history)
    const compiled = object(raw.compiled_binding), domain = object(compiled.ion_domain), layout = object(compiled.state_layout), model = object(compiled.ion_model)
    check(compiled.original_prepared_sha256 === summary.prepared_sha256 && summary.prepared_sha256 === worker.prepared_sha256
      && same(domain.contacts, ['blocking', 'blocking']) && domain.layer === grid.ion_domain_layer
      && same(domain.electrical_node_indices, arrays.ion_node_indices.values) && same(domain.cell_measures_m, arrays.ion_cell_measures_m.values)
      && arrays.ion_cell_measures_m.values.every(v => v > 0) && layout.size === raw.state_size
      && layout.electrical_node_count === n && layout.ion_node_count === m && same(layout.ion_species ?? ['positive'], species)
      && summary.ion_site_contract === (model.site_contract ?? 'independent_finite_sites'), 'Compiled ionic domain/layout differs')
    for (const name of ['face_diffusivity_m2_s', ...(species.length === 2 ? ['negative_face_diffusivity_m2_s'] : [])]) {
      check(Array.isArray(model[name]) && model[name].length === m - 1 && model[name].every(v => typeof v === 'number' && Number.isFinite(v) && v > 0), 'Only positive diffusivity is supported')
    }
    for (const name of ['nodes_m', 'ion_nodes_m', 'ion_node_indices', 'ion_cell_measures_m']) {
      check(same(rawArray(object(raw.grid)[name], name === 'ion_node_indices' ? '<i8' : '<f8'), arrays[name]), 'Raw grid differs from stored ionic arrays')
    }
    const states = history ? tuple(history.states).map(object) : [], attempts = history ? tuple(history.attempts).map(object) : []
    check(states.length === count && raw.state_size === 3 * n + species.length * m, 'Ionic state layout differs')
    arrays.state_coordinates = { shape: [count, Number(raw.state_size)], values: states.flatMap(s => rawArray(s.u).values) }
    units.state_coordinates = 'mixed: first N in V; remaining coordinates dimensionless'
    for (const s of species) {
      const density = arrays[s + '_ion_density_m3']
      arrays[s + '_ion_cell_inventory_m2'] = { shape: density.shape, values: density.values.map((v, i) => v * arrays.ion_cell_measures_m.values[i % m]) }
      units[s + '_ion_cell_inventory_m2'] = 'm^-2'
    }
    for (const [name, a] of Object.entries(arrays)) {
      const shape = name === 'nodes_m' ? [n] : ['ion_nodes_m', 'ion_node_indices', 'ion_cell_measures_m'].includes(name) ? [m]
        : name.startsWith('requested_') ? [requested.length] : name === 'accepted_times_s' ? [count]
          : name === 'state_coordinates' ? [count, Number(raw.state_size)] : name.includes('current_density') ? [Math.max(0, count - 1), name.startsWith('terminal_') ? 2 : n + 1]
            : name.endsWith('_ion_inventory_m2') ? [count] : [count, name.includes('_ion_') ? m : n]
      check(same(a.shape, shape) && a.values.length === shape.reduce((a, b) => a * b, 1), 'Ionic logical array shape mismatch')
    }
    check(same(protocol, { times_s: requested, voltage_V: arrays.requested_voltage_V.values, generation_scale: arrays.requested_generation_scale.values })
      && arrays.requested_generation_scale.values.every(v => v === 0), 'Recorded ionic driving samples differ or are not dark')
    const complete = attempt.state === 'succeeded', initial = object(raw.initial), initialDoc = object(JSON.parse(initialJson))
    check(summary.outcome === (complete ? 'completed' : count ? 'partial' : 'initial_failed') && summary.reason === worker.reason
      && summary.accepted_count === count && summary.requested_count === requested.length && summary.attempt_count === attempts.length,
    'Terminal state cannot relabel the ionic numerical outcome')
    check(complete ? count === requested.length : count < requested.length, 'Incorrect completed/failed ionic prefix')
    check(history ? attempts.length === Math.max(0, count - 1) + (complete ? 0 : 1) && attempts.every((a, i) =>
      a.accepted === (i < count - 1) && a.start_s === requested[i] && a.end_s === requested[i + 1]) : count === 0 && initial.accepted === false,
    'First failure or unattempted tail was lost')
    check(capture.initial_artifact_sha256 === hashes.ionic_initial_steady && capture.time_solve_started === initial.accepted
      && capture.scope === 'cold_ionic_compile_steady_and_time_in_worker' && capture.prepared_sha256 === worker.prepared_sha256
      && object(initialDoc.summary).accepted === initial.accepted && object(initialDoc.summary).qualification === false
      && initial.compiled_identity === summary.compiled_identity && object(initialDoc.summary).compiled_identity === summary.compiled_identity
      && same(initial.solver_controls, controls.initial_steady), 'Cold initial evidence differs')
    for (const name of ['times_s', 'voltage_V', 'generation_scale']) {
      check(same(rawArray(object(raw.protocol)[name]), arrays['requested_' + name]), 'Raw requested protocol differs')
    }
    if (history) {
      check(history.physical_certificate_status === 'not_evaluated' && history.compiled_identity === summary.compiled_identity
        && same(history.binding_manifest, raw.compiled_binding) && object(history.initial_provenance).source_sha256 === hashes.ionic_initial_steady
        && history.original_prepared_identity === worker.prepared_sha256 && same(rawArray(history.accepted_times_s), arrays.accepted_times_s), 'History initial/source provenance differs')
      const initialState = rawArray(object(initial.state).u), words = new Uint8Array(initialState.values.length * 8), wordView = new DataView(words.buffer)
      initialState.values.forEach((v, i) => wordView.setFloat64(i * 8, v, true))
      check(same(initialState, rawArray(states[0].u)) && same(capture.initial_state, { dtype: '<f8', shape: initialState.shape, sha256: await digest(words) }), 'Cold initial state identity differs')
      for (const [name, value] of Object.entries(object(controls.transient))) check(same(object(history.solver_controls)[name], value), 'Recorded transient controls differ')
      for (const name of ['times_s', 'voltage_V', 'generation_scale']) check(same(rawArray(object(history.protocol)[name]), arrays['requested_' + name]), 'History protocol differs')
      for (let i = 0; i < count; i++) {
        const state = states[i], electronic = object(state.electronic_state)
        check(state.device_sha256 === summary.compiled_identity && electronic.voltage_V === arrays.requested_voltage_V.values[i]
          && electronic.device_sha256 === object(compiled.electronic_component).compiled_sha256
          && same(rawArray(electronic.u).values, rawArray(state.u).values.slice(0, 3 * n)), 'State belongs to another point')
        for (const name of ['potential_V', 'electron_density_m3', 'hole_density_m3']) check(same(rawArray(electronic[name]).values, arrays[name].values.slice(i * n, (i + 1) * n)), 'Electronic fields differ from raw producer')
        for (const s of species) {
          check(same(rawArray(state[s + '_ion_density_m3']).values, arrays[s + '_ion_density_m3'].values.slice(i * m, (i + 1) * m)), 'Ionic density differs from producer')
          check(Object.is(state[s === 'positive' ? 'domain_inventory_m2' : 'negative_domain_inventory_m2'], arrays[s + '_ion_inventory_m2'].values[i]), 'Species inventory differs from producer')
        }
      }
      attempts.filter(a => a.accepted).forEach((a, i) => { for (const kind of currents) {
        const key = kind + '_current_density_A_m2'; check(same(rawArray(object(a.accounting)[key]).values, arrays[key].values.slice(i * (n + 1), (i + 1) * (n + 1))), 'Current accounting differs from producer')
      } })
      for (const kind of terminalCurrents) {
        const faces = arrays[kind + '_current_density_A_m2'].values, ports = arrays['terminal_' + kind + '_current_density_A_m2'].values
        check(ports.every((v, i) => Object.is(v, faces[Math.floor(i / 2) * (n + 1) + (i % 2 ? n : 0)])), 'Terminal current is not an exact port selection')
      }
      check(arrays.conduction_current_density_A_m2.values.every((v, i) => Object.is(v, arrays.carrier_conduction_current_density_A_m2.values[i] + arrays.ion_conduction_current_density_A_m2.values[i]))
        && arrays.total_current_density_A_m2.values.every((v, i) => Object.is(v, arrays.conduction_current_density_A_m2.values[i] + arrays.displacement_current_density_A_m2.values[i])), 'Ionic current components differ from their totals')
    }
    check(capture.result_sha256 === await digest(new TextEncoder().encode(canonical({ summary, payloads: hashes }))), 'Ionic result digest differs')
    if (job.submitted) check(worker.prepared_sha256 === job.submitted.device.preparedSha256 && registry.sha256 === job.submitted.catalog.registry
      && same(protocol, { times_s: job.submitted.input.times_s, voltage_V: job.submitted.input.voltage_V, generation_scale: job.submitted.input.generation_scale }), 'Result differs from the submitted device/protocol')
    return { state: attempt.state, arrays, storedArrays, units, species, summary, worker, attempts, diagnosticsJson, initialJson, manifestJson, originalJson: snapshot.json }
  }
}
