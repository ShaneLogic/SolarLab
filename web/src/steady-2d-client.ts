/** Fixed, installed full-2D steady jobs. Opening and reconnecting only read saved bytes. */
import { RunClientError, StoredRunClient } from './run-client'
import type { ArtifactRecord, AttemptIdentity, StoredObject } from './run-client'
import { readSteadyArray, requestJobService } from './steady-jv-client'
import type { FloatArray } from './steady-jv-client'
import { readIonicArray } from './ionic-transient-client'

export const STEADY_2D_EXPERIMENT = 'steady_2d'
export const STEADY_2D_SCOPE = 'uniform_no_grain_classical_2d_steady'
const schemaId = 'solarlab.steady-2d.result.v1'
const coordinateUnit = 'mixed: phi[V], electron/hole dimensionless log coordinates; C order'
const residualUnit = 'mixed: Gauss[C/m depth], carrier[particles/(m depth*s)], contact[V or dimensionless log]'
export const STEADY_2D_UNITS: Record<string, string> = {
  x_m: 'm', y_m: 'm', cell_areas_m2: 'm^2', face_lengths_m: 'm', face_spacing_m: 'm', face_pairs: '1',
  accepted_indices: '1', accepted_voltage_V: 'V', accepted_generation_scale: '1', state_coordinates: coordinateUnit,
  potential_V: 'V', electron_density_m3: 'm^-3', hole_density_m3: 'm^-3', face_current_density_A_m2: 'A/m^2',
  lateral_current_density_A_m2: 'A/m^2', stack_current_density_A_m2: 'A/m^2', terminal_current_per_depth_A_m: 'A/m',
  cross_section_current_per_depth_A_m: 'A/m', terminal_mean_current_density_A_m2: 'A/m^2', original_residual: residualUnit,
  face_displacement_C_m2: 'C/m^2', carrier_face_particle_flux_m2_s: 'm^-2 s^-1',
}
const storedUnits: Record<string, string> = { x_m: 'm', y_m: 'm', cell_areas_m2: 'm^2', face_geometry_m: 'm',
  face_pairs: '1', accepted_indices: '1', state_coordinates: coordinateUnit, potential_V: 'V', carrier_density_m3: 'm^-3',
  face_current_density_A_m2: 'A/m^2', section_terminal_current_A_m: 'A/m', original_residual: residualUnit }
const components: Record<string, string[]> = { face_geometry_m: ['face_lengths_m', 'face_spacing_m'],
  carrier_density_m3: ['electron_density_m3', 'hole_density_m3'],
  section_terminal_current_A_m: ['lower_terminal', 'stack_cross_sections', 'upper_terminal'] }
const geometry = { lateral_length_m: 1e-6, Nx: 4, Ny_per_layer: 4, lateral_bc: 'neumann', geometry_policy: 'rectangular_half_endpoint_neumann_2d_v1' }

export interface Steady2DInput {
  voltage_V: number; generation_scales: number[]; spatial: StoredObject; controls: StoredObject; initial: StoredObject
}
export interface Steady2DRoute {
  id: string; version: number; input: Steady2DInput; generations: number[][]
  devices: readonly { id: string; preparedSha256: string }[]
}
export interface Steady2DCatalog {
  registry: string; schema: string; routes: readonly Steady2DRoute[]
  ceiling: { wall_seconds: number; rss_bytes: number; output_bytes: number }
}
export interface Steady2DJob {
  reader: StoredRunClient; target: Readonly<AttemptIdentity>
  submitted?: { catalog: Steady2DCatalog; device: Steady2DRoute['devices'][number]; input: Steady2DInput }
}
export interface Steady2DResult {
  state: 'succeeded' | 'failed'; registered: boolean; arrays: Record<string, FloatArray>; units: Record<string, string>
  storedArrays: Record<string, FloatArray>; attempts: StoredObject[]; targets: StoredObject[]; summary: StoredObject; worker: StoredObject
  originalJson: string; diagnosticsJson: string; manifestJson: string
}

function check(value: unknown, message: string): asserts value { if (!value) throw new RunClientError('protocol', message) }
function object(value: unknown): StoredObject {
  check(value !== null && typeof value === 'object' && !Array.isArray(value), 'Expected a 2D record'); return value as StoredObject
}
function id(value: unknown): string {
  check(typeof value === 'string' && /^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$/.test(value), 'Invalid installed/stored identifier'); return value
}
function sha(value: unknown): string { check(typeof value === 'string' && /^[a-f0-9]{64}$/.test(value), 'Missing identity digest'); return value }
function number(value: unknown): number { check(typeof value === 'number' && Number.isFinite(value), 'Expected a finite number'); return value }
function same(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object' || Array.isArray(a) !== Array.isArray(b)) return false
  const x = a as StoredObject, y = b as StoredObject, keys = Object.keys(x)
  return keys.length === Object.keys(y).length && keys.every(k => Object.hasOwn(y, k) && same(x[k], y[k]))
}
/** Stored reads also admit the installation's explicit failure-test budget. */
export function admitsSteady2DControls(policy: StoredObject, controls: unknown): boolean {
  return same(policy.controls, controls) || same(policy.insufficient_budget_controls, controls)
}
const text = (bytes: Uint8Array) => new TextDecoder('utf-8', { fatal: true }).decode(bytes)
async function digest(bytes: Uint8Array): Promise<string> {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', Uint8Array.from(bytes))), b => b.toString(16).padStart(2, '0')).join('')
}
async function document(value: unknown): Promise<{ data: StoredObject; sha256: string }> {
  const doc = object(value); check(typeof doc.base64 === 'string', 'Missing original request document')
  const bytes = Uint8Array.from(atob(doc.base64), c => c.charCodeAt(0)), hash = sha(doc.sha256)
  check(await digest(bytes) === hash, 'Stored request/catalog/schema bytes changed')
  return { data: object(JSON.parse(text(bytes))), sha256: hash }
}
function tuple(value: unknown): unknown[] {
  const values = Array.isArray(value) ? value : object(value).$tuple
  check(Array.isArray(values), 'Missing retained sequence'); return values
}
function rawArray(value: unknown, dtype = '<f8'): FloatArray {
  const data = object(object(value).$array), real = object(data.real)
  check(data.schema === 'solarlab.numeric-array.v1' && data.dtype === dtype && data.imag === null && Array.isArray(data.shape)
    && Array.isArray(real.values) && Array.isArray(real.mask) && real.mask.length === real.values.length && real.mask.every(v => v === false)
    && Array.isArray(real.nonfinite_words) && real.nonfinite_words.length === real.values.length && real.nonfinite_words.every(v => v === null),
  'Accepted raw array is nonfinite or unsupported')
  const shape = data.shape.map(number), values = real.values.map(number)
  check(shape.every(n => Number.isSafeInteger(n) && n >= 0) && shape.reduce((a, b) => a * b, 1) === values.length, 'Invalid raw array shape')
  return { shape, values }
}
// Same source-aware JSON digest convention as the stored ionic J-V client.
function canonicalRecorded(value: unknown): string {
  if (typeof value === 'number') throw new RunClientError('verification_unavailable', 'Original numeric spelling is unavailable')
  if (Array.isArray(value)) return '[' + value.map(canonicalRecorded).join(',') + ']'
  if (value && typeof value === 'object') {
    if (value instanceof Number) return String((value as Number & { source: string }).source)
    return '{' + Object.keys(value).sort().map(k => canonicalRecorded(k) + ':' + canonicalRecorded(object(value)[k])).join(',') + '}'
  }
  return JSON.stringify(value).replace(/[\u007f-\uffff]/g, c => '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0'))
}
function recordedJson(json: string): StoredObject {
  return object(JSON.parse(json, (_k: string, value: unknown, context?: { source?: string }) =>
    typeof value === 'number' && context?.source ? Object.assign(new Number(value), { source: context.source }) : value))
}

/** Reuse existing readers; the only extra layouts are int64 face pairs and the rank-four carrier pack. */
function readPack(bytes: Uint8Array, record: ArtifactRecord): FloatArray {
  const shape = record.metadata.shape
  check(Array.isArray(shape), 'Missing 2D pack shape')
  if (record.metadata.dtype === '<f8' && shape.length <= 2) return readSteadyArray(bytes, record, true)
  if (shape.length === 1 || shape.length === 3) return readIonicArray(bytes, record)
  check(record.format === 'npy' && bytes.length >= 12 && Array.from(bytes.slice(0, 6)).join(',') === '147,78,85,77,80,89', 'Invalid 2D NPY')
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), version = bytes[6], prefix = version === 1 ? 10 : 12
  check((version === 1 || version === 2) && bytes[7] === 0, 'Unsupported NPY version')
  const length = version === 1 ? view.getUint16(8, true) : view.getUint32(8, true)
  check(length <= 4096 && prefix + length <= bytes.length, 'Invalid NPY header')
  const match = /^\{'descr': '(<[fi]8)', 'fortran_order': False, 'shape': \((\d+(?:,\s*\d+)*,?)\),?\s*\}\s*$/.exec(text(bytes.slice(prefix, prefix + length)))
  check(match, 'Only plain C-order 2D packs are supported')
  const dimensions = match[2].split(',').filter(v => v.trim()).map(Number), integer = match[1] === '<i8'
  check(dimensions.length === (integer ? 2 : 4) && dimensions.every(n => Number.isSafeInteger(n) && n >= 0), 'Unexpected 2D pack rank')
  const count = dimensions.reduce((a, b) => a * b, 1)
  check(Number.isSafeInteger(count) && prefix + length + count * 8 === bytes.length && same(shape, dimensions)
    && record.metadata.dtype === match[1] && record.metadata.order === 'C', '2D NPY metadata/size mismatch')
  const values = Array.from({ length: count }, (_, i) => integer ? Number(view.getBigInt64(prefix + length + i * 8, true)) : view.getFloat64(prefix + length + i * 8, true))
  check(values.every(v => Number.isFinite(v) && (!integer || Number.isSafeInteger(v))), 'Invalid accepted array values')
  return { shape: dimensions, values }
}

export class Steady2DClient {
  readonly endpoint: string
  constructor(endpoint: string | URL) {
    const url = new URL(endpoint)
    check(['http:', 'https:'].includes(url.protocol) && !url.username && !url.password && !url.search && !url.hash, 'Use an explicit HTTP endpoint')
    this.endpoint = url.href.replace(/\/$/, '') + '/'
  }
  async discover(signal?: AbortSignal): Promise<Steady2DCatalog> {
    const { value, response } = await requestJobService(this.endpoint, 'experiments', { signal })
    const registry = sha(response.headers.get('X-Solarlab-Registry')), schema = sha(response.headers.get('X-Solarlab-Job-Schema'))
    check(value.schema === 'solarlab.execution-catalog.v1' && value.job_schema_sha256 === schema
      && response.headers.get('etag') === `"${registry}"` && Array.isArray(value.bindings), 'Inconsistent installed catalog')
    const cap = object(value.resource_ceiling)
    check(cap.calls === null && cap.accepted_steps === null, 'Unsupported counter caps')
    const ceiling = { wall_seconds: number(cap.wall_seconds), rss_bytes: number(cap.rss_bytes), output_bytes: number(cap.output_bytes) }
    check(Object.values(ceiling).every(v => v > 0) && ceiling.wall_seconds <= 30, 'Invalid resource ceiling')
    const routes: Steady2DRoute[] = []
    for (const raw of value.bindings) {
      const entry = object(raw)
      if (entry.engine !== 'solarlab' || entry.driver !== STEADY_2D_EXPERIMENT || entry.experiment_id !== STEADY_2D_EXPERIMENT
        || entry.version !== 1 || entry.scope !== STEADY_2D_SCOPE) continue
      const schema = object(entry.input_schema), properties = object(schema.properties), installed = object(entry.preparation_context)
      const keys = ['device_id', 'voltage_V', 'generation_scales', 'spatial', 'controls', 'initial'].sort()
      check(schema.additionalProperties === false && same(Object.keys(properties).sort(), keys)
        && Array.isArray(schema.required) && same([...schema.required].sort(), keys), 'Unsupported 2D input schema')
      const policy = object(installed.fixed_numerical_policy)
      check(same(policy.spatial, geometry) && policy.voltage_V === .002 && same(policy.generation_scales, [[0], [1], [0, 1]])
        && same(policy.shape_Ny_Nx, [13, 5]) && policy.numerical_coordinates === 195 && policy.auxiliary_seed_is_2d_completion === false
        && policy.scientific_qualification === false && policy.scope === STEADY_2D_SCOPE && entry.scientific_qualification === false
        && same(object(properties.generation_scales).enum, policy.generation_scales)
        && same(object(properties.initial).const, policy.initial)
        && same(object(properties.controls).anyOf, [{ const: policy.controls }, { const: policy.insufficient_budget_controls }])
        && object(properties.voltage_V).minimum === .002 && object(properties.voltage_V).maximum === .002
        && Array.isArray(installed.devices), 'Unsupported fixed no-grain 2D protocol')
      const spatialSchema = object(object(schema.$defs)._SpatialInput), spatialProperties = object(spatialSchema.properties)
      check(spatialSchema.additionalProperties === false && same(Object.keys(spatialProperties).sort(), Object.keys(geometry).sort())
        && object(spatialProperties.Nx).minimum === 4 && object(spatialProperties.Nx).maximum === 4
        && object(spatialProperties.Ny_per_layer).minimum === 4 && object(spatialProperties.Ny_per_layer).maximum === 4
        && object(spatialProperties.lateral_length_m).minimum === 1e-6 && object(spatialProperties.lateral_length_m).maximum === 1e-6
        && object(spatialProperties.lateral_bc).const === 'neumann' && object(spatialProperties.geometry_policy).const === geometry.geometry_policy,
      'Installed geometry schema differs')
      const devices = installed.devices.map(raw => { const d = object(raw); return { id: id(d.device_id), preparedSha256: sha(d.prepared_sha256) } })
      check(new Set(devices.map(d => d.id)).size === devices.length && new Set(devices.map(d => d.preparedSha256)).size === devices.length, 'Ambiguous installed devices')
      routes.push({ id: STEADY_2D_EXPERIMENT, version: 1, devices, generations: policy.generation_scales as number[][],
        input: { voltage_V: .002, generation_scales: [0, 1], spatial: object(policy.spatial), controls: object(policy.controls), initial: object(policy.initial) } })
    }
    check(routes.length <= 1, 'Ambiguous 2D registrations')
    return { registry, schema, routes, ceiling }
  }
  async submit(catalog: Steady2DCatalog, route: Steady2DRoute, deviceId: string, input: Steady2DInput,
    runId = `steady-2d-${crypto.randomUUID()}`): Promise<Steady2DJob> {
    check(catalog.routes.includes(route), 'Select the installed 2D route')
    const device = route.devices.find(d => d.id === deviceId); check(device, 'Select an installed device')
    check(route.generations.some(g => same(g, input.generation_scales)) && same({ ...input, generation_scales: route.input.generation_scales }, route.input),
      'Use the installed dark/G1 targets, geometry, initial policy and full controls')
    const frozen = JSON.parse(JSON.stringify(input)) as Steady2DInput
    const { value, response } = await requestJobService(this.endpoint, 'jobs', { method: 'POST', headers: {
      'Content-Type': 'application/json', 'X-Solarlab-Registry': catalog.registry, 'X-Solarlab-Job-Schema': catalog.schema,
    }, body: JSON.stringify({ schema_version: 'solarlab.job-submission.v1', run_id: id(runId), experiment_id: route.id, version: 1,
      engine: 'solarlab', input: { device_id: deviceId, ...frozen }, limits: { ...catalog.ceiling, calls: null, accepted_steps: null } }) })
    check(response.status === 202 && value.status === 'accepted' && value.run_id === runId && value.execution_catalog_sha256 === catalog.registry
      && value.job_schema_sha256 === catalog.schema, '2D acceptance identity mismatch')
    return { ...this.open(id(value.store_id), runId, id(value.attempt_id)), submitted: { catalog, device, input: frozen } }
  }
  open(storeId: string, runId: string, attemptId: string): Steady2DJob {
    return { reader: new StoredRunClient(this.endpoint, id(storeId), 64 * 1024 ** 2), target: Object.freeze({ runId: id(runId), attemptId: id(attemptId) }) }
  }
  async cancel(job: Steady2DJob): Promise<void> {
    const run = (await job.reader.getRun(job.target.runId)).value
    check(run.current_attempt_id === job.target.attemptId && ['queued', 'running'].includes(run.state), 'Cancellation target is stale or terminal')
    const { value } = await requestJobService(this.endpoint, `jobs/${encodeURIComponent(job.target.runId)}/attempts/${encodeURIComponent(job.target.attemptId)}/cancel`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: `{"expected_generation":${run.generation.toString()},"reason":"User requested 2D steady cancellation"}`,
    })
    check(value.run_id === job.target.runId && value.attempt_id === job.target.attemptId, 'Cancellation identity mismatch')
  }
  async readResult(job: Steady2DJob, signal?: AbortSignal): Promise<Steady2DResult | undefined> {
    const [snapshot, run] = await Promise.all([job.reader.getAttempt(job.target, signal), job.reader.getRun(job.target.runId, signal)])
    const attempt = snapshot.value; check(attempt.terminal, 'Attempt is not committed')
    if (attempt.state === 'cancelled') { check(!attempt.terminal.artifacts.length, 'Cancelled attempt cannot invent fields'); return undefined }
    check(attempt.state === 'succeeded' || attempt.state === 'failed', 'Unsupported terminal state')
    if (!attempt.terminal.artifacts.length && attempt.state === 'failed') return undefined
    const worker = object(attempt.result?.worker_result), metadata = object(run.value.request.input_metadata), identity = object(run.value.request.identity)
    const documents = object(object(metadata.transport).documents)
    const [registry, submitted, schema] = await Promise.all([document(documents.registry), document(documents.request), document(documents.schema)])
    const summary = object(worker.steady_2d_result), capture = object(worker.capture), execution = object(metadata.execution_manifest)
    check(worker.schema === 'solarlab.steady-2d.worker-result.v1' && worker.engine === 'solarlab' && worker.scientific_qualification === false
      && summary.schema === schemaId && summary.dimension === 2 && summary.scientific_qualification === false
      && summary.physical_certificate_status === 'not_evaluated' && summary.mesh_qualification === 'not_evaluated' && summary.photovoltaic_metrics === null,
    'Not the bound full-2D worker')
    for (const key of ['H_input', 'H_physics', 'H_execution']) check(sha(worker[key]) === identity[key], '2D execution identity mismatch')
    check(worker.manifest_sha256 === execution.manifest_sha256 && worker.H_execution === execution.H_execution
      && capture.request_sha256 === submitted.sha256 && summary.prepared_sha256 === worker.prepared_sha256, 'Manifest/request capture mismatch')
    const expected = object(worker.payloads), keys = [...Object.keys(storedUnits), 'steady_2d_diagnostics']
    check(same(Object.keys(expected).sort(), keys.sort()) && attempt.terminal.artifacts.length === 14, 'Incomplete 2D artifact roles')
    const storedArrays: Record<string, FloatArray> = {}, names = new Set<string>(), hashes: Record<string, string> = {}
    let manifestJson = '', diagnosticsJson = ''
    for (const reference of attempt.terminal.artifacts) {
      const artifact = await job.reader.getArtifact(job.target, reference, signal), record = artifact.record.value, name = String(record.name)
      check(!names.has(name) && String(record.generation) === String(attempt.generation), 'Duplicate or foreign-generation artifact'); names.add(name)
      if (name === 'execution_manifest') { check(record.sha256 === worker.manifest_sha256, 'Wrong execution manifest'); manifestJson = text(artifact.bytes); continue }
      const binding = object(expected[name]); hashes[name] = sha(record.sha256)
      check(binding.sha256 === record.sha256 && binding.format === record.format && same(binding.metadata, record.metadata)
        && record.metadata.schema === schemaId && record.metadata.compiled_identity === summary.compiled_identity
        && record.metadata.geometry_identity === summary.geometry_identity && record.metadata.flatten_order === summary.flatten_order,
      'Payload binding mismatch')
      if (Object.hasOwn(storedUnits, name)) {
        check(record.metadata.unit === storedUnits[name] && record.metadata.dtype === (['face_pairs', 'accepted_indices'].includes(name) ? '<i8' : '<f8')
          && same(record.metadata.components, components[name] ?? []), 'Unit/dtype/component order mismatch')
        storedArrays[name] = readPack(artifact.bytes, record)
      } else { check(name === 'steady_2d_diagnostics' && record.metadata.role === 'original_auxiliary_and_2d_attempts', 'Wrong diagnostics role'); diagnosticsJson = text(artifact.bytes) }
    }
    check(names.size === 14 && Object.keys(storedArrays).length === 12 && manifestJson && diagnosticsJson, 'Incomplete 2D result')
    const manifest = object(JSON.parse(manifestJson)), bound = object(object(manifest.execution_identity).payload), roles = object(manifest.content_roles)
    const role = async (name: string) => {
      const doc = object(roles[name]); check(typeof doc.content_hex === 'string' && /^(?:[a-f0-9]{2})+$/.test(doc.content_hex), 'Invalid manifest role bytes')
      const bytes = Uint8Array.from(doc.content_hex.match(/../g)!, v => parseInt(v, 16))
      check(await digest(bytes) === sha(doc.sha256), 'Manifest role digest differs'); return object(JSON.parse(text(bytes)))
    }
    const [protocol, preparation, grid, controls, initial] = await Promise.all(['protocol', 'preparation', 'grid', 'numeric_controls', 'initial_state'].map(role))
    const sources: StoredObject = {}
    check(Array.isArray(manifest.implementation_content) && Array.isArray(preparation.adapter_sources), 'Missing source capture')
    for (const raw of [...manifest.implementation_content, ...preparation.adapter_sources]) { const s = object(raw); sources[String(s.id)] = sha(s.sha256) }
    check(bound.H_input === worker.H_input && bound.H_physics === worker.H_physics && object(bound.engine).name === 'solarlab'
      && same(capture.source_sha256, sources) && capture.build_source_commit === object(bound.implementation).build_source_commit
      && same(capture.runtime, bound.environment) && capture.manifest_sha256 === worker.manifest_sha256
      && capture.prepared_sha256 === worker.prepared_sha256 && capture.compiled_identity === summary.compiled_identity
      && capture.geometry_identity === summary.geometry_identity && preparation.prepared_sha256 === worker.prepared_sha256 && preparation.dimension === 2,
    'Actual source/runtime/configuration binding differs')
    const raw = object(JSON.parse(diagnosticsJson))
    check(submitted.data.schema === 'solarlab.steady-2d.worker-request.v1' && submitted.data.engine === 'solarlab'
      && submitted.data.prepared_sha256 === worker.prepared_sha256 && same(submitted.data, raw.request)
      && same(submitted.data.protocol, protocol) && same(submitted.data.spatial, grid) && same(submitted.data.controls, controls)
      && same(submitted.data.initial, initial) && same(preparation.resolved_device, raw.prepared_declaration), 'Recorded request differs from manifest/raw producer')
    const registered = registry.data.schema === 'solarlab.execution-catalog.v1'
    if (registered) {
      check(Array.isArray(registry.data.bindings), 'Missing recorded catalog')
      const entries = registry.data.bindings.map(object).filter(e => e.driver === STEADY_2D_EXPERIMENT); check(entries.length === 1, 'Ambiguous recorded 2D route')
      const entry = entries[0], installed = object(entry.preparation_context), policy = object(installed.fixed_numerical_policy)
      const selected = tuple(installed.devices).map(object).filter(d => d.prepared_sha256 === worker.prepared_sha256)
      check(registry.data.job_schema_sha256 === schema.sha256 && worker.execution_catalog_sha256 === registry.sha256
        && entry.experiment_id === STEADY_2D_EXPERIMENT && entry.version === 1 && entry.engine === 'solarlab' && entry.scope === STEADY_2D_SCOPE
        && entry.result_schema === worker.schema && entry.worker_source_sha256 === sources['solarlab/experiments/two_dimensional/steady_worker.py']
        && worker.experiment_id === STEADY_2D_EXPERIMENT && worker.experiment_version === 1 && selected.length === 1
        && selected[0].device_id === worker.registered_device_id && worker.registered_model_registry_sha256 === entry.model_registry_sha256
        && same(worker.server_sources, installed.server_sources) && same(worker.server_runtime, installed.server_runtime)
        && same(worker.registered_numerical_policy, policy) && same(worker.registered_spatial_declaration, geometry)
        && same(policy.spatial, geometry) && same(policy.initial, initial)
        && admitsSteady2DControls(policy, controls)
        && protocol.voltage_V === policy.voltage_V && tuple(policy.generation_scales).some(g => same(g, protocol.generation_scales))
        && worker.http_request_sha256 === submitted.sha256
        && object(installed.implementation).build_source_commit === capture.build_source_commit, 'Stored HTTP registration differs from its producer')
      const exactRegistry = recordedJson(text(Uint8Array.from(atob(String(object(documents.registry).base64)), c => c.charCodeAt(0))))
      const exactEntry = (exactRegistry.bindings as unknown[]).map(object).find(e => e.driver === STEADY_2D_EXPERIMENT)!
      check(worker.registered_input_schema_sha256 === await digest(new TextEncoder().encode(canonicalRecorded(exactEntry.input_schema))), 'Recorded input schema digest differs')
    } else {
      check(!worker.execution_catalog_sha256 && object(documents.registry).id === 'solarlab.steady-2d.worker-registry.v1'
        && object(documents.schema).id === 'solarlab.steady-2d.worker-request.v1', 'Unknown historical 2D registry')
    }
    const { arrays, attempts, targets } = validateArrays(raw, summary, storedArrays)
    check(attempt.state === (summary.completed ? 'succeeded' : 'failed') && worker.reason === summary.reason, 'Terminal contradicts original full-2D outcome')
    for (const attempt of attempts) {
      if (attempt.result_2d === null) continue
      for (const [name, hash] of Object.entries(object(object(attempt.result_2d).source_sha256))) {
        check(sources[name.replaceAll('.', '/') + '.py'] === hash, 'Original numerical source differs from manifest')
      }
    }
    const exactSummary = object(object(object(recordedJson(snapshot.json).result).worker_result)).steady_2d_result
    check(capture.result_sha256 === await digest(new TextEncoder().encode(canonicalRecorded({ summary: exactSummary, payloads: hashes }))), 'Result digest differs')
    if (job.submitted) check(worker.prepared_sha256 === job.submitted.device.preparedSha256 && registry.sha256 === job.submitted.catalog.registry
      && same(protocol, { voltage_V: job.submitted.input.voltage_V, generation_scales: job.submitted.input.generation_scales })
      && same(controls, job.submitted.input.controls) && same(initial, job.submitted.input.initial), 'Result differs from submitted protocol')
    return { state: attempt.state, registered, arrays, units: STEADY_2D_UNITS, storedArrays, attempts, targets, summary, worker,
      originalJson: snapshot.json, diagnosticsJson, manifestJson }
  }
}

function validateArrays(raw: StoredObject, summary: StoredObject, packed: Record<string, FloatArray>) {
  const g = object(raw.geometry), attempts = tuple(raw.attempts).map(object), targets = tuple(raw.requested_targets).map(object)
  const count = number(summary.accepted_count), ny = 13, nx = 5, n = ny * nx, f = 112
  check(raw.schema === schemaId && raw.qualification === false && raw.prepared_identity === summary.prepared_sha256
    && raw.compiled_identity === summary.compiled_identity && g.identity === summary.geometry_identity
    && same(tuple(g.shape_Ny_Nx), [ny, nx]) && same(summary.shape_Ny_Nx, [ny, nx]) && g.lateral_width_m === 1e-6
    && summary.lateral_width_m === g.lateral_width_m && g.x_face_count === 52
    && summary.flatten_order === 'C: y-major, x-fastest; coordinates [phi,z_n,z_p]'
    && summary.face_order === 'x faces then y faces; positive coordinate direction', '2D geometry/flattening identity differs')
  check(Number.isSafeInteger(count) && count >= 0 && count <= targets.length && attempts.length > 0
    && targets.length <= 2 && same(targets, tuple(object(object(raw.request).protocol).generation_scales).map(scale => ({ voltage_V: .002, generation_scale: scale })))
    && [[0], [1], [0, 1]].some(scales => same(targets.map(t => t.generation_scale), scales))
    && attempts.every((a, i) => same(a.target, targets[i]) && a.auxiliary_seed_is_2d_completion === false
      && typeof a.auxiliary_1d_seed_called === 'boolean' && typeof a.full_2d_adapter_called === 'boolean' && a.accepted === (i < count))
    && summary.requested_count === targets.length && summary.attempt_count === attempts.length
    && summary.unattempted_count === targets.length - attempts.length
    && (summary.completed === true ? count === targets.length && attempts.length === count : attempts.length === count + 1),
  'Requested/accepted/failed/unattempted targets differ')
  const failed = attempts[count]
  check(same(summary.first_failure, failed ? { index: count, target: failed.target, termination: failed.termination, error: failed.error, full_2d_adapter_called: failed.full_2d_adapter_called } : null)
    && summary.reason === (failed ? failed.termination : null), 'First failed target was lost')
  const arrays: Record<string, FloatArray> = {}
  for (const name of ['x_m', 'y_m', 'cell_areas_m2', 'face_lengths_m', 'face_spacing_m', 'face_pairs']) arrays[name] = rawArray(g[name], name === 'face_pairs' ? '<i8' : '<f8')
  arrays.accepted_indices = { shape: [count], values: Array.from({ length: count }, (_, i) => i) }
  arrays.accepted_voltage_V = { shape: [count], values: targets.slice(0, count).map(t => number(t.voltage_V)) }
  arrays.accepted_generation_scale = { shape: [count], values: targets.slice(0, count).map(t => number(t.generation_scale)) }
  const shapeByName: Record<string, number[]> = { state_coordinates: [3 * n], potential_V: [ny, nx], electron_density_m3: [ny, nx], hole_density_m3: [ny, nx],
    face_current_density_A_m2: [f], lateral_current_density_A_m2: [ny, nx - 1], stack_current_density_A_m2: [ny - 1, nx],
    terminal_current_per_depth_A_m: [2], cross_section_current_per_depth_A_m: [ny - 1], terminal_mean_current_density_A_m2: [2],
    original_residual: [3 * n], face_displacement_C_m2: [f], carrier_face_particle_flux_m2_s: [2, f] }
  for (const [name, shape] of Object.entries(shapeByName)) {
    const state = ['state_coordinates', 'potential_V', 'electron_density_m3', 'hole_density_m3'].includes(name)
    const values = attempts.slice(0, count).flatMap(a => {
      const result = object(a.result_2d), array = rawArray(object(result[state ? 'state' : 'diagnostics'])[name === 'state_coordinates' ? 'u' : name])
      check(same(array.shape, shape), 'Original 2D array shape differs'); return array.values
    })
    arrays[name] = { shape: [count, ...shape], values }
  }
  for (const [name, shape] of Object.entries({ x_m: [nx], y_m: [ny], cell_areas_m2: [ny, nx], face_lengths_m: [f], face_spacing_m: [f], face_pairs: [f, 2] }))
    check(same(arrays[name].shape, shape), '2D geometry array shape differs')
  const pairs: number[] = []
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx - 1; x++) pairs.push(y * nx + x, y * nx + x + 1)
  for (let y = 0; y < ny - 1; y++) for (let x = 0; x < nx; x++) pairs.push(y * nx + x, (y + 1) * nx + x)
  check(same(arrays.face_pairs.values, pairs) && ['x_m', 'y_m'].every(name => arrays[name].values.every((v, i, all) => !i || v > all[i - 1]))
    && ['cell_areas_m2', 'face_lengths_m', 'face_spacing_m'].every(name => arrays[name].values.every(v => v > 0)), '2D directed-face connectivity or measures differ')
  const expected = { ...arrays,
    face_geometry_m: { shape: [2, f], values: [...arrays.face_lengths_m.values, ...arrays.face_spacing_m.values] },
    carrier_density_m3: { shape: [2, count, ny, nx], values: [...arrays.electron_density_m3.values, ...arrays.hole_density_m3.values] },
    section_terminal_current_A_m: { shape: [count, ny + 1], values: Array.from({ length: count }, (_, i) => [arrays.terminal_current_per_depth_A_m.values[i * 2],
      ...arrays.cross_section_current_per_depth_A_m.values.slice(i * (ny - 1), (i + 1) * (ny - 1)), arrays.terminal_current_per_depth_A_m.values[i * 2 + 1]]).flat() } }
  for (const name of Object.keys(packed)) check(same(packed[name], expected[name as keyof typeof expected]), 'NPY bytes differ from original 2D producer arrays')
  for (const a of attempts.slice(0, count)) {
    const r = object(a.result_2d), state = object(r.state), diagnostics = object(r.diagnostics), controls = object(r.solver_controls)
    check(r.accepted === true && a.full_2d_adapter_called === true && r.termination === 'converged' && object(r.newton).accepted === true
      && r.compiled_identity === summary.compiled_identity && state.device_sha256 === summary.compiled_identity
      && r.operating_voltage_V === object(a.target).voltage_V && r.generation_scale === object(a.target).generation_scale
      && r.physical_certificate_status === 'not_evaluated' && controls.full_2d_newton_calls === 1 && controls.hidden_seed_or_fallback_calls === 0
      && same(object(controls.numerical).electronic, object(object(object(raw.request).controls).full_2d).electronic)
      && same(rawArray(state.u), rawArray(r.candidate_coordinates)) && same(rawArray(object(r.newton).state), rawArray(r.candidate_coordinates))
      && same(rawArray(state.potential_V).values, rawArray(state.u).values.slice(0, n))
      && same(tuple(diagnostics.original_rows), Array.from({ length: 3 * n }, (_, i) => i))
      && tuple(r.checks).length > 0 && tuple(r.checks).map(object).every(c => c.passed === true && number(c.value) <= number(c.limit)),
    'Accepted fields lack original full-2D Newton evidence')
  }
  check(arrays.terminal_current_per_depth_A_m.values.every((v, i) => Object.is(v / 1e-6, arrays.terminal_mean_current_density_A_m2.values[i])), 'Terminal width normalization differs')
  const counts = object(summary.computation_counts), full = attempts.filter(a => a.result_2d !== null).map(a => object(a.result_2d))
  const seeds = attempts.flatMap(a => a.auxiliary_1d_seed !== null ? [...tuple(object(a.auxiliary_1d_seed).preparation), object(a.auxiliary_1d_seed).newton].map(object)
    : a.failed_auxiliary_seed_newton !== null ? [object(a.failed_auxiliary_seed_newton)] : [])
  check(counts.auxiliary_1d_seed_calls === attempts.filter(a => a.auxiliary_1d_seed_called === true).length
    && counts.auxiliary_1d_observed_newton_results === seeds.length
    && counts.auxiliary_1d_observed_residual_evaluations === seeds.reduce((s, r) => s + number(r.residual_evaluations), 0)
    && counts.auxiliary_1d_observed_jacobian_evaluations === seeds.reduce((s, r) => s + number(r.jacobian_evaluations), 0)
    && counts.full_2d_adapter_calls === attempts.filter(a => a.full_2d_adapter_called === true).length
    && counts.full_2d_newton_calls === full.reduce((s, r) => s + number(object(r.solver_controls).full_2d_newton_calls), 0)
    && ['residual', 'jacobian', 'diagnostics'].every(name => object(counts.full_2d_evaluation_counts)[name] === full.reduce((s, r) => s + number(object(r.evaluation_counts)[name]), 0)),
  'Auxiliary and full-2D computation counts differ')
  return { arrays, attempts, targets }
}
