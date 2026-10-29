/** Installed illuminated ionic steady J-V; reopening is strictly a stored read. */
import { RunClientError, StoredRunClient } from './run-client'
import type { AttemptIdentity, StoredObject } from './run-client'
import { requestJobService } from './steady-jv-client'
import type { FloatArray } from './steady-jv-client'
import { readIonicArray } from './ionic-transient-client'

export const IONIC_JV_EXPERIMENT = 'ionic_steady_jv'
export const IONIC_JV_SCOPE = 'illuminated_shared_site_positive_D_single_blocking_domain_ionic_steady_jv'
export const IONIC_JV_UNITS: Record<string, string> = {
  requested_voltage_V: 'V', voltage_V: 'V', nodes_m: 'm', ion_node_indices: '1', ion_cell_measures_m: 'm',
  state_coordinates: 'mixed: first N in V; remaining coordinates dimensionless', potential_V: 'V',
  carrier_density_m3: 'm^-3', ion_density_m3: 'm^-3', ion_inventory_m2: 'm^-2',
  current_density_A_m2: 'A/m^2', output_current_A_m2: 'A/m^2',
}
const currentNames = ['electronic_current_density_A_m2', 'ionic_current_density_A_m2', 'total_current_density_A_m2']
export interface IonicJVInput {
  voltage_V: number[]; generation_scale: number; incident_power_W_m2: null; intervals_per_layer: number[]
  controls: StoredObject; initial: { policy: string; voltage_V: number; wall_s: number }
}
export interface IonicJVRoute {
  id: string; version: number; domain: string; input: IonicJVInput
  devices: readonly { id: string; preparedSha256: string }[]
}
export interface IonicJVCatalog {
  registry: string; schema: string; routes: readonly IonicJVRoute[]
  ceiling: { wall_seconds: number; rss_bytes: number; output_bytes: number }
}
export interface IonicJVJob {
  reader: StoredRunClient; target: Readonly<AttemptIdentity>
  submitted?: { catalog: IonicJVCatalog; device: IonicJVRoute['devices'][number]; input: IonicJVInput }
}
export interface IonicJVResult {
  state: 'succeeded' | 'failed'; registered: boolean; arrays: Record<string, FloatArray>; units: Record<string, string>
  storedArrays: Record<string, FloatArray>; species: string[]; attempts: StoredObject[]; summary: StoredObject; worker: StoredObject
  originalJson: string; diagnosticsJson: string; initialJson: string; manifestJson: string
}

function check(value: unknown, message: string): asserts value { if (!value) throw new RunClientError('protocol', message) }
function object(value: unknown): StoredObject {
  check(value !== null && typeof value === 'object' && !Array.isArray(value), 'Expected an ionic J-V record'); return value as StoredObject
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
/** Preserve Python's original number spellings (including 1.0) for the result digest.
 * The shared reader already requires source-aware JSON parsing for stored integers. */
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

export class IonicJVClient {
  readonly endpoint: string
  constructor(endpoint: string | URL) {
    const url = new URL(endpoint)
    check(['http:', 'https:'].includes(url.protocol) && !url.username && !url.password && !url.search && !url.hash, 'Use an explicit HTTP endpoint')
    this.endpoint = url.href.replace(/\/$/, '') + '/'
  }
  async discover(signal?: AbortSignal): Promise<IonicJVCatalog> {
    const { value, response } = await requestJobService(this.endpoint, 'experiments', { signal })
    const registry = sha(response.headers.get('X-Solarlab-Registry')), schema = sha(response.headers.get('X-Solarlab-Job-Schema'))
    check(value.schema === 'solarlab.execution-catalog.v1' && value.job_schema_sha256 === schema
      && response.headers.get('etag') === `"${registry}"` && Array.isArray(value.bindings), 'Inconsistent installed catalog')
    const cap = object(value.resource_ceiling)
    check(cap.calls === null && cap.accepted_steps === null, 'Unsupported counter caps')
    const ceiling = { wall_seconds: number(cap.wall_seconds), rss_bytes: number(cap.rss_bytes), output_bytes: number(cap.output_bytes) }
    check(Object.values(ceiling).every(v => v > 0) && ceiling.wall_seconds <= 30, 'Invalid resource ceiling')
    const routes: IonicJVRoute[] = []
    for (const raw of value.bindings) {
      const entry = object(raw)
      if (entry.engine !== 'solarlab' || entry.driver !== IONIC_JV_EXPERIMENT || entry.experiment_id !== IONIC_JV_EXPERIMENT
        || entry.version !== 1 || entry.scope !== IONIC_JV_SCOPE) continue
      const schema = object(entry.input_schema), properties = object(schema.properties), installed = object(entry.preparation_context)
      const keys = ['device_id', 'voltage_V', 'generation_scale', 'incident_power_W_m2', 'intervals_per_layer', 'controls', 'initial'].sort()
      check(schema.additionalProperties === false && same(Object.keys(properties).sort(), keys)
        && Array.isArray(schema.required) && same([...schema.required].sort(), keys), 'Unsupported ionic J-V input schema')
      const policy = object(installed.fixed_numerical_policy), protocol = object(policy.protocol), initial = object(policy.initial)
      check(same(protocol.voltage_V, Array.from({ length: 51 }, (_, i) => i / 40)) && protocol.generation_scale === 1
        && protocol.incident_power_W_m2 === null && same(policy.intervals_per_layer, [12, 12, 12])
        && same(initial, { policy: 'dark_direct_density_seed_v1', voltage_V: .002, wall_s: 25 })
        && same(object(properties.voltage_V).const, protocol.voltage_V) && same(object(properties.controls).const, policy.controls)
        && same(installed.ion_domain_endpoints, ['blocking', 'blocking']) && Array.isArray(installed.devices), 'Unsupported fixed illuminated protocol')
      const devices = installed.devices.map(raw => { const d = object(raw); return { id: id(d.device_id), preparedSha256: sha(d.prepared_sha256) } })
      check(new Set(devices.map(d => d.id)).size === devices.length && new Set(devices.map(d => d.preparedSha256)).size === devices.length, 'Ambiguous installed devices')
      routes.push({ id: IONIC_JV_EXPERIMENT, version: 1, domain: id(installed.ion_domain_layer), devices,
        input: { ...protocol, intervals_per_layer: policy.intervals_per_layer, controls: policy.controls, initial } as unknown as IonicJVInput })
    }
    check(routes.length <= 1, 'Ambiguous ionic J-V registrations')
    return { registry, schema, routes, ceiling }
  }
  async submit(catalog: IonicJVCatalog, route: IonicJVRoute, deviceId: string, input: IonicJVInput,
    runId = `ionic-jv-${crypto.randomUUID()}`): Promise<IonicJVJob> {
    check(catalog.routes.includes(route), 'Select the installed ionic J-V route')
    const device = route.devices.find(d => d.id === deviceId); check(device, 'Select an installed device')
    check(same(input, route.input), 'Use the exact installed protocol, initial policy and full controls')
    const frozen = JSON.parse(JSON.stringify(input)) as IonicJVInput
    const { value, response } = await requestJobService(this.endpoint, 'jobs', { method: 'POST', headers: {
      'Content-Type': 'application/json', 'X-Solarlab-Registry': catalog.registry, 'X-Solarlab-Job-Schema': catalog.schema,
    }, body: JSON.stringify({ schema_version: 'solarlab.job-submission.v1', run_id: id(runId), experiment_id: route.id, version: 1,
      engine: 'solarlab', input: { device_id: deviceId, ...frozen }, limits: { ...catalog.ceiling, calls: null, accepted_steps: null } }) })
    check(response.status === 202 && value.status === 'accepted' && value.run_id === runId && value.execution_catalog_sha256 === catalog.registry
      && value.job_schema_sha256 === catalog.schema, 'Ionic J-V acceptance identity mismatch')
    return { ...this.open(id(value.store_id), runId, id(value.attempt_id)), submitted: { catalog, device, input: frozen } }
  }
  open(storeId: string, runId: string, attemptId: string): IonicJVJob {
    return { reader: new StoredRunClient(this.endpoint, id(storeId), 64 * 1024 ** 2), target: Object.freeze({ runId: id(runId), attemptId: id(attemptId) }) }
  }
  async cancel(job: IonicJVJob): Promise<void> {
    const run = (await job.reader.getRun(job.target.runId)).value
    check(run.current_attempt_id === job.target.attemptId && ['queued', 'running'].includes(run.state), 'Cancellation target is stale or terminal')
    const { value } = await requestJobService(this.endpoint, `jobs/${encodeURIComponent(job.target.runId)}/attempts/${encodeURIComponent(job.target.attemptId)}/cancel`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: `{"expected_generation":${run.generation.toString()},"reason":"User requested ionic J-V cancellation"}`,
    })
    check(value.run_id === job.target.runId && value.attempt_id === job.target.attemptId, 'Cancellation identity mismatch')
  }
  async readResult(job: IonicJVJob, signal?: AbortSignal): Promise<IonicJVResult | undefined> {
    const [snapshot, run] = await Promise.all([job.reader.getAttempt(job.target, signal), job.reader.getRun(job.target.runId, signal)])
    const attempt = snapshot.value; check(attempt.terminal, 'Attempt is not committed')
    if (attempt.state === 'cancelled') { check(!attempt.terminal.artifacts.length, 'Cancelled attempt cannot invent a curve'); return undefined }
    check(attempt.state === 'succeeded' || attempt.state === 'failed', 'Unsupported terminal state')
    if (!attempt.terminal.artifacts.length && attempt.state === 'failed') return undefined
    const worker = object(attempt.result?.worker_result), metadata = object(run.value.request.input_metadata), identity = object(run.value.request.identity)
    const documents = object(object(metadata.transport).documents)
    const [registry, submitted, schema] = await Promise.all([document(documents.registry), document(documents.request), document(documents.schema)])
    const capture = object(worker.capture), summary = object(worker.ionic_jv_result), execution = object(metadata.execution_manifest)
    check(worker.schema === 'solarlab.ionic-jv.worker-result.v1' && worker.engine === 'solarlab' && worker.scientific_qualification === false
      && summary.schema === 'solarlab.ionic-jv.result.v1' && summary.physical_certificate_status === 'not_evaluated', 'Not the bound ionic J-V worker')
    for (const key of ['H_input', 'H_physics', 'H_execution']) check(sha(worker[key]) === identity[key], 'Ionic J-V execution identity mismatch')
    check(worker.manifest_sha256 === execution.manifest_sha256 && worker.H_execution === execution.H_execution
      && capture.request_sha256 === submitted.sha256 && summary.prepared_sha256 === worker.prepared_sha256, 'Manifest/request capture mismatch')
    const registered = registry.data.schema === 'solarlab.execution-catalog.v1'
    const species = tuple(summary.ion_species).map(String)
    check(same(species, ['positive', 'negative']) && summary.ion_site_contract === 'shared_finite_sites', 'Unsupported ionic J-V species/site contract')
    const components: Record<string, string[]> = { carrier_density_m3: ['electron_density_m3', 'hole_density_m3'],
      ion_density_m3: species.map(s => s + '_ion_density_m3'), ion_inventory_m2: species.map(s => s + '_ion_inventory_m2'), current_density_A_m2: currentNames }
    const expected = object(worker.payloads), keys = [...Object.keys(IONIC_JV_UNITS), 'ionic_jv_initial', 'ionic_jv_diagnostics']
    check(same(Object.keys(expected).sort(), keys.sort()) && attempt.terminal.artifacts.length === 15, 'Incomplete ionic J-V artifact roles')
    const storedArrays: Record<string, FloatArray> = {}, names = new Set<string>(), hashes: Record<string, string> = {}
    let manifestJson = '', diagnosticsJson = '', initialJson = ''
    for (const reference of attempt.terminal.artifacts) {
      const artifact = await job.reader.getArtifact(job.target, reference, signal), record = artifact.record.value, name = String(record.name)
      check(!names.has(name) && String(record.generation) === String(attempt.generation), 'Duplicate or foreign-generation artifact'); names.add(name)
      if (name === 'execution_manifest') { check(record.sha256 === worker.manifest_sha256, 'Wrong execution manifest'); manifestJson = text(artifact.bytes); continue }
      const binding = object(expected[name]); hashes[name] = String(record.sha256)
      check(binding.sha256 === record.sha256 && binding.format === record.format && same(binding.metadata, record.metadata)
        && record.metadata.schema === summary.schema && record.metadata.compiled_identity === summary.compiled_identity, 'Payload binding mismatch')
      if (Object.hasOwn(IONIC_JV_UNITS, name)) {
        check(record.metadata.unit === IONIC_JV_UNITS[name] && record.metadata.dtype === (name === 'ion_node_indices' ? '<i8' : '<f8')
          && same(record.metadata.components, components[name] ?? []), 'Unit/dtype/component order mismatch')
        storedArrays[name] = readIonicArray(artifact.bytes, record)
      } else if (name === 'ionic_jv_initial') { check(record.metadata.role === 'exact_initial_evidence', 'Wrong initial role'); initialJson = text(artifact.bytes) }
      else { check(name === 'ionic_jv_diagnostics' && record.metadata.role === 'original_ionic_jv_attempts', 'Wrong diagnostics role'); diagnosticsJson = text(artifact.bytes) }
    }
    check(names.size === 15 && Object.keys(storedArrays).length === 12 && manifestJson && diagnosticsJson && initialJson, 'Incomplete ionic J-V result')
    const manifest = object(JSON.parse(manifestJson)), bound = object(object(manifest.execution_identity).payload), roles = object(manifest.content_roles)
    const role = async (name: string) => {
      const doc = object(roles[name]); check(typeof doc.content_hex === 'string' && /^(?:[a-f0-9]{2})+$/.test(doc.content_hex), 'Invalid manifest role bytes')
      const bytes = Uint8Array.from(doc.content_hex.match(/../g)!, v => parseInt(v, 16))
      check(await digest(bytes) === sha(doc.sha256), 'Manifest role digest differs'); return object(JSON.parse(text(bytes)))
    }
    const [protocol, preparation, grid, controls, recipe] = await Promise.all(['protocol', 'preparation', 'grid', 'numeric_controls', 'initial_state'].map(role))
    const sources: StoredObject = {}
    check(Array.isArray(manifest.implementation_content) && Array.isArray(preparation.adapter_sources), 'Missing source capture')
    for (const value of [...manifest.implementation_content, ...preparation.adapter_sources]) { const s = object(value); sources[String(s.id)] = sha(s.sha256) }
    check(bound.H_input === worker.H_input && bound.H_physics === worker.H_physics && object(bound.engine).name === 'solarlab'
      && capture.scope === 'ionic_compile_initialization_and_steady_jv_in_worker' && same(capture.source_sha256, sources)
      && capture.build_source_commit === object(bound.implementation).build_source_commit && same(capture.runtime, bound.environment)
      && capture.manifest_sha256 === worker.manifest_sha256 && capture.compiled_identity === summary.compiled_identity
      && capture.prepared_sha256 === worker.prepared_sha256 && preparation.prepared_sha256 === worker.prepared_sha256, 'Actual source/runtime/configuration binding differs')
    check(submitted.data.schema === 'solarlab.ionic-jv.worker-request.v1' && submitted.data.engine === 'solarlab'
      && submitted.data.prepared_sha256 === worker.prepared_sha256 && same(submitted.data.protocol, protocol)
      && same(submitted.data.grid, grid) && same(submitted.data.controls, controls) && same(submitted.data.initial, recipe), 'Recorded request differs from manifest')
    if (registered) {
      const catalog = registry.data; check(Array.isArray(catalog.bindings), 'Missing recorded catalog')
      const entries = catalog.bindings.map(object).filter(e => e.driver === IONIC_JV_EXPERIMENT); check(entries.length === 1, 'Ambiguous recorded ionic J-V route')
      const entry = entries[0], installed = object(entry.preparation_context)
      check(catalog.job_schema_sha256 === schema.sha256 && worker.execution_catalog_sha256 === registry.sha256
        && entry.experiment_id === IONIC_JV_EXPERIMENT && entry.version === 1 && entry.engine === 'solarlab' && entry.scope === IONIC_JV_SCOPE
        && entry.result_schema === worker.schema && entry.worker_source_sha256 === sources['solarlab/experiments/jv/ionic_worker.py']
        && worker.experiment_id === IONIC_JV_EXPERIMENT && worker.experiment_version === 1 && worker.http_request_sha256 === submitted.sha256
        && entry.model_registry_sha256 === worker.registered_model_registry_sha256 && entry.model_registry_sha256 === object(bound.effective_binding).registry_sha256
        && same(installed.implementation, { build_source_commit: object(bound.implementation).build_source_commit,
          source_manifest_sha256: await digest(new TextEncoder().encode(canonicalRecorded(manifest.implementation_content.map(raw => {
            const s = object(raw); return { id: s.id, sha256: s.sha256 }
          })))) }) && installed.current_repository_head === execution.current_repository_head
        && same(installed.server_sources, worker.server_sources) && same(installed.server_runtime, worker.server_runtime)
        && same(installed.fixed_numerical_policy, worker.registered_numerical_policy) && Array.isArray(installed.devices), 'Registered source/schema/model binding differs')
      const devices = installed.devices.map(object).filter(d => d.prepared_sha256 === worker.prepared_sha256)
      const policy = object(installed.fixed_numerical_policy)
      check(same(policy.protocol, protocol) && same(policy.controls, controls) && same(policy.intervals_per_layer, grid.intervals_per_layer)
        && same(policy.initial, { policy: recipe.policy, voltage_V: recipe.voltage_V, wall_s: recipe.wall_s }), 'Recorded installed protocol differs')
      const definitions = object(catalog.registry).experiments; check(Array.isArray(definitions), 'Missing experiment definitions')
      const definition = definitions.map(object).find(d => d.id === IONIC_JV_EXPERIMENT && d.version === 1)
      check(devices.length === 1 && devices[0].device_id === worker.registered_device_id && definition
        && object(definition.parameter_schema).sha256 === worker.registered_input_schema_sha256, 'Registered device/input schema differs')
    } else {
      check(registry.data.worker === IONIC_JV_EXPERIMENT && registry.data.engine === 'solarlab' && registry.data.fallback === 'forbidden'
        && registry.data.scope === 'illuminated_positive_D_one_blocking_domain_ionic_steady_jv' && schema.data.engine === 'solarlab'
        && worker.execution_catalog_sha256 === undefined && worker.registered_device_id === undefined, 'Historical process identity was relabelled')
    }
    const arrays: Record<string, FloatArray> = {}, units: Record<string, string> = {}
    for (const [name, pack] of Object.entries(storedArrays)) {
      if (!components[name]) { arrays[name] = pack; units[name] = IONIC_JV_UNITS[name]; continue }
      const labels = components[name]; check(pack.shape[0] === labels.length, 'Wrong packed component count')
      const shape = pack.shape.slice(1), size = shape.reduce((a, b) => a * b, 1)
      labels.forEach((key, i) => { arrays[key] = { shape, values: pack.values.slice(i * size, (i + 1) * size) }; units[key] = IONIC_JV_UNITS[name] })
    }
    const x = arrays.nodes_m.values, indices = arrays.ion_node_indices.values, v = arrays.voltage_V.values, requested = arrays.requested_voltage_V.values
    const n = x.length, m = indices.length, count = v.length
    check(n === 37 && m === 13 && x.every((v, i) => !i || v > x[i - 1])
      && indices.every((v, i) => Number.isSafeInteger(v) && v >= 0 && v < n && (!i || v === indices[i - 1] + 1))
      && arrays.ion_cell_measures_m.values.every(v => v > 0) && count <= requested.length
      && requested.every((v, i) => !i || v > requested[i - 1]) && v.every((v, i) => Object.is(v, requested[i])), 'Invalid accepted prefix or ionic grid')
    arrays.ion_nodes_m = { shape: [m], values: indices.map(i => x[i]) }; units.ion_nodes_m = 'm'
    for (const s of species) {
      const density = arrays[s + '_ion_density_m3']
      arrays[s + '_ion_cell_inventory_m2'] = { shape: density.shape, values: density.values.map((v, i) => v * arrays.ion_cell_measures_m.values[i % m]) }
      units[s + '_ion_cell_inventory_m2'] = 'm^-2'
    }
    for (const name of currentNames) {
      arrays['terminal_' + name] = { shape: [count, 2], values: v.flatMap((_, i) => [arrays[name].values[i * (n + 1)], arrays[name].values[i * (n + 1) + n]]) }
      units['terminal_' + name] = 'A/m^2'
    }
    for (const [name, a] of Object.entries(arrays)) {
      const shape = name === 'nodes_m' ? [n] : ['ion_nodes_m', 'ion_node_indices', 'ion_cell_measures_m'].includes(name) ? [m]
        : name === 'requested_voltage_V' ? [requested.length] : ['voltage_V', 'output_current_A_m2'].includes(name) || name.endsWith('_ion_inventory_m2') ? [count]
          : [count, name === 'state_coordinates' ? 3 * n + species.length * m : name.includes('current_density') ? name.startsWith('terminal_') ? 2 : n + 1 : name.includes('_ion_') ? m : n]
      check(same(a.shape, shape), 'Ionic J-V logical array shape differs')
    }
    const raw = object(JSON.parse(diagnosticsJson)), curve = raw.curve === null ? undefined : object(raw.curve)
    const attempts = curve ? tuple(curve.attempts).map(object) : [], complete = attempt.state === 'succeeded', failed = attempts.find(a => a.accepted === false)
    check(same(raw.protocol, protocol) && same(raw.controls, controls) && same(raw.initial_recipe, recipe)
      && same(protocol.voltage_V, requested) && protocol.generation_scale === summary.generation_scale && protocol.incident_power_W_m2 === summary.incident_power_W_m2
      && summary.requested_count === requested.length && summary.accepted_count === count && summary.attempt_count === attempts.length
      && summary.outcome === (complete ? 'completed' : curve ? 'partial' : 'initial_failed') && worker.reason === summary.reason
      && (complete ? count === requested.length && !failed : count < requested.length), 'Terminal state or requested/accepted samples differ')
    check(attempts.every((a, i) => a.index === i && a.voltage_V === requested[i] && a.accepted === (i < count))
      && attempts.length === count + (failed ? 1 : 0) && (!curve || complete || failed), 'First failed attempt or unattempted tail was lost')
    check(same(summary.first_failure, failed ? Object.fromEntries(['index', 'voltage_V', 'termination', 'solver_called', 'error'].map(k => [k, failed[k]])) : null), 'First failure differs')
    const compiled = object(raw.compiled_binding), domain = object(compiled.ion_domain), layout = object(compiled.state_layout), model = object(compiled.ion_model)
    check(summary.compiled_identity === await digest(new TextEncoder().encode(canonicalRecorded(recordedJson(diagnosticsJson).compiled_binding))), 'Compiled identity digest differs')
    check(compiled.original_prepared_sha256 === worker.prepared_sha256 && same(domain.contacts, ['blocking', 'blocking'])
      && domain.layer === grid.ion_domain_layer && same(domain.electrical_node_indices, indices) && same(domain.cell_measures_m, arrays.ion_cell_measures_m.values)
      && layout.electrical_node_count === n && layout.ion_node_count === m && layout.size === 3 * n + species.length * m
      && same(layout.ion_species, species) && model.site_contract === summary.ion_site_contract, 'Compiled physics/grid binding differs')
    for (const name of ['face_diffusivity_m2_s', 'negative_face_diffusivity_m2_s']) check(Array.isArray(model[name])
      && model[name].length === m - 1 && model[name].every(v => typeof v === 'number' && v > 0 && Number.isFinite(v)), 'Positive diffusivity required')
    for (const name of ['nodes_m', 'ion_node_indices', 'ion_cell_measures_m']) check(same(rawArray(object(raw.grid)[name], name === 'ion_node_indices' ? '<i8' : '<f8'), arrays[name]), 'Raw grid differs')
    const initial = object(JSON.parse(initialJson))
    check(capture.initial_artifact_sha256 === hashes.ionic_jv_initial && object(initial.summary).qualification === false
      && object(initial.summary).compiled_identity === summary.compiled_identity && capture.curve_started === Boolean(curve), 'Initial evidence differs')
    if (curve) {
      check(curve.compiled_identity === summary.compiled_identity && same(curve.binding_manifest, compiled)
        && curve.original_prepared_identity === worker.prepared_sha256 && same(curve.metrics, summary.metrics)
        && same(rawArray(curve.voltage_V), arrays.voltage_V) && same(rawArray(curve.output_current_A_m2), arrays.output_current_A_m2), 'Original curve differs')
      const provenance = object(curve.initial_provenance), initialState = rawArray(object(provenance.state).u)
      const words = new Uint8Array(initialState.values.length * 8), view = new DataView(words.buffer)
      initialState.values.forEach((v, i) => view.setFloat64(i * 8, v, true))
      const initialHash = await digest(words)
      check(provenance.policy === 'saved_accepted' && provenance.source_sha256 === hashes.ionic_jv_initial
        && provenance.state_sha256 === initialHash && provenance.generation_scale === protocol.generation_scale
        && same(capture.initial_state, { dtype: '<f8', shape: initialState.shape, sha256: initialHash })
        && same(object(object(initial.result).state).u, initialState.values) && object(initial.summary).accepted === true
        && same(object(initial.result).solver_controls, controls.steady) && same(curve.solver_controls, controls), 'Exact initial state/control admission differs')
      for (let i = 0; i < count; i++) {
        const solution = object(attempts[i].solution), state = object(solution.state), electronic = object(state.electronic_state), diagnostics = object(solution.diagnostics)
        check(solution.accepted === true && state.device_sha256 === summary.compiled_identity && electronic.voltage_V === v[i]
          && same(rawArray(state.u).values, arrays.state_coordinates.values.slice(i * (3 * n + species.length * m), (i + 1) * (3 * n + species.length * m))), 'Accepted state differs')
        for (const name of ['potential_V', 'electron_density_m3', 'hole_density_m3']) check(same(rawArray(electronic[name]).values, arrays[name].values.slice(i * n, (i + 1) * n)), 'Electronic fields differ')
        for (const s of species) check(same(rawArray(state[s + '_ion_density_m3']).values, arrays[s + '_ion_density_m3'].values.slice(i * m, (i + 1) * m))
          && Object.is(state[s === 'positive' ? 'domain_inventory_m2' : 'negative_domain_inventory_m2'], arrays[s + '_ion_inventory_m2'].values[i]), 'Species density/inventory differs')
        for (const name of currentNames) check(same(rawArray(diagnostics[name]).values, arrays[name].values.slice(i * (n + 1), (i + 1) * (n + 1))), 'Steady current accounting differs')
      }
      const counts = object(summary.computation_counts), cold = object(counts.cold_steady), illumination = object(counts.illumination), sweep = object(counts.curve)
      check(sweep.solver_calls === curve.solver_calls && same(sweep.evaluation_counts, curve.evaluation_counts)
        && curve.solver_calls === attempts.filter(a => a.solver_called === true).length
        && counts.total_solver_calls === number(cold.solver_calls) + number(illumination.component_solver_calls) + number(illumination.coupled_solver_calls) + number(sweep.solver_calls), 'Computation counts differ')
      check([1, -1].includes(Number(curve.voltage_polarity)) && arrays.total_current_density_A_m2.values.every((v, i) =>
        Object.is(v, arrays.electronic_current_density_A_m2.values[i] + arrays.ionic_current_density_A_m2.values[i]))
        && arrays.output_current_A_m2.values.every((v, i) => Object.is(v, -Number(curve.voltage_polarity) * arrays.total_current_density_A_m2.values[i * (n + 1)])), 'Current sum or output polarity differs')
    }
    check(summary.incident_power_W_m2 !== null || object(summary.metrics).efficiency === null, 'Efficiency needs known incident power')
    const exact = recordedJson(snapshot.json)
    const exactSummary = object(object(object(exact).result).worker_result).ionic_jv_result
    check(capture.result_sha256 === await digest(new TextEncoder().encode(canonicalRecorded({ summary: exactSummary, payloads: hashes }))), 'Result digest differs')
    if (job.submitted) check(worker.prepared_sha256 === job.submitted.device.preparedSha256 && registry.sha256 === job.submitted.catalog.registry
      && same(protocol, { voltage_V: job.submitted.input.voltage_V, generation_scale: job.submitted.input.generation_scale, incident_power_W_m2: job.submitted.input.incident_power_W_m2 }), 'Result differs from submitted protocol')
    return { state: attempt.state, registered, arrays, units, storedArrays, species, attempts, summary, worker, originalJson: snapshot.json, diagnosticsJson, initialJson, manifestJson }
  }
}
