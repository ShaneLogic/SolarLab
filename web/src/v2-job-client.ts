/**
 * V2 solarlab_server job submission, durable SSE subscription, and result readback.
 *
 * Uses exclusively V2 routes (`/experiments`, `/jobs`, `/runs/{run}/attempts/{attempt}/events`,
 * `/results/{run}/attempts/{attempt}`, `/artifacts/{id}/content`) with engine='solarlab'
 * and zero fallback to retired legacy endpoints.
 */
import { StoredRunClient, RunClientError } from './run-client'
import type { AttemptIdentity, StoredObject } from './run-client'
import { readSteadyArray, requestJobService, voltageGrid, STEADY_SCOPE } from './steady-jv-client'
import type { FloatArray } from './steady-jv-client'
import { resolveV2ServiceEndpoint } from './v2-workstation-client'
import type {
  DarkJVResult,
  DispatchableJobKind,
  JobStreamHandlers,
  JV2DResult,
  JVResult,
  SpatialSnapshot2D,
  TPVResult,
} from './types'

interface EncodedV2JobHandle {
  v: 2
  endpoint: string
  storeId: string
  runId: string
  attemptId: string
  kind: DispatchableJobKind
  driver: string
  catalogSha256: string
  voltageV?: number
}

function encodeHandle(handle: EncodedV2JobHandle): string {
  return 'v2:' + btoa(JSON.stringify(handle))
}

function decodeHandle(jobId: string): EncodedV2JobHandle {
  if (!jobId.startsWith('v2:')) {
    throw new RunClientError('protocol', 'Invalid V2 job handle; legacy job identifiers are not accepted for live dispatch')
  }
  const parsed = JSON.parse(atob(jobId.slice(3))) as EncodedV2JobHandle
  if (!parsed || parsed.v !== 2 || !parsed.storeId || !parsed.runId || !parsed.attemptId) {
    throw new RunClientError('protocol', 'Malformed V2 job handle')
  }
  return parsed
}

function asObject(value: unknown, msg = 'Expected object'): StoredObject {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new RunClientError('protocol', msg)
  }
  return value as StoredObject
}

function makeRunId(prefix: string): string {
  const token = typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID().replace(/[^A-Za-z0-9_-]/g, '')
    : Math.random().toString(36).slice(2, 12) + Date.now().toString(36)
  return `${prefix}_${token}`.slice(0, 96)
}

/**
 * Normalize a workstation configuration path to an installed preset ID:
 * `research_1d`, `research_1d.yaml`, and `examples/research_1d/device.json`
 * all map to `research_1d`.
 */
export function normalizePresetId(configPath: string): string {
  const parts = configPath.trim().replace(/\\/g, '/').split('/').filter(Boolean)
  let leaf = parts[parts.length - 1] ?? ''
  if (/^device\.(json|ya?ml)$/i.test(leaf) && parts.length >= 2) leaf = parts[parts.length - 2]
  return leaf.replace(/\.(json|ya?ml)$/i, '')
}

/**
 * Select the installed V2 route device for the requested configuration.
 * A non-null configuration that is not installed on the route is refused
 * explicitly; it is never silently replaced by another registered device.
 */
function selectRouteDevice(prep: StoredObject, configPath: string | null, route: string): string {
  const devices = Array.isArray(prep.devices) ? prep.devices.map(d => asObject(d)) : []
  if (devices.length === 0) {
    throw new RunClientError('protocol', `Installed ${route} route has no registered devices`)
  }
  const ids = devices.map(d => String(d.device_id))
  if (configPath === null) return ids[0]
  const raw = configPath.trim()
  const wanted = normalizePresetId(raw)
  const match = wanted.length > 0
    ? ids.find(id => id === raw || id === wanted || normalizePresetId(id) === wanted)
    : undefined
  if (match === undefined) {
    throw new Error(
      `Configuration "${configPath}" is not installed on the V2 ${route} route (installed: ${ids.join(', ')}); `
      + 'refusing to substitute a different device.',
    )
  }
  return match
}

export async function startJob(
  kind: DispatchableJobKind,
  _device: unknown,
  params: Record<string, unknown>,
  configPath: string | null = null,
  endpoint?: string | URL,
): Promise<string> {
  const base = resolveV2ServiceEndpoint(endpoint)
  const { value: catalog, response: expResp } = await requestJobService(base, 'experiments')
  const registrySha = expResp.headers.get('X-Solarlab-Registry') ?? ''
  const schemaSha = expResp.headers.get('X-Solarlab-Job-Schema') ?? ''
  if (!/^[a-f0-9]{64}$/.test(registrySha) || !/^[a-f0-9]{64}$/.test(schemaSha)) {
    throw new RunClientError('protocol', 'Missing X-Solarlab-Registry or X-Solarlab-Job-Schema header on GET /experiments')
  }
  if (catalog.schema !== 'solarlab.execution-catalog.v1' || !Array.isArray(catalog.bindings)) {
    throw new RunClientError('protocol', 'Invalid execution catalog document from GET /experiments')
  }
  const ceiling = asObject(catalog.resource_ceiling, 'Missing resource_ceiling')
  const bindings = catalog.bindings.map(b => asObject(b))

  let selectedEntry: StoredObject | undefined
  let inputPayload: Record<string, unknown> | undefined
  let driver = ''

  if (kind === 'jv' || kind === 'dark_jv') {
    selectedEntry = bindings.find(
      b => b.engine === 'solarlab' && b.driver === 'steady_jv' && b.scope === STEADY_SCOPE,
    )
    if (selectedEntry) {
      const prep = asObject(selectedEntry.preparation_context)
      const deviceId = selectRouteDevice(prep, configPath, 'steady_jv')
      const nPoints = Math.max(2, Math.min(101, Math.round(Number(params.n_points ?? 25))))
      const vMax = Number.isFinite(Number(params.V_max)) && Number(params.V_max) !== 0 ? Number(params.V_max) : 1.1
      const requestedGrid = Math.round(Number(params.N_grid ?? 36))
      const intervals = Math.max(2, Math.min(256, Math.max(4, Math.round(requestedGrid / 3))))
      const isDark = kind === 'dark_jv' || params.illuminated === false
      inputPayload = {
        device_id: deviceId,
        voltage_V: voltageGrid(0, vMax, nPoints),
        intervals_per_layer: intervals,
        generation_scale: isDark ? 0 : 1,
        incident_power_W_m2: null,
      }
      driver = 'steady_jv'
    }
  } else if (kind === 'tpv') {
    selectedEntry = bindings.find(
      b => b.engine === 'solarlab' && b.driver === 'transient_1d',
    )
    if (selectedEntry) {
      const prep = asObject(selectedEntry.preparation_context)
      const deviceId = selectRouteDevice(prep, configPath, 'transient_1d')
      const sampleCount = 9
      const tEnd = Math.min(1, Math.max(1e-6, Number(params.t_end ?? 1e-4)))
      const times = Array.from({ length: sampleCount }, (_, i) => (tEnd * i) / (sampleCount - 1))
      const wallCeiling = Math.min(30, Number(ceiling.wall_seconds ?? 30))
      inputPayload = {
        device_id: deviceId,
        times_s: times,
        voltage_V: times.map((_, i) => (i === 0 ? 0 : 0.02)),
        generation_scale: times.map((_, i) => (i === 1 ? 1.05 : 1.0)),
        intervals_per_layer: 12,
        controls: {
          max_steps: sampleCount - 1,
          wall_s: Math.min(20, wallCeiling),
          newton_budget: {
            max_iterations: 30,
            max_residual_evaluations: 240,
            max_jacobian_evaluations: 30,
            max_backtracks: 12,
          },
        },
      }
      driver = 'transient_1d'
    }
  } else if (kind === 'jv_2d') {
    selectedEntry = bindings.find(
      b => b.engine === 'solarlab' && b.driver === 'steady_2d',
    )
    if (selectedEntry) {
      if (params.microstructure) {
        throw new Error(
          'Grain-boundary 2D microstructure is not installed on this V2 steady_2d HTTP route; only uniform Neumann 2D steady is registered.',
        )
      }
      const prep = asObject(selectedEntry.preparation_context)
      const deviceId = selectRouteDevice(prep, configPath, 'steady_2d')
      const policy = asObject(prep.fixed_numerical_policy)
      const illum = params.illuminated !== false
      inputPayload = {
        device_id: deviceId,
        voltage_V: Number(policy.voltage_V ?? 0.002),
        generation_scales: illum ? [0, 1] : [0],
        spatial: asObject(policy.spatial),
        controls: asObject(policy.controls),
        initial: asObject(policy.initial),
      }
      driver = 'steady_2d'
    }
  }

  if (!selectedEntry || !inputPayload) {
    throw new Error(
      `Experiment "${kind}" has no installed V2 solarlab execution route in GET /experiments; legacy execution fallback is retired.`,
    )
  }

  const runId = makeRunId(driver)
  const wallLimit = Math.min(30, Number(ceiling.wall_seconds ?? 30))
  const submission = {
    schema_version: 'solarlab.job-submission.v1',
    run_id: runId,
    experiment_id: String(selectedEntry.experiment_id),
    version: Number(selectedEntry.version),
    engine: 'solarlab',
    input: inputPayload,
    limits: {
      wall_seconds: wallLimit,
      rss_bytes: Number(ceiling.rss_bytes),
      output_bytes: Number(ceiling.output_bytes),
      accepted_steps: null,
      calls: null,
    },
  }

  const { value: accepted, response: postResp } = await requestJobService(base, 'jobs', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Solarlab-Registry': registrySha,
      'X-Solarlab-Job-Schema': schemaSha,
    },
    body: JSON.stringify(submission),
  })
  if (postResp.status !== 202 || accepted.status !== 'accepted') {
    throw new RunClientError('protocol', 'Unexpected POST /jobs acceptance response')
  }

  return encodeHandle({
    v: 2,
    endpoint: base,
    storeId: String(accepted.store_id),
    runId: String(accepted.run_id),
    attemptId: String(accepted.attempt_id),
    kind,
    driver,
    catalogSha256: registrySha,
    ...(driver === 'steady_2d' && typeof inputPayload.voltage_V === 'number'
      ? { voltageV: inputPayload.voltage_V }
      : {}),
  })
}

async function adaptV2Result(
  handle: EncodedV2JobHandle,
  reader: StoredRunClient,
  target: AttemptIdentity,
): Promise<Record<string, unknown>> {
  const snapshot = (await reader.getAttempt(target)).value
  if (!snapshot.terminal || snapshot.state !== 'succeeded') {
    throw new RunClientError('protocol', `Attempt ended in state ${snapshot.state}`)
  }
  const worker = asObject(snapshot.result?.worker_result, 'Missing worker_result')
  const manifestSha = String(worker.manifest_sha256 ?? '')
  const manifestEnvelope = {
    scientific_qualification: false,
    execution: { worker_result: worker },
    execution_manifest: {
      engine: 'solarlab',
      driver: handle.driver,
      manifest_sha256: manifestSha,
      scientific_qualification: false,
    },
  }

  if (handle.driver === 'steady_jv') {
    const arrays: Record<string, FloatArray> = {}
    for (const ref of snapshot.terminal.artifacts) {
      const art = await reader.getArtifact(target, ref)
      const name = String(art.record.value.name)
      if (name === 'voltage_V' || name === 'output_current_A_m2') {
        arrays[name] = readSteadyArray(art.bytes, art.record.value)
      }
    }
    const V = arrays.voltage_V?.values ?? []
    const J = arrays.output_current_A_m2?.values ?? []
    const doc = asObject(worker.steady_result)
    const rawMetrics = asObject(doc.metrics)
    const vocRaw = rawMetrics.open_circuit_voltage_V
    const vocBracketed = typeof vocRaw === 'number' && Number.isFinite(vocRaw)
    const metrics = {
      V_oc: vocBracketed ? vocRaw : 0,
      J_sc: typeof rawMetrics.short_circuit_current_A_m2 === 'number' ? rawMetrics.short_circuit_current_A_m2 : (J[0] ?? 0),
      FF: typeof rawMetrics.fill_factor === 'number' && Number.isFinite(rawMetrics.fill_factor) ? rawMetrics.fill_factor : 0,
      PCE: typeof rawMetrics.efficiency === 'number' && Number.isFinite(rawMetrics.efficiency) ? rawMetrics.efficiency : 0,
      voc_bracketed: vocBracketed,
    }
    if (handle.kind === 'dark_jv') {
      const darkPayload: DarkJVResult = {
        V,
        J: J.map(v => Math.abs(v)),
        n_ideality: Number.NaN,
        J_0: Number.NaN,
        V_fit_lo: V[0] ?? 0,
        V_fit_hi: V[V.length - 1] ?? 0,
      }
      return { ...darkPayload, ...manifestEnvelope }
    }
    const jvPayload: JVResult = {
      V_fwd: V,
      J_fwd: J,
      V_rev: [...V].reverse(),
      J_rev: [...J].reverse(),
      metrics_fwd: metrics,
      metrics_rev: metrics,
      hysteresis_index: 0,
    }
    return { ...jvPayload, ...manifestEnvelope }
  }

  if (handle.driver === 'transient_1d') {
    const arrays: Record<string, FloatArray> = {}
    for (const ref of snapshot.terminal.artifacts) {
      const art = await reader.getArtifact(target, ref)
      const name = String(art.record.value.name)
      if (
        name === 'accepted_times_s'
        || name === 'requested_voltage_V'
        || name === 'terminal_total_current_density_A_m2'
      ) {
        arrays[name] = readSteadyArray(art.bytes, art.record.value, true)
      }
    }
    const t = arrays.accepted_times_s?.values ?? []
    const V = (arrays.requested_voltage_V?.values ?? []).slice(0, t.length)
    const termJ = arrays.terminal_total_current_density_A_m2?.values ?? []
    const J = t.map((_, i) => (i === 0 ? 0 : termJ[(i - 1) * 2] ?? 0))
    const v0 = V[0] ?? 0
    const deltaV = V.map(v => v - v0)
    const tpvPayload: TPVResult = {
      t,
      V,
      J,
      delta_V: deltaV,
      tau: null,
      V_oc: v0,
      delta_V0: deltaV[1] ?? 0,
      non_identifiable_reason: 'voltage_step_transient_1d_no_exponential_fit',
    }
    return { ...tpvPayload, ...manifestEnvelope }
  }

  if (handle.driver === 'steady_2d') {
    const arrays: Record<string, FloatArray> = {}
    for (const ref of snapshot.terminal.artifacts) {
      const art = await reader.getArtifact(target, ref)
      const name = String(art.record.value.name)
      if (
        name === 'x_m'
        || name === 'y_m'
        || name === 'section_terminal_current_A_m'
        || name === 'potential_V'
        || name === 'carrier_density_m3'
      ) {
        arrays[name] = readSteadyArray(art.bytes, art.record.value, true, 4)
      }
    }
    const summary = asObject(worker.steady_2d_result, 'Missing steady_2d_result summary')
    const xArr = arrays.x_m
    const yArr = arrays.y_m
    const secArr = arrays.section_terminal_current_A_m
    const phiArr = arrays.potential_V
    const carrierArr = arrays.carrier_density_m3
    if (!xArr || !yArr || !secArr || !phiArr || !carrierArr) {
      throw new RunClientError('protocol', 'Incomplete steady_2d NPY artifacts')
    }
    if (xArr.shape.length !== 1 || yArr.shape.length !== 1) {
      throw new RunClientError('protocol', 'Invalid steady_2d grid axis rank')
    }
    const nx = xArr.shape[0]
    const ny = yArr.shape[0]
    const lateralWidthM = Number(summary.lateral_width_m)
    if (!Number.isFinite(lateralWidthM) || lateralWidthM <= 0) {
      throw new RunClientError('protocol', 'Invalid steady_2d lateral_width_m')
    }
    if (secArr.shape.length !== 2 || secArr.shape[1] < 2) {
      throw new RunClientError('protocol', 'Invalid steady_2d section_terminal_current_A_m shape')
    }
    const k = secArr.shape[0]
    const cols = secArr.shape[1]
    if (
      phiArr.shape.length !== 3
      || phiArr.shape[0] !== k
      || phiArr.shape[1] !== ny
      || phiArr.shape[2] !== nx
      || carrierArr.shape.length !== 4
      || carrierArr.shape[0] !== 2
      || carrierArr.shape[1] !== k
      || carrierArr.shape[2] !== ny
      || carrierArr.shape[3] !== nx
    ) {
      throw new RunClientError('protocol', 'Mismatched steady_2d field dimensions')
    }

    const grid_x = xArr.values.map(v => v * 1e9)
    const grid_y = yArr.values.map(v => v * 1e9)
    const policy = worker.registered_numerical_policy && typeof worker.registered_numerical_policy === 'object'
      ? (worker.registered_numerical_policy as Record<string, unknown>)
      : {}
    const spatialDecl = worker.registered_spatial_declaration && typeof worker.registered_spatial_declaration === 'object'
      ? (worker.registered_spatial_declaration as Record<string, unknown>)
      : {}
    const lateralBc = spatialDecl.lateral_bc === 'periodic' ? 'periodic' : 'neumann'
    const operatingVoltageV = typeof handle.voltageV === 'number' && Number.isFinite(handle.voltageV)
      ? handle.voltageV
      : (typeof policy.voltage_V === 'number' && Number.isFinite(policy.voltage_V) ? policy.voltage_V : 0.002)

    const slice2D = (flat: number[], offset: number, rows: number, rowCols: number): number[][] =>
      Array.from({ length: rows }, (_, r) =>
        flat.slice(offset + r * rowCols, offset + (r + 1) * rowCols),
      )
    const zeroGrid = (rows: number, rowCols: number): number[][] =>
      Array.from({ length: rows }, () => Array.from({ length: Math.max(0, rowCols) }, () => 0))

    let V: number[] = []
    let J: number[] = []
    let snapshots: SpatialSnapshot2D[] = []
    if (k > 0) {
      const targetIdx = k - 1
      const lowerA_m = secArr.values[targetIdx * cols]
      const upperA_m = secArr.values[targetIdx * cols + (cols - 1)]
      const meanTerminalJ = (0.5 * (lowerA_m + upperA_m)) / lateralWidthM
      V = [operatingVoltageV]
      J = [meanTerminalJ]
      const cellCount = ny * nx
      snapshots = [
        {
          V: operatingVoltageV,
          x: grid_x,
          y: grid_y,
          phi: slice2D(phiArr.values, targetIdx * cellCount, ny, nx),
          n: slice2D(carrierArr.values, targetIdx * cellCount, ny, nx),
          p: slice2D(carrierArr.values, (k + targetIdx) * cellCount, ny, nx),
          Jx_n: zeroGrid(ny, nx - 1),
          Jy_n: zeroGrid(ny - 1, nx),
          Jx_p: zeroGrid(ny, nx - 1),
          Jy_p: zeroGrid(ny - 1, nx),
        },
      ]
    }

    const jv2dPayload: JV2DResult = {
      V,
      J,
      grid_x,
      grid_y,
      lateral_bc: lateralBc,
      snapshots,
      metrics: { V_oc: 0, J_sc: 0, FF: 0, PCE: 0, voc_bracketed: false },
    }
    return { ...jv2dPayload, steady_2d_summary: summary, ...manifestEnvelope }
  }

  return manifestEnvelope
}

export function streamJobEvents<TResult>(
  jobId: string,
  handlers: JobStreamHandlers<TResult>,
): () => void {
  let handle: EncodedV2JobHandle
  try {
    handle = decodeHandle(jobId)
  } catch (err) {
    queueMicrotask(() => {
      handlers.onError(err instanceof Error ? err.message : String(err))
      handlers.onDone()
    })
    return () => {}
  }

  const reader = new StoredRunClient(handle.endpoint, handle.storeId, 64 * 1024 ** 2)
  const target: AttemptIdentity = Object.freeze({ runId: handle.runId, attemptId: handle.attemptId })
  let finished = false
  const finish = () => {
    if (finished) return
    finished = true
    handlers.onDone()
  }

  const sub = reader.subscribe(target, doc => {
    const ev = doc.value
    if (ev.kind === 'queued' || ev.kind === 'running') {
      handlers.onProgress({
        stage: ev.kind,
        current: ev.kind === 'running' ? 1 : 0,
        total: 2,
        eta_s: null,
        message: `SolarLab worker ${ev.kind}`,
      })
      return
    }
    if (ev.kind === 'progress') {
      const ui = (ev.payload?.ui_progress ?? {}) as Record<string, unknown>
      handlers.onProgress({
        stage: typeof ui.stage === 'string' ? ui.stage : 'progress',
        current: typeof ui.current === 'number' ? ui.current : 1,
        total: typeof ui.total === 'number' ? ui.total : 1,
        eta_s: typeof ui.eta_s === 'number' ? ui.eta_s : null,
        message: typeof ui.message === 'string' ? ui.message : String(ui.stage ?? 'running'),
      })
      return
    }
    if (ev.kind === 'terminal') {
      const payload = ev.payload as { state: string; result?: Record<string, unknown> }
      if (payload.state !== 'succeeded') {
        const resObj = payload.result ?? {}
        const wr = (resObj.worker_result ?? resObj) as Record<string, unknown>
        const detail = wr.detail ?? wr.reason ?? payload.state
        handlers.onError(String(detail))
        finish()
        return
      }
      void adaptV2Result(handle, reader, target)
        .then(adapted => {
          handlers.onResult(adapted as unknown as TResult)
          finish()
        })
        .catch(err => {
          handlers.onError(err instanceof Error ? err.message : String(err))
          finish()
        })
    }
  })

  void sub.done
    .then(end => {
      if (!finished && (end.reason === 'disconnected' || end.reason === 'terminal_acknowledged')) {
        if (end.transportError) {
          handlers.onError(end.transportError instanceof Error ? end.transportError.message : String(end.transportError))
        }
        finish()
      }
    })
    .catch(err => {
      if (!finished) {
        handlers.onError(err instanceof Error ? err.message : String(err))
        finish()
      }
    })

  return () => {
    sub.unsubscribe()
  }
}
