// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { webcrypto } from './test-helpers/schema-fixture.mjs'
import {
  checkUserConfigExists,
  fetchOpticalMaterials,
  getConfig,
  listConfigs,
  saveUserConfig,
} from './v2-workstation-client'
import { normalizePresetId, startJob, streamJobEvents } from './v2-job-client'
import { extractExecutionPhysics, formatQualificationBadge } from './workstation/execution-physics'
import { STEADY_SCOPE } from './steady-jv-client'

const endpoint = 'http://127.0.0.1:8000/'
const registrySha = '1'.repeat(64)
const schemaSha = '2'.repeat(64)
const manifestSha = '3'.repeat(64)

function jsonResponse(body: unknown, headers: Record<string, string> = {}, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...headers },
  })
}

function makeFloat64Npy(values: number[], shape: number[] = [values.length]): Uint8Array<ArrayBuffer> {
  const shapeTuple = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`
  const header = `{'descr': '<f8', 'fortran_order': False, 'shape': ${shapeTuple}, }\n`
  const pad = (16 - ((10 + header.length) % 16)) % 16
  const padded = header.slice(0, -1) + ' '.repeat(pad) + '\n'
  const out = new Uint8Array(new ArrayBuffer(10 + padded.length + values.length * 8))
  const view = new DataView(out.buffer)
  out.set([147, 78, 85, 77, 80, 89, 1, 0])
  view.setUint16(8, padded.length, true)
  out.set(new TextEncoder().encode(padded), 10)
  values.forEach((v, i) => view.setFloat64(10 + padded.length + i * 8, v, true))
  return out
}

async function sha256Hex(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await webcrypto.subtle.digest('SHA-256', bytes)
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('')
}

describe('P09-02 V2 workstation and job client cutover', () => {
  const storage = new Map<string, string>()

  beforeEach(() => {
    storage.clear()
    vi.stubGlobal('crypto', webcrypto)
    vi.stubGlobal('localStorage', {
      getItem: (k: string) => storage.get(k) ?? null,
      setItem: (k: string, v: string) => { storage.set(k, v) },
      removeItem: (k: string) => { storage.delete(k) },
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('loads configs and optical materials from V2 /configs and /materials routes without /api calls', async () => {
    const calledUrls: string[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: string | URL) => {
      const url = String(input)
      calledUrls.push(url)
      if (url.endsWith('/configs')) {
        return jsonResponse({ sources: [{ id: 'research_1d.yaml', sha256: 'a'.repeat(64), bytes: 64 }] })
      }
      if (url.endsWith('/configs/research_1d.yaml')) {
        return new Response(
          'device:\n  V_bi: 1.05\n  Phi: 2.5e21\nlayers:\n  - name: Absorber\n    role: absorber\n    thickness: 4.0e-7\n',
          { status: 200, headers: { 'Content-Type': 'application/octet-stream' } },
        )
      }
      if (url.endsWith('/materials')) {
        return jsonResponse({
          resource_library_sha256: 'b'.repeat(64),
          resources: [
            { name: 'MAPbI3', kind: 'nk' },
            { name: 'AM15G', kind: 'spectrum' },
          ],
        })
      }
      throw new Error(`Unexpected URL: ${url}`)
    }))

    const entries = await listConfigs(endpoint)
    expect(entries.map(e => e.name)).toEqual(['research_1d.yaml'])
    const cfg = await getConfig('research_1d.yaml', endpoint)
    expect(cfg.device.V_bi).toBe(1.05)
    expect(cfg.layers[0].role).toBe('absorber')

    const mats = await fetchOpticalMaterials(endpoint)
    expect(mats).toEqual(['MAPbI3'])

    const reserved = await checkUserConfigExists('research_1d', endpoint)
    expect(reserved).toEqual({ exists: true, namespace: 'shipped' })
    const saved = await saveUserConfig('custom_stack', cfg, false, endpoint)
    expect(saved).toEqual({ ok: true, saved: 'custom_stack' })
    const reloaded = await getConfig('custom_stack.yaml', endpoint)
    expect(reloaded.device.V_bi).toBe(1.05)
    expect(calledUrls.every(u => !u.includes('/api/'))).toBe(true)
  })

  it('submits V2 engine="solarlab" jobs, streams durable /runs SSE, reads NPY artifacts, and preserves UNQUALIFIED badge', async () => {
    const vBytes = makeFloat64Npy([0, 0.5, 1.0])
    const jBytes = makeFloat64Npy([200, 180, -20])
    const vSha = await sha256Hex(vBytes)
    const jSha = await sha256Hex(jBytes)
    const vManifestSha = '4'.repeat(64)
    const jManifestSha = '5'.repeat(64)
    const calledUrls: string[] = []
    const captured: { submission: Record<string, unknown> | null } = { submission: null }

    vi.stubGlobal('fetch', vi.fn(async (input: string | URL, init?: RequestInit) => {
      const url = String(input)
      calledUrls.push(url)
      if (url === `${endpoint}experiments`) {
        return jsonResponse(
          {
            schema: 'solarlab.execution-catalog.v1',
            job_schema_sha256: schemaSha,
            resource_ceiling: { wall_seconds: 30, rss_bytes: 1073741824, output_bytes: 67108864, accepted_steps: null, calls: null },
            bindings: [
              {
                experiment_id: 'steady_jv',
                version: 1,
                engine: 'solarlab',
                driver: 'steady_jv',
                scope: STEADY_SCOPE,
                scientific_qualification: false,
                preparation_context: {
                  devices: [{ device_id: 'research_1d', prepared_sha256: 'c'.repeat(64) }],
                },
              },
            ],
          },
          {
            'X-Solarlab-Registry': registrySha,
            'X-Solarlab-Job-Schema': schemaSha,
            ETag: `"${registrySha}"`,
          },
        )
      }
      if (url === `${endpoint}jobs`) {
        const parsed = JSON.parse(String(init?.body)) as Record<string, unknown>
        captured.submission = parsed
        return jsonResponse(
          {
            status: 'accepted',
            store_id: 'store_v2',
            run_id: parsed.run_id,
            attempt_id: 'att_1',
            execution_catalog_sha256: registrySha,
            job_schema_sha256: schemaSha,
            scientific_qualification: false,
          },
          {},
          202,
        )
      }
      const runId = String(captured.submission?.run_id ?? '')
      if (url.includes('/attempts/att_1/events')) {
        const sse = [
          `id: 1\nevent: queued\ndata: ${JSON.stringify({ sequence: 1, event_key: 'queued', run_id: runId, attempt_id: 'att_1', generation: 0, kind: 'queued', created_ns: 1, payload: {} })}\n\n`,
          `id: 2\nevent: terminal\ndata: ${JSON.stringify({ sequence: 2, event_key: 'terminal', run_id: runId, attempt_id: 'att_1', generation: 1, kind: 'terminal', created_ns: 2, payload: { state: 'succeeded', result: { worker_result: { manifest_sha256: manifestSha, scientific_qualification: false } }, artifacts: [] } })}\n\n`,
        ].join('')
        return new Response(sse, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream', 'X-RunStore-ID': 'store_v2' },
        })
      }
      if (url.endsWith('/attempts/att_1')) {
        const attemptResult = {
          worker_result: {
            manifest_sha256: manifestSha,
            scientific_qualification: false,
            steady_result: {
              metrics: {
                open_circuit_voltage_V: 0.9,
                short_circuit_current_A_m2: 200,
                fill_factor: 0.75,
                efficiency: null,
              },
            },
          },
        }
        return jsonResponse(
          {
            attempt_id: 'att_1',
            run_id: runId,
            attempt_number: 1,
            generation: 1,
            state: 'succeeded',
            created_ns: 1,
            started_ns: 1,
            finished_ns: 2,
            owner_id: 'sup',
            result: attemptResult,
            terminal: {
              state: 'succeeded',
              result: attemptResult,
              artifacts: [
                { artifact_id: 'art_v', manifest_sha256: vManifestSha },
                { artifact_id: 'art_j', manifest_sha256: jManifestSha },
              ],
            },
          },
          { 'X-RunStore-ID': 'store_v2' },
        )
      }
      if (url === `${endpoint}artifacts/art_v`) {
        return jsonResponse(
          {
            artifact_id: 'art_v',
            run_id: runId,
            attempt_id: 'att_1',
            generation: 1,
            name: 'voltage_V',
            format: 'npy',
            phase: 'committed',
            sha256: vSha,
            manifest_sha256: vManifestSha,
            size_bytes: vBytes.byteLength,
            relative_path: 'objects/v',
            metadata: { dtype: '<f8', order: 'C', shape: [3] },
          },
          { 'X-RunStore-ID': 'store_v2' },
        )
      }
      if (url === `${endpoint}artifacts/art_v/content`) {
        return new Response(vBytes, {
          status: 200,
          headers: { 'Content-Type': 'application/octet-stream', 'X-RunStore-ID': 'store_v2', ETag: `"${vSha}"` },
        })
      }
      if (url === `${endpoint}artifacts/art_j`) {
        return jsonResponse(
          {
            artifact_id: 'art_j',
            run_id: runId,
            attempt_id: 'att_1',
            generation: 1,
            name: 'output_current_A_m2',
            format: 'npy',
            phase: 'committed',
            sha256: jSha,
            manifest_sha256: jManifestSha,
            size_bytes: jBytes.byteLength,
            relative_path: 'objects/j',
            metadata: { dtype: '<f8', order: 'C', shape: [3] },
          },
          { 'X-RunStore-ID': 'store_v2' },
        )
      }
      if (url === `${endpoint}artifacts/art_j/content`) {
        return new Response(jBytes, {
          status: 200,
          headers: { 'Content-Type': 'application/octet-stream', 'X-RunStore-ID': 'store_v2', ETag: `"${jSha}"` },
        })
      }
      throw new Error(`Unexpected URL: ${url}`)
    }))

    const handle = await startJob('jv', null, { n_points: 3, V_max: 1.0, N_grid: 36 }, null, endpoint)
    expect(captured.submission?.engine).toBe('solarlab')

    const completed = await new Promise<Record<string, unknown>>((resolve, reject) => {
      let resultObj: Record<string, unknown> | undefined
      streamJobEvents<Record<string, unknown>>(handle, {
        onProgress: () => {},
        onResult: r => { resultObj = r },
        onError: msg => reject(new Error(msg)),
        onDone: () => {
          if (resultObj) resolve(resultObj)
          else reject(new Error('stream closed without result'))
        },
      })
    })

    const { pure, receipt } = extractExecutionPhysics(completed)
    expect(pure.V_fwd).toEqual([0, 0.5, 1.0])
    expect(pure.J_fwd).toEqual([200, 180, -20])
    expect(receipt.engine).toBe('solarlab')
    expect(receipt.scientificQualification).toBe(false)
    expect(formatQualificationBadge(receipt).label).toBe('Unqualified · Engineering Preview')
    expect(calledUrls.every(u => !u.includes('/api/'))).toBe(true)

    // Uninstalled capability refuses cleanly without falling back to legacy endpoints.
    await expect(startJob('eqe', null, {}, null, endpoint)).rejects.toThrow('legacy execution fallback is retired')
  })

  it('decodes 1D-4D steady_2d NPY artifacts into real grid_x, grid_y, terminal mean J, and 2D snapshots', async () => {
    const nx = 3
    const ny = 4
    const k = 2
    const xBytes = makeFloat64Npy([0, 4.0e-7, 8.0e-7], [nx])
    const yBytes = makeFloat64Npy([0, 1.0e-7, 2.5e-7, 4.0e-7], [ny])
    const secBytes = makeFloat64Npy(
      [
        1.6e-9, 1.6e-9, 1.6e-9, 1.6e-9, 1.6e-9,
        -1.76e-4, -1.76e-4, -1.76e-4, -1.76e-4, -1.76e-4,
      ],
      [k, ny + 1],
    )
    const phiValues = Array.from({ length: k * ny * nx }, (_, i) => 0.01 * i)
    const phiBytes = makeFloat64Npy(phiValues, [k, ny, nx])
    const carrierValues = Array.from({ length: 2 * k * ny * nx }, (_, i) => 1.0e20 + 1.0e18 * i)
    const carrierBytes = makeFloat64Npy(carrierValues, [2, k, ny, nx])

    const artifacts = [
      { id: 'art_x', name: 'x_m', bytes: xBytes, shape: [nx], msha: '6'.repeat(64) },
      { id: 'art_y', name: 'y_m', bytes: yBytes, shape: [ny], msha: '7'.repeat(64) },
      { id: 'art_sec', name: 'section_terminal_current_A_m', bytes: secBytes, shape: [k, ny + 1], msha: '8'.repeat(64) },
      { id: 'art_phi', name: 'potential_V', bytes: phiBytes, shape: [k, ny, nx], msha: '9'.repeat(64) },
      { id: 'art_car', name: 'carrier_density_m3', bytes: carrierBytes, shape: [2, k, ny, nx], msha: 'a'.repeat(64) },
    ]
    const shas = new Map<string, string>()
    for (const a of artifacts) {
      shas.set(a.id, await sha256Hex(a.bytes))
    }

    const captured: { submission: Record<string, unknown> | null } = { submission: null }
    vi.stubGlobal('fetch', vi.fn(async (input: string | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === `${endpoint}experiments`) {
        return jsonResponse(
          {
            schema: 'solarlab.execution-catalog.v1',
            job_schema_sha256: schemaSha,
            resource_ceiling: { wall_seconds: 30, rss_bytes: 1073741824, output_bytes: 67108864, accepted_steps: null, calls: null },
            bindings: [
              {
                experiment_id: 'steady_2d',
                version: 1,
                engine: 'solarlab',
                driver: 'steady_2d',
                scope: 'uniform_no_grain_classical_2d_steady',
                scientific_qualification: false,
                preparation_context: {
                  devices: [{ device_id: 'research_1d', prepared_sha256: 'c'.repeat(64) }],
                  fixed_numerical_policy: {
                    voltage_V: 0.002,
                    spatial: { lateral_length_m: 8.0e-7, Nx: 2, Ny_per_layer: 1, lateral_bc: 'neumann', geometry_policy: 'rectangular_half_endpoint_neumann_2d_v1' },
                    controls: { auxiliary_1d: {}, full_2d: {} },
                    initial: { scheme: 'auxiliary_1d_broadcast' },
                  },
                },
              },
            ],
          },
          {
            'X-Solarlab-Registry': registrySha,
            'X-Solarlab-Job-Schema': schemaSha,
            ETag: `"${registrySha}"`,
          },
        )
      }
      if (url === `${endpoint}jobs`) {
        const parsed = JSON.parse(String(init?.body)) as Record<string, unknown>
        captured.submission = parsed
        return jsonResponse(
          {
            status: 'accepted',
            store_id: 'store_2d',
            run_id: parsed.run_id,
            attempt_id: 'att_2d',
            execution_catalog_sha256: registrySha,
            job_schema_sha256: schemaSha,
            scientific_qualification: false,
          },
          {},
          202,
        )
      }
      const runId = String(captured.submission?.run_id ?? '')
      if (url.includes('/attempts/att_2d/events')) {
        const sse = [
          `id: 1\nevent: terminal\ndata: ${JSON.stringify({ sequence: 1, event_key: 'terminal', run_id: runId, attempt_id: 'att_2d', generation: 1, kind: 'terminal', created_ns: 2, payload: { state: 'succeeded', result: { worker_result: { manifest_sha256: manifestSha, scientific_qualification: false } }, artifacts: [] } })}\n\n`,
        ].join('')
        return new Response(sse, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream', 'X-RunStore-ID': 'store_2d' },
        })
      }
      if (url.endsWith('/attempts/att_2d')) {
        const attemptResult = {
          worker_result: {
            manifest_sha256: manifestSha,
            scientific_qualification: false,
            registered_numerical_policy: { voltage_V: 0.002 },
            registered_spatial_declaration: { lateral_bc: 'neumann' },
            steady_2d_result: {
              schema: 'solarlab.steady-2d.result.v1',
              accepted_count: k,
              lateral_width_m: 8.0e-7,
              shape_Ny_Nx: [ny, nx],
              scientific_qualification: false,
            },
          },
        }
        return jsonResponse(
          {
            attempt_id: 'att_2d',
            run_id: runId,
            attempt_number: 1,
            generation: 1,
            state: 'succeeded',
            created_ns: 1,
            started_ns: 1,
            finished_ns: 2,
            owner_id: 'sup',
            result: attemptResult,
            terminal: {
              state: 'succeeded',
              result: attemptResult,
              artifacts: artifacts.map(a => ({ artifact_id: a.id, manifest_sha256: a.msha })),
            },
          },
          { 'X-RunStore-ID': 'store_2d' },
        )
      }
      for (const a of artifacts) {
        const sha = shas.get(a.id)!
        if (url === `${endpoint}artifacts/${a.id}`) {
          return jsonResponse(
            {
              artifact_id: a.id,
              run_id: runId,
              attempt_id: 'att_2d',
              generation: 1,
              name: a.name,
              format: 'npy',
              phase: 'committed',
              sha256: sha,
              manifest_sha256: a.msha,
              size_bytes: a.bytes.byteLength,
              relative_path: `objects/${a.id}`,
              metadata: { dtype: '<f8', order: 'C', shape: a.shape },
            },
            { 'X-RunStore-ID': 'store_2d' },
          )
        }
        if (url === `${endpoint}artifacts/${a.id}/content`) {
          return new Response(a.bytes, {
            status: 200,
            headers: { 'Content-Type': 'application/octet-stream', 'X-RunStore-ID': 'store_2d', ETag: `"${sha}"` },
          })
        }
      }
      throw new Error(`Unexpected URL: ${url}`)
    }))

    const handle = await startJob('jv_2d', null, { illuminated: true }, 'examples/research_1d/device.json', endpoint)
    const completed = await new Promise<Record<string, unknown>>((resolve, reject) => {
      let resultObj: Record<string, unknown> | undefined
      streamJobEvents<Record<string, unknown>>(handle, {
        onProgress: () => {},
        onResult: r => { resultObj = r },
        onError: msg => reject(new Error(msg)),
        onDone: () => {
          if (resultObj) resolve(resultObj)
          else reject(new Error('stream closed without result'))
        },
      })
    })

    const { pure, receipt } = extractExecutionPhysics(completed)
    expect(pure.grid_x).toEqual([0, 400, 800])
    expect(pure.grid_y).toEqual([0, 100, 250, 400])
    expect(pure.V).toEqual([0.002])
    expect((pure.J as number[]).length).toBe(1)
    expect((pure.J as number[])[0]).toBeCloseTo(-220.0, 10)
    const snaps = pure.snapshots as Array<{ V: number; phi: number[][]; n: number[][]; p: number[][] }>
    expect(snaps).toHaveLength(1)
    expect(snaps[0].V).toBe(0.002)
    expect(snaps[0].phi).toHaveLength(ny)
    expect(snaps[0].phi[0]).toEqual([0.12, 0.13, 0.14])
    expect(snaps[0].n[0][0]).toBeCloseTo(1.0e20 + 1.0e18 * 12, 4)
    expect(snaps[0].p[0][0]).toBeCloseTo(1.0e20 + 1.0e18 * 36, 4)
    expect(receipt.scientificQualification).toBe(false)
  })

  it('normalizes configPath to select matching installed V2 device and refuses uninstalled presets without falling back to devices[0]', async () => {
    expect(normalizePresetId('research_1d')).toBe('research_1d')
    expect(normalizePresetId('research_1d.yaml')).toBe('research_1d')
    expect(normalizePresetId('examples/research_1d/device.json')).toBe('research_1d')

    const submittedDeviceIds: string[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: string | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === `${endpoint}experiments`) {
        return jsonResponse(
          {
            schema: 'solarlab.execution-catalog.v1',
            job_schema_sha256: schemaSha,
            resource_ceiling: { wall_seconds: 30, rss_bytes: 1073741824, output_bytes: 67108864, accepted_steps: null, calls: null },
            bindings: [
              {
                experiment_id: 'steady_jv',
                version: 1,
                engine: 'solarlab',
                driver: 'steady_jv',
                scope: STEADY_SCOPE,
                scientific_qualification: false,
                preparation_context: {
                  devices: [
                    { device_id: 'research_1d', prepared_sha256: 'c'.repeat(64) },
                    { device_id: 'classical_pin', prepared_sha256: 'd'.repeat(64) },
                  ],
                },
              },
            ],
          },
          {
            'X-Solarlab-Registry': registrySha,
            'X-Solarlab-Job-Schema': schemaSha,
            ETag: `"${registrySha}"`,
          },
        )
      }
      if (url === `${endpoint}jobs`) {
        const parsed = JSON.parse(String(init?.body)) as { run_id: string; input: { device_id: string } }
        submittedDeviceIds.push(parsed.input.device_id)
        return jsonResponse(
          {
            status: 'accepted',
            store_id: 'store_v2',
            run_id: parsed.run_id,
            attempt_id: 'att_1',
            execution_catalog_sha256: registrySha,
            job_schema_sha256: schemaSha,
            scientific_qualification: false,
          },
          {},
          202,
        )
      }
      throw new Error(`Unexpected URL: ${url}`)
    }))

    await startJob('jv', null, { n_points: 3, V_max: 1.0 }, 'classical_pin.yaml', endpoint)
    await startJob('jv', null, { n_points: 3, V_max: 1.0 }, 'examples/research_1d/device.json', endpoint)
    expect(submittedDeviceIds).toEqual(['classical_pin', 'research_1d'])

    await expect(startJob('jv', null, {}, 'calado2016_ion_sweep.yaml', endpoint)).rejects.toThrow(
      'Configuration "calado2016_ion_sweep.yaml" is not installed on the V2 steady_jv route',
    )
    await expect(startJob('jv', null, {}, 'custom_stack.yaml', endpoint)).rejects.toThrow(
      'Configuration "custom_stack.yaml" is not installed on the V2 steady_jv route',
    )
    expect(submittedDeviceIds).toEqual(['classical_pin', 'research_1d'])
  })
})
