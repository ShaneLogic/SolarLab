import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  newPlot: vi.fn(() => Promise.resolve()),
  purge: vi.fn(),
  resize: vi.fn(() => Promise.resolve()),
  startJob: vi.fn(),
  streamJobEvents: vi.fn(),
  listConfigs: vi.fn(() => Promise.resolve([{ name: 'tandem_lin2019.yaml', device_type: 'tandem_2T' }])),
  mountTandemStackVisualizer: vi.fn(() => ({ update: vi.fn(), clear: vi.fn() })),
}))

vi.mock('plotly.js-basic-dist-min', () => ({
  default: {
    newPlot: mocks.newPlot,
    purge: mocks.purge,
    Plots: { resize: mocks.resize },
  },
}))

vi.mock('../job-stream', () => ({
  startJob: mocks.startJob,
  streamJobEvents: mocks.streamJobEvents,
}))

vi.mock('../api', () => ({
  BASE: '',
  listConfigs: mocks.listConfigs,
}))

vi.mock('../stack/tandem-stack-visualizer', () => ({
  mountTandemStackVisualizer: mocks.mountTandemStackVisualizer,
}))

import {
  extractExecutionPhysics,
  formatQualificationBadge,
  resolveRunExecutionPhysics,
} from './execution-physics'
import {
  EXPERIMENT_CATALOG,
  getExperimentTreeLabelHTML,
  getSelectableExperimentGroups,
} from './experiment-catalog'
import { renderTreeHTML } from './tree'
import { mountExperimentPane } from './panes/experiment-pane'
import {
  mountMainPlotPane,
  renderDegradation,
  renderJV,
  renderJV2D,
  renderTPV,
  renderVocGrainSweep,
} from './panes/main-plot-pane'
import { mountTPVPane } from './panes/tpv-pane'
import { mountJVPane } from './panes/jv-pane'
import { mountJV2DPane } from './panes/jv-2d-pane'
import { mountDegradationPane } from './panes/degradation-pane'
import { mountVocGrainSweepPane } from './panes/voc-grain-sweep-pane'
import { mountTandemPane } from './panes/tandem-pane'
import type { Workspace, Run } from './types'
import type {
  DegResult,
  DeviceConfig,
  JV2DResult,
  JVResult,
  TandemJVPayload,
  TPVResult,
  VocGrainSweepResult,
} from '../types'

const SAMPLE_CONFIG: DeviceConfig = {
  device: { V_bi: 1.1, Phi: 1.4e21, mode: 'full' },
  layers: [],
}

const flushAsync = (): Promise<void> => new Promise(resolve => setTimeout(resolve, 0))

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  document.body.replaceChildren()
})

describe('P06-08: execution-manifest physics & scientific_qualification badges (no TIER_FLAGS guessing)', () => {
  it('extracts backend execution_manifest & scientific_qualification=false and strips envelope keys from pure payload', () => {
    const rawPayload = {
      V_fwd: [0, 1.0],
      J_fwd: [200, -10],
      V_rev: [1.0, 0],
      J_rev: [-10, 200],
      metrics_fwd: { V_oc: 0.95, J_sc: 200, FF: 0.75, PCE: 0.1425, voc_bracketed: true },
      metrics_rev: { V_oc: 0.95, J_sc: 200, FF: 0.75, PCE: 0.1425, voc_bracketed: true },
      hysteresis_index: 0.0,
      active_physics: 'legacy compatibility shim',
      scientific_qualification: false,
      run_id: 'run-99',
      attempt_id: 'att-1',
      execution: {
        worker_result: {
          engine: 'legacy',
          driver: 'legacy_jv_compat',
          scientific_qualification: false,
          manifest_sha256: 'd'.repeat(64),
        },
      },
      execution_manifest: {
        engine: 'legacy',
        driver: 'legacy_jv_compat',
        manifest_sha256: 'd'.repeat(64),
        scientific_qualification: false,
        effective_physics: {
          transport: 'boltzmann_sg',
          recombination: 'bulk_srh_only',
        },
      },
    }

    const { pure, receipt } = extractExecutionPhysics(rawPayload)
    expect('execution' in pure).toBe(false)
    expect('execution_manifest' in pure).toBe(false)
    expect('active_physics' in pure).toBe(false)
    expect('scientific_qualification' in pure).toBe(false)
    expect('run_id' in pure).toBe(false)
    expect(pure.V_fwd).toEqual([0, 1.0])

    expect(receipt.source).toBe('execution_manifest')
    expect(receipt.engine).toBe('legacy')
    expect(receipt.driver).toBe('legacy_jv_compat')
    expect(receipt.scientificQualification).toBe(false)
    expect(receipt.badgeLabel).toBe('LEGACY · legacy_jv_compat')
    expect(receipt.summary).toContain('transport=boltzmann_sg')

    const qual = formatQualificationBadge(receipt)
    expect(qual.qualified).toBe(false)
    expect(qual.explicitUnqualified).toBe(true)
    expect(qual.label).toBe('Unqualified · Engineering Preview')
  })

  it('renders backend execution_manifest & Unqualified · Engineering Preview in main-plot-pane and tree even when deviceSnapshot.device.mode is "full"', () => {
    const rawReceipt = extractExecutionPhysics({
      execution_manifest: {
        engine: 'solarlab',
        driver: 'metastable_split_preview',
        scientific_qualification: false,
        manifest_sha256: 'e'.repeat(64),
        effective_physics: { defects: 'operator_split_preview' },
      },
    }).receipt

    const run: Run = {
      id: 'run-unqual',
      timestamp: 1_700_000_000_000,
      durationMs: 42,
      activePhysics: rawReceipt.summary,
      executionPhysics: rawReceipt,
      scientificQualification: false,
      // Intentionally set deviceSnapshot.device.mode = 'full' to prove post-run UI ignores it!
      deviceSnapshot: SAMPLE_CONFIG,
      result: {
        kind: 'jv',
        data: {
          V_fwd: [0, 1],
          J_fwd: [200, -10],
          V_rev: [1, 0],
          J_rev: [-10, 200],
          metrics_fwd: { V_oc: 0.92, J_sc: 200, FF: 0.72, PCE: 0.132, voc_bracketed: true },
          metrics_rev: { V_oc: 0.92, J_sc: 200, FF: 0.72, PCE: 0.132, voc_bracketed: true },
          hysteresis_index: 0,
        },
      },
    }

    const ws: Workspace = {
      version: 1,
      id: 'ws-1',
      name: 'Manifest Test',
      layout: null,
      activeDeviceId: 'dev-1',
      activeExperimentId: 'exp-1',
      activeRunId: 'run-unqual',
      devices: [
        {
          id: 'dev-1',
          name: 'Declared Full Stack',
          tier: 'full',
          config: SAMPLE_CONFIG,
          experiments: [{ id: 'exp-1', kind: 'jv', params: {}, runs: [run] }],
        },
      ],
    }

    const host = document.createElement('div')
    document.body.appendChild(host)
    const plotHandle = mountMainPlotPane(host)
    plotHandle.update(ws)

    const header = host.querySelector<HTMLElement>('#mpp-header')!
    expect(header.textContent).not.toContain('FULL ·')
    expect(
      host.querySelector('[data-test="run-manifest-badge"]')?.textContent,
    ).toBe('SOLARLAB · metastable_split_preview')
    const qualBadge = host.querySelector<HTMLElement>('[data-test="run-qualification-badge"]')!
    expect(qualBadge.textContent).toBe('Unqualified · Engineering Preview')
    expect(qualBadge.getAttribute('data-qualified')).toBe('false')

    const treeHtml = renderTreeHTML(ws)
    expect(treeHtml).toContain('data-test="tree-run-qualification"')
    expect(treeHtml).toContain('PREVIEW')
    plotHandle.dispose()

    const fallbackReceipt = resolveRunExecutionPhysics({
      activePhysics: 'unknown',
      scientificQualification: true,
    })
    expect(formatQualificationBadge(fallbackReceipt).label).toBe('Scientifically Qualified')
  })
})

describe('P06-08: explicit diagnostic states for failed points, non-identifiable tau, and invalid V_oc', () => {
  it('surfaces tau === null + non_identifiable_reason in renderTPV and mountTPVPane', async () => {
    const tpvPayload: TPVResult = {
      t: [0, 1e-6, 2e-6, 3e-6],
      V: [1.05, 1.0501, 1.05009, 1.05009],
      J: [0, 0, 0, 0],
      delta_V: [0, 1e-4, 9e-5, 9e-5],
      tau: null,
      V_oc: 1.05,
      delta_V0: 1e-4,
      non_identifiable_reason: 'plateau_unresolved_ion_drift',
      fit: {
        tau_s: null,
        amplitude_V: 1e-4,
        status: 'insufficient_decay_window',
        non_identifiable_reason: 'plateau_unresolved_ion_drift',
        points: 4,
        r_squared: null,
        normalized_max_error: null,
      },
    }

    const plotHost = document.createElement('div')
    renderTPV(plotHost, tpvPayload)
    const tpvWarn = plotHost.querySelector<HTMLElement>('[data-test="tpv-diagnostic-warning"]')
    expect(tpvWarn).not.toBeNull()
    expect(tpvWarn?.getAttribute('role')).toBe('alert')
    expect(tpvWarn?.textContent).toContain('Lifetime τ is non-identifiable')
    expect(tpvWarn?.textContent).toContain('plateau_unresolved_ion_drift')

    const capturedRuns: Run[] = []
    mocks.startJob.mockResolvedValueOnce('job-tpv-1')
    mocks.streamJobEvents.mockImplementationOnce((_jobId, handlers) => {
      handlers.onResult({
        ...tpvPayload,
        scientific_qualification: false,
        execution_manifest: { engine: 'solarlab', driver: 'tpv_1d', scientific_qualification: false },
      })
      handlers.onDone?.()
    })

    const paneHost = document.createElement('div')
    document.body.appendChild(paneHost)
    mountTPVPane(paneHost, {
      getActiveDevice: () => ({ id: 'd1', config: SAMPLE_CONFIG }),
      onRunComplete: (_id, run) => {
        capturedRuns.push(run)
      },
    })
    paneHost.querySelector<HTMLButtonElement>('#btn-tpv')!.click()
    await flushAsync()

    expect(paneHost.querySelector('#status-tpv')?.textContent).toContain(
      'τ non-identifiable (plateau_unresolved_ion_drift)',
    )
    expect(capturedRuns[0]?.scientificQualification).toBe(false)
  })

  it('surfaces unbracketed V_oc and failed sweep points in 1D J-V, 2D J-V, Tandem, Degradation, and V_oc(L_g)', async () => {
    // 1. 1D J-V
    const jvBad: JVResult = {
      V_fwd: [0, 0.4, 0.8],
      J_fwd: [210, 195, 80],
      V_rev: [0.8, 0.4, 0],
      J_rev: [85, 196, 210],
      metrics_fwd: { V_oc: 0, J_sc: 210, FF: 0, PCE: 0, voc_bracketed: false },
      metrics_rev: { V_oc: 0, J_sc: 210, FF: 0, PCE: 0, voc_bracketed: false },
      hysteresis_index: 0,
      failed_points: [{ index: 3, bias_V: 1.0, reason: 'Newton step stalled' }],
    }
    const jvPlot = document.createElement('div')
    renderJV(jvPlot, jvBad)
    const jvWarn = jvPlot.querySelector<HTMLElement>('[data-test="jv-diagnostic-warning"]')
    expect(jvWarn?.getAttribute('role')).toBe('alert')
    expect(jvWarn?.textContent).toContain('Forward sweep V_oc not bracketed')
    expect(jvWarn?.textContent).toContain('Newton step stalled')

    mocks.startJob.mockResolvedValueOnce('job-jv-1')
    mocks.streamJobEvents.mockImplementationOnce((_jobId, handlers) => {
      handlers.onResult(jvBad)
      handlers.onDone?.()
    })
    const jvPane = document.createElement('div')
    document.body.appendChild(jvPane)
    mountJVPane(jvPane, {
      getActiveDevice: () => ({ id: 'd1', config: SAMPLE_CONFIG }),
      onRunComplete: () => {},
    })
    jvPane.querySelector<HTMLButtonElement>('#btn-jvp')!.click()
    await flushAsync()
    expect(jvPane.querySelector('#status-jvp')?.textContent).toContain('fwd V_oc not bracketed')
    expect(jvPane.querySelector('#status-jvp')?.textContent).toContain('1 failed point(s)')

    // 2. 2D J-V
    const jv2dBad: JV2DResult = {
      V: [0, 0.5],
      J: [-200, -120],
      grid_x: [0, 500],
      grid_y: [0, 250, 500],
      lateral_bc: 'neumann',
      snapshots: [],
      metrics: { V_oc: 0, J_sc: 200, FF: 0, PCE: 0, voc_bracketed: false },
      failed_points: [{ index: 2, bias_V: 1.0, reason: '2D GMRES breakdown' }],
    }
    const jv2dPlot = document.createElement('div')
    renderJV2D(jv2dPlot, jv2dBad)
    expect(
      jv2dPlot.querySelector('[data-test="jv2d-voc-not-bracketed"]')?.getAttribute('role'),
    ).toBe('alert')
    expect(
      jv2dPlot.querySelector('[data-test="jv2d-failed-points-warning"]')?.textContent,
    ).toContain('2D GMRES breakdown')

    mocks.startJob.mockResolvedValueOnce('job-jv2d-1')
    mocks.streamJobEvents.mockImplementationOnce((_jobId, handlers) => {
      handlers.onResult({
        ...jv2dBad,
        execution_manifest: {
          engine: 'solarlab',
          driver: 'jv_2d_fvm',
          scientific_qualification: true,
          effective_physics: { poisson: '2d_anisotropic' },
        },
      })
      handlers.onDone?.()
    })
    const jv2dPane = document.createElement('div')
    document.body.appendChild(jv2dPane)
    mountJV2DPane(jv2dPane, {
      getActiveDevice: () => ({ id: 'd1', config: SAMPLE_CONFIG }),
      onRunComplete: () => {},
    })
    jv2dPane.querySelector<HTMLButtonElement>('#btn-jv2d')!.click()
    await flushAsync()
    expect(jv2dPane.querySelector('#status-jv2d')?.textContent).toContain('V_oc not bracketed')
    expect(jv2dPane.querySelector('#jv2d-active-physics')?.textContent).toContain(
      'poisson=2d_anisotropic',
    )

    // 3. Tandem J-V
    const tandemBad: TandemJVPayload = {
      V: [0, 1.0],
      J: [190, 80],
      V_top: [0, 0.5],
      V_bot: [0, 0.5],
      metrics: {
        V_oc: 0,
        open_circuit_voltage_V: null,
        voc_bracketed: false,
        J_sc: 190,
        FF: 0,
        PCE: 0,
      },
      benchmark: null,
      failed_points: [{ index: 2, label: 'J_match=0', reason: 'subcell bracket miss' }],
    }
    mocks.startJob.mockResolvedValueOnce('job-tandem-1')
    mocks.streamJobEvents.mockImplementationOnce((_jobId, handlers) => {
      handlers.onResult({
        ...tandemBad,
        scientific_qualification: false,
        execution_manifest: {
          engine: 'solarlab',
          driver: 'tandem_2t_series',
          scientific_qualification: false,
        },
      })
      handlers.onDone?.()
    })
    const tandemPane = document.createElement('div')
    document.body.appendChild(tandemPane)
    mountTandemPane(tandemPane)
    await flushAsync()
    tandemPane.querySelector<HTMLButtonElement>('#btn-tandem')!.click()
    await flushAsync()

    expect(
      tandemPane.querySelector('[data-test="tandem-qualification-badge"]')?.textContent,
    ).toBe('Unqualified · Engineering Preview')
    const tandemWarn = tandemPane.querySelector('[data-test="tandem-diagnostic-warning"]')
    expect(tandemWarn?.getAttribute('role')).toBe('alert')
    expect(tandemWarn?.textContent).toContain('Tandem V_oc not bracketed')
    expect(tandemWarn?.textContent).toContain('subcell bracket miss')
    expect(tandemPane.querySelector('.tandem-metrics-pre')?.textContent).toContain(
      'V_oc  = unavailable (not bracketed)',
    )

    // 4. Degradation
    const pceRaw = [0.18, 0.16, Number.NaN]
    const degBad: DegResult = {
      times: [0, 50, 100],
      PCE: [...pceRaw],
      V_oc: [1.08, 1.05, Number.NaN],
      J_sc: [210, 208, 200],
      valid: [true, true, false],
      failed_points: [{ index: 2, label: 't=100 s', reason: 'J-V probe failed' }],
    }
    const degPlot = document.createElement('div')
    renderDegradation(degPlot, degBad)
    expect(degBad.PCE).toEqual(pceRaw)
    expect(
      degPlot.querySelector('[data-test="degradation-diagnostic-warning"]')?.textContent,
    ).toContain('J-V probe failed')

    mocks.startJob.mockResolvedValueOnce('job-deg-1')
    mocks.streamJobEvents.mockImplementationOnce((_jobId, handlers) => {
      handlers.onResult(degBad)
      handlers.onDone?.()
    })
    const degPane = document.createElement('div')
    document.body.appendChild(degPane)
    mountDegradationPane(degPane, {
      getActiveDevice: () => ({ id: 'd1', config: SAMPLE_CONFIG }),
      onRunComplete: () => {},
    })
    degPane.querySelector<HTMLButtonElement>('#btn-deg')!.click()
    await flushAsync()
    expect(degPane.querySelector('#status-deg')?.textContent).toContain(
      'failed/invalid trajectory step(s)',
    )

    // 5. V_oc(L_g) grain sweep
    const vgsBad: VocGrainSweepResult = {
      grain_sizes_nm: [200, 500, 1000],
      V_oc_V: [0.98, 0, 1.06],
      voc_bracketed: [true, false, true],
      J_sc_Am2: [205, 206, 208],
      FF: [0.74, 0, 0.79],
      failed_points: [{ index: 1, label: 'L_g=500 nm', reason: 'V_oc above V_max' }],
    }
    const vgsPlot = document.createElement('div')
    renderVocGrainSweep(vgsPlot, vgsBad)
    const vgsWarn = vgsPlot.querySelector('[data-test="voc-grain-sweep-diagnostic-warning"]')
    expect(vgsWarn?.getAttribute('role')).toBe('alert')
    expect(vgsWarn?.textContent).toContain('L_g=500 nm')
    expect(vgsWarn?.textContent).toContain('V_oc above V_max')

    mocks.startJob.mockResolvedValueOnce('job-vgs-1')
    mocks.streamJobEvents.mockImplementationOnce((_jobId, handlers) => {
      handlers.onResult(vgsBad)
      handlers.onDone?.()
    })
    const vgsPane = document.createElement('div')
    document.body.appendChild(vgsPane)
    mountVocGrainSweepPane(vgsPane, {
      getActiveDevice: () => ({ id: 'd1', config: SAMPLE_CONFIG }),
      onRunComplete: () => {},
    })
    vgsPane.querySelector<HTMLButtonElement>('#btn-vgs')!.click()
    await flushAsync()
    expect(vgsPane.querySelector('#status-vgs')?.textContent).toContain('980.0–1060.0 mV')
    expect(vgsPane.querySelector('#status-vgs')?.textContent).toContain(
      '1 failed/unbracketed point(s)',
    )
  })
})

describe('P06-08: registry/descriptor catalog dispatch and unknown-capability fallback', () => {
  it('renders selectable experiment groups from EXPERIMENT_CATALOG and falls back gracefully on newly registered backend capabilities', () => {
    expect(Object.keys(EXPERIMENT_CATALOG)).toHaveLength(15)
    const groups = getSelectableExperimentGroups()
    expect(groups.flatMap(g => g.entries).some(i => i.kind === 'current_decomp')).toBe(false)
    expect(groups.flatMap(g => g.entries).some(i => i.kind === 'spatial')).toBe(false)
    expect(getExperimentTreeLabelHTML('celiv_future_v2')).toBe('celiv_future_v2')

    const futureExpWs: Workspace = {
      version: 1,
      id: 'ws-future',
      name: 'Future Capability',
      layout: null,
      activeDeviceId: 'dev-1',
      activeExperimentId: 'exp-future',
      activeRunId: 'run-future',
      devices: [
        {
          id: 'dev-1',
          name: 'Device',
          tier: 'full',
          config: SAMPLE_CONFIG,
          experiments: [
            {
              id: 'exp-future',
              kind: 'celiv_future_v2' as unknown as Workspace['devices'][number]['experiments'][number]['kind'],
              params: {},
              runs: [
                {
                  id: 'run-future',
                  timestamp: 0,
                  durationMs: 10,
                  activePhysics: 'SOLARLAB · celiv_1d',
                  scientificQualification: true,
                  deviceSnapshot: SAMPLE_CONFIG,
                  result: {
                    kind: 'celiv_future_v2',
                    data: { peak_current_A_m2: 42.5 },
                  } as unknown as Run['result'],
                },
              ],
            },
          ],
        },
      ],
    }

    const expHost = document.createElement('div')
    const expHandle = mountExperimentPane(expHost, {
      getActiveDevice: () => ({ id: 'dev-1', config: SAMPLE_CONFIG }),
      onRunComplete: () => {},
    })
    expHandle.selectExperiment(
      'celiv_future_v2' as unknown as Workspace['devices'][number]['experiments'][number]['kind'],
    )
    expect(
      expHost.querySelector('[data-test="experiment-capability-fallback"]')?.textContent,
    ).toContain('celiv_future_v2')

    const plotHost = document.createElement('div')
    document.body.appendChild(plotHost)
    const plotHandle = mountMainPlotPane(plotHost)
    plotHandle.update(futureExpWs)
    expect(
      plotHost.querySelector('[data-test="plot-capability-fallback"]')?.textContent,
    ).toContain('42.5')
    plotHandle.dispose()
  })
})
