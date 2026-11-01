import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  config: null as unknown,
  changeListener: null as ((config: unknown) => void) | null,
  startJob: vi.fn(),
  streamJobEvents: vi.fn(),
  mountDevicePanel: vi.fn(),
}))

vi.mock('plotly.js-basic-dist-min', () => ({
  default: { newPlot: vi.fn(), purge: vi.fn() },
  newPlot: vi.fn(),
  purge: vi.fn(),
}))

vi.mock('../device-panel', () => ({
  mountDevicePanel: mocks.mountDevicePanel,
}))

vi.mock('../job-stream', () => ({
  startJob: mocks.startJob,
  streamJobEvents: mocks.streamJobEvents,
}))

import Plotly from 'plotly.js-basic-dist-min'
import { mountJVPanel, renderJVResults } from './jv'
import type { DeviceConfig, InterfaceChargeJVEvidence, JVResult } from '../types'

const chargedConfig: DeviceConfig = {
  device: {
    Phi: 1e18,
    interface_charge_closure: 'equilibrium_referenced',
    interface_charge_rebaseline_acknowledged: true,
  },
  layers: [],
}

function result(): JVResult {
  const metrics = {
    V_oc: 0.0788, J_sc: 0.0133, FF: 0.55, PCE: 5.8e-7, voc_bracketed: true,
  }
  const evidence = {
    model: 'interface-charge-jv-evidence-v1',
    protocol_sha256: 'a'.repeat(64),
    dark_state_sha256: 'd'.repeat(64),
    grid_sha256: 'b'.repeat(64),
    stack_sha256: 'c'.repeat(64),
    points: [{}, {}, {}],
    continuation_bridge_count: 0,
    minimum_occupancy: 0.7139,
    maximum_occupancy: 0.7194,
    maximum_absolute_sheet_charge_C_m2: 8.73e-5,
    maximum_absolute_trace_potential_shift_V: 2.41e-4,
    maximum_normalized_gauss_residual: 3.16e-16,
    maximum_normalized_cell_residual: 3.12e-7,
    maximum_continuity_bound_A_m2: 3.13e-7,
    maximum_contact_fermi_level_span_eV: 1e-5,
    maximum_scaled_local_jacobian_condition: 1.19e4,
    tolerance_factor: 1,
    protocol: { charge_law: '-q*N_t*(f-f_eq)' },
  } as unknown as InterfaceChargeJVEvidence
  return {
    V_fwd: [0, 0.05, 0.1], J_fwd: [0.0133, 0.0087, -0.0116],
    V_rev: [0, 0.05, 0.1], J_rev: [0.0133, 0.0087, -0.0116],
    metrics_fwd: metrics, metrics_rev: metrics, hysteresis_index: 0,
    interface_charge_evidence: evidence,
  }
}

beforeEach(() => {
  document.body.replaceChildren()
  mocks.config = chargedConfig
  mocks.changeListener = null
  mocks.startJob.mockReset()
  mocks.startJob.mockResolvedValue('charged-jv-job')
  mocks.streamJobEvents.mockReset()
  mocks.mountDevicePanel.mockReset()
  mocks.mountDevicePanel.mockResolvedValue({
    getConfig: () => mocks.config,
    onChange: (listener: (config: unknown) => void) => {
      mocks.changeListener = listener
    },
  })
  vi.mocked(Plotly.newPlot).mockClear()
})

describe('simple J-V panel charged-interface closure', () => {
  it('locks the execution controls and submits the exact charged slice', async () => {
    const root = document.createElement('div')
    document.body.appendChild(root)
    await mountJVPanel(root)

    const solver = root.querySelector<HTMLSelectElement>('#jv-solver')!
    const rate = root.querySelector<HTMLInputElement>('#jv-rate')!
    const dark = root.querySelector<HTMLInputElement>('#jv-dark')!
    expect(solver.value).toBe('quasi_fermi')
    expect(solver.disabled).toBe(true)
    expect(rate.value).toBe('0')
    expect(rate.disabled).toBe(true)
    expect(dark.checked).toBe(false)
    expect(dark.disabled).toBe(true)

    ;(root.querySelector('#jv-vmax') as HTMLInputElement).value = '0.1'
    ;(root.querySelector('#jv-np') as HTMLInputElement).value = '5'
    ;(root.querySelector('#btn-jv') as HTMLButtonElement).click()
    await vi.waitFor(() => expect(mocks.startJob).toHaveBeenCalledOnce())

    expect(mocks.startJob.mock.calls[0]).toEqual([
      'jv',
      chargedConfig,
      expect.objectContaining({
        n_points: 5,
        v_rate: 0,
        V_max: 0.1,
        illuminated: true,
        solver: 'quasi_fermi',
        iface_states: false,
        interface_boundary: true,
        interface_transport_model: 'fermi_dirac_richardson',
      }),
    ])
  })

  it('renders one curve, one metric block, and the charged evidence strip', () => {
    const container = document.createElement('div')
    document.body.appendChild(container)
    renderJVResults(container, result())

    expect(container.textContent).toContain('Certified zero-scan QF/DC')
    expect(container.textContent).not.toContain('Reverse')
    expect(container.textContent).not.toContain('Hysteresis Index')
    const summary = container.querySelector<HTMLElement>(
      '[data-test="interface-charge-jv-evidence-summary"]',
    )!
    expect(summary.textContent).toContain('3 requested points + 0 bridges')
    expect(summary.textContent).toContain('Max Gauss 3.160e-16')
    const traces = vi.mocked(Plotly.newPlot).mock.calls[0][1] as Array<{ name: string }>
    expect(traces).toHaveLength(1)
    expect(traces[0].name).toBe('Charged QF/DC')
  })
})
