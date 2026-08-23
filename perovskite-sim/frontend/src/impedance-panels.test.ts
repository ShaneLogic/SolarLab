import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  startJob: vi.fn(),
  streamJobEvents: vi.fn(),
  mountDevicePanel: vi.fn(),
}))

vi.mock('./job-stream', () => ({
  startJob: mocks.startJob,
  streamJobEvents: mocks.streamJobEvents,
}))

vi.mock('./device-panel', () => ({
  mountDevicePanel: mocks.mountDevicePanel,
}))

vi.mock('plotly.js-basic-dist-min', () => ({
  default: {
    newPlot: vi.fn(),
    purge: vi.fn(),
  },
}))

import { mountImpedancePanel } from './panels/impedance'
import { mountImpedancePane } from './workstation/panes/impedance-pane'
import type { DeviceConfig, ISResult, JobStreamHandlers } from './types'

const device: DeviceConfig = {
  device: { Phi: 0 },
  layers: [],
}

let container: HTMLDivElement

beforeEach(() => {
  document.body.replaceChildren()
  container = document.createElement('div')
  document.body.appendChild(container)
  mocks.startJob.mockReset()
  mocks.startJob.mockResolvedValue('job-1')
  mocks.streamJobEvents.mockReset()
  mocks.mountDevicePanel.mockReset()
  mocks.mountDevicePanel.mockResolvedValue({ getConfig: () => device })
})

afterEach(() => {
  document.body.replaceChildren()
})

describe('impedance points-per-cycle controls', () => {
  it('legacy panel defaults to 40 and forwards the selected value', async () => {
    await mountImpedancePanel(container)
    const points = document.getElementById('is-ppc') as HTMLInputElement
    expect(points.value).toBe('40')
    points.value = '64'

    ;(document.getElementById('btn-is') as HTMLButtonElement).click()
    await Promise.resolve()

    expect(mocks.startJob).toHaveBeenCalledWith(
      'impedance',
      device,
      expect.objectContaining({ points_per_cycle: 64 }),
    )
  })

  it('workstation pane forwards the default 40 points per cycle', async () => {
    mountImpedancePane(container, {
      getActiveDevice: () => ({ id: 'device-1', config: device }),
      onRunComplete: vi.fn(),
    })
    const points = document.getElementById('imp-ppc') as HTMLInputElement
    expect(points.value).toBe('40')

    ;(document.getElementById('btn-imp') as HTMLButtonElement).click()
    await Promise.resolve()

    expect(mocks.startJob).toHaveBeenCalledWith(
      'impedance',
      device,
      expect.objectContaining({ points_per_cycle: 40 }),
    )
  })

  it('legacy panel marks a result without certificate blocks as unclassified', async () => {
    await mountImpedancePanel(container)
    ;(document.getElementById('btn-is') as HTMLButtonElement).click()
    await vi.waitFor(() => {
      expect(mocks.streamJobEvents).toHaveBeenCalledTimes(1)
    })

    const handlers = mocks.streamJobEvents.mock.calls[0][1] as JobStreamHandlers<ISResult>
    handlers.onResult({
      frequencies: [1e3],
      Z_real: [1],
      Z_imag: [-1],
    })

    const warning = container.querySelector<HTMLElement>(
      '[data-test="impedance-evidence-warning"]',
    )
    expect(warning?.textContent).toContain('Legacy impedance result')
    expect(warning?.textContent).toContain('unclassified')
  })

  it('marks a pre-envelope frequency assessment as legacy unclassified', async () => {
    await mountImpedancePanel(container)
    ;(document.getElementById('btn-is') as HTMLButtonElement).click()
    await vi.waitFor(() => {
      expect(mocks.streamJobEvents).toHaveBeenCalledTimes(1)
    })

    const handlers = mocks.streamJobEvents.mock.calls[0][1] as JobStreamHandlers<ISResult>
    handlers.onResult({
      frequencies: [1e3],
      Z_real: [1],
      Z_imag: [-1],
      protocol: {
        method: 'transient_ion_aware',
        V_dc: 0,
        delta_V: 0.01,
        illuminated: false,
        dc_settle_time: 1e-3,
        n_cycles: 5,
        n_extract: 2,
        points_per_cycle: 40,
      },
      operating_point: {
        certified: true,
        reasons: [],
      } as unknown as NonNullable<ISResult['operating_point']>,
      frequency_window: {
        f_min_Hz: 1e3,
        f_max_Hz: 1e3,
        has_mobile_ions: true,
        characteristic_frequency_bracketed: true,
        ionic_branch_covered: true,
        ionic_timescales: [],
        warnings: [],
      },
      grid_assessment: {
        certified: true,
        warnings: [],
      } as unknown as NonNullable<ISResult['grid_assessment']>,
    })

    expect(
      container.querySelector<HTMLElement>(
        '[data-test="impedance-evidence-warning"]',
      )?.textContent,
    ).toContain('Legacy impedance result')
  })
})
