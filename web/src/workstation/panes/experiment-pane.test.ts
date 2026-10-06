import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('../../job-stream', () => ({ startJob: vi.fn(), streamJobEvents: vi.fn() }))

import { startJob } from '../../job-stream'
import type { DeviceConfig } from '../../types'
import { mountExperimentPane, type ExperimentPaneHandle } from './experiment-pane'

const config: DeviceConfig = { device: { Phi: 1e21 }, layers: [] }
let container: HTMLElement
let handle: ExperimentPaneHandle

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(startJob).mockResolvedValue('selected-experiment')
  container = document.createElement('div')
  document.body.replaceChildren(container)
  handle = mountExperimentPane(container, {
    getActiveDevice: () => ({ id: 'reference', config }),
    onRunComplete: () => {},
  })
})

afterEach(() => document.body.replaceChildren())

describe('workspace experiment selection', () => {
  it('opens TPV and preserves its numerical inputs when selected again', () => {
    handle.selectExperiment('tpv')
    const tolerance = container.querySelector<HTMLInputElement>('#tpv-rtol')!
    tolerance.value = '1e-7'
    handle.selectExperiment('tpv')
    expect(container.querySelector('#tpv-rtol')).toBe(tolerance)
    container.querySelector<HTMLButtonElement>('#btn-tpv')!.click()
    expect(startJob).toHaveBeenCalledWith('tpv', config, expect.objectContaining({ rtol: 1e-7 }))
  })

  it('opens the two-dimensional form and dispatches the selected experiment', () => {
    handle.selectExperiment('jv_2d')
    container.querySelector<HTMLButtonElement>('#btn-jv2d')!.click()
    expect(startJob).toHaveBeenCalledWith('jv_2d', config, expect.objectContaining({ Nx: 10 }))
  })

  it.each(['current_decomp', 'spatial'] as const)('restores the %s J-V output mode', kind => {
    handle.selectExperiment(kind)
    const decomp = container.querySelector<HTMLInputElement>('#jvp-decomp')!
    const spatial = container.querySelector<HTMLInputElement>('#jvp-spatial')!
    expect(decomp.checked).toBe(kind === 'current_decomp')
    expect(spatial.checked).toBe(kind === 'spatial')
    expect(decomp.disabled).toBe(false)
    expect(spatial.disabled).toBe(false)
    handle.selectExperiment('jv')
    expect(decomp.checked).toBe(false)
    expect(spatial.checked).toBe(false)
  })
})
