import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('../../job-stream', () => ({
  startJob: vi.fn(),
  streamJobEvents: vi.fn(),
}))

import { startJob, streamJobEvents } from '../../job-stream'
import type { DeviceConfig } from '../../types'
import { mountTPVPane } from './tpv-pane'

const config: DeviceConfig = { device: { Phi: 1e21 }, layers: [] }
let container: HTMLElement
let onRunComplete: ReturnType<typeof vi.fn>

function input(id: string): HTMLInputElement {
  return container.querySelector<HTMLInputElement>(`#${id}`)!
}

function runButton(): HTMLButtonElement {
  return container.querySelector<HTMLButtonElement>('#btn-tpv')!
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(startJob).mockResolvedValue('tpv-browser-test')
  container = document.createElement('div')
  document.body.replaceChildren(container)
  onRunComplete = vi.fn()
  mountTPVPane(container, {
    getActiveDevice: () => ({ id: 'reference', config }),
    onRunComplete,
  })
})

afterEach(() => document.body.replaceChildren())

describe('TPV numerical accuracy and failure recovery', () => {
  it('keeps automatic time stepping and existing accuracy defaults', () => {
    runButton().click()
    expect(startJob).toHaveBeenCalledWith('tpv', config, {
      N_grid: 80, delta_G_frac: 0.05, t_pulse: 1e-6,
      t_decay: 50e-6, n_points: 200,
      rtol: 1e-4, atol: 1e-6, voltage_atol: 1e-9,
    })
  })

  it('refines accuracy without changing the measurement history or gates', () => {
    input('tpv-rtol').value = '1e-6'
    input('tpv-atol').value = '1e-8'
    input('tpv-voltage-atol').value = '1e-11'
    input('tpv-max-step').value = '5e-8'
    runButton().click()
    expect(startJob).toHaveBeenCalledWith('tpv', config, {
      N_grid: 80, delta_G_frac: 0.05, t_pulse: 1e-6,
      t_decay: 50e-6, n_points: 200,
      rtol: 1e-6, atol: 1e-8, voltage_atol: 1e-11, max_step: 5e-8,
    })
  })

  it.each([
    ['tpv-rtol', '0'], ['tpv-atol', '-1'],
    ['tpv-voltage-atol', ''], ['tpv-max-step', '-1'],
  ])('rejects invalid %s before starting a numerical job', (id, value) => {
    input(id).value = value
    runButton().click()
    expect(startJob).not.toHaveBeenCalled()
    expect(container.querySelector('[role="status"]')!.textContent).toContain('finite and positive')
    expect(runButton().disabled).toBe(false)
  })

  it('shows the physical failure reason and permits a refined retry', async () => {
    runButton().click()
    await vi.waitFor(() => expect(streamJobEvents).toHaveBeenCalledOnce())
    const handlers = vi.mocked(streamJobEvents).mock.calls[0][1]
    handlers.onError('OpenCircuitError: integrated charge exceeds 1e-6 V\nTraceback (most recent call last):\ninternal file')
    handlers.onDone()
    expect(container.textContent).toContain('integrated charge exceeds 1e-6 V')
    expect(container.textContent).not.toContain('Traceback')
    expect(container.textContent).not.toContain('internal file')
    expect(onRunComplete).not.toHaveBeenCalled()
    expect(runButton().disabled).toBe(false)
    input('tpv-max-step').value = '5e-8'
    runButton().click()
    expect(startJob).toHaveBeenCalledTimes(2)
    expect(vi.mocked(startJob).mock.calls[1][2]).toMatchObject({ max_step: 5e-8 })
    expect(container.querySelector('[role="status"]')!.textContent).toBe('Starting job…')
  })
})
