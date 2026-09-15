import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  newPlot: vi.fn(),
  purge: vi.fn(),
  resize: vi.fn(),
}))

vi.mock('plotly.js-basic-dist-min', () => ({
  default: { newPlot: mocks.newPlot, purge: mocks.purge, Plots: { resize: mocks.resize } },
}))

import type { Workspace } from '../types'
import { mountMainPlotPane, type MainPlotHandle } from './main-plot-pane'

function workspace(): Workspace {
  const config = { device: { Phi: 1e21 }, layers: [] }
  return {
    version: 1, id: 'workspace', name: 'Resize regression', layout: null,
    activeDeviceId: 'device', activeExperimentId: 'jv', activeRunId: 'run',
    devices: [{
      id: 'device', name: 'Device', tier: 'full', config,
      experiments: [{
        id: 'jv', kind: 'jv', params: {},
        runs: [{
          id: 'run', timestamp: 0, durationMs: 1, activePhysics: 'Full', deviceSnapshot: config,
          result: {
            kind: 'jv',
            data: {
              V_fwd: [0, 1], J_fwd: [200, -10], V_rev: [1, 0], J_rev: [-10, 200],
              metrics_fwd: { V_oc: 0.9, J_sc: 200, FF: 0.7, PCE: 0.126 },
              metrics_rev: { V_oc: 0.9, J_sc: 200, FF: 0.7, PCE: 0.126 },
              hysteresis_index: 0,
            },
          },
        }],
      }],
    }],
  }
}

let container: HTMLElement
let handle: MainPlotHandle
let notifyResize: () => void
let width: number
let height: number
let observed: Element[]
let disconnect: ReturnType<typeof vi.fn>

beforeEach(() => {
  vi.clearAllMocks()
  width = 600
  height = 500
  observed = []
  disconnect = vi.fn()
  class TestResizeObserver {
    constructor(callback: ResizeObserverCallback) {
      notifyResize = () => callback([], this as unknown as ResizeObserver)
    }
    observe(target: Element): void { observed.push(target) }
    disconnect = disconnect
    unobserve = vi.fn()
  }
  vi.stubGlobal('ResizeObserver', TestResizeObserver)
  mocks.newPlot.mockImplementation((plot: HTMLElement) => {
    plot.classList.add('js-plotly-plot')
    Object.defineProperties(plot, {
      clientWidth: { configurable: true, get: () => Math.max(0, width - 24) },
      clientHeight: { configurable: true, get: () => Math.max(0, height - 48) },
    })
    return Promise.resolve()
  })
  mocks.purge.mockImplementation((plot: HTMLElement) => plot.classList.remove('js-plotly-plot'))
  mocks.resize.mockResolvedValue(undefined)
  container = document.createElement('div')
  Object.defineProperties(container, {
    clientWidth: { get: () => width },
    clientHeight: { get: () => height },
  })
  document.body.replaceChildren(container)
  handle = mountMainPlotPane(container)
})

afterEach(() => {
  handle.dispose()
  document.body.replaceChildren()
  vi.unstubAllGlobals()
})

describe('docked main plot sizing', () => {
  it('resizes an existing J–V figure when only the dock becomes narrower', () => {
    handle.update(workspace())
    const plot = container.querySelector<HTMLElement>('#jv1d-plot-inner')!

    expect(observed).toEqual([container])
    width = 280
    notifyResize()

    expect(mocks.resize).toHaveBeenCalledWith(plot)
    expect(mocks.newPlot).toHaveBeenCalledTimes(1)
  })

  it('waits for a hidden tab to become visible and skips a detached pane', () => {
    handle.update(workspace())
    width = 0
    height = 0
    notifyResize()
    expect(mocks.resize).not.toHaveBeenCalled()

    width = 320
    height = 400
    notifyResize()
    expect(mocks.resize).toHaveBeenCalledTimes(1)

    container.remove()
    notifyResize()
    expect(mocks.resize).toHaveBeenCalledTimes(1)
  })

  it('purges the old figure and resizes only the replacement result', () => {
    handle.update(workspace())
    const oldPlot = container.querySelector('.js-plotly-plot')!
    handle.update(workspace())
    const newPlot = container.querySelector('.js-plotly-plot')!

    expect(newPlot).not.toBe(oldPlot)
    expect(mocks.purge).toHaveBeenCalledWith(oldPlot)
    notifyResize()
    expect(mocks.resize).toHaveBeenCalledTimes(1)
    expect(mocks.resize).toHaveBeenCalledWith(newPlot)
  })

  it('clears the active figure without retaining it in the resize callback', () => {
    handle.update(workspace())
    const oldPlot = container.querySelector('.js-plotly-plot')!
    handle.update({ ...workspace(), activeRunId: null })
    notifyResize()

    expect(container.textContent).toContain('No result selected')
    expect(mocks.purge).toHaveBeenCalledWith(oldPlot)
    expect(mocks.resize).not.toHaveBeenCalled()
  })

  it('disconnects and purges when Golden Layout destroys the pane', () => {
    handle.update(workspace())
    const plot = container.querySelector('.js-plotly-plot')!
    handle.dispose()
    handle.dispose()
    notifyResize()
    handle.update(workspace())

    expect(disconnect).toHaveBeenCalledTimes(1)
    expect(mocks.purge).toHaveBeenCalledWith(plot)
    expect(mocks.resize).not.toHaveBeenCalled()
    expect(mocks.newPlot).toHaveBeenCalledTimes(1)
  })

  it('handles a tab being hidden during an asynchronous Plotly resize', async () => {
    handle.update(workspace())
    mocks.resize.mockRejectedValueOnce(new Error('Resize must be passed a displayed plot div element.'))
    notifyResize()
    await Promise.resolve()
    expect(mocks.resize).toHaveBeenCalledTimes(1)
  })
})
