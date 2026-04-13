import { describe, it, expect, beforeEach, vi } from 'vitest'
import { renderTreeHTML, attachTreeHandlers } from './tree'
import { createEmptyWorkspace, addDevice, addExperiment, addRun } from './state'
import type { Device, Experiment, Run } from './types'

function makeDevice(id: string, name: string, tier: 'legacy' | 'fast' | 'full' = 'full'): Device {
  return {
    id,
    name,
    tier,
    config: { device: { V_bi: 1.1, Phi: 1.4e21 }, layers: [] },
    experiments: [],
  }
}

function makeExperiment(id: string, kind: 'jv' | 'impedance' | 'degradation' = 'jv'): Experiment {
  return { id, kind, params: {}, runs: [] }
}

function makeRun(id: string): Run {
  return {
    id,
    timestamp: new Date(2026, 0, 1, 14, 30).getTime(),
    result: { placeholder: true },
    activePhysics: 'FULL',
    durationMs: 100,
    deviceSnapshot: { device: { V_bi: 1.1, Phi: 1.4e21 }, layers: [] },
  }
}

describe('renderTreeHTML', () => {
  it('renders the Devices folder header even when empty', () => {
    const ws = createEmptyWorkspace('W')
    const html = renderTreeHTML(ws)
    expect(html).toContain('Devices')
    expect(html).toContain('Results')
  })

  it('renders each device with its tier badge', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1', 'MAPbI3', 'full'))
    ws = addDevice(ws, makeDevice('d2', 'CIGS', 'legacy'))
    const html = renderTreeHTML(ws)
    expect(html).toContain('MAPbI3')
    expect(html).toContain('CIGS')
    expect(html).toContain('FULL')
    expect(html).toContain('LEGACY')
  })

  it('marks the active device with an active class', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1', 'Alpha'))
    const html = renderTreeHTML(ws)
    expect(html).toContain('data-device-id="d1"')
    expect(html).toContain('tree-node-active')
  })

  it('escapes device names with HTML special characters', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1', '<script>x</script>'))
    const html = renderTreeHTML(ws)
    expect(html).not.toContain('<script>x</script>')
    expect(html).toContain('&lt;script&gt;')
  })
})

describe('attachTreeHandlers', () => {
  let container: HTMLDivElement

  beforeEach(() => {
    container = document.createElement('div')
    document.body.appendChild(container)
  })

  it('fires onSelectDevice when a device node is clicked', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1', 'Alpha'))
    container.innerHTML = renderTreeHTML(ws)
    const received: string[] = []
    attachTreeHandlers(container, { onSelectDevice: (id) => received.push(id) })
    const node = container.querySelector<HTMLElement>('[data-device-id="d1"]')!
    node.click()
    expect(received).toEqual(['d1'])
  })
})

describe('hierarchical rendering', () => {
  it('renders experiment nodes under their device', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1', 'Alpha'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1', 'jv'))
    const html = renderTreeHTML(ws)
    expect(html).toContain('data-device-id="d1"')
    expect(html).toContain('data-experiment-id="e1"')
    expect(html).toMatch(/J.V/i)
  })

  it('renders run nodes under their experiment with timestamp', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1', 'Alpha'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1', 'jv'))
    ws = addRun(ws, 'd1', 'e1', makeRun('r1'))
    const html = renderTreeHTML(ws)
    expect(html).toContain('data-run-id="r1"')
  })

  it('escapes malicious device names in experiment context', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1', '<img src=x onerror=alert(1)>'))
    const html = renderTreeHTML(ws)
    expect(html).not.toContain('<img')
    expect(html).toContain('&lt;img')
  })
})

describe('attachTreeHandlers dispatch', () => {
  it('dispatches onSelectExperiment when an experiment node is clicked', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1', 'Alpha'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1', 'jv'))
    const el = document.createElement('div')
    el.innerHTML = renderTreeHTML(ws)
    const spy = vi.fn()
    attachTreeHandlers(el, {
      onSelectDevice: () => {},
      onSelectExperiment: spy,
      onSelectRun: () => {},
    })
    el.querySelector<HTMLElement>('[data-experiment-id="e1"]')!.click()
    expect(spy).toHaveBeenCalledWith('d1', 'e1')
  })

  it('dispatches onSelectRun when a run node is clicked', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1', 'Alpha'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1', 'jv'))
    ws = addRun(ws, 'd1', 'e1', makeRun('r1'))
    const el = document.createElement('div')
    el.innerHTML = renderTreeHTML(ws)
    const spy = vi.fn()
    attachTreeHandlers(el, {
      onSelectDevice: () => {},
      onSelectExperiment: () => {},
      onSelectRun: spy,
    })
    el.querySelector<HTMLElement>('[data-run-id="r1"]')!.click()
    expect(spy).toHaveBeenCalledWith('d1', 'e1', 'r1')
  })
})
