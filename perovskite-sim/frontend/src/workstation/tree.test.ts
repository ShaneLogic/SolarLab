import { describe, it, expect, beforeEach } from 'vitest'
import { renderTreeHTML, attachTreeHandlers } from './tree'
import { createEmptyWorkspace, addDevice } from './state'
import type { Device } from './types'

function makeDevice(id: string, name: string, tier: 'legacy' | 'fast' | 'full' = 'full'): Device {
  return {
    id,
    name,
    tier,
    config: { device: { V_bi: 1.1, Phi: 1.4e21 }, layers: [] },
    experiments: [],
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
