/**
 * vitest — J–V pane solver/iface gating.
 *
 * Interface-plane states only take effect in the steady-state Newton driver;
 * the transient sweep ignores the iface_states param. The pane must make the
 * no-op combo (iface ticked, transient) impossible by gating the "Interface-
 * plane states" checkbox on the "Steady-state solver" toggle.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { mountJVPane } from './jv-pane'

const opts = { getActiveDevice: () => null, onRunComplete: () => {} }

let container: HTMLElement

beforeEach(() => {
  document.body.replaceChildren()
  container = document.createElement('div')
  document.body.appendChild(container)
})

afterEach(() => {
  document.body.replaceChildren()
})

function boxes() {
  return {
    ss: document.getElementById('jvp-ss') as HTMLInputElement,
    iface: document.getElementById('jvp-iface') as HTMLInputElement,
  }
}

describe('J–V pane interface-plane-states gating', () => {
  it('iface checkbox starts disabled (steady-state off by default)', () => {
    mountJVPane(container, opts)
    expect(boxes().iface.disabled).toBe(true)
  })

  it('checking steady-state enables the iface checkbox', () => {
    mountJVPane(container, opts)
    const { ss, iface } = boxes()
    ss.checked = true
    ss.dispatchEvent(new Event('change'))
    expect(iface.disabled).toBe(false)
  })

  it('unchecking steady-state disables AND clears the iface checkbox', () => {
    mountJVPane(container, opts)
    const { ss, iface } = boxes()
    ss.checked = true
    ss.dispatchEvent(new Event('change'))
    iface.checked = true
    ss.checked = false
    ss.dispatchEvent(new Event('change'))
    expect(iface.disabled).toBe(true)
    expect(iface.checked).toBe(false)
  })
})
