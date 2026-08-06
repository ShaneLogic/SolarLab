/**
 * vitest — J–V pane solver/iface gating.
 *
 * Interface-plane states only take effect in the steady-state Newton driver;
 * the transient sweep ignores the iface_states param. The pane must make the
 * no-op combo (iface ticked, transient) impossible by gating the "Interface-
 * plane states" checkbox on the solver selection.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { mountJVPane } from './jv-pane'
import type { DeviceConfig } from '../../types'

const opts = { getActiveDevice: () => null, onRunComplete: () => {} }

const csiConfig: DeviceConfig = {
  simulation_hints: { min_N_grid: 200 },
  electrical_grid: {
    interval_weights: { n_emitter: 1, p_base: 4 },
    alphas: { n_emitter: 2, p_base: 3 },
  },
  device: {
    V_bi: 0.8928964399850017,
    Phi: 2.7e21,
    jv_solver_policy: 'cancellation_safe_qf_required',
  },
  layers: [],
}

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
    solver: document.getElementById('jvp-solver') as HTMLSelectElement,
    iface: document.getElementById('jvp-iface') as HTMLInputElement,
  }
}

describe('J–V pane interface-plane-states gating', () => {
  it('iface checkbox starts disabled (steady-state off by default)', () => {
    mountJVPane(container, opts)
    expect(boxes().iface.disabled).toBe(true)
  })

  it('selecting steady-state enables the iface checkbox', () => {
    mountJVPane(container, opts)
    const { solver, iface } = boxes()
    solver.value = 'steady_state'
    solver.dispatchEvent(new Event('change'))
    expect(iface.disabled).toBe(false)
  })

  it('selecting quasi-Fermi disables and clears the iface checkbox', () => {
    mountJVPane(container, opts)
    const { solver, iface } = boxes()
    solver.value = 'steady_state'
    solver.dispatchEvent(new Event('change'))
    iface.checked = true
    solver.value = 'quasi_fermi'
    solver.dispatchEvent(new Event('change'))
    expect(iface.disabled).toBe(true)
    expect(iface.checked).toBe(false)
  })

  it('exposes the cancellation-safe quasi-Fermi solver explicitly', () => {
    mountJVPane(container, opts)
    const options = Array.from(boxes().solver.options)
    expect(options.map(option => option.value)).toEqual([
      'transient',
      'steady_state',
      'quasi_fermi',
    ])
  })

  it('rejects a c-Si grid below the preserved config minimum', () => {
    mountJVPane(container, {
      getActiveDevice: () => ({ id: 'csi', config: csiConfig }),
      onRunComplete: () => {},
    })
    const { solver } = boxes()
    solver.value = 'quasi_fermi'
    solver.dispatchEvent(new Event('change'))
    ;(document.getElementById('jvp-N') as HTMLInputElement).value = '100'
    ;(document.getElementById('btn-jvp') as HTMLButtonElement).click()
    expect(document.getElementById('status-jvp')?.textContent).toContain(
      'N_grid >= 200',
    )
  })
})
