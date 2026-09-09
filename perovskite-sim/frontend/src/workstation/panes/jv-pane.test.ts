/**
 * vitest — J–V pane solver/iface gating.
 *
 * Interface-plane states only take effect in the steady-state Newton driver;
 * the transient sweep ignores the iface_states param. The pane must make the
 * no-op combo (iface ticked, transient) impossible by gating the "Interface-
 * plane states" checkbox on the solver selection.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import figureReferenceJSON from '../../../../tests/fixtures/IonDiffusivityFigureReference.json?raw'

vi.mock('../../job-stream', () => ({
  startJob: vi.fn(),
  streamJobEvents: vi.fn(),
}))

import { mountJVPane } from './jv-pane'
import type { DeviceConfig } from '../../types'
import { startJob } from '../../job-stream'
import { mountExperimentPane } from './experiment-pane'

const figureReference = JSON.parse(figureReferenceJSON)

function figureConfig(): DeviceConfig {
  const config = structuredClone(figureReference.request.device) as DeviceConfig
  const { N_grid, n_points, v_rate, V_max, waveform, waveform_controls } = figureReference.request.params
  config.simulation_hints = {
    jv_sweep: { N_grid, n_points, v_rate, V_max, waveform, waveform_controls },
  }
  return config
}

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

const chargedInterfaceConfig: DeviceConfig = {
  device: {
    Phi: 1e18,
    interface_charge_closure: 'equilibrium_referenced',
    interface_charge_rebaseline_acknowledged: true,
  },
  layers: [],
}

function explicitDefectConfig(transition: 'neutral' | 'acceptor'): DeviceConfig {
  return {
    device: { Phi: 2e21 },
    layers: [{
      name: 'absorber', role: 'absorber', thickness: 5e-7, eps_r: 20,
      mu_n: 1e-4, mu_p: 1e-4, ni: 1e15, N_D: 0, N_A: 0,
      D_ion: 0, P_lim: 0, P0: 0, tau_n: 1e-6, tau_p: 1e-6,
      n1: 1e15, p1: 1e15, B_rad: 0, C_n: 0, C_p: 0, alpha: 0,
      defect_schema_version: 'solarlab-explicit-bulk-defects-v1',
      defect_model: 'explicit_quasi_steady',
      bulk_defects: [{
        name: 'D1',
        distribution: {
          kind: 'single_level', normalization: 'integrated_total',
          total_density_m3: 1e21, center_eV_above_vb: 0.7,
        },
        charge_transition: transition,
        neutral_reference: transition === 'neutral' ? 'all_occupancies' : 'empty',
        kinetics: {
          sigma_n_m2: 1e-19, sigma_p_m2: 1e-19,
          thermal_velocity_n_m_s: 1e5, thermal_velocity_p_m_s: 1e5,
        },
        degeneracy: 1,
      }],
    }],
  }
}

let container: HTMLElement

beforeEach(() => {
  document.body.replaceChildren()
  container = document.createElement('div')
  document.body.appendChild(container)
  vi.mocked(startJob).mockReset()
  vi.mocked(startJob).mockResolvedValue('charged-jv-job')
})

afterEach(() => {
  document.body.replaceChildren()
})

function boxes() {
  return {
    solver: document.getElementById('jvp-solver') as HTMLSelectElement,
    iface: document.getElementById('jvp-iface') as HTMLInputElement,
    interfaceBoundary: document.getElementById(
      'jvp-interface-boundary',
    ) as HTMLInputElement,
    interfaceTransport: document.getElementById(
      'jvp-interface-transport',
    ) as HTMLSelectElement,
  }
}

describe('J–V pane interface-plane-states gating', () => {
  it('iface checkbox starts disabled (steady-state off by default)', () => {
    mountJVPane(container, opts)
    expect(boxes().iface.disabled).toBe(true)
    expect(boxes().interfaceBoundary.disabled).toBe(true)
    expect(boxes().interfaceTransport.disabled).toBe(true)
  })

  it('selecting steady-state enables the iface checkbox', () => {
    mountJVPane(container, opts)
    const { solver, iface } = boxes()
    solver.value = 'steady_state'
    solver.dispatchEvent(new Event('change'))
    expect(iface.disabled).toBe(false)
    expect(boxes().interfaceBoundary.disabled).toBe(true)
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
    expect(boxes().interfaceBoundary.disabled).toBe(false)
    expect(boxes().interfaceTransport.disabled).toBe(true)
  })

  it('physical interface response is enabled only for quasi-Fermi', () => {
    mountJVPane(container, opts)
    const { solver, interfaceBoundary } = boxes()
    solver.value = 'quasi_fermi'
    solver.dispatchEvent(new Event('change'))
    interfaceBoundary.checked = true
    interfaceBoundary.dispatchEvent(new Event('change'))
    expect(boxes().interfaceTransport.disabled).toBe(false)
    boxes().interfaceTransport.value = 'scaps_thermionic'
    solver.value = 'transient'
    solver.dispatchEvent(new Event('change'))
    expect(interfaceBoundary.disabled).toBe(true)
    expect(interfaceBoundary.checked).toBe(false)
    expect(boxes().interfaceTransport.disabled).toBe(true)
    expect(boxes().interfaceTransport.value).toBe('fermi_richardson')
  })

  it('lists all supported interface transport closures', () => {
    mountJVPane(container, opts)
    expect(Array.from(boxes().interfaceTransport.options).map(o => o.value)).toEqual([
      'fermi_richardson',
      'fermi_dirac_richardson',
      'scaps_thermionic',
      'scaps_thermal_velocity',
    ])
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

  it('locks charged explicit defects to the QF solver when the control is focused', () => {
    const charged = explicitDefectConfig('acceptor')
    mountJVPane(container, {
      getActiveDevice: () => ({ id: 'charged', config: charged }),
      onRunComplete: () => {},
    })
    boxes().solver.focus()
    expect(boxes().solver.value).toBe('quasi_fermi')
    expect(boxes().solver.disabled).toBe(true)
    expect(boxes().interfaceBoundary.disabled).toBe(false)
  })

  it('keeps the QF lock for charged distributed v3 defects', () => {
    const charged = explicitDefectConfig('acceptor')
    const layer = charged.layers[0]
    layer.defect_schema_version = 'solarlab-explicit-bulk-defects-v3'
    layer.bulk_defects![0].distribution.energy_reference = 'above_valence_band'
    layer.bulk_defects![0].spatial_profile = {
      coordinate: 'normalized_layer_coordinate',
      interpolation: 'piecewise_linear',
      density_normalization: 'layer_average_unity',
      knots: [
        { position_fraction: 0, density_multiplier: 0.8 },
        { position_fraction: 1, density_multiplier: 1.2 },
      ],
    }
    mountJVPane(container, {
      getActiveDevice: () => ({ id: 'charged-v3', config: charged }),
      onRunComplete: () => {},
    })

    boxes().solver.focus()
    expect(boxes().solver.value).toBe('quasi_fermi')
    expect(boxes().solver.disabled).toBe(true)
  })

  it('keeps the transient solver available for neutral explicit defects', () => {
    const neutral = explicitDefectConfig('neutral')
    mountJVPane(container, {
      getActiveDevice: () => ({ id: 'neutral', config: neutral }),
      onRunComplete: () => {},
    })
    boxes().solver.focus()
    expect(boxes().solver.value).toBe('transient')
    expect(boxes().solver.disabled).toBe(false)
  })

  it('locks charged interface J-V controls and submits the exact API slice', async () => {
    mountJVPane(container, {
      getActiveDevice: () => ({ id: 'charged-interface', config: chargedInterfaceConfig }),
      onRunComplete: () => {},
    })

    expect(boxes().solver.value).toBe('quasi_fermi')
    expect(boxes().solver.disabled).toBe(true)
    expect(boxes().iface.disabled).toBe(true)
    expect(boxes().interfaceBoundary.checked).toBe(true)
    expect(boxes().interfaceBoundary.disabled).toBe(true)
    expect(boxes().interfaceTransport.value).toBe('fermi_dirac_richardson')
    expect(boxes().interfaceTransport.disabled).toBe(true)
    expect((document.getElementById('jvp-rate') as HTMLInputElement).value).toBe('0')
    expect((document.getElementById('jvp-rate') as HTMLInputElement).disabled).toBe(true)
    expect((document.getElementById('jvp-decomp') as HTMLInputElement).disabled).toBe(true)
    expect((document.getElementById('jvp-spatial') as HTMLInputElement).disabled).toBe(true)

    ;(document.getElementById('jvp-vmax') as HTMLInputElement).value = '0.1'
    ;(document.getElementById('jvp-np') as HTMLInputElement).value = '5'
    ;(document.getElementById('btn-jvp') as HTMLButtonElement).click()

    await vi.waitFor(() => expect(startJob).toHaveBeenCalledOnce())
    expect(vi.mocked(startJob).mock.calls[0]).toEqual([
      'jv',
      chargedInterfaceConfig,
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

describe('figure-reference J-V inputs', () => {
  it('submits the captured figure history from its initial device preset', async () => {
    const config = figureConfig()
    mountJVPane(container, {
      getActiveDevice: () => ({ id: 'figure', config }), onRunComplete: () => {},
    })
    expect(boxes().solver.value).toBe('transient')
    expect((document.getElementById('jvp-waveform-mode') as HTMLSelectElement).value).toBe('continuous')
    expect((document.getElementById('jvp-waveform-turnaround') as HTMLInputElement).value).toBe('3')
    expect((document.getElementById('jvp-waveform-atol') as HTMLInputElement).value).toBe('1')
    document.getElementById('btn-jvp')!.click()
    await vi.waitFor(() => expect(startJob).toHaveBeenCalledOnce())
    expect(vi.mocked(startJob).mock.calls[0]).toEqual(['jv', config, figureReference.request.params])
  })

  it.each([
    ['D_ion', 0], ['D_ion', 5.17e-19], ['D_ion', 5.17e-18], ['P0', 0], ['P0', 2e24],
  ] as const)('keeps edited absorber %s=%s and the figure history on run', async (field, value) => {
    const config = figureConfig()
    const pane = mountJVPane(container, {
      getActiveDevice: () => ({ id: 'figure', config }), onRunComplete: () => {},
    })
    const absorber = config.layers.find(layer => layer.role === 'absorber')!
    absorber[field] = value
    pane.updateDevice()
    document.getElementById('btn-jvp')!.click()
    await vi.waitFor(() => expect(startJob).toHaveBeenCalledOnce())
    const [, sent, params] = vi.mocked(startJob).mock.calls[0]
    expect((sent as DeviceConfig).layers.find(layer => layer.role === 'absorber')![field]).toBe(value)
    expect(params).toEqual(figureReference.request.params)
  })

  it('updates a mounted experiment after preset selection and preserves later scan edits', async () => {
    let config: DeviceConfig = { device: { Phi: 2e21 }, layers: [] }
    const pane = mountExperimentPane(container, {
      getActiveDevice: () => ({ id: 'device', config }), onRunComplete: () => {},
    })
    config = figureConfig()
    pane.updateDevice()
    const rate = document.getElementById('jvp-rate') as HTMLInputElement
    expect(rate.value).toBe('0.04')
    rate.value = '0.08'
    config.layers[1].D_ion = 5.17e-19
    pane.updateDevice()
    expect(rate.value).toBe('0.08')
    document.getElementById('btn-jvp')!.click()
    await vi.waitFor(() => expect(startJob).toHaveBeenCalledOnce())
    expect(vi.mocked(startJob).mock.calls[0][2]).toEqual({
      ...figureReference.request.params, v_rate: 0.08,
    })
  })

  it('restores ordinary scan inputs when leaving the figure preset', () => {
    let config: DeviceConfig = { device: { Phi: 2e21 }, layers: [] }
    const pane = mountJVPane(container, {
      getActiveDevice: () => ({ id: 'device', config }), onRunComplete: () => {},
    })
    const grid = document.getElementById('jvp-N') as HTMLInputElement
    grid.value = '80'
    config = figureConfig()
    pane.updateDevice()
    expect(grid.value).toBe('60')
    config = { device: { Phi: 2e21 }, layers: [] }
    pane.updateDevice()
    expect(grid.value).toBe('80')
    expect((document.getElementById('jvp-rate') as HTMLInputElement).value).toBe('1')
    expect((document.getElementById('jvp-waveform-mode') as HTMLSelectElement).value).toBe('standard')
  })

  it('restores the same figure protocol only on an explicit preset reset', () => {
    const config = figureConfig()
    const pane = mountExperimentPane(container, {
      getActiveDevice: () => ({ id: 'figure', config }), onRunComplete: () => {},
    })
    const rate = document.getElementById('jvp-rate') as HTMLInputElement
    const hold = document.getElementById('jvp-waveform-turnaround') as HTMLInputElement
    rate.value = '0.2'
    hold.value = '0'
    boxes().solver.focus()
    pane.updateDevice()
    expect(rate.value).toBe('0.2')
    expect(hold.value).toBe('0')
    pane.updateDevice(true)
    expect(rate.value).toBe('0.04')
    expect(hold.value).toBe('3')
  })

  it('restores the figure rate after leaving a charged-interface solver lock', () => {
    let config = chargedInterfaceConfig
    const pane = mountJVPane(container, {
      getActiveDevice: () => ({ id: 'device', config }), onRunComplete: () => {},
    })
    expect((document.getElementById('jvp-rate') as HTMLInputElement).value).toBe('0')
    config = figureConfig()
    pane.updateDevice()
    expect((document.getElementById('jvp-rate') as HTMLInputElement).value).toBe('0.04')
    expect(boxes().solver.value).toBe('transient')
    expect(boxes().solver.disabled).toBe(false)
  })
})
