import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getScapsReferenceCatalog, getScapsReferencePoint } from './api'
import { mountScapsReferenceControls } from './scaps-reference-controls'
import type { DeviceConfig } from './types'

vi.mock('./api', () => ({ getScapsReferenceCatalog: vi.fn(), getScapsReferencePoint: vi.fn() }))
const config: DeviceConfig = {
  device: { Phi: 2.5e21, mode: 'fast', optical_model: 'beer_lambert_spectral' }, layers: [],
  simulation_hints: { reference_study: 'scaps_beer_lambert_v1', jv_sweep: { N_grid: 30, n_points: 40, v_rate: 5, V_max: 1.6 } },
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(getScapsReferenceCatalog).mockResolvedValue({ sweeps: { CHI_ETL: { title: 'Band offset', unit: 'eV', points: [{ x: -1 }, { x: .7 }] } } })
  vi.mocked(getScapsReferencePoint).mockResolvedValue(config)
})

describe('SCAPS report parameter loading', () => {
  it('loads normal device parameters, without launching a simulation', async () => {
    const root = document.createElement('div')
    const onLoad = vi.fn()
    mountScapsReferenceControls(root, onLoad).update(config)
    await vi.waitFor(() => expect(root.querySelectorAll('option')).toHaveLength(1))
    root.querySelector('input')!.value = '0.5'
    root.querySelector('button')!.click()
    await vi.waitFor(() => expect(onLoad).toHaveBeenCalledWith(config))
    expect(getScapsReferencePoint).toHaveBeenCalledWith('CHI_ETL', .5)
  })

  it('rejects blank and out-of-range values', async () => {
    const root = document.createElement('div')
    mountScapsReferenceControls(root, vi.fn()).update(config)
    await vi.waitFor(() => expect(root.querySelectorAll('option')).toHaveLength(1))
    root.querySelector('input')!.value = '8'
    root.querySelector('button')!.click()
    expect(getScapsReferencePoint).not.toHaveBeenCalled()
  })

  it('stays absent from other reference models', () => {
    const root = document.createElement('div')
    mountScapsReferenceControls(root, vi.fn()).update({ device: config.device, layers: [] })
    expect(root.hidden).toBe(true)
    expect(getScapsReferenceCatalog).not.toHaveBeenCalled()
  })

  it('labels high-doping extensions and retains their grid defaults', async () => {
    vi.mocked(getScapsReferenceCatalog).mockResolvedValue({ sweeps: {
      Nd_PVK: { title: 'PVK donor doping', unit: 'cm^-3', points: [{ x: 1e8 }, { x: 1e18 }, { x: 1e19 }, { x: 1e20 }] },
    } })
    const extended: DeviceConfig = {
      ...config,
      simulation_hints: {
        ...config.simulation_hints,
        reference_point: { sweep: 'Nd_PVK', value: 1e19, scope: 'high_doping_extrapolation' },
        jv_sweep: { N_grid: 90, n_points: 40, v_rate: 5, V_max: 1.6 },
        min_N_grid: 90,
      },
    }
    vi.mocked(getScapsReferencePoint).mockResolvedValue(extended)
    const root = document.createElement('div')
    const onLoad = vi.fn()
    mountScapsReferenceControls(root, onLoad).update(config)
    await vi.waitFor(() => expect(root.querySelectorAll('option')).toHaveLength(1))
    expect(root.querySelector('input')!.max).toBe('100000000000000000000')
    root.querySelector('input')!.value = '1e19'
    root.querySelector('button')!.click()
    await vi.waitFor(() => expect(onLoad).toHaveBeenCalledWith(extended))
    expect(getScapsReferencePoint).toHaveBeenCalledWith('Nd_PVK', 1e19)
    expect(root.querySelector('[role="status"]')!.textContent).toContain('high-doping extrapolation')
  })
})
