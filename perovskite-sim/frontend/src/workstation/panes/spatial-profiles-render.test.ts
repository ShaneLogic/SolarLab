import { expect, it, vi } from 'vitest'
vi.mock('plotly.js-basic-dist-min', () => ({ default: { newPlot: vi.fn(), purge: vi.fn() } }))
import Plotly from 'plotly.js-basic-dist-min'
import { renderSpatialProfiles } from './main-plot-pane'
import type { SpatialSnapshot } from '../../types'

it('shows both ion species and selects actual reverse snapshots', () => {
  const el = document.createElement('div')
  const snapshot: SpatialSnapshot = {
    x: [0, 1, 2], phi: [0, 0.4, 1], E: [-1, -2],
    n: [1, 2, 3], p: [3, 2, 1], P: [0, 1e25, 0],
    P_neg: [0, 2e24, 0], rho: [0, 1, 0], V_app: 0,
  }
  renderSpatialProfiles(el, { V_fwd: [0], V_rev: [1], snapshots_fwd: [snapshot],
    snapshots_rev: [{ ...snapshot, V_app: 1, P: [0, 2e25, 0] }] })
  const traces = () => vi.mocked(Plotly.newPlot).mock.calls.at(-1)![1] as Array<{ name: string; y: number[] }>
  expect(traces().find(trace => trace.name === 'c')!.y).toEqual([null, 1e25, null])
  expect(traces().find(trace => trace.name === 'a')!.y).toEqual([null, 2e24, null])
  const branch = el.querySelector('select')!
  branch.value = 'rev'
  branch.dispatchEvent(new Event('change'))
  expect(traces().find(trace => trace.name === 'c')!.y).toEqual([null, 2e25, null])
  expect(el.querySelector('output')!.textContent).toBe('1.000 V')
  expect(snapshot.P[0]).toBe(0)
})
