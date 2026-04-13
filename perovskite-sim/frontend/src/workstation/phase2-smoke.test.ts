import { describe, it, expect } from 'vitest'
import {
  createEmptyWorkspace,
  addDevice,
  addExperiment,
  addRun,
  setActiveRun,
  setActiveExperiment,
  findRun,
} from './state'
import type { Device, Run } from './types'
import { renderTreeHTML } from './tree'

function device(id: string, tier: 'legacy' | 'fast' | 'full' = 'full'): Device {
  return {
    id,
    name: `dev-${id}`,
    tier,
    config: {
      device: { V_bi: 1.1, Phi: 1.4e21 },
      layers: [],
    } as unknown as Device['config'],
    experiments: [],
  }
}

function jvRun(id: string): Run {
  return {
    id,
    timestamp: Date.now(),
    result: {
      kind: 'jv',
      data: {
        V_fwd: [0, 0.5, 1],
        J_fwd: [0, 50, 200],
        V_rev: [1, 0.5, 0],
        J_rev: [200, 50, 0],
        metrics_fwd: { V_oc: 1.08, J_sc: 200, FF: 0.75, PCE: 0.162 },
        metrics_rev: { V_oc: 1.09, J_sc: 200, FF: 0.76, PCE: 0.163 },
        hysteresis_index: 0.01,
      },
    } as unknown as Run['result'],
    activePhysics: 'FULL',
    durationMs: 1234,
    deviceSnapshot: {
      device: { V_bi: 1.1, Phi: 1.4e21 },
      layers: [],
    } as unknown as Run['deviceSnapshot'],
  }
}

describe('Phase 2 headline: create device → run experiment → run persists → re-activate', () => {
  it('end-to-end state transitions', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, device('d1', 'full'))
    ws = addExperiment(ws, 'd1', { id: 'e1', kind: 'jv', params: {}, runs: [] })
    ws = addRun(ws, 'd1', 'e1', jvRun('r1'))
    ws = setActiveRun(ws, 'd1', 'e1', 'r1')
    expect(ws.activeRunId).toBe('r1')
    expect(findRun(ws, 'd1', 'e1', 'r1')?.id).toBe('r1')

    const html = renderTreeHTML(ws)
    expect(html).toContain('data-run-id="r1"')
    expect(html).toContain('data-experiment-id="e1"')
    expect(html).toContain('FULL')
  })

  it('setActiveExperiment clears activeRunId', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, device('d1'))
    ws = addExperiment(ws, 'd1', { id: 'e1', kind: 'jv', params: {}, runs: [] })
    ws = addRun(ws, 'd1', 'e1', jvRun('r1'))
    ws = setActiveRun(ws, 'd1', 'e1', 'r1')
    ws = setActiveExperiment(ws, 'd1', 'e1')
    expect(ws.activeRunId).toBeNull()
  })
})
