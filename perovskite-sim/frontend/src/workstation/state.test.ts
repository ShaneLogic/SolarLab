import { describe, it, expect, beforeEach } from 'vitest'
import {
  createEmptyWorkspace,
  addDevice,
  removeDevice,
  setActiveDevice,
  saveWorkspace,
  loadWorkspace,
  STORAGE_KEY,
  addExperiment,
  removeExperiment,
  setActiveExperiment,
  addRun,
  removeRun,
  setActiveRun,
  findRun,
} from './state'
import type { Device, Experiment, Run } from './types'

function makeDevice(id: string, name = 'Test'): Device {
  return {
    id,
    name,
    tier: 'full',
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
    timestamp: Date.now(),
    result: { placeholder: true },
    activePhysics: 'FULL',
    durationMs: 123,
    deviceSnapshot: { device: { V_bi: 1.1, Phi: 1.4e21 }, layers: [] },
  }
}

describe('createEmptyWorkspace', () => {
  it('returns a workspace with version 1, no devices, nothing active', () => {
    const ws = createEmptyWorkspace('My Workspace')
    expect(ws.version).toBe(1)
    expect(ws.name).toBe('My Workspace')
    expect(ws.devices).toEqual([])
    expect(ws.activeDeviceId).toBeNull()
    expect(ws.activeExperimentId).toBeNull()
    expect(ws.activeRunId).toBeNull()
    expect(ws.layout).toBeNull()
    expect(typeof ws.id).toBe('string')
    expect(ws.id.length).toBeGreaterThan(0)
  })
})

describe('addDevice', () => {
  it('returns a new workspace with the device appended — original is untouched', () => {
    const ws = createEmptyWorkspace('W')
    const dev = makeDevice('d1')
    const next = addDevice(ws, dev)
    expect(next.devices).toHaveLength(1)
    expect(next.devices[0].id).toBe('d1')
    expect(ws.devices).toHaveLength(0) // immutability check
  })

  it('sets activeDeviceId to the new device when no device was active', () => {
    const ws = createEmptyWorkspace('W')
    const next = addDevice(ws, makeDevice('d1'))
    expect(next.activeDeviceId).toBe('d1')
  })

  it('leaves activeDeviceId alone when a device was already active', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1'))
    ws = addDevice(ws, makeDevice('d2'))
    expect(ws.activeDeviceId).toBe('d1')
  })
})

describe('removeDevice', () => {
  it('removes the matching device and returns a new workspace', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1'))
    ws = addDevice(ws, makeDevice('d2'))
    const next = removeDevice(ws, 'd1')
    expect(next.devices.map(d => d.id)).toEqual(['d2'])
  })

  it('clears activeDeviceId if the active device was removed', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1'))
    const next = removeDevice(ws, 'd1')
    expect(next.activeDeviceId).toBeNull()
  })
})

describe('setActiveDevice', () => {
  it('sets activeDeviceId when the id exists', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1'))
    ws = addDevice(ws, makeDevice('d2'))
    const next = setActiveDevice(ws, 'd2')
    expect(next.activeDeviceId).toBe('d2')
  })

  it('returns the same workspace reference when the id does not exist', () => {
    const ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    const next = setActiveDevice(ws, 'unknown')
    expect(next).toBe(ws)
  })
})

describe('saveWorkspace / loadWorkspace', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('roundtrips a workspace through localStorage', () => {
    let ws = createEmptyWorkspace('Roundtrip')
    ws = addDevice(ws, makeDevice('d1', 'Alpha'))
    saveWorkspace(ws)
    const loaded = loadWorkspace()
    expect(loaded).not.toBeNull()
    expect(loaded!.name).toBe('Roundtrip')
    expect(loaded!.devices[0].name).toBe('Alpha')
  })

  it('returns null when nothing is stored', () => {
    expect(loadWorkspace()).toBeNull()
  })

  it('returns null when the stored blob has a different schema version', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 99 }))
    expect(loadWorkspace()).toBeNull()
  })

  it('returns null when the stored blob is not JSON', () => {
    localStorage.setItem(STORAGE_KEY, 'not json')
    expect(loadWorkspace()).toBeNull()
  })
})

describe('addExperiment', () => {
  it('appends experiment to the named device, leaves other devices alone', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1'))
    ws = addDevice(ws, makeDevice('d2'))
    const next = addExperiment(ws, 'd1', makeExperiment('e1'))
    expect(next.devices.find(d => d.id === 'd1')!.experiments).toHaveLength(1)
    expect(next.devices.find(d => d.id === 'd2')!.experiments).toHaveLength(0)
  })

  it('is a no-op when the device id is unknown', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    const next = addExperiment(ws, 'unknown', makeExperiment('e1'))
    expect(next).toBe(ws)
  })

  it('returns a new workspace reference (immutability)', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    const next = addExperiment(ws, 'd1', makeExperiment('e1'))
    expect(next).not.toBe(ws)
    expect(ws.devices[0].experiments).toHaveLength(0)
  })
})

describe('removeExperiment', () => {
  it('removes the experiment from its device', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e2'))
    const next = removeExperiment(ws, 'd1', 'e1')
    expect(next.devices[0].experiments.map(e => e.id)).toEqual(['e2'])
  })

  it('clears activeExperimentId when the removed experiment was active', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    ws = setActiveExperiment(ws, 'd1', 'e1')
    const next = removeExperiment(ws, 'd1', 'e1')
    expect(next.activeExperimentId).toBeNull()
  })
})

describe('setActiveExperiment', () => {
  it('sets both activeDeviceId and activeExperimentId', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    const next = setActiveExperiment(ws, 'd1', 'e1')
    expect(next.activeDeviceId).toBe('d1')
    expect(next.activeExperimentId).toBe('e1')
  })

  it('is a no-op when device or experiment id is unknown', () => {
    const ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    expect(setActiveExperiment(ws, 'd1', 'missing')).toBe(ws)
    expect(setActiveExperiment(ws, 'missing', 'e1')).toBe(ws)
  })
})

describe('addRun', () => {
  it('appends run under the named experiment', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    const next = addRun(ws, 'd1', 'e1', makeRun('r1'))
    expect(next.devices[0].experiments[0].runs.map(r => r.id)).toEqual(['r1'])
  })

  it('is a no-op when device or experiment is unknown', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    expect(addRun(ws, 'd1', 'missing', makeRun('r1'))).toBe(ws)
    expect(addRun(ws, 'missing', 'e1', makeRun('r1'))).toBe(ws)
  })
})

describe('removeRun', () => {
  it('removes the run and clears activeRunId if it was active', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    ws = addRun(ws, 'd1', 'e1', makeRun('r1'))
    ws = setActiveRun(ws, 'd1', 'e1', 'r1')
    const next = removeRun(ws, 'd1', 'e1', 'r1')
    expect(next.devices[0].experiments[0].runs).toHaveLength(0)
    expect(next.activeRunId).toBeNull()
  })
})

describe('setActiveRun', () => {
  it('sets activeDevice/Experiment/Run', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    ws = addRun(ws, 'd1', 'e1', makeRun('r1'))
    const next = setActiveRun(ws, 'd1', 'e1', 'r1')
    expect(next.activeDeviceId).toBe('d1')
    expect(next.activeExperimentId).toBe('e1')
    expect(next.activeRunId).toBe('r1')
  })
})

describe('findRun', () => {
  it('returns the run when it exists', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    ws = addRun(ws, 'd1', 'e1', makeRun('r1'))
    const found = findRun(ws, 'd1', 'e1', 'r1')
    expect(found?.id).toBe('r1')
  })

  it('returns undefined for an unknown triple', () => {
    expect(findRun(createEmptyWorkspace('W'), 'x', 'y', 'z')).toBeUndefined()
  })
})
