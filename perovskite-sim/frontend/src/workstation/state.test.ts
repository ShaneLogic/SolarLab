import { describe, it, expect, beforeEach } from 'vitest'
import {
  createEmptyWorkspace,
  addDevice,
  removeDevice,
  setActiveDevice,
  saveWorkspace,
  loadWorkspace,
  STORAGE_KEY,
} from './state'
import type { Device } from './types'

function makeDevice(id: string, name = 'Test'): Device {
  return {
    id,
    name,
    tier: 'full',
    config: { device: { V_bi: 1.1, Phi: 1.4e21 }, layers: [] },
    experiments: [],
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
