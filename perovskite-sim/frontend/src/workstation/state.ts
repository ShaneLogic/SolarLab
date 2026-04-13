import type { Device, Workspace } from './types'

export const STORAGE_KEY = 'solarsim:workspace:v1'

function randomId(): string {
  // Sufficient uniqueness for a single-user local workspace; not a crypto concern.
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}

export function createEmptyWorkspace(name: string): Workspace {
  return {
    version: 1,
    id: randomId(),
    name,
    devices: [],
    activeDeviceId: null,
    layout: null,
  }
}

export function addDevice(ws: Workspace, device: Device): Workspace {
  return {
    ...ws,
    devices: [...ws.devices, device],
    activeDeviceId: ws.activeDeviceId ?? device.id,
  }
}

export function removeDevice(ws: Workspace, deviceId: string): Workspace {
  const devices = ws.devices.filter(d => d.id !== deviceId)
  const activeDeviceId = ws.activeDeviceId === deviceId ? null : ws.activeDeviceId
  return { ...ws, devices, activeDeviceId }
}

export function setActiveDevice(ws: Workspace, deviceId: string): Workspace {
  if (!ws.devices.some(d => d.id === deviceId)) return ws
  if (ws.activeDeviceId === deviceId) return ws
  return { ...ws, activeDeviceId: deviceId }
}

export function saveWorkspace(ws: Workspace): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(ws))
  } catch (e) {
    // localStorage can throw on quota exceeded; failing to persist should not crash the UI.
    console.error('saveWorkspace failed:', e)
  }
}

export function loadWorkspace(): Workspace | null {
  const raw = localStorage.getItem(STORAGE_KEY)
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw) as Partial<Workspace>
    if (parsed.version !== 1) return null
    // Minimal structural check — later phases can add migrations here.
    if (!Array.isArray(parsed.devices)) return null
    return parsed as Workspace
  } catch {
    return null
  }
}
