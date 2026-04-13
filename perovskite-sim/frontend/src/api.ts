import type {
  DeviceConfig,
  JVResult,
  ISResult,
  DegResult,
  JVParams,
  ISParams,
  DegParams,
} from './types'

const BASE = 'http://127.0.0.1:8000'

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${await res.text()}`)
  }
  const data = await res.json()
  return data.result as T
}

export async function listConfigs(): Promise<string[]> {
  const res = await fetch(`${BASE}/api/configs`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const data = await res.json()
  return data.configs as string[]
}

export async function getConfig(name: string): Promise<DeviceConfig> {
  const res = await fetch(`${BASE}/api/configs/${encodeURIComponent(name)}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const data = await res.json()
  return data.config as DeviceConfig
}

export async function runJV(device: DeviceConfig, params: JVParams): Promise<JVResult> {
  const res = await fetch(`${BASE}/api/jv`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ device, ...params }),
  })
  return handle<JVResult>(res)
}

export async function runImpedance(device: DeviceConfig, params: ISParams): Promise<ISResult> {
  const res = await fetch(`${BASE}/api/impedance`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ device, ...params }),
  })
  return handle<ISResult>(res)
}

export async function runDegradation(device: DeviceConfig, params: DegParams): Promise<DegResult> {
  const res = await fetch(`${BASE}/api/degradation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ device, ...params }),
  })
  return handle<DegResult>(res)
}
