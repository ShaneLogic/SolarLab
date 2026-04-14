import type {
  DeviceConfig,
  JVResult,
  ISResult,
  DegResult,
  JVParams,
  ISParams,
  DegParams,
  ConfigEntry,
  LayerTemplate,
} from './types'

const BASE = 'http://127.0.0.1:8000'

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${await res.text()}`)
  }
  const data = await res.json()
  return data.result as T
}

export async function listConfigs(): Promise<ConfigEntry[]> {
  const res = await fetch(`${BASE}/api/configs`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const data = await res.json()
  return data.configs as ConfigEntry[]
}

export async function getConfig(name: string): Promise<DeviceConfig> {
  const res = await fetch(`${BASE}/api/configs/${encodeURIComponent(name)}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const data = await res.json()
  return data.config as DeviceConfig
}

export async function fetchOpticalMaterials(): Promise<string[]> {
  const res = await fetch(`${BASE}/api/optical-materials`)
  if (!res.ok) throw new Error(`fetchOpticalMaterials failed: ${res.status}`)
  const data = (await res.json()) as { materials: string[] }
  return data.materials
}

export async function fetchLayerTemplates(): Promise<Record<string, LayerTemplate>> {
  const res = await fetch(`${BASE}/api/layer-templates`)
  if (!res.ok) throw new Error(`fetchLayerTemplates failed: ${res.status}`)
  const data = (await res.json()) as { templates: Record<string, LayerTemplate> }
  return data.templates
}

export interface SaveUserConfigResult {
  ok: true
  saved: string
}

export interface SaveUserConfigError {
  ok: false
  status: number
  detail: string
}

export type SaveUserConfigResponse = SaveUserConfigResult | SaveUserConfigError

export async function saveUserConfig(
  name: string,
  config: DeviceConfig,
  overwrite = false,
): Promise<SaveUserConfigResponse> {
  const res = await fetch(`${BASE}/api/configs/user`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, config, overwrite }),
  })
  if (res.ok) {
    const data = await res.json()
    return { ok: true, saved: data.saved as string }
  }
  let detail = `HTTP ${res.status}`
  try {
    const data = await res.json()
    if (data?.detail) detail = String(data.detail)
  } catch { /* non-JSON body */ }
  return { ok: false, status: res.status, detail }
}

export async function checkUserConfigExists(
  name: string,
): Promise<{ exists: boolean; namespace: 'shipped' | 'user' | null }> {
  const entries = await listConfigs()
  const match = entries.find(
    e => e.name === `${name}.yaml` || e.name === `${name}.yml`,
  )
  if (!match) return { exists: false, namespace: null }
  return { exists: true, namespace: match.namespace }
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
