/**
 * V2 solarlab_server configuration, material, and browser-preset client.
 *
 * Uses exclusively V2 routes (`/configs`, `/configs/{id}`, `/materials`,
 * `/configuration-preview/device`) with zero legacy wire endpoints.
 */
import type {
  ConfigEntry,
  DeviceConfig,
  LayerConfig,
  LayerRole,
  LayerTemplate,
} from './types'

const USER_PRESETS_STORAGE_KEY = 'solarlab.v2.user_presets.v1'

export function resolveV2ServiceEndpoint(explicit?: string | URL): string {
  if (explicit !== undefined) {
    const url = new URL(explicit)
    return url.href.replace(/\/$/, '') + '/'
  }
  const configured = import.meta.env.VITE_SOLARLAB_SERVICE_BASE
  if (typeof configured === 'string' && configured.trim().length > 0) {
    const url = new URL(configured, globalThis.location?.origin ?? 'http://127.0.0.1:8000')
    return url.href.replace(/\/$/, '') + '/'
  }
  const origin = globalThis.location?.origin
  if (origin && /^https?:\/\//.test(origin)) {
    return origin.replace(/\/$/, '') + '/'
  }
  return 'http://127.0.0.1:8000/'
}

interface StoredUserPresetMap {
  [stem: string]: DeviceConfig
}

function readLocalUserPresets(): StoredUserPresetMap {
  try {
    const raw = globalThis.localStorage?.getItem(USER_PRESETS_STORAGE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return {}
    return parsed as StoredUserPresetMap
  } catch {
    return {}
  }
}

function writeLocalUserPresets(map: StoredUserPresetMap): void {
  globalThis.localStorage?.setItem(USER_PRESETS_STORAGE_KEY, JSON.stringify(map))
}

function parseScalar(raw: string): unknown {
  const value = raw.trim()
  if (value === '' || value === 'null' || value === '~') return null
  if (value === 'true') return true
  if (value === 'false') return false
  if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
    return value.slice(1, -1)
  }
  const num = Number(value)
  if (Number.isFinite(num)) return num
  return value
}

function parseQuantityNumber(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string') {
    const trimmed = value.trim()
    const match = /^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$/.exec(trimmed)
    if (match) {
      const n = Number(match[1])
      const unit = match[2].trim()
      if (unit === 'nm') return n * 1e-9
      if (unit === 'um' || unit === 'µm') return n * 1e-6
      if (unit === 'cm^2/(V s)' || unit === 'cm^2/Vs') return n * 1e-4
      if (unit === 'cm^-3') return n * 1e6
      if (unit === 'cm^-1') return n * 1e2
      return n
    }
  }
  return fallback
}

function normalizeRole(raw: unknown): LayerRole {
  const s = String(raw ?? 'absorber')
  if (
    s === 'ETL'
    || s === 'HTL'
    || s === 'absorber'
    || s === 'front_contact'
    || s === 'back_contact'
    || s === 'substrate'
  ) {
    return s
  }
  const upper = s.toUpperCase()
  if (upper === 'ETL' || upper === 'HTL') return upper
  return 'absorber'
}

function layerFromMapping(raw: Record<string, unknown>, materialParams?: Record<string, unknown>): LayerConfig {
  const merged = { ...(materialParams ?? {}), ...(raw.parameters as Record<string, unknown> | undefined ?? {}), ...raw }
  const num = (key: string, fallback: number): number =>
    merged[key] !== undefined && merged[key] !== null ? parseQuantityNumber(merged[key], fallback) : fallback
  return {
    name: String(raw.name ?? raw.id ?? 'layer'),
    role: normalizeRole(raw.role),
    thickness: num('thickness', 2e-7),
    eps_r: num('eps_r', 10),
    mu_n: num('mu_n', 1e-4),
    mu_p: num('mu_p', 1e-4),
    ni: num('ni', 1e11),
    N_D: num('N_D', 0),
    N_A: num('N_A', 0),
    D_ion: num('D_ion', 0),
    P0: num('P0', 0),
    P_lim: num('P_lim', 1e30),
    D_ion_neg: merged.D_ion_neg !== undefined ? num('D_ion_neg', 0) : undefined,
    P0_neg: merged.P0_neg !== undefined ? num('P0_neg', 0) : undefined,
    tau_n: num('tau_n', 1e-6),
    tau_p: num('tau_p', 1e-6),
    n1: num('n1', 1e11),
    p1: num('p1', 1e11),
    B_rad: num('B_rad', 0),
    C_n: num('C_n', 0),
    C_p: num('C_p', 0),
    alpha: num('alpha', 0),
    Eg: merged.Eg !== undefined ? num('Eg', 1.6) : undefined,
    chi: merged.chi !== undefined ? num('chi', 3.9) : undefined,
    optical_material: typeof merged.optical_material === 'string' ? merged.optical_material : undefined,
  }
}

export function parseConfigDocument(text: string): DeviceConfig {
  const trimmed = text.replace(/^\uFEFF/, '').trim()
  if (trimmed.startsWith('{')) {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>
    if (parsed.device && Array.isArray(parsed.layers)) {
      return parsed as unknown as DeviceConfig
    }
    const dto = (parsed.device && typeof parsed.device === 'object' && !Array.isArray(parsed.device) && Array.isArray((parsed.device as Record<string, unknown>).layers)
      ? parsed.device
      : parsed) as Record<string, unknown>
    const materialsById = new Map<string, Record<string, unknown>>()
    if (Array.isArray(dto.materials)) {
      for (const m of dto.materials) {
        if (m && typeof m === 'object') {
          const rec = m as Record<string, unknown>
          if (typeof rec.id === 'string' && rec.parameters && typeof rec.parameters === 'object') {
            materialsById.set(rec.id, rec.parameters as Record<string, unknown>)
          }
        }
      }
    }
    const settings = (dto.settings as Record<string, unknown> | undefined) ?? {}
    const rawLayers = Array.isArray(dto.layers) ? dto.layers : []
    const layers = rawLayers
      .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === 'object'))
      .map(item => {
        const matId = typeof item.material === 'string' ? item.material : undefined
        return layerFromMapping(item, matId ? materialsById.get(matId) : undefined)
      })
    return {
      device: {
        V_bi: settings.V_bi !== undefined && settings.V_bi !== null ? parseQuantityNumber(settings.V_bi, 1.05) : 1.05,
        Phi: parseQuantityNumber(settings.Phi, 2.5e21),
        T: settings.T !== undefined ? parseQuantityNumber(settings.T, 300) : 300,
        mode: 'full',
      },
      layers,
    }
  }

  // Minimal deterministic YAML reader for shipped standard device presets (`device:` + `layers:`).
  const device: Record<string, unknown> = {}
  const layers: Array<Record<string, unknown>> = []
  let section: 'none' | 'device' | 'interfaces' | 'layers' = 'none'
  let currentLayer: Record<string, unknown> | null = null
  const interfaces: Array<[number, number]> = []

  for (const rawLine of trimmed.split(/\r?\n/)) {
    const commentIdx = rawLine.indexOf('#')
    const line = (commentIdx >= 0 ? rawLine.slice(0, commentIdx) : rawLine).replace(/\s+$/, '')
    if (!line.trim()) continue
    if (/^device:\s*$/.test(line)) {
      section = 'device'
      currentLayer = null
      continue
    }
    if (/^layers:\s*$/.test(line)) {
      section = 'layers'
      currentLayer = null
      continue
    }
    if (/^[A-Za-z0-9_]+:/.test(line)) {
      section = 'none'
      currentLayer = null
      continue
    }
    if (section === 'device' && /^\s{2}interfaces:\s*$/.test(line)) {
      section = 'interfaces'
      continue
    }
    if (section === 'interfaces') {
      const pairMatch = /^\s*-\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]/.exec(line)
      if (pairMatch) {
        interfaces.push([Number(pairMatch[1]) || 0, Number(pairMatch[2]) || 0])
        continue
      }
      if (/^\s{2}[A-Za-z0-9_]+:/.test(line)) {
        section = 'device'
      }
    }
    if (section === 'device') {
      const kv = /^\s{2}([A-Za-z0-9_]+):\s*(.*)$/.exec(line)
      if (kv) device[kv[1]] = parseScalar(kv[2])
      continue
    }
    if (section === 'layers') {
      const itemStart = /^\s{2}-\s*([A-Za-z0-9_]+):\s*(.*)$/.exec(line)
      if (itemStart) {
        currentLayer = { [itemStart[1]]: parseScalar(itemStart[2]) }
        layers.push(currentLayer)
        continue
      }
      const kv = /^\s{4}([A-Za-z0-9_]+):\s*(.*)$/.exec(line)
      if (kv && currentLayer) {
        currentLayer[kv[1]] = parseScalar(kv[2])
      }
    }
  }
  if (interfaces.length > 0) device.interfaces = interfaces
  return {
    device: {
      V_bi: typeof device.V_bi === 'number' ? device.V_bi : 1.05,
      Phi: typeof device.Phi === 'number' ? device.Phi : 2.5e21,
      ...device,
    } as DeviceConfig['device'],
    layers: layers.map(l => layerFromMapping(l)),
  }
}

export async function listConfigs(endpoint?: string | URL): Promise<ConfigEntry[]> {
  const base = resolveV2ServiceEndpoint(endpoint)
  const shipped: ConfigEntry[] = []
  try {
    const res = await fetch(new URL('configs', base), { cache: 'no-store' })
    if (res.ok) {
      const data = (await res.json()) as { sources?: Array<{ id: string }> }
      for (const item of data.sources ?? []) {
        if (typeof item.id === 'string' && item.id.length > 0) {
          shipped.push({
            name: item.id,
            namespace: 'shipped',
            device_type: item.id.toLowerCase().includes('tandem') ? 'tandem_2T' : 'single_junction',
            tier_compat: ['legacy', 'fast', 'full'],
          })
        }
      }
    }
  } catch {
    // Offline or configuration preview context unconfigured; still surface local user presets.
  }
  const userMap = readLocalUserPresets()
  const userEntries: ConfigEntry[] = Object.keys(userMap)
    .sort()
    .map(stem => ({
      name: `${stem}.yaml`,
      namespace: 'user',
      device_type: 'single_junction',
      tier_compat: ['legacy', 'fast', 'full'],
    }))
  return [...shipped, ...userEntries]
}

export async function getConfig(name: string, endpoint?: string | URL): Promise<DeviceConfig> {
  const stem = name.replace(/\.ya?ml$/i, '')
  const userMap = readLocalUserPresets()
  if (Object.hasOwn(userMap, stem)) {
    return structuredClone(userMap[stem])
  }
  const base = resolveV2ServiceEndpoint(endpoint)
  const encoded = name.split('/').map(encodeURIComponent).join('/')
  const res = await fetch(new URL(`configs/${encoded}`, base), { cache: 'no-store' })
  if (!res.ok) {
    throw new Error(`GET /configs/${name} failed: HTTP ${res.status}`)
  }
  const text = await res.text()
  return parseConfigDocument(text)
}

export async function fetchOpticalMaterials(endpoint?: string | URL): Promise<string[]> {
  const base = resolveV2ServiceEndpoint(endpoint)
  const res = await fetch(new URL('materials', base), { cache: 'no-store' })
  if (!res.ok) throw new Error(`GET /materials failed: HTTP ${res.status}`)
  const data = (await res.json()) as { resources?: Array<{ name: string; kind: string }> }
  return (data.resources ?? [])
    .filter(item => item.kind === 'nk' && typeof item.name === 'string')
    .map(item => item.name)
}

const BUILTIN_LAYER_TEMPLATES: Record<string, LayerTemplate> = {
  MAPbI3_absorber: {
    role: 'absorber',
    optical_material: null,
    description: 'Intrinsic MAPbI3 perovskite absorber',
    source: 'SolarLab V2 built-in template',
    defaults: {
      name: 'MAPbI3',
      role: 'absorber',
      thickness: 4e-7,
      eps_r: 24.1,
      mu_n: 2e-4,
      mu_p: 2e-4,
      ni: 1e11,
      N_D: 0,
      N_A: 0,
      D_ion: 0,
      P0: 0,
      tau_n: 5e-7,
      tau_p: 5e-7,
      n1: 1e11,
      p1: 1e11,
      B_rad: 5e-17,
      C_n: 1e-40,
      C_p: 1e-40,
      alpha: 1e7,
      Eg: 1.6,
      chi: 3.9,
    },
  },
  TiO2_ETL: {
    role: 'ETL',
    optical_material: null,
    description: 'Compact n-type TiO2 electron transport layer',
    source: 'SolarLab V2 built-in template',
    defaults: {
      name: 'TiO2',
      role: 'ETL',
      thickness: 1e-7,
      eps_r: 9.0,
      mu_n: 5e-5,
      mu_p: 1e-6,
      ni: 1e7,
      N_D: 1e24,
      N_A: 0,
      D_ion: 0,
      P0: 0,
      tau_n: 1e-6,
      tau_p: 1e-6,
      n1: 1e7,
      p1: 1e7,
      B_rad: 0,
      C_n: 0,
      C_p: 0,
      alpha: 0,
      Eg: 3.2,
      chi: 4.0,
    },
  },
  Spiro_HTL: {
    role: 'HTL',
    optical_material: null,
    description: 'Doped p-type Spiro-OMeTAD hole transport layer',
    source: 'SolarLab V2 built-in template',
    defaults: {
      name: 'Spiro',
      role: 'HTL',
      thickness: 1e-7,
      eps_r: 3.5,
      mu_n: 1e-6,
      mu_p: 2e-5,
      ni: 1e8,
      N_D: 0,
      N_A: 1e24,
      D_ion: 0,
      P0: 0,
      tau_n: 1e-6,
      tau_p: 1e-6,
      n1: 1e8,
      p1: 1e8,
      B_rad: 0,
      C_n: 0,
      C_p: 0,
      alpha: 0,
      Eg: 2.2,
      chi: 3.2,
    },
  },
}

export async function fetchLayerTemplates(): Promise<Record<string, LayerTemplate>> {
  return structuredClone(BUILTIN_LAYER_TEMPLATES)
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

export async function checkUserConfigExists(
  name: string,
  endpoint?: string | URL,
): Promise<{ exists: boolean; namespace: 'shipped' | 'user' | null }> {
  const entries = await listConfigs(endpoint)
  const match = entries.find(
    e => e.name === name || e.name === `${name}.yaml` || e.name === `${name}.yml`,
  )
  if (!match) return { exists: false, namespace: null }
  return { exists: true, namespace: match.namespace }
}

export async function saveUserConfig(
  name: string,
  config: DeviceConfig,
  overwrite = false,
  endpoint?: string | URL,
): Promise<SaveUserConfigResponse> {
  const trimmed = name.trim().replace(/\.ya?ml$/i, '')
  if (!/^[a-zA-Z0-9_-]{1,64}$/.test(trimmed)) {
    return { ok: false, status: 422, detail: 'Invalid preset filename' }
  }
  const probe = await checkUserConfigExists(trimmed, endpoint)
  if (probe.exists && probe.namespace === 'shipped') {
    return { ok: false, status: 409, detail: `"${trimmed}" is reserved by a shipped preset` }
  }
  if (probe.exists && probe.namespace === 'user' && !overwrite) {
    return { ok: false, status: 409, detail: `"${trimmed}" already exists` }
  }
  const map = readLocalUserPresets()
  map[trimmed] = structuredClone(config)
  writeLocalUserPresets(map)
  return { ok: true, saved: trimmed }
}
