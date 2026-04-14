export type LayerRole =
  | 'substrate'
  | 'front_contact'
  | 'ETL'
  | 'absorber'
  | 'HTL'
  | 'back_contact'

export const LAYER_ROLES: readonly LayerRole[] = [
  'substrate',
  'front_contact',
  'ETL',
  'absorber',
  'HTL',
  'back_contact',
]

export function isLayerRole(v: unknown): v is LayerRole {
  return typeof v === 'string' && (LAYER_ROLES as readonly string[]).includes(v)
}

export interface LayerConfig {
  name: string
  role: LayerRole
  thickness: number
  eps_r: number
  mu_n: number
  mu_p: number
  ni: number
  N_D: number
  N_A: number
  D_ion: number
  P_lim: number
  P0: number
  tau_n: number
  tau_p: number
  n1: number
  p1: number
  B_rad: number
  C_n: number
  C_p: number
  alpha: number
  chi?: number
  Eg?: number
  optical_material?: string | null
  incoherent?: boolean
}

export type SimulationModeName = 'legacy' | 'fast' | 'full'

export interface DeviceConfig {
  device: {
    V_bi: number
    Phi: number
    interfaces?: Array<[number, number]>
    T?: number
    mode?: SimulationModeName
  }
  layers: LayerConfig[]
}

export interface JVMetrics {
  V_oc: number
  J_sc: number
  FF: number
  PCE: number
}

export interface JVResult {
  V_fwd: number[]
  J_fwd: number[]
  V_rev: number[]
  J_rev: number[]
  metrics_fwd: JVMetrics
  metrics_rev: JVMetrics
  hysteresis_index: number
}

export interface ISResult {
  frequencies: number[]
  Z_real: number[]
  Z_imag: number[]
}

export interface DegResult {
  times: number[]
  PCE: number[]
  V_oc: number[]
  J_sc: number[]
}

export interface JVParams {
  N_grid: number
  n_points: number
  v_rate: number
  V_max: number | null
}

export interface ISParams {
  N_grid: number
  V_dc: number
  n_freq: number
  f_min: number
  f_max: number
}

export interface DegParams {
  N_grid: number
  V_bias: number
  t_end: number
  n_snapshots: number
}

export interface ProgressEvent {
  stage: string
  current: number
  total: number
  eta_s: number | null
  message: string
}

export interface JobStartResponse {
  status: string
  job_id: string
}

export interface JobStreamHandlers<TResult> {
  onProgress: (ev: ProgressEvent) => void
  onResult: (result: TResult) => void
  onError: (message: string) => void
  onDone: () => void
}

// ── Phase 2b layer builder ──────────────────────────────────────────────────

export interface LayerTemplate {
  role: LayerRole
  optical_material: string | null
  description: string
  source: string
  defaults: Partial<LayerConfig>
}

export interface ValidationIssue {
  layerIdx: number | null   // null = stack-level issue
  field: string | null
  message: string
}

export interface ValidationReport {
  errors: ValidationIssue[]
  warnings: ValidationIssue[]
}

export type StackAction =
  | { type: 'select'; idx: number }
  | { type: 'delete'; idx: number }
  | { type: 'reorder'; from: number; to: number }
  | { type: 'insert'; atIdx: number; layer: LayerConfig }
  | { type: 'edit-interface'; idx: number; pair: readonly [number, number] }

export type Namespace = 'shipped' | 'user'

export interface ConfigEntry {
  name: string
  namespace: Namespace
}

// ── Tandem cell (Phase 3) ────────────────────────────────────────────────────

export interface TandemJunctionLayer {
  name: string
  role: string
  thickness_nm?: number
}

export interface TandemBenchmark {
  V_oc?: number
  J_sc?: number
  FF?: number
  PCE?: number
  source?: string
}

/** Mirrors the POST /api/tandem request body. */
export interface TandemConfigView {
  config_path: string
  N_grid?: number
  n_points?: number
}

/** Mirrors the JSON response from POST /api/tandem.
 * Snake_case keys are intentional — they match the backend response directly. */
export interface TandemJVPayload {
  V_oc: number
  J_sc: number
  FF: number
  PCE: number
  V_top: number
  V_bot: number
  top_layers?: TandemJunctionLayer[]
  bot_layers?: TandemJunctionLayer[]
  benchmark?: TandemBenchmark
}
