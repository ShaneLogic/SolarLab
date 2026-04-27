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

export interface TPVResult {
  t: number[]
  V: number[]
  J: number[]
  V_oc: number
  tau: number
  delta_V0: number
}

// ── Phase 2 characterisation experiments ────────────────────────────────────

export interface DarkJVResult {
  V: number[]
  J: number[]
  n_ideality: number
  J_0: number
  V_fit_lo: number
  V_fit_hi: number
}

export interface SunsVocResult {
  suns: number[]
  V_oc: number[]
  J_sc: number[]
  J_pseudo_V: number[]
  J_pseudo_J: number[]
  pseudo_FF: number
}

export interface VocTResult {
  T_arr: number[]           // K
  V_oc_arr: number[]        // V
  J_sc_arr: number[]        // A/m^2
  slope: number             // V/K
  intercept_0K: number      // V (≈ E_A in eV)
  E_A_eV: number            // eV
  R_squared: number
}

export interface EQEResult {
  wavelengths_nm: number[]
  EQE: number[]
  J_sc_per_lambda: number[]
  J_sc_integrated: number
  Phi_incident: number
}

export interface ELResult {
  wavelengths_nm: number[]
  EL_spectrum: number[]           // photons / m^2 / s / nm
  absorber_absorptance: number[]  // dimensionless, [0, 1]
  V_inj: number                   // V
  J_inj: number                   // A/m^2 (signed; negative under solar sign convention)
  J_em_rad: number                // A/m^2
  EQE_EL: number                  // [-]
  delta_V_nr_mV: number           // mV
  T: number                       // K
}

export interface MottSchottkyResult {
  V: number[]
  C: number[]
  one_over_C2: number[]
  V_bi_fit: number
  N_eff_fit: number
  V_fit_lo: number
  V_fit_hi: number
  frequency: number
  eps_r_used: number
}

export interface CurrentDecompResult {
  V_fwd: number[]
  V_rev: number[]
  Jn_fwd: number[]
  Jp_fwd: number[]
  Jion_fwd: number[]
  Jdisp_fwd: number[]
  Jtotal_fwd: number[]
  Jn_rev: number[]
  Jp_rev: number[]
  Jion_rev: number[]
  Jdisp_rev: number[]
  Jtotal_rev: number[]
}

export interface SpatialSnapshot {
  x: number[]       // nm
  phi: number[]     // V
  E: number[]       // V/m
  n: number[]       // m^-3
  p: number[]       // m^-3
  P: number[]       // m^-3
  rho: number[]     // C/m^3
  V_app: number
}

export interface SpatialProfileResult {
  V_fwd: number[]
  V_rev: number[]
  snapshots_fwd: SpatialSnapshot[]
  snapshots_rev: SpatialSnapshot[]
}

// ── Stage-A 2D J-V (Phase 6) ────────────────────────────────────────────────

export interface SpatialSnapshot2D {
  V: number
  x: number[]            // nm, length Nx
  y: number[]            // nm, length Ny
  phi: number[][]        // (Ny, Nx)
  n: number[][]          // (Ny, Nx)
  p: number[][]          // (Ny, Nx)
  Jx_n: number[][]       // (Ny, Nx-1)
  Jy_n: number[][]       // (Ny-1, Nx)
  Jx_p: number[][]       // (Ny, Nx-1)
  Jy_p: number[][]       // (Ny-1, Nx)
}

export interface JV2DResult {
  V: number[]                       // applied bias, V
  J: number[]                       // terminal current density, A/m^2
  grid_x: number[]                  // lateral nodes, nm
  grid_y: number[]                  // vertical nodes, nm
  lateral_bc: 'periodic' | 'neumann'
  snapshots: SpatialSnapshot2D[]    // empty when save_snapshots=false
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
  device_type?: string
  // Phase 2 tier gate: list of physics tiers this preset runs correctly
  // under. Legacy configs with chi=Eg=0 can only run legacy/fast (FULL
  // collapses compute_V_bi). Optional for backwards compatibility with
  // older backend snapshots; callers should treat missing as ['legacy', 'fast'].
  tier_compat?: ReadonlyArray<'legacy' | 'fast' | 'full'>
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

export interface TandemJVMetrics {
  V_oc: number
  J_sc: number
  FF: number
  PCE: number
}

/** Mirrors the JSON response from POST /api/tandem.
 * Snake_case keys are intentional — they match the backend response directly. */
export interface TandemJVPayload {
  V: number[]
  J: number[]
  V_top: number[]
  V_bot: number[]
  metrics: TandemJVMetrics
  benchmark: TandemBenchmark | null
  top_layers?: TandemJunctionLayer[]
  bot_layers?: TandemJunctionLayer[]
}
