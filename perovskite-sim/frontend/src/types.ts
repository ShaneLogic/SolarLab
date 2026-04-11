export interface LayerConfig {
  name: string
  role: string
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
}

export interface DeviceConfig {
  device: {
    V_bi: number
    Phi: number
    interfaces?: Array<[number, number]>
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
