export interface IodinePresetSummary {
  id: string
  name: string
  description: string
  reference: string
  source_id: string
  sha256: string
  deck: Record<string, unknown>
}

export interface IodinePresetCatalog {
  schema: 'solarlab.iodine-presets.v1'
  presets: IodinePresetSummary[]
  scientific_acceptance: 'not_evaluated'
  scientific_qualification: false
}

export interface IodineStateProfiles {
  electrical_nodes_nm: number[]
  chem_nodes_nm: number[]
  potential_V: number[]
  electron_density_m3: number[]
  hole_density_m3: number[]
  iodide_m3: number[]
  iodine_atom_m3: number[]
  iodine_molecule_m3: number[]
  species_inventories_m2: Record<string, number>
  total_iodine_atoms_m2: number
}

export interface IodineFrozenJV {
  voltages_V: number[]
  output_current_A_m2: number[]
  metrics: Record<string, number | null>
}

export interface IodineSteadyResponse {
  schema: 'solarlab.iodine-steady-result.v1'
  deck_id: string
  deck_name: string
  device_sha256: string
  domain_mode: string
  operating_state_mode: string
  voltage_V: number
  generation_scale: number
  profiles: IodineStateProfiles
  frozen_jv: IodineFrozenJV | null
  wall_s: number
  scientific_acceptance: 'not_evaluated'
  scientific_qualification: false
}

export interface IodineImpedanceSweep {
  chem_mode: 'mobile' | 'frozen'
  frequencies_Hz: number[]
  admittance_real_S_m2: number[]
  admittance_imag_S_m2: number[]
  impedance_real_Ohm_m2: number[]
  impedance_imag_Ohm_m2: number[]
  apparent_capacitance_F_m2: number[]
  max_maxwell_spread_A_m2_V: number
  max_backward_error: number
}

export interface IodineImpedanceResponse {
  schema: 'solarlab.iodine-impedance-result.v1'
  deck_id: string
  deck_name: string
  device_sha256: string
  domain_mode: string
  operating_state_mode: string
  voltage_V: number
  generation_scale: number
  frequencies_Hz: number[]
  sweeps: Record<string, IodineImpedanceSweep>
  redox_excess_capacitance_F_m2: number[] | null
  profiles: IodineStateProfiles
  wall_s: number
  scientific_acceptance: 'not_evaluated'
  scientific_qualification: false
}

export interface IodineAgingCheckpoint {
  time_s: number
  stage_name: string
  in_device_iodine_m2: number
  cumulative_escaped_iodine_atoms_m2: number
  cumulative_immobilized_iodine_atoms_m2: number
  conservation_residual_m2: number
  profiles: IodineStateProfiles
  frozen_jv: IodineFrozenJV | null
  dynamic_jv: Record<string, unknown> | null
}

export interface IodineAgingResponse {
  schema: 'solarlab.iodine-aging-result.v1'
  deck_id: string
  deck_name: string
  protocol_sha256: string
  device_sha256: string
  domain_mode: string
  reference_iodine_inventory_m2: number
  max_relative_conservation_error: number
  accepted_bdf_steps: number
  completed: boolean
  wall_s: number
  trajectory: {
    times_s: number[]
    in_device_iodine_m2: number[]
    cumulative_escaped_iodine_atoms_m2: number[]
    cumulative_immobilized_iodine_atoms_m2: number[]
  }
  checkpoints: IodineAgingCheckpoint[]
  persisted_run_store: { store_id: string; run_id: string; attempt_id: string } | null
  scientific_acceptance: 'not_evaluated'
  scientific_qualification: false
}

export interface IodineOverrides {
  intervals_per_layer?: number | number[]
  k_a_cm3_s?: number
  mu_bl_cm2_V_s?: number
  ebl_partition?: number
  volumetric_escape_s_1?: number
  contact_corrosion_m_s?: number
  voltage_V?: number
  generation_scale?: number
  compute_frozen_jv?: boolean
  f_min_Hz?: number
  f_max_Hz?: number
  n_frequencies?: number
  chem_mode?: 'mobile' | 'frozen' | 'both'
  probe_mode?: 'none' | 'frozen_chemistry_jv' | 'dynamic_jv_scan' | 'both'
}

export interface IodineRunRequest {
  preset?: string
  deck?: Record<string, unknown>
  overrides?: IodineOverrides
  persist_to_store?: boolean
  run_id?: string
}

function assertRecord(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be a JSON object`)
  }
  return value as Record<string, unknown>
}

export class IodineAgingClient {
  private readonly base: URL
  private readonly fetchImpl: typeof fetch

  constructor(endpoint: string | URL, fetchImpl: typeof fetch = globalThis.fetch.bind(globalThis)) {
    this.base = new URL(String(endpoint))
    if (!this.base.pathname.endsWith('/')) this.base.pathname += '/'
    this.fetchImpl = fetchImpl
  }

  private url(path: string): string {
    return new URL(path.replace(/^\//, ''), this.base).toString()
  }

  private async requestJson(path: string, init: RequestInit): Promise<Record<string, unknown>> {
    const resp = await this.fetchImpl(this.url(path), init)
    const text = await resp.text()
    let parsed: unknown = null
    try {
      parsed = text ? JSON.parse(text) : null
    } catch {
      parsed = null
    }
    if (!resp.ok) {
      const detail = parsed && typeof parsed === 'object' && 'detail' in parsed
        ? String((parsed as Record<string, unknown>).detail)
        : `HTTP ${resp.status}`
      throw new Error(detail)
    }
    return assertRecord(parsed, path)
  }

  async listPresets(signal?: AbortSignal): Promise<IodinePresetCatalog> {
    const doc = await this.requestJson('iodine/presets', { method: 'GET', signal })
    if (doc.schema !== 'solarlab.iodine-presets.v1' || doc.scientific_qualification !== false) {
      throw new Error('Invalid iodine preset catalog schema or qualification flag')
    }
    return doc as unknown as IodinePresetCatalog
  }

  async runSteady(request: IodineRunRequest, signal?: AbortSignal): Promise<IodineSteadyResponse> {
    const doc = await this.requestJson('iodine/steady', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    })
    if (doc.schema !== 'solarlab.iodine-steady-result.v1' || doc.scientific_qualification !== false) {
      throw new Error('Invalid iodine steady response schema or qualification flag')
    }
    return doc as unknown as IodineSteadyResponse
  }

  async runImpedance(request: IodineRunRequest, signal?: AbortSignal): Promise<IodineImpedanceResponse> {
    const doc = await this.requestJson('iodine/impedance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    })
    if (doc.schema !== 'solarlab.iodine-impedance-result.v1' || doc.scientific_qualification !== false) {
      throw new Error('Invalid iodine impedance response schema or qualification flag')
    }
    return doc as unknown as IodineImpedanceResponse
  }

  async runAging(request: IodineRunRequest, signal?: AbortSignal): Promise<IodineAgingResponse> {
    const doc = await this.requestJson('iodine/aging', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    })
    if (doc.schema !== 'solarlab.iodine-aging-result.v1' || doc.scientific_qualification !== false) {
      throw new Error('Invalid iodine aging response schema or qualification flag')
    }
    return doc as unknown as IodineAgingResponse
  }
}
