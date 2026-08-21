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
  Nc300?: number
  Nv300?: number
  N_D: number
  N_A: number
  // Optional continuous dopant profile. N_A/N_D are the selected edge values;
  // populated bulk asymptotes activate the profile. Absent means uniform.
  N_A_bulk?: number
  N_D_bulk?: number
  doping_profile_shape?: 'gaussian'
  doping_decay_length?: number
  doping_edge?: 'front' | 'back'
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
  // Stage B(c.2) field-dependent mobility μ(E) — optional, FULL-tier-only.
  // Sentinel 0.0 disables that branch on this layer (Caughey-Thomas: v_sat=0
  // → no saturation; Poole-Frenkel: pf_gamma=0 → no field enhancement).
  // ct_beta_{n,p} default to 2.0 (Canali silicon-electron exponent) when
  // omitted; only meaningful when the corresponding v_sat is non-zero.
  v_sat_n?: number
  v_sat_p?: number
  ct_beta_n?: number
  ct_beta_p?: number
  pf_gamma_n?: number
  pf_gamma_p?: number
  // Continuous bandgap grading — optional, FULL-tier-only. The scalar chi/Eg
  // above are the FRONT endpoints; these are the BACK (far-face) endpoints +
  // profile. A layer is graded iff Eg_back or chi_back is set (and the
  // device-level band_grading flag is on). Absent → uniform layer.
  Eg_back?: number
  chi_back?: number
  grading_profile?: 'linear' | 'parabolic' | 'exponential'
  grading_direction?: 'front_to_back' | 'back_to_front'
  grading_bowing?: number
  grading_char_length?: number
  grading_N_mult?: number
}

export type SimulationModeName = 'legacy' | 'fast' | 'full'
export type BuiltInPotentialMode =
  | 'legacy_manual'
  | 'semiconductor_work_function'
  | 'metal_work_function'

/**
 * Phase E1.8 — SCAPS-style heterojunction interface defect fields. Mirrors
 * the YAML schema parsed by ``perovskite_sim.scaps_compat.loader`` and the
 * ``backend/main.py:stack_from_dict`` plumbing: the 5 fields below feed a
 * single ``InterfaceDefect`` slot on ``DeviceStack.interface_defects[k]``
 * plus the SRV pair on ``DeviceStack.interfaces[k]`` (v = σ·v_th·N_t_areal,
 * cgs→SI). Sentinel discipline matches ``S_*_left/right``:
 *   undefined → field absent in YAML, no defect on this interface
 *   null      → explicitly disabled (also no defect, but persists in JSON)
 *   number    → finite value, included in σ·v_th·N_t computation
 *
 * The backend treats the whole slot as "absent" iff the slot itself is
 * null or undefined. A slot with mixed-null fields is malformed and is
 * rejected at the boundary (every field must be non-null when the slot
 * is populated).
 */
export interface InterfaceDefectFields {
  sigma_n_cm2: number | null
  sigma_p_cm2: number | null
  N_t_cm2: number | null
  v_th_cm_s: number | null
  E_t_eV_below_cb: number | null
  calibration_factor?: number
  iface_state_calibration_factor?: number
}

export interface DeviceConfig {
  simulation_hints?: {
    min_N_grid?: number
    notes?: string
  }
  electrical_grid?: {
    interval_weights?: Record<string, number>
    alphas?: Record<string, number>
  }
  device: {
    /** Deprecated compatibility input used by shipped benchmark presets. */
    V_bi?: number
    /** Explicit manual magnitude, valid only in legacy_manual mode. */
    V_bi_override?: number
    built_in_potential_mode?: BuiltInPotentialMode
    work_function_left_eV?: number
    work_function_right_eV?: number
    Phi: number
    interfaces?: Array<[number, number]>
    /**
     * Phase E1.8 — per-interface SCAPS defect dicts, one slot per internal
     * interface of ``layers`` (``layers.length − 1`` of them). Alignment is
     * FULL-layer, substrate included: on a substrate-prefixed stack k=0 is
     * the glass|HTL boundary and the hetero-interfaces start at k=1 — do NOT
     * assume k=0 is HTL/absorber. Indexing this by the *electrical* interface
     * number is what caused the E10.1 glass regression on the Python side.
     * Each slot is either null (no defect on this heterointerface) or a
     * populated ``InterfaceDefectFields`` object. The backend computes
     * the SRV pair on ``stack.interfaces[k]`` from σ·v_th·N_t_areal and
     * builds ``stack.interface_defects[k] = InterfaceDefect(E_t_eV=…)``;
     * see ``backend/main.py:stack_from_dict``. When this field is
     * present, it takes precedence over the legacy ``interfaces`` SRV
     * pairs on the same index.
     */
    interface_defects?: Array<InterfaceDefectFields | null>
    T?: number
    mode?: SimulationModeName
    // Stage B(c.1) Robin / selective contacts — optional, FULL-tier-only.
    // The naming convention here matches the YAML schema (left = HTL side =
    // 2D top, right = ETL side = 2D bottom). Frontend UI labels these as
    // Top contact (HTL) / Bottom contact (ETL); see active-physics.ts and
    // config-editor.ts for the user-facing translation.
    // - undefined → field absent in YAML, default Dirichlet ohmic contact
    // - null      → explicitly disabled (also Dirichlet, but persists in YAML)
    // - number    → Robin flux with this surface recombination velocity (m/s)
    S_n_left?: number | null
    S_p_left?: number | null
    S_n_right?: number | null
    S_p_right?: number | null
    // SCAPS-validation physics flags — device-level, FULL-tier-only.
    // Mirror load_device_from_yaml / stack_from_dict. Some defaults are on;
    // explicit false values must therefore survive an editor round-trip.
    dos_band_potentials?: boolean
    flat_band_contacts?: boolean
    interface_plane_closure?: boolean
    interface_plane_projection?: boolean
    het_recomb_despike?: number
    // Continuous bandgap grading master switch (device-level, FULL-tier-only).
    // Absent → off → uniform layers (bit-identical). See physics/grading.py.
    band_grading?: boolean
    // Intra-band TFE tunnelling at heterointerfaces (device-level, FULL-tier).
    // Folds a static Padovani-Stratton enhancement into A* at TE-capped faces.
    // Absent → off (bit-identical). tunnel_mass_eff = tunnelling effective
    // mass / m_e (only used when interface_tunneling on). See physics/tunneling.py.
    interface_tunneling?: boolean
    tunnel_mass_eff?: number
    te_physical_norm?: boolean
    ion_steric_diffusion_only?: boolean
    ion_steric_shared_site?: boolean
    autoloop_generated_lever?: boolean
    flat_band_metal_contacts?: boolean
    contact_phi_B_eV?: number
    interface_two_sided?: boolean
    interface_shared_occupancy?: boolean
    interface_plane_generation?: boolean
    jv_solver_policy?: 'general' | 'cancellation_safe_qf_required'
  }
  layers: LayerConfig[]
}

export interface JVMetrics {
  V_oc: number
  J_sc: number
  FF: number
  PCE: number
  /** True iff the J(V) sweep crossed zero. False means V_max stopped
   *  short of V_oc; V_oc / FF / PCE are sentinel zeros and the UI
   *  should warn the user to expand the sweep range. Optional for
   *  back-compat with 1D consumers that pre-date the flag. */
  voc_bracketed?: boolean
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

export interface ComplexNumber {
  real: number
  imag: number
}

export interface ImpedanceProtocol {
  method: 'transient_ion_aware' | 'qf_frequency_ion_free'
  V_dc: number
  delta_V: number
  illuminated: boolean
  dc_settle_time: number | null
  n_cycles: number | null
  n_extract: number | null
  points_per_cycle: number | null
}

export interface ContactThermodynamicCertificate {
  status:
    | 'certified'
    | 'inconsistent'
    | 'compatible_unverified'
    | 'not_assessable'
  built_in_potential_mode: string
  tolerance_eV: number
  fermi_level_span_eV: number | null
  potential_mismatch_V: number | null
  metal_work_function_mismatch_eV: number | null
  contact_quasi_fermi_levels_eV: number[]
  message: string
}

export interface OperatingPointCertificate {
  certified: boolean
  numerically_certified: boolean
  thermodynamically_certified: boolean
  source:
    | 'finite_time_preconditioned'
    | 'dark_equilibrium'
    | 'qf_residual_certified'
  carrier_area_rate_A_m2: number
  ion_area_rate_A_m2: number
  max_ionic_face_current_A_m2: number
  dc_face_current_spread_A_m2: number
  carrier_area_rate_limit_A_m2: number | null
  ion_area_rate_limit_A_m2: number | null
  ionic_face_current_limit_A_m2: number | null
  dc_face_current_spread_limit_A_m2: number | null
  contact_thermodynamics: ContactThermodynamicCertificate
  reasons: string[]
}

export interface IonicTimescale {
  species: 'positive' | 'negative'
  region_start_m: number
  region_end_m: number
  region_length_m: number
  diffusion_coefficient_m2_s: number
  equilibrium_density_m3: number
  debye_length_m: number
  dielectric_frequency_Hz: number
  blocking_charge_frequency_Hz: number
  diffusion_frequency_Hz: number
}

export interface FrequencyWindowAssessment {
  f_min_Hz: number
  f_max_Hz: number
  has_mobile_ions: boolean
  characteristic_frequency_bracketed: boolean | null
  ionic_branch_covered: boolean | null
  ionic_timescales: IonicTimescale[]
  warnings: string[]
}

export interface GridAssessment {
  certified: boolean
  override_used: boolean
  guarded_cell_count: number
  offender_count: number
  max_guarded_cell_debye_ratio: number | null
  max_cell_debye_ratio_limit: number
  warnings: string[]
}

export interface ImpedanceDiagnostics {
  admittance_S_m2: ComplexNumber[] | null
  admittance_faces_S_m2: ComplexNumber[][] | null
  max_relative_face_spread: number[] | null
  reciprocal_condition: number[] | null
  backward_error: number[] | null
  electron_storage_response_F_m2: ComplexNumber[] | null
  hole_storage_response_F_m2: ComplexNumber[] | null
}

export interface ISResult {
  frequencies: number[]
  Z_real: number[]
  Z_imag: number[]
  protocol?: ImpedanceProtocol | null
  operating_point?: OperatingPointCertificate | null
  frequency_window?: FrequencyWindowAssessment | null
  grid_assessment?: GridAssessment | null
  diagnostics?: ImpedanceDiagnostics | null
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
  V_bi_fit: number  // apparent p-n depletion-model value; API name retained
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
  /** Layer 2 of the Phase 6 acceptance follow-up. The backend extracts
   *  V_oc / J_sc / FF / PCE via the centralised ``compute_metrics`` and
   *  reports ``voc_bracketed=false`` when V_max stopped short of V_oc.
   *  Optional for back-compat with payloads from a backend that pre-
   *  dates the field; the renderer falls back to "no metrics" in that
   *  case. ``J_sc`` is in A/m² (same convention as the J array; the
   *  backend has already sign-normalised to J_sc>0). */
  metrics?: JVMetrics
}

/** V_oc(L_g) sweep result — Stage-B headline experiment.
 *  Arrays are aligned: ``V_oc_V[k]`` corresponds to ``grain_sizes_nm[k]``. */
export interface VocGrainSweepResult {
  grain_sizes_nm: number[]
  V_oc_V: number[]
  J_sc_Am2: number[]
  FF: number[]
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
  delta_V?: number
  n_cycles?: number
  n_extract?: number
  points_per_cycle?: number
  dc_settle_time?: number
  illuminated?: boolean
  method?:
    | 'transient'
    | 'transient_ion_aware'
    | 'quasi_fermi_frequency'
    | 'qf_frequency_ion_free'
  require_operating_point_certificate?: boolean
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
