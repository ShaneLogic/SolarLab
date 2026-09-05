import type {
  BulkDefectChargeTransition,
  BulkDefectDistributionKind,
  BulkDefectNeutralReference,
  BulkDefectSchemaVersion,
  BulkDefectSpecies,
  BulkDefectSpatialKnot,
  BulkDefectSpatialProfile,
  BulkDefectWidthConvention,
  BuiltInPotentialMode,
  DeviceConfig,
  InterfaceDefectFields,
  LayerConfig,
  LayerRole,
  SimulationModeName,
} from './types'
import { isLayerRole } from './types'
import { isFieldVisible } from './workstation/tier-gating'

const MODE_OPTIONS: ReadonlyArray<{ value: SimulationModeName; label: string }> = [
  { value: 'full', label: 'Full (all physics upgrades)' },
  { value: 'fast', label: 'Fast (build-once physics)' },
  { value: 'legacy', label: 'Legacy (IonMonger-compatible)' },
]

const BUILT_IN_POTENTIAL_OPTIONS: ReadonlyArray<{
  value: BuiltInPotentialMode
  label: string
}> = [
  { value: 'semiconductor_work_function', label: 'Semiconductor work functions' },
  { value: 'metal_work_function', label: 'Explicit metal work functions' },
  { value: 'legacy_manual', label: 'Legacy manual override' },
]

const EXPLICIT_DEFECT_SCHEMA_V1 = 'solarlab-explicit-bulk-defects-v1' as const
const EXPLICIT_DEFECT_SCHEMA_V2 = 'solarlab-explicit-bulk-defects-v2' as const
const EXPLICIT_DEFECT_SCHEMA_V3 = 'solarlab-explicit-bulk-defects-v3' as const
const EXPLICIT_DEFECT_SCHEMA_VERSIONS = new Set<BulkDefectSchemaVersion>([
  EXPLICIT_DEFECT_SCHEMA_V1,
  EXPLICIT_DEFECT_SCHEMA_V2,
  EXPLICIT_DEFECT_SCHEMA_V3,
])
const DEFECT_DISTRIBUTION_LABELS: ReadonlyArray<{
  value: BulkDefectDistributionKind
  label: string
}> = [
  { value: 'single_level', label: 'Single level' },
  { value: 'gaussian', label: 'Gaussian' },
  { value: 'uniform', label: 'Uniform' },
  { value: 'conduction_band_tail', label: 'Conduction-band tail' },
  { value: 'valence_band_tail', label: 'Valence-band tail' },
]
const DEFECT_NEUTRAL_REFERENCE: Record<
  Exclude<BulkDefectChargeTransition, 'unresolved'>,
  Exclude<BulkDefectNeutralReference, 'unresolved'>
> = {
  neutral: 'all_occupancies',
  acceptor: 'empty',
  donor: 'filled',
}

type DefectDisplayUnit = 'si' | 'scaps_cgs'
type DefectDimension = 'density' | 'cross_section' | 'velocity'

const DEFECT_DISPLAY_FACTORS: Record<
  DefectDisplayUnit,
  Record<DefectDimension, number>
> = {
  si: { density: 1, cross_section: 1, velocity: 1 },
  scaps_cgs: { density: 1e-6, cross_section: 1e4, velocity: 1e2 },
}

function isModeName(v: unknown): v is SimulationModeName {
  return v === 'full' || v === 'fast' || v === 'legacy'
}

function isBuiltInPotentialMode(v: unknown): v is BuiltInPotentialMode {
  return v === 'legacy_manual'
    || v === 'semiconductor_work_function'
    || v === 'metal_work_function'
}

// Discriminator for per-layer field rendering. Most parameters are numeric;
// optical_material is a select populated from the backend's n,k CSV scan, and
// incoherent is a boolean checkbox (thick substrate Fresnel handling in TMM).
type FieldKind = 'numeric' | 'numeric-optional' | 'select' | 'select-optical-material' | 'boolean'

interface FieldDef {
  key: keyof LayerConfig
  label: string
  kind: FieldKind
  unit?: string
  /** Hover tooltip — short physical-meaning explainer. */
  tooltip?: string
  /** Placeholder hint for an empty numeric input — used to convey the
   *  "0.0 / disabled" sentinel for opt-in physics fields. */
  placeholder?: string
  /** Option list for ``kind: 'select'`` (generic string-enum dropdown). */
  options?: ReadonlyArray<string>
}

// Groups of parameters for a single layer. Grouping makes long forms scannable.
interface ParamGroup {
  title: string
  fields: FieldDef[]
  /** When true, render the group inside a collapsed-by-default
   *  ``<details>`` so the form stays scannable. Used by the
   *  "Advanced 2D Physics" group (FULL-tier-only μ(E) fields). */
  collapsed?: boolean
}

const LAYER_GROUPS: ParamGroup[] = [
  {
    title: 'Geometry & Electrostatics',
    fields: [
      { key: 'thickness', label: 'Thickness', kind: 'numeric', unit: 'm' },
      { key: 'eps_r', label: '<i>ε</i><sub>r</sub>', kind: 'numeric', unit: '' },
      { key: 'chi', label: '<i>χ</i>', kind: 'numeric', unit: 'eV' },
      { key: 'Eg', label: '<i>E</i><sub>g</sub>', kind: 'numeric', unit: 'eV' },
    ],
  },
  {
    title: 'Transport',
    fields: [
      { key: 'mu_n', label: '<i>μ</i><sub>n</sub>', kind: 'numeric', unit: 'm²/(V·s)' },
      { key: 'mu_p', label: '<i>μ</i><sub>p</sub>', kind: 'numeric', unit: 'm²/(V·s)' },
      { key: 'ni', label: '<i>n</i><sub>i</sub>', kind: 'numeric', unit: 'm⁻³' },
      {
        key: 'Nc300', label: '<i>N</i><sub>C,300</sub>', kind: 'numeric-optional', unit: 'm⁻³',
        placeholder: 'required for semiconductor ΔW',
        tooltip: 'Effective conduction-band density of states at 300 K.',
      },
      {
        key: 'Nv300', label: '<i>N</i><sub>V,300</sub>', kind: 'numeric-optional', unit: 'm⁻³',
        placeholder: 'required for semiconductor ΔW',
        tooltip: 'Effective valence-band density of states at 300 K.',
      },
      { key: 'N_D', label: '<i>N</i><sub>D</sub>', kind: 'numeric', unit: 'm⁻³' },
      { key: 'N_A', label: '<i>N</i><sub>A</sub>', kind: 'numeric', unit: 'm⁻³' },
    ],
  },
  {
    title: 'Recombination',
    fields: [
      { key: 'tau_n', label: '<i>τ</i><sub>n</sub>', kind: 'numeric', unit: 's' },
      { key: 'tau_p', label: '<i>τ</i><sub>p</sub>', kind: 'numeric', unit: 's' },
      { key: 'n1', label: '<i>n</i><sub>1</sub>', kind: 'numeric', unit: 'm⁻³' },
      { key: 'p1', label: '<i>p</i><sub>1</sub>', kind: 'numeric', unit: 'm⁻³' },
      { key: 'B_rad', label: '<i>B</i><sub>rad</sub>', kind: 'numeric', unit: 'm³/s' },
      { key: 'C_n', label: '<i>C</i><sub>n</sub>', kind: 'numeric', unit: 'm⁶/s' },
      { key: 'C_p', label: '<i>C</i><sub>p</sub>', kind: 'numeric', unit: 'm⁶/s' },
    ],
  },
  {
    title: 'Ions & Optics',
    fields: [
      {
        key: 'D_ion', label: '<i>D</i><sub>c</sub>', kind: 'numeric', unit: 'm²/s',
        tooltip: 'Diffusion coefficient of the positive mobile ionic species c at 300 K. Positive-ion migration requires both D_c > 0 and an available positive-ion population. Set to 0 to freeze this species.',
      },
      {
        key: 'P0', label: '<i>c</i><sub>0</sub>', kind: 'numeric', unit: 'm⁻³',
        tooltip: 'Initial reference concentration of positive mobile ions. The layer starts with c = c₀; its ionic space-charge contribution is +q(c − c₀), including the fixed compensating background. This uniform initial profile need not be ion-relaxed equilibrium. 1e18 cm⁻³ = 1e24 m⁻³.',
      },
      {
        key: 'P_lim', label: '<i>c</i><sub>lim</sub>', kind: 'numeric', unit: 'm⁻³',
        tooltip: 'Finite-site concentration scale for positive-ion crowding. Single-species occupancy is c/c_lim; shared-site occupancy is (c + a)/c_lim. For a common site reservoir, use c_lim = a_lim and c₀ + a₀ < c_lim. This is a site-density scale, not the initial ion concentration.',
      },
      {
        key: 'D_ion_neg', label: '<i>D</i><sub>a</sub>',
        kind: 'numeric-optional', unit: 'm²/s', placeholder: 'empty — single species',
        tooltip: 'Diffusion coefficient of the negative mobile ionic species a at 300 K. Its drift is reversed relative to c. Set D_a > 0 and a₀ > 0 to include mobile negative ions. Leave empty for single-species transport. Runs in Fast and Full tiers; Legacy forces it off.',
      },
      {
        key: 'P0_neg', label: '<i>a</i><sub>0</sub>',
        kind: 'numeric-optional', unit: 'm⁻³', placeholder: 'empty — none',
        tooltip: 'Initial reference concentration of negative mobile ions. The layer starts with a = a₀; its ionic space-charge contribution is −q(a − a₀), including the fixed compensating background. a₀ need not equal c₀. Also set D_a > 0 in Fast or Full to activate this species. 1e18 cm⁻³ = 1e24 m⁻³.',
      },
      {
        key: 'P_lim_neg', label: '<i>a</i><sub>lim</sub>',
        kind: 'numeric-optional', unit: 'm⁻³', placeholder: 'empty — 1e30 (no limit)',
        tooltip: 'Finite-site concentration scale for negative-ion crowding. Independent-site occupancy is a/a_lim; shared-site occupancy is (c + a)/a_lim. For a common site reservoir, set a_lim = c_lim. Leaving it empty gives 1e30, making crowding negligible at ordinary ion concentrations.',
      },
      {
        key: 'E_a_ion', label: '<i>E</i><sub>a,ion</sub>',
        kind: 'numeric-optional', unit: 'eV', placeholder: 'empty — 0.58',
        tooltip: 'Arrhenius activation energy for the ionic diffusivity: D_c(T) = D_c,300·exp[−E_a/k_B·(1/T − 1/300)], with the same scaling for D_a. One activation energy covers both species. An explicit 0 removes the temperature dependence. Applies in Fast and Full; Legacy pins T to 300 K and ignores it.',
      },
      { key: 'alpha', label: '<i>α</i>', kind: 'numeric', unit: 'm⁻¹' },
      { key: 'optical_material', label: 'Optical material', kind: 'select-optical-material' },
      { key: 'incoherent', label: 'Incoherent layer', kind: 'boolean' },
    ],
  },
  // Stage B(c.2) field-dependent mobility μ(E). Hidden under FAST/LEGACY
  // by tier-gating.ts (FULL only). Collapsed by default; sentinel "0"
  // disables the corresponding model on this layer.
  {
    title: 'Advanced 2D Physics — Field-dependent mobility μ(E)',
    collapsed: true,
    fields: [
      {
        key: 'v_sat_n', label: '<i>v</i><sub>sat</sub><sup>n</sup>',
        kind: 'numeric', unit: 'm/s', placeholder: '0 — disabled',
        tooltip: 'Caughey-Thomas saturation velocity for electrons. Caps drift mobility under high field. Typical perovskite ~1e5 m/s. Set 0 to disable.',
      },
      {
        key: 'v_sat_p', label: '<i>v</i><sub>sat</sub><sup>p</sup>',
        kind: 'numeric', unit: 'm/s', placeholder: '0 — disabled',
        tooltip: 'Caughey-Thomas saturation velocity for holes. Set 0 to disable.',
      },
      {
        key: 'ct_beta_n', label: '<i>β</i><sup>n</sup>',
        kind: 'numeric', unit: '', placeholder: '2.0',
        tooltip: 'Caughey-Thomas exponent for electrons. β=2 (Canali silicon-electron form) is the safe default; β=1 (Thornber form) for silicon holes. Only meaningful when v_sat is non-zero.',
      },
      {
        key: 'ct_beta_p', label: '<i>β</i><sup>p</sup>',
        kind: 'numeric', unit: '', placeholder: '2.0',
        tooltip: 'Caughey-Thomas exponent for holes. Only meaningful when v_sat is non-zero.',
      },
      {
        key: 'pf_gamma_n', label: '<i>γ</i><sub>PF</sub><sup>n</sup>',
        kind: 'numeric', unit: '(V/m)^-0.5', placeholder: '0 — disabled',
        tooltip: 'Poole-Frenkel coefficient for electrons. Field-assisted hopping; typical disordered HTL ~3e-4 (V/m)^-0.5. Set 0 to disable.',
      },
      {
        key: 'pf_gamma_p', label: '<i>γ</i><sub>PF</sub><sup>p</sup>',
        kind: 'numeric', unit: '(V/m)^-0.5', placeholder: '0 — disabled',
        tooltip: 'Poole-Frenkel coefficient for holes. Set 0 to disable.',
      },
    ],
  },
  // Continuous bandgap grading. FULL-tier-only (tier-gating GRADING_KEYS);
  // takes effect only when the device "Bandgap grading" flag is on AND the
  // layer sets a back endpoint. The front endpoints are the scalar χ / E_g
  // in the Geometry group; these are the back-face endpoints + profile.
  // Empty Eg_back/chi_back → uniform layer (numeric-optional omits the key).
  {
    title: 'Bandgap grading (front = χ / E_g above)',
    collapsed: true,
    fields: [
      {
        key: 'Eg_back', label: '<i>E</i><sub>g</sub><sup>back</sup>',
        kind: 'numeric-optional', unit: 'eV', placeholder: 'uniform',
        tooltip: 'Band gap at the back face. Set (with the device Bandgap-grading flag) to grade E_g front→back via the SCAPS material law. Empty = uniform layer.',
      },
      {
        key: 'chi_back', label: '<i>χ</i><sup>back</sup>',
        kind: 'numeric-optional', unit: 'eV', placeholder: 'uniform',
        tooltip: 'Electron affinity at the back face. chi_back < chi raises E_C toward the back (electron back-surface field). Empty = uniform.',
      },
      {
        key: 'grading_profile', label: 'profile', kind: 'select',
        options: ['linear', 'parabolic', 'exponential'],
        tooltip: 'Composition profile y(x): linear, parabolic, or exponential (notch).',
      },
      {
        key: 'grading_direction', label: 'direction', kind: 'select',
        options: ['front_to_back', 'back_to_front'],
        tooltip: 'Which face the grade runs toward (back_to_front flips y).',
      },
      {
        key: 'grading_bowing', label: '<i>b</i>',
        kind: 'numeric-optional', unit: 'eV', placeholder: '0',
        tooltip: 'Alloy bowing in E_g(y) = (1-y)·E_front + y·E_back − b·y(1-y). 0 = linear.',
      },
      {
        key: 'grading_char_length', label: '<i>L</i><sub>grade</sub>',
        kind: 'numeric-optional', unit: 'm', placeholder: 'linear',
        tooltip: 'Characteristic length for the exponential (notch) profile. Unused for linear/parabolic.',
      },
      {
        key: 'grading_N_mult', label: '<i>N</i><sub>mult</sub>',
        kind: 'numeric-optional', unit: '', placeholder: '1',
        tooltip: 'Mesh refinement factor for this layer — raise (2–4) for steep notches.',
      },
    ],
  },
]

// Module-level cache for optical-material option list. Populated once per
// mount via setOpticalMaterialOptions() — see device-panel.ts. If the fetch
// fails the cache stays empty and the dropdown simply has no options beyond
// the "(none — Beer-Lambert)" sentinel.
let opticalMaterialOptions: ReadonlyArray<string> = []

export function setOpticalMaterialOptions(options: ReadonlyArray<string>): void {
  opticalMaterialOptions = [...options]
}

// Minimal HTML escape for untrusted-ish string values (material filenames
// come from backend auto-scan of on-disk CSVs — trusted but defensive).
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function fmt(v: unknown): string {
  if (v === undefined || v === null || v === '') return ''
  const n = typeof v === 'number' ? v : Number(v)
  if (!Number.isFinite(n)) return ''
  if (n === 0) return '0'
  const abs = Math.abs(n)
  if (abs >= 1e-3 && abs < 1e4) return String(n)
  return n.toExponential(3)
}

interface NumAttrOpts {
  /** Hint shown when the input is empty — used for "0 / disabled" sentinels. */
  placeholder?: string
  /** Native HTML5 tooltip surfaced on hover. */
  title?: string
}

function numAttr(id: string, value: unknown, opts?: NumAttrOpts): string {
  const placeholderAttr = opts?.placeholder ? ` placeholder="${escapeHtml(opts.placeholder)}"` : ''
  const titleAttr = opts?.title ? ` title="${escapeHtml(opts.title)}"` : ''
  return `<input type="text" class="num-input" id="${id}" value="${fmt(value)}" spellcheck="false"${placeholderAttr}${titleAttr}>`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function hasExactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort()
  const wanted = [...expected].sort()
  return actual.length === wanted.length && actual.every((key, index) => key === wanted[index])
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function minimumLayerBandGap(layer: LayerConfig, bandGradingActive: boolean): number | null {
  if (!isFiniteNumber(layer.Eg) || layer.Eg <= 0) return null
  const front = layer.Eg
  if (!bandGradingActive) return front
  const back = isFiniteNumber(layer.Eg_back) && layer.Eg_back > 0 ? layer.Eg_back : front
  const bowing = isFiniteNumber(layer.grading_bowing) ? layer.grading_bowing : 0
  const candidates = [front, back]
  if (bowing > 0) {
    const stationary = (front + bowing - back) / (2 * bowing)
    if (stationary > 0 && stationary < 1) {
      candidates.push(
        (1 - stationary) * front
          + stationary * back
          - bowing * stationary * (1 - stationary),
      )
    }
  }
  return Math.min(...candidates)
}

function spatialProfileUnsupportedReason(value: unknown): string | null {
  if (!isRecord(value) || !hasExactKeys(value, [
    'coordinate', 'interpolation', 'density_normalization', 'knots',
  ])) return 'spatial profile schema is not canonical'
  if (
    value.coordinate !== 'normalized_layer_coordinate'
    || value.interpolation !== 'piecewise_linear'
    || value.density_normalization !== 'layer_average_unity'
    || !Array.isArray(value.knots)
    || value.knots.length < 2
  ) return 'spatial profile convention is not editable'
  const knots: BulkDefectSpatialKnot[] = []
  for (const knot of value.knots) {
    if (!isRecord(knot) || !hasExactKeys(knot, [
      'position_fraction', 'density_multiplier',
    ])) return 'spatial profile knot schema is not canonical'
    if (
      !isFiniteNumber(knot.position_fraction)
      || knot.position_fraction < 0
      || knot.position_fraction > 1
      || !isFiniteNumber(knot.density_multiplier)
      || knot.density_multiplier <= 0
    ) return 'spatial profile knot values are invalid'
    knots.push({
      position_fraction: knot.position_fraction,
      density_multiplier: knot.density_multiplier,
    })
  }
  if (knots[0].position_fraction !== 0 || knots.at(-1)?.position_fraction !== 1) {
    return 'spatial profile requires exact 0 and 1 endpoints'
  }
  if (knots.some((knot, index) => (
    index > 0 && knot.position_fraction <= knots[index - 1].position_fraction
  ))) return 'spatial profile positions are not strictly increasing'
  const integral = knots.slice(1).reduce((sum, right, index) => {
    const left = knots[index]
    return sum + 0.5
      * (left.density_multiplier + right.density_multiplier)
      * (right.position_fraction - left.position_fraction)
  }, 0)
  if (Math.abs(integral - 1) > 1e-12) {
    return 'spatial profile layer average is not unity'
  }
  return null
}

function distributionUnsupportedReason(
  value: unknown,
  schema: BulkDefectSchemaVersion,
  layer: LayerConfig,
  bandGradingActive: boolean,
): string | null {
  if (!isRecord(value)) return 'distribution is not a mapping'
  const kind = value.kind
  if (!DEFECT_DISTRIBUTION_LABELS.some(option => option.value === kind)) {
    return 'distribution kind is unsupported'
  }
  if (schema === EXPLICIT_DEFECT_SCHEMA_V1 && kind !== 'single_level') {
    return 'v1 distributed metadata remains read-only'
  }
  const expected = [
    'kind', 'normalization', 'total_density_m3', 'center_eV_above_vb',
  ]
  if (schema !== EXPLICIT_DEFECT_SCHEMA_V1) expected.push('energy_reference')
  if (kind !== 'single_level') expected.push('width_eV', 'width_convention')
  if (
    kind === 'gaussian'
    || kind === 'conduction_band_tail'
    || kind === 'valence_band_tail'
  ) expected.push('support_width_multiplier')
  if (!hasExactKeys(value, expected)) return 'distribution schema is not canonical'
  if (
    value.normalization !== 'integrated_total'
    || !isFiniteNumber(value.total_density_m3)
    || value.total_density_m3 <= 0
    || !isFiniteNumber(value.center_eV_above_vb)
    || value.center_eV_above_vb < 0
  ) return 'distribution density or energy is invalid'
  if (
    schema !== EXPLICIT_DEFECT_SCHEMA_V1
    && value.energy_reference !== 'above_valence_band'
  ) return 'energy reference is unresolved'
  if (kind === 'uniform' && value.width_convention !== 'uniform_full_width') {
    return 'uniform width convention is unresolved'
  }
  if (
    (kind === 'conduction_band_tail' || kind === 'valence_band_tail')
    && value.width_convention !== 'scaps_characteristic_energy'
  ) return 'band-tail width convention is unresolved'
  if (
    kind === 'gaussian'
    && value.width_convention !== 'gaussian_standard_deviation'
    && value.width_convention !== 'scaps_characteristic_energy'
  ) return 'Gaussian width convention is unresolved'
  if (kind !== 'single_level' && (!isFiniteNumber(value.width_eV) || value.width_eV <= 0)) {
    return 'distribution width is invalid'
  }
  const hasFiniteSupport = kind === 'gaussian'
    || kind === 'conduction_band_tail'
    || kind === 'valence_band_tail'
  if (
    hasFiniteSupport
    && (!isFiniteNumber(value.support_width_multiplier) || value.support_width_multiplier <= 0)
  ) return 'distribution support multiplier is invalid'

  const gap = minimumLayerBandGap(layer, bandGradingActive)
  if (gap === null) return null
  const center = value.center_eV_above_vb
  const width = isFiniteNumber(value.width_eV) ? value.width_eV : 0
  const supportMultiplier = isFiniteNumber(value.support_width_multiplier)
    ? value.support_width_multiplier
    : 0
  if (center > gap) return 'distribution center lies outside the local band gap'
  let lower = center
  let upper = center
  if (kind === 'uniform') {
    lower -= width / 2
    upper += width / 2
  } else if (kind === 'gaussian') {
    const halfWidth = 0.5 * supportMultiplier * width
    lower -= halfWidth
    upper += halfWidth
  } else if (kind === 'conduction_band_tail') {
    lower -= supportMultiplier * width
  } else if (kind === 'valence_band_tail') {
    upper += supportMultiplier * width
  }
  if (lower < -1e-14 || upper > gap + 1e-14) {
    return 'distribution support lies outside the local band gap'
  }
  return null
}

function bulkDefectEditorUnsupportedReason(
  layer: LayerConfig,
  bandGradingActive: boolean,
): string | null {
  const raw = layer as unknown as Record<string, unknown>
  const keys = ['defect_schema_version', 'defect_model', 'bulk_defects'] as const
  const present = keys.filter(key => raw[key] !== undefined)
  if (present.length === 0) return null
  if (present.length !== keys.length) return 'incomplete versioned defect document'
  if (!EXPLICIT_DEFECT_SCHEMA_VERSIONS.has(raw.defect_schema_version as BulkDefectSchemaVersion)) {
    return 'unsupported schema version'
  }
  const schema = raw.defect_schema_version as BulkDefectSchemaVersion
  if (raw.defect_model !== 'effective_lifetime' && raw.defect_model !== 'explicit_quasi_steady') {
    return 'unsupported defect model'
  }
  if (!Array.isArray(raw.bulk_defects)) return 'bulk_defects is not an array'
  if (raw.defect_model === 'explicit_quasi_steady' && raw.bulk_defects.length === 0) {
    return 'explicit model has no species'
  }
  let hasSpatialProfile = false
  const names: string[] = []
  for (const [index, value] of raw.bulk_defects.entries()) {
    if (!isRecord(value)) return `species ${index + 1} is not a mapping`
    const hasProfile = Object.prototype.hasOwnProperty.call(value, 'spatial_profile')
    const expectedSpeciesKeys = [
      'name', 'distribution', 'charge_transition', 'neutral_reference', 'kinetics', 'degeneracy',
      ...(hasProfile ? ['spatial_profile'] : []),
    ]
    if (!hasExactKeys(value, expectedSpeciesKeys)) {
      return `species ${index + 1} has unsupported metadata`
    }
    if (schema !== EXPLICIT_DEFECT_SCHEMA_V3 && hasProfile) {
      return `species ${index + 1} spatial profile is not allowed by the schema`
    }
    if (hasProfile) {
      const profileReason = spatialProfileUnsupportedReason(value.spatial_profile)
      if (profileReason) return `species ${index + 1} ${profileReason}`
      hasSpatialProfile = true
    }
    const distributionReason = distributionUnsupportedReason(
      value.distribution,
      schema,
      layer,
      bandGradingActive,
    )
    if (distributionReason) return `species ${index + 1} ${distributionReason}`
    if (!isRecord(value.kinetics) || !hasExactKeys(value.kinetics, [
      'sigma_n_m2', 'sigma_p_m2', 'thermal_velocity_n_m_s', 'thermal_velocity_p_m_s',
    ])) return `species ${index + 1} kinetics has unsupported metadata`
    if (
      !isFiniteNumber(value.kinetics.sigma_n_m2)
      || value.kinetics.sigma_n_m2 < 0
      || !isFiniteNumber(value.kinetics.sigma_p_m2)
      || value.kinetics.sigma_p_m2 < 0
      || !isFiniteNumber(value.kinetics.thermal_velocity_n_m_s)
      || value.kinetics.thermal_velocity_n_m_s <= 0
      || !isFiniteNumber(value.kinetics.thermal_velocity_p_m_s)
      || value.kinetics.thermal_velocity_p_m_s <= 0
      || !isFiniteNumber(value.degeneracy)
      || value.degeneracy <= 0
    ) return `species ${index + 1} kinetics or degeneracy is invalid`
    if (
      value.charge_transition !== 'neutral'
      && value.charge_transition !== 'acceptor'
      && value.charge_transition !== 'donor'
    ) return `species ${index + 1} charge transition is unresolved`
    const expectedReference = DEFECT_NEUTRAL_REFERENCE[value.charge_transition]
    if (value.neutral_reference !== expectedReference) {
      return `species ${index + 1} neutral reference is inconsistent`
    }
    if (
      raw.defect_model === 'explicit_quasi_steady'
      && (typeof value.name !== 'string' || value.name.trim() === '')
    ) return `species ${index + 1} requires a name`
    if (schema === EXPLICIT_DEFECT_SCHEMA_V3 && (
      typeof value.name !== 'string' || value.name.trim() === ''
    )) return `species ${index + 1} requires a name in v3`
    if (value.name !== null && typeof value.name !== 'string') {
      return `species ${index + 1} name is invalid`
    }
    if (typeof value.name === 'string') names.push(value.name.trim())
  }
  if (new Set(names).size !== names.length) return 'defect species names are not unique'
  if (schema === EXPLICIT_DEFECT_SCHEMA_V3 && !hasSpatialProfile) {
    return 'v3 document has no spatial profile'
  }
  return null
}

function defaultBulkDefectSpecies(layer: LayerConfig, index: number): BulkDefectSpecies {
  const gap = typeof layer.Eg === 'number' && Number.isFinite(layer.Eg) && layer.Eg > 0
    ? layer.Eg
    : 1.5
  return {
    name: `defect_${index + 1}`,
    distribution: {
      kind: 'single_level',
      normalization: 'integrated_total',
      total_density_m3: 1e21,
      center_eV_above_vb: gap / 2,
    },
    charge_transition: 'neutral',
    neutral_reference: 'all_occupancies',
    kinetics: {
      sigma_n_m2: 1e-19,
      sigma_p_m2: 1e-19,
      thermal_velocity_n_m_s: 1e5,
      thermal_velocity_p_m_s: 1e5,
    },
    degeneracy: 1,
  }
}

function defectNumberInput(
  field: string,
  value: number,
  dimension?: DefectDimension,
): string {
  const dimensionAttr = dimension ? ` data-defect-dimension="${dimension}"` : ''
  return `<input type="text" class="num-input" data-defect-field="${field}"${dimensionAttr} data-defect-canonical="${String(value)}" value="${fmt(value)}" spellcheck="false">`
}

function defaultBulkDefectSpatialProfile(): BulkDefectSpatialProfile {
  return {
    coordinate: 'normalized_layer_coordinate',
    interpolation: 'piecewise_linear',
    density_normalization: 'layer_average_unity',
    knots: [
      { position_fraction: 0, density_multiplier: 1 },
      { position_fraction: 1, density_multiplier: 1 },
    ],
  }
}

function widthConventionForKind(
  kind: BulkDefectDistributionKind,
  current?: BulkDefectWidthConvention,
): BulkDefectWidthConvention {
  if (kind === 'single_level') return 'not_applicable'
  if (kind === 'uniform') return 'uniform_full_width'
  if (kind === 'conduction_band_tail' || kind === 'valence_band_tail') {
    return 'scaps_characteristic_energy'
  }
  return current === 'scaps_characteristic_energy'
    ? 'scaps_characteristic_energy'
    : 'gaussian_standard_deviation'
}

function renderWidthConventionOptions(
  kind: BulkDefectDistributionKind,
  current?: BulkDefectWidthConvention,
): string {
  const selected = widthConventionForKind(kind, current)
  if (kind === 'gaussian') {
    return `
      <option value="gaussian_standard_deviation"${selected === 'gaussian_standard_deviation' ? ' selected' : ''}>Standard deviation</option>
      <option value="scaps_characteristic_energy"${selected === 'scaps_characteristic_energy' ? ' selected' : ''}>SCAPS characteristic energy</option>`
  }
  const labels: Record<Exclude<BulkDefectWidthConvention, 'unresolved' | 'gaussian_standard_deviation'>, string> = {
    not_applicable: 'Not applicable',
    scaps_characteristic_energy: 'SCAPS characteristic energy',
    uniform_full_width: 'Full support width',
  }
  return `<option value="${selected}" selected>${labels[selected as keyof typeof labels]}</option>`
}

function renderBulkDefectSpatialKnot(knot: BulkDefectSpatialKnot, index: number, count: number): string {
  const endpoint = index === 0 || index === count - 1
  return `
    <div class="bulk-defect-knot" data-defect-profile-knot>
      <label class="param">
        <span class="param-label"><span class="sym">Position</span><span class="unit">x/L</span></span>
        ${defectNumberInput('profile_position', knot.position_fraction)}
      </label>
      <label class="param">
        <span class="param-label"><span class="sym">Density multiplier</span></span>
        ${defectNumberInput('profile_multiplier', knot.density_multiplier)}
      </label>
      <button type="button" class="btn btn-ghost bulk-defect-knot-remove" data-defect-knot-remove title="Remove profile knot" aria-label="Remove profile knot"${endpoint ? ' disabled' : ''}>&times;</button>
    </div>`
}

function renderBulkDefectSpecies(species: BulkDefectSpecies): string {
  const transition = species.charge_transition as Exclude<BulkDefectChargeTransition, 'unresolved'>
  const distribution = species.distribution
  const kind = distribution.kind
  const profile = species.spatial_profile ?? defaultBulkDefectSpatialProfile()
  const hasWidth = kind !== 'single_level'
  const hasSupport = kind === 'gaussian'
    || kind === 'conduction_band_tail'
    || kind === 'valence_band_tail'
  return `
    <div class="bulk-defect-species" data-defect-species data-defect-kind="${kind}">
      <div class="bulk-defect-species-head">
        <label class="param bulk-defect-name">
          <span class="param-label"><span class="sym">Species name</span></span>
          <input type="text" class="num-input" data-defect-field="name" value="${escapeHtml(species.name ?? '')}" spellcheck="false">
        </label>
        <button type="button" class="btn btn-ghost bulk-defect-remove" data-defect-remove title="Remove defect species" aria-label="Remove defect species">&times;</button>
      </div>
      <div class="param-grid bulk-defect-grid">
        <label class="param">
          <span class="param-label"><span class="sym">Energy distribution</span></span>
          <select class="num-input" data-defect-field="distribution_kind">
            ${DEFECT_DISTRIBUTION_LABELS.map(option => (
              `<option value="${option.value}"${kind === option.value ? ' selected' : ''}>${option.label}</option>`
            )).join('')}
          </select>
        </label>
        <label class="param">
          <span class="param-label"><span class="sym"><i>N</i><sub>t</sub></span><span class="unit" data-defect-unit-label="density">m⁻³</span></span>
          ${defectNumberInput('total_density', species.distribution.total_density_m3, 'density')}
        </label>
        <label class="param">
          <span class="param-label"><span class="sym"><i>E</i><sub>t</sub> above VB</span><span class="unit">eV</span></span>
          ${defectNumberInput('center_energy', species.distribution.center_eV_above_vb)}
        </label>
        <label class="param" data-defect-width-field${hasWidth ? '' : ' hidden'}>
          <span class="param-label"><span class="sym">Distribution width</span><span class="unit">eV</span></span>
          ${defectNumberInput('width', distribution.width_eV ?? 0.05)}
        </label>
        <label class="param" data-defect-width-field${hasWidth ? '' : ' hidden'}>
          <span class="param-label"><span class="sym">Width convention</span></span>
          <select class="num-input" data-defect-field="width_convention">
            ${renderWidthConventionOptions(kind, distribution.width_convention)}
          </select>
        </label>
        <label class="param" data-defect-support-field${hasSupport ? '' : ' hidden'}>
          <span class="param-label"><span class="sym">Support multiplier</span></span>
          ${defectNumberInput('support_multiplier', distribution.support_width_multiplier ?? 6)}
        </label>
        <label class="param">
          <span class="param-label"><span class="sym">Energy reference</span></span>
          <input type="text" class="num-input" data-defect-field="energy_reference" value="above_valence_band" readonly>
        </label>
        <label class="param">
          <span class="param-label"><span class="sym">Charge transition</span></span>
          <select class="num-input" data-defect-field="charge_transition">
            <option value="neutral"${transition === 'neutral' ? ' selected' : ''}>neutral</option>
            <option value="acceptor"${transition === 'acceptor' ? ' selected' : ''}>acceptor (0/-)</option>
            <option value="donor"${transition === 'donor' ? ' selected' : ''}>donor (+/0)</option>
          </select>
        </label>
        <label class="param">
          <span class="param-label"><span class="sym">Neutral reference</span></span>
          <input type="text" class="num-input" data-defect-field="neutral_reference" value="${DEFECT_NEUTRAL_REFERENCE[transition]}" readonly>
        </label>
        <label class="param">
          <span class="param-label"><span class="sym"><i>σ</i><sub>n</sub></span><span class="unit" data-defect-unit-label="cross_section">m²</span></span>
          ${defectNumberInput('sigma_n', species.kinetics.sigma_n_m2, 'cross_section')}
        </label>
        <label class="param">
          <span class="param-label"><span class="sym"><i>σ</i><sub>p</sub></span><span class="unit" data-defect-unit-label="cross_section">m²</span></span>
          ${defectNumberInput('sigma_p', species.kinetics.sigma_p_m2, 'cross_section')}
        </label>
        <label class="param">
          <span class="param-label"><span class="sym"><i>v</i><sub>th,n</sub></span><span class="unit" data-defect-unit-label="velocity">m/s</span></span>
          ${defectNumberInput('thermal_velocity_n', species.kinetics.thermal_velocity_n_m_s, 'velocity')}
        </label>
        <label class="param">
          <span class="param-label"><span class="sym"><i>v</i><sub>th,p</sub></span><span class="unit" data-defect-unit-label="velocity">m/s</span></span>
          ${defectNumberInput('thermal_velocity_p', species.kinetics.thermal_velocity_p_m_s, 'velocity')}
        </label>
        <label class="param">
          <span class="param-label"><span class="sym">Degeneracy</span></span>
          ${defectNumberInput('degeneracy', species.degeneracy)}
        </label>
      </div>
      <div class="bulk-defect-profile">
        <label class="bulk-defect-enable">
          <input type="checkbox" data-defect-profile-enabled${species.spatial_profile ? ' checked' : ''}>
          <span>Spatial density profile</span>
        </label>
        <div class="bulk-defect-profile-body" data-defect-profile-body${species.spatial_profile ? '' : ' hidden'}>
          <div class="bulk-defect-profile-head">
            <span class="param-label"><span class="sym">Piecewise-linear knots</span></span>
            <output class="bulk-defect-profile-average" data-defect-profile-average></output>
          </div>
          <div class="bulk-defect-knot-list" data-defect-knot-list>
            ${profile.knots.map((knot, index) => (
              renderBulkDefectSpatialKnot(knot, index, profile.knots.length)
            )).join('')}
          </div>
          <button type="button" class="btn btn-ghost bulk-defect-knot-add" data-defect-knot-add title="Add profile knot" aria-label="Add profile knot">+</button>
        </div>
      </div>
    </div>`
}

function renderBulkDefectEditor(
  layer: LayerConfig,
  idx: number,
  bandGradingActive: boolean,
): string {
  const reason = bulkDefectEditorUnsupportedReason(layer, bandGradingActive)
  const hasDocument = layer.defect_schema_version !== undefined
  if (reason) {
    return `
      <details class="param-group bulk-defect-editor bulk-defect-readonly" data-test="bulk-defect-editor" data-layer="${idx}">
        <summary><h5>Explicit bulk defects</h5><span class="experimental-badge">Experimental</span></summary>
        <p class="bulk-defect-warning">Loaded metadata is preserved read-only: ${escapeHtml(reason)}.</p>
      </details>`
  }
  const model = layer.defect_model ?? 'explicit_quasi_steady'
  const species = layer.bulk_defects ?? []
  return `
    <details class="param-group bulk-defect-editor" id="layer-${idx}-bulk-defect-editor" data-test="bulk-defect-editor" data-layer="${idx}">
      <summary><h5>Explicit bulk defects</h5><span class="experimental-badge">Experimental</span></summary>
      <p class="bulk-defect-warning">Charged species run only on the research QF/DC J–V path. Transient, ion-coupled and 2D execution remain unsupported.</p>
      <label class="bulk-defect-enable">
        <input type="checkbox" id="layer-${idx}-defect-enabled"${hasDocument ? ' checked' : ''}>
        <span>Versioned bulk-defect document</span>
      </label>
      <div class="bulk-defect-body" data-defect-body${hasDocument ? '' : ' hidden'}>
        <div class="bulk-defect-toolbar">
          <label class="param">
            <span class="param-label"><span class="sym">Execution model</span></span>
            <select class="num-input" id="layer-${idx}-defect-model">
              <option value="explicit_quasi_steady"${model === 'explicit_quasi_steady' ? ' selected' : ''}>Explicit quasi-steady</option>
              <option value="effective_lifetime"${model === 'effective_lifetime' ? ' selected' : ''}>Effective lifetime</option>
            </select>
          </label>
          <label class="param">
            <span class="param-label"><span class="sym">Display units</span></span>
            <select class="num-input" id="layer-${idx}-defect-units" data-current-unit="si">
              <option value="si">SI</option>
              <option value="scaps_cgs">SCAPS cgs</option>
            </select>
          </label>
          <label class="param">
            <span class="param-label"><span class="sym">Document schema</span></span>
            <output class="num-input bulk-defect-schema" data-defect-schema-label></output>
          </label>
        </div>
        <p class="param-help bulk-defect-model-note" data-defect-model-note></p>
        <div class="bulk-defect-list" data-defect-list>${species.map(renderBulkDefectSpecies).join('')}</div>
        <button type="button" class="btn btn-ghost bulk-defect-add" data-defect-add>+ Add species</button>
      </div>
    </details>`
}

function renderOpticalMaterialSelect(layerIdx: number, currentValue: string | null | undefined): string {
  const id = `layer-${layerIdx}-optical_material`
  const selectedNone = currentValue == null || currentValue === '' ? ' selected' : ''
  const opts = [`<option value=""${selectedNone}>(none — Beer-Lambert)</option>`]
    .concat(
      opticalMaterialOptions.map(m => {
        const safe = escapeHtml(m)
        const sel = currentValue === m ? ' selected' : ''
        return `<option value="${safe}"${sel}>${safe}</option>`
      }),
    )
    .join('')
  return `<select class="num-input" id="${id}" data-layer="${layerIdx}" data-field="optical_material">${opts}</select>`
}

function renderIncoherentCheckbox(layerIdx: number, currentValue: boolean | undefined): string {
  const id = `layer-${layerIdx}-incoherent`
  const checked = currentValue ? ' checked' : ''
  return `<input type="checkbox" id="${id}" data-layer="${layerIdx}" data-field="incoherent"${checked}>`
}

function renderField(layer: LayerConfig, idx: number, f: FieldDef): string {
  const id = `layer-${idx}-${String(f.key)}`
  const unit = f.unit ? `<span class="unit">${f.unit}</span>` : ''
  let control: string
  switch (f.kind) {
    case 'numeric':
    case 'numeric-optional':
      // Same control; the kinds differ only on READ — 'numeric-optional'
      // treats an empty input as "absent" (omits the key) rather than 0.
      control = numAttr(id, layer[f.key] as number | undefined, {
        placeholder: f.placeholder,
        title: f.tooltip,
      })
      break
    case 'select': {
      const cur = (layer[f.key] as string | undefined) ?? (f.options?.[0] ?? '')
      const opts = (f.options ?? [])
        .map(o => `<option value="${escapeHtml(o)}"${cur === o ? ' selected' : ''}>${escapeHtml(o)}</option>`)
        .join('')
      control = `<select class="num-input" id="${id}">${opts}</select>`
      break
    }
    case 'select-optical-material':
      control = renderOpticalMaterialSelect(idx, layer.optical_material)
      break
    case 'boolean':
      control = renderIncoherentCheckbox(idx, layer.incoherent)
      break
  }
  const labelTitle = f.tooltip ? ` title="${escapeHtml(f.tooltip)}"` : ''
  return `
        <label class="param"${labelTitle}>
          <span class="param-label"><span class="sym">${f.label}</span>${unit}</span>
          ${control}
        </label>`
}

function isVisibleField(f: FieldDef, tier: SimulationModeName | undefined): boolean {
  if (!tier) return true
  return isFieldVisible(String(f.key), tier)
}

function renderLayer(
  layer: LayerConfig,
  idx: number,
  tier?: SimulationModeName,
  forceOpen: boolean = false,
  bandGradingActive: boolean = false,
): string {
  const groups = LAYER_GROUPS.map(group => {
    const visibleFields = group.fields.filter(f => isVisibleField(f, tier))
    if (visibleFields.length === 0) return ''
    const rows = visibleFields.map(f => renderField(layer, idx, f)).join('')
    if (group.collapsed) {
      return `
      <details class="param-group">
        <summary><h5>${group.title}</h5></summary>
        <div class="param-grid">${rows}</div>
      </details>`
    }
    return `
      <div class="param-group">
        <h5>${group.title}</h5>
        <div class="param-grid">${rows}</div>
      </div>`
  }).join('')
  const bulkDefects = !tier || tier === 'full'
    ? renderBulkDefectEditor(layer, idx, bandGradingActive)
    : ''

  const openAttr = (forceOpen || idx === 0) ? 'open' : ''
  return `
    <details class="param-card" ${openAttr}>
      <summary>
        <span class="layer-index">${idx + 1}</span>
        <input type="text" class="layer-name" id="layer-${idx}-name" value="${escapeHtml(layer.name)}" spellcheck="false">
        <select class="layer-role" id="layer-${idx}-role">
          <option value="substrate" ${layer.role === 'substrate' ? 'selected' : ''}>substrate</option>
          <option value="front_contact" ${layer.role === 'front_contact' ? 'selected' : ''}>front contact</option>
          <option value="ETL" ${layer.role === 'ETL' ? 'selected' : ''}>ETL</option>
          <option value="absorber" ${layer.role === 'absorber' ? 'selected' : ''}>absorber</option>
          <option value="HTL" ${layer.role === 'HTL' ? 'selected' : ''}>HTL</option>
          <option value="back_contact" ${layer.role === 'back_contact' ? 'selected' : ''}>back contact</option>
        </select>
      </summary>
      <div class="param-card-body">${groups}${bulkDefects}</div>
    </details>`
}

function renderInterfaces(config: DeviceConfig): string {
  const n = Math.max(0, config.layers.length - 1)
  const rows: string[] = []
  for (let i = 0; i < n; i++) {
    const pair = config.device.interfaces?.[i] ?? [0, 0]
    const left = config.layers[i]?.name ?? `layer ${i + 1}`
    const right = config.layers[i + 1]?.name ?? `layer ${i + 2}`
    rows.push(`
      <div class="iface-row">
        <span class="iface-label">${escapeHtml(left)} / ${escapeHtml(right)}</span>
        <label class="param">
          <span class="param-label"><span class="sym"><i>v</i><sub>n</sub></span><span class="unit">m/s</span></span>
          ${numAttr(`iface-${i}-vn`, pair[0])}
        </label>
        <label class="param">
          <span class="param-label"><span class="sym"><i>v</i><sub>p</sub></span><span class="unit">m/s</span></span>
          ${numAttr(`iface-${i}-vp`, pair[1])}
        </label>
      </div>`)
  }
  if (!rows.length) return ''
  return `
    <div class="param-group">
      <h5>Interface Recombination (SRV)</h5>
      <div class="iface-list">${rows.join('')}</div>
    </div>`
}

/**
 * FULL-tier-only Stage B(c.1) Robin / selective-contacts panel. Maps the
 * four ``DeviceConfig.device.S_{n,p}_{left,right}`` fields to UI labels
 * "Top contact (HTL side)" / "Bottom contact (ETL side)" — matches the
 * y-axis convention exposed by the workstation 2D pane and
 * ``MaterialArrays2D.S_{n,p}_{top,bot}``. The YAML keys remain the
 * original 1D names (``S_n_left`` etc.) for backwards compatibility with
 * the 1D Phase 3.3 hook; this is a UI-only relabel for the 2D mental
 * model. Empty input is the "absent" sentinel (round-trips as ``null``);
 * an explicit ``0`` is the "perfectly blocking (Neumann)" limit; large
 * values (≥ 10³ m/s) approach the ohmic limit.
 */
function renderRobinContacts(config: DeviceConfig): string {
  const d = config.device
  const help = '<p class="param-help">Surface recombination velocities at the outer contacts. Empty = disabled (Dirichlet ohmic); explicit <code>0</code> = perfectly blocking (Neumann); ≥ 10³ m/s approaches the ohmic limit. <strong>Top</strong> = HTL side (y=0, YAML <code>S_*_left</code>); <strong>Bottom</strong> = ETL side (y=Ny−1, YAML <code>S_*_right</code>).</p>'
  return `
      <details class="param-group">
        <summary><h5>Advanced 2D Physics — Robin contacts (B(c.1))</h5></summary>
        ${help}
        <div class="param-grid">
          <label class="param" title="Electron surface velocity at the top contact (HTL side, YAML S_n_left). Empty disables; 0 is blocking; ≥ 1e3 m/s approaches ohmic.">
            <span class="param-label"><span class="sym"><i>S</i><sub>n</sub><sup>top</sup></span><span class="unit">m/s</span></span>
            ${numAttr('dev-S-n-top', d.S_n_left, { placeholder: '0 — disabled', title: 'Top electron Robin S (YAML S_n_left)' })}
          </label>
          <label class="param" title="Hole surface velocity at the top contact (HTL side, YAML S_p_left). Empty disables; 0 is blocking; ≥ 1e3 m/s approaches ohmic.">
            <span class="param-label"><span class="sym"><i>S</i><sub>p</sub><sup>top</sup></span><span class="unit">m/s</span></span>
            ${numAttr('dev-S-p-top', d.S_p_left, { placeholder: '0 — disabled', title: 'Top hole Robin S (YAML S_p_left)' })}
          </label>
          <label class="param" title="Electron surface velocity at the bottom contact (ETL side, YAML S_n_right). Empty disables; 0 is blocking; ≥ 1e3 m/s approaches ohmic.">
            <span class="param-label"><span class="sym"><i>S</i><sub>n</sub><sup>bot</sup></span><span class="unit">m/s</span></span>
            ${numAttr('dev-S-n-bot', d.S_n_right, { placeholder: '0 — disabled', title: 'Bottom electron Robin S (YAML S_n_right)' })}
          </label>
          <label class="param" title="Hole surface velocity at the bottom contact (ETL side, YAML S_p_right). Empty disables; 0 is blocking; ≥ 1e3 m/s approaches ohmic.">
            <span class="param-label"><span class="sym"><i>S</i><sub>p</sub><sup>bot</sup></span><span class="unit">m/s</span></span>
            ${numAttr('dev-S-p-bot', d.S_p_right, { placeholder: '0 — disabled', title: 'Bottom hole Robin S (YAML S_p_right)' })}
          </label>
        </div>
      </details>`
}

/**
 * FULL-tier-only SCAPS-validation physics panel (device-level). Surfaces the
 * five flags the YAML loader and ``stack_from_dict`` both parse — DOS band
 * potentials, flat-band contacts, interface-plane closure / projection, and
 * the heterointerface bulk-Auger de-spike fraction — so a user loading a
 * parity preset (e.g. scaps_mirror_v2) can see and round-trip them instead
 * of having them silently stripped at the inline-device boundary.
 */
function renderScapsPhysics(config: DeviceConfig): string {
  const d = config.device
  const help = '<p class="param-help">SCAPS-validation physics (device-level). <strong>DOS band potentials</strong> adds the V<sub>T</sub>·ln(N<sub>C</sub>/N<sub>V</sub>) quasi-Fermi step at DOS-contrast heterojunctions (closes the V<sub>oc</sub> gap; needs per-layer N<sub>C</sub>/N<sub>V</sub>). <strong>Flat-band contacts</strong> uses SCAPS finite-S metal contacts. <strong>Interface-plane closure / projection</strong> evaluate interface recombination on the band-bending-suppressed plane. <strong>De-spike f</strong> blends the heterointerface-node density toward the neighbour geometric mean in the bulk Auger rate (0 = off, 0.53 = SCAPS-emulation).</p>'
  const cb = (
    id: string, label: string, on: boolean, title: string, disabled = false,
  ): string => `
          <label class="param" title="${title}">
            <span class="param-label">${label}</span>
            <input type="checkbox" id="${id}"${on ? ' checked' : ''}${disabled ? ' disabled' : ''}>
          </label>`
  // jv_sweep.py:940 gates shared-site behind the diffusion-only steric form:
  //   shared = ion_steric_diffusion_only and ion_steric_shared_site and ...
  // With the whole-flux form selected the box has no effect, so it is disabled
  // and says so rather than silently doing nothing. The backend default is
  // true, so only an explicit false disables it.
  const stericDiffusionOnly = d.ion_steric_diffusion_only ?? true
  const sharedSiteTitle = stericDiffusionOnly
    ? 'With two mobile species, shared-site crowding uses θ_c = (c + a)/c_lim and θ_a = (c + a)/a_lim. For one common reservoir, set c_lim = a_lim. When unchecked, the occupancies are c/c_lim and a/a_lim instead (YAML ion_steric_shared_site; backend default true).'
    : 'No effect while ion_steric_diffusion_only is off: the whole-flux steric form does not consult the shared-site assumption (jv_sweep.py). Re-enable ion_steric_diffusion_only in the YAML to use this. The stored value is preserved.'
  return `
      <details class="param-group">
        <summary><h5>SCAPS comparison controls</h5></summary>
        ${help}
        <div class="param-grid">
          ${cb('dev-dos', 'DOS band potentials', d.dos_band_potentials ?? true, 'V_T·ln(DOS) quasi-Fermi step (YAML dos_band_potentials)')}
          ${cb('dev-flatband', 'Flat-band contacts', !!d.flat_band_contacts, 'SCAPS finite-S metal contacts (YAML flat_band_contacts)')}
          ${cb('dev-iface-closure', 'Interface-plane closure', !!d.interface_plane_closure, 'QSS plane-density interface SRH, recombination-only; trap electrostatic charge is parked (YAML interface_plane_closure)')}
          ${cb('dev-iface-proj', 'Interface-plane projection', !!d.interface_plane_projection, 'phi-projected interface densities (YAML interface_plane_projection)')}
          <label class="param" title="Calibrated heterointerface bulk-Auger de-spike scaffold (YAML het_recomb_despike). 0 = off; 0.53 = SCAPS comparison lane.">
            <span class="param-label"><span class="sym">de-spike <i>f</i></span></span>
            ${numAttr('dev-despike', d.het_recomb_despike, { placeholder: '0 — off', title: 'het_recomb_despike (0 = off, 0.53 = SCAPS-emulation)' })}
          </label>
          ${cb('dev-ion-shared-site', 'Ions share one lattice site', d.ion_steric_shared_site ?? true, sharedSiteTitle, !stericDiffusionOnly)}
          ${cb('dev-band-grading', 'Bandgap grading', !!d.band_grading, 'Electrical chi/Eg grading. An explicit graded_optics CIGS block may use the same composition coordinate for n,k (YAML band_grading).')}
          ${cb('dev-iface-tunnel', 'Interface tunnelling (TFE)', !!d.interface_tunneling, 'Intra-band thermionic-field-emission through CB/VB spikes — static Padovani-Stratton enhancement of A* at TE-capped faces (YAML interface_tunneling).')}
          <label class="param" title="Tunnelling effective mass relative to the free-electron mass (YAML tunnel_mass_eff). Only used when Interface tunnelling is on.">
            <span class="param-label"><span class="sym"><i>m</i><sub>tun</sub>/<i>m</i><sub>e</sub></span></span>
            ${numAttr('dev-tunnel-mass', d.tunnel_mass_eff, { placeholder: '0.2', title: 'tunnel_mass_eff (relative to m_e; default 0.2)' })}
          </label>
        </div>
      </details>`
}

/**
 * Phase E1.8 — FULL-tier-only per-heterointerface SCAPS defect panel.
 * Collapsed ``<details>`` placed below the Robin contacts panel; one row per
 * internal interface of ``config.layers`` — FULL-layer aligned, so a
 * substrate-prefixed stack gets a leading glass|HTL row rather than starting
 * at HTL/absorber. Row labels come from the adjacent layer names, so they
 * stay correct either way. Each row exposes the 5 SCAPS fields
 * (σ_n, σ_p, N_t areal, v_th, E_t below CB) typed ``number | null`` —
 * empty input is the "absent" sentinel (round-trips as ``null``).
 * Mirrors the YAML schema parsed by ``scaps_compat/loader.py`` and the
 * backend ``stack_from_dict`` plumbing.
 */
function renderInterfaceDefects(config: DeviceConfig): string {
  const n = Math.max(0, config.layers.length - 1)
  if (n === 0) return ''
  const help = '<p class="param-help">SCAPS-style per-heterointerface SRH defect. Each row contributes ``σ·v_th·N_t`` surface velocities to <code>DeviceStack.interfaces[k]</code> and an <code>InterfaceDefect(E_t_eV)</code> entry to <code>DeviceStack.interface_defects[k]</code>. Empty fields = absent (no defect on this interface). Calibration ratio between SCAPS direct N_t and SolarLab effective N_t is ~10⁻⁴ for PVK/ETL.</p>'
  const rows: string[] = []
  for (let i = 0; i < n; i++) {
    const defect = config.device.interface_defects?.[i] ?? null
    const left = config.layers[i]?.name ?? `layer ${i + 1}`
    const right = config.layers[i + 1]?.name ?? `layer ${i + 2}`
    rows.push(`
      <div class="iface-row">
        <span class="iface-label">${escapeHtml(left)} / ${escapeHtml(right)}</span>
        <label class="param" title="Electron capture cross-section [cm²]">
          <span class="param-label"><span class="sym">σ<sub>n</sub></span><span class="unit">cm²</span></span>
          ${numAttr(`idef-${i}-sigma-n`, defect?.sigma_n_cm2, { placeholder: '— disabled' })}
        </label>
        <label class="param" title="Hole capture cross-section [cm²]">
          <span class="param-label"><span class="sym">σ<sub>p</sub></span><span class="unit">cm²</span></span>
          ${numAttr(`idef-${i}-sigma-p`, defect?.sigma_p_cm2, { placeholder: '— disabled' })}
        </label>
        <label class="param" title="Areal trap density at the interface plane [cm⁻²]">
          <span class="param-label"><span class="sym"><i>N</i><sub>t</sub></span><span class="unit">cm⁻²</span></span>
          ${numAttr(`idef-${i}-N-t`, defect?.N_t_cm2, { placeholder: '— disabled' })}
        </label>
        <label class="param" title="Thermal velocity (typically 1e7 cm/s) [cm/s]">
          <span class="param-label"><span class="sym"><i>v</i><sub>th</sub></span><span class="unit">cm/s</span></span>
          ${numAttr(`idef-${i}-v-th`, defect?.v_th_cm_s, { placeholder: '— disabled' })}
        </label>
        <label class="param" title="Trap energy referenced as E_C(reference side) − E_t [eV]; reference is the absorber if exactly one adjacent layer is an absorber, else the lower-Eg side.">
          <span class="param-label"><span class="sym"><i>E</i><sub>t</sub></span><span class="unit">eV</span></span>
          ${numAttr(`idef-${i}-E-t`, defect?.E_t_eV_below_cb, { placeholder: '— disabled' })}
        </label>
      </div>`)
  }
  return `
      <details class="param-group">
        <summary><h5>Interface Defects (FULL only) — Phase E1.5</h5></summary>
        ${help}
        <div class="iface-list">${rows.join('')}</div>
      </details>`
}

function renderModeOptions(current: SimulationModeName): string {
  return MODE_OPTIONS
    .map(o => `<option value="${o.value}"${o.value === current ? ' selected' : ''}>${o.label}</option>`)
    .join('')
}

function inferredBuiltInPotentialMode(config: DeviceConfig): BuiltInPotentialMode {
  const explicit = config.device.built_in_potential_mode
  if (isBuiltInPotentialMode(explicit)) return explicit
  // Mirror the backend parser exactly. Any old manual key denotes an
  // un-migrated compatibility payload, including the historical case where
  // flat_band_contacts implicitly selected the band-derived Poisson value.
  // A genuinely new payload with no manual key starts on the fail-closed
  // semiconductor-work-function path.
  if (config.device.V_bi !== undefined || config.device.V_bi_override !== undefined) {
    return 'legacy_manual'
  }
  return 'semiconductor_work_function'
}

function renderBuiltInPotentialOptions(current: BuiltInPotentialMode): string {
  return BUILT_IN_POTENTIAL_OPTIONS
    .map(o => `<option value="${o.value}"${o.value === current ? ' selected' : ''}>${o.label}</option>`)
    .join('')
}

export function renderDeviceEditor(
  container: HTMLElement,
  config: DeviceConfig,
  tier?: SimulationModeName,
  selectedLayerIdx?: number,
): void {
  const singleLayer = selectedLayerIdx != null && tier === 'full'
  const layerHtml = singleLayer
    ? renderLayer(
      config.layers[selectedLayerIdx!], selectedLayerIdx!, tier, true, !!config.device.band_grading,
    )
    : config.layers.map((layer, idx) => (
      renderLayer(layer, idx, tier, false, !!config.device.band_grading)
    )).join('')
  const currentMode: SimulationModeName = isModeName(config.device.mode) ? config.device.mode : 'full'
  const builtInPotentialMode = inferredBuiltInPotentialMode(config)
  const manualVbi = config.device.V_bi_override ?? config.device.V_bi ?? 1.1
  const currentT = config.device.T ?? 300
  const showT = !tier || isFieldVisible('T', tier)
  const tField = showT ? `
          <label class="param">
            <span class="param-label"><span class="sym"><i>T</i></span><span class="unit">K</span></span>
            ${numAttr('dev-T', currentT)}
          </label>` : ''
  const deviceGroup = singleLayer ? '' : `
      <div class="param-group">
        <h5>Device</h5>
        <div class="param-grid">
          <label class="param">
            <span class="param-label"><span class="sym">Mode</span></span>
            <select class="num-input" id="dev-mode">${renderModeOptions(currentMode)}</select>
          </label>${tField}
          <label class="param" title="Select the physical source of the Poisson built-in potential">
            <span class="param-label"><span class="sym">Built-in potential</span></span>
            <select class="num-input" id="dev-vbi-mode">${renderBuiltInPotentialOptions(builtInPotentialMode)}</select>
          </label>
          <label class="param" data-vbi-mode="legacy_manual">
            <span class="param-label"><span class="sym"><i>V</i><sub>bi</sub> override</span><span class="unit">V</span></span>
            ${numAttr('dev-Vbi', manualVbi, { title: 'Compatibility-only positive magnitude' })}
          </label>
          <label class="param" data-vbi-mode="metal_work_function">
            <span class="param-label"><span class="sym"><i>W</i><sub>left</sub></span><span class="unit">eV</span></span>
            ${numAttr('dev-W-left', config.device.work_function_left_eV, { title: 'Left metal work function below vacuum' })}
          </label>
          <label class="param" data-vbi-mode="metal_work_function">
            <span class="param-label"><span class="sym"><i>W</i><sub>right</sub></span><span class="unit">eV</span></span>
            ${numAttr('dev-W-right', config.device.work_function_right_eV, { title: 'Right metal work function below vacuum' })}
          </label>
          <label class="param">
            <span class="param-label"><span class="sym"><i>Φ</i></span><span class="unit">m⁻²·s⁻¹</span></span>
            ${numAttr('dev-Phi', config.device.Phi)}
          </label>
        </div>
      </div>`
  const interfacesHtml = singleLayer ? '' : renderInterfaces(config)
  // Stage B(c.1) Robin contacts panel — FULL-tier-only because the
  // ``use_selective_contacts`` flag is off in LEGACY/FAST (mode.py:54-86).
  // Hidden in the single-layer drill-down too, where the panel would lose
  // context (it is a device-level setting, not a per-layer one).
  const robinHtml = !singleLayer && tier === 'full' ? renderRobinContacts(config) : ''
  // Phase E1.8 — interface defects panel placed below Robin contacts.
  // FULL-tier-gated (matches the underlying ``InterfaceDefect`` solver
  // hook from Phase E1.5). Hidden in single-layer drill-down because
  // the panel is device-level.
  const interfaceDefectsHtml =
    !singleLayer && tier === 'full' ? renderInterfaceDefects(config) : ''
  // SCAPS-validation physics panel — FULL-tier-only and device-level, same
  // gating as Robin contacts. Hidden panels round-trip as a no-op because
  // readDeviceEditor falls back to original.device.* when the inputs are
  // absent (and single-layer drill-down returns original.device verbatim).
  const scapsHtml = !singleLayer && tier === 'full' ? renderScapsPhysics(config) : ''
  container.innerHTML = `
    <div class="editor">
      ${deviceGroup}
      ${robinHtml}
      ${scapsHtml}
      ${interfaceDefectsHtml}
      ${interfacesHtml}
      <div class="layer-list">${layerHtml}</div>
    </div>`
  const potentialSelect = container.querySelector<HTMLSelectElement>('#dev-vbi-mode')
  const syncPotentialFields = (): void => {
    const selected = potentialSelect?.value
    container.querySelectorAll<HTMLElement>('[data-vbi-mode]').forEach(field => {
      field.hidden = field.dataset.vbiMode !== selected
    })
  }
  potentialSelect?.addEventListener('change', syncPotentialFields)
  syncPotentialFields()
  wireBulkDefectEditors(container, config)
}

function defectDisplayFactor(unit: DefectDisplayUnit, dimension: string): number {
  if (dimension !== 'density' && dimension !== 'cross_section' && dimension !== 'velocity') {
    return 1
  }
  return DEFECT_DISPLAY_FACTORS[unit][dimension]
}

function convertDefectInputs(
  root: ParentNode,
  from: DefectDisplayUnit,
  to: DefectDisplayUnit,
): void {
  root.querySelectorAll<HTMLInputElement>('[data-defect-dimension]').forEach(input => {
    const rawValue = input.value.trim()
    if (rawValue === '') return
    const value = Number(rawValue)
    if (!Number.isFinite(value)) return
    const dimension = input.dataset.defectDimension ?? ''
    const storedCanonical = Number(input.dataset.defectCanonical)
    const expectedDisplay = Number.isFinite(storedCanonical)
      ? fmt(storedCanonical * defectDisplayFactor(from, dimension))
      : ''
    const canonical = Number.isFinite(storedCanonical) && input.value.trim() === expectedDisplay
      ? storedCanonical
      : value / defectDisplayFactor(from, dimension)
    input.dataset.defectCanonical = String(canonical)
    input.value = fmt(canonical * defectDisplayFactor(to, dimension))
  })
}

function updateDefectUnitLabels(root: ParentNode, unit: DefectDisplayUnit): void {
  const labels: Record<DefectDimension, Record<DefectDisplayUnit, string>> = {
    density: { si: 'm⁻³', scaps_cgs: 'cm⁻³' },
    cross_section: { si: 'm²', scaps_cgs: 'cm²' },
    velocity: { si: 'm/s', scaps_cgs: 'cm/s' },
  }
  root.querySelectorAll<HTMLElement>('[data-defect-unit-label]').forEach(label => {
    const dimension = label.dataset.defectUnitLabel
    if (dimension === 'density' || dimension === 'cross_section' || dimension === 'velocity') {
      label.textContent = labels[dimension][unit]
    }
  })
}

function syncDefectTransition(row: Element): void {
  const transition = row.querySelector<HTMLSelectElement>(
    '[data-defect-field="charge_transition"]',
  )?.value
  const reference = row.querySelector<HTMLInputElement>(
    '[data-defect-field="neutral_reference"]',
  )
  if (
    reference
    && (transition === 'neutral' || transition === 'acceptor' || transition === 'donor')
  ) reference.value = DEFECT_NEUTRAL_REFERENCE[transition]
}

function defectDistributionKind(row: Element): BulkDefectDistributionKind | null {
  const value = row.querySelector<HTMLSelectElement>(
    '[data-defect-field="distribution_kind"]',
  )?.value
  return DEFECT_DISTRIBUTION_LABELS.some(option => option.value === value)
    ? value as BulkDefectDistributionKind
    : null
}

function syncDefectDistribution(row: Element): void {
  const kind = defectDistributionKind(row)
  if (kind === null) return
  row.setAttribute('data-defect-kind', kind)
  const hasWidth = kind !== 'single_level'
  const hasSupport = kind === 'gaussian'
    || kind === 'conduction_band_tail'
    || kind === 'valence_band_tail'
  row.querySelectorAll<HTMLElement>('[data-defect-width-field]').forEach(field => {
    field.hidden = !hasWidth
  })
  row.querySelectorAll<HTMLElement>('[data-defect-support-field]').forEach(field => {
    field.hidden = !hasSupport
  })
  const convention = row.querySelector<HTMLSelectElement>(
    '[data-defect-field="width_convention"]',
  )
  if (convention) convention.innerHTML = renderWidthConventionOptions(
    kind,
    convention.value as BulkDefectWidthConvention,
  )
}

function currentDefectInputValue(
  input: HTMLInputElement,
  unit: DefectDisplayUnit = 'si',
): number {
  const displayed = Number(input.value.trim())
  const dimension = input.dataset.defectDimension ?? ''
  const factor = defectDisplayFactor(unit, dimension)
  const storedCanonical = Number(input.dataset.defectCanonical)
  const expectedDisplay = Number.isFinite(storedCanonical)
    ? fmt(storedCanonical * factor)
    : ''
  return Number.isFinite(storedCanonical) && input.value.trim() === expectedDisplay
    ? storedCanonical
    : displayed / factor
}

function defectProfileKnotsFromRow(row: Element): BulkDefectSpatialKnot[] | null {
  const knots = Array.from(row.querySelectorAll<HTMLElement>('[data-defect-profile-knot]'))
    .map(knot => {
      const position = knot.querySelector<HTMLInputElement>('[data-defect-field="profile_position"]')
      const multiplier = knot.querySelector<HTMLInputElement>('[data-defect-field="profile_multiplier"]')
      if (!position || !multiplier) return null
      return {
        position_fraction: currentDefectInputValue(position),
        density_multiplier: currentDefectInputValue(multiplier),
      }
    })
  return knots.some(knot => knot === null)
    ? null
    : knots as BulkDefectSpatialKnot[]
}

function syncDefectProfileEndpoints(row: Element): void {
  const knots = Array.from(row.querySelectorAll<HTMLElement>('[data-defect-profile-knot]'))
  knots.forEach((knot, index) => {
    const remove = knot.querySelector<HTMLButtonElement>('[data-defect-knot-remove]')
    if (remove) remove.disabled = index === 0 || index === knots.length - 1
  })
}

function updateDefectProfileAverage(row: Element): void {
  const output = row.querySelector<HTMLOutputElement>('[data-defect-profile-average]')
  if (!output) return
  const knots = defectProfileKnotsFromRow(row)
  if (!knots || knots.length < 2 || knots.some(knot => (
    !Number.isFinite(knot.position_fraction)
    || !Number.isFinite(knot.density_multiplier)
  ))) {
    output.textContent = 'Layer average: invalid'
    output.dataset.valid = 'false'
    return
  }
  const integral = knots.slice(1).reduce((sum, right, index) => {
    const left = knots[index]
    return sum + 0.5
      * (left.density_multiplier + right.density_multiplier)
      * (right.position_fraction - left.position_fraction)
  }, 0)
  output.textContent = `Layer average: ${integral.toPrecision(7)}`
  output.dataset.valid = String(Math.abs(integral - 1) <= 1e-12)
}

function syncDefectProfile(row: Element): void {
  const enabled = row.querySelector<HTMLInputElement>('[data-defect-profile-enabled]')
  const body = row.querySelector<HTMLElement>('[data-defect-profile-body]')
  if (!enabled || !body) return
  body.hidden = !enabled.checked
  syncDefectProfileEndpoints(row)
  updateDefectProfileAverage(row)
}

function inferDefectSchemaVersion(
  editor: Element,
  original: BulkDefectSchemaVersion | undefined,
): BulkDefectSchemaVersion {
  const rows = Array.from(editor.querySelectorAll<HTMLElement>('[data-defect-species]'))
  if (rows.some(row => row.querySelector<HTMLInputElement>(
    '[data-defect-profile-enabled]',
  )?.checked)) return EXPLICIT_DEFECT_SCHEMA_V3
  if (
    original === EXPLICIT_DEFECT_SCHEMA_V2
    || original === EXPLICIT_DEFECT_SCHEMA_V3
    || rows.some(row => defectDistributionKind(row) !== 'single_level')
  ) return EXPLICIT_DEFECT_SCHEMA_V2
  return EXPLICIT_DEFECT_SCHEMA_V1
}

function updateDefectSchemaLabel(
  editor: Element,
  original: BulkDefectSchemaVersion | undefined,
): void {
  const output = editor.querySelector<HTMLOutputElement>('[data-defect-schema-label]')
  if (!output) return
  const schema = inferDefectSchemaVersion(editor, original)
  output.value = schema.endsWith('-v1') ? 'v1' : schema.endsWith('-v2') ? 'v2' : 'v3'
  output.dataset.schemaVersion = schema
}

function addDefectProfileKnot(row: Element): void {
  const list = row.querySelector<HTMLElement>('[data-defect-knot-list]')
  const knots = defectProfileKnotsFromRow(row)
  if (!list || !knots || knots.length < 2) return
  let insertAfter = -1
  let widestGap = 0
  for (let index = 0; index < knots.length - 1; index += 1) {
    const gap = knots[index + 1].position_fraction - knots[index].position_fraction
    if (gap > widestGap) {
      widestGap = gap
      insertAfter = index
    }
  }
  if (insertAfter < 0 || widestGap <= 0) return
  const left = knots[insertAfter]
  const right = knots[insertAfter + 1]
  const midpoint = 0.5 * (left.position_fraction + right.position_fraction)
  const multiplier = 0.5 * (left.density_multiplier + right.density_multiplier)
  knots.splice(insertAfter + 1, 0, {
    position_fraction: midpoint,
    density_multiplier: multiplier,
  })
  list.innerHTML = knots.map((knot, index) => (
    renderBulkDefectSpatialKnot(knot, index, knots.length)
  )).join('')
  syncDefectProfile(row)
}

function wireBulkDefectEditors(container: HTMLElement, config: DeviceConfig): void {
  config.layers.forEach((layer, idx) => {
    const editor = container.querySelector<HTMLElement>(`#layer-${idx}-bulk-defect-editor`)
    if (!editor) return
    const enabled = editor.querySelector<HTMLInputElement>(`#layer-${idx}-defect-enabled`)
    const body = editor.querySelector<HTMLElement>('[data-defect-body]')
    const model = editor.querySelector<HTMLSelectElement>(`#layer-${idx}-defect-model`)
    const units = editor.querySelector<HTMLSelectElement>(`#layer-${idx}-defect-units`)
    const list = editor.querySelector<HTMLElement>('[data-defect-list]')
    const add = editor.querySelector<HTMLButtonElement>('[data-defect-add]')
    const note = editor.querySelector<HTMLElement>('[data-defect-model-note]')
    const originalSchema = layer.defect_schema_version
    if (!enabled || !body || !model || !units || !list || !add || !note) return

    let nextSpeciesIndex = list.children.length
    const appendDefaultSpecies = (): void => {
      const names = new Set(
        Array.from(list.querySelectorAll<HTMLInputElement>('[data-defect-field="name"]'))
          .map(input => input.value),
      )
      let species: BulkDefectSpecies
      do {
        species = defaultBulkDefectSpecies(layer, nextSpeciesIndex)
        nextSpeciesIndex += 1
      } while (species.name !== null && names.has(species.name))
      list.insertAdjacentHTML('beforeend', renderBulkDefectSpecies(species))
      const row = list.lastElementChild
      const currentUnit: DefectDisplayUnit = units.value === 'scaps_cgs' ? 'scaps_cgs' : 'si'
      if (row && currentUnit !== 'si') convertDefectInputs(row, 'si', currentUnit)
      if (row) {
        syncDefectDistribution(row)
        syncDefectProfile(row)
      }
      updateDefectUnitLabels(editor, currentUnit)
      updateDefectSchemaLabel(editor, originalSchema)
    }

    const syncState = (ensureSpecies: boolean): void => {
      body.hidden = !enabled.checked
      if (!enabled.checked) return
      if (ensureSpecies && model.value === 'explicit_quasi_steady' && list.children.length === 0) {
        appendDefaultSpecies()
      }
      note.textContent = model.value === 'explicit_quasi_steady'
        ? 'Explicit species active in the QF/DC constitutive closure.'
        : 'Lifetime SRH active; listed species remain provenance metadata.'
      updateDefectSchemaLabel(editor, originalSchema)
    }

    enabled.addEventListener('change', () => syncState(true))
    model.addEventListener('change', () => syncState(true))
    add.addEventListener('click', appendDefaultSpecies)
    list.addEventListener('click', event => {
      const target = event.target as Element | null
      const removeSpecies = target?.closest('[data-defect-remove]')
      if (removeSpecies) {
        removeSpecies.closest('[data-defect-species]')?.remove()
        updateDefectSchemaLabel(editor, originalSchema)
        return
      }
      const addKnot = target?.closest('[data-defect-knot-add]')
      if (addKnot) {
        const row = addKnot.closest('[data-defect-species]')
        if (row) addDefectProfileKnot(row)
        return
      }
      const removeKnot = target?.closest<HTMLButtonElement>('[data-defect-knot-remove]')
      if (removeKnot && !removeKnot.disabled) {
        const row = removeKnot.closest('[data-defect-species]')
        removeKnot.closest('[data-defect-profile-knot]')?.remove()
        if (row) syncDefectProfile(row)
      }
    })
    list.addEventListener('change', event => {
      const target = event.target as HTMLElement | null
      const row = target?.closest('[data-defect-species]')
      if (!target || !row) return
      if (target.dataset.defectField === 'charge_transition') syncDefectTransition(row)
      if (target.dataset.defectField === 'distribution_kind') syncDefectDistribution(row)
      if (target.matches('[data-defect-profile-enabled]')) syncDefectProfile(row)
      updateDefectSchemaLabel(editor, originalSchema)
    })
    list.addEventListener('input', event => {
      const row = (event.target as Element | null)?.closest('[data-defect-species]')
      if (row) updateDefectProfileAverage(row)
    })
    units.addEventListener('change', () => {
      const from: DefectDisplayUnit = units.dataset.currentUnit === 'scaps_cgs'
        ? 'scaps_cgs'
        : 'si'
      const to: DefectDisplayUnit = units.value === 'scaps_cgs' ? 'scaps_cgs' : 'si'
      convertDefectInputs(list, from, to)
      updateDefectUnitLabels(editor, to)
      units.dataset.currentUnit = to
    })
    syncState(false)
    updateDefectUnitLabels(editor, 'si')
    list.querySelectorAll<HTMLElement>('[data-defect-species]').forEach(row => {
      syncDefectDistribution(row)
      syncDefectProfile(row)
    })
    updateDefectSchemaLabel(editor, originalSchema)
  })
}

function parseNum(id: string, fallback: number): number {
  const el = document.getElementById(id) as HTMLInputElement | null
  if (!el) return fallback
  const v = Number(el.value)
  return Number.isFinite(v) ? v : fallback
}

/**
 * Like ``parseNum`` but the empty-string case is a meaningful sentinel
 * (``null`` = "absent / disabled") rather than a fallback. Used for the
 * Stage B(c.1) Robin S fields where the backend distinguishes
 * ``None`` (Dirichlet ohmic), ``0`` (Neumann blocking), and a positive
 * finite value (Robin). When the input element does not exist (e.g.
 * because the panel is hidden under a non-FULL tier), the original
 * value is preserved verbatim — including ``undefined``, which is the
 * "field never set on this preset" state.
 */
function parseNumOrNull(
  id: string,
  fallback: number | null | undefined,
): number | null | undefined {
  const el = document.getElementById(id) as HTMLInputElement | null
  if (!el) return fallback
  const raw = el.value.trim()
  if (raw === '') return null
  const v = Number(raw)
  return Number.isFinite(v) ? v : fallback
}

function parseText(id: string, fallback: string): string {
  const el = document.getElementById(id) as HTMLInputElement | HTMLSelectElement | null
  return el?.value ?? fallback
}

function parseRole(id: string, fallback: LayerRole): LayerRole {
  const el = document.getElementById(id) as HTMLSelectElement | null
  const v = el?.value
  return isLayerRole(v) ? v : fallback
}

function parseOpticalMaterial(layerIdx: number, fallback: string | null | undefined): string | null {
  const el = document.getElementById(`layer-${layerIdx}-optical_material`) as HTMLSelectElement | null
  if (!el) return fallback ?? null
  const v = el.value
  return v === '' ? null : v
}

function parseCheckbox(id: string, fallback: boolean): boolean {
  const el = document.getElementById(id) as HTMLInputElement | null
  if (!el) return fallback
  return el.checked
}

function requiredDefectNumber(
  row: Element,
  field: string,
  label: string,
  unit: DefectDisplayUnit,
  constraint: 'positive' | 'nonnegative',
): number {
  const input = row.querySelector<HTMLInputElement>(`[data-defect-field="${field}"]`)
  if (!input) throw new Error(`Missing ${label} input`)
  const rawValue = input.value.trim()
  if (rawValue === '') throw new Error(`${label} must not be empty`)
  const displayed = Number(rawValue)
  if (!Number.isFinite(displayed)) throw new Error(`${label} must be finite`)
  const value = currentDefectInputValue(input, unit)
  if (!Number.isFinite(value) || (constraint === 'positive' ? value <= 0 : value < 0)) {
    throw new Error(`${label} must be finite and ${constraint === 'positive' ? 'positive' : 'non-negative'}`)
  }
  return value
}

function readDefectSpatialProfile(
  row: Element,
  prefix: string,
): BulkDefectSpatialProfile | undefined {
  const enabled = row.querySelector<HTMLInputElement>('[data-defect-profile-enabled]')
  if (!enabled?.checked) return undefined
  const knotRows = Array.from(row.querySelectorAll<HTMLElement>('[data-defect-profile-knot]'))
  if (knotRows.length < 2) throw new Error(`${prefix} spatial profile requires at least two knots`)
  const knots = knotRows.map((knot, index) => {
    const position = requiredDefectNumber(
      knot,
      'profile_position',
      `${prefix} profile knot ${index + 1} position`,
      'si',
      'nonnegative',
    )
    if (position > 1) {
      throw new Error(`${prefix} profile knot ${index + 1} position must lie in [0, 1]`)
    }
    return {
      position_fraction: position,
      density_multiplier: requiredDefectNumber(
        knot,
        'profile_multiplier',
        `${prefix} profile knot ${index + 1} density multiplier`,
        'si',
        'positive',
      ),
    }
  })
  if (knots[0].position_fraction !== 0 || knots.at(-1)?.position_fraction !== 1) {
    throw new Error(`${prefix} spatial profile requires exact 0 and 1 endpoints`)
  }
  knots.slice(1).forEach((knot, index) => {
    if (knot.position_fraction <= knots[index].position_fraction) {
      throw new Error(`${prefix} spatial profile positions must be strictly increasing`)
    }
  })
  const integral = knots.slice(1).reduce((sum, right, index) => {
    const left = knots[index]
    return sum + 0.5
      * (left.density_multiplier + right.density_multiplier)
      * (right.position_fraction - left.position_fraction)
  }, 0)
  if (Math.abs(integral - 1) > 1e-12) {
    throw new Error(
      `${prefix} spatial profile layer average must equal unity; integral=${integral}`,
    )
  }
  return {
    coordinate: 'normalized_layer_coordinate',
    interpolation: 'piecewise_linear',
    density_normalization: 'layer_average_unity',
    knots,
  }
}

function readBulkDefectEditor(
  next: LayerConfig,
  idx: number,
  bandGradingActive: boolean,
): LayerConfig {
  const editor = document.getElementById(`layer-${idx}-bulk-defect-editor`)
  const enabled = document.getElementById(`layer-${idx}-defect-enabled`) as HTMLInputElement | null
  if (!editor || !enabled) return next
  if (!enabled.checked) {
    delete next.defect_schema_version
    delete next.defect_model
    delete next.bulk_defects
    return next
  }
  const modelInput = document.getElementById(`layer-${idx}-defect-model`) as HTMLSelectElement | null
  const unitInput = document.getElementById(`layer-${idx}-defect-units`) as HTMLSelectElement | null
  const model = modelInput?.value
  if (model !== 'effective_lifetime' && model !== 'explicit_quasi_steady') {
    throw new Error(`Layer ${idx + 1} has an unsupported defect model`)
  }
  const unit: DefectDisplayUnit = unitInput?.value === 'scaps_cgs' ? 'scaps_cgs' : 'si'
  const rows = Array.from(editor.querySelectorAll<HTMLElement>('[data-defect-species]'))
  if (model === 'explicit_quasi_steady' && rows.length === 0) {
    throw new Error(`Layer ${idx + 1} explicit defect model requires at least one species`)
  }
  const schema = inferDefectSchemaVersion(editor, next.defect_schema_version)
  const species: BulkDefectSpecies[] = rows.map((row, speciesIndex) => {
    const prefix = `Layer ${idx + 1} defect ${speciesIndex + 1}`
    const rawName = row.querySelector<HTMLInputElement>('[data-defect-field="name"]')?.value.trim() ?? ''
    const name = rawName === '' ? null : rawName
    if (model === 'explicit_quasi_steady' && name === null) {
      throw new Error(`${prefix} requires a non-empty name`)
    }
    if (schema === EXPLICIT_DEFECT_SCHEMA_V3 && name === null) {
      throw new Error(`${prefix} v3 spatial profile requires a non-empty name`)
    }
    const transitionValue = row.querySelector<HTMLSelectElement>(
      '[data-defect-field="charge_transition"]',
    )?.value
    if (
      transitionValue !== 'neutral'
      && transitionValue !== 'acceptor'
      && transitionValue !== 'donor'
    ) throw new Error(`${prefix} has an unsupported charge transition`)
    const density = requiredDefectNumber(
      row, 'total_density', `${prefix} density`, unit, 'positive',
    )
    const center = requiredDefectNumber(
      row, 'center_energy', `${prefix} energy`, unit, 'nonnegative',
    )
    const kind = defectDistributionKind(row)
    if (kind === null) throw new Error(`${prefix} has an unsupported energy distribution`)
    const distribution: BulkDefectSpecies['distribution'] = {
      kind,
      normalization: 'integrated_total',
      total_density_m3: density,
      center_eV_above_vb: center,
    }
    if (schema !== EXPLICIT_DEFECT_SCHEMA_V1) {
      distribution.energy_reference = 'above_valence_band'
    }
    if (kind !== 'single_level') {
      distribution.width_eV = requiredDefectNumber(
        row, 'width', `${prefix} distribution width`, unit, 'positive',
      )
      const conventionValue = row.querySelector<HTMLSelectElement>(
        '[data-defect-field="width_convention"]',
      )?.value
      const expectedConvention = widthConventionForKind(
        kind,
        conventionValue as BulkDefectWidthConvention,
      )
      if (conventionValue !== expectedConvention) {
        throw new Error(`${prefix} has an unsupported width convention`)
      }
      distribution.width_convention = expectedConvention
    }
    if (
      kind === 'gaussian'
      || kind === 'conduction_band_tail'
      || kind === 'valence_band_tail'
    ) {
      distribution.support_width_multiplier = requiredDefectNumber(
        row, 'support_multiplier', `${prefix} support multiplier`, unit, 'positive',
      )
    }
    const distributionReason = distributionUnsupportedReason(
      distribution,
      schema,
      next,
      bandGradingActive,
    )
    if (distributionReason) throw new Error(`${prefix} ${distributionReason}`)
    const value: BulkDefectSpecies = {
      name,
      distribution,
      charge_transition: transitionValue,
      neutral_reference: DEFECT_NEUTRAL_REFERENCE[transitionValue],
      kinetics: {
        sigma_n_m2: requiredDefectNumber(
          row, 'sigma_n', `${prefix} electron cross-section`, unit, 'nonnegative',
        ),
        sigma_p_m2: requiredDefectNumber(
          row, 'sigma_p', `${prefix} hole cross-section`, unit, 'nonnegative',
        ),
        thermal_velocity_n_m_s: requiredDefectNumber(
          row, 'thermal_velocity_n', `${prefix} electron thermal velocity`, unit, 'positive',
        ),
        thermal_velocity_p_m_s: requiredDefectNumber(
          row, 'thermal_velocity_p', `${prefix} hole thermal velocity`, unit, 'positive',
        ),
      },
      degeneracy: requiredDefectNumber(
        row, 'degeneracy', `${prefix} degeneracy`, unit, 'positive',
      ),
    }
    const spatialProfile = readDefectSpatialProfile(row, prefix)
    if (spatialProfile) value.spatial_profile = spatialProfile
    return value
  })
  const names = species.flatMap(item => item.name === null ? [] : [item.name])
  if (new Set(names).size !== names.length) {
    throw new Error(`Layer ${idx + 1} defect species names must be unique`)
  }
  next.defect_schema_version = schema
  next.defect_model = model
  next.bulk_defects = species
  return next
}

export function readDeviceEditor(
  original: DeviceConfig,
  selectedLayerIdx?: number,
): DeviceConfig {
  const singleLayer = selectedLayerIdx != null
  const layers: LayerConfig[] = original.layers.map((layer, idx) => {
    if (singleLayer && idx !== selectedLayerIdx) {
      return layer
    }
    const next: LayerConfig = { ...layer }
    next.name = parseText(`layer-${idx}-name`, layer.name)
    next.role = parseRole(`layer-${idx}-role`, layer.role)
    for (const group of LAYER_GROUPS) {
      for (const f of group.fields) {
        const id = `layer-${idx}-${String(f.key)}`
        switch (f.kind) {
          case 'numeric': {
            const original_v = (layer[f.key] as number | undefined) ?? 0
            ;(next as unknown as Record<string, number>)[f.key as string] = parseNum(id, original_v)
            break
          }
          case 'numeric-optional': {
            // Empty input → null → omit the key (absent / disabled sentinel),
            // so an empty Eg_back/chi_back keeps the layer ungraded rather
            // than coercing to 0 (which would mis-mark it graded).
            const val = parseNumOrNull(id, (layer[f.key] as number | undefined) ?? null)
            const rec = next as unknown as Record<string, unknown>
            if (val === null || val === undefined) delete rec[f.key as string]
            else rec[f.key as string] = val
            break
          }
          case 'select': {
            const cur = (layer[f.key] as string | undefined) ?? (f.options?.[0] ?? '')
            ;(next as unknown as Record<string, string>)[f.key as string] = parseText(id, cur)
            break
          }
          case 'select-optical-material': {
            next.optical_material = parseOpticalMaterial(idx, layer.optical_material)
            break
          }
          case 'boolean': {
            next.incoherent = parseCheckbox(id, layer.incoherent ?? false)
            break
          }
        }
      }
    }
    // Keep ungraded layers clean: with no back endpoint the layer is not
    // graded, so strip any stray grading-spec keys (the 'select' reads always
    // emit a profile/direction default) — leaves the payload bit-identical to
    // a layer that never had grading fields.
    if (next.Eg_back == null && next.chi_back == null) {
      const g = next as unknown as Record<string, unknown>
      delete g.grading_profile
      delete g.grading_direction
      delete g.grading_bowing
      delete g.grading_char_length
      delete g.grading_N_mult
    }
    const bandGradingActive = parseCheckbox(
      'dev-band-grading',
      !!original.device.band_grading,
    )
    return readBulkDefectEditor(next, idx, bandGradingActive)
  })

  if (singleLayer) {
    return { device: original.device, layers }
  }

  const interfaces: Array<[number, number]> = []
  for (let i = 0; i < layers.length - 1; i++) {
    const existing = original.device.interfaces?.[i] ?? [0, 0]
    interfaces.push([
      parseNum(`iface-${i}-vn`, existing[0]),
      parseNum(`iface-${i}-vp`, existing[1]),
    ])
  }
  // Phase E1.8 — read interface defects panel. Each slot is "absent"
  // (null in the round-trip payload) iff EVERY field is empty input;
  // a fully-populated slot serialises into an ``InterfaceDefectFields``
  // object for backend ``stack_from_dict``. Mixed half-populated slots
  // are not allowed by contract (backend rejects), so the reader
  // collapses them to null to surface the user's intent cleanly.
  const interface_defects: Array<InterfaceDefectFields | null> = []
  for (let i = 0; i < layers.length - 1; i++) {
    const existing = original.device.interface_defects?.[i] ?? null
    const coreFields: InterfaceDefectFields = {
      sigma_n_cm2: parseNumOrNull(`idef-${i}-sigma-n`, existing?.sigma_n_cm2 ?? null) ?? null,
      sigma_p_cm2: parseNumOrNull(`idef-${i}-sigma-p`, existing?.sigma_p_cm2 ?? null) ?? null,
      N_t_cm2: parseNumOrNull(`idef-${i}-N-t`, existing?.N_t_cm2 ?? null) ?? null,
      v_th_cm_s: parseNumOrNull(`idef-${i}-v-th`, existing?.v_th_cm_s ?? null) ?? null,
      E_t_eV_below_cb: parseNumOrNull(`idef-${i}-E-t`, existing?.E_t_eV_below_cb ?? null) ?? null,
    }
    const allNull = Object.values(coreFields).every(v => v == null)
    if (allNull) {
      interface_defects.push(null)
      continue
    }
    const parsed: InterfaceDefectFields = {
      ...coreFields,
      ...(existing?.calibration_factor !== undefined
        ? { calibration_factor: existing.calibration_factor }
        : {}),
      ...(existing?.iface_state_calibration_factor !== undefined
        ? { iface_state_calibration_factor: existing.iface_state_calibration_factor }
        : {}),
    }
    interface_defects.push(parsed)
  }
  const anyDefectPopulated = interface_defects.some(d => d != null)
  const interfaceDefectsField = anyDefectPopulated
    ? { interface_defects }
    : (original.device.interface_defects !== undefined
      ? { interface_defects }
      : {})
  // SCAPS-validation physics flags. Read only when the FULL-tier panel is
  // rendered; otherwise parseCheckbox / parseNumOrNull fall back to the
  // original value so a non-FULL round-trip preserves them verbatim. Most
  // flags are spread in only when truthy / non-zero so non-SCAPS configs
  // keep a clean payload. DOS band potentials is different: its backend
  // default is ON, so an unchecked box must serialize an explicit ``false``.
  const scapsPhysicsField: Record<string, boolean | number> = {}
  const dosBandPotentials = parseCheckbox(
    'dev-dos', original.device.dos_band_potentials ?? true,
  )
  if (!dosBandPotentials) {
    scapsPhysicsField.dos_band_potentials = false
  } else if (original.device.dos_band_potentials !== undefined) {
    scapsPhysicsField.dos_band_potentials = true
  }
  if (parseCheckbox('dev-flatband', !!original.device.flat_band_contacts))
    scapsPhysicsField.flat_band_contacts = true
  if (parseCheckbox('dev-iface-closure', !!original.device.interface_plane_closure))
    scapsPhysicsField.interface_plane_closure = true
  if (parseCheckbox('dev-iface-proj', !!original.device.interface_plane_projection))
    scapsPhysicsField.interface_plane_projection = true
  const despike = parseNumOrNull('dev-despike', original.device.het_recomb_despike ?? null)
  if (typeof despike === 'number' && despike !== 0)
    scapsPhysicsField.het_recomb_despike = despike
  // Shared-site is a physical assumption with a backend default of TRUE, so —
  // like dos_band_potentials — an unchecked box has to serialise an explicit
  // false rather than simply being omitted.
  // When the box is disabled (diffusion-only steric form off) it carries no
  // user intent, so the stored value wins — reading the DOM there would let a
  // rendered default overwrite what the config actually said.
  const sharedSiteBox = document.getElementById('dev-ion-shared-site')
  const sharedSite = (sharedSiteBox instanceof HTMLInputElement && sharedSiteBox.disabled)
    ? (original.device.ion_steric_shared_site ?? true)
    : parseCheckbox('dev-ion-shared-site', original.device.ion_steric_shared_site ?? true)
  if (!sharedSite) {
    scapsPhysicsField.ion_steric_shared_site = false
  } else if (original.device.ion_steric_shared_site !== undefined) {
    scapsPhysicsField.ion_steric_shared_site = true
  }
  if (parseCheckbox('dev-band-grading', !!original.device.band_grading))
    scapsPhysicsField.band_grading = true
  if (parseCheckbox('dev-iface-tunnel', !!original.device.interface_tunneling)) {
    scapsPhysicsField.interface_tunneling = true
    const tmass = parseNumOrNull('dev-tunnel-mass', original.device.tunnel_mass_eff ?? null)
    if (typeof tmass === 'number') scapsPhysicsField.tunnel_mass_eff = tmass
  }
  const rawMode = parseText('dev-mode', original.device.mode ?? 'full')
  const mode: SimulationModeName = isModeName(rawMode) ? rawMode : 'full'
  const rawPotentialMode = parseText(
    'dev-vbi-mode', inferredBuiltInPotentialMode(original),
  )
  const potentialMode: BuiltInPotentialMode = isBuiltInPotentialMode(rawPotentialMode)
    ? rawPotentialMode
    : inferredBuiltInPotentialMode(original)
  const builtInPotentialField: Partial<DeviceConfig['device']> = {}
  if (potentialMode === 'legacy_manual') {
    const value = parseNum(
      'dev-Vbi', original.device.V_bi_override ?? original.device.V_bi ?? 1.1,
    )
    if (
      original.device.built_in_potential_mode === undefined
      && original.device.V_bi_override === undefined
      && original.device.V_bi !== undefined
    ) {
      // Preserve old benchmark payloads until they are deliberately migrated.
      builtInPotentialField.V_bi = value
    } else {
      builtInPotentialField.built_in_potential_mode = 'legacy_manual'
      builtInPotentialField.V_bi_override = value
    }
  } else if (potentialMode === 'semiconductor_work_function') {
    builtInPotentialField.built_in_potential_mode = potentialMode
  } else {
    builtInPotentialField.built_in_potential_mode = potentialMode
    const left = parseNumOrNull(
      'dev-W-left', original.device.work_function_left_eV ?? null,
    )
    const right = parseNumOrNull(
      'dev-W-right', original.device.work_function_right_eV ?? null,
    )
    if (typeof left === 'number') builtInPotentialField.work_function_left_eV = left
    if (typeof right === 'number') builtInPotentialField.work_function_right_eV = right
  }
  const T = parseNum('dev-T', original.device.T ?? 300)
  const hiddenPhysicsKeys = [
    'te_physical_norm',
    'ion_steric_diffusion_only',
    'autoloop_generated_lever',
    'flat_band_metal_contacts',
    'contact_phi_B_eV',
    'interface_two_sided',
    'interface_shared_occupancy',
    'interface_plane_generation',
    'jv_solver_policy',
    'interface_charge_closure',
    'interface_charge_rebaseline_acknowledged',
    'graded_optics',
  ] as const
  const hiddenPhysicsField: Record<string, boolean | number | string> = {}
  for (const key of hiddenPhysicsKeys) {
    const value = original.device[key]
    if (value !== undefined) hiddenPhysicsField[key] = value
  }
  // Structured device-level physics the editor renders no control for. Kept
  // separate from `hiddenPhysicsField` because that map is deliberately typed
  // to scalars, and widening it to admit a nested object would let any future
  // key through unchecked.
  //
  // Dropping this one is not cosmetic: `wkb_tunnelling_intraband_spike.yaml`
  // is offered in the preset dropdown, and without the passthrough a Run on
  // it silently produced a tunnelling-free J-V. With it, the backend's
  // capability guard refuses instead - the channels are certified only on the
  // guarded QF/DC lane, not on the driver this pane runs - which is the
  // honest answer.
  const structuredPhysicsField: Pick<
    DeviceConfig['device'], 'tunnelling_channels'
  > = {}
  if (original.device.tunnelling_channels !== undefined) {
    structuredPhysicsField.tunnelling_channels =
      original.device.tunnelling_channels
  }
  return {
    ...(original.simulation_hints === undefined
      ? {}
      : { simulation_hints: original.simulation_hints }),
    ...(original.electrical_grid === undefined
      ? {}
      : { electrical_grid: original.electrical_grid }),
    device: {
      ...builtInPotentialField,
      Phi: parseNum('dev-Phi', original.device.Phi),
      interfaces,
      T,
      mode,
      // Stage B(c.1) Robin contacts. Read by ID matching the renderRobin-
      // Contacts panel; when the panel is not rendered (non-FULL tier or
      // single-layer drill-down) ``parseNumOrNull`` returns the
      // ``original.device.S_*`` value, so a round-trip through readDevice-
      // Editor in those modes is a no-op.
      S_n_left: parseNumOrNull('dev-S-n-top', original.device.S_n_left),
      S_p_left: parseNumOrNull('dev-S-p-top', original.device.S_p_left),
      S_n_right: parseNumOrNull('dev-S-n-bot', original.device.S_n_right),
      S_p_right: parseNumOrNull('dev-S-p-bot', original.device.S_p_right),
      // Phase E1.8 — spread the interface_defects field conditionally
      // so absent → still absent (no spurious null array in the payload
      // for presets that pre-date E1.5 or for non-FULL tier round-trips
      // where the panel is hidden).
      ...interfaceDefectsField,
      ...hiddenPhysicsField,
      ...structuredPhysicsField,
      ...scapsPhysicsField,
    },
    layers,
  }
}
