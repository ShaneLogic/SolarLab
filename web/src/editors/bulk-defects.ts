import type {
  BulkDefectChargeTransition,
  BulkDefectDistributionKind,
  BulkDefectNeutralReference,
  BulkDefectSchemaVersion,
  BulkDefectSpecies,
  BulkDefectSpatialKnot,
  BulkDefectSpatialProfile,
  BulkDefectWidthConvention,
  DeviceConfig,
  LayerConfig,
} from '../types'
import { formatNumberInput as fmt } from '../number-input'
import { escapeHtml } from './inputs'

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

export function renderBulkDefectEditor(
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

export function wireBulkDefectEditors(container: HTMLElement, config: DeviceConfig): void {
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
      const target = event.target as HTMLElement | null
      if (target) delete target.dataset.defectCanonical
      const row = target?.closest('[data-defect-species]')
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

export function readBulkDefectEditor(
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
