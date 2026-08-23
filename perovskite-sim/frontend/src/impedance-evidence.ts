import type { ISResult } from './types'

export const LEGACY_IMPEDANCE_EVIDENCE_WARNING =
  'Legacy impedance result: certificate evidence is incomplete; classification is unclassified.'

function hasFrequencyWindowCertificate(result: ISResult): boolean {
  const frequencyWindow = result.frequency_window
  if (!frequencyWindow) return false
  return [
    'full_timescale_envelope_bracketed',
    'recommended_f_min_Hz',
    'recommended_f_max_Hz',
    'branch_margin_decades',
    'max_allowed_sampling_gap_decades',
    'max_observed_sampling_gap_decades',
    'ionic_branch_assessments',
  ].every(field => Object.prototype.hasOwnProperty.call(frequencyWindow, field))
}

export function collectImpedanceEvidenceWarnings(result: ISResult): string[] {
  const warnings: string[] = []
  const requiresIonAwareEvidence = (
    result.protocol?.method === 'ion_aware_frequency_certified'
  )
  const completeEvidence = Boolean(
    result.protocol
      && result.operating_point
      && hasFrequencyWindowCertificate(result)
      && result.grid_assessment
      && (!requiresIonAwareEvidence || result.ion_aware_evidence),
  )

  if (!completeEvidence) warnings.push(LEGACY_IMPEDANCE_EVIDENCE_WARNING)
  warnings.push(...(result.frequency_window?.warnings ?? []))

  if (result.operating_point && !result.operating_point.certified) {
    const reasons = result.operating_point.reasons.filter(Boolean)
    warnings.push(
      reasons.length > 0
        ? `DC operating point uncertified: ${reasons.join(', ')}`
        : 'DC operating point uncertified: no failure reasons were returned.',
    )
  }

  if (result.grid_assessment && !result.grid_assessment.certified) {
    const gridWarnings = result.grid_assessment.warnings.filter(Boolean)
    warnings.push(
      ...(gridWarnings.length > 0
        ? gridWarnings
        : ['Interface grid uncertified: no grid warning details were returned.']),
    )
  }

  const ionAware = result.ion_aware_evidence
  if (ionAware && !ionAware.numerically_certified) {
    warnings.push(
      ionAware.reasons.length > 0
        ? `Ion-aware linearization uncertified: ${ionAware.reasons.join(', ')}`
        : 'Ion-aware linearization uncertified: no failure reasons were returned.',
    )
  }
  if (ionAware && !ionAware.frequency_window_certified) {
    warnings.push('Ion-aware frequency window is not certified.')
  }
  if (ionAware) {
    const failedPoints = ionAware.frequency_point_certificates.filter(
      point => !point.numerically_certified,
    )
    if (failedPoints.length > 0) {
      warnings.push(
        `${failedPoints.length} of ${ionAware.frequency_point_certificates.length} `
        + 'frequency points failed numerical certification.',
      )
    }
  }

  return [...new Set(warnings.filter(Boolean))]
}

function triState(
  value: boolean | null | undefined,
  positive: string,
  negative: string,
): string {
  if (value === true) return positive
  if (value === false) return negative
  if (value === null) return 'not applicable'
  return 'unclassified'
}

function compactNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return 'unavailable'
  }
  if (value === 0) return '0'
  const magnitude = Math.abs(value)
  if (magnitude >= 1e3 || magnitude < 1e-3) return value.toExponential(2)
  return String(Number(value.toPrecision(3)))
}

export function summarizeImpedanceEvidence(result: ISResult): string[] {
  const protocol = result.protocol
  const operatingPoint = result.operating_point
  const frequencyWindow = result.frequency_window
  const grid = result.grid_assessment
  const ionAware = result.ion_aware_evidence

  const protocolSummary = protocol
    ? `Protocol: ${protocol.method}${
      typeof protocol.points_per_cycle === 'number'
        ? `; ${protocol.points_per_cycle} points/cycle`
        : ''
    }`
    : 'Protocol: unclassified'

  const operatingPointSummary = operatingPoint
    ? `Operating point: ${operatingPoint.certified ? 'certified' : 'uncertified'}`
    : 'Operating point: unclassified'

  const frequencySummary = frequencyWindow
    ? `Frequency window: blocking-charge frequency ${triState(
      frequencyWindow.characteristic_frequency_bracketed,
      'bracketed',
      'not bracketed',
    )}; full ionic envelope ${triState(
      frequencyWindow.full_timescale_envelope_bracketed,
      'bracketed',
      'not bracketed',
    )}; ionic branch ${triState(
      frequencyWindow.ionic_branch_covered,
      'covered',
      'not covered',
    )}; recommended range: [${compactNumber(
      frequencyWindow.recommended_f_min_Hz,
    )}, ${compactNumber(frequencyWindow.recommended_f_max_Hz)}] Hz`
    : 'Frequency window: unclassified'

  const gridSummary = grid
    ? `Grid: ${grid.certified ? 'certified' : 'uncertified'}; guarded cells: ${
      grid.guarded_cell_count
    }; offenders: ${grid.offender_count}; max cell/Debye ratio: ${compactNumber(
      grid.max_guarded_cell_debye_ratio,
    )} (limit ${compactNumber(grid.max_cell_debye_ratio_limit)}); override: ${
      grid.override_used ? 'used' : 'no'
    }`
    : 'Grid: unclassified'

  const ionAwareSummary = ionAware
    ? `Ion-aware engine: ${ionAware.numerically_certified ? 'numerically certified' : 'uncertified'}; `
      + `${ionAware.frequency_point_certificates.filter(point => point.numerically_certified).length}`
      + `/${ionAware.frequency_point_certificates.length} frequency points; `
      + `FD refinement ${ionAware.perturbation_assessments.every(item => item.passed) ? 'passed' : 'failed'}; `
      + `max face spread ${compactNumber(ionAware.max_relative_face_spread)}; `
      + `max inventory response ${compactNumber(ionAware.max_ion_inventory_response_relative)}`
    : protocol?.method === 'ion_aware_frequency_certified'
      ? 'Ion-aware engine: unclassified'
      : null

  return [
    protocolSummary,
    operatingPointSummary,
    frequencySummary,
    gridSummary,
    ...(ionAwareSummary ? [ionAwareSummary] : []),
  ]
}
