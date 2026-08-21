import type { ISResult } from './types'

export const LEGACY_IMPEDANCE_EVIDENCE_WARNING =
  'Legacy impedance result: certificate evidence is incomplete; classification is unclassified.'

export function collectImpedanceEvidenceWarnings(result: ISResult): string[] {
  const warnings: string[] = []
  const completeEvidence = Boolean(
    result.protocol
      && result.operating_point
      && result.frequency_window
      && result.grid_assessment,
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
    ? `Frequency window: characteristic frequency ${triState(
      frequencyWindow.characteristic_frequency_bracketed,
      'bracketed',
      'not bracketed',
    )}; ionic branch ${triState(
      frequencyWindow.ionic_branch_covered,
      'covered',
      'not covered',
    )}`
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

  return [protocolSummary, operatingPointSummary, frequencySummary, gridSummary]
}
