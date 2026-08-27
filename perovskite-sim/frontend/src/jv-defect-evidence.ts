import type { JVBulkDefectEvidence } from './types'

function scientific(value: number): string {
  return Number.isFinite(value) ? value.toExponential(3) : 'non-finite'
}

export function summarizeJVBulkDefectEvidence(
  evidence: JVBulkDefectEvidence | null | undefined,
): string[] {
  if (!evidence) return []
  return [
    `Explicit defects: ${evidence.species_identifiers.length} species / ${evidence.points_completed} points`,
    `Model ID: ${evidence.model_identity_sha256.slice(0, 12)}...`,
    `Occupancy: [${evidence.minimum_occupancy.toFixed(4)}, ${evidence.maximum_occupancy.toFixed(4)}]`,
    `min kinetic denominator: ${scientific(evidence.minimum_kinetic_denominator_s1)} s^-1`,
    `max |rho_t|: ${scientific(evidence.maximum_absolute_charge_density_C_m3)} C m^-3`,
    `max |R_t|: ${scientific(evidence.maximum_absolute_recombination_rate_m3_s)} m^-3 s^-1`,
  ]
}
