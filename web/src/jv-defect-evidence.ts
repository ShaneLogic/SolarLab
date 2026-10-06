import type { JVBulkDefectEvidence } from './types'

function scientific(value: number): string {
  return Number.isFinite(value) ? value.toExponential(3) : 'non-finite'
}

export function summarizeJVBulkDefectEvidence(
  evidence: JVBulkDefectEvidence | null | undefined,
): string[] {
  if (!evidence) return []
  const lines = [
    `Explicit defects: ${evidence.species_identifiers.length} species / ${evidence.points_completed} points`,
    `Model ID: ${evidence.model_identity_sha256.slice(0, 12)}...`,
    `Occupancy: [${evidence.minimum_occupancy.toFixed(4)}, ${evidence.maximum_occupancy.toFixed(4)}]`,
    `min kinetic denominator: ${scientific(evidence.minimum_kinetic_denominator_s1)} s^-1`,
    `max |rho_t|: ${scientific(evidence.maximum_absolute_charge_density_C_m3)} C m^-3`,
    `max |R_t|: ${scientific(evidence.maximum_absolute_recombination_rate_m3_s)} m^-3 s^-1`,
  ]
  const profileHashes = evidence.spatial_profile_sha256s ?? []
  const minima = evidence.minimum_density_multipliers ?? []
  const maxima = evidence.maximum_density_multipliers ?? []
  if (
    evidence.spatial_closure === 'layer-density-profile-v1'
    && profileHashes.length === evidence.species_identifiers.length
    && minima.length === profileHashes.length
    && maxima.length === profileHashes.length
  ) {
    const profiledCount = profileHashes.filter((value) => value !== null).length
    lines.push(
      `Spatial density: ${profiledCount}/${profileHashes.length} profiled species; m(x) [${Math.min(...minima).toFixed(3)}, ${Math.max(...maxima).toFixed(3)}]`,
    )
  }
  return lines
}
