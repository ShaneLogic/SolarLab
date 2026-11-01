import type { InterfaceChargeJVEvidence } from './types'

function shortHash(value: string): string {
  return value.length >= 12 ? `${value.slice(0, 12)}...` : value
}

function exp(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toExponential(digits) : 'non-finite'
}

export function summarizeInterfaceChargeJVEvidence(
  evidence: InterfaceChargeJVEvidence | null | undefined,
): string[] {
  if (!evidence) return []
  return [
    `Charged QF/DC: ${evidence.points.length} requested points + ${evidence.continuation_bridge_count} bridges | protocol ${shortHash(evidence.protocol_sha256)}`,
    `Dark ${shortHash(evidence.dark_state_sha256)} | grid ${shortHash(evidence.grid_sha256)} | stack ${shortHash(evidence.stack_sha256)}`,
    `Occupancy [${evidence.minimum_occupancy.toFixed(4)}, ${evidence.maximum_occupancy.toFixed(4)}] | max |sigma_if| ${exp(evidence.maximum_absolute_sheet_charge_C_m2)} C m^-2 | max |delta_phi_trace| ${exp(evidence.maximum_absolute_trace_potential_shift_V)} V`,
    `Max Gauss ${exp(evidence.maximum_normalized_gauss_residual)} | cell residual ${exp(evidence.maximum_normalized_cell_residual)} | continuity ${exp(evidence.maximum_continuity_bound_A_m2)} A m^-2`,
    `Contact span ${(1e3 * evidence.maximum_contact_fermi_level_span_eV).toFixed(3)} meV | local Jacobian ${exp(evidence.maximum_scaled_local_jacobian_condition)} | tolerance x${evidence.tolerance_factor.toPrecision(3)}`,
  ]
}

export function interfaceChargeJVEvidenceTitle(
  evidence: InterfaceChargeJVEvidence,
): string {
  return [
    evidence.model,
    `protocol sha256:${evidence.protocol_sha256}`,
    `dark sha256:${evidence.dark_state_sha256}`,
    `grid sha256:${evidence.grid_sha256}`,
    `stack sha256:${evidence.stack_sha256}`,
    `charge law ${evidence.protocol.charge_law}`,
  ].join(' | ')
}
