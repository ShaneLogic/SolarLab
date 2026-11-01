import type { DynamicDefectTransientResult } from './types'

const REFERENCE_CERTIFICATE = (
  '9eab2f9e251b8d4c0f7f3f07e0baeea9bb6497126ef8d8111eba1803947e5beb'
)

function scientific(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toExponential(digits) : 'non-finite'
}

export function summarizeDynamicDefectTransientEvidence(
  result: DynamicDefectTransientResult,
): string[] {
  const evidence = result.evidence
  if (!evidence) return ['Unclassified dynamic-defect transient']
  const certificate = evidence.engine_certificate
  return [
    evidence.certified
      ? 'Certified interface defect + positive-ion transient'
      : 'Uncertified interface defect + positive-ion transient',
    `Protocol ${evidence.protocol_sha256.slice(0, 12)} · ${result.times_s.length} time points · ${result.protocol.actual_grid_nodes} nodes · Δt factor ${result.protocol.time_step_refinement_factor}`,
    `max |Δf| ${scientific(evidence.maximum_interface_occupancy_motion)} · ion motion ${scientific(evidence.maximum_positive_ion_relative_motion)}`,
    `centroid shift ${scientific(evidence.maximum_positive_ion_centroid_shift_m * 1e9)} nm · |ΔQ| ${scientific(evidence.maximum_integrated_charge_change_C_m2)} C m⁻²`,
    `residual ${scientific(certificate.maximum_scaled_nonlinear_residual)} · charge ${scientific(certificate.maximum_charge_balance_relative_error)} · inventory ${scientific(certificate.maximum_ion_inventory_relative_drift)}`,
    `time refinement state/current ${scientific(certificate.maximum_refinement_state_change)} / ${scientific(certificate.maximum_refinement_current_relative_change)}`,
    `Reference ${evidence.reference_lane_id} · ${evidence.reference_certificate_sha256.slice(0, 12)}`,
  ]
}

export function collectDynamicDefectTransientEvidenceWarnings(
  result: DynamicDefectTransientResult,
): string[] {
  const evidence = result.evidence
  if (!evidence) {
    return ['Legacy dynamic-defect transient result: evidence is unclassified.']
  }
  const warnings = [...evidence.reasons]
  if (evidence.model !== 'dynamic-defect-transient-evidence-v1') {
    warnings.push('unknown dynamic-defect transient evidence schema')
  }
  if (evidence.protocol.method !== 'dynamic_defect_transient_certified') {
    warnings.push('protocol method is not the certified transient method')
  }
  if (evidence.protocol_sha256.length !== 64 || evidence.state_sha256.length !== 64) {
    warnings.push('protocol or state content address is malformed')
  }
  if (evidence.reference_certificate_sha256 !== REFERENCE_CERTIFICATE) {
    warnings.push('reference certificate does not match D6-E3c v5')
  }
  if (!evidence.dc_operating_point_certified) {
    warnings.push('DC operating point is uncertified')
  }
  if (!evidence.dark_reference_certified) {
    warnings.push('dark reference is uncertified')
  }
  if (!evidence.microscopic_binding_certified) {
    warnings.push('microscopic interface binding is uncertified')
  }
  if (!evidence.numerically_certified) {
    warnings.push('transient numerical gates are uncertified')
  }
  if (!evidence.public_projection_certified) {
    warnings.push('public result projection is uncertified')
  }
  if (!evidence.certified && warnings.length === 0) {
    warnings.push('dynamic-defect transient is uncertified without a declared reason')
  }
  return [...new Set(warnings)]
}
