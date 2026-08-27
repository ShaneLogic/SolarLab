import type { DeviceConfig } from './types'

export function requiresQuasiFermiBulkDefectSolver(config: DeviceConfig): boolean {
  return config.layers.some(layer => (
    layer.defect_model === 'explicit_quasi_steady'
    && (layer.bulk_defects ?? []).some(species => (
      species.charge_transition === 'acceptor'
      || species.charge_transition === 'donor'
    ))
  ))
}

export function requiresQuasiFermiJVSolver(config: DeviceConfig): boolean {
  return (
    config.device.jv_solver_policy === 'cancellation_safe_qf_required'
    || requiresQuasiFermiBulkDefectSolver(config)
  )
}
