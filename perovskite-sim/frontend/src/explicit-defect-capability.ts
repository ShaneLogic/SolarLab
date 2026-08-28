import type { DeviceConfig } from './types'

export interface DynamicDefectImpedancePreset {
  N_grid: number
  n_freq: number
  f_min: number
  f_max: number
}

export function hasExplicitBulkDefects(config: DeviceConfig): boolean {
  return config.layers.some(layer => (
    layer.defect_model === 'explicit_quasi_steady'
    && (layer.bulk_defects ?? []).length > 0
  ))
}

export function hasExplicitInterfaceDefects(config: DeviceConfig): boolean {
  return (config.device.interface_defects ?? []).some(defect => defect !== null)
}

export function hasActiveMobileIons(config: DeviceConfig): boolean {
  return config.layers.some(layer => {
    const extended = layer as typeof layer & {
      D_ion_neg?: number
      P0_neg?: number
    }
    return (
      (layer.D_ion > 0 && layer.P0 > 0)
      || ((extended.D_ion_neg ?? 0) > 0 && (extended.P0_neg ?? 0) > 0)
    )
  })
}

export function dynamicDefectImpedancePreset(
  config: DeviceConfig,
): DynamicDefectImpedancePreset {
  const minimumGrid = config.simulation_hints?.min_N_grid ?? 0
  const hasDefect = hasExplicitBulkDefects(config) || hasExplicitInterfaceDefects(config)
  if (hasDefect && hasActiveMobileIons(config)) {
    return {
      N_grid: Math.max(8, minimumGrid),
      n_freq: 19,
      f_min: 1e-3,
      f_max: 1e6,
    }
  }
  if (hasExplicitInterfaceDefects(config)) {
    return {
      N_grid: Math.max(8, minimumGrid),
      n_freq: 45,
      f_min: 1e-8,
      f_max: 1e14,
    }
  }
  return {
    N_grid: Math.max(12, minimumGrid),
    n_freq: 33,
    f_min: 1e-4,
    f_max: 1e12,
  }
}

export function requiresQuasiFermiBulkDefectSolver(config: DeviceConfig): boolean {
  return config.layers.some(layer => (
    layer.defect_model === 'explicit_quasi_steady'
    && (layer.bulk_defects ?? []).some(species => (
      species.charge_transition === 'acceptor'
      || species.charge_transition === 'donor'
    ))
  ))
}

export function requiresChargedInterfaceJVSolver(config: DeviceConfig): boolean {
  return config.device.interface_charge_closure === 'equilibrium_referenced'
}

export function requiresQuasiFermiJVSolver(config: DeviceConfig): boolean {
  return (
    config.device.jv_solver_policy === 'cancellation_safe_qf_required'
    || requiresQuasiFermiBulkDefectSolver(config)
    || requiresChargedInterfaceJVSolver(config)
  )
}
