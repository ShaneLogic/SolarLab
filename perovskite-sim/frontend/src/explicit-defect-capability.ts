import type { DeviceConfig } from './types'

export interface DynamicDefectImpedancePreset {
  N_grid: number
  n_freq: number
  f_min: number
  f_max: number
}

export interface DynamicDefectTransientEligibility {
  eligible: boolean
  reasons: string[]
  N_grid: number
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
  return config.layers.some(layer => (
    (layer.D_ion > 0 && layer.P0 > 0)
    || ((layer.D_ion_neg ?? 0) > 0 && (layer.P0_neg ?? 0) > 0)
  ))
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

export function dynamicDefectTransientEligibility(
  config: DeviceConfig,
): DynamicDefectTransientEligibility {
  const reasons: string[] = []
  let substratePrefix = 0
  while (config.layers[substratePrefix]?.role === 'substrate') substratePrefix += 1
  const layers = config.layers.slice(substratePrefix)
  if (layers.length !== 2) {
    reasons.push('requires exactly two electrical layers')
  }
  if (layers.some(layer => (
    layer.defect_model === 'explicit_quasi_steady'
    && (layer.bulk_defects?.length ?? 0) > 0
  ))) {
    reasons.push('bulk explicit defects are outside the v1 transient capability')
  }
  const interfaceDefects = (config.device.interface_defects ?? [])
    .slice(substratePrefix, substratePrefix + Math.max(0, layers.length - 1))
  const populatedDefects = interfaceDefects.filter(defect => defect !== null)
  const interfaceCount = populatedDefects.length
  if (interfaceCount !== 1) {
    reasons.push('requires exactly one microscopic interface defect')
  } else {
    const defect = populatedDefects[0]!
    const microscopicValues = [
      defect.sigma_n_cm2,
      defect.sigma_p_cm2,
      defect.N_t_cm2,
      defect.v_th_cm_s,
      defect.E_t_eV_below_cb,
    ]
    if (
      microscopicValues.some(value => (
        typeof value !== 'number' || !Number.isFinite(value)
      ))
      || microscopicValues.slice(0, 4).some(value => Number(value) <= 0)
    ) {
      reasons.push('requires a complete positive microscopic interface document')
    }
    if (
      (defect.calibration_factor ?? 1) !== 1
      || (defect.iface_state_calibration_factor ?? 1) !== 1
    ) {
      reasons.push('empirical interface calibration is outside the v1 capability')
    }
  }
  if (config.device.interface_charge_closure !== 'equilibrium_referenced') {
    reasons.push('requires equilibrium-referenced interface charge')
  }
  if (config.device.interface_charge_rebaseline_acknowledged !== true) {
    reasons.push('requires interface-charge rebaseline acknowledgement')
  }
  const activePositive = layers
    .map((layer, index) => ({ layer, index }))
    .filter(({ layer }) => layer.D_ion > 0 && layer.P0 > 0)
  if (activePositive.length !== 1) {
    reasons.push('requires exactly one active positive-ion layer')
  } else if (activePositive[0].layer.role !== 'absorber') {
    reasons.push('the active positive-ion layer must be the absorber')
  }
  const activeNegative = layers.some(
    layer => (layer.D_ion_neg ?? 0) > 0 && (layer.P0_neg ?? 0) > 0,
  )
  if (activeNegative) {
    reasons.push('active negative ions are outside the v1 transient capability')
  }
  return {
    eligible: reasons.length === 0,
    reasons,
    N_grid: Math.max(4, config.simulation_hints?.min_N_grid ?? 0),
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
