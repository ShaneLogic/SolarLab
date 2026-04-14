import type {
  DeviceConfig,
  ValidationIssue,
  ValidationReport,
} from '../types'

export function validate(config: DeviceConfig): ValidationReport {
  const errors: ValidationIssue[] = []
  const warnings: ValidationIssue[] = []
  const layers = config.layers

  // 1. Exactly one absorber.
  const absorberIdxs = layers
    .map((l, i) => (l.role === 'absorber' ? i : -1))
    .filter(i => i >= 0)
  if (absorberIdxs.length !== 1) {
    errors.push({
      layerIdx: null,
      field: null,
      message: `Stack needs exactly one absorber layer (found ${absorberIdxs.length})`,
    })
  }

  // 2. Unique names.
  const nameCounts = new Map<string, number[]>()
  layers.forEach((l, i) => {
    const list = nameCounts.get(l.name) ?? []
    list.push(i)
    nameCounts.set(l.name, list)
  })
  for (const [name, idxs] of nameCounts) {
    if (idxs.length > 1) {
      for (const i of idxs) {
        errors.push({
          layerIdx: i,
          field: 'name',
          message: `Duplicate layer name "${name}"`,
        })
      }
    }
  }

  // 3. Positive thickness.
  layers.forEach((l, i) => {
    if (!(typeof l.thickness === 'number' && l.thickness > 0)) {
      errors.push({
        layerIdx: i,
        field: 'thickness',
        message: 'Thickness must be positive',
      })
    }
  })

  // 4. At most one substrate.
  const substrateIdxs = layers
    .map((l, i) => (l.role === 'substrate' ? i : -1))
    .filter(i => i >= 0)
  if (substrateIdxs.length > 1) {
    for (const i of substrateIdxs) {
      errors.push({
        layerIdx: i,
        field: 'role',
        message: 'At most one substrate layer is allowed',
      })
    }
  }

  // 5. Substrate constraints.
  if (substrateIdxs.length === 1) {
    const i = substrateIdxs[0]
    const sub = layers[i]
    if (i !== 0) {
      errors.push({
        layerIdx: i,
        field: 'role',
        message: 'Substrate must be the first layer',
      })
    }
    if (!sub.incoherent) {
      errors.push({
        layerIdx: i,
        field: 'incoherent',
        message: 'Substrate must be marked incoherent',
      })
    }
    if (sub.optical_material == null || sub.optical_material === '') {
      errors.push({
        layerIdx: i,
        field: 'optical_material',
        message: 'Substrate must have an optical material',
      })
    }
  }

  // Warnings
  const tmmCount = layers.filter(
    l => l.optical_material != null && l.optical_material !== '',
  ).length
  if (tmmCount > 0 && tmmCount < layers.length) {
    warnings.push({
      layerIdx: null,
      field: 'optical_material',
      message: 'Mixed TMM / Beer-Lambert layers — TMM-less layers fall back per Phase 2a',
    })
  }
  if (tmmCount === 0 && layers.length > 0) {
    warnings.push({
      layerIdx: null,
      field: 'optical_material',
      message: 'TMM is dormant — set optical_material to enable',
    })
  }

  return { errors, warnings }
}
