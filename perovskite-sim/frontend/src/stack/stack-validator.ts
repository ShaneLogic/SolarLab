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

  // 6. Built-in-potential source contract. Mirror the backend's compatibility
  // inference so malformed physical configs fail before a job is submitted.
  const explicitPotentialMode = config.device.built_in_potential_mode
  const potentialMode = explicitPotentialMode
    ?? (config.device.V_bi !== undefined || config.device.V_bi_override !== undefined
      ? 'legacy_manual'
      : 'semiconductor_work_function')
  const hasLegacyInput = config.device.V_bi !== undefined
    || config.device.V_bi_override !== undefined

  if (
    (potentialMode === 'semiconductor_work_function'
      || potentialMode === 'metal_work_function')
    && hasLegacyInput
  ) {
    errors.push({
      layerIdx: null,
      field: 'built_in_potential_mode',
      message: 'Physical work-function mode cannot include V_bi or V_bi_override',
    })
  }

  if (potentialMode === 'metal_work_function') {
    for (const [field, value] of [
      ['work_function_left_eV', config.device.work_function_left_eV],
      ['work_function_right_eV', config.device.work_function_right_eV],
    ] as const) {
      if (!(typeof value === 'number' && Number.isFinite(value) && value > 0)) {
        errors.push({
          layerIdx: null,
          field,
          message: `${field} must be finite and positive in metal-work-function mode`,
        })
      }
    }
  } else if (potentialMode === 'semiconductor_work_function') {
    const electrical = layers
      .map((item, index) => ({ item, index }))
      .filter(({ item }) => item.role !== 'substrate')
    const contacts = electrical.length > 0
      ? [electrical[0], electrical[electrical.length - 1]]
      : []
    for (const { item, index } of contacts.filter(
      (contact, contactIndex, all) => all.findIndex(c => c.index === contact.index) === contactIndex,
    )) {
      for (const field of ['chi', 'Eg', 'Nc300', 'Nv300'] as const) {
        const value = item[field]
        if (!(typeof value === 'number' && Number.isFinite(value) && value > 0)) {
          errors.push({
            layerIdx: index,
            field,
            message: `${field} must be finite and positive at a semiconductor contact`,
          })
        }
      }
      for (const field of ['N_A', 'N_D'] as const) {
        const value = item[field]
        if (!(Number.isFinite(value) && value >= 0)) {
          errors.push({
            layerIdx: index,
            field,
            message: `${field} must be finite and non-negative at a semiconductor contact`,
          })
        }
      }
    }
  } else if (explicitPotentialMode === 'legacy_manual') {
    const value = config.device.V_bi_override ?? config.device.V_bi
    if (!(typeof value === 'number' && Number.isFinite(value) && value >= 0)) {
      errors.push({
        layerIdx: null,
        field: 'V_bi_override',
        message: 'Legacy manual mode requires a finite, non-negative V_bi_override',
      })
    }
  }

  // 7. Composition-resolved CIGS optics is a strict, research-only opt-in.
  // Mirror the backend activation contract here so malformed jobs do not get
  // as far as the solver worker.  A dormant layer block is preserved but is
  // deliberately not interpreted until the device master gate is enabled.
  const cigsLayerIdxs = layers
    .map((item, index) => (item.cigs_graded_optics != null ? index : -1))
    .filter(index => index >= 0)
  if (config.device.graded_optics === true) {
    if (config.device.band_grading !== true) {
      errors.push({
        layerIdx: null,
        field: 'graded_optics',
        message: 'Graded CIGS optics requires bandgap grading',
      })
    }
    if (cigsLayerIdxs.length === 0) {
      errors.push({
        layerIdx: null,
        field: 'graded_optics',
        message: 'Graded optics is enabled but no layer declares a CIGS optical model',
      })
    }
    for (const index of cigsLayerIdxs) {
      const item = layers[index]
      const model = item.cigs_graded_optics!
      if (item.role !== 'absorber') {
        errors.push({
          layerIdx: index,
          field: 'cigs_graded_optics',
          message: 'Graded CIGS optics is restricted to absorber layers',
        })
      }
      if (item.Eg_back == null && item.chi_back == null) {
        errors.push({
          layerIdx: index,
          field: 'cigs_graded_optics',
          message: 'Graded CIGS optics requires an Eg_back or chi_back endpoint',
        })
      }
      for (const field of ['ggi_front', 'ggi_back'] as const) {
        const value = model[field]
        if (!(Number.isFinite(value) && value >= 0 && value <= 1)) {
          errors.push({
            layerIdx: index,
            field,
            message: `${field} must be finite and lie in [0, 1]`,
          })
        }
      }
      if (!(Number.isFinite(model.cgi) && model.cgi >= 0.75 && model.cgi <= 1)) {
        errors.push({
          layerIdx: index,
          field: 'cgi',
          message: 'cgi must be finite and lie in [0.75, 1]',
        })
      }
      const slices = model.slices ?? 25
      if (!(Number.isInteger(slices) && slices >= 1 && slices <= 512)) {
        errors.push({
          layerIdx: index,
          field: 'slices',
          message: 'slices must be an integer in [1, 512]',
        })
      }
      const quadrature = model.kk_quadrature_order ?? 192
      if (!(Number.isInteger(quadrature) && quadrature >= 48 && quadrature <= 2048)) {
        errors.push({
          layerIdx: index,
          field: 'kk_quadrature_order',
          message: 'kk_quadrature_order must be an integer in [48, 2048]',
        })
      }
      if (model.model !== undefined && model.model !== 'minoura_2015') {
        errors.push({
          layerIdx: index,
          field: 'model',
          message: 'Only the minoura_2015 CIGS optical model is supported',
        })
      }
    }
  }

  // Warnings
  const gradedOpticsActive = config.device.graded_optics === true
    && config.device.band_grading === true
  const tmmCount = layers.filter(l =>
    (l.optical_material != null && l.optical_material !== '')
    || (gradedOpticsActive && l.cigs_graded_optics != null),
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
      message: 'TMM is dormant — set optical_material or activate graded CIGS optics',
    })
  }

  return { errors, warnings }
}
