import type { SimulationModeName } from '../types'

/**
 * Keys of layer / device fields that belong to physics upgrades gated by
 * the tiered SimulationMode. Fields not listed here are always visible.
 *
 * Keep this in sync with perovskite_sim/models/mode.py: a field belongs to
 * a tier iff the mode that enables its physics is included in that tier.
 */
const TMM_KEYS = ['optical_material', 'n_optical', 'incoherent'] as const
const DUAL_ION_KEYS = ['D_ion_neg', 'P_lim_neg', 'E_a_ion_neg'] as const
const TRAP_PROFILE_KEYS = ['trap_N_t_interface', 'trap_N_t_bulk', 'trap_decay_length'] as const
const TEMPERATURE_KEYS = ['T'] as const

/**
 * Stage B(c.x) per-RHS hooks — the four flags
 * use_selective_contacts, use_field_dependent_mobility,
 * use_radiative_reabsorption, use_photon_recycling that LEGACY and FAST
 * tiers leave OFF (mode.py:54-86). Only FULL turns them on, so the
 * corresponding parameter fields belong only in the FULL editor surface.
 *
 * Photon recycling and radiative reabsorption have no parameters — they
 * are pure tier flags — so this list only covers the contact velocities
 * (B(c.1)) and the field-mobility per-layer params (B(c.2)).
 */
const PER_RHS_KEYS = [
  // Stage B(c.1) Robin / selective contacts (device-level)
  'S_n_left', 'S_p_left', 'S_n_right', 'S_p_right',
  // Stage B(c.2) field-dependent mobility μ(E) (per-layer)
  'v_sat_n', 'v_sat_p',
  'ct_beta_n', 'ct_beta_p',
  'pf_gamma_n', 'pf_gamma_p',
] as const

/** Keys hidden in FAST mode: no TMM, no dual ions, no trap profile, no T,
 *  no per-RHS hooks (B(c.1) / B(c.2) parameter fields). */
const FAST_HIDDEN = new Set<string>([
  ...TMM_KEYS,
  ...DUAL_ION_KEYS,
  ...TRAP_PROFILE_KEYS,
  ...TEMPERATURE_KEYS,
  ...PER_RHS_KEYS,
])

/** Keys hidden in LEGACY mode — identical to FAST today (mode.py:54-86). */
const LEGACY_HIDDEN = new Set<string>(FAST_HIDDEN)

const HIDDEN_BY_TIER: Record<SimulationModeName, Set<string>> = {
  legacy: LEGACY_HIDDEN,
  fast: FAST_HIDDEN,
  full: new Set<string>(),
}

export function isFieldVisible(key: string, tier: SimulationModeName): boolean {
  return !HIDDEN_BY_TIER[tier].has(key)
}

export function hiddenKeysForTier(tier: SimulationModeName): string[] {
  return [...HIDDEN_BY_TIER[tier]]
}

/**
 * Phase 2b layer-builder gate. The custom-stack visualizer, add/remove/
 * reorder controls, template library, and Save-As path are full-tier-only
 * because adding/removing layers in legacy/fast tiers risks producing
 * configs that silently diverge from IonMonger / DriftFusion benchmark
 * conventions — exactly what those tiers exist to preserve.
 */
export function isLayerBuilderEnabled(tier: SimulationModeName): boolean {
  return tier === 'full'
}
