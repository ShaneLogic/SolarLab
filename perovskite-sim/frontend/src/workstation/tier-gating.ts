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

/** Keys hidden in FAST mode (no TMM, no dual ions, no trap profile, no T input). */
const FAST_HIDDEN = new Set<string>([
  ...TMM_KEYS,
  ...DUAL_ION_KEYS,
  ...TRAP_PROFILE_KEYS,
  ...TEMPERATURE_KEYS,
])

/** Keys hidden in LEGACY mode — identical to FAST today (mode.py:54-61). */
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
