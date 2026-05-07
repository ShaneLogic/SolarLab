/**
 * Frontend mirror of the backend ``_describe_active_physics`` helper
 * (backend/main.py:49-78). Single source of truth for the workstation's
 * pre-run "Active physics" summary on jv-2d-pane and the device-card
 * tier badge.
 *
 * Divergence from the backend, by design:
 *   - Backend returns a verbose dotted form for the post-run console:
 *     ``"FULL  band offsets · TE · TMM · dual ions · trap profile · ...
 *     · Robin contacts"``. It includes every fragment, on or off, so the
 *     console is a complete physics receipt.
 *   - Frontend returns a compact comma-joined form for the pre-run
 *     summary: ``"Active physics: Microstructure, Robin contacts, μ(E),
 *     Photon recycling"``. It includes ONLY active extras (the things
 *     that diverge from baseline 2D drift-diffusion). When zero
 *     fragments are active, it returns the literal sentinel
 *     ``"Active physics: baseline 2D drift-diffusion"``.
 *
 * The full backend string is still surfaced after a run lands in
 * Run.activePhysics — frontend display = pre-run preview, backend
 * display = post-run truth.
 */
import type { DeviceConfig, SimulationModeName } from './types'

/**
 * Tier-flag resolver — mirrors perovskite_sim/models/mode.py:54-86.
 * Keep these tables in sync with the Python defaults; the
 * tier-gating.ts unit tests pin the same constants from a different
 * angle (UI visibility), so a regression in either file shows up
 * quickly in npm test.
 */
interface TierFlags {
  use_thermionic_emission: boolean
  use_tmm_optics: boolean
  use_photon_recycling: boolean
  use_radiative_reabsorption: boolean
  use_field_dependent_mobility: boolean
  use_selective_contacts: boolean
}

const TIER_FLAGS: Record<SimulationModeName, TierFlags> = {
  legacy: {
    use_thermionic_emission: false,
    use_tmm_optics: false,
    use_photon_recycling: false,
    use_radiative_reabsorption: false,
    use_field_dependent_mobility: false,
    use_selective_contacts: false,
  },
  fast: {
    use_thermionic_emission: true,
    use_tmm_optics: true,
    use_photon_recycling: true,
    use_radiative_reabsorption: false,
    use_field_dependent_mobility: false,
    use_selective_contacts: false,
  },
  full: {
    use_thermionic_emission: true,
    use_tmm_optics: true,
    use_photon_recycling: true,
    use_radiative_reabsorption: true,
    use_field_dependent_mobility: true,
    use_selective_contacts: true,
  },
}

export function resolveTierFlags(tier: SimulationModeName): TierFlags {
  return TIER_FLAGS[tier]
}

/**
 * Optional microstructure block on the device config. Not yet a
 * first-class field on ``DeviceConfig`` (no UI editor surfaces it
 * directly — only jv-2d-pane.ts injects it inline as a per-run
 * override) but it IS round-tripped through the YAML loader as
 * ``DeviceStack.microstructure``. The active-physics summary needs to
 * read it, so we accept it via a structural cast.
 */
interface MaybeMicrostructure {
  microstructure?: {
    grain_boundaries?: unknown[]
  }
}

function hasMicrostructure(device: DeviceConfig): boolean {
  const ms = (device as DeviceConfig & MaybeMicrostructure).microstructure
  return Array.isArray(ms?.grain_boundaries) && ms.grain_boundaries.length > 0
}

function hasRobinContacts(device: DeviceConfig): boolean {
  const d = device.device
  for (const s of [d.S_n_left, d.S_p_left, d.S_n_right, d.S_p_right]) {
    if (s != null && s !== 0) return true
  }
  return false
}

function hasFieldMobility(device: DeviceConfig): boolean {
  for (const layer of device.layers) {
    if ((layer.v_sat_n ?? 0) > 0) return true
    if ((layer.v_sat_p ?? 0) > 0) return true
    if ((layer.pf_gamma_n ?? 0) > 0) return true
    if ((layer.pf_gamma_p ?? 0) > 0) return true
  }
  return false
}

function hasTmmOptics(device: DeviceConfig): boolean {
  return device.layers.some(L => Boolean(L.optical_material))
}

/**
 * Compact comma-joined active-physics summary for the workstation UI.
 *
 * Fragment ordering is thematic, not alphabetical — geometric features →
 * boundary conditions → transport modifications → optics. Matches the
 * example in the spec:
 *   "Active physics: Microstructure, Robin contacts, μ(E), Photon recycling"
 *
 * Each fragment fires only when both the corresponding tier flag is on
 * AND the device config supplies the required parameters. This means a
 * FULL-tier device with no Robin S values still shows "Active physics:
 * Photon recycling, Reabsorption, TMM" — the four per-RHS flag fragments
 * appear only when actually configured, not just when permitted.
 *
 * @returns A user-facing string starting with "Active physics: ". When
 * no fragment is active, returns "Active physics: baseline 2D drift-
 * diffusion" (the sentinel used by jv-2d-pane and the device-card
 * tooltip).
 */
export function describeActivePhysics(device: DeviceConfig): string {
  const tier: SimulationModeName = device.device.mode ?? 'full'
  const flags = resolveTierFlags(tier)
  const fragments: string[] = []

  // Geometric features
  if (hasMicrostructure(device)) {
    fragments.push('Microstructure')
  }

  // Boundary conditions
  if (flags.use_selective_contacts && hasRobinContacts(device)) {
    fragments.push('Robin contacts')
  }

  // Transport modifications
  if (flags.use_field_dependent_mobility && hasFieldMobility(device)) {
    fragments.push('μ(E)')
  }

  // Optics
  if (flags.use_radiative_reabsorption) {
    fragments.push('Reabsorption')
  }
  if (flags.use_photon_recycling) {
    fragments.push('Photon recycling')
  }
  if (flags.use_tmm_optics && hasTmmOptics(device)) {
    fragments.push('TMM')
  }

  if (fragments.length === 0) {
    return 'Active physics: baseline 2D drift-diffusion'
  }
  return 'Active physics: ' + fragments.join(', ')
}
