/**
 * vitest — the pre-run tier physics summary must not contradict the post-run
 * backend receipt.
 *
 * shell.ts hardcoded FAST to LEGACY's string on the stated belief that "FAST
 * today has identical physics flags to LEGACY (see mode.py)". That is false on
 * six flags: FAST runs thermionic emission, TMM, dual ions, trap profiles,
 * temperature scaling and photon recycling. A FAST user was told "flat bands ·
 * Beer-Lambert · single ion · uniform tau · T=300K" before the run, then got
 * the backend's accurate string after it.
 *
 * backend/main.py:_describe_active_physics derives every fragment from the
 * mode flags precisely so the indicator "can't silently drift from the physics
 * that actually ran". These tests hold the frontend to the same contract.
 */
import { describe, it, expect } from 'vitest'
import { resolveTierFlags, tierPhysicsSummary } from './active-physics'

describe('tier flag mirror matches mode.py', () => {
  it('LEGACY runs no upgrade', () => {
    const f = resolveTierFlags('legacy')
    for (const on of Object.values(f)) expect(on).toBe(false)
  })

  it('FAST differs from LEGACY — it is not a LEGACY alias', () => {
    const legacy = resolveTierFlags('legacy')
    const fast = resolveTierFlags('fast')
    expect(fast).not.toEqual(legacy)
  })

  it('FAST runs the six build-once upgrades', () => {
    const f = resolveTierFlags('fast')
    expect(f.use_thermionic_emission).toBe(true)
    expect(f.use_tmm_optics).toBe(true)
    expect(f.use_dual_ions).toBe(true)
    expect(f.use_trap_profile).toBe(true)
    expect(f.use_temperature_scaling).toBe(true)
    expect(f.use_photon_recycling).toBe(true)
  })

  it('FAST leaves exactly the three per-RHS upgrades off', () => {
    const f = resolveTierFlags('fast')
    expect(f.use_radiative_reabsorption).toBe(false)
    expect(f.use_field_dependent_mobility).toBe(false)
    expect(f.use_selective_contacts).toBe(false)
  })

  it('FULL runs everything', () => {
    const f = resolveTierFlags('full')
    for (const on of Object.values(f)) expect(on).toBe(true)
  })
})

describe('tierPhysicsSummary', () => {
  it('describes LEGACY as the stripped baseline', () => {
    const s = tierPhysicsSummary('legacy')
    expect(s).toContain('flat bands')
    expect(s).toContain('Beer-Lambert')
    expect(s).toContain('single ion')
    expect(s).toContain('T=300K')
  })

  it('does not describe FAST as LEGACY', () => {
    expect(tierPhysicsSummary('fast')).not.toBe(tierPhysicsSummary('legacy'))
  })

  it('credits FAST with the upgrades it actually runs', () => {
    const s = tierPhysicsSummary('fast')
    expect(s).toContain('TE')
    expect(s).toContain('TMM')
    expect(s).toContain('dual ions')
    expect(s).toContain('trap profile')
    expect(s).toContain('T-scaling')
    expect(s).toContain('photon recycling')
  })

  it('does not credit FAST with the per-RHS upgrades it skips', () => {
    const s = tierPhysicsSummary('fast')
    expect(s).not.toContain('Robin contacts')
    expect(s).not.toContain('μ(E)')
    expect(s).not.toContain('PR reabsorption')
  })

  it('credits FULL with the per-RHS upgrades', () => {
    const s = tierPhysicsSummary('full')
    expect(s).toContain('Robin contacts')
    expect(s).toContain('μ(E)')
    expect(s).toContain('PR reabsorption')
  })

  /**
   * Exact cross-language pin. These three strings are the verbatim output of
   * backend/main.py:_describe_active_physics (minus its tier-name prefix,
   * which shell.ts renders separately through tierLabel), captured by running
   * it against configs/dual_ion_demo.yaml at each tier on 2026-09-04:
   *
   *   LEGACY  flat bands · Beer-Lambert · single ion · uniform τ · T=300K
   *   FAST    band offsets · TE · TMM · dual ions · trap profile · T-scaling · photon recycling
   *   FULL    ... · PR reabsorption · μ(E) · Robin contacts
   *
   * If the backend helper changes, this fails and the frontend follows it.
   */
  it('reproduces the backend string exactly, tier for tier', () => {
    expect(tierPhysicsSummary('legacy')).toBe(
      'flat bands · Beer-Lambert · single ion · uniform τ · T=300K',
    )
    expect(tierPhysicsSummary('fast')).toBe(
      'band offsets · TE · TMM · dual ions · trap profile · T-scaling · photon recycling',
    )
    expect(tierPhysicsSummary('full')).toBe(
      'band offsets · TE · TMM · dual ions · trap profile · T-scaling'
      + ' · photon recycling · PR reabsorption · μ(E) · Robin contacts',
    )
  })
})
