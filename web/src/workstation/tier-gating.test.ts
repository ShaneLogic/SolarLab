import { describe, it, expect } from 'vitest'
import { isFieldVisible, hiddenKeysForTier, isLayerBuilderEnabled } from './tier-gating'

describe('isFieldVisible', () => {
  it('always shows the core Geometry & Electrostatics fields regardless of tier', () => {
    for (const tier of ['legacy', 'fast', 'full'] as const) {
      expect(isFieldVisible('thickness', tier)).toBe(true)
      expect(isFieldVisible('eps_r', tier)).toBe(true)
      expect(isFieldVisible('mu_n', tier)).toBe(true)
    }
  })

  // Corrected 2026-09-04, same defect as the dual-ion entry below:
  // ``use_tmm_optics`` is off only in LEGACY (mode.py FAST/FULL both True).
  it('hides TMM fields in legacy only — FAST runs use_tmm_optics', () => {
    for (const key of ['optical_material', 'n_optical', 'incoherent', 'cigs_graded_optics']) {
      expect(isFieldVisible(key, 'legacy')).toBe(false)
      expect(isFieldVisible(key, 'fast')).toBe(true)
      expect(isFieldVisible(key, 'full')).toBe(true)
    }
  })

  // Corrected 2026-09-04: this previously pinned the dual-ion fields as hidden
  // in FAST as well, which contradicted mode.py — ``use_dual_ions`` is off only
  // in LEGACY and on in both FAST and FULL. The editor was hiding fields from a
  // tier whose physics runs them.
  it('hides dual-ion fields in legacy only — FAST runs use_dual_ions', () => {
    for (const key of ['D_ion_neg', 'P0_neg', 'P_lim_neg'] as const) {
      expect(isFieldVisible(key, 'legacy')).toBe(false)
      expect(isFieldVisible(key, 'fast')).toBe(true)
      expect(isFieldVisible(key, 'full')).toBe(true)
    }
  })

  it('hides trap-profile fields in legacy only — FAST runs use_trap_profile', () => {
    for (const key of ['trap_N_t_interface', 'trap_N_t_bulk', 'trap_decay_length']) {
      expect(isFieldVisible(key, 'legacy')).toBe(false)
      expect(isFieldVisible(key, 'fast')).toBe(true)
      expect(isFieldVisible(key, 'full')).toBe(true)
    }
  })

  it('hides device-level T in legacy only — FAST runs use_temperature_scaling', () => {
    expect(isFieldVisible('T', 'legacy')).toBe(false)
    expect(isFieldVisible('T', 'fast')).toBe(true)
    expect(isFieldVisible('T', 'full')).toBe(true)
  })

  // Grading is not a mode.py flag at all: device.py:574 gates it on the tier
  // NAME — ``bool(band_grading) and sim_mode.name != "legacy"`` — so FAST runs
  // it too, and the fields were hidden from a tier that executes them.
  it('hides grading fields in legacy only', () => {
    for (const key of [
      'Eg_back', 'chi_back', 'grading_profile', 'grading_direction',
      'grading_bowing', 'grading_char_length', 'grading_N_mult',
    ]) {
      expect(isFieldVisible(key, 'legacy')).toBe(false)
      expect(isFieldVisible(key, 'fast')).toBe(true)
      expect(isFieldVisible(key, 'full')).toBe(true)
    }
  })

  it('hides Stage B(c.1) Robin contact fields in legacy and fast', () => {
    for (const k of ['S_n_left', 'S_p_left', 'S_n_right', 'S_p_right']) {
      expect(isFieldVisible(k, 'legacy')).toBe(false)
      expect(isFieldVisible(k, 'fast')).toBe(false)
      expect(isFieldVisible(k, 'full')).toBe(true)
    }
  })

  it('hides Stage B(c.2) field-mobility fields in legacy and fast', () => {
    for (const k of [
      'v_sat_n', 'v_sat_p', 'ct_beta_n', 'ct_beta_p',
      'pf_gamma_n', 'pf_gamma_p',
    ]) {
      expect(isFieldVisible(k, 'legacy')).toBe(false)
      expect(isFieldVisible(k, 'fast')).toBe(false)
      expect(isFieldVisible(k, 'full')).toBe(true)
    }
  })

  it('unknown field keys default to visible (fail-open)', () => {
    expect(isFieldVisible('some_new_future_key', 'legacy')).toBe(true)
  })
})

describe('hiddenKeysForTier', () => {
  it('legacy hides everything fast hides, and more', () => {
    const legacy = hiddenKeysForTier('legacy')
    const fast = hiddenKeysForTier('fast')
    for (const k of fast) expect(legacy).toContain(k)
    expect(legacy.length).toBeGreaterThan(fast.length)
  })

  /**
   * Structural guard against the defect this file has now been corrected for
   * three times (TMM, trap profile / T / grading, dual ions): a field group
   * getting hidden from FAST although FAST's physics runs it.
   *
   * mode.py has exactly three flags that FAST leaves off — radiative
   * reabsorption, field-dependent mobility, selective contacts. Photon
   * recycling and thermionic emission are ON in FAST and carry no parameter
   * fields. So the ONLY keys FAST may hide are the per-RHS parameters. If a
   * future key group is added to FAST_HIDDEN, this fails and forces the author
   * back to the flag matrix.
   */
  it('fast hides only the per-RHS parameters that mode.py leaves off there', () => {
    expect(new Set(hiddenKeysForTier('fast'))).toEqual(new Set([
      'S_n_left', 'S_p_left', 'S_n_right', 'S_p_right',
      'v_sat_n', 'v_sat_p', 'ct_beta_n', 'ct_beta_p', 'pf_gamma_n', 'pf_gamma_p',
    ]))
  })

  it('full hides nothing', () => {
    expect(hiddenKeysForTier('full')).toEqual([])
  })
})

describe('isLayerBuilderEnabled', () => {
  it('returns true only for full tier', () => {
    expect(isLayerBuilderEnabled('full')).toBe(true)
    expect(isLayerBuilderEnabled('fast')).toBe(false)
    expect(isLayerBuilderEnabled('legacy')).toBe(false)
  })
})
