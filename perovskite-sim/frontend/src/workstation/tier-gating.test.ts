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

  it('hides TMM-only fields in legacy and fast', () => {
    expect(isFieldVisible('optical_material', 'legacy')).toBe(false)
    expect(isFieldVisible('optical_material', 'fast')).toBe(false)
    expect(isFieldVisible('optical_material', 'full')).toBe(true)
  })

  it('incoherent field is hidden in legacy and fast, visible in full', () => {
    expect(isFieldVisible('incoherent', 'legacy')).toBe(false)
    expect(isFieldVisible('incoherent', 'fast')).toBe(false)
    expect(isFieldVisible('incoherent', 'full')).toBe(true)
  })

  it('hides dual-ion fields in legacy and fast', () => {
    expect(isFieldVisible('D_ion_neg', 'legacy')).toBe(false)
    expect(isFieldVisible('D_ion_neg', 'fast')).toBe(false)
    expect(isFieldVisible('D_ion_neg', 'full')).toBe(true)
    expect(isFieldVisible('P_lim_neg', 'legacy')).toBe(false)
  })

  it('hides trap-profile fields in legacy and fast', () => {
    expect(isFieldVisible('trap_N_t_interface', 'legacy')).toBe(false)
    expect(isFieldVisible('trap_N_t_bulk', 'legacy')).toBe(false)
    expect(isFieldVisible('trap_N_t_interface', 'full')).toBe(true)
  })

  it('hides device-level T input in legacy (fixed 300 K)', () => {
    expect(isFieldVisible('T', 'legacy')).toBe(false)
    expect(isFieldVisible('T', 'fast')).toBe(false)
    expect(isFieldVisible('T', 'full')).toBe(true)
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
  it('legacy hides everything fast hides plus T', () => {
    const legacy = hiddenKeysForTier('legacy')
    const fast = hiddenKeysForTier('fast')
    for (const k of fast) expect(legacy).toContain(k)
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
