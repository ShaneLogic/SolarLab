/**
 * Vitest cases for ``renderImpedance`` / ``renderDegradation`` /
 * ``renderTPV`` / ``renderVocGrainSweep`` workstation panes.
 *
 * Covers publication styling, result evidence, and raw array preservation
 * using the shared Plotly mock pattern.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.mock('plotly.js-basic-dist-min', () => ({
  default: {
    newPlot: vi.fn(),
    purge: vi.fn(),
  },
  newPlot: vi.fn(),
  purge: vi.fn(),
}))

import {
  renderImpedance,
  renderDegradation,
  renderTPV,
  renderVocGrainSweep,
} from './main-plot-pane'
import type {
  ISResult,
  DegResult,
  TPVResult,
  VocGrainSweepResult,
} from '../../types'
import { PUBLICATION_FONT_FAMILY } from '../../plot-theme'
import Plotly from 'plotly.js-basic-dist-min'

const newPlotMock = vi.mocked(Plotly.newPlot)

function _layout(): Record<string, any> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  return calls[calls.length - 1][2] as Record<string, any>
}
function _traces(): Array<Record<string, any>> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  return calls[calls.length - 1][1] as Array<Record<string, any>>
}
function _config(): Record<string, any> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  return calls[calls.length - 1][3] as Record<string, any>
}

// ── Impedance (Nyquist) ──────────────────────────────────────────────────

function makeIS(): ISResult {
  return {
    frequencies: [1e2, 1e3, 1e4, 1e5],
    Z_real:      [120,  90,  60,  20],   // Ω·m²
    Z_imag:      [-15, -40, -55, -10],   // Ω·m² (raw, will be sign-flipped)
    protocol: {
      method: 'transient_ion_aware',
      V_dc: 0.9,
      delta_V: 0.01,
      illuminated: true,
      dc_settle_time: 1e-3,
      n_cycles: 5,
      n_extract: 2,
      points_per_cycle: 40,
    },
    operating_point: {
      certified: true,
      numerically_certified: true,
      thermodynamically_certified: true,
      source: 'finite_time_preconditioned',
      carrier_area_rate_A_m2: 1e-3,
      ion_area_rate_A_m2: 1e-8,
      max_ionic_face_current_A_m2: 1e-9,
      dc_face_current_spread_A_m2: 1e-3,
      carrier_area_rate_limit_A_m2: 1e-1,
      ion_area_rate_limit_A_m2: 1e-6,
      ionic_face_current_limit_A_m2: 1e-6,
      dc_face_current_spread_limit_A_m2: 1e-1,
      contact_thermodynamics: {
        status: 'certified',
        built_in_potential_mode: 'semiconductor_work_function',
        tolerance_eV: 5e-3,
        fermi_level_span_eV: 1e-4,
        potential_mismatch_V: 1e-4,
        metal_work_function_mismatch_eV: null,
        contact_quasi_fermi_levels_eV: [-4.8, -4.8],
        message: 'certified',
      },
      reasons: [],
    },
    frequency_window: {
      f_min_Hz: 1e2,
      f_max_Hz: 1e5,
      has_mobile_ions: true,
      characteristic_frequency_bracketed: true,
      ionic_branch_covered: true,
      ionic_timescales: [],
      warnings: [],
      full_timescale_envelope_bracketed: true,
      recommended_f_min_Hz: 1e2,
      recommended_f_max_Hz: 1e5,
      branch_margin_decades: 1,
      max_allowed_sampling_gap_decades: 1,
      max_observed_sampling_gap_decades: 1,
      ionic_branch_assessments: [],
    },
    grid_assessment: {
      certified: true,
      override_used: false,
      guarded_cell_count: 6,
      offender_count: 0,
      max_guarded_cell_debye_ratio: 0.25,
      max_cell_debye_ratio_limit: 0.5,
      warnings: [],
    },
    diagnostics: {
      admittance_S_m2: [{ real: 1e-3, imag: 2e-3 }],
      admittance_faces_S_m2: [[{ real: 1e-3, imag: 2e-3 }]],
      max_relative_face_spread: [1e-6],
      reciprocal_condition: [0.1],
      backward_error: [1e-12],
      electron_storage_response_F_m2: [{ real: 1e-5, imag: -1e-6 }],
      hole_storage_response_F_m2: [{ real: 2e-5, imag: -2e-6 }],
    },
  }
}

describe('renderImpedance', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders without a style selector', () => {
    renderImpedance(el, makeIS())
    expect(el.querySelector('[data-test="impedance-toolbar"]')).toBeNull()
    expect(el.querySelector('[data-test="impedance-style-mode"]')).toBeNull()
    expect(el.querySelector('[data-test="impedance-evidence-warning"]')).toBeNull()
    const summary = el.querySelector<HTMLElement>('[data-test="impedance-evidence-summary"]')
    expect(summary?.textContent).toContain('blocking-charge frequency bracketed')
    expect(summary?.textContent).toContain('full ionic envelope bracketed')
    expect(summary?.textContent).toContain('ionic branch covered')
    expect(summary?.textContent).toContain('recommended range: [100, 1.00e+5] Hz')
    expect(summary?.textContent).toContain('Grid: certified')
    expect(summary?.textContent).toContain('guarded cells: 6')
    expect(summary?.textContent).toContain('offenders: 0')
  })

  it('classifies a result without all four evidence blocks as legacy unclassified', () => {
    const legacy = makeIS()
    delete legacy.grid_assessment
    renderImpedance(el, legacy)
    const warning = el.querySelector<HTMLElement>(
      '[data-test="impedance-evidence-warning"]',
    )
    expect(warning?.textContent).toContain('Legacy impedance result')
    expect(warning?.textContent).toContain('unclassified')
  })

  it('surfaces DC and frequency-window evidence warnings', () => {
    const result = makeIS()
    result.operating_point = {
      ...result.operating_point!,
      certified: false,
      numerically_certified: false,
      thermodynamically_certified: false,
      reasons: ['ion_area_rate_exceeds_limit'],
    }
    result.frequency_window = {
      ...result.frequency_window!,
      characteristic_frequency_bracketed: false,
      full_timescale_envelope_bracketed: false,
      ionic_branch_covered: false,
      warnings: ['ionic_blocking_charge_frequency_not_bracketed'],
    }

    renderImpedance(el, result)

    const warning = el.querySelector<HTMLElement>(
      '[data-test="impedance-evidence-warning"]',
    )
    expect(warning).not.toBeNull()
    expect(warning!.textContent).toContain('ionic_blocking_charge_frequency_not_bracketed')
    expect(warning!.textContent).toContain('ion_area_rate_exceeds_limit')
  })

  it('surfaces grid warnings and evidence when the grid is uncertified', () => {
    const result = makeIS()
    result.grid_assessment = {
      ...result.grid_assessment!,
      certified: false,
      override_used: true,
      warnings: ['underresolved_interface_grid_override'],
    }

    renderImpedance(el, result)

    const warning = el.querySelector<HTMLElement>(
      '[data-test="impedance-evidence-warning"]',
    )
    const summary = el.querySelector<HTMLElement>(
      '[data-test="impedance-evidence-summary"]',
    )
    expect(warning?.textContent).toContain('underresolved_interface_grid_override')
    expect(summary?.textContent).toContain('Grid: uncertified')
    expect(summary?.textContent).toContain('override: used')
  })

  it('publication: hollow circle muted blue, modebar hidden, scaleanchor preserved', () => {
    renderImpedance(el, makeIS())
    const layout = _layout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.yaxis.scaleanchor).toBe('x')
    expect(_config()!.displayModeBar).toBe(false)
    const data = _traces()![0]
    expect(data.marker.symbol).toBe('circle-open')
    expect(data.line.color).toBe('#2B6FA3')
  })

  it('raw Z_real / Z_imag arrays unchanged across renders', () => {
    const r = makeIS()
    const re_pre = [...r.Z_real]
    const im_pre = [...r.Z_imag]
    renderImpedance(el, r)
    renderImpedance(el, r)
    expect(r.Z_real).toEqual(re_pre)
    expect(r.Z_imag).toEqual(im_pre)
    // x is the raw Z_real reference (no .slice() / .map()).
    expect(_traces()![0].x).toBe(r.Z_real)
  })
})

// ── Degradation (PCE / PCE_0 vs t) ───────────────────────────────────────

function makeDeg(): DegResult {
  return {
    times:   [0, 100, 1000, 10000],
    PCE:     [0.20, 0.195, 0.18, 0.16],
    V_oc:    [1.10, 1.09, 1.07, 1.04],
    J_sc:    [220, 218, 215, 210],
  }
}

describe('renderDegradation', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders without a style selector', () => {
    renderDegradation(el, makeDeg())
    expect(el.querySelector('[data-test="degradation-toolbar"]')).toBeNull()
    expect(el.querySelector('[data-test="degradation-style-mode"]')).toBeNull()
  })

  it('publication: hollow circle muted blue, modebar hidden', () => {
    renderDegradation(el, makeDeg())
    expect(_layout()!.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(_config()!.displayModeBar).toBe(false)
    const data = _traces()![0]
    expect(data.marker.symbol).toBe('circle-open')
    expect(data.line.color).toBe('#2B6FA3')
    expect(data.name).toContain('PCE')
  })

  it('raw PCE / times arrays unchanged across renders', () => {
    const r = makeDeg()
    const pce_pre = [...r.PCE]
    const t_pre = [...r.times]
    renderDegradation(el, r)
    renderDegradation(el, r)
    expect(r.PCE).toEqual(pce_pre)
    expect(r.times).toEqual(t_pre)
    expect(_traces()![0].x).toBe(r.times)
  })
})

// ── TPV (ΔV decay) ───────────────────────────────────────────────────────

function makeTPV(): TPVResult {
  return {
    t:        [0, 1e-6, 2e-6, 5e-6, 1e-5],
    V:        [1.105, 1.103, 1.1015, 1.1005, 1.1001],
    J:        [0, -0.5, -0.4, -0.2, -0.05],
    V_oc:     1.100,
    tau:      2.0e-6,
    delta_V0: 5.0e-3,
  }
}

describe('renderTPV', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders without a style selector', () => {
    renderTPV(el, makeTPV())
    expect(el.querySelector('[data-test="tpv-toolbar"]')).toBeNull()
    expect(el.querySelector('[data-test="tpv-style-mode"]')).toBeNull()
  })

  it('publication: solid muted-blue line (lines mode, no markers), font swap', () => {
    renderTPV(el, makeTPV())
    expect(_layout()!.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(_config()!.displayModeBar).toBe(false)
    const data = _traces()![0]
    expect(data.mode).toBe('lines')
    expect(data.line.color).toBe('#2B6FA3')
    expect(data.line.width).toBe(1.75)
    // TPV trace is a smooth decay → no markers in publication.
    expect(data.marker).toBeUndefined()
  })

  it('publication annotation: V_oc / τ / ΔV_0 stacked with <br> at upper-RIGHT', () => {
    renderTPV(el, makeTPV())
    const ann = _layout()!.annotations as Array<Record<string, any>>
    expect(ann[0].xanchor).toBe('right')
    expect(ann[0].yanchor).toBe('top')
    expect(ann[0].text).toContain('<br>')
    expect(ann[0].font.family).toBe(PUBLICATION_FONT_FAMILY)
  })

  it('raw t / V arrays unchanged across renders', () => {
    const r = makeTPV()
    const t_pre = [...r.t]
    const V_pre = [...r.V]
    renderTPV(el, r)
    renderTPV(el, r)
    expect(r.t).toEqual(t_pre)
    expect(r.V).toEqual(V_pre)
  })

  it.each(['engineering', 'publication'] as const)('uses the matched control with legacy %s style state', mode => {
    const result = makeTPV()
    result.V_reference = [1.104, 1.102, 1.1005, 1.0995, 1.0991]
    el.dataset.plotStyleMode = mode
    renderTPV(el, result)
    const voltage = _traces()![0].y as number[]
    voltage.forEach(value => expect(value).toBeCloseTo(1, 9))
  })

  it.each(['engineering', 'publication'] as const)('does not invent a lifetime with legacy %s style state', mode => {
    const result = makeTPV()
    result.tau = null
    result.fit = {
      tau_s: null, amplitude_V: 0, status: 'no_signal', points: 5,
      r_squared: null, normalized_max_error: null,
    }
    el.dataset.plotStyleMode = mode
    renderTPV(el, result)
    const annotation = _layout()!.annotations[0].text as string
    expect(annotation).toContain('No resolved pulse')
    expect(annotation).not.toContain('0.0 \u00B5s')
    expect(annotation).not.toContain('NaN')
  })

  it('rejects a lifetime attached to an invalid trajectory', () => {
    const result = makeTPV()
    result.valid = result.t.map(() => false)
    renderTPV(el, result)
    expect(_layout()!.annotations[0].text).toContain('Invalid response')
    expect(_layout()!.annotations[0].text).not.toContain('2.0 \u00B5s')
    expect(_traces()![0].y).toEqual(result.t.map(() => null))
  })
})

// ── V_oc(L_g) Grain Sweep ────────────────────────────────────────────────

function makeVGS(): VocGrainSweepResult {
  return {
    grain_sizes_nm: [10, 30, 100, 300, 1000],
    V_oc_V:         [0.85, 0.92, 0.99, 1.04, 1.07],
    J_sc_Am2:       [340, 360, 380, 395, 410],
    FF:             [0.62, 0.69, 0.74, 0.78, 0.81],
  }
}

describe('renderVocGrainSweep', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders without a style selector', () => {
    renderVocGrainSweep(el, makeVGS())
    expect(el.querySelector('[data-test="voc-grain-sweep-toolbar"]')).toBeNull()
    expect(el.querySelector('[data-test="voc-grain-sweep-style-mode"]')).toBeNull()
  })

  it('publication: log x-axis with dtick=1 + hollow circle muted blue', () => {
    renderVocGrainSweep(el, makeVGS())
    const layout = _layout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.xaxis.type).toBe('log')
    expect(layout.xaxis.dtick).toBe(1)
    const data = _traces()![0]
    expect(data.marker.symbol).toBe('circle-open')
    expect(data.line.color).toBe('#2B6FA3')
    expect(data.name).toContain('V<sub>oc</sub>(L<sub>g</sub>)')
  })

  it('publication annotation lives at lower-RIGHT (rising curve, empty quadrant)', () => {
    renderVocGrainSweep(el, makeVGS())
    const ann = _layout()!.annotations as Array<Record<string, any>>
    expect(ann).toHaveLength(1)
    expect(ann[0].xanchor).toBe('right')
    expect(ann[0].yanchor).toBe('bottom')
    // Per-grain table format with HTML subscripts.
    expect(ann[0].text).toContain('L<sub>g</sub>')
    expect(ann[0].text).toContain('<br>')
    expect(ann[0].font.family).toBe(PUBLICATION_FONT_FAMILY)
  })

  it('raw arrays unchanged across renders', () => {
    const r = makeVGS()
    const g_pre = [...r.grain_sizes_nm]
    const v_pre = [...r.V_oc_V]
    renderVocGrainSweep(el, r)
    renderVocGrainSweep(el, r)
    expect(r.grain_sizes_nm).toEqual(g_pre)
    expect(r.V_oc_V).toEqual(v_pre)
    expect(_traces()![0].x).toBe(r.grain_sizes_nm)
  })
})
