/**
 * Vitest cases for ``renderJV`` (1D J-V workstation pane).
 *
 * Mirrors the 2D pane test pattern: Plotly is mocked because jsdom
 * has no canvas, so layout / trace / config arguments are read from
 * ``newPlot.mock.calls``. Publication styling and metric selection
 * must preserve the raw V/J arrays.
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

import { renderJV } from './main-plot-pane'
import type { InterfaceChargeJVEvidence, JVResult, JVMetrics } from '../../types'
import { PUBLICATION_FONT_FAMILY } from '../../plot-theme'
import Plotly from 'plotly.js-basic-dist-min'

const newPlotMock = vi.mocked(Plotly.newPlot)

function makeMetrics(overrides: Partial<JVMetrics> = {}): JVMetrics {
  return {
    V_oc: 0.951,
    J_sc: 220.0,
    FF: 0.823,
    PCE: 0.1722,
    voc_bracketed: true,
    ...overrides,
  }
}

function makeResult(overrides: Partial<JVResult> = {}): JVResult {
  return {
    V_fwd: [0.0, 0.4, 0.8, 1.0],
    J_fwd: [-220.0, -180.0, -50.0, +120.0],   // 1D: J<0 at V=0 with photocurrent / signed solar convention
    V_rev: [1.0, 0.8, 0.4, 0.0],
    J_rev: [+115.0, -55.0, -185.0, -222.0],
    metrics_fwd: makeMetrics(),
    metrics_rev: makeMetrics({ V_oc: 0.948, J_sc: 218.0, FF: 0.815, PCE: 0.1685 }),
    hysteresis_index: 0.012,
    ...overrides,
  }
}

function makeInterfaceChargeEvidence(): InterfaceChargeJVEvidence {
  return {
    model: 'interface-charge-jv-evidence-v1',
    capability: 'equilibrium_referenced_interface_charge_qf_dc_v1',
    protocol: {
      voltages_V: [0, 0.05, 0.1], temperature_K: 300, P_in_W_m2: 1000,
      solver_controls: {
        illumination_steps: [0, 1], finite_difference_step: 7e-6,
        newton_residual_tolerance: 4e-7, max_newton_iterations: 60,
        poisson_tolerance_V: 1e-12, poisson_max_iterations: 100,
        continuity_tolerance_A_m2: 1e-4, current_spread_tolerance_A_m2: 1e-4,
        poisson_residual_tolerance: 1e-8, minimum_voltage_step_V: 1e-3,
        max_voltage_bridge_points: 256,
      },
      acceptance: {
        max_normalized_cell_residual: 4e-7, max_interface_local_residual: 1e-7,
        max_normalized_gauss_residual: 1e-10,
        max_scaled_local_jacobian_condition: 1e8,
        max_continuity_bound_A_m2: 1e-4, max_face_current_spread_A_m2: 1e-4,
        max_poisson_residual: 1e-8, max_contact_fermi_level_span_eV: 5e-3,
        require_contact_thermodynamic_certificate: true,
        require_dark_charge_off_bit_identity: true, require_voc_bracket: true,
      },
      capability: 'equilibrium_referenced_interface_charge_qf_dc_v1',
      solver: 'quasi_fermi', illumination: 'stack_baseline_one_sun',
      branch_semantics: 'ascending_zero_scan_rate',
      initial_state_source: 'certified_charge_off_dark_reference',
      interface_topology: 'two_sided_trace',
      interface_transport_model: 'fermi_dirac_richardson',
      interface_transmission: 1, charge_law: '-q*N_t*(f-f_eq)',
      stop_after_voc: true, mpp_interpolation: 'sampled',
      schema_version: 'interface-charge-jv-protocol-v1',
    },
    protocol_sha256: 'a'.repeat(64), grid_sha256: 'b'.repeat(64),
    stack_sha256: 'c'.repeat(64), dark_state_sha256: 'd'.repeat(64),
    dark_contact_thermodynamic_status: 'certified',
    dark_contact_fermi_level_span_eV: 0,
    interface_defect_document_sha256: ['e'.repeat(64)],
    capture_velocities_m_s: [[0.03, 0.05]], trap_density_m2: [1e17],
    equilibrium_occupancy: [0.7], dark_charge_off_bit_identity_verified: true,
    points: [] as never[], continuation_bridges: [] as never[],
    continuation_bridge_count: 1, tolerance_factor: 1,
    minimum_occupancy: 0.7139, maximum_occupancy: 0.7194,
    maximum_absolute_sheet_charge_C_m2: 8.73e-5,
    maximum_absolute_trace_potential_shift_V: 2.41e-4,
    maximum_normalized_gauss_residual: 3.16e-16,
    maximum_scaled_local_jacobian_condition: 1.19e4,
    maximum_interface_local_residual: 1.27e-13,
    maximum_normalized_cell_residual: 3.12e-7,
    maximum_continuity_bound_A_m2: 3.13e-7,
    maximum_face_current_spread_A_m2: 3.12e-7,
    maximum_poisson_residual: 4.59e-15,
    maximum_contact_fermi_level_span_eV: 1e-5,
    limitations: ['ion-free'],
  }
}

function _lastNewPlotLayout(): Record<string, any> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  return calls[calls.length - 1][2] as Record<string, any>
}

function _lastNewPlotTraces(): Array<Record<string, any>> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  return calls[calls.length - 1][1] as Array<Record<string, any>>
}

function _lastNewPlotConfig(): Record<string, any> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  return calls[calls.length - 1][3] as Record<string, any>
}

describe('renderJV — controls and evidence', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders without a style selector', () => {
    renderJV(el, makeResult())
    expect(el.querySelector('[data-test="jv1d-toolbar"]')).toBeNull()
    expect(el.querySelector('[data-test="jv1d-style-mode"]')).toBeNull()
  })

  it.each([undefined, 'engineering', 'publication', 'unknown'])(
    'keeps publication styling across renders with legacy state %s', style => {
      if (style !== undefined) el.dataset.plotStyleMode = style
      const result = makeResult()
      renderJV(el, result)
      renderJV(el, result)
      expect(_lastNewPlotLayout()!.font.family).toBe(PUBLICATION_FONT_FAMILY)
      expect(_lastNewPlotConfig()!.displayModeBar).toBe(false)
      expect(el.querySelector('.plot-style-select')).toBeNull()
      expect(el.querySelectorAll('.jv1d-plot')).toHaveLength(1)
    },
  )

  it('omits the explicit-defect strip for legacy J-V results', () => {
    renderJV(el, makeResult())
    expect(el.querySelector('[data-test="jv-defect-evidence-summary"]')).toBeNull()
  })

  it('renders QF/DC model identity and constitutive extrema without changing traces', () => {
    const digest = 'a'.repeat(64)
    renderJV(el, makeResult({
      bulk_defect_evidence: {
        model: 'monovalent-device-mb-qf-dc-v1',
        model_identity_sha256: digest,
        species_identifiers: ['V_I', 'I_i'],
        charge_transitions: ['acceptor', 'donor'],
        points_completed: 11,
        minimum_occupancy: 0.0123,
        maximum_occupancy: 0.9876,
        minimum_kinetic_denominator_s1: 2.5e5,
        maximum_absolute_charge_density_C_m3: 4.2e3,
        maximum_absolute_recombination_rate_m3_s: 8.1e27,
        spatial_closure: 'layer-density-profile-v1',
        spatial_profile_sha256s: ['b'.repeat(64), null],
        minimum_density_multipliers: [0.5, 1.0],
        maximum_density_multipliers: [1.5, 1.0],
      },
    }))
    const summary = el.querySelector<HTMLElement>('[data-test="jv-defect-evidence-summary"]')!
    expect(summary).not.toBeNull()
    expect(summary.textContent).toContain('2 species / 11 points')
    expect(summary.textContent).toContain('aaaaaaaaaaaa...')
    expect(summary.textContent).toContain('Occupancy: [0.0123, 0.9876]')
    expect(summary.textContent).toContain('4.200e+3 C m^-3')
    expect(summary.textContent).toContain('1/2 profiled species')
    expect(summary.textContent).toContain('m(x) [0.500, 1.500]')
    expect(summary.title).toContain(digest)
    expect(_lastNewPlotTraces()).toHaveLength(2)
  })

  it('renders charged evidence and one zero-scan curve with publication styling', () => {
    const evidence = makeInterfaceChargeEvidence()
    evidence.points.push({} as never, {} as never, {} as never)
    renderJV(el, makeResult({ interface_charge_evidence: evidence }))

    const summary = el.querySelector<HTMLElement>(
      '[data-test="interface-charge-jv-evidence-summary"]',
    )!
    expect(summary.textContent).toContain('3 requested points + 1 bridges')
    expect(summary.textContent).toContain('aaaaaaaaaaaa...')
    expect(summary.textContent).toContain('Max Gauss 3.160e-16')
    expect(summary.title).toContain(`dark sha256:${'d'.repeat(64)}`)
    expect(_lastNewPlotTraces()).toHaveLength(1)
    expect(_lastNewPlotTraces()![0].name).toBe('Charged QF/DC')

    expect(_lastNewPlotTraces()).toHaveLength(1)
    const annotation = (_lastNewPlotLayout()!.annotations as Array<{ text: string }>)[0]
    expect(annotation.text).toContain('Charged QF/DC')
  })
})

describe('renderJV — publication rendering', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('uses publication by default with Nature-style layout', () => {
    renderJV(el, makeResult())
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.plot_bgcolor).toBe('#ffffff')
    expect(layout.margin).toEqual({ t: 18, r: 18, b: 48, l: 58 })
    expect(layout.xaxis.showgrid).toBe(false)
    expect(layout.yaxis.showgrid).toBe(false)
    expect(layout.yaxis.zeroline).toBe(true)
  })

  it('publication mode hides the Plotly modebar', () => {
    renderJV(el, makeResult())
    expect(_lastNewPlotConfig()!.displayModeBar).toBe(false)
  })

  it('publication forward trace: hollow circle, muted blue, lines+markers', () => {
    renderJV(el, makeResult())
    const traces = _lastNewPlotTraces()!
    const fwd = traces[0]
    expect(fwd.name).toBe('Forward')
    expect(fwd.mode).toBe('lines+markers')
    expect(fwd.marker.symbol).toBe('circle-open')
    expect(fwd.marker.color).toBe('rgba(0,0,0,0)')
    expect(fwd.marker.line.color).toBe('#2B6FA3')
    expect(fwd.marker.line.width).toBe(1.2)
    expect(fwd.marker.size).toBe(5)
    expect(fwd.line.color).toBe('#2B6FA3')
    expect(fwd.line.width).toBe(1.75)
  })

  it('publication reverse trace: hollow circle, muted red, dashed line', () => {
    renderJV(el, makeResult())
    const rev = _lastNewPlotTraces()![1]
    expect(rev.name).toBe('Reverse')
    expect(rev.mode).toBe('lines+markers')
    expect(rev.marker.symbol).toBe('circle-open')
    expect(rev.marker.color).toBe('rgba(0,0,0,0)')
    expect(rev.marker.line.color).toBe('#C44536')
    expect(rev.line.color).toBe('#C44536')
    expect(rev.line.dash).toBe('dash')
    expect(rev.line.width).toBe(1.75)
  })

  it('both bracketed: prefer forward metrics, label "Forward"', () => {
    // Default fixture has metrics_fwd.voc_bracketed=true AND
    // metrics_rev.voc_bracketed=true (both bracket). Annotation must
    // prefer forward.
    renderJV(el, makeResult())
    const annotations = _lastNewPlotLayout()!.annotations as Array<{ text: string }>
    expect(annotations).toHaveLength(1)
    const text = annotations[0].text
    expect(text.startsWith('<b>Forward:</b><br>')).toBe(true)
    expect(text).toContain('0.951 V')
    expect(text).toContain('22.00 mA cm⁻²')
    expect(text).toContain('82.3%')
    expect(text).toContain('17.22%')
    // Reverse-only numerics must NOT leak into the annotation text.
    expect(text).not.toContain('0.948 V')
    expect(text).not.toContain('21.80')
  })

  it('forward-only bracketed: label "Forward", forward metrics', () => {
    renderJV(el, makeResult({
      metrics_fwd: { V_oc: 0.951, J_sc: 220.0, FF: 0.823, PCE: 0.1722, voc_bracketed: true },
      metrics_rev: { V_oc: 0.0, J_sc: 218.0, FF: 0.0, PCE: 0.0, voc_bracketed: false },
    }))
    const text = (_lastNewPlotLayout()!.annotations as Array<{ text: string }>)[0].text
    expect(text.startsWith('<b>Forward:</b><br>')).toBe(true)
    expect(text).toContain('0.951 V')
    expect(text).toContain('22.00 mA cm⁻²')
    expect(text).not.toContain('not bracketed')
  })

  it('reverse-only bracketed: label "Reverse", reverse metrics', () => {
    // Hysteretic case: forward fails to bracket V_oc, reverse brackets.
    renderJV(el, makeResult({
      metrics_fwd: { V_oc: 0.0, J_sc: 405.0, FF: 0.0, PCE: 0.0, voc_bracketed: false },
      metrics_rev: { V_oc: 0.860, J_sc: 404.9, FF: 0.865, PCE: 0.3011, voc_bracketed: true },
    }))
    const annotations = _lastNewPlotLayout()!.annotations as Array<{ text: string }>
    expect(annotations).toHaveLength(1)
    const text = annotations[0].text
    expect(text.startsWith('<b>Reverse:</b><br>')).toBe(true)
    expect(text).toContain('0.860 V')
    expect(text).toContain('40.49 mA cm⁻²')
    expect(text).toContain('86.5%')
    expect(text).toContain('30.11%')
    // No "not bracketed" — we picked the bracketed (reverse) sweep.
    expect(text).not.toContain('not bracketed')
    // Forward sentinel-zero values must NOT leak in.
    expect(text).not.toContain('0.000 V')
    expect(text).not.toContain('0.0%')
  })

  it('reverse-only bracketed: y-range tightens around metrics_rev.J_sc', () => {
    // Range source must follow the picked sweep so the publication
    // panel is internally consistent. metrics_rev.J_sc=400 → 40 mA/cm²
    // → tight range = [-0.15·40, +1.12·40] = [-6.0, +44.8].
    renderJV(el, makeResult({
      metrics_fwd: { V_oc: 0.0, J_sc: 405.0, FF: 0.0, PCE: 0.0, voc_bracketed: false },
      metrics_rev: { V_oc: 0.860, J_sc: 400.0, FF: 0.865, PCE: 0.3011, voc_bracketed: true },
    }))
    const layout = _lastNewPlotLayout()!
    const [ymin, ymax] = layout.yaxis.range as [number, number]
    expect(ymin).toBeCloseTo(-6.0, 6)
    expect(ymax).toBeCloseTo(+44.8, 6)
  })

  it('reverse-only bracketed: x-range capped at metrics_rev.V_oc + 0.18', () => {
    // metrics_rev.V_oc = 0.860 → cap at 1.040 V. Sweep extends to
    // V=1.0, so max(V)+0.05 = 1.05. min(cap, max+0.05) = 1.040.
    renderJV(el, makeResult({
      metrics_fwd: { V_oc: 0.0, J_sc: 405.0, FF: 0.0, PCE: 0.0, voc_bracketed: false },
      metrics_rev: { V_oc: 0.860, J_sc: 400.0, FF: 0.865, PCE: 0.3011, voc_bracketed: true },
    }))
    const [xmin, xmax] = _lastNewPlotLayout()!.xaxis.range as [number, number]
    expect(xmin).toBeCloseTo(-0.05, 6)
    expect(xmax).toBeCloseTo(+1.040, 6)
  })

  it('both bracketed: axis bounds include both branches', () => {
    // Reverse sets the current envelope; forward sets the larger V_oc.
    renderJV(el, makeResult({
      metrics_fwd: { V_oc: 0.951, J_sc: 200.0, FF: 0.823, PCE: 0.1722, voc_bracketed: true },
      metrics_rev: { V_oc: 0.860, J_sc: 400.0, FF: 0.865, PCE: 0.3011, voc_bracketed: true },
    }))
    const layout = _lastNewPlotLayout()!
    const [ymin, ymax] = layout.yaxis.range as [number, number]
    expect(ymin).toBeCloseTo(-6.0, 6)
    expect(ymax).toBeCloseTo(+44.8, 6)
    const [, xmax] = layout.xaxis.range as [number, number]
    // V_oc + 0.18 = 1.131; max(V) + 0.05 = 1.05 → cap at 1.05.
    expect(xmax).toBeCloseTo(1.05, 6)
  })

  it('keeps the reverse operating region visible when forward power collapses', () => {
    renderJV(el, makeResult({
      metrics_fwd: { V_oc: 0.25, J_sc: 10, FF: 0.32, PCE: 0.0008, voc_bracketed: true },
      metrics_rev: { V_oc: 0.78, J_sc: 160, FF: 0.63, PCE: 0.0783, voc_bracketed: true },
    }))
    const layout = _lastNewPlotLayout()!
    expect(layout.yaxis.range[1]).toBeCloseTo(17.92, 6)
    expect(layout.xaxis.range[1]).toBeCloseTo(0.96, 6)
    expect(layout.annotations[0].text).toContain('Forward:')
    expect(layout.annotations[0].text).toContain('0.250 V')
    expect(layout.annotations[0].y).toBe(0.72)
  })

  it('shows the photovoltaic quadrant by default and preserves full-history data', () => {
    const result = makeResult({
      V_fwd: [-1, 0, 0.5, 1.2], J_fwd: [160, 10, -30, -1000],
      V_rev: [1.2, 0.5, 0, -1], J_rev: [-1000, 150, 160, 165],
    })
    renderJV(el, result)
    expect(_lastNewPlotLayout()!.xaxis.range[0]).toBe(-0.05)
    const originalTraces = structuredClone(_lastNewPlotTraces())
    const range = el.querySelector<HTMLSelectElement>('[data-test="jv1d-range-mode"]')!
    range.value = 'full'
    range.dispatchEvent(new Event('change'))
    expect(_lastNewPlotLayout()!.xaxis.range).toBeUndefined()
    expect(_lastNewPlotLayout()!.yaxis.range).toBeUndefined()
    expect(_lastNewPlotTraces()).toEqual(originalTraces)
    renderJV(el, result)
    expect(el.querySelector<HTMLSelectElement>('[data-test="jv1d-range-mode"]')!.value).toBe('full')
  })

  it('neither bracketed: autorange y/x; annotation Forward + "not bracketed"', () => {
    renderJV(el, makeResult({
      metrics_fwd: { V_oc: 0.0, J_sc: 220.0, FF: 0.0, PCE: 0.0, voc_bracketed: false },
      metrics_rev: { V_oc: 0.0, J_sc: 218.0, FF: 0.0, PCE: 0.0, voc_bracketed: false },
    }))
    const layout = _lastNewPlotLayout()!
    // y-range: autoranged because the picked sweep is not bracketed.
    expect(layout.yaxis.range).toBeUndefined()
    // x-range: still applies the -0.05 V left margin and max(V)+0.05
    // cap (no V_oc cap because nothing is bracketed).
    const [xmin, xmax] = layout.xaxis.range as [number, number]
    expect(xmin).toBeCloseTo(-0.05, 6)
    expect(xmax).toBeCloseTo(1.05, 6)   // max(V)=1.0 + 0.05
    // Annotation falls back to Forward + "not bracketed" via picker rule 3.
    const ann = layout.annotations as Array<{ text: string }>
    expect(ann).toHaveLength(1)
    expect(ann[0].text.startsWith('<b>Forward:</b><br>')).toBe(true)
    expect(ann[0].text).toContain('not bracketed')
  })

  it('legacy/undefined payload: autorange y, no annotation, publication style still applies', () => {
    renderJV(el, makeResult({
      metrics_fwd: { V_oc: 0.95, J_sc: 220, FF: 0.82, PCE: 0.17 },     // no voc_bracketed
      metrics_rev: { V_oc: 0.94, J_sc: 218, FF: 0.81, PCE: 0.165 },
    }))
    const layout = _lastNewPlotLayout()!
    // y-range autoranged because no sweep was picked.
    expect(layout.yaxis.range).toBeUndefined()
    // Annotation suppressed (no voc_bracketed → no fake numbers).
    expect((layout.annotations as Array<unknown>) ?? []).toHaveLength(0)
    // Publication style still applies.
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.xaxis.showgrid).toBe(false)
  })

  it('neither bracketed: label "Forward", "V_oc: not bracketed", J_sc only', () => {
    renderJV(el, makeResult({
      metrics_fwd: { V_oc: 0.0, J_sc: 220.0, FF: 0.0, PCE: 0.0, voc_bracketed: false },
      metrics_rev: { V_oc: 0.0, J_sc: 218.0, FF: 0.0, PCE: 0.0, voc_bracketed: false },
    }))
    const text = (_lastNewPlotLayout()!.annotations as Array<{ text: string }>)[0].text
    expect(text.startsWith('<b>Forward:</b><br>')).toBe(true)
    expect(text).toContain('not bracketed')
    expect(text).toContain('22.00 mA cm⁻²')   // metrics_fwd.J_sc / 10
    // No fake V_oc / FF / PCE.
    expect(text).not.toContain('0.000 V')
    expect(text).not.toContain('0.0%')
    expect(text).not.toContain('0.00%')
  })

  it('voc_bracketed=undefined (legacy payload): publication style applies, annotation omitted', () => {
    renderJV(el, makeResult({
      metrics_fwd: { V_oc: 0.95, J_sc: 220, FF: 0.82, PCE: 0.17 },     // no voc_bracketed
      metrics_rev: { V_oc: 0.94, J_sc: 218, FF: 0.81, PCE: 0.165 },
    }))
    const layout = _lastNewPlotLayout()!
    // Style still applies — Helvetica/Arial, white bg, no grid.
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    // Annotation is suppressed (no voc_bracketed → no fake numbers).
    const annotations = (layout.annotations as Array<unknown>) ?? []
    expect(annotations).toHaveLength(0)
  })

  it('publication operational y-range follows J_sc', () => {
    // Envelope = [-0.15·J_sc_mA, +1.12·J_sc_mA]. J_sc=220 → 22 mA/cm².
    // → publication range ≈ [-3.30, +24.64].
    renderJV(el, makeResult())
    const range = _lastNewPlotLayout()!.yaxis.range as [number, number]
    expect(range[0]).toBeCloseTo(-3.30, 6)
    expect(range[1]).toBeCloseTo(+24.64, 6)
  })

  it('publication x-axis: -0.05 V left margin when sweep starts at V=0', () => {
    renderJV(el, makeResult())
    const layout = _lastNewPlotLayout()!
    const [xmin, xmax] = layout.xaxis.range as [number, number]
    expect(xmin).toBeCloseTo(-0.05, 6)
    // V_oc + 0.18 = 1.131; max(V) + 0.05 = 1.05 → cap at 1.05.
    expect(xmax).toBeCloseTo(1.05, 6)
    expect(layout.xaxis.zeroline).toBe(true)
    expect(layout.xaxis.zerolinecolor).toBe('#000000')
  })

  it('publication x-axis: capped at V_oc + 0.18 when sweep extends past V_oc', () => {
    renderJV(el, makeResult({
      V_fwd: [0.0, 0.5, 1.0, 1.5],
      J_fwd: [-220.0, -180.0, -50.0, +400.0],
      V_rev: [1.5, 1.0, 0.5, 0.0],
      J_rev: [+395.0, -55.0, -185.0, -222.0],
    }))
    // V_oc + 0.18 = 0.951 + 0.18 = 1.131 < max(V) + 0.05 = 1.55
    const [, xmax] = _lastNewPlotLayout()!.xaxis.range as [number, number]
    expect(xmax).toBeCloseTo(1.131, 6)
  })

  it('raw V/J trace data are unchanged across renders (no mutation, byte-identical)', () => {
    const result = makeResult()
    const V_fwd_pre = [...result.V_fwd]
    const J_fwd_pre = [...result.J_fwd]
    const V_rev_pre = [...result.V_rev]
    const J_rev_pre = [...result.J_rev]
    // Initial render.
    renderJV(el, result)
    const tracesInitial = _lastNewPlotTraces()!
    const yInitialFwd = (tracesInitial[0].y as number[]).slice()
    const yInitialRev = (tracesInitial[1].y as number[]).slice()
    // Repeated render.
    renderJV(el, result)
    const tracesRepeated = _lastNewPlotTraces()!
    const yRepeatedFwd = (tracesRepeated[0].y as number[]).slice()
    const yRepeatedRev = (tracesRepeated[1].y as number[]).slice()
    // Trace y arrays equal between renders (post-flip-and-scale).
    expect(yRepeatedFwd).toEqual(yInitialFwd)
    expect(yRepeatedRev).toEqual(yInitialRev)
    // Forward traces use raw r.V_fwd reference for x — must remain identical.
    expect(tracesInitial[0].x).toBe(result.V_fwd)
    expect(tracesRepeated[0].x).toBe(result.V_fwd)
    // Raw input arrays remain bit-identical.
    expect(result.V_fwd).toEqual(V_fwd_pre)
    expect(result.J_fwd).toEqual(J_fwd_pre)
    expect(result.V_rev).toEqual(V_rev_pre)
    expect(result.J_rev).toEqual(J_rev_pre)
  })

  it('publication legend sits at upper-RIGHT (avoids overlapping J(V=0) plateau)', () => {
    // 1D J-V curves enter the panel from the upper-LEFT corner where
    // (V=0, J=+J_sc) lives. publicationLayout's default legend at
    // (0.02, 0.98) → upper-left would overlap the curve plateau.
    // 1D pane must override to upper-RIGHT (the curves exit toward
    // the lower-right at V_oc, leaving the upper-right empty).
    renderJV(el, makeResult())
    const layout = _lastNewPlotLayout()!
    expect(layout.legend.x).toBe(0.98)
    expect(layout.legend.y).toBe(0.98)
    expect(layout.legend.xanchor).toBe('right')
    expect(layout.legend.yanchor).toBe('top')
    // Override re-pins the publication transparent / borderless styling
    // (publicationLayout spreads overrides last, so an incomplete
    // legend object would silently restore Plotly defaults).
    expect(layout.legend.bgcolor).toBe('rgba(255,255,255,0)')
    expect(layout.legend.borderwidth).toBe(0)
  })
})
