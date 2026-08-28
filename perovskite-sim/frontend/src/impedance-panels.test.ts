import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  startJob: vi.fn(),
  streamJobEvents: vi.fn(),
  mountDevicePanel: vi.fn(),
}))

vi.mock('./job-stream', () => ({
  startJob: mocks.startJob,
  streamJobEvents: mocks.streamJobEvents,
}))

vi.mock('./device-panel', () => ({
  mountDevicePanel: mocks.mountDevicePanel,
}))

vi.mock('plotly.js-basic-dist-min', () => ({
  default: {
    newPlot: vi.fn(),
    purge: vi.fn(),
  },
}))

import { mountImpedancePanel } from './panels/impedance'
import { mountImpedancePane } from './workstation/panes/impedance-pane'
import {
  collectImpedanceEvidenceWarnings,
  summarizeImpedanceEvidence,
} from './impedance-evidence'
import type { DeviceConfig, ISResult, JobStreamHandlers } from './types'

const device: DeviceConfig = {
  device: { Phi: 0 },
  layers: [],
}

let container: HTMLDivElement

beforeEach(() => {
  document.body.replaceChildren()
  container = document.createElement('div')
  document.body.appendChild(container)
  mocks.startJob.mockReset()
  mocks.startJob.mockResolvedValue('job-1')
  mocks.streamJobEvents.mockReset()
  mocks.mountDevicePanel.mockReset()
  mocks.mountDevicePanel.mockResolvedValue({ getConfig: () => device })
})

afterEach(() => {
  document.body.replaceChildren()
})

describe('impedance points-per-cycle controls', () => {
  it('panel preserves the transient default and forwards time resolution', async () => {
    await mountImpedancePanel(container)
    const points = document.getElementById('is-ppc') as HTMLInputElement
    expect(points.value).toBe('40')
    expect(points.disabled).toBe(false)
    points.value = '64'

    ;(document.getElementById('btn-is') as HTMLButtonElement).click()
    await Promise.resolve()

    expect(mocks.startJob).toHaveBeenCalledWith(
      'impedance',
      device,
      expect.objectContaining({
        N_grid: 40,
        n_freq: 15,
        f_min: 10,
        f_max: 1e5,
        method: 'transient_ion_aware',
        points_per_cycle: 64,
        require_frequency_window_certificate: false,
      }),
    )
  })

  it('workstation pane preserves the transient default', async () => {
    mountImpedancePane(container, {
      getActiveDevice: () => ({ id: 'device-1', config: device }),
      onRunComplete: vi.fn(),
    })
    const points = document.getElementById('imp-ppc') as HTMLInputElement
    expect(points.value).toBe('40')
    expect(points.disabled).toBe(false)

    ;(document.getElementById('btn-imp') as HTMLButtonElement).click()
    await Promise.resolve()

    expect(mocks.startJob).toHaveBeenCalledWith(
      'impedance',
      device,
      expect.objectContaining({
        N_grid: 40,
        n_freq: 15,
        f_min: 10,
        f_max: 1e5,
        method: 'transient_ion_aware',
        points_per_cycle: 40,
        require_frequency_window_certificate: false,
      }),
    )
  })

  it('applies the certified low-frequency preset only when selected', async () => {
    await mountImpedancePanel(container)
    const method = document.getElementById('is-method') as HTMLSelectElement
    method.value = 'ion_aware_frequency_certified'
    method.dispatchEvent(new Event('change'))

    expect((document.getElementById('is-N') as HTMLInputElement).value).toBe('60')
    expect((document.getElementById('is-fmin') as HTMLInputElement).value).toBe('0.000001')
    expect((document.getElementById('is-fmax') as HTMLInputElement).value).toBe('10')
    expect((document.getElementById('is-nf') as HTMLInputElement).value).toBe('29')
    expect((document.getElementById('is-ppc') as HTMLInputElement).disabled).toBe(true)
    expect(
      (document.getElementById('is-window-strict') as HTMLInputElement).disabled,
    ).toBe(false)

    ;(document.getElementById('btn-is') as HTMLButtonElement).click()
    await Promise.resolve()
    expect(mocks.startJob).toHaveBeenCalledWith(
      'impedance',
      device,
      expect.objectContaining({
        method: 'ion_aware_frequency_certified',
        require_frequency_window_certificate: true,
      }),
    )
  })

  it('restores safe high-frequency defaults when transient is reselected', async () => {
    await mountImpedancePanel(container)
    const method = document.getElementById('is-method') as HTMLSelectElement
    method.value = 'ion_aware_frequency_certified'
    method.dispatchEvent(new Event('change'))
    method.value = 'transient_ion_aware'
    method.dispatchEvent(new Event('change'))

    expect((document.getElementById('is-fmin') as HTMLInputElement).value).toBe('10')
    expect((document.getElementById('is-fmax') as HTMLInputElement).value).toBe('100000')
    expect((document.getElementById('is-nf') as HTMLInputElement).value).toBe('15')
    expect((document.getElementById('is-ppc') as HTMLInputElement).disabled).toBe(false)
    expect(
      (document.getElementById('is-window-strict') as HTMLInputElement).disabled,
    ).toBe(true)

    ;(document.getElementById('btn-is') as HTMLButtonElement).click()
    await Promise.resolve()
    expect(mocks.startJob).toHaveBeenCalledWith(
      'impedance',
      device,
      expect.objectContaining({
        method: 'transient_ion_aware',
        require_frequency_window_certificate: false,
      }),
    )
  })

  it('panel applies and forwards the dynamic-defect production contract', async () => {
    await mountImpedancePanel(container)
    const method = document.getElementById('is-method') as HTMLSelectElement
    method.value = 'dynamic_defect_frequency_certified'
    method.dispatchEvent(new Event('change'))

    expect((document.getElementById('is-N') as HTMLInputElement).value).toBe('12')
    expect((document.getElementById('is-nf') as HTMLInputElement).value).toBe('33')
    expect((document.getElementById('is-fmin') as HTMLInputElement).value).toBe('0.0001')
    expect((document.getElementById('is-fmax') as HTMLInputElement).value).toBe('1000000000000')
    expect((document.getElementById('is-ppc') as HTMLInputElement).disabled).toBe(true)
    expect((document.getElementById('is-defect-order') as HTMLInputElement).disabled).toBe(false)
    expect((document.getElementById('is-strict') as HTMLInputElement).checked).toBe(true)

    ;(document.getElementById('btn-is') as HTMLButtonElement).click()
    await Promise.resolve()

    expect(mocks.startJob).toHaveBeenCalledWith(
      'impedance',
      device,
      expect.objectContaining({
        method: 'dynamic_defect_frequency_certified',
        N_grid: 12,
        n_freq: 33,
        f_min: 1e-4,
        f_max: 1e12,
        defect_energy_quadrature_order: 32,
        require_operating_point_certificate: true,
        require_frequency_window_certificate: true,
      }),
    )
  })

  it('workstation pane forwards the same dynamic-defect controls', async () => {
    mountImpedancePane(container, {
      getActiveDevice: () => ({ id: 'device-1', config: device }),
      onRunComplete: vi.fn(),
    })
    const method = document.getElementById('imp-method') as HTMLSelectElement
    method.value = 'dynamic_defect_frequency_certified'
    method.dispatchEvent(new Event('change'))
    ;(document.getElementById('imp-defect-order') as HTMLInputElement).value = '48'

    ;(document.getElementById('btn-imp') as HTMLButtonElement).click()
    await Promise.resolve()

    expect(mocks.startJob).toHaveBeenCalledWith(
      'impedance',
      device,
      expect.objectContaining({
        method: 'dynamic_defect_frequency_certified',
        defect_energy_quadrature_order: 48,
        require_operating_point_certificate: true,
        require_frequency_window_certificate: true,
      }),
    )
  })

  it('legacy panel marks a result without certificate blocks as unclassified', async () => {
    await mountImpedancePanel(container)
    ;(document.getElementById('btn-is') as HTMLButtonElement).click()
    await vi.waitFor(() => {
      expect(mocks.streamJobEvents).toHaveBeenCalledTimes(1)
    })

    const handlers = mocks.streamJobEvents.mock.calls[0][1] as JobStreamHandlers<ISResult>
    handlers.onResult({
      frequencies: [1e3],
      Z_real: [1],
      Z_imag: [-1],
    })

    const warning = container.querySelector<HTMLElement>(
      '[data-test="impedance-evidence-warning"]',
    )
    expect(warning?.textContent).toContain('Legacy impedance result')
    expect(warning?.textContent).toContain('unclassified')
  })

  it('marks a pre-envelope frequency assessment as legacy unclassified', async () => {
    await mountImpedancePanel(container)
    ;(document.getElementById('btn-is') as HTMLButtonElement).click()
    await vi.waitFor(() => {
      expect(mocks.streamJobEvents).toHaveBeenCalledTimes(1)
    })

    const handlers = mocks.streamJobEvents.mock.calls[0][1] as JobStreamHandlers<ISResult>
    handlers.onResult({
      frequencies: [1e3],
      Z_real: [1],
      Z_imag: [-1],
      protocol: {
        method: 'transient_ion_aware',
        V_dc: 0,
        delta_V: 0.01,
        illuminated: false,
        dc_settle_time: 1e-3,
        n_cycles: 5,
        n_extract: 2,
        points_per_cycle: 40,
      },
      operating_point: {
        certified: true,
        reasons: [],
      } as unknown as NonNullable<ISResult['operating_point']>,
      frequency_window: {
        f_min_Hz: 1e3,
        f_max_Hz: 1e3,
        has_mobile_ions: true,
        characteristic_frequency_bracketed: true,
        ionic_branch_covered: true,
        ionic_timescales: [],
        warnings: [],
      },
      grid_assessment: {
        certified: true,
        warnings: [],
      } as unknown as NonNullable<ISResult['grid_assessment']>,
    })

    expect(
      container.querySelector<HTMLElement>(
        '[data-test="impedance-evidence-warning"]',
      )?.textContent,
    ).toContain('Legacy impedance result')
  })
})

describe('ion-aware impedance evidence', () => {
  function certifiedResult(): ISResult {
    return {
      frequencies: [1e-3, 1],
      Z_real: [1, 2],
      Z_imag: [-1, -2],
      protocol: {
        method: 'ion_aware_frequency_certified',
        V_dc: 0.9,
        delta_V: 0.01,
        illuminated: true,
        dc_settle_time: null,
        n_cycles: null,
        n_extract: null,
        points_per_cycle: null,
      },
      operating_point: {
        certified: true,
        reasons: [],
      } as unknown as NonNullable<ISResult['operating_point']>,
      frequency_window: {
        f_min_Hz: 1e-3,
        f_max_Hz: 1,
        has_mobile_ions: true,
        characteristic_frequency_bracketed: true,
        ionic_branch_covered: true,
        ionic_timescales: [],
        warnings: [],
        full_timescale_envelope_bracketed: true,
        recommended_f_min_Hz: 1e-3,
        recommended_f_max_Hz: 1,
        branch_margin_decades: 1,
        max_allowed_sampling_gap_decades: 0.5,
        max_observed_sampling_gap_decades: 0.25,
        ionic_branch_assessments: [],
      },
      grid_assessment: {
        certified: true,
        override_used: false,
        guarded_cell_count: 4,
        offender_count: 0,
        max_guarded_cell_debye_ratio: 0.2,
        max_cell_debye_ratio_limit: 0.5,
        warnings: [],
      },
      ion_aware_evidence: {
        numerically_certified: true,
        thermodynamically_certified: true,
        frequency_window_certified: true,
        certified: true,
        max_relative_face_spread: 2e-8,
        max_ion_inventory_response_relative: 3e-13,
        perturbation_assessments: [{ passed: true }],
        frequency_point_certificates: [
          { frequency_Hz: 1e-3, numerically_certified: true },
          { frequency_Hz: 1, numerically_certified: true },
        ],
        reasons: [],
      } as unknown as NonNullable<ISResult['ion_aware_evidence']>,
    }
  }

  it('summarizes per-frequency and finite-difference evidence', () => {
    const result = certifiedResult()
    expect(collectImpedanceEvidenceWarnings(result)).toEqual([])
    expect(summarizeImpedanceEvidence(result).join(' ')).toContain(
      '2/2 frequency points',
    )
    expect(summarizeImpedanceEvidence(result).join(' ')).toContain(
      'FD refinement passed',
    )
  })

  it('surfaces failed ion-aware point evidence', () => {
    const result = certifiedResult()
    result.ion_aware_evidence = {
      ...result.ion_aware_evidence!,
      numerically_certified: false,
      reasons: ['componentwise_backward_error_exceeds_limit'],
      frequency_point_certificates: [
        {
          ...result.ion_aware_evidence!.frequency_point_certificates[0],
          numerically_certified: false,
        },
        result.ion_aware_evidence!.frequency_point_certificates[1],
      ],
    }

    const warnings = collectImpedanceEvidenceWarnings(result).join(' ')
    expect(warnings).toContain('componentwise_backward_error_exceeds_limit')
    expect(warnings).toContain('1 of 2 frequency points')
  })
})

describe('dynamic-defect impedance evidence', () => {
  it('classifies a complete dynamic certificate without a legacy warning', () => {
    const result = {
      frequencies: [1e-4, 1, 1e12],
      Z_real: [1, 2, 3],
      Z_imag: [-1, -2, -3],
      protocol: {
        method: 'dynamic_defect_frequency_certified',
        V_dc: 0,
        delta_V: 0.01,
        illuminated: false,
        dc_settle_time: null,
        n_cycles: null,
        n_extract: null,
        points_per_cycle: null,
        dynamic_defect_protocol: {
          defect_energy_quadrature_order: 32,
        },
      },
      grid_assessment: {
        certified: true,
        override_used: false,
        guarded_cell_count: 2,
        offender_count: 0,
        max_guarded_cell_debye_ratio: 0.2,
        max_cell_debye_ratio_limit: 0.5,
        warnings: [],
      },
      dynamic_defect_evidence: {
        certified: true,
        numerically_certified: true,
        thermodynamically_certified: true,
        dc_operating_point_certified: true,
        frequency_window_certified: true,
        capability: 'bulk_dynamic_defect',
        interface_current_observation: 'ordinary_finite_volume_faces',
        maximum_all_face_admittance_spread: 1e-10,
        maximum_refinement_relative_change: 2e-5,
        maximum_bulk_trap_balance_relative_error: 3e-8,
        maximum_interface_trap_balance_relative_error: null,
        frequency_window: {
          certified: true,
          trap_low_frequency_limit_covered: true,
          trap_high_frequency_limit_covered: true,
          every_trap_relaxation_frequency_bracketed: true,
          requested_minimum_frequency_Hz: 1e-4,
          requested_maximum_frequency_Hz: 1e12,
        },
        reasons: [],
      },
    } as unknown as ISResult

    expect(collectImpedanceEvidenceWarnings(result)).toEqual([])
    const summary = summarizeImpedanceEvidence(result).join(' ')
    expect(summary).toContain('energy order 32')
    expect(summary).toContain('bulk_dynamic_defect')
    expect(summary).toContain('relaxation frequencies bracketed')
  })
})
