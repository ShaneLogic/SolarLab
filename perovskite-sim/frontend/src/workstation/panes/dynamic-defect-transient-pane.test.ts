import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  startJob: vi.fn(),
  streamJobEvents: vi.fn(),
  newPlot: vi.fn(),
  purge: vi.fn(),
}))

vi.mock('../../job-stream', () => ({
  startJob: mocks.startJob,
  streamJobEvents: mocks.streamJobEvents,
}))

vi.mock('plotly.js-basic-dist-min', () => ({
  default: {
    newPlot: mocks.newPlot,
    purge: mocks.purge,
  },
}))

import { dynamicDefectTransientEligibility } from '../../explicit-defect-capability'
import type {
  DeviceConfig,
  DynamicDefectTransientResult,
  JobStreamHandlers,
} from '../../types'
import { mountDynamicDefectTransientPane } from './dynamic-defect-transient-pane'
import { mountExperimentPane } from './experiment-pane'
import { renderDynamicDefectTransient } from './main-plot-pane'


function layer(
  name: string,
  role: 'substrate' | 'absorber' | 'ETL',
  diffusivity: number,
): DeviceConfig['layers'][number] {
  return {
    name,
    role,
    thickness: 1e-7,
    eps_r: 10,
    mu_n: 1e-3,
    mu_p: 1e-3,
    ni: 2.5e12,
    N_D: 0,
    N_A: 0,
    D_ion: diffusivity,
    P_lim: 2e22,
    P0: 1e22,
    tau_n: 1e-6,
    tau_p: 1e-6,
    n1: 2.5e12,
    p1: 2.5e12,
    B_rad: 0,
    C_n: 0,
    C_p: 0,
    alpha: 0,
  }
}

function eligibleConfig(): DeviceConfig {
  return {
    device: {
      Phi: 0,
      interface_charge_closure: 'equilibrium_referenced',
      interface_charge_rebaseline_acknowledged: true,
      interface_defects: [{
        sigma_n_cm2: 6e-18,
        sigma_p_cm2: 1e-17,
        N_t_cm2: 5e10,
        v_th_cm_s: 1e7,
        E_t_eV_below_cb: 0.55,
      }],
    },
    layers: [layer('absorber', 'absorber', 1e-14), layer('etl', 'ETL', 0)],
  }
}

function result(): DynamicDefectTransientResult {
  const policy = {
    refinement_substeps: [1, 2, 4],
    maximum_newton_iterations: 100,
    maximum_line_search_steps: 40,
    maximum_near_acceptance_nonmonotone_steps: 2,
    maximum_scaled_nonlinear_residual: 0.05,
    maximum_refinement_state_change: 0.02,
    maximum_refinement_current_relative_change: 0.05,
    maximum_charge_balance_relative_error: 1e-10,
    maximum_all_face_current_spread_relative: 2e-6,
    maximum_two_sided_interface_total_current_relative_error: 2e-6,
    maximum_ion_inventory_relative_drift: 1e-9,
    site_occupancy_ceiling: 0.999,
  }
  const protocol = {
    schema_version: 'dynamic-defect-transient-protocol-v1' as const,
    method: 'dynamic_defect_transient_certified' as const,
    capability: 'interface_defect_plus_positive_ions' as const,
    illuminated: false as const,
    times_s: [0, 1e-8, 1e-6, 1e-4],
    voltage_V: [0, 0.05, 0.05, 0.05],
    voltage_interpolation: 'right_continuous_step_and_hold' as const,
    requested_grid_intervals: 4,
    actual_grid_nodes: 4,
    grid_sha256: '1'.repeat(64),
    stack_sha256: '2'.repeat(64),
    interface_defect_document_sha256: ['3'.repeat(64)],
    active_positive_ion_layer_indices: [0],
    defect_energy_quadrature_order: 32,
    interface_current_observation: 'symmetric_adjacent_physical_faces' as const,
    time_step_refinement_factor: 1,
    solver_policy: policy,
    reference_lane_id: 'dynamic-defect-ion-transient-timescale-reference-resolved-v5',
    reference_certificate_sha256: (
      '9eab2f9e251b8d4c0f7f3f07e0baeea9bb6497126ef8d8111eba1803947e5beb'
    ),
  }
  const certificate = {
    dc_operating_point_certified: true,
    dark_reference_certified: true,
    microscopic_binding_certified: true,
    maximum_scaled_nonlinear_residual: 1e-3,
    maximum_charge_balance_relative_error: 1e-12,
    maximum_all_face_current_spread_relative: 1e-7,
    maximum_two_sided_interface_total_current_relative_error: 1e-7,
    maximum_ion_inventory_relative_drift: 1e-14,
    maximum_refinement_state_change: 1e-4,
    maximum_refinement_current_relative_change: 2e-4,
    certified: true,
    reasons: [],
  }
  return {
    grid_m: [0, 0.5e-7, 1e-7, 2e-7],
    times_s: protocol.times_s,
    voltage_V: protocol.voltage_V,
    terminal_total_current_A_m2: [0, 1, 0.8, 0.5],
    total_current_faces_A_m2: [[0, 0, 0], [1, 1, 1], [0.8, 0.8, 0.8], [0.5, 0.5, 0.5]],
    interface_total_current_A_m2: [[[0, 0]], [[1, 1]], [[0.8, 0.8]], [[0.5, 0.5]]],
    interface_occupancy: [[0.5], [0.51], [0.515], [0.52]],
    interface_occupancy_change: [[0], [0.01], [0.015], [0.02]],
    positive_ion_centroid_m: [5e-8, 5.1e-8, 5.2e-8, 5.3e-8],
    positive_ion_centroid_shift_m: [0, 1e-9, 2e-9, 3e-9],
    integrated_charge_change_C_m2: [0, 1e-10, 2e-10, 3e-10],
    electron_density_m3: [[1, 1, 1, 1]],
    hole_density_m3: [[1, 1, 1, 1]],
    positive_ion_density_m3: [[1, 1, 1, 1]],
    electrostatic_potential_V: [[0, 0, 0, 0]],
    protocol,
    evidence: {
      model: 'dynamic-defect-transient-evidence-v1',
      protocol,
      protocol_sha256: '4'.repeat(64),
      capability: 'interface_defect_plus_positive_ions',
      engine_scope: 'research_two_sided_interface_defect_mobile_ion_transient_only',
      engine_version: 'interface-defect-ion-transient-v2',
      state_sha256: '5'.repeat(64),
      reference_lane_id: protocol.reference_lane_id,
      reference_certificate_sha256: protocol.reference_certificate_sha256,
      engine_certificate: certificate,
      dc_operating_point_certified: true,
      dark_reference_certified: true,
      microscopic_binding_certified: true,
      numerically_certified: true,
      public_projection_certified: true,
      certified: true,
      maximum_interface_occupancy_motion: 0.02,
      maximum_positive_ion_relative_motion: 1e-3,
      maximum_positive_ion_centroid_shift_m: 3e-9,
      maximum_integrated_charge_change_C_m2: 3e-10,
      maximum_terminal_current_A_m2: 1,
      reasons: [],
      limitations: [],
    },
  }
}

let container: HTMLDivElement

beforeEach(() => {
  document.body.replaceChildren()
  container = document.createElement('div')
  document.body.appendChild(container)
  mocks.startJob.mockReset()
  mocks.startJob.mockResolvedValue('dit-job')
  mocks.streamJobEvents.mockReset()
  mocks.newPlot.mockReset()
  mocks.purge.mockReset()
})

afterEach(() => {
  document.body.replaceChildren()
})

describe('dynamic-defect transient capability and pane', () => {
  it('is selectable as a dedicated transient experiment', () => {
    mountExperimentPane(container, {
      getActiveDevice: () => ({ id: 'device-1', config: eligibleConfig() }),
      onRunComplete: vi.fn(),
    })

    const option = container.querySelector<HTMLElement>(
      '[data-kind="dynamic_defect_transient"]',
    )
    expect(option?.textContent).toContain('Defect–Ion Transient')
  })

  it('matches the backend narrow topology before submit', () => {
    const config = eligibleConfig()
    expect(dynamicDefectTransientEligibility(config)).toMatchObject({
      eligible: true,
      reasons: [],
      N_grid: 4,
    })

    const negative = eligibleConfig()
    ;(negative.layers[0] as typeof negative.layers[0] & {
      D_ion_neg: number
      P0_neg: number
    }).D_ion_neg = 1e-14
    ;(negative.layers[0] as typeof negative.layers[0] & {
      D_ion_neg: number
      P0_neg: number
    }).P0_neg = 1e22
    expect(dynamicDefectTransientEligibility(negative).reasons.join(' ')).toContain(
      'negative ions',
    )

    const substrate = eligibleConfig()
    substrate.layers.unshift(layer('glass', 'substrate', 0))
    const electricalDefect = substrate.device.interface_defects![0]
    substrate.device.interface_defects = [electricalDefect, electricalDefect]
    expect(dynamicDefectTransientEligibility(substrate)).toMatchObject({
      eligible: true,
      reasons: [],
    })

    const incomplete = eligibleConfig()
    incomplete.device.interface_defects![0]!.sigma_n_cm2 = null
    expect(dynamicDefectTransientEligibility(incomplete).reasons.join(' ')).toContain(
      'complete positive microscopic interface document',
    )

    const calibrated = eligibleConfig()
    calibrated.device.interface_defects![0]!.calibration_factor = 1e-4
    expect(dynamicDefectTransientEligibility(calibrated).reasons.join(' ')).toContain(
      'empirical interface calibration',
    )
  })

  it('submits the exact dark step history and commits certified results', async () => {
    const config = eligibleConfig()
    const onRunComplete = vi.fn()
    mountDynamicDefectTransientPane(container, {
      getActiveDevice: () => ({ id: 'device-1', config }),
      onRunComplete,
    })

    expect(container.querySelector('[data-test="dit-capability"]')?.textContent).toContain(
      'Eligible',
    )
    ;(document.getElementById('btn-dit') as HTMLButtonElement).click()
    await vi.waitFor(() => expect(mocks.startJob).toHaveBeenCalledTimes(1))
    expect(mocks.startJob).toHaveBeenCalledWith(
      'dynamic_defect_transient',
      config,
      {
        N_grid: 4,
        times_s: [0, 1e-8, 1e-6, 1e-4],
        voltage_V: [0, 0.05, 0.05, 0.05],
        illuminated: false,
        method: 'dynamic_defect_transient_certified',
      },
    )
    await vi.waitFor(() => expect(mocks.streamJobEvents).toHaveBeenCalledTimes(1))
    const handlers = mocks.streamJobEvents.mock.calls[0][1] as JobStreamHandlers<
      DynamicDefectTransientResult & { active_physics?: string }
    >
    handlers.onResult({ ...result(), active_physics: 'FULL · dynamic defects' })
    expect(onRunComplete).toHaveBeenCalledTimes(1)
    expect(onRunComplete.mock.calls[0][1].result.kind).toBe(
      'dynamic_defect_transient',
    )
  })

  it('rejects non-increasing sample times without starting a job', () => {
    const config = eligibleConfig()
    mountDynamicDefectTransientPane(container, {
      getActiveDevice: () => ({ id: 'device-1', config }),
      onRunComplete: vi.fn(),
    })
    ;(document.getElementById('dit-t2') as HTMLInputElement).value = '1e-9'
    ;(document.getElementById('btn-dit') as HTMLButtonElement).click()

    expect(mocks.startJob).not.toHaveBeenCalled()
    expect(document.getElementById('status-dit')?.textContent).toContain(
      '0 < early < intermediate < end',
    )
  })
})

describe('dynamic-defect transient plot evidence', () => {
  it('renders certified evidence and current plus occupancy traces', () => {
    const data = result()

    renderDynamicDefectTransient(container, data)

    expect(
      container.querySelector(
        '[data-test="dynamic-defect-transient-evidence-warning"]',
      ),
    ).toBeNull()
    expect(
      container.querySelector(
        '[data-test="dynamic-defect-transient-evidence-summary"]',
      )?.textContent,
    ).toContain('Certified interface defect + positive-ion transient')
    expect(
      container.querySelector(
        '[data-test="dynamic-defect-transient-evidence-summary"]',
      )?.textContent,
    ).toContain('Δt factor 1')
    expect(mocks.newPlot).toHaveBeenCalledTimes(1)
    const traces = mocks.newPlot.mock.calls[0][1] as Array<Record<string, unknown>>
    expect(traces).toHaveLength(2)
    expect(traces[0].name).toBe('Terminal current')
    expect(traces[1].name).toBe('Interface Δf')
    expect(traces[1].yaxis).toBe('y2')
  })

  it('surfaces declared certification reasons', () => {
    const data = result()
    data.evidence.certified = false
    data.evidence.numerically_certified = false
    data.evidence.reasons = ['time_refinement_state_not_converged']

    renderDynamicDefectTransient(container, data)

    expect(
      container.querySelector(
        '[data-test="dynamic-defect-transient-evidence-warning"]',
      )?.textContent,
    ).toContain('time_refinement_state_not_converged')
  })
})
