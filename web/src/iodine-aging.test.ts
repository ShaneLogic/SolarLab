import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Plotly from 'plotly.js-basic-dist-min'
import { IodineAgingClient } from './iodine-aging-client'
import { mountIodineAgingPanel } from './panels/iodine-aging'
import type { IodineAgingPanel } from './panels/iodine-aging'

vi.mock('plotly.js-basic-dist-min', () => ({
  default: {
    newPlot: vi.fn().mockResolvedValue(undefined),
    purge: vi.fn(),
  },
}))

const SAMPLE_CATALOG = {
  schema: 'solarlab.iodine-presets.v1',
  scientific_acceptance: 'not_evaluated',
  scientific_qualification: false,
  presets: [
    {
      id: 'bitton_tessler_2023_blocking',
      name: 'Bitton & Tessler (2023) Blocking Absorber Stack',
      description: 'Blocking 5-reaction iodine deck',
      reference: 'Bitton & Tessler (2023)',
      source_id: 'solarlab/materials/data/iodine_presets/bitton_tessler_2023_blocking.yaml',
      sha256: '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
      deck: { schema_version: 'solarlab.iodine-deck.v1', id: 'bitton_tessler_2023_blocking' },
    },
  ],
}

const SAMPLE_PROFILES = {
  electrical_nodes_nm: [0, 50, 300, 550, 600],
  chem_nodes_nm: [50, 300, 550],
  potential_V: [0.0, 0.2, 0.45, 0.7, 0.9],
  electron_density_m3: [1e18, 1e20, 1e21, 1e19, 1e16],
  hole_density_m3: [1e16, 1e19, 1e21, 1e20, 1e18],
  iodide_m3: [1.2e24, 1.0e24, 0.8e24],
  iodine_atom_m3: [1e19, 2e19, 3e19],
  iodine_molecule_m3: [1e17, 5e17, 9e17],
  species_inventories_m2: { iodide: 5.0e17, iodine_atom: 1.0e13, iodine_molecule: 2.5e11 },
  total_iodine_atoms_m2: 5.000105e17,
}

const SAMPLE_STEADY = {
  schema: 'solarlab.iodine-steady-result.v1',
  deck_id: 'bitton_tessler_2023_blocking',
  deck_name: 'Bitton & Tessler (2023) Blocking Absorber Stack',
  device_sha256: 'deadbeef',
  domain_mode: 'absorber_blocking',
  operating_state_mode: 'closed_inventory_steady',
  voltage_V: 0.9,
  generation_scale: 1.0,
  profiles: SAMPLE_PROFILES,
  frozen_jv: {
    voltages_V: [0.0, 0.6, 1.12],
    output_current_A_m2: [215.0, 195.0, 0.0],
    metrics: { open_circuit_voltage_V: 1.12, short_circuit_current_A_m2: 215.0 },
  },
  wall_s: 0.42,
  scientific_acceptance: 'not_evaluated',
  scientific_qualification: false,
}

const SAMPLE_IMPEDANCE = {
  schema: 'solarlab.iodine-impedance-result.v1',
  deck_id: 'bitton_tessler_2023_blocking',
  deck_name: 'Bitton & Tessler (2023) Blocking Absorber Stack',
  device_sha256: 'deadbeef',
  domain_mode: 'absorber_blocking',
  operating_state_mode: 'closed_inventory_steady',
  voltage_V: 0.9,
  generation_scale: 1.0,
  frequencies_Hz: [0.01, 10.0, 1.0e5],
  sweeps: {
    mobile: {
      chem_mode: 'mobile',
      frequencies_Hz: [0.01, 10.0, 1.0e5],
      admittance_real_S_m2: [10, 20, 50],
      admittance_imag_S_m2: [0.05, 1.2, 60],
      impedance_real_Ohm_m2: [0.1, 0.04, 0.008],
      impedance_imag_Ohm_m2: [-0.0005, -0.002, -0.01],
      apparent_capacitance_F_m2: [0.79, 0.019, 9.5e-5],
      max_maxwell_spread_A_m2_V: 1e-11,
      max_backward_error: 1e-14,
    },
  },
  redox_excess_capacitance_F_m2: null,
  profiles: SAMPLE_PROFILES,
  wall_s: 0.35,
  scientific_acceptance: 'not_evaluated',
  scientific_qualification: false,
}

const SAMPLE_AGING = {
  schema: 'solarlab.iodine-aging-result.v1',
  deck_id: 'bitton_tessler_2023_blocking',
  deck_name: 'Bitton & Tessler (2023) Blocking Absorber Stack',
  protocol_sha256: 'aaaa',
  device_sha256: 'bbbb',
  domain_mode: 'absorber_blocking',
  reference_iodine_inventory_m2: 5.0e17,
  max_relative_conservation_error: 1.2e-13,
  accepted_bdf_steps: 28,
  completed: true,
  wall_s: 0.78,
  trajectory: {
    times_s: [0, 60, 120],
    in_device_iodine_m2: [5.0e17, 5.0e17, 5.0e17],
    cumulative_escaped_iodine_atoms_m2: [0, 0, 0],
    cumulative_immobilized_iodine_atoms_m2: [0, 0, 0],
  },
  checkpoints: [
    {
      time_s: 0,
      stage_name: 'initial',
      in_device_iodine_m2: 5.0e17,
      cumulative_escaped_iodine_atoms_m2: 0,
      cumulative_immobilized_iodine_atoms_m2: 0,
      conservation_residual_m2: 0,
      profiles: SAMPLE_PROFILES,
      frozen_jv: SAMPLE_STEADY.frozen_jv,
      dynamic_jv: null,
    },
  ],
  persisted_run_store: null,
  scientific_acceptance: 'not_evaluated',
  scientific_qualification: false,
}

describe('IodineAgingClient & mountIodineAgingPanel', () => {
  let root: HTMLElement
  let panel: IodineAgingPanel | undefined

  beforeEach(() => {
    vi.clearAllMocks()
    root = document.createElement('div')
    document.body.append(root)
  })

  afterEach(() => {
    panel?.dispose()
    panel = undefined
    root.remove()
  })

  it('rejects responses that violate scientific_qualification === false', async () => {
    const badFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ...SAMPLE_CATALOG, scientific_qualification: true }), { status: 200 }),
    )
    const client = new IodineAgingClient('http://localhost:8000', badFetch as unknown as typeof fetch)
    await expect(client.listPresets()).rejects.toThrow(/qualification/)
  })

  it('mounts presets and executes steady, impedance, and aging modes', async () => {
    const mockFetch = vi.fn(async (input: RequestInfo | URL) => {
      const u = String(input)
      if (u.endsWith('/iodine/presets')) return new Response(JSON.stringify(SAMPLE_CATALOG), { status: 200 })
      if (u.endsWith('/iodine/steady')) return new Response(JSON.stringify(SAMPLE_STEADY), { status: 200 })
      if (u.endsWith('/iodine/impedance')) return new Response(JSON.stringify(SAMPLE_IMPEDANCE), { status: 200 })
      if (u.endsWith('/iodine/aging')) return new Response(JSON.stringify(SAMPLE_AGING), { status: 200 })
      return new Response(JSON.stringify({ detail: 'not found' }), { status: 404 })
    })

    panel = mountIodineAgingPanel(root, 'http://localhost:8000/', mockFetch as unknown as typeof fetch)
    await panel.ready
    expect(root.querySelector('[data-role=state]')?.textContent).toBe('ready')
    expect(root.querySelector('[data-role=preset-info]')?.textContent).toContain('Bitton & Tessler')

    const modeSelect = root.querySelector<HTMLSelectElement>('select[name=mode]')!
    const kaInput = root.querySelector<HTMLInputElement>('input[name=k_a_cm3_s]')!
    const submitBtn = root.querySelector<HTMLButtonElement>('button[type=submit]')!

    kaInput.value = '2e-11'
    modeSelect.value = 'steady'
    submitBtn.click()
    await vi.waitFor(() => expect(root.querySelector('[data-role=state]')?.textContent).toBe('succeeded'))
    expect(Plotly.newPlot).toHaveBeenCalledTimes(2)
    expect(root.querySelector('[data-role=summary]')?.textContent).toContain('scientific_qualification=false')

    modeSelect.value = 'impedance'
    submitBtn.click()
    await vi.waitFor(() => expect(Plotly.newPlot).toHaveBeenCalledTimes(3))

    modeSelect.value = 'aging'
    submitBtn.click()
    await vi.waitFor(() => expect(Plotly.newPlot).toHaveBeenCalledTimes(5))
    expect(panel.lastResult?.schema).toBe('solarlab.iodine-aging-result.v1')
  })
})
