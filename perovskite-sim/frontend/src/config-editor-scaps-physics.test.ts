/**
 * vitest — the FULL-tier SCAPS-validation physics panel.
 *
 * Surfaces the five device-level flags the YAML loader + stack_from_dict
 * parse (dos_band_potentials, flat_band_contacts, interface_plane_closure,
 * interface_plane_projection, het_recomb_despike) so a parity preset loaded
 * in the live editor round-trips them instead of having them silently
 * stripped at the inline-device boundary. Pins render → read round-trip,
 * tier gating, and the clean-payload contract (no spurious false / 0).
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { renderDeviceEditor, readDeviceEditor } from './config-editor'
import type { DeviceConfig, LayerConfig } from './types'

function emptyLayer(name: string, role: LayerConfig['role']): LayerConfig {
  return {
    name,
    role,
    thickness: 1e-7,
    eps_r: 1, mu_n: 0, mu_p: 0, ni: 1e10, N_D: 0, N_A: 0,
    D_ion: 0, P_lim: 0, P0: 0,
    tau_n: 1e-6, tau_p: 1e-6, n1: 1e10, p1: 1e10,
    B_rad: 0, C_n: 0, C_p: 0, alpha: 0,
  }
}

function cfg(extras: Partial<DeviceConfig['device']> = {}): DeviceConfig {
  return {
    device: { V_bi: 1.3, Phi: 2.5e21, mode: 'full', ...extras },
    layers: [
      emptyLayer('HTL', 'HTL'),
      emptyLayer('PVK', 'absorber'),
      emptyLayer('ETL', 'ETL'),
    ],
  }
}

let container: HTMLElement

beforeEach(() => {
  document.body.replaceChildren()
  container = document.createElement('div')
  document.body.appendChild(container)
})

afterEach(() => {
  document.body.replaceChildren()
})

describe('SCAPS-validation physics panel tier gating', () => {
  it('renders in FULL tier', () => {
    renderDeviceEditor(container, cfg(), 'full')
    expect(container.innerHTML).toContain('SCAPS-validation physics')
  })

  it('hidden in FAST tier', () => {
    renderDeviceEditor(container, cfg({ mode: 'fast' }), 'fast')
    expect(container.innerHTML).not.toContain('SCAPS-validation physics')
  })

  it('hidden in LEGACY tier', () => {
    renderDeviceEditor(container, cfg({ mode: 'legacy' }), 'legacy')
    expect(container.innerHTML).not.toContain('SCAPS-validation physics')
  })

  it('hidden in single-layer drill-down even on FULL', () => {
    renderDeviceEditor(container, cfg(), 'full', 1)
    expect(container.innerHTML).not.toContain('SCAPS-validation physics')
  })
})

describe('SCAPS-validation physics panel structure', () => {
  it('renders the four checkboxes + despike field by ID', () => {
    renderDeviceEditor(container, cfg(), 'full')
    for (const id of ['dev-dos', 'dev-flatband', 'dev-iface-closure', 'dev-iface-proj', 'dev-despike']) {
      expect(document.getElementById(id), `missing ${id}`).not.toBeNull()
    }
  })

  it('reflects the input config (checked + despike value)', () => {
    renderDeviceEditor(container, cfg({ dos_band_potentials: true, het_recomb_despike: 0.53 }), 'full')
    expect((document.getElementById('dev-dos') as HTMLInputElement).checked).toBe(true)
    expect((document.getElementById('dev-flatband') as HTMLInputElement).checked).toBe(false)
    expect((document.getElementById('dev-despike') as HTMLInputElement).value).toBe('0.53')
  })

  it('renders an absent DOS flag as enabled to match the backend default', () => {
    renderDeviceEditor(container, cfg(), 'full')
    expect((document.getElementById('dev-dos') as HTMLInputElement).checked).toBe(true)
  })

  it('renders an explicit DOS false opt-out as disabled', () => {
    renderDeviceEditor(container, cfg({ dos_band_potentials: false }), 'full')
    expect((document.getElementById('dev-dos') as HTMLInputElement).checked).toBe(false)
  })
})

describe('SCAPS-validation physics round-trip', () => {
  it('round-trips set flags back into device', () => {
    const c = cfg({
      dos_band_potentials: true,
      interface_plane_projection: true,
      het_recomb_despike: 0.53,
    })
    renderDeviceEditor(container, c, 'full')
    const out = readDeviceEditor(c)
    expect(out.device.dos_band_potentials).toBe(true)
    expect(out.device.interface_plane_projection).toBe(true)
    expect(out.device.het_recomb_despike).toBe(0.53)
    // unset flags stay absent (not false)
    expect('flat_band_contacts' in out.device).toBe(false)
    expect('interface_plane_closure' in out.device).toBe(false)
  })

  it('unchecking DOS serializes an explicit false opt-out', () => {
    const c = cfg({ dos_band_potentials: true })
    renderDeviceEditor(container, c, 'full')
    ;(document.getElementById('dev-dos') as HTMLInputElement).checked = false
    const out = readDeviceEditor(c)
    expect(out.device.dos_band_potentials).toBe(false)
  })

  it('disabling default-on DOS from an absent field serializes false', () => {
    const c = cfg()
    renderDeviceEditor(container, c, 'full')
    ;(document.getElementById('dev-dos') as HTMLInputElement).checked = false
    const out = readDeviceEditor(c)
    expect(out.device.dos_band_potentials).toBe(false)
  })

  it('clean payload for a non-SCAPS config (no spurious false / 0)', () => {
    const c = cfg()
    renderDeviceEditor(container, c, 'full')
    const out = readDeviceEditor(c)
    for (const k of ['dos_band_potentials', 'flat_band_contacts', 'interface_plane_closure', 'interface_plane_projection', 'het_recomb_despike']) {
      expect(k in out.device, `spurious ${k}`).toBe(false)
    }
  })

  it('non-FULL round-trip preserves original flags verbatim', () => {
    const c = cfg({ mode: 'fast', dos_band_potentials: true, het_recomb_despike: 0.53 })
    renderDeviceEditor(container, c, 'fast')
    const out = readDeviceEditor(c)
    expect(out.device.dos_band_potentials).toBe(true)
    expect(out.device.het_recomb_despike).toBe(0.53)
  })

  it('preserves hidden physics flags including explicit false opt-outs', () => {
    const c = cfg({
      te_physical_norm: true,
      ion_steric_diffusion_only: false,
      ion_steric_shared_site: false,
      flat_band_metal_contacts: true,
      contact_phi_B_eV: 0.42,
      interface_two_sided: true,
      interface_shared_occupancy: true,
      interface_plane_generation: true,
      jv_solver_policy: 'cancellation_safe_qf_required',
    })
    renderDeviceEditor(container, c, 'full')
    const out = readDeviceEditor(c)
    expect(out.device.te_physical_norm).toBe(true)
    expect(out.device.ion_steric_diffusion_only).toBe(false)
    expect(out.device.ion_steric_shared_site).toBe(false)
    expect(out.device.flat_band_metal_contacts).toBe(true)
    expect(out.device.contact_phi_B_eV).toBe(0.42)
    expect(out.device.interface_two_sided).toBe(true)
    expect(out.device.interface_shared_occupancy).toBe(true)
    expect(out.device.interface_plane_generation).toBe(true)
    expect(out.device.jv_solver_policy).toBe('cancellation_safe_qf_required')
  })

  it('preserves top-level electrical-grid and simulation-hint contracts', () => {
    const c: DeviceConfig = {
      ...cfg({ jv_solver_policy: 'cancellation_safe_qf_required' }),
      simulation_hints: { min_N_grid: 200, notes: 'certified c-Si grid' },
      electrical_grid: {
        interval_weights: { n_emitter: 1, p_base: 4 },
        alphas: { n_emitter: 2, p_base: 3 },
      },
    }
    renderDeviceEditor(container, c, 'full')
    const out = readDeviceEditor(c)
    expect(out.simulation_hints).toEqual(c.simulation_hints)
    expect(out.electrical_grid).toEqual(c.electrical_grid)
  })
})
