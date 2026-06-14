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

  it('unchecking a flag drops it from the payload', () => {
    const c = cfg({ dos_band_potentials: true })
    renderDeviceEditor(container, c, 'full')
    ;(document.getElementById('dev-dos') as HTMLInputElement).checked = false
    const out = readDeviceEditor(c)
    expect('dos_band_potentials' in out.device).toBe(false)
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
})
