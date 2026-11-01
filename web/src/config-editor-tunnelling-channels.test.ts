/**
 * vitest — FE-1: the D8 tunnelling-channel document must survive the editor.
 *
 * `readDeviceEditor` rebuilds `device` from an allowlist, so any device-level
 * key it does not name is dropped. `tunnelling_channels` was absent, and
 * `wkb_tunnelling_intraband_spike.yaml` IS offered in the preset dropdown
 * (measured: 50 shipped configs, that one among them) — so loading it and
 * pressing Run silently produced a tunnelling-free J-V.
 *
 * Passing it through does NOT give the user tunnelling in their J-V. The
 * backend refuses: the channels are certified only on the guarded QF/DC lane,
 * and `assemble_rhs` raises `ExplicitDefectCapabilityError` on the transient
 * driver this pane runs. That refusal is the point — an explicit "this
 * preset's physics is not available here" replaces a silent wrong answer.
 *
 * The editor renders no control for the document and deliberately does not
 * validate it: the canonical schema is `solarlab-wkb-tunnelling-channels-v1`
 * in Python, and a partial mirror in TypeScript would drift from it.
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

const DOCUMENT = {
  schema_version: 'solarlab-wkb-tunnelling-channels-v1',
  intraband: {
    enabled: true,
    electron_effective_mass_rel: 0.2,
    hole_effective_mass_rel: 0.2,
    carrier: 'electron',
    energy_quadrature_order: 96,
  },
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
  container = document.createElement('div')
  document.body.appendChild(container)
})

afterEach(() => {
  container.remove()
})

describe('tunnelling-channel passthrough', () => {
  it('survives a render/read round-trip unchanged', () => {
    const original = cfg({ tunnelling_channels: DOCUMENT })
    renderDeviceEditor(container, original, 'full')

    const read = readDeviceEditor(original)

    expect(read.device.tunnelling_channels).toEqual(DOCUMENT)
  })

  it('is carried verbatim rather than reconstructed', () => {
    // A partial TypeScript mirror of the Python schema would drift from it,
    // so the editor must not rebuild the document field by field. Deep
    // equality against an object the editor never inspected proves it did
    // not: an unknown extra key survives too.
    const exotic = { ...DOCUMENT, some_future_channel: { enabled: false } }
    const original = cfg({ tunnelling_channels: exotic })
    renderDeviceEditor(container, original, 'full')

    const read = readDeviceEditor(original)

    expect(read.device.tunnelling_channels).toEqual(exotic)
  })

  it('is absent, not null or empty, when the preset does not declare it', () => {
    // Every shipped preset except one omits the key. Emitting `null` or `{}`
    // would move the semantic hash of all of them — the same class of drift
    // the D8-E1 hash-normalisation rule exists to prevent.
    const original = cfg()
    renderDeviceEditor(container, original, 'full')

    const read = readDeviceEditor(original)

    expect('tunnelling_channels' in read.device).toBe(false)
  })

  it('survives the single-layer drill-down, where no device panel renders', () => {
    // The drill-down path returns the device verbatim, so this was already
    // state-dependent before the fix: the field survived only when no layer
    // happened to be selected. Pin both paths.
    const original = cfg({ tunnelling_channels: DOCUMENT })
    renderDeviceEditor(container, original, 'full', 1)

    const read = readDeviceEditor(original, 1)

    expect(read.device.tunnelling_channels).toEqual(DOCUMENT)
  })

  it('is not gated by tier, because the backend gate is the real one', () => {
    // Hiding it in fast/legacy would recreate the silent-strip bug for those
    // tiers. The capability guard lives in the solver and fires regardless of
    // tier, so the editor's job is only to not lose the field.
    for (const tier of ['legacy', 'fast', 'full'] as const) {
      const original = cfg({ tunnelling_channels: DOCUMENT, mode: tier })
      const scratch = document.createElement('div')
      document.body.appendChild(scratch)
      renderDeviceEditor(scratch, original, tier)

      const read = readDeviceEditor(original)

      expect(read.device.tunnelling_channels).toEqual(DOCUMENT)
      scratch.remove()
    }
  })
})
