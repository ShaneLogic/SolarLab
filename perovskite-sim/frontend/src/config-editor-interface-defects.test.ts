/**
 * Phase E1.12 — vitest tests for the Phase E1.8 Interface Defects panel.
 *
 * The Interface Defects panel surfaces the Phase E1.5
 * ``DeviceStack.interface_defects`` capability in the workstation live
 * editor, FULL-tier-gated, one row per heterointerface. These tests pin
 * the render → read round-trip and the FULL-tier gating so regressions
 * in `config-editor.ts` surface immediately without requiring a full
 * E2E run.
 *
 * Contract:
 * 1. Panel renders ONLY in FULL tier (hidden in LEGACY / FAST / single-
 *    layer drill-down).
 * 2. Panel renders one row per heterointerface (n - 1 rows for n layers).
 * 3. Each row carries 5 numeric inputs by ID: ``idef-{i}-{sigma-n,
 *    sigma-p,N-t,v-th,E-t}``.
 * 4. Populated slot in the input config renders the field values in
 *    the corresponding inputs; absent slot leaves inputs empty.
 * 5. ``readDeviceEditor`` round-trips a populated slot back into
 *    ``DeviceConfig.device.interface_defects[k]``; fully-empty slot
 *    collapses to ``null``; mixed half-populated slot also collapses
 *    to ``null`` (backend contract).
 * 6. Absent ``interface_defects`` on the original config + all-empty
 *    inputs → field stays absent in the round-tripped payload (no
 *    spurious null array on legacy presets).
 * 7. Round-trip across renderDeviceEditor → readDeviceEditor preserves
 *    the populated slot byte-for-byte.
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


function threeLayerConfig(
  extras: Partial<DeviceConfig['device']> = {},
): DeviceConfig {
  return {
    device: {
      V_bi: 1.3,
      Phi: 2.5e21,
      mode: 'full',
      ...extras,
    },
    layers: [
      emptyLayer('HTL', 'HTL'),
      emptyLayer('PVK', 'absorber'),
      emptyLayer('ETL', 'ETL'),
    ],
  }
}


function pvkEtlDefect() {
  return {
    sigma_n_cm2: 1.0e-15,
    sigma_p_cm2: 1.0e-15,
    N_t_cm2: 1.0e8,
    v_th_cm_s: 1.0e7,
    E_t_eV_below_cb: 0.6,
  }
}


let container: HTMLElement


beforeEach(() => {
  // Clean every leftover element from prior tests so duplicate IDs
  // do not poison ``document.getElementById`` lookups. jsdom is a
  // single shared global across the whole vitest run; without this
  // cleanup, stale inputs from earlier tests collide with the freshly-
  // rendered ones and the reader sees the old DOM.
  document.body.replaceChildren()
  container = document.createElement('div')
  document.body.appendChild(container)
})


afterEach(() => {
  document.body.replaceChildren()
})


// ────── Tier gating ──────────────────────────────────────────────────


describe('Interface Defects panel tier gating', () => {
  it('renders in FULL tier', () => {
    renderDeviceEditor(container, threeLayerConfig(), 'full')
    expect(container.innerHTML).toContain('Interface Defects (FULL only)')
  })

  it('hidden in FAST tier', () => {
    renderDeviceEditor(container, threeLayerConfig({ mode: 'fast' }), 'fast')
    expect(container.innerHTML).not.toContain('Interface Defects')
  })

  it('hidden in LEGACY tier', () => {
    renderDeviceEditor(
      container, threeLayerConfig({ mode: 'legacy' }), 'legacy',
    )
    expect(container.innerHTML).not.toContain('Interface Defects')
  })

  it('hidden in single-layer drill-down even on FULL tier', () => {
    renderDeviceEditor(container, threeLayerConfig(), 'full', 1)
    expect(container.innerHTML).not.toContain('Interface Defects')
  })
})


// ────── Row count + input IDs ────────────────────────────────────────


describe('Interface Defects panel structure', () => {
  it('renders one row per heterointerface (n-1 for n layers)', () => {
    renderDeviceEditor(container, threeLayerConfig(), 'full')
    // 3 layers → 2 heterointerfaces → 10 inputs (5 fields × 2 rows)
    expect(container.querySelectorAll('input[id^="idef-"]').length).toBe(10)
  })

  it('uses idef-{i}-{field} ID convention', () => {
    renderDeviceEditor(container, threeLayerConfig(), 'full')
    for (const i of [0, 1]) {
      for (const f of ['sigma-n', 'sigma-p', 'N-t', 'v-th', 'E-t']) {
        const el = document.getElementById(`idef-${i}-${f}`)
        expect(el, `missing input idef-${i}-${f}`).not.toBeNull()
        expect(el?.tagName).toBe('INPUT')
      }
    }
  })

  it('labels each row by adjacent layer names', () => {
    renderDeviceEditor(container, threeLayerConfig(), 'full')
    expect(container.innerHTML).toContain('HTL / PVK')
    expect(container.innerHTML).toContain('PVK / ETL')
  })

  it('empty when stack has only one layer (no interfaces)', () => {
    const cfg: DeviceConfig = {
      device: { V_bi: 1.0, Phi: 1e21, mode: 'full' },
      layers: [emptyLayer('A', 'absorber')],
    }
    renderDeviceEditor(container, cfg, 'full')
    expect(container.innerHTML).not.toContain('Interface Defects')
  })
})


// ────── Render value population ──────────────────────────────────────


describe('Interface Defects panel value population', () => {
  it('populated slot renders field values', () => {
    const cfg = threeLayerConfig({
      interface_defects: [null, pvkEtlDefect()],
    })
    renderDeviceEditor(container, cfg, 'full')

    const sigmaN = document.getElementById('idef-1-sigma-n') as HTMLInputElement
    const N_t = document.getElementById('idef-1-N-t') as HTMLInputElement
    const E_t = document.getElementById('idef-1-E-t') as HTMLInputElement

    expect(Number(sigmaN.value)).toBeCloseTo(1.0e-15, 18)
    expect(Number(N_t.value)).toBeCloseTo(1.0e8, 6)
    expect(Number(E_t.value)).toBeCloseTo(0.6, 4)
  })

  it('absent slot renders empty inputs', () => {
    const cfg = threeLayerConfig({
      interface_defects: [null, pvkEtlDefect()],
    })
    renderDeviceEditor(container, cfg, 'full')

    // HTL/PVK (index 0) is null → all 5 inputs empty
    for (const f of ['sigma-n', 'sigma-p', 'N-t', 'v-th', 'E-t']) {
      const el = document.getElementById(`idef-0-${f}`) as HTMLInputElement
      expect(el.value).toBe('')
    }
  })

  it('absent interface_defects field renders all inputs empty', () => {
    renderDeviceEditor(container, threeLayerConfig(), 'full')
    for (const i of [0, 1]) {
      for (const f of ['sigma-n', 'sigma-p', 'N-t', 'v-th', 'E-t']) {
        const el = document.getElementById(`idef-${i}-${f}`) as HTMLInputElement
        expect(el.value).toBe('')
      }
    }
  })
})


// ────── Read round-trip ──────────────────────────────────────────────


describe('readDeviceEditor round-trip', () => {
  it('reads populated slot back into interface_defects[k]', () => {
    const cfg = threeLayerConfig({
      interface_defects: [null, pvkEtlDefect()],
    })
    renderDeviceEditor(container, cfg, 'full')
    const out = readDeviceEditor(cfg)
    expect(out.device.interface_defects?.[0]).toBeNull()
    expect(out.device.interface_defects?.[1]).toEqual(pvkEtlDefect())
  })

  it('all-empty inputs collapse the slot to null', () => {
    const cfg = threeLayerConfig({ interface_defects: [null, null] })
    renderDeviceEditor(container, cfg, 'full')
    const out = readDeviceEditor(cfg)
    expect(out.device.interface_defects).toEqual([null, null])
  })

  it('absent on the input config + empty inputs → field stays absent', () => {
    // No interface_defects key on the input config; reader should not
    // emit the field at all so legacy YAMLs round-trip unchanged.
    const cfg = threeLayerConfig()
    renderDeviceEditor(container, cfg, 'full')
    const out = readDeviceEditor(cfg)
    expect('interface_defects' in out.device).toBe(false)
  })

  it('user editing populated slot to empty round-trips as null', () => {
    const cfg = threeLayerConfig({
      interface_defects: [null, pvkEtlDefect()],
    })
    renderDeviceEditor(container, cfg, 'full')
    // Simulate the user clearing every field in row 1.
    for (const f of ['sigma-n', 'sigma-p', 'N-t', 'v-th', 'E-t']) {
      const el = document.getElementById(`idef-1-${f}`) as HTMLInputElement
      el.value = ''
    }
    const out = readDeviceEditor(cfg)
    // Both slots are null now BUT the field is still present in the
    // payload because the original config carried it (legacy callers
    // depend on the round-trip preserving the field's existence when
    // the user starts from a populated preset).
    expect(out.device.interface_defects).toEqual([null, null])
  })

  it('round-trip preserves a fully-populated slot byte-for-byte', () => {
    const cfg = threeLayerConfig({
      interface_defects: [pvkEtlDefect(), pvkEtlDefect()],
    })
    renderDeviceEditor(container, cfg, 'full')
    const out = readDeviceEditor(cfg)
    expect(out.device.interface_defects).toEqual([
      pvkEtlDefect(), pvkEtlDefect(),
    ])
  })
})
