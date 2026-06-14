import type { DeviceConfig, InterfaceDefectFields, LayerConfig, LayerRole, SimulationModeName } from './types'
import { isLayerRole } from './types'
import { isFieldVisible } from './workstation/tier-gating'

const MODE_OPTIONS: ReadonlyArray<{ value: SimulationModeName; label: string }> = [
  { value: 'full', label: 'Full (all physics upgrades)' },
  { value: 'fast', label: 'Fast (Beer–Lambert, no TE)' },
  { value: 'legacy', label: 'Legacy (IonMonger-compatible)' },
]

function isModeName(v: unknown): v is SimulationModeName {
  return v === 'full' || v === 'fast' || v === 'legacy'
}

// Discriminator for per-layer field rendering. Most parameters are numeric;
// optical_material is a select populated from the backend's n,k CSV scan, and
// incoherent is a boolean checkbox (thick substrate Fresnel handling in TMM).
type FieldKind = 'numeric' | 'select-optical-material' | 'boolean'

interface FieldDef {
  key: keyof LayerConfig
  label: string
  kind: FieldKind
  unit?: string
  /** Hover tooltip — short physical-meaning explainer. */
  tooltip?: string
  /** Placeholder hint for an empty numeric input — used to convey the
   *  "0.0 / disabled" sentinel for opt-in physics fields. */
  placeholder?: string
}

// Groups of parameters for a single layer. Grouping makes long forms scannable.
interface ParamGroup {
  title: string
  fields: FieldDef[]
  /** When true, render the group inside a collapsed-by-default
   *  ``<details>`` so the form stays scannable. Used by the
   *  "Advanced 2D Physics" group (FULL-tier-only μ(E) fields). */
  collapsed?: boolean
}

const LAYER_GROUPS: ParamGroup[] = [
  {
    title: 'Geometry & Electrostatics',
    fields: [
      { key: 'thickness', label: 'Thickness', kind: 'numeric', unit: 'm' },
      { key: 'eps_r', label: '<i>ε</i><sub>r</sub>', kind: 'numeric', unit: '' },
      { key: 'chi', label: '<i>χ</i>', kind: 'numeric', unit: 'eV' },
      { key: 'Eg', label: '<i>E</i><sub>g</sub>', kind: 'numeric', unit: 'eV' },
    ],
  },
  {
    title: 'Transport',
    fields: [
      { key: 'mu_n', label: '<i>μ</i><sub>n</sub>', kind: 'numeric', unit: 'm²/(V·s)' },
      { key: 'mu_p', label: '<i>μ</i><sub>p</sub>', kind: 'numeric', unit: 'm²/(V·s)' },
      { key: 'ni', label: '<i>n</i><sub>i</sub>', kind: 'numeric', unit: 'm⁻³' },
      { key: 'N_D', label: '<i>N</i><sub>D</sub>', kind: 'numeric', unit: 'm⁻³' },
      { key: 'N_A', label: '<i>N</i><sub>A</sub>', kind: 'numeric', unit: 'm⁻³' },
    ],
  },
  {
    title: 'Recombination',
    fields: [
      { key: 'tau_n', label: '<i>τ</i><sub>n</sub>', kind: 'numeric', unit: 's' },
      { key: 'tau_p', label: '<i>τ</i><sub>p</sub>', kind: 'numeric', unit: 's' },
      { key: 'n1', label: '<i>n</i><sub>1</sub>', kind: 'numeric', unit: 'm⁻³' },
      { key: 'p1', label: '<i>p</i><sub>1</sub>', kind: 'numeric', unit: 'm⁻³' },
      { key: 'B_rad', label: '<i>B</i><sub>rad</sub>', kind: 'numeric', unit: 'm³/s' },
      { key: 'C_n', label: '<i>C</i><sub>n</sub>', kind: 'numeric', unit: 'm⁶/s' },
      { key: 'C_p', label: '<i>C</i><sub>p</sub>', kind: 'numeric', unit: 'm⁶/s' },
    ],
  },
  {
    title: 'Ions & Optics',
    fields: [
      { key: 'D_ion', label: '<i>D</i><sub>ion</sub>', kind: 'numeric', unit: 'm²/s' },
      { key: 'P_lim', label: '<i>P</i><sub>lim</sub>', kind: 'numeric', unit: 'm⁻³' },
      { key: 'P0', label: '<i>P</i><sub>0</sub>', kind: 'numeric', unit: 'm⁻³' },
      { key: 'alpha', label: '<i>α</i>', kind: 'numeric', unit: 'm⁻¹' },
      { key: 'optical_material', label: 'Optical material', kind: 'select-optical-material' },
      { key: 'incoherent', label: 'Incoherent layer', kind: 'boolean' },
    ],
  },
  // Stage B(c.2) field-dependent mobility μ(E). Hidden under FAST/LEGACY
  // by tier-gating.ts (FULL only). Collapsed by default; sentinel "0"
  // disables the corresponding model on this layer.
  {
    title: 'Advanced 2D Physics — Field-dependent mobility μ(E)',
    collapsed: true,
    fields: [
      {
        key: 'v_sat_n', label: '<i>v</i><sub>sat</sub><sup>n</sup>',
        kind: 'numeric', unit: 'm/s', placeholder: '0 — disabled',
        tooltip: 'Caughey-Thomas saturation velocity for electrons. Caps drift mobility under high field. Typical perovskite ~1e5 m/s. Set 0 to disable.',
      },
      {
        key: 'v_sat_p', label: '<i>v</i><sub>sat</sub><sup>p</sup>',
        kind: 'numeric', unit: 'm/s', placeholder: '0 — disabled',
        tooltip: 'Caughey-Thomas saturation velocity for holes. Set 0 to disable.',
      },
      {
        key: 'ct_beta_n', label: '<i>β</i><sup>n</sup>',
        kind: 'numeric', unit: '', placeholder: '2.0',
        tooltip: 'Caughey-Thomas exponent for electrons. β=2 (Canali silicon-electron form) is the safe default; β=1 (Thornber form) for silicon holes. Only meaningful when v_sat is non-zero.',
      },
      {
        key: 'ct_beta_p', label: '<i>β</i><sup>p</sup>',
        kind: 'numeric', unit: '', placeholder: '2.0',
        tooltip: 'Caughey-Thomas exponent for holes. Only meaningful when v_sat is non-zero.',
      },
      {
        key: 'pf_gamma_n', label: '<i>γ</i><sub>PF</sub><sup>n</sup>',
        kind: 'numeric', unit: '(V/m)^-0.5', placeholder: '0 — disabled',
        tooltip: 'Poole-Frenkel coefficient for electrons. Field-assisted hopping; typical disordered HTL ~3e-4 (V/m)^-0.5. Set 0 to disable.',
      },
      {
        key: 'pf_gamma_p', label: '<i>γ</i><sub>PF</sub><sup>p</sup>',
        kind: 'numeric', unit: '(V/m)^-0.5', placeholder: '0 — disabled',
        tooltip: 'Poole-Frenkel coefficient for holes. Set 0 to disable.',
      },
    ],
  },
]

// Module-level cache for optical-material option list. Populated once per
// mount via setOpticalMaterialOptions() — see device-panel.ts. If the fetch
// fails the cache stays empty and the dropdown simply has no options beyond
// the "(none — Beer-Lambert)" sentinel.
let opticalMaterialOptions: ReadonlyArray<string> = []

export function setOpticalMaterialOptions(options: ReadonlyArray<string>): void {
  opticalMaterialOptions = [...options]
}

// Minimal HTML escape for untrusted-ish string values (material filenames
// come from backend auto-scan of on-disk CSVs — trusted but defensive).
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function fmt(v: unknown): string {
  if (v === undefined || v === null || v === '') return ''
  const n = typeof v === 'number' ? v : Number(v)
  if (!Number.isFinite(n)) return ''
  if (n === 0) return '0'
  const abs = Math.abs(n)
  if (abs >= 1e-3 && abs < 1e4) return String(n)
  return n.toExponential(3)
}

interface NumAttrOpts {
  /** Hint shown when the input is empty — used for "0 / disabled" sentinels. */
  placeholder?: string
  /** Native HTML5 tooltip surfaced on hover. */
  title?: string
}

function numAttr(id: string, value: unknown, opts?: NumAttrOpts): string {
  const placeholderAttr = opts?.placeholder ? ` placeholder="${escapeHtml(opts.placeholder)}"` : ''
  const titleAttr = opts?.title ? ` title="${escapeHtml(opts.title)}"` : ''
  return `<input type="text" class="num-input" id="${id}" value="${fmt(value)}" spellcheck="false"${placeholderAttr}${titleAttr}>`
}

function renderOpticalMaterialSelect(layerIdx: number, currentValue: string | null | undefined): string {
  const id = `layer-${layerIdx}-optical_material`
  const selectedNone = currentValue == null || currentValue === '' ? ' selected' : ''
  const opts = [`<option value=""${selectedNone}>(none — Beer-Lambert)</option>`]
    .concat(
      opticalMaterialOptions.map(m => {
        const safe = escapeHtml(m)
        const sel = currentValue === m ? ' selected' : ''
        return `<option value="${safe}"${sel}>${safe}</option>`
      }),
    )
    .join('')
  return `<select class="num-input" id="${id}" data-layer="${layerIdx}" data-field="optical_material">${opts}</select>`
}

function renderIncoherentCheckbox(layerIdx: number, currentValue: boolean | undefined): string {
  const id = `layer-${layerIdx}-incoherent`
  const checked = currentValue ? ' checked' : ''
  return `<input type="checkbox" id="${id}" data-layer="${layerIdx}" data-field="incoherent"${checked}>`
}

function renderField(layer: LayerConfig, idx: number, f: FieldDef): string {
  const id = `layer-${idx}-${String(f.key)}`
  const unit = f.unit ? `<span class="unit">${f.unit}</span>` : ''
  let control: string
  switch (f.kind) {
    case 'numeric':
      control = numAttr(id, layer[f.key] as number | undefined, {
        placeholder: f.placeholder,
        title: f.tooltip,
      })
      break
    case 'select-optical-material':
      control = renderOpticalMaterialSelect(idx, layer.optical_material)
      break
    case 'boolean':
      control = renderIncoherentCheckbox(idx, layer.incoherent)
      break
  }
  const labelTitle = f.tooltip ? ` title="${escapeHtml(f.tooltip)}"` : ''
  return `
        <label class="param"${labelTitle}>
          <span class="param-label"><span class="sym">${f.label}</span>${unit}</span>
          ${control}
        </label>`
}

function isVisibleField(f: FieldDef, tier: SimulationModeName | undefined): boolean {
  if (!tier) return true
  return isFieldVisible(String(f.key), tier)
}

function renderLayer(
  layer: LayerConfig,
  idx: number,
  tier?: SimulationModeName,
  forceOpen: boolean = false,
): string {
  const groups = LAYER_GROUPS.map(group => {
    const visibleFields = group.fields.filter(f => isVisibleField(f, tier))
    if (visibleFields.length === 0) return ''
    const rows = visibleFields.map(f => renderField(layer, idx, f)).join('')
    if (group.collapsed) {
      return `
      <details class="param-group">
        <summary><h5>${group.title}</h5></summary>
        <div class="param-grid">${rows}</div>
      </details>`
    }
    return `
      <div class="param-group">
        <h5>${group.title}</h5>
        <div class="param-grid">${rows}</div>
      </div>`
  }).join('')

  const openAttr = (forceOpen || idx === 0) ? 'open' : ''
  return `
    <details class="param-card" ${openAttr}>
      <summary>
        <span class="layer-index">${idx + 1}</span>
        <input type="text" class="layer-name" id="layer-${idx}-name" value="${escapeHtml(layer.name)}" spellcheck="false">
        <select class="layer-role" id="layer-${idx}-role">
          <option value="substrate" ${layer.role === 'substrate' ? 'selected' : ''}>substrate</option>
          <option value="front_contact" ${layer.role === 'front_contact' ? 'selected' : ''}>front contact</option>
          <option value="ETL" ${layer.role === 'ETL' ? 'selected' : ''}>ETL</option>
          <option value="absorber" ${layer.role === 'absorber' ? 'selected' : ''}>absorber</option>
          <option value="HTL" ${layer.role === 'HTL' ? 'selected' : ''}>HTL</option>
          <option value="back_contact" ${layer.role === 'back_contact' ? 'selected' : ''}>back contact</option>
        </select>
      </summary>
      <div class="param-card-body">${groups}</div>
    </details>`
}

function renderInterfaces(config: DeviceConfig): string {
  const n = Math.max(0, config.layers.length - 1)
  const rows: string[] = []
  for (let i = 0; i < n; i++) {
    const pair = config.device.interfaces?.[i] ?? [0, 0]
    const left = config.layers[i]?.name ?? `layer ${i + 1}`
    const right = config.layers[i + 1]?.name ?? `layer ${i + 2}`
    rows.push(`
      <div class="iface-row">
        <span class="iface-label">${escapeHtml(left)} / ${escapeHtml(right)}</span>
        <label class="param">
          <span class="param-label"><span class="sym"><i>v</i><sub>n</sub></span><span class="unit">m/s</span></span>
          ${numAttr(`iface-${i}-vn`, pair[0])}
        </label>
        <label class="param">
          <span class="param-label"><span class="sym"><i>v</i><sub>p</sub></span><span class="unit">m/s</span></span>
          ${numAttr(`iface-${i}-vp`, pair[1])}
        </label>
      </div>`)
  }
  if (!rows.length) return ''
  return `
    <div class="param-group">
      <h5>Interface Recombination (SRV)</h5>
      <div class="iface-list">${rows.join('')}</div>
    </div>`
}

/**
 * FULL-tier-only Stage B(c.1) Robin / selective-contacts panel. Maps the
 * four ``DeviceConfig.device.S_{n,p}_{left,right}`` fields to UI labels
 * "Top contact (HTL side)" / "Bottom contact (ETL side)" — matches the
 * y-axis convention exposed by the workstation 2D pane and
 * ``MaterialArrays2D.S_{n,p}_{top,bot}``. The YAML keys remain the
 * original 1D names (``S_n_left`` etc.) for backwards compatibility with
 * the 1D Phase 3.3 hook; this is a UI-only relabel for the 2D mental
 * model. Empty input is the "absent" sentinel (round-trips as ``null``);
 * an explicit ``0`` is the "perfectly blocking (Neumann)" limit; large
 * values (≥ 10³ m/s) approach the ohmic limit.
 */
function renderRobinContacts(config: DeviceConfig): string {
  const d = config.device
  const help = '<p class="param-help">Surface recombination velocities at the outer contacts. Empty = disabled (Dirichlet ohmic); explicit <code>0</code> = perfectly blocking (Neumann); ≥ 10³ m/s approaches the ohmic limit. <strong>Top</strong> = HTL side (y=0, YAML <code>S_*_left</code>); <strong>Bottom</strong> = ETL side (y=Ny−1, YAML <code>S_*_right</code>).</p>'
  return `
      <details class="param-group">
        <summary><h5>Advanced 2D Physics — Robin contacts (B(c.1))</h5></summary>
        ${help}
        <div class="param-grid">
          <label class="param" title="Electron surface velocity at the top contact (HTL side, YAML S_n_left). Empty disables; 0 is blocking; ≥ 1e3 m/s approaches ohmic.">
            <span class="param-label"><span class="sym"><i>S</i><sub>n</sub><sup>top</sup></span><span class="unit">m/s</span></span>
            ${numAttr('dev-S-n-top', d.S_n_left, { placeholder: '0 — disabled', title: 'Top electron Robin S (YAML S_n_left)' })}
          </label>
          <label class="param" title="Hole surface velocity at the top contact (HTL side, YAML S_p_left). Empty disables; 0 is blocking; ≥ 1e3 m/s approaches ohmic.">
            <span class="param-label"><span class="sym"><i>S</i><sub>p</sub><sup>top</sup></span><span class="unit">m/s</span></span>
            ${numAttr('dev-S-p-top', d.S_p_left, { placeholder: '0 — disabled', title: 'Top hole Robin S (YAML S_p_left)' })}
          </label>
          <label class="param" title="Electron surface velocity at the bottom contact (ETL side, YAML S_n_right). Empty disables; 0 is blocking; ≥ 1e3 m/s approaches ohmic.">
            <span class="param-label"><span class="sym"><i>S</i><sub>n</sub><sup>bot</sup></span><span class="unit">m/s</span></span>
            ${numAttr('dev-S-n-bot', d.S_n_right, { placeholder: '0 — disabled', title: 'Bottom electron Robin S (YAML S_n_right)' })}
          </label>
          <label class="param" title="Hole surface velocity at the bottom contact (ETL side, YAML S_p_right). Empty disables; 0 is blocking; ≥ 1e3 m/s approaches ohmic.">
            <span class="param-label"><span class="sym"><i>S</i><sub>p</sub><sup>bot</sup></span><span class="unit">m/s</span></span>
            ${numAttr('dev-S-p-bot', d.S_p_right, { placeholder: '0 — disabled', title: 'Bottom hole Robin S (YAML S_p_right)' })}
          </label>
        </div>
      </details>`
}

/**
 * FULL-tier-only SCAPS-validation physics panel (device-level). Surfaces the
 * five flags the YAML loader and ``stack_from_dict`` both parse — DOS band
 * potentials, flat-band contacts, interface-plane closure / projection, and
 * the heterointerface bulk-Auger de-spike fraction — so a user loading a
 * parity preset (e.g. scaps_mirror_v2) can see and round-trip them instead
 * of having them silently stripped at the inline-device boundary.
 */
function renderScapsPhysics(config: DeviceConfig): string {
  const d = config.device
  const help = '<p class="param-help">SCAPS-validation physics (device-level). <strong>DOS band potentials</strong> adds the V<sub>T</sub>·ln(N<sub>C</sub>/N<sub>V</sub>) quasi-Fermi step at DOS-contrast heterojunctions (closes the V<sub>oc</sub> gap; needs per-layer N<sub>C</sub>/N<sub>V</sub>). <strong>Flat-band contacts</strong> uses SCAPS finite-S metal contacts. <strong>Interface-plane closure / projection</strong> evaluate interface recombination on the band-bending-suppressed plane. <strong>De-spike f</strong> blends the heterointerface-node density toward the neighbour geometric mean in the bulk Auger rate (0 = off, 0.53 = SCAPS-emulation).</p>'
  const cb = (id: string, label: string, on: boolean, title: string): string => `
          <label class="param" title="${title}">
            <span class="param-label">${label}</span>
            <input type="checkbox" id="${id}"${on ? ' checked' : ''}>
          </label>`
  return `
      <details class="param-group">
        <summary><h5>SCAPS-validation physics</h5></summary>
        ${help}
        <div class="param-grid">
          ${cb('dev-dos', 'DOS band potentials', !!d.dos_band_potentials, 'V_T·ln(DOS) quasi-Fermi step (YAML dos_band_potentials)')}
          ${cb('dev-flatband', 'Flat-band contacts', !!d.flat_band_contacts, 'SCAPS finite-S metal contacts (YAML flat_band_contacts)')}
          ${cb('dev-iface-closure', 'Interface-plane closure', !!d.interface_plane_closure, 'QSS plane-density interface SRH (YAML interface_plane_closure)')}
          ${cb('dev-iface-proj', 'Interface-plane projection', !!d.interface_plane_projection, 'phi-projected interface densities (YAML interface_plane_projection)')}
          <label class="param" title="Heterointerface bulk-Auger de-spike fraction (YAML het_recomb_despike). 0 = off; 0.53 = SCAPS-emulation.">
            <span class="param-label"><span class="sym">de-spike <i>f</i></span></span>
            ${numAttr('dev-despike', d.het_recomb_despike, { placeholder: '0 — off', title: 'het_recomb_despike (0 = off, 0.53 = SCAPS-emulation)' })}
          </label>
        </div>
      </details>`
}

/**
 * Phase E1.8 — FULL-tier-only per-heterointerface SCAPS defect panel.
 * Collapsed ``<details>`` placed below the Robin contacts panel; one
 * row per electrical heterointerface (HTL/PVK, PVK/ETL, …) auto-derived
 * from ``config.layers``. Each row exposes the 5 SCAPS fields
 * (σ_n, σ_p, N_t areal, v_th, E_t below CB) typed ``number | null`` —
 * empty input is the "absent" sentinel (round-trips as ``null``).
 * Mirrors the YAML schema parsed by ``scaps_compat/loader.py`` and the
 * backend ``stack_from_dict`` plumbing.
 */
function renderInterfaceDefects(config: DeviceConfig): string {
  const n = Math.max(0, config.layers.length - 1)
  if (n === 0) return ''
  const help = '<p class="param-help">SCAPS-style per-heterointerface SRH defect. Each row contributes ``σ·v_th·N_t`` surface velocities to <code>DeviceStack.interfaces[k]</code> and an <code>InterfaceDefect(E_t_eV)</code> entry to <code>DeviceStack.interface_defects[k]</code>. Empty fields = absent (no defect on this interface). Calibration ratio between SCAPS direct N_t and SolarLab effective N_t is ~10⁻⁴ for PVK/ETL — see <code>docs/scaps_validation_report.md</code>.</p>'
  const rows: string[] = []
  for (let i = 0; i < n; i++) {
    const defect = config.device.interface_defects?.[i] ?? null
    const left = config.layers[i]?.name ?? `layer ${i + 1}`
    const right = config.layers[i + 1]?.name ?? `layer ${i + 2}`
    rows.push(`
      <div class="iface-row">
        <span class="iface-label">${escapeHtml(left)} / ${escapeHtml(right)}</span>
        <label class="param" title="Electron capture cross-section [cm²]">
          <span class="param-label"><span class="sym">σ<sub>n</sub></span><span class="unit">cm²</span></span>
          ${numAttr(`idef-${i}-sigma-n`, defect?.sigma_n_cm2, { placeholder: '— disabled' })}
        </label>
        <label class="param" title="Hole capture cross-section [cm²]">
          <span class="param-label"><span class="sym">σ<sub>p</sub></span><span class="unit">cm²</span></span>
          ${numAttr(`idef-${i}-sigma-p`, defect?.sigma_p_cm2, { placeholder: '— disabled' })}
        </label>
        <label class="param" title="Areal trap density at the interface plane [cm⁻²]">
          <span class="param-label"><span class="sym"><i>N</i><sub>t</sub></span><span class="unit">cm⁻²</span></span>
          ${numAttr(`idef-${i}-N-t`, defect?.N_t_cm2, { placeholder: '— disabled' })}
        </label>
        <label class="param" title="Thermal velocity (typically 1e7 cm/s) [cm/s]">
          <span class="param-label"><span class="sym"><i>v</i><sub>th</sub></span><span class="unit">cm/s</span></span>
          ${numAttr(`idef-${i}-v-th`, defect?.v_th_cm_s, { placeholder: '— disabled' })}
        </label>
        <label class="param" title="Trap energy referenced as E_C(reference side) − E_t [eV]; reference is the absorber if exactly one adjacent layer is an absorber, else the lower-Eg side.">
          <span class="param-label"><span class="sym"><i>E</i><sub>t</sub></span><span class="unit">eV</span></span>
          ${numAttr(`idef-${i}-E-t`, defect?.E_t_eV_below_cb, { placeholder: '— disabled' })}
        </label>
      </div>`)
  }
  return `
      <details class="param-group">
        <summary><h5>Interface Defects (FULL only) — Phase E1.5</h5></summary>
        ${help}
        <div class="iface-list">${rows.join('')}</div>
      </details>`
}

function renderModeOptions(current: SimulationModeName): string {
  return MODE_OPTIONS
    .map(o => `<option value="${o.value}"${o.value === current ? ' selected' : ''}>${o.label}</option>`)
    .join('')
}

export function renderDeviceEditor(
  container: HTMLElement,
  config: DeviceConfig,
  tier?: SimulationModeName,
  selectedLayerIdx?: number,
): void {
  const singleLayer = selectedLayerIdx != null && tier === 'full'
  const layerHtml = singleLayer
    ? renderLayer(config.layers[selectedLayerIdx!], selectedLayerIdx!, tier, true)
    : config.layers.map((layer, idx) => renderLayer(layer, idx, tier)).join('')
  const currentMode: SimulationModeName = isModeName(config.device.mode) ? config.device.mode : 'full'
  const currentT = config.device.T ?? 300
  const showT = !tier || isFieldVisible('T', tier)
  const tField = showT ? `
          <label class="param">
            <span class="param-label"><span class="sym"><i>T</i></span><span class="unit">K</span></span>
            ${numAttr('dev-T', currentT)}
          </label>` : ''
  const deviceGroup = singleLayer ? '' : `
      <div class="param-group">
        <h5>Device</h5>
        <div class="param-grid">
          <label class="param">
            <span class="param-label"><span class="sym">Mode</span></span>
            <select class="num-input" id="dev-mode">${renderModeOptions(currentMode)}</select>
          </label>${tField}
          <label class="param">
            <span class="param-label"><span class="sym"><i>V</i><sub>bi</sub></span><span class="unit">V</span></span>
            ${numAttr('dev-Vbi', config.device.V_bi)}
          </label>
          <label class="param">
            <span class="param-label"><span class="sym"><i>Φ</i></span><span class="unit">m⁻²·s⁻¹</span></span>
            ${numAttr('dev-Phi', config.device.Phi)}
          </label>
        </div>
      </div>`
  const interfacesHtml = singleLayer ? '' : renderInterfaces(config)
  // Stage B(c.1) Robin contacts panel — FULL-tier-only because the
  // ``use_selective_contacts`` flag is off in LEGACY/FAST (mode.py:54-86).
  // Hidden in the single-layer drill-down too, where the panel would lose
  // context (it is a device-level setting, not a per-layer one).
  const robinHtml = !singleLayer && tier === 'full' ? renderRobinContacts(config) : ''
  // Phase E1.8 — interface defects panel placed below Robin contacts.
  // FULL-tier-gated (matches the underlying ``InterfaceDefect`` solver
  // hook from Phase E1.5). Hidden in single-layer drill-down because
  // the panel is device-level.
  const interfaceDefectsHtml =
    !singleLayer && tier === 'full' ? renderInterfaceDefects(config) : ''
  // SCAPS-validation physics panel — FULL-tier-only and device-level, same
  // gating as Robin contacts. Hidden panels round-trip as a no-op because
  // readDeviceEditor falls back to original.device.* when the inputs are
  // absent (and single-layer drill-down returns original.device verbatim).
  const scapsHtml = !singleLayer && tier === 'full' ? renderScapsPhysics(config) : ''
  container.innerHTML = `
    <div class="editor">
      ${deviceGroup}
      ${robinHtml}
      ${scapsHtml}
      ${interfaceDefectsHtml}
      ${interfacesHtml}
      <div class="layer-list">${layerHtml}</div>
    </div>`
}

function parseNum(id: string, fallback: number): number {
  const el = document.getElementById(id) as HTMLInputElement | null
  if (!el) return fallback
  const v = Number(el.value)
  return Number.isFinite(v) ? v : fallback
}

/**
 * Like ``parseNum`` but the empty-string case is a meaningful sentinel
 * (``null`` = "absent / disabled") rather than a fallback. Used for the
 * Stage B(c.1) Robin S fields where the backend distinguishes
 * ``None`` (Dirichlet ohmic), ``0`` (Neumann blocking), and a positive
 * finite value (Robin). When the input element does not exist (e.g.
 * because the panel is hidden under a non-FULL tier), the original
 * value is preserved verbatim — including ``undefined``, which is the
 * "field never set on this preset" state.
 */
function parseNumOrNull(
  id: string,
  fallback: number | null | undefined,
): number | null | undefined {
  const el = document.getElementById(id) as HTMLInputElement | null
  if (!el) return fallback
  const raw = el.value.trim()
  if (raw === '') return null
  const v = Number(raw)
  return Number.isFinite(v) ? v : fallback
}

function parseText(id: string, fallback: string): string {
  const el = document.getElementById(id) as HTMLInputElement | HTMLSelectElement | null
  return el?.value ?? fallback
}

function parseRole(id: string, fallback: LayerRole): LayerRole {
  const el = document.getElementById(id) as HTMLSelectElement | null
  const v = el?.value
  return isLayerRole(v) ? v : fallback
}

function parseOpticalMaterial(layerIdx: number, fallback: string | null | undefined): string | null {
  const el = document.getElementById(`layer-${layerIdx}-optical_material`) as HTMLSelectElement | null
  if (!el) return fallback ?? null
  const v = el.value
  return v === '' ? null : v
}

function parseCheckbox(id: string, fallback: boolean): boolean {
  const el = document.getElementById(id) as HTMLInputElement | null
  if (!el) return fallback
  return el.checked
}

export function readDeviceEditor(
  original: DeviceConfig,
  selectedLayerIdx?: number,
): DeviceConfig {
  const singleLayer = selectedLayerIdx != null
  const layers: LayerConfig[] = original.layers.map((layer, idx) => {
    if (singleLayer && idx !== selectedLayerIdx) {
      return layer
    }
    const next: LayerConfig = { ...layer }
    next.name = parseText(`layer-${idx}-name`, layer.name)
    next.role = parseRole(`layer-${idx}-role`, layer.role)
    for (const group of LAYER_GROUPS) {
      for (const f of group.fields) {
        const id = `layer-${idx}-${String(f.key)}`
        switch (f.kind) {
          case 'numeric': {
            const original_v = (layer[f.key] as number | undefined) ?? 0
            ;(next as unknown as Record<string, number>)[f.key as string] = parseNum(id, original_v)
            break
          }
          case 'select-optical-material': {
            next.optical_material = parseOpticalMaterial(idx, layer.optical_material)
            break
          }
          case 'boolean': {
            next.incoherent = parseCheckbox(id, layer.incoherent ?? false)
            break
          }
        }
      }
    }
    return next
  })

  if (singleLayer) {
    return { device: original.device, layers }
  }

  const interfaces: Array<[number, number]> = []
  for (let i = 0; i < layers.length - 1; i++) {
    const existing = original.device.interfaces?.[i] ?? [0, 0]
    interfaces.push([
      parseNum(`iface-${i}-vn`, existing[0]),
      parseNum(`iface-${i}-vp`, existing[1]),
    ])
  }
  // Phase E1.8 — read interface defects panel. Each slot is "absent"
  // (null in the round-trip payload) iff EVERY field is empty input;
  // a fully-populated slot serialises into an ``InterfaceDefectFields``
  // object for backend ``stack_from_dict``. Mixed half-populated slots
  // are not allowed by contract (backend rejects), so the reader
  // collapses them to null to surface the user's intent cleanly.
  const interface_defects: Array<InterfaceDefectFields | null> = []
  for (let i = 0; i < layers.length - 1; i++) {
    const existing = original.device.interface_defects?.[i] ?? null
    const parsed: InterfaceDefectFields = {
      sigma_n_cm2: parseNumOrNull(`idef-${i}-sigma-n`, existing?.sigma_n_cm2 ?? null) ?? null,
      sigma_p_cm2: parseNumOrNull(`idef-${i}-sigma-p`, existing?.sigma_p_cm2 ?? null) ?? null,
      N_t_cm2: parseNumOrNull(`idef-${i}-N-t`, existing?.N_t_cm2 ?? null) ?? null,
      v_th_cm_s: parseNumOrNull(`idef-${i}-v-th`, existing?.v_th_cm_s ?? null) ?? null,
      E_t_eV_below_cb: parseNumOrNull(`idef-${i}-E-t`, existing?.E_t_eV_below_cb ?? null) ?? null,
    }
    const allNull = Object.values(parsed).every(v => v == null)
    interface_defects.push(allNull ? null : parsed)
  }
  const anyDefectPopulated = interface_defects.some(d => d != null)
  const interfaceDefectsField = anyDefectPopulated
    ? { interface_defects }
    : (original.device.interface_defects !== undefined
      ? { interface_defects }
      : {})
  // SCAPS-validation physics flags. Read only when the FULL-tier panel is
  // rendered; otherwise parseCheckbox / parseNumOrNull fall back to the
  // original value so a non-FULL round-trip preserves them verbatim. Each
  // flag is spread in only when truthy / non-zero so non-SCAPS configs keep
  // a clean device payload (no spurious ``false`` / ``0`` fields).
  const scapsPhysicsField: Record<string, boolean | number> = {}
  if (parseCheckbox('dev-dos', !!original.device.dos_band_potentials))
    scapsPhysicsField.dos_band_potentials = true
  if (parseCheckbox('dev-flatband', !!original.device.flat_band_contacts))
    scapsPhysicsField.flat_band_contacts = true
  if (parseCheckbox('dev-iface-closure', !!original.device.interface_plane_closure))
    scapsPhysicsField.interface_plane_closure = true
  if (parseCheckbox('dev-iface-proj', !!original.device.interface_plane_projection))
    scapsPhysicsField.interface_plane_projection = true
  const despike = parseNumOrNull('dev-despike', original.device.het_recomb_despike ?? null)
  if (typeof despike === 'number' && despike !== 0)
    scapsPhysicsField.het_recomb_despike = despike
  const rawMode = parseText('dev-mode', original.device.mode ?? 'full')
  const mode: SimulationModeName = isModeName(rawMode) ? rawMode : 'full'
  const T = parseNum('dev-T', original.device.T ?? 300)
  return {
    device: {
      V_bi: parseNum('dev-Vbi', original.device.V_bi),
      Phi: parseNum('dev-Phi', original.device.Phi),
      interfaces,
      T,
      mode,
      // Stage B(c.1) Robin contacts. Read by ID matching the renderRobin-
      // Contacts panel; when the panel is not rendered (non-FULL tier or
      // single-layer drill-down) ``parseNumOrNull`` returns the
      // ``original.device.S_*`` value, so a round-trip through readDevice-
      // Editor in those modes is a no-op.
      S_n_left: parseNumOrNull('dev-S-n-top', original.device.S_n_left),
      S_p_left: parseNumOrNull('dev-S-p-top', original.device.S_p_left),
      S_n_right: parseNumOrNull('dev-S-n-bot', original.device.S_n_right),
      S_p_right: parseNumOrNull('dev-S-p-bot', original.device.S_p_right),
      // Phase E1.8 — spread the interface_defects field conditionally
      // so absent → still absent (no spurious null array in the payload
      // for presets that pre-date E1.5 or for non-FULL tier round-trips
      // where the panel is hidden).
      ...interfaceDefectsField,
      ...scapsPhysicsField,
    },
    layers,
  }
}
