import type { DeviceConfig, LayerConfig, LayerRole, SimulationModeName } from './types'
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
}

// Groups of parameters for a single layer. Grouping makes long forms scannable.
interface ParamGroup {
  title: string
  fields: FieldDef[]
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

function numAttr(id: string, value: unknown): string {
  return `<input type="text" class="num-input" id="${id}" value="${fmt(value)}" spellcheck="false">`
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
      control = numAttr(id, layer[f.key] as number | undefined)
      break
    case 'select-optical-material':
      control = renderOpticalMaterialSelect(idx, layer.optical_material)
      break
    case 'boolean':
      control = renderIncoherentCheckbox(idx, layer.incoherent)
      break
  }
  return `
        <label class="param">
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
  container.innerHTML = `
    <div class="editor">
      ${deviceGroup}
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
    },
    layers,
  }
}
