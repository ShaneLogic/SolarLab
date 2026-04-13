import type { DeviceConfig, LayerConfig, SimulationModeName } from './types'

const MODE_OPTIONS: ReadonlyArray<{ value: SimulationModeName; label: string }> = [
  { value: 'full', label: 'Full (all physics upgrades)' },
  { value: 'fast', label: 'Fast (Beer–Lambert, no TE)' },
  { value: 'legacy', label: 'Legacy (IonMonger-compatible)' },
]

function isModeName(v: unknown): v is SimulationModeName {
  return v === 'full' || v === 'fast' || v === 'legacy'
}

// Groups of parameters for a single layer. Grouping makes long forms scannable.
interface ParamGroup {
  title: string
  fields: Array<{ key: keyof LayerConfig; label: string; unit: string }>
}

const LAYER_GROUPS: ParamGroup[] = [
  {
    title: 'Geometry & Electrostatics',
    fields: [
      { key: 'thickness', label: 'Thickness', unit: 'm' },
      { key: 'eps_r', label: '<i>ε</i><sub>r</sub>', unit: '' },
      { key: 'chi', label: '<i>χ</i>', unit: 'eV' },
      { key: 'Eg', label: '<i>E</i><sub>g</sub>', unit: 'eV' },
    ],
  },
  {
    title: 'Transport',
    fields: [
      { key: 'mu_n', label: '<i>μ</i><sub>n</sub>', unit: 'm²/(V·s)' },
      { key: 'mu_p', label: '<i>μ</i><sub>p</sub>', unit: 'm²/(V·s)' },
      { key: 'ni', label: '<i>n</i><sub>i</sub>', unit: 'm⁻³' },
      { key: 'N_D', label: '<i>N</i><sub>D</sub>', unit: 'm⁻³' },
      { key: 'N_A', label: '<i>N</i><sub>A</sub>', unit: 'm⁻³' },
    ],
  },
  {
    title: 'Recombination',
    fields: [
      { key: 'tau_n', label: '<i>τ</i><sub>n</sub>', unit: 's' },
      { key: 'tau_p', label: '<i>τ</i><sub>p</sub>', unit: 's' },
      { key: 'n1', label: '<i>n</i><sub>1</sub>', unit: 'm⁻³' },
      { key: 'p1', label: '<i>p</i><sub>1</sub>', unit: 'm⁻³' },
      { key: 'B_rad', label: '<i>B</i><sub>rad</sub>', unit: 'm³/s' },
      { key: 'C_n', label: '<i>C</i><sub>n</sub>', unit: 'm⁶/s' },
      { key: 'C_p', label: '<i>C</i><sub>p</sub>', unit: 'm⁶/s' },
    ],
  },
  {
    title: 'Ions & Optics',
    fields: [
      { key: 'D_ion', label: '<i>D</i><sub>ion</sub>', unit: 'm²/s' },
      { key: 'P_lim', label: '<i>P</i><sub>lim</sub>', unit: 'm⁻³' },
      { key: 'P0', label: '<i>P</i><sub>0</sub>', unit: 'm⁻³' },
      { key: 'alpha', label: '<i>α</i>', unit: 'm⁻¹' },
    ],
  },
]

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

function renderLayer(layer: LayerConfig, idx: number): string {
  const groups = LAYER_GROUPS.map(group => {
    const rows = group.fields.map(f => {
      const id = `layer-${idx}-${String(f.key)}`
      const unit = f.unit ? `<span class="unit">${f.unit}</span>` : ''
      return `
        <label class="param">
          <span class="param-label"><span class="sym">${f.label}</span>${unit}</span>
          ${numAttr(id, layer[f.key] as number | undefined)}
        </label>`
    }).join('')
    return `
      <div class="param-group">
        <h5>${group.title}</h5>
        <div class="param-grid">${rows}</div>
      </div>`
  }).join('')

  return `
    <details class="layer-card" ${idx === 0 ? 'open' : ''}>
      <summary>
        <span class="layer-index">${idx + 1}</span>
        <input type="text" class="layer-name" id="layer-${idx}-name" value="${layer.name}" spellcheck="false">
        <select class="layer-role" id="layer-${idx}-role">
          <option value="HTL" ${layer.role === 'HTL' ? 'selected' : ''}>HTL</option>
          <option value="absorber" ${layer.role === 'absorber' ? 'selected' : ''}>absorber</option>
          <option value="ETL" ${layer.role === 'ETL' ? 'selected' : ''}>ETL</option>
        </select>
      </summary>
      <div class="layer-body">${groups}</div>
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
        <span class="iface-label">${left} / ${right}</span>
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

export function renderDeviceEditor(container: HTMLElement, config: DeviceConfig): void {
  const layers = config.layers.map(renderLayer).join('')
  const currentMode: SimulationModeName = isModeName(config.device.mode) ? config.device.mode : 'full'
  const currentT = config.device.T ?? 300
  container.innerHTML = `
    <div class="editor">
      <div class="param-group">
        <h5>Device</h5>
        <div class="param-grid">
          <label class="param">
            <span class="param-label"><span class="sym">Mode</span></span>
            <select class="num-input" id="dev-mode">${renderModeOptions(currentMode)}</select>
          </label>
          <label class="param">
            <span class="param-label"><span class="sym"><i>T</i></span><span class="unit">K</span></span>
            ${numAttr('dev-T', currentT)}
          </label>
          <label class="param">
            <span class="param-label"><span class="sym"><i>V</i><sub>bi</sub></span><span class="unit">V</span></span>
            ${numAttr('dev-Vbi', config.device.V_bi)}
          </label>
          <label class="param">
            <span class="param-label"><span class="sym"><i>Φ</i></span><span class="unit">m⁻²·s⁻¹</span></span>
            ${numAttr('dev-Phi', config.device.Phi)}
          </label>
        </div>
      </div>
      ${renderInterfaces(config)}
      <div class="layer-list">${layers}</div>
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

export function readDeviceEditor(original: DeviceConfig): DeviceConfig {
  const layers: LayerConfig[] = original.layers.map((layer, idx) => {
    const next: LayerConfig = { ...layer }
    next.name = parseText(`layer-${idx}-name`, layer.name)
    next.role = parseText(`layer-${idx}-role`, layer.role)
    for (const group of LAYER_GROUPS) {
      for (const f of group.fields) {
        const id = `layer-${idx}-${String(f.key)}`
        const original_v = (layer[f.key] as number | undefined) ?? 0
        ;(next as unknown as Record<string, number>)[f.key as string] = parseNum(id, original_v)
      }
    }
    return next
  })

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
