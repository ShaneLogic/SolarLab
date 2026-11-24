import Plotly from 'plotly.js-basic-dist-min'
import { baseLayout, axisTitle, plotConfig, PALETTE } from '../plot-theme'
import { IodineAgingClient } from '../iodine-aging-client'
import type {
  IodineAgingResponse,
  IodineImpedanceResponse,
  IodineOverrides,
  IodinePresetCatalog,
  IodinePresetSummary,
  IodineSteadyResponse,
} from '../iodine-aging-client'

export type IodineAnalysisMode = 'steady' | 'impedance' | 'aging'

export interface IodineAgingPanel {
  readonly ready: Promise<void>
  readonly lastResult: IodineSteadyResponse | IodineImpedanceResponse | IodineAgingResponse | undefined
  dispose(): void
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  if (text !== undefined) node.textContent = text
  return node
}

const errorMessage = (err: unknown) => (err instanceof Error ? err.message : String(err))

export function mountIodineAgingPanel(
  root: HTMLElement,
  endpoint: string | URL,
  fetchImpl?: typeof fetch,
): IodineAgingPanel {
  const client = new IodineAgingClient(endpoint, fetchImpl)
  let catalog: IodinePresetCatalog | undefined
  let lastResult: IodineSteadyResponse | IodineImpedanceResponse | IodineAgingResponse | undefined
  let disposed = false
  let abort = new AbortController()
  const plots: HTMLElement[] = []

  const card = element('section')
  card.className = 'card'
  card.dataset.panel = 'iodine-aging'
  card.append(
    element('h2', '1D Iodine Chemistry & Aging · new core'),
    element(
      'p',
      'Coupled 1D (φ, n, p, I⁻, I⁰, I₂) 5-reaction drift-diffusion, small-signal redox capacitance, volatile I₂ outgassing, and Faradaic contact corrosion. Physical status: not_evaluated; scientific qualification: false.',
    ),
  )

  const form = element('form')
  const fields = element('fieldset')
  fields.style.cssText = 'border:0;padding:0'
  const grid = element('div')
  grid.className = 'form-grid'
  fields.append(grid)

  function control<T extends HTMLInputElement | HTMLSelectElement>(
    name: string,
    label: string,
    input: T,
  ): T {
    const wrap = element('label')
    wrap.className = 'form-group'
    input.name = name
    input.setAttribute('aria-label', label)
    wrap.append(element('span', label), input)
    grid.append(wrap)
    return input
  }

  const presetSelect = control('preset', 'Packaged iodine preset', element('select'))
  const modeSelect = control('mode', 'Simulation mode', element('select'))
  for (const [val, text] of [
    ['steady', 'Steady profiles & frozen J–V'],
    ['impedance', 'Small-signal AC redox capacitance'],
    ['aging', 'Multi-stage BDF aging trajectory'],
  ] as const) {
    const opt = element('option', text)
    opt.value = val
    modeSelect.append(opt)
  }

  const kaInput = control('k_a_cm3_s', 'Hole oxidation rate k_a [cm³/s] (optional)', element('input'))
  kaInput.type = 'number'
  kaInput.step = 'any'
  kaInput.placeholder = 'Preset default (e.g. 1e-11)'

  const voltageInput = control('voltage_V', 'Operating bias [V] (steady/AC)', element('input'))
  voltageInput.type = 'number'
  voltageInput.step = 'any'
  voltageInput.placeholder = '0.9'

  const genInput = control('generation_scale', 'Illumination [Suns] (steady/AC)', element('input'))
  genInput.type = 'number'
  genInput.step = 'any'
  genInput.placeholder = '1.0'

  const outgasInput = control('volumetric_escape_s_1', 'Volumetric I₂ outgassing [s⁻¹] (optional)', element('input'))
  outgasInput.type = 'number'
  outgasInput.step = 'any'
  outgasInput.placeholder = 'Preset default'

  const presetInfo = element('p')
  presetInfo.dataset.role = 'preset-info'
  const deckDetails = element('details')
  const deckPre = element('pre')
  deckPre.style.cssText = 'white-space:pre-wrap;max-height:260px;overflow:auto'
  deckDetails.append(element('summary', 'Inspect declarative solarlab.iodine-deck.v1 YAML/JSON mapping'), deckPre)

  const runBtn = element('button', 'Run iodine simulation')
  runBtn.type = 'submit'
  runBtn.className = 'btn btn-primary'
  fields.append(presetInfo, deckDetails, runBtn)
  fields.disabled = true
  form.append(fields)

  const stateEl = element('p', 'Loading packaged iodine presets…')
  stateEl.dataset.role = 'state'
  stateEl.setAttribute('role', 'status')
  const errorEl = element('p')
  errorEl.className = 'status error'
  errorEl.setAttribute('role', 'alert')
  const output = element('div')
  output.dataset.role = 'result'

  card.append(form, stateEl, errorEl, output)
  root.replaceChildren(card)

  function selectedPreset(): IodinePresetSummary | undefined {
    return catalog?.presets.find(p => p.id === presetSelect.value)
  }

  function refreshPresetDetails(): void {
    const p = selectedPreset()
    if (!p) return
    presetInfo.textContent = `${p.description} (${p.reference}) · sha256=${p.sha256.slice(0, 12)}…`
    deckPre.textContent = JSON.stringify(p.deck, null, 2)
  }

  presetSelect.addEventListener('change', refreshPresetDetails)

  function purgeOutput(): void {
    for (const div of plots.splice(0)) Plotly.purge(div)
    output.replaceChildren()
  }

  function addPlotHost(): HTMLDivElement {
    const div = element('div')
    div.style.cssText = 'width:100%;min-height:360px;margin-top:16px'
    output.append(div)
    plots.push(div)
    return div
  }

  async function renderSteady(res: IodineSteadyResponse): Promise<void> {
    purgeOutput()
    const summary = element(
      'p',
      `Deck ${res.deck_id} (${res.domain_mode}, ${res.operating_state_mode}) · V=${res.voltage_V} V, G=${res.generation_scale} Sun · Total I=${res.profiles.total_iodine_atoms_m2.toExponential(4)} m⁻² · wall=${res.wall_s.toFixed(2)} s · scientific_qualification=false`,
    )
    summary.dataset.role = 'summary'
    output.append(summary)

    const p = res.profiles
    const cmCm3 = p.iodide_m3.map(v => Math.max(v * 1e-6, 1))
    const c0Cm3 = p.iodine_atom_m3.map(v => Math.max(v * 1e-6, 1))
    const c2Cm3 = p.iodine_molecule_m3.map(v => Math.max(v * 1e-6, 1))
    const nCm3 = p.electron_density_m3.map(v => Math.max(v * 1e-6, 1))
    const hCm3 = p.hole_density_m3.map(v => Math.max(v * 1e-6, 1))

    await Plotly.newPlot(
      addPlotHost(),
      [
        { x: p.chem_nodes_nm, y: cmCm3, name: '[I⁻]', mode: 'lines', line: { color: PALETTE.primary, width: 2.2 } },
        { x: p.chem_nodes_nm, y: c0Cm3, name: '[I⁰]', mode: 'lines', line: { color: PALETTE.reverse, width: 2.2 } },
        { x: p.chem_nodes_nm, y: c2Cm3, name: '[I₂]', mode: 'lines', line: { color: PALETTE.accent, width: 2.2 } },
        { x: p.electrical_nodes_nm, y: nCm3, name: 'n (e⁻)', mode: 'lines', line: { color: PALETTE.secondary, dash: 'dash' } },
        { x: p.electrical_nodes_nm, y: hCm3, name: 'p (h⁺)', mode: 'lines', line: { color: PALETTE.neutral, dash: 'dot' } },
      ],
      baseLayout({
        title: { text: 'Steady-State Species & Carrier Profiles' },
        xaxis: { title: axisTitle('Position x [nm]') },
        yaxis: { title: axisTitle('Concentration [cm⁻³]'), type: 'log' },
      }),
      plotConfig(`iodine-steady-profiles-${res.deck_id}`),
    )

    if (res.frozen_jv) {
      const jMaCm2 = res.frozen_jv.output_current_A_m2.map(j => j * 0.1)
      await Plotly.newPlot(
        addPlotHost(),
        [
          {
            x: res.frozen_jv.voltages_V,
            y: jMaCm2,
            name: 'Frozen-chemistry J–V',
            mode: 'lines+markers',
            line: { color: PALETTE.primary, width: 2.2 },
          },
        ],
        baseLayout({
          title: { text: 'Frozen-Chemistry Electronic J–V Snapshot' },
          xaxis: { title: axisTitle('Voltage [V]') },
          yaxis: { title: axisTitle('Output Current Density [mA/cm²]') },
        }),
        plotConfig(`iodine-steady-jv-${res.deck_id}`),
      )
    }
  }

  async function renderImpedance(res: IodineImpedanceResponse): Promise<void> {
    purgeOutput()
    const summary = element(
      'p',
      `Deck ${res.deck_id} AC small-signal @ V=${res.voltage_V} V, G=${res.generation_scale} Sun · ${res.frequencies_Hz.length} frequencies · wall=${res.wall_s.toFixed(2)} s · scientific_qualification=false`,
    )
    summary.dataset.role = 'summary'
    output.append(summary)

    const capTraces = Object.entries(res.sweeps).map(([mode, sw], idx) => ({
      x: sw.frequencies_Hz,
      y: sw.apparent_capacitance_F_m2.map(c => Math.max(Math.abs(c) * 100, 1e-12)),
      name: `|C_app| (${mode})`,
      mode: 'lines+markers',
      line: { color: idx === 0 ? PALETTE.primary : PALETTE.reverse, width: 2.2 },
    }))
    await Plotly.newPlot(
      addPlotHost(),
      capTraces,
      baseLayout({
        title: { text: 'Apparent Capacitance Spectrum |C_app(f)|' },
        xaxis: { title: axisTitle('Frequency f [Hz]'), type: 'log' },
        yaxis: { title: axisTitle('|C_app| [µF/cm²]'), type: 'log' },
      }),
      plotConfig(`iodine-capacitance-${res.deck_id}`),
    )
  }

  async function renderAging(res: IodineAgingResponse): Promise<void> {
    purgeOutput()
    const summary = element(
      'p',
      `Deck ${res.deck_id} · ${res.accepted_bdf_steps} BDF steps · max conservation error=${res.max_relative_conservation_error.toExponential(3)} · wall=${res.wall_s.toFixed(2)} s · scientific_qualification=false`,
    )
    summary.dataset.role = 'summary'
    output.append(summary)

    const t = res.trajectory
    await Plotly.newPlot(
      addPlotHost(),
      [
        { x: t.times_s, y: t.in_device_iodine_m2, name: 'In-device I [m⁻²]', mode: 'lines', line: { color: PALETTE.primary, width: 2.2 } },
        { x: t.times_s, y: t.cumulative_escaped_iodine_atoms_m2, name: 'Escaped I₂ atoms [m⁻²]', mode: 'lines', line: { color: PALETTE.reverse, width: 2.2 } },
        { x: t.times_s, y: t.cumulative_immobilized_iodine_atoms_m2, name: 'Immobilized contact I [m⁻²]', mode: 'lines', line: { color: PALETTE.accent, width: 2.2 } },
      ],
      baseLayout({
        title: { text: 'Elemental Iodine Inventory & Loss Trajectory' },
        xaxis: { title: axisTitle('Aging Time t [s]') },
        yaxis: { title: axisTitle('Sheet Density [atoms / m²]') },
      }),
      plotConfig(`iodine-aging-kinetics-${res.deck_id}`),
    )

    const jvCheckpoints = res.checkpoints.filter(c => c.frozen_jv !== null)
    if (jvCheckpoints.length > 0) {
      const colors = [PALETTE.primary, PALETTE.reverse, PALETTE.accent, PALETTE.secondary]
      await Plotly.newPlot(
        addPlotHost(),
        jvCheckpoints.map((ckpt, idx) => ({
          x: ckpt.frozen_jv!.voltages_V,
          y: ckpt.frozen_jv!.output_current_A_m2.map(j => j * 0.1),
          name: `t=${ckpt.time_s}s (${ckpt.stage_name})`,
          mode: 'lines',
          line: { color: colors[idx % colors.length], width: 2.2 },
        })),
        baseLayout({
          title: { text: 'Checkpoint Frozen-Chemistry J–V Evolution' },
          xaxis: { title: axisTitle('Voltage [V]') },
          yaxis: { title: axisTitle('Output Current Density [mA/cm²]') },
        }),
        plotConfig(`iodine-aging-jv-${res.deck_id}`),
      )
    }
  }

  function collectOverrides(): IodineOverrides {
    const overrides: IodineOverrides = {}
    if (kaInput.value.trim() !== '') overrides.k_a_cm3_s = Number(kaInput.value)
    if (voltageInput.value.trim() !== '') overrides.voltage_V = Number(voltageInput.value)
    if (genInput.value.trim() !== '') overrides.generation_scale = Number(genInput.value)
    if (outgasInput.value.trim() !== '') overrides.volumetric_escape_s_1 = Number(outgasInput.value)
    return overrides
  }

  form.addEventListener('submit', event => {
    event.preventDefault()
    if (disposed || !presetSelect.value) return
    errorEl.textContent = ''
    fields.disabled = true
    const mode = modeSelect.value as IodineAnalysisMode
    stateEl.textContent = `Running ${mode} simulation…`
    const req = { preset: presetSelect.value, overrides: collectOverrides() }
    void (async () => {
      try {
        if (mode === 'steady') {
          const res = await client.runSteady(req, abort.signal)
          if (disposed) return
          lastResult = res
          await renderSteady(res)
        } else if (mode === 'impedance') {
          const res = await client.runImpedance(req, abort.signal)
          if (disposed) return
          lastResult = res
          await renderImpedance(res)
        } else {
          const res = await client.runAging(req, abort.signal)
          if (disposed) return
          lastResult = res
          await renderAging(res)
        }
        stateEl.textContent = 'succeeded'
      } catch (err) {
        if (!disposed) {
          errorEl.textContent = errorMessage(err)
          stateEl.textContent = 'failed'
        }
      } finally {
        if (!disposed) fields.disabled = false
      }
    })()
  })

  const ready = (async () => {
    try {
      const found = await client.listPresets(abort.signal)
      if (disposed) return
      catalog = found
      presetSelect.replaceChildren()
      for (const p of found.presets) {
        const opt = element('option', `${p.name} (${p.id})`)
        opt.value = p.id
        presetSelect.append(opt)
      }
      refreshPresetDetails()
      fields.disabled = found.presets.length === 0
      stateEl.textContent = found.presets.length > 0 ? 'ready' : 'unavailable'
    } catch (err) {
      if (!disposed) {
        errorEl.textContent = `Iodine aging service unavailable: ${errorMessage(err)}`
        stateEl.textContent = 'unavailable'
      }
    }
  })()

  return {
    ready,
    get lastResult() {
      return lastResult
    },
    dispose() {
      disposed = true
      abort.abort()
      purgeOutput()
    },
  }
}
