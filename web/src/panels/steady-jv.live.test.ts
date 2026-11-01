// @vitest-environment node
/** Run explicitly with SOLARLAB_STEADY_UI_ENDPOINT pointing to the Vite proxy.
 * Only Plotly's canvas is replaced in jsdom; HTTP, SSE, worker and artifacts are real. */
import { afterEach, expect, it, vi } from 'vitest'
import Plotly from 'plotly.js-basic-dist-min'
import { mountSteadyJVPanel } from './steady-jv'
import { webcrypto } from '../test-helpers/schema-fixture.mjs'

vi.mock('plotly.js-basic-dist-min', () => ({ default: { newPlot: vi.fn().mockResolvedValue(undefined), purge: vi.fn() } }))
afterEach(() => { vi.unstubAllGlobals() })

// This opt-in test runs in Node/jsdom; keep Node globals out of the app's DOM-only types.
const env = (globalThis as unknown as { process: { env: Record<string, string | undefined> } }).process.env

it.skipIf(!env.SOLARLAB_STEADY_UI_ENDPOINT)('runs one actual 37-node DC job through the browser-facing proxy and stored JS reader', async () => {
  const domModule: string = 'jsdom'
  const { JSDOM } = await import(/* @vite-ignore */ domModule) as { JSDOM: new (html: string) => { window: Window & typeof globalThis } }
  const dom = new JSDOM('<!doctype html><body></body>')
  vi.stubGlobal('document', dom.window.document)
  vi.stubGlobal('crypto', webcrypto)
  const root = document.createElement('div'); document.body.append(root)
  const panel = mountSteadyJVPanel(root, env.SOLARLAB_STEADY_UI_ENDPOINT!)
  try {
    await panel.ready
    expect(root.querySelector('[role=alert]')?.textContent).toBe('')
    expect(root.querySelector<HTMLSelectElement>('select[name=device]')?.value).toBe('research_1d')
    root.querySelector<HTMLButtonElement>('button[type=submit]')!.click()
    await panel.completion
    expect(root.querySelector('[role=alert]')?.textContent).toBe('')
    expect(root.querySelector('[data-role=state]')?.textContent).toBe('succeeded')
    const result = panel.result!
    expect(result.arrays.potential_V.shape).toEqual([3, 37])
    expect(result.arrays.voltage_V.values).toEqual([0, .05, .1])
    expect(Object.keys(result.arrays)).toHaveLength(8)
    expect(result.evidence.every(point => (point as { accepted: boolean }).accepted)).toBe(true)
    expect(result.metrics.open_circuit_voltage_V).toBeNull()
    expect(root.querySelector('[data-metric=open_circuit_voltage_V]')?.textContent).toContain('Unknown')
    const events = [...root.querySelectorAll('[data-role=history] li')].map(node => node.textContent!)
    expect(events.some(text => text.endsWith('queued'))).toBe(true)
    expect(events.some(text => text.endsWith('running'))).toBe(true)
    expect(events.filter(text => text.endsWith('terminal'))).toHaveLength(1)
    expect(Plotly.newPlot).toHaveBeenCalledTimes(4)
    const jv = vi.mocked(Plotly.newPlot).mock.calls.find(([div]) => (div as HTMLElement).dataset.plot === 'J–V')!
    expect(jv[1][0].y).toEqual(result.arrays.output_current_A_m2.values)
    if (env.SOLARLAB_STEADY_UI_EVIDENCE) {
      const fsModule: string = 'node:fs/promises'
      const { writeFile } = await import(/* @vite-ignore */ fsModule) as { writeFile(path: string, content: string): Promise<void> }
      const arrays = Object.fromEntries(await Promise.all(Object.entries(result.arrays).map(async ([name, array]) => {
        const hash = await crypto.subtle.digest('SHA-256', new Float64Array(array.values).buffer)
        return [name, { shape: array.shape, word_sha256: Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join('') }]
      })))
      await writeFile(env.SOLARLAB_STEADY_UI_EVIDENCE, JSON.stringify({ endpoint: env.SOLARLAB_STEADY_UI_ENDPOINT,
        backend_commit: env.SOLARLAB_STEADY_UI_COMMIT, target: panel.job!.target,
        store_id: panel.job!.reader.storeId, events, terminal_count: 1, metrics: result.metrics,
        worker: result.worker, plots: vi.mocked(Plotly.newPlot).mock.calls.map(([div, traces, layout]) => ({ name: (div as HTMLElement).dataset.plot, traces, layout })),
        arrays,
      }, null, 2) + '\n')
    }
  } finally { panel.dispose(); root.remove(); dom.window.close() }
}, 90000)
