import { describe, expect, it } from 'vitest'
import { presetLabel, presetPreferredMode, researchPresetEntries } from './preset-catalog'
import type { ConfigEntry } from './types'

describe('focused research preset catalog', () => {
  it('offers the two research baselines in a stable order and preserves user presets', () => {
    const entries: ConfigEntry[] = [
      { name: 'calado2016_fig1f.yaml', namespace: 'shipped' },
      { name: 'cigs_baseline.yaml', namespace: 'shipped' },
      { name: 'driftfusion_calado2016_repro.yaml', namespace: 'shipped' },
      { name: 'scaps_mirror_v2_robin_strong.yaml', namespace: 'shipped' },
      { name: 'scaps_mirror_v2.yaml', namespace: 'shipped' },
      { name: 'my_scan.yaml', namespace: 'user' },
      { name: 'my_tandem.yaml', namespace: 'user', device_type: 'tandem_2T' },
    ]
    const original = structuredClone(entries)
    expect(researchPresetEntries(entries).map(e => e.name)).toEqual([
      'scaps_mirror_v2.yaml', 'calado2016_fig1f.yaml', 'my_scan.yaml',
    ])
    expect(entries).toEqual(original)
  })

  it('does not hide a user configuration with a retired bundled name', () => {
    const entry: ConfigEntry = { name: 'cigs_baseline.yaml', namespace: 'user' }
    expect(researchPresetEntries([entry])).toEqual([entry])
  })

  it('does not fabricate a missing research configuration', () => {
    expect(researchPresetEntries([])).toEqual([])
  })

  it('distinguishes the two studies and their reference physics modes', () => {
    expect(presetLabel('scaps_mirror_v2.yaml')).toBe('SCAPS parity - Reference v2')
    expect(presetPreferredMode('scaps_mirror_v2.yaml')).toBe('fast')
    expect(presetLabel('calado2016_fig1f.yaml')).toBe('Calado 2016 - Ion hysteresis')
    expect(presetPreferredMode('calado2016_fig1f.yaml')).toBe('legacy')
    expect(presetLabel('my_scan.yml')).toBe('my_scan')
    expect(presetPreferredMode('my_scan.yml')).toBeUndefined()
  })
})
