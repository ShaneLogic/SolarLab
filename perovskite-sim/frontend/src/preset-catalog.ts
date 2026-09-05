import type { ConfigEntry, SimulationModeName } from './types'

export const RESEARCH_PRESETS: ReadonlyArray<{
  name: string
  label: string
  mode: SimulationModeName
}> = [
  { name: 'scaps_mirror_v2.yaml', label: 'SCAPS parity - Reference v2', mode: 'fast' },
  { name: 'calado2016_fig1f.yaml', label: 'Calado 2016 - Ion hysteresis', mode: 'legacy' },
]

export function researchPresetEntries(entries: ReadonlyArray<ConfigEntry>): ConfigEntry[] {
  const singleCell = entries.filter(e => !(e.device_type ?? '').startsWith('tandem'))
  // New bundled studies must be added explicitly; user presets stay available.
  const shipped = RESEARCH_PRESETS.flatMap(p => {
    const entry = singleCell.find(e => e.namespace === 'shipped' && e.name === p.name)
    return entry ? [entry] : []
  })
  return [...shipped, ...singleCell.filter(e => e.namespace === 'user')]
}

export function presetLabel(name: string): string {
  return RESEARCH_PRESETS.find(p => p.name === name)?.label ?? name.replace(/\.ya?ml$/, '')
}

export function presetPreferredMode(name: string): SimulationModeName | undefined {
  return RESEARCH_PRESETS.find(p => p.name === name)?.mode
}
