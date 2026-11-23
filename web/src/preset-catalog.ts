import type { ConfigEntry, SimulationModeName } from './types'

export interface ReferenceModelDescription {
  formulation: string
  mechanisms: ReadonlyArray<string>
  evidence: string
}

// One frontend entry per reference model; protocol variants share that entry.
export const RESEARCH_PRESETS: ReadonlyArray<{
  name: string
  label: string
  mode: SimulationModeName
  description: ReferenceModelDescription
}> = [
  {
    name: 'scaps_pvk_xy.yaml', label: 'SCAPS native (PVK-xy)', mode: 'fast',
    description: {
      formulation: 'Native zero-fit SCAPS-1D heterojunction model',
      mechanisms: [
        'Electron and hole transport; recalculated flat-band contacts',
        '7-level Gaussian PVK/ETL & single HTL/PVK interface SRH (Et = 0.30 eV above EV)',
        'Bulk SRH, radiative and Auger recombination with zero calibration factors',
      ],
      evidence: 'Native PVK-xy.scaps + AM1.5G .spe zero-fit alignment (median |dVoc| = 0.60 mV across 248 numeric cases).',
    },
  },
  {
    name: 'scaps_mirror_v2.yaml', label: 'SCAPS reproduction', mode: 'fast',
    description: {
      formulation: 'Electronic heterojunction model',
      mechanisms: [
        'Electron and hole transport; no mobile ions',
        'Band offsets; bulk and interface SRH recombination',
        'TMM photogeneration; radiative and Auger recombination',
      ],
      evidence: 'Calibrated SCAPS comparison; validation is target-specific.',
    },
  },
  {
    name: 'calado2016_ion_sweep.yaml', label: 'Calado 2016 - Ion hysteresis', mode: 'full',
    description: {
      formulation: 'Mixed ionic-electronic p-i-n model',
      mechanisms: [
        'Electron, hole and positive-ion drift-diffusion',
        'Contact-region SRH and radiative recombination',
        'Uniform photogeneration; continuous voltage history',
      ],
      evidence: 'Internal figure regression; original-paper agreement remains open.',
    },
  },
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

export function presetDescription(name: string): ReferenceModelDescription | undefined {
  return RESEARCH_PRESETS.find(p => p.name === name)?.description
}
