import type { DeviceConfig, SimulationModeName } from '../types'

/** The root object persisted to localStorage. */
export interface Workspace {
  /** Schema version — bump on breaking changes; load() falls back to empty workspace on mismatch. */
  version: 1
  id: string
  name: string
  devices: Device[]
  /** ID of the currently-selected device, or null for "nothing selected". */
  activeDeviceId: string | null
  /** ID of the currently-selected experiment within the active device, or null. */
  activeExperimentId: string | null
  /** ID of the currently-selected run within the active experiment, or null. */
  activeRunId: string | null
  /**
   * Opaque Golden Layout config blob — serialised state of the dockable area.
   * `unknown` by design: we never inspect it, only round-trip it to Golden Layout.
   */
  layout: unknown | null
}

/** A device node in the project tree. Phase 1 always has exactly one (the seeded default). */
export interface Device {
  id: string
  name: string
  tier: SimulationModeName
  config: DeviceConfig
  /** Empty in Phase 1 — experiments are wired up in Phase 2. */
  experiments: Experiment[]
}

export type ExperimentKind = 'jv' | 'impedance' | 'degradation'

/** Phase 2 will populate this. Defined here to keep the type stable across phases. */
export interface Experiment {
  id: string
  kind: ExperimentKind
  params: Record<string, unknown>
  runs: Run[]
}

/** Phase 2 will populate this. */
export interface Run {
  id: string
  timestamp: number
  result: unknown
  activePhysics: string
  durationMs: number
  /** Frozen DeviceConfig snapshot at the moment the run was dispatched. */
  deviceSnapshot: DeviceConfig
}

/** Phase 4 will populate this. */
export interface CompareView {
  id: string
  name: string
  kind: 'jv' | 'impedance' | 'degradation'
  runRefs: Array<{ deviceId: string; experimentId: string; runId: string }>
}
