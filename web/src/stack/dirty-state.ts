import type { DeviceConfig } from '../types'

/**
 * Return true if `current` differs from `loaded`, comparing the two
 * device configs by deep value equality. The implementation canonicalises
 * each side via JSON.stringify with a sorted-key replacer so insertion
 * order does not produce false positives (the device-pane state spreads
 * objects, which can change key order).
 */
export function isDirty(loaded: DeviceConfig, current: DeviceConfig): boolean {
  return canonical(loaded) !== canonical(current)
}

function canonical(value: unknown): string {
  return JSON.stringify(value, (_key, v) => {
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      const sorted: Record<string, unknown> = {}
      for (const k of Object.keys(v as Record<string, unknown>).sort()) {
        sorted[k] = (v as Record<string, unknown>)[k]
      }
      return sorted
    }
    return v
  })
}
