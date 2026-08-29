import { LayoutConfig } from 'golden-layout'
import type { ResolvedLayoutConfig } from 'golden-layout'

export function restoreLayoutConfig(
  persisted: unknown,
  fallback: LayoutConfig,
): LayoutConfig {
  if (persisted === null || persisted === undefined) return fallback
  if (typeof persisted !== 'object') {
    throw new TypeError('Persisted GoldenLayout state must be an object')
  }

  const config = persisted as LayoutConfig | ResolvedLayoutConfig
  return LayoutConfig.isResolved(config)
    ? LayoutConfig.fromResolved(config)
    : config
}
