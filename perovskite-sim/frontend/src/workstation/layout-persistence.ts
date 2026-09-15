import { LayoutConfig } from 'golden-layout'
import type { ItemConfig, ResolvedLayoutConfig } from 'golden-layout'

function withContainerMinimums<T extends ItemConfig>(item: T): T {
  // Golden Layout sums the minimum sizes of a stack's tabs during dragging.
  // Tabs share one area, so apply the minimum at the row/column level instead.
  return {
    ...item,
    ...(item.type === 'component' ? { minSize: '0px', minWidth: undefined, minHeight: undefined } : {}),
    ...(item.content ? { content: item.content.map(withContainerMinimums) } : {}),
  }
}

export function configureDockLayout(config: LayoutConfig): LayoutConfig {
  return {
    ...config,
    root: config.root ? withContainerMinimums(config.root) : undefined,
    settings: { ...config.settings, responsiveMode: 'always' },
    dimensions: {
      ...config.dimensions,
      defaultMinItemWidth: '240px',
      borderWidth: 8,
      borderGrabWidth: 16,
    },
  }
}

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
