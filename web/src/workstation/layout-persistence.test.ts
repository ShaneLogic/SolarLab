import { describe, expect, it } from 'vitest'
import {
  LayoutConfig,
  ResolvedComponentItemConfig,
  ResolvedLayoutConfig,
  SizeUnitEnum,
} from 'golden-layout'
import type { LayoutConfig as LayoutConfigType } from 'golden-layout'
import { restoreLayoutConfig } from './layout-persistence'

const fallback: LayoutConfigType = {
  root: {
    type: 'component',
    componentType: 'fallback',
  },
}

const savedLayout: LayoutConfigType = {
  root: {
    type: 'row',
    content: [
      {
        type: 'component',
        componentType: 'device',
        width: 42,
      },
      {
        type: 'component',
        componentType: 'main-plot',
        width: 58,
      },
    ],
  },
}

describe('restoreLayoutConfig', () => {
  it('uses the fallback when no layout was persisted', () => {
    expect(restoreLayoutConfig(null, fallback)).toBe(fallback)
  })

  it('keeps an unresolved LayoutConfig unchanged', () => {
    expect(restoreLayoutConfig(savedLayout, fallback)).toBe(savedLayout)
  })

  it('converts a JSON-round-tripped ResolvedLayoutConfig before loading', () => {
    const resolved = {
      ...ResolvedLayoutConfig.createDefault(),
      root: {
        ...ResolvedComponentItemConfig.createDefault('main-plot'),
        size: 58,
        sizeUnit: SizeUnitEnum.Percent,
      },
    }
    const persisted = JSON.parse(JSON.stringify(resolved)) as unknown

    const restored = restoreLayoutConfig(persisted, fallback)

    expect(LayoutConfig.isResolved(restored)).toBe(false)
    expect(restored.root).toMatchObject({
      type: 'component',
      componentType: 'main-plot',
      size: '58%',
    })
  })

  it('rejects malformed primitive state so the caller can use its fallback', () => {
    expect(() => restoreLayoutConfig(42, fallback)).toThrow(
      'Persisted GoldenLayout state must be an object',
    )
  })
})
