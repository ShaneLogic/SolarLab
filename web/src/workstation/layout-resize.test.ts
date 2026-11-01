import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { GoldenLayout, type LayoutConfig } from 'golden-layout'
import { configureDockLayout, restoreLayoutConfig } from './layout-persistence'

const layouts: GoldenLayout[] = []

beforeEach(() => {
  vi.useFakeTimers()
  // jsdom does not perform CSS layout. Golden Layout itself writes the pixel
  // sizes; expose those sizes through the browser measurements it consumes.
  for (const [property, dimension] of [['offsetWidth', 'width'], ['offsetHeight', 'height']] as const) {
    vi.spyOn(HTMLElement.prototype, property, 'get').mockImplementation(function (this: HTMLElement) {
      const size = this.style[dimension]
      if (size.endsWith('%')) {
        const parentSize = this.parentElement?.[property] ?? 0
        return parentSize * Number.parseFloat(size) / 100
      }
      return Number.parseFloat(size) || 0
    })
  }
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  })
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => (
    setTimeout(() => callback(performance.now()), 16)
  ))
  vi.stubGlobal('cancelAnimationFrame', (handle: number) => clearTimeout(handle))
})

afterEach(() => {
  for (const layout of layouts.splice(0)) layout.destroy()
  document.body.replaceChildren()
  vi.clearAllTimers()
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

const previousLayout: LayoutConfig = {
  dimensions: { defaultMinItemWidth: '320px' },
  root: {
    type: 'row',
    content: [
      {
        type: 'stack', size: '33.33333333333333%',
        content: [{ type: 'component', componentType: 'device' }],
      },
      {
        type: 'stack', size: '33.33333333333333%',
        content: [
          { type: 'component', componentType: 'experiments' },
          { type: 'component', componentType: 'tandem' },
        ],
      },
      {
        type: 'stack', size: '33.33333333333334%',
        content: [{ type: 'component', componentType: 'main-plot' }],
      },
    ],
  },
}

function mount(config: LayoutConfig): GoldenLayout {
  const container = document.createElement('div')
  // Three 400 px panels plus two 8 px dividers.
  container.style.width = '1216px'
  container.style.height = '800px'
  document.body.appendChild(container)
  const layout = new GoldenLayout(container)
  layouts.push(layout)
  for (const name of ['device', 'experiments', 'tandem', 'main-plot']) {
    layout.registerComponentFactoryFunction(name, component => {
      component.element.textContent = name
    })
  }
  layout.loadLayout(configureDockLayout(config))
  vi.runOnlyPendingTimers()
  return layout
}

function widths(layout: GoldenLayout): number[] {
  return layout.rootItem!.contentItems.map(item => item.element.offsetWidth)
}

function expectSameSplit(actual: number[], expected: number[]): void {
  expect(actual).toHaveLength(expected.length)
  // Loading percentages involves another allocation and rounding pass.
  expected.forEach((width, index) => {
    expect(Math.abs(actual[index] - width)).toBeLessThanOrEqual(2)
  })
}

function pointer(target: EventTarget, type: string, x: number): void {
  // MouseEvent supplies pageX in jsdom; Golden Layout also checks isPrimary.
  const event = new MouseEvent(type, { clientX: x, clientY: 100, bubbles: true, cancelable: true })
  Object.defineProperty(event, 'isPrimary', { value: true })
  target.dispatchEvent(event)
}

function dragMiddleRightDivider(layout: GoldenLayout, offset: number): void {
  const divider = layout.rootItem!.element.querySelectorAll<HTMLElement>(':scope > .lm_splitter')[1]
  expect(divider).toBeDefined()
  const handle = divider.querySelector<HTMLElement>('.lm_drag_handle')!
  pointer(handle, 'pointerdown', 800)
  pointer(document, 'pointermove', 800 + offset)
  pointer(document, 'pointerup', 800 + offset)
  vi.runOnlyPendingTimers()
}

describe('docked panel resizing', () => {
  it('shrinks the two-tab experiment panel to 240 px, grows the plot, and retains the split after reload', () => {
    const layout = mount(previousLayout)
    const before = widths(layout)
    expect(before).toEqual([400, 400, 400])
    expect(layout.rootItem!.contentItems[1].contentItems).toHaveLength(2)

    dragMiddleRightDivider(layout, -160)
    const after = widths(layout)
    // Golden Layout rounds percentage allocations to whole CSS pixels.
    expect(after[1]).toBeGreaterThanOrEqual(240)
    expect(after[1]).toBeLessThanOrEqual(241)
    expect(after[2]).toBeGreaterThanOrEqual(559)
    expect(after[2]).toBeLessThanOrEqual(560)
    expect(Math.abs(after[0] - before[0])).toBeLessThanOrEqual(1)

    const persisted = JSON.parse(JSON.stringify(layout.saveLayout())) as unknown
    const reloaded = mount(restoreLayoutConfig(persisted, previousLayout))
    const restored = widths(reloaded)
    expectSameSplit(restored, after)
    expect(reloaded.rootItem!.contentItems[1].contentItems).toHaveLength(2)

    dragMiddleRightDivider(reloaded, 120)
    const expanded = widths(reloaded)
    expect(expanded[1] - restored[1]).toBeGreaterThanOrEqual(119)
    expect(expanded[1] - restored[1]).toBeLessThanOrEqual(121)
    expect(restored[2] - expanded[2]).toBeGreaterThanOrEqual(119)
    expect(restored[2] - expanded[2]).toBeLessThanOrEqual(121)
    dragMiddleRightDivider(reloaded, -120)
    expectSameSplit(widths(reloaded), restored)
  })

  it('keeps all three panels usable when the divider is dragged past the minimum width', () => {
    const layout = mount(previousLayout)
    dragMiddleRightDivider(layout, -1000)
    const resized = widths(layout)
    expect(resized[1]).toBeGreaterThanOrEqual(240)
    expect(resized[1]).toBeLessThanOrEqual(241)
    resized.forEach(width => expect(width).toBeGreaterThanOrEqual(240))
    expect(resized[2]).toBeGreaterThan(400)
    expect(resized.reduce((total, width) => total + width, 0)).toBe(1200)
  })
})
