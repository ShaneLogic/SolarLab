import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('../../job-stream', () => ({
  startJob: vi.fn(),
  streamJobEvents: vi.fn(),
}))

vi.mock('../../active-physics', () => ({
  describeActivePhysics: vi.fn(() => 'Active physics: test'),
}))

import type { DeviceConfig } from '../../types'
import { mountJV2DPane } from './jv-2d-pane'


describe('2D J-V microstructure boundary constraint', () => {
  let container: HTMLDivElement

  beforeEach(() => {
    document.body.innerHTML = ''
    container = document.createElement('div')
    document.body.appendChild(container)
  })

  function mount(config: Record<string, unknown>): void {
    mountJV2DPane(container, {
      getActiveDevice: () => ({
        id: 'device-1',
        config: config as unknown as DeviceConfig,
      }),
      onRunComplete: vi.fn(),
    })
  }

  it('keeps periodic available for a laterally uniform device', () => {
    mount({})
    const select = container.querySelector<HTMLSelectElement>('#jv2d-bc')!
    const periodic = select.querySelector<HTMLOptionElement>('option[value="periodic"]')!
    expect(select.value).toBe('periodic')
    expect(periodic.disabled).toBe(false)
  })

  it('selects and locks Neumann when the active config contains a GB', () => {
    mount({ microstructure: { grain_boundaries: [{}] } })
    const select = container.querySelector<HTMLSelectElement>('#jv2d-bc')!
    const periodic = select.querySelector<HTMLOptionElement>('option[value="periodic"]')!
    expect(select.value).toBe('neumann')
    expect(periodic.disabled).toBe(true)
  })

  it('selects and locks Neumann when the inline GB control is enabled', () => {
    mount({})
    const checkbox = container.querySelector<HTMLInputElement>('#jv2d-gb-en')!
    const select = container.querySelector<HTMLSelectElement>('#jv2d-bc')!
    const periodic = select.querySelector<HTMLOptionElement>('option[value="periodic"]')!
    checkbox.checked = true
    checkbox.dispatchEvent(new Event('change'))
    expect(select.value).toBe('neumann')
    expect(periodic.disabled).toBe(true)

    checkbox.checked = false
    checkbox.dispatchEvent(new Event('change'))
    expect(periodic.disabled).toBe(false)
  })
})
