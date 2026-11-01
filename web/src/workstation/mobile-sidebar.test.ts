import { beforeEach, describe, expect, it } from 'vitest'
import { bindMobileSidebarToggle } from './mobile-sidebar'

let toggle: HTMLButtonElement
let sidebar: HTMLElement

beforeEach(() => {
  document.body.innerHTML = `
    <button type="button" aria-controls="tree"></button>
    <aside id="tree"><button class="tree-node">Device</button></aside>`
  toggle = document.querySelector('button')!
  sidebar = document.querySelector('aside')!
})

describe('bindMobileSidebarToggle', () => {
  it('starts closed with an accessible menu control', () => {
    bindMobileSidebarToggle(toggle, sidebar)

    expect(sidebar.classList.contains('workstation-sidebar-open')).toBe(false)
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(toggle.getAttribute('aria-label')).toBe('Show workspace tree')
    expect(toggle.textContent).toBe('☰')
  })

  it('opens from the toggle and closes on Escape', () => {
    bindMobileSidebarToggle(toggle, sidebar)

    toggle.click()
    expect(sidebar.classList.contains('workstation-sidebar-open')).toBe(true)
    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    expect(toggle.getAttribute('aria-label')).toBe('Hide workspace tree')

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    expect(sidebar.classList.contains('workstation-sidebar-open')).toBe(false)
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
  })

  it('closes after a workspace tree selection', () => {
    bindMobileSidebarToggle(toggle, sidebar)
    toggle.click()

    sidebar.querySelector<HTMLButtonElement>('.tree-node')!.click()

    expect(sidebar.classList.contains('workstation-sidebar-open')).toBe(false)
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
  })
})
