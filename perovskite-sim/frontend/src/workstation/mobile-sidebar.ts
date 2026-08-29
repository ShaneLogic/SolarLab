const OPEN_CLASS = 'workstation-sidebar-open'

function setMobileSidebarOpen(
  toggle: HTMLButtonElement,
  sidebar: HTMLElement,
  open: boolean,
): void {
  sidebar.classList.toggle(OPEN_CLASS, open)
  toggle.setAttribute('aria-expanded', String(open))
  toggle.setAttribute('aria-label', open ? 'Hide workspace tree' : 'Show workspace tree')
  toggle.title = open ? 'Hide workspace tree' : 'Show workspace tree'
  toggle.textContent = open ? '×' : '☰'
}

export function bindMobileSidebarToggle(
  toggle: HTMLButtonElement,
  sidebar: HTMLElement,
): () => void {
  const toggleSidebar = (): void => {
    setMobileSidebarOpen(
      toggle,
      sidebar,
      toggle.getAttribute('aria-expanded') !== 'true',
    )
  }
  const closeAfterSelection = (event: MouseEvent): void => {
    const target = event.target as Element | null
    if (target?.closest('.tree-node')) setMobileSidebarOpen(toggle, sidebar, false)
  }
  const closeOnEscape = (event: KeyboardEvent): void => {
    if (event.key === 'Escape') setMobileSidebarOpen(toggle, sidebar, false)
  }

  setMobileSidebarOpen(toggle, sidebar, false)
  toggle.addEventListener('click', toggleSidebar)
  sidebar.addEventListener('click', closeAfterSelection)
  document.addEventListener('keydown', closeOnEscape)

  return () => {
    toggle.removeEventListener('click', toggleSidebar)
    sidebar.removeEventListener('click', closeAfterSelection)
    document.removeEventListener('keydown', closeOnEscape)
    setMobileSidebarOpen(toggle, sidebar, false)
  }
}
