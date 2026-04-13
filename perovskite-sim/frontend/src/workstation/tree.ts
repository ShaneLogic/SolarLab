import type { Workspace } from './types'
import type { SimulationModeName } from '../types'

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function tierBadge(tier: SimulationModeName): string {
  const label = tier.toUpperCase()
  return `<span class="tier-badge tier-badge-${tier}">${label}</span>`
}

export function renderTreeHTML(ws: Workspace): string {
  const deviceNodes = ws.devices
    .map(d => {
      const active = d.id === ws.activeDeviceId ? ' tree-node-active' : ''
      return `
        <div class="tree-node tree-node-device${active}" data-device-id="${escapeHtml(d.id)}">
          <span class="tree-icon">🔬</span>
          <span class="tree-label">${escapeHtml(d.name)}</span>
          ${tierBadge(d.tier)}
        </div>`
    })
    .join('')

  return `
    <div class="tree-section">
      <div class="tree-section-header">📁 Devices</div>
      <div class="tree-section-body">${deviceNodes || '<div class="tree-empty">(no devices yet)</div>'}</div>
    </div>
    <div class="tree-section">
      <div class="tree-section-header">📁 Results / Compare</div>
      <div class="tree-section-body"><div class="tree-empty">(Phase 4)</div></div>
    </div>`
}

export interface TreeHandlers {
  onSelectDevice: (deviceId: string) => void
}

export function attachTreeHandlers(container: HTMLElement, handlers: TreeHandlers): void {
  container.addEventListener('click', (e) => {
    const target = e.target as HTMLElement
    const node = target.closest<HTMLElement>('[data-device-id]')
    if (node) {
      const id = node.dataset.deviceId
      if (id) handlers.onSelectDevice(id)
    }
  })
}
