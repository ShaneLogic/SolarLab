import type { Workspace, Experiment, Run } from './types'
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
  return `<span class="tier-badge tier-badge-${tier}">${tier.toUpperCase()}</span>`
}

function experimentLabel(kind: Experiment['kind']): string {
  switch (kind) {
    case 'jv': return 'J–V Sweep'
    case 'impedance': return 'Impedance'
    case 'degradation': return 'Degradation'
    case 'tpv': return 'TPV'
  }
}

function runLabel(r: Run): string {
  const t = new Date(r.timestamp)
  const hh = String(t.getHours()).padStart(2, '0')
  const mm = String(t.getMinutes()).padStart(2, '0')
  return `Run ${hh}:${mm}`
}

function renderRun(deviceId: string, experimentId: string, r: Run, activeRunId: string | null): string {
  const active = r.id === activeRunId ? ' tree-node-active' : ''
  return `
    <div class="tree-node tree-node-run${active}"
         data-device-id="${escapeHtml(deviceId)}"
         data-experiment-id="${escapeHtml(experimentId)}"
         data-run-id="${escapeHtml(r.id)}">
      <span class="tree-icon">▶</span>
      <span class="tree-label">${escapeHtml(runLabel(r))}</span>
    </div>`
}

function renderExperiment(deviceId: string, e: Experiment, ws: Workspace): string {
  const active = e.id === ws.activeExperimentId ? ' tree-node-active' : ''
  const runs = e.runs.map(r => renderRun(deviceId, e.id, r, ws.activeRunId)).join('')
  return `
    <div class="tree-node tree-node-experiment${active}"
         data-device-id="${escapeHtml(deviceId)}"
         data-experiment-id="${escapeHtml(e.id)}">
      <span class="tree-icon">🧪</span>
      <span class="tree-label">${escapeHtml(experimentLabel(e.kind))}</span>
    </div>
    <div class="tree-children">${runs}</div>`
}

export function renderTreeHTML(ws: Workspace): string {
  const deviceNodes = ws.devices
    .map(d => {
      const active = d.id === ws.activeDeviceId ? ' tree-node-active' : ''
      const experiments = d.experiments.map(e => renderExperiment(d.id, e, ws)).join('')
      return `
        <div class="tree-node tree-node-device${active}" data-device-id="${escapeHtml(d.id)}">
          <span class="tree-icon">🔬</span>
          <span class="tree-label">${escapeHtml(d.name)}</span>
          ${tierBadge(d.tier)}
        </div>
        <div class="tree-children">${experiments}</div>`
    })
    .join('')

  return `
    <div class="tree-section">
      <div class="tree-section-header">📁 Devices</div>
      <div class="tree-section-body">${deviceNodes || '<div class="tree-empty">(no devices yet)</div>'}</div>
    </div>
    <div class="tree-section">
      <div class="tree-section-header">📁 Results / Compare</div>
      <div class="tree-section-body"><div class="tree-empty">(no runs yet)</div></div>
    </div>`
}

export interface TreeHandlers {
  onSelectDevice: (deviceId: string) => void
  onSelectExperiment?: (deviceId: string, experimentId: string) => void
  onSelectRun?: (deviceId: string, experimentId: string, runId: string) => void
}

export function attachTreeHandlers(container: HTMLElement, handlers: TreeHandlers): void {
  container.addEventListener('click', (e) => {
    const target = e.target as HTMLElement
    const runNode = target.closest<HTMLElement>('[data-run-id]')
    if (runNode) {
      const deviceId = runNode.dataset.deviceId
      const experimentId = runNode.dataset.experimentId
      const runId = runNode.dataset.runId
      if (deviceId && experimentId && runId) {
        handlers.onSelectRun?.(deviceId, experimentId, runId)
      }
      return
    }
    const expNode = target.closest<HTMLElement>('[data-experiment-id]')
    if (expNode && !expNode.hasAttribute('data-run-id')) {
      const deviceId = expNode.dataset.deviceId
      const experimentId = expNode.dataset.experimentId
      if (deviceId && experimentId) {
        handlers.onSelectExperiment?.(deviceId, experimentId)
      }
      return
    }
    const devNode = target.closest<HTMLElement>('[data-device-id]')
    if (devNode && !devNode.hasAttribute('data-experiment-id')) {
      const id = devNode.dataset.deviceId
      if (id) handlers.onSelectDevice(id)
    }
  })
}
