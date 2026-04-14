import type { TandemJunctionLayer } from '../types'

/**
 * Minimal side-by-side tandem stack visualizer.
 *
 * Three columns: top sub-cell | tunnel junction | bottom sub-cell.
 * Each column renders layer names/roles as stacked divs — no Plotly, no
 * complex interaction. Pure DOM, pure HTML strings.
 *
 * Deviation from plan: does NOT wrap the existing mountStackVisualizer because
 * that API requires a full DeviceConfig + ValidationReport and is designed for
 * the interactive device editor. This visualizer is intentionally read-only.
 */

const HTML_ESCAPE: Record<string, string> = {
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}
function esc(s: string): string {
  return s.replace(/[&<>"']/g, c => HTML_ESCAPE[c])
}

function renderLayerPill(layer: TandemJunctionLayer): string {
  const role = esc(layer.role ?? 'unknown')
  const name = esc(layer.name ?? '')
  const thick = layer.thickness_nm != null ? ` <span class="ts-thickness">${layer.thickness_nm} nm</span>` : ''
  return `<div class="ts-layer ts-role-${role}" title="${name}">${name}${thick}</div>`
}

function renderColumn(title: string, layers: TandemJunctionLayer[]): string {
  const pills = layers.map(renderLayerPill).join('\n')
  return `
    <div class="ts-column">
      <div class="ts-column-title">${esc(title)}</div>
      <div class="ts-column-layers">${pills || '<div class="ts-empty">—</div>'}</div>
    </div>`
}

export interface TandemStackVisualizerHandle {
  /** Update the visualizer with fresh layer data from a completed run. */
  update(topLayers: TandemJunctionLayer[], botLayers: TandemJunctionLayer[]): void
  /** Clear to placeholder state (before first run). */
  clear(): void
}

export function mountTandemStackVisualizer(
  container: HTMLElement,
): TandemStackVisualizerHandle {
  container.classList.add('tandem-stack-viz')
  container.innerHTML = '<div class="ts-placeholder">Run a tandem simulation to see the stack.</div>'

  return {
    update(topLayers, botLayers) {
      container.innerHTML = `
        <div class="ts-row">
          ${renderColumn('Top sub-cell', topLayers)}
          <div class="ts-junction-divider" aria-label="recombination junction">
            <span class="ts-junction-label">Tunnel / recombination junction</span>
          </div>
          ${renderColumn('Bottom sub-cell', botLayers)}
        </div>`
    },
    clear() {
      container.innerHTML = '<div class="ts-placeholder">Run a tandem simulation to see the stack.</div>'
    },
  }
}
