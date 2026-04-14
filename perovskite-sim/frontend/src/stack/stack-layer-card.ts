import type { LayerConfig } from '../types'
import { logScaleHeight } from './log-scale-height'

const HTML_ESCAPE: Record<string, string> = {
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}
function esc(s: string): string {
  return s.replace(/[&<>"']/g, c => HTML_ESCAPE[c])
}

function fmtThickness(metres: number): string {
  if (!Number.isFinite(metres) || metres <= 0) return '—'
  if (metres >= 1e-3) return `${(metres * 1000).toPrecision(3)} mm`
  if (metres >= 1e-6) return `${(metres * 1e6).toPrecision(3)} µm`
  return `${(metres * 1e9).toPrecision(3)} nm`
}

/**
 * Render one layer card as an HTML string. The visualizer composes these
 * with the interface strips. Selection / hover / drag affordances are
 * styled via CSS classes and `data-*` attributes; event wiring is done
 * by the visualizer at delegation time.
 */
export function renderLayerCard(
  layer: LayerConfig,
  idx: number,
  selected: boolean,
  errorFields: ReadonlySet<string>,
): string {
  const role = layer.role || 'absorber'
  const heightPx = logScaleHeight(layer.thickness)
  const selectedCls = selected ? ' is-selected' : ''
  const errorCls = errorFields.size > 0 ? ' is-error' : ''
  const opticalPill = layer.optical_material
    ? `<span class="layer-card-pill" title="optical_material: ${esc(layer.optical_material)}">${esc(layer.optical_material)}</span>`
    : ''
  return `
    <div class="layer-card layer-card-${esc(role)}${selectedCls}${errorCls}"
         data-idx="${idx}"
         draggable="true"
         style="min-height:${heightPx}px"
         role="button"
         tabindex="0"
         aria-selected="${selected}"
         aria-label="Layer ${idx + 1}: ${esc(layer.name)} (${esc(role)})">
      <span class="layer-card-handle" aria-hidden="true">⋮⋮</span>
      <div class="layer-card-body">
        <div class="layer-card-name">${esc(layer.name)}</div>
        <div class="layer-card-meta">${fmtThickness(layer.thickness)} · ${esc(role)}${layer.incoherent ? ' · incoherent' : ''}</div>
      </div>
      ${opticalPill}
      <div class="layer-card-controls">
        <button class="layer-card-up" data-action="up" data-idx="${idx}" aria-label="Move up">↑</button>
        <button class="layer-card-down" data-action="down" data-idx="${idx}" aria-label="Move down">↓</button>
        <button class="layer-card-delete" data-action="delete" data-idx="${idx}" aria-label="Delete layer">✕</button>
      </div>
    </div>`
}
