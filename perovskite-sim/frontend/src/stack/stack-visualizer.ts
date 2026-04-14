import type { DeviceConfig, StackAction, ValidationReport } from '../types'
import { renderLayerCard } from './stack-layer-card'
import { renderInterfaceStrip } from './stack-interface-strip'
import { reconcileInterfaces } from './reconcile-interfaces'

const HTML_ESCAPE: Record<string, string> = {
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}
function esc(s: string): string {
  return s.replace(/[&<>"']/g, c => HTML_ESCAPE[c])
}

export interface StackVisualizerHandle {
  render(
    config: DeviceConfig,
    selectedIdx: number,
    report: ValidationReport,
  ): void
}

/**
 * Mount the stack visualizer into `container`. Pure render: state lives
 * in the parent (device-pane). All user interactions bubble out through
 * `onAction` so the parent owns the immutable state updates.
 *
 * The returned handle's `render` method is idempotent and always rebuilds
 * the inner HTML — this keeps the implementation small and the DOM in
 * sync with parent state without diffing.
 */
export function mountStackVisualizer(
  container: HTMLElement,
  onAction: (action: StackAction) => void,
): StackVisualizerHandle {
  container.classList.add('stack-visualizer')
  let dragSrcIdx: number | null = null

  function handleClick(ev: MouseEvent): void {
    const target = ev.target as HTMLElement
    const actionEl = target.closest<HTMLElement>('[data-action]')
    if (actionEl) {
      const action = actionEl.dataset.action!
      const idxStr = actionEl.dataset.idx
      const ifaceIdxStr = actionEl.dataset.ifaceIdx
      ev.stopPropagation()
      if (action === 'delete' && idxStr) {
        onAction({ type: 'delete', idx: Number(idxStr) })
        return
      }
      if (action === 'up' && idxStr) {
        const i = Number(idxStr)
        if (i > 0) onAction({ type: 'reorder', from: i, to: i - 1 })
        return
      }
      if (action === 'down' && idxStr) {
        const i = Number(idxStr)
        onAction({ type: 'reorder', from: i, to: i + 1 })
        return
      }
      if (action === 'insert' && idxStr) {
        container.dispatchEvent(
          new CustomEvent('stack-insert-request', { detail: { atIdx: Number(idxStr) } }),
        )
        return
      }
      if (action === 'edit-iface' && ifaceIdxStr) {
        container.dispatchEvent(
          new CustomEvent('stack-edit-iface', { detail: { ifaceIdx: Number(ifaceIdxStr) } }),
        )
        return
      }
    }
    // Click on a layer card body → select.
    const card = target.closest<HTMLElement>('.layer-card')
    if (card?.dataset.idx) {
      onAction({ type: 'select', idx: Number(card.dataset.idx) })
    }
  }

  function handleDragStart(ev: DragEvent): void {
    const card = (ev.target as HTMLElement).closest<HTMLElement>('.layer-card')
    if (!card?.dataset.idx) return
    dragSrcIdx = Number(card.dataset.idx)
    ev.dataTransfer?.setData('text/plain', String(dragSrcIdx))
    if (ev.dataTransfer) ev.dataTransfer.effectAllowed = 'move'
  }

  function handleDragOver(ev: DragEvent): void {
    ev.preventDefault()
    if (ev.dataTransfer) ev.dataTransfer.dropEffect = 'move'
  }

  function handleDrop(ev: DragEvent): void {
    ev.preventDefault()
    const card = (ev.target as HTMLElement).closest<HTMLElement>('.layer-card')
    if (card?.dataset.idx == null || dragSrcIdx == null) return
    const toIdx = Number(card.dataset.idx)
    if (toIdx !== dragSrcIdx) {
      onAction({ type: 'reorder', from: dragSrcIdx, to: toIdx })
    }
    dragSrcIdx = null
  }

  container.addEventListener('click', handleClick)
  container.addEventListener('dragstart', handleDragStart)
  container.addEventListener('dragover', handleDragOver)
  container.addEventListener('drop', handleDrop)

  return {
    render(config, selectedIdx, report) {
      const errorFieldsByLayer = new Map<number, Set<string>>()
      for (const e of report.errors) {
        if (e.layerIdx == null) continue
        const set = errorFieldsByLayer.get(e.layerIdx) ?? new Set<string>()
        if (e.field) set.add(e.field)
        errorFieldsByLayer.set(e.layerIdx, set)
      }

      const layers = config.layers
      const interfaces = config.device.interfaces ?? []

      const parts: string[] = []
      parts.push(
        '<div class="stack-visualizer-sun" aria-hidden="true">',
        '  <div class="stack-visualizer-sun-label">☀ AM1.5G</div>',
        '  <div class="stack-visualizer-sun-rays">↓ ↓ ↓ ↓ ↓</div>',
        '</div>',
        '<div class="stack-visualizer-frame">',
      )

      // Inter-layer "+" gap above the first layer.
      parts.push(
        `<div class="stack-insert-gap"><button class="stack-insert-btn" data-action="insert" data-idx="0" aria-label="Insert layer at top">+</button></div>`,
      )

      for (let i = 0; i < layers.length; i++) {
        const errs = errorFieldsByLayer.get(i) ?? new Set<string>()
        parts.push(renderLayerCard(layers[i], i, i === selectedIdx, errs))
        if (i < layers.length - 1) {
          const pair = (interfaces[i] ?? [0, 0]) as readonly [number, number]
          const isDefault = pair[0] === 0 && pair[1] === 0
          parts.push(renderInterfaceStrip(i, pair, isDefault))
        }
      }

      // Inter-layer "+" gap below the last layer.
      parts.push(
        `<div class="stack-insert-gap"><button class="stack-insert-btn" data-action="insert" data-idx="${layers.length}" aria-label="Insert layer at bottom">+</button></div>`,
      )

      parts.push('</div>')

      // Validation banner (first error only — surface space is tight).
      if (report.errors.length > 0) {
        parts.push(
          `<div class="stack-error-banner" role="alert">${esc(report.errors[0].message)}</div>`,
        )
      }

      // Legend.
      parts.push(
        '<div class="stack-legend">',
        '  <span class="legend-chip legend-substrate">substrate</span>',
        '  <span class="legend-chip legend-front_contact">front contact</span>',
        '  <span class="legend-chip legend-ETL">ETL</span>',
        '  <span class="legend-chip legend-absorber">absorber</span>',
        '  <span class="legend-chip legend-HTL">HTL</span>',
        '  <span class="legend-chip legend-back_contact">back contact</span>',
        '</div>',
      )

      // Stack-level actions.
      parts.push(
        '<div class="stack-actions">',
        '  <button class="btn btn-ghost" data-stack-action="add">＋ Add layer…</button>',
        '  <button class="btn btn-ghost" data-stack-action="save-as">Save as…</button>',
        '  <button class="btn btn-ghost" data-stack-action="download-yaml">↓ YAML</button>',
        '</div>',
      )

      container.innerHTML = parts.join('\n')
      // Suppress unused-import warning — reconcileInterfaces is exported
      // for callers that need it; the visualizer itself doesn't.
      void reconcileInterfaces
    },
  }
}
