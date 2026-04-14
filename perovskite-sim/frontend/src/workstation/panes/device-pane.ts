import { mountDevicePanel } from '../../device-panel'
import type { DevicePanel } from '../../device-panel'
import type { SimulationModeName } from '../../types'
import { isLayerBuilderEnabled } from '../tier-gating'

/**
 * Build the Device pane contents into the given container.
 *
 * In full tier (Phase 2b), the pane is a CSS grid with two columns:
 * - left: the stack visualizer (rendered by mountDevicePanel which delegates
 *   to the visualizer when the builder is enabled);
 * - right: the per-layer detail editor.
 *
 * In fast / legacy tiers, the pane keeps the existing single-column
 * accordion editor — no behavior change, no regression risk for benchmark
 * workflows.
 */
export async function mountDevicePane(
  container: HTMLElement,
  tabId: string,
  tier: SimulationModeName = 'full',
): Promise<DevicePanel> {
  container.classList.add('pane', 'pane-device')
  const inner = document.createElement('div')
  inner.className = 'pane-body'
  if (isLayerBuilderEnabled(tier)) {
    inner.classList.add('device-pane-grid')
  }
  container.appendChild(inner)
  return mountDevicePanel(inner, tabId, { tier })
}
