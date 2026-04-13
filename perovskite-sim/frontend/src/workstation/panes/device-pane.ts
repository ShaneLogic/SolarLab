import { mountDevicePanel } from '../../device-panel'
import type { DevicePanel } from '../../device-panel'
import type { SimulationModeName } from '../../types'

/**
 * Build the Device pane contents into the given container.
 *
 * The container is provided by Golden Layout (or any host). We simply
 * delegate to the existing `mountDevicePanel` so Phase 1 inherits all
 * of its behaviour (preset dropdown, per-layer editor, reset button)
 * without a rewrite.
 *
 * Returns the `DevicePanel` handle so the host (Golden Layout wiring)
 * can call `getConfig()` and `onChange()`.
 */
export async function mountDevicePane(
  container: HTMLElement,
  tabId: string,
  tier: SimulationModeName = 'full',
): Promise<DevicePanel> {
  container.classList.add('pane', 'pane-device')
  const inner = document.createElement('div')
  inner.className = 'pane-body'
  container.appendChild(inner)
  return mountDevicePanel(inner, tabId, { tier })
}
