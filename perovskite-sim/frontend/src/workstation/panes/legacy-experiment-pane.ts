import { mountJVPanel } from '../../panels/jv'
import { mountImpedancePanel } from '../../panels/impedance'
import { mountDegradationPanel } from '../../panels/degradation'

type TabKey = 'jv' | 'is' | 'deg'

interface TabDef {
  key: TabKey
  label: string
  mount: (el: HTMLElement) => Promise<void>
}

const TABS: TabDef[] = [
  { key: 'jv', label: 'J-V Sweep', mount: mountJVPanel },
  { key: 'is', label: 'Impedance', mount: mountImpedancePanel },
  { key: 'deg', label: 'Degradation', mount: mountDegradationPanel },
]

export async function mountLegacyExperimentPane(container: HTMLElement): Promise<void> {
  container.classList.add('pane', 'pane-legacy')
  container.innerHTML = `
    <div class="legacy-tabs" role="tablist">
      ${TABS.map((t, i) => `
        <button class="legacy-tab${i === 0 ? ' active' : ''}" data-legacy-tab="${t.key}" role="tab">${t.label}</button>
      `).join('')}
    </div>
    <div class="legacy-body">
      ${TABS.map(t => `<section class="legacy-section" data-legacy-section="${t.key}" hidden></section>`).join('')}
    </div>`

  const mounted: Record<TabKey, boolean> = { jv: false, is: false, deg: false }

  async function activate(key: TabKey): Promise<void> {
    container.querySelectorAll<HTMLElement>('.legacy-tab').forEach(b => {
      b.classList.toggle('active', b.dataset.legacyTab === key)
    })
    container.querySelectorAll<HTMLElement>('.legacy-section').forEach(s => {
      s.hidden = s.dataset.legacySection !== key
    })
    if (!mounted[key]) {
      mounted[key] = true
      const section = container.querySelector<HTMLElement>(`[data-legacy-section="${key}"]`)!
      const tab = TABS.find(t => t.key === key)!
      try {
        await tab.mount(section)
      } catch (e) {
        mounted[key] = false
        section.innerHTML = `<div class="card error-card">Failed to load: ${(e as Error).message}</div>`
      }
    }
  }

  container.querySelectorAll<HTMLButtonElement>('.legacy-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.legacyTab as TabKey | undefined
      if (key) void activate(key)
    })
  })

  await activate('jv')
}
