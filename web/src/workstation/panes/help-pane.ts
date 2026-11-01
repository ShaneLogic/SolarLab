import { tutorialHTML } from '../../panels/tutorial'
import { parametersHTML } from '../../panels/parameters'
import { algorithmHTML } from '../../panels/algorithm'

type TabKey = 'tutorial' | 'parameters' | 'algorithm'

interface TabDef {
  key: TabKey
  label: string
  html: () => string
}

const TABS: TabDef[] = [
  { key: 'tutorial', label: 'Tutorial', html: tutorialHTML },
  { key: 'parameters', label: 'Parameters', html: parametersHTML },
  { key: 'algorithm', label: 'Algorithm', html: algorithmHTML },
]

export function mountHelpPane(container: HTMLElement): void {
  container.classList.add('pane', 'pane-help')
  container.innerHTML = `
    <div class="help-tabs" role="tablist">
      ${TABS.map((t, i) => `
        <button class="help-tab${i === 0 ? ' active' : ''}" data-help-tab="${t.key}" role="tab">${t.label}</button>
      `).join('')}
    </div>
    <div class="help-body" id="help-body">${TABS[0].html()}</div>`

  const body = container.querySelector<HTMLDivElement>('#help-body')!
  container.querySelectorAll<HTMLButtonElement>('.help-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.helpTab as TabKey | undefined
      if (!key) return
      const tab = TABS.find(t => t.key === key)
      if (!tab) return
      container.querySelectorAll('.help-tab').forEach(b => b.classList.remove('active'))
      btn.classList.add('active')
      body.innerHTML = tab.html()
    })
  })
}
