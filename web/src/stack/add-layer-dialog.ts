import type { LayerConfig, LayerRole, LayerTemplate } from '../types'

const ROLES: ReadonlyArray<LayerRole> = [
  'substrate', 'front_contact', 'ETL', 'absorber', 'HTL', 'back_contact',
]

const HTML_ESCAPE: Record<string, string> = {
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}
function esc(s: string): string {
  return s.replace(/[&<>"']/g, c => HTML_ESCAPE[c])
}

function blankLayer(name: string, role: LayerRole): LayerConfig {
  return {
    name, role, thickness: 1e-7, eps_r: 1,
    mu_n: 0, mu_p: 0, ni: 0, N_D: 0, N_A: 0,
    D_ion: 0, P_lim: 0, P0: 0,
    tau_n: 0, tau_p: 0, n1: 0, p1: 0,
    B_rad: 0, C_n: 0, C_p: 0, alpha: 0,
  }
}

function templateToLayer(name: string, t: LayerTemplate): LayerConfig {
  return {
    ...blankLayer(name, t.role),
    optical_material: t.optical_material,
    ...(t.defaults as Partial<LayerConfig>),
    name,
    role: t.role,
  }
}

/**
 * Open the Add Layer dialog. Returns a promise that resolves with the new
 * LayerConfig (or null if the user cancels).
 *
 * The dialog is a single inline overlay — no library, no portal magic.
 */
export function openAddLayerDialog(
  templates: Record<string, LayerTemplate>,
): Promise<LayerConfig | null> {
  return new Promise(resolve => {
    const overlay = document.createElement('div')
    overlay.className = 'modal-overlay'
    overlay.innerHTML = `
      <div class="modal" role="dialog" aria-label="Add Layer">
        <div class="modal-header">Add layer</div>
        <div class="modal-tabs">
          <button class="modal-tab is-active" data-tab="template">Template</button>
          <button class="modal-tab" data-tab="blank">Blank</button>
        </div>
        <div class="modal-body" id="add-layer-body"></div>
        <div class="modal-footer">
          <button class="btn btn-ghost" data-cancel>Cancel</button>
          <button class="btn btn-primary" id="add-layer-ok" disabled>Add</button>
        </div>
      </div>`
    document.body.appendChild(overlay)

    const body = overlay.querySelector<HTMLElement>('#add-layer-body')!
    const okBtn = overlay.querySelector<HTMLButtonElement>('#add-layer-ok')!
    let pending: LayerConfig | null = null

    function close(result: LayerConfig | null): void {
      overlay.remove()
      resolve(result)
    }

    function renderTemplateTab(): void {
      const items = Object.entries(templates)
        .map(([key, t]) => `
          <button class="template-item" data-template-key="${esc(key)}">
            <div class="template-name">${esc(key)}</div>
            <div class="template-role">${esc(t.role)}</div>
            <div class="template-desc">${esc(t.description)}</div>
            <div class="template-source">${esc(t.source)}</div>
          </button>`)
        .join('')
      body.innerHTML = `<div class="template-list">${items || '<em>No templates available</em>'}</div>`
    }

    function renderBlankTab(): void {
      const roleOpts = ROLES.map(r => `<option value="${r}">${r}</option>`).join('')
      body.innerHTML = `
        <label class="param">
          <span class="param-label">Name</span>
          <input type="text" id="blank-name" class="num-input" value="" spellcheck="false">
        </label>
        <label class="param">
          <span class="param-label">Role</span>
          <select id="blank-role" class="num-input">${roleOpts}</select>
        </label>`
      const nameInput = body.querySelector<HTMLInputElement>('#blank-name')!
      const roleSelect = body.querySelector<HTMLSelectElement>('#blank-role')!
      function refresh(): void {
        const name = nameInput.value.trim()
        if (!name) {
          pending = null
          okBtn.disabled = true
          return
        }
        pending = blankLayer(name, roleSelect.value as LayerRole)
        okBtn.disabled = false
      }
      nameInput.addEventListener('input', refresh)
      roleSelect.addEventListener('change', refresh)
    }

    overlay.addEventListener('click', ev => {
      const target = ev.target as HTMLElement
      if (target.dataset.cancel != null || target === overlay) {
        close(null)
        return
      }
      const tabBtn = target.closest<HTMLElement>('[data-tab]')
      if (tabBtn) {
        overlay.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('is-active'))
        tabBtn.classList.add('is-active')
        if (tabBtn.dataset.tab === 'template') renderTemplateTab()
        else renderBlankTab()
        pending = null
        okBtn.disabled = true
        return
      }
      const tplBtn = target.closest<HTMLElement>('[data-template-key]')
      if (tplBtn) {
        const key = tplBtn.dataset.templateKey!
        const tmpl = templates[key]
        if (tmpl) {
          pending = templateToLayer(key, tmpl)
          okBtn.disabled = false
          overlay.querySelectorAll('.template-item').forEach(el => el.classList.remove('is-selected'))
          tplBtn.classList.add('is-selected')
        }
      }
    })

    okBtn.addEventListener('click', () => close(pending))

    renderTemplateTab()
  })
}
