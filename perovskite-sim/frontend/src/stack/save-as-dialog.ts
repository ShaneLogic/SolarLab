import type { DeviceConfig } from '../types'
import { checkUserConfigExists, saveUserConfig } from '../api'

const FILENAME_RE = /^[a-zA-Z0-9_-]{1,64}$/
const HTML_ESCAPE: Record<string, string> = {
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}
function esc(s: string): string {
  return s.replace(/[&<>"']/g, c => HTML_ESCAPE[c])
}

export interface SaveAsResult {
  saved: string
}

/**
 * Open the Save-As dialog. Returns the saved name on success, or null
 * if the user cancels. Errors from the backend are surfaced inline.
 */
export function openSaveAsDialog(
  config: DeviceConfig,
): Promise<SaveAsResult | null> {
  return new Promise(resolve => {
    const overlay = document.createElement('div')
    overlay.className = 'modal-overlay'
    overlay.innerHTML = `
      <div class="modal" role="dialog" aria-label="Save device as">
        <div class="modal-header">Save device as user preset</div>
        <div class="modal-body">
          <label class="param">
            <span class="param-label">Filename</span>
            <input type="text" id="save-as-name" class="num-input" placeholder="my_custom_stack" spellcheck="false" autocomplete="off">
          </label>
          <div id="save-as-hint" class="save-as-hint"></div>
        </div>
        <div class="modal-footer">
          <button class="btn btn-ghost" data-cancel>Cancel</button>
          <button class="btn btn-primary" id="save-as-ok" disabled>Save</button>
        </div>
      </div>`
    document.body.appendChild(overlay)

    const input = overlay.querySelector<HTMLInputElement>('#save-as-name')!
    const okBtn = overlay.querySelector<HTMLButtonElement>('#save-as-ok')!
    const hint = overlay.querySelector<HTMLElement>('#save-as-hint')!
    let overwriteAllowed = false
    let probeToken = 0

    function setHint(text: string, kind: 'ok' | 'warn' | 'error'): void {
      hint.textContent = text
      hint.className = `save-as-hint save-as-hint-${kind}`
    }

    async function refresh(): Promise<void> {
      const name = input.value.trim()
      overwriteAllowed = false
      if (!name) {
        setHint('', 'ok')
        okBtn.disabled = true
        okBtn.textContent = 'Save'
        return
      }
      if (!FILENAME_RE.test(name)) {
        setHint('Use letters, digits, hyphen, underscore (max 64).', 'error')
        okBtn.disabled = true
        okBtn.textContent = 'Save'
        return
      }
      const myToken = ++probeToken
      setHint('Checking…', 'ok')
      okBtn.disabled = true
      try {
        const probe = await checkUserConfigExists(name)
        if (myToken !== probeToken) return  // stale
        if (probe.exists && probe.namespace === 'shipped') {
          setHint(`"${name}" is reserved by a shipped preset.`, 'error')
          okBtn.disabled = true
          okBtn.textContent = 'Save'
          return
        }
        if (probe.exists && probe.namespace === 'user') {
          setHint(`"${name}" already exists. Click Overwrite to replace it.`, 'warn')
          okBtn.disabled = false
          okBtn.textContent = 'Overwrite'
          overwriteAllowed = true
          return
        }
        setHint(`"${name}" is available.`, 'ok')
        okBtn.disabled = false
        okBtn.textContent = 'Save'
      } catch (e) {
        setHint(`Probe failed: ${(e as Error).message}`, 'error')
        okBtn.disabled = true
      }
    }

    function close(result: SaveAsResult | null): void {
      overlay.remove()
      resolve(result)
    }

    let debounceId: number | undefined
    input.addEventListener('input', () => {
      window.clearTimeout(debounceId)
      debounceId = window.setTimeout(() => { void refresh() }, 250)
    })

    overlay.addEventListener('click', ev => {
      const target = ev.target as HTMLElement
      if (target.dataset.cancel != null || target === overlay) close(null)
    })

    okBtn.addEventListener('click', async () => {
      const name = input.value.trim()
      if (!FILENAME_RE.test(name)) return
      okBtn.disabled = true
      const result = await saveUserConfig(name, config, overwriteAllowed)
      if (result.ok) {
        close({ saved: result.saved })
      } else {
        setHint(`Save failed: ${esc(result.detail)}`, 'error')
        okBtn.disabled = false
      }
    })

    input.focus()
  })
}
