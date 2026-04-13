export function setStatus(id: string, msg: string, isError = false): void {
  const el = document.getElementById(id)
  if (!el) return
  el.textContent = msg
  el.className = isError ? 'status error' : 'status'
}

export function metricCard(label: string, value: string): string {
  return `<div class="metric-card"><div class="label">${label}</div><div class="value">${value}</div></div>`
}

export function numField(id: string, label: string, value: number | string, step?: string): string {
  return `
    <label class="form-group">
      <span>${label}</span>
      <input type="number" id="${id}" value="${value}"${step ? ` step="${step}"` : ''}>
    </label>`
}

export function readNum(id: string, fallback: number): number {
  const el = document.getElementById(id) as HTMLInputElement | null
  if (!el) return fallback
  const v = Number(el.value)
  return Number.isFinite(v) ? v : fallback
}
