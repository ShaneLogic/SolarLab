export type NumberInputNotation = 'scientific' | 'integer'

const observedDocuments = new WeakSet<Document>()

function observeNumberInputEdits(ownerDocument: Document): void {
  if (observedDocuments.has(ownerDocument)) return
  observedDocuments.add(ownerDocument)
  // Capture before existing editor listeners read the form. Even retyping the
  // displayed rounded value is an explicit edit and must replace the original.
  ownerDocument.addEventListener('input', event => {
    const input = event.target
    if (!(input instanceof HTMLInputElement)) return
    delete input.dataset.numberValue
    delete input.dataset.numberDisplay
  }, true)
}

function finiteNumber(value: unknown): number | null {
  if (typeof value !== 'number' && typeof value !== 'string') return null
  if (typeof value === 'string' && value.trim() === '') return null
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

/** Compact defaults for physical quantities; counts keep their integer notation. */
export function formatNumberInput(value: unknown, notation: NumberInputNotation = 'scientific'): string {
  const number = finiteNumber(value)
  if (number === null) return ''
  if (number === 0) return '0'
  if (notation === 'integer') return String(number)
  const [mantissa, exponent] = number.toExponential(5).split('e')
  return `${mantissa.replace(/\.?0+$/, '')}e${Number(exponent)}`
}

/** Keep the full value separately so rendering a rounded default never changes a simulation. */
export function numberInputAttributes(value: unknown, notation: NumberInputNotation = 'scientific'): string {
  observeNumberInputEdits(document)
  const display = formatNumberInput(value, notation)
  const canonical = finiteNumber(value)
  return `value="${display}" data-number-display="${display}" data-number-value="${canonical ?? ''}"`
}

export function setNumberInputValue(
  input: HTMLInputElement,
  value: number,
  notation: NumberInputNotation = input.step === '1' ? 'integer' : 'scientific',
): void {
  observeNumberInputEdits(input.ownerDocument)
  input.value = formatNumberInput(value, notation)
  input.dataset.numberDisplay = input.value
  input.dataset.numberValue = Number.isFinite(value) ? String(value) : ''
}

export function readNumberInput(input: HTMLInputElement): number {
  const canonical = finiteNumber(input.dataset.numberValue)
  if (canonical !== null && input.value === input.dataset.numberDisplay) return canonical
  return Number(input.value)
}
