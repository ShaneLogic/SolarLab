import type { FullParameterInput } from '../generated/configuration-inputs'

type Quantity = NonNullable<FullParameterInput['mu_n']>

export type QuantityValue = { kind: 'omit' } | { kind: 'value'; value: Quantity | null }
export type QuantityEdit = QuantityValue | { kind: 'incomplete'; text: string; message: string }

/** Pass the field metadata and required membership from its generated schema. */
export interface QuantityInputOptions {
  name: string
  metadata: { readonly unit: string; readonly nullable?: boolean; readonly title?: string }
  required: boolean
  initial: QuantityValue
  onChange?: (edit: QuantityEdit) => void
}

export interface QuantityInput {
  read(): QuantityEdit
  set(value: QuantityValue): void
  dispose(): void
}

let nextId = 0

function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string) {
  const node = document.createElement(tag)
  if (text !== undefined) node.textContent = text
  return node
}

/** Edit only an override. The caller applies it to its original input and asks
 * the backend to resolve units and physical constraints. Rendering never does. */
export function mountQuantityInput(root: HTMLElement, options: QuantityInputOptions): QuantityInput {
  // Generated quantity fields omit this flag when their schema excludes null.
  const nullable = options.metadata.nullable === true
  const id = `quantity-${++nextId}`
  const field = element('div')
  field.className = 'form-group'; field.dataset.field = options.name
  const title = options.metadata.title || options.name
  const label = element('label', title)
  label.htmlFor = id
  const mode = element('select')
  mode.setAttribute('aria-label', `${title}: value source`)
  mode.dataset.role = 'quantity-source'
  mode.add(new Option('Value', 'value'))
  if (!options.required) mode.add(new Option('Inherit / use default', 'omit'))
  if (nullable) mode.add(new Option('Clear value (null)', 'null'))
  const input = element('input')
  input.type = 'text'; input.id = id; input.spellcheck = false
  input.dataset.role = 'quantity-value'
  const unit = element('span', options.metadata.unit)
  unit.id = `${id}-unit`
  const error = element('span')
  error.id = `${id}-error`; error.className = 'status error'; error.setAttribute('role', 'status')
  input.setAttribute('aria-describedby', `${unit.id} ${error.id}`)
  field.append(label, mode, input, unit, error)
  root.replaceChildren(field)

  let original: QuantityValue = { kind: 'omit' }
  let edited = false
  let disposed = false

  function incomplete(message: string): QuantityEdit {
    return { kind: 'incomplete', text: input.value, message }
  }

  function read(): QuantityEdit {
    if (mode.value === 'omit' && !options.required) return { kind: 'omit' }
    if (mode.value === 'null' && nullable) return { kind: 'value', value: null }
    if (mode.value !== 'value') return incomplete('Select a permitted value source.')
    if (!edited && original.kind === 'value' && original.value === null && !nullable) {
      return incomplete(options.required ? 'This field does not accept null. Enter a value.'
        : 'This field does not accept null. Enter a value or omit its override.')
    }
    if (input.value.trim() === '') return incomplete(options.required
      ? 'Enter the required value.' : 'Enter a value or select inheritance explicitly.')
    const value = !edited && original.kind === 'value' ? original.value : input.value
    if (typeof value === 'number' && !Number.isFinite(value)) return incomplete('Enter a finite value.')
    // Keep unit expressions and decimal strings intact; conversion and physical
    // validation belong to the same Python validator used by the preview API.
    return { kind: 'value', value }
  }

  function show() {
    input.disabled = mode.value !== 'value'
    const result = read()
    error.textContent = result.kind === 'incomplete' ? result.message : ''
    input.setAttribute('aria-invalid', String(result.kind === 'incomplete'))
  }

  function set(value: QuantityValue) {
    if (disposed) return
    original = value.kind === 'omit' ? { kind: 'omit' } : { kind: 'value', value: value.value }
    edited = false
    input.value = value.kind === 'omit' || value.value === null ? ''
      : Object.is(value.value, -0) ? '-0' : String(value.value)
    mode.value = value.kind === 'omit' ? options.required ? 'value' : 'omit'
      : value.value === null && nullable ? 'null' : 'value'
    show()
  }

  function onInput() {
    edited = true
    show(); options.onChange?.(read())
  }

  function onMode() {
    // Switching source is explicit, but leaving and returning to a value does
    // not round it or replace its original number/string representation.
    show(); options.onChange?.(read())
  }

  input.addEventListener('input', onInput)
  mode.addEventListener('change', onMode)
  set(options.initial)
  return {
    read, set,
    dispose() {
      disposed = true
      input.removeEventListener('input', onInput)
      mode.removeEventListener('change', onMode)
      field.remove()
    },
  }
}
