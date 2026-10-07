import { mountQuantityInput } from './quantity-input'
import {
  allowsNull, cellInput, DeviceEditError, exactInputEqual, fieldSchema,
  layerReferences, moveLayer, renameLayer, scalarField, schemaVariants, selectedEntity,
  selections, writeConnection, writeField,
} from './device-edits'
import type { Cell, EditorInput, FieldEdit, FieldSchema, FieldValue, Path, Selection } from './device-edits'

export interface DeviceDraft {
  input: EditorInput
  selection: Selection
  dirty: boolean
  incomplete: Path[]
}

export interface DeviceEditorOptions {
  input: EditorInput
  /** A preview request, never an execution or validity decision. */
  onApply(input: EditorInput): void
}

export interface DeviceEditor {
  read(): DeviceDraft
  setInput(input: EditorInput): void
  apply(): boolean
  discard(): void
  dispose(): void
}

let nextId = 0
function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string) {
  const node = document.createElement(tag)
  if (text !== undefined) node.textContent = text
  return node
}
function text(value: unknown): string {
  return Object.is(value, -0) ? '-0' : typeof value === 'bigint' ? value.toString() : String(value)
}
function supplied(owner: object | undefined, field: string): FieldValue {
  return owner && Object.hasOwn(owner, field) ? { kind: 'value', value: Reflect.get(owner, field) } : { kind: 'omit' }
}

/** Non-quantity scalar controls use only the generated primitive/enum metadata.
 * Numeric-only integers require a complete safe token; unit input goes through
 * the quantity control unchanged, including decimal strings and numeric -0. */
function mountScalar(root: HTMLElement, name: string, metadata: FieldSchema, required: boolean,
  initial: FieldValue, onChange: (edit: FieldEdit) => void) {
  if (metadata.unit !== undefined) return mountQuantityInput(root, {
    name, metadata: { title: metadata.title, unit: metadata.unit, nullable: allowsNull(metadata) }, required,
    initial: initial as Parameters<typeof mountQuantityInput>[1]['initial'], onChange,
  })
  const group = element('div'); group.className = 'form-group'
  const id = `device-field-${++nextId}`, label = element('label', metadata.title ?? name)
  label.htmlFor = id
  const mode = element('select'); mode.dataset.role = 'value-source'
  mode.setAttribute('aria-label', `${metadata.title ?? name}: value source`)
  mode.add(new Option('Value', 'value'))
  if (!required) mode.add(new Option('Inherit / omit override', 'omit'))
  if (allowsNull(metadata)) mode.add(new Option('Clear value (null)', 'null'))
  const variants = schemaVariants(metadata).filter(item => item.type !== 'null')
  const boolean = variants.every(item => item.type === 'boolean')
  const choices = variants.flatMap(item => item.enum ?? (Object.hasOwn(item, 'const') ? [item.const!] : []))
  const numeric = variants.every(item => item.type === 'integer' || item.type === 'number')
  const input = boolean || choices.length ? element('select') : element('input')
  input.id = id; input.dataset.role = 'scalar-value'
  if (input instanceof HTMLInputElement) { input.type = 'text'; input.spellcheck = false }
  else {
    input.add(new Option('Choose a value', ''))
    for (const value of boolean ? [true, false] : [...new Set(choices)]) input.add(new Option(text(value), text(value)))
  }
  let edited = false, disposed = false
  const original = initial.kind === 'value' ? initial.value : undefined
  const shown = original === null || original === undefined ? '' : text(original)
  if (input instanceof HTMLSelectElement && shown && ![...input.options].some(option => option.value === shown)) {
    input.add(new Option(`${shown} (supplied value)`, shown))
  }
  input.value = shown
  mode.value = initial.kind === 'omit' ? required ? 'value' : 'omit' : original === null && allowsNull(metadata) ? 'null' : 'value'
  const error = element('span'); error.className = 'status error'; error.id = `${id}-error`
  input.setAttribute('aria-describedby', error.id)
  function read(): FieldEdit {
    if (mode.value === 'omit' && !required) return { kind: 'omit' }
    if (mode.value === 'null' && allowsNull(metadata)) return { kind: 'value', value: null }
    const incomplete = (message: string): FieldEdit => ({ kind: 'incomplete', text: input.value, message })
    if (mode.value !== 'value') return incomplete('Select a permitted value source.')
    if (input.value.length < (metadata.minLength ?? 0)) return incomplete('Enter a value.')
    if (!edited && initial.kind === 'value' && original !== null) return initial
    if (!edited && original === null) return incomplete('Enter a value or choose a permitted source.')
    if (!input.value && (required || numeric || boolean || choices.length || !edited)) return incomplete('Enter or choose a value.')
    if (boolean) return { kind: 'value', value: input.value === 'true' }
    if (numeric) {
      // This branch never receives a unit expression or a blank input.
      if (!/^-?(?:0|[1-9][0-9]*)$/.test(input.value)) return incomplete('Enter a complete whole number.')
      const value = Number(input.value)
      if (!Number.isSafeInteger(value)) return incomplete('This number cannot be represented exactly by the input type.')
      return { kind: 'value', value }
    }
    return { kind: 'value', value: input.value }
  }
  function show() {
    input.disabled = mode.value !== 'value'
    const edit = read()
    error.textContent = edit.kind === 'incomplete' ? edit.message : ''
    input.setAttribute('aria-invalid', String(edit.kind === 'incomplete'))
  }
  function change() { if (!disposed) { edited = true; show(); onChange(read()) } }
  function source() { if (!disposed) { show(); onChange(read()) } }
  const event = input instanceof HTMLInputElement ? 'input' : 'change'
  input.addEventListener(event, change); mode.addEventListener('change', source)
  group.append(label, mode, input, error); root.replaceChildren(group); show()
  return { read, dispose() { disposed = true; input.removeEventListener(event, change); mode.removeEventListener('change', source); group.remove() } }
}

/** A draft editor for existing items. Apply emits a copied draft for preview;
 * Discard restores the caller's input, including omitted and empty containers. */
export function mountDeviceEditor(root: HTMLElement, options: DeviceEditorOptions): DeviceEditor {
  let baseline = structuredClone(options.input), draft = structuredClone(options.input)
  let selection: Selection = selections(draft, draft.schema_version === 'solarlab.device-preparation.v1' ? 'device' : 'top_cell')[0]
  let disposed = false, parameter = '', connectionsOpen = false
  const pending = new Map<string, { edit: Extract<FieldEdit, { kind: 'incomplete' }>; path: Path }>()
  const controls: { dispose(): void }[] = []
  const listeners: (() => void)[] = []
  const renderListeners: (() => void)[] = []
  let rendering = false
  const card = element('section'); card.className = 'card'; card.dataset.editor = 'device'
  const status = element('p'); status.className = 'status'; status.setAttribute('role', 'status'); status.dataset.role = 'draft-status'
  const message = element('p'); message.className = 'status error'; message.setAttribute('role', 'alert'); message.dataset.role = 'edit-error'
  const cell = element('select'); cell.dataset.role = 'cell'; cell.setAttribute('aria-label', 'Device or tandem cell')
  const entity = element('select'); entity.dataset.role = 'entity'; entity.setAttribute('aria-label', 'Layer, named material or settings')
  const fields = element('div'), topology = element('div'), retained = element('div')
  const applyButton = button('Apply draft to preview', 'apply', apply)
  const discardButton = button('Discard draft', 'discard', discard)
  const actions = element('div'); actions.className = 'actions'; actions.style.flexWrap = 'wrap'; actions.append(applyButton, discardButton)
  card.append(element('h3', 'Device layers and materials'),
    element('p', 'Edit a draft of the supplied input. The preview shows the last applied values and their origins. Applying checks preparation; it does not run a simulation.'),
    cell, entity, status, message, actions, fields, topology, retained)
  root.replaceChildren(card)

  function listen(node: HTMLElement, event: string, callback: () => void) {
    const guarded = () => { if (!disposed) callback() }
    node.addEventListener(event, guarded)
    ;(rendering ? renderListeners : listeners).push(() => node.removeEventListener(event, guarded))
  }
  function button(label: string, action: string, callback: () => void) {
    const node = element('button', label); node.type = 'button'; node.className = 'btn btn-ghost'; node.dataset.action = action
    listen(node, 'click', () => { try { callback() } catch (error) { showError(error) } })
    return node
  }
  function showError(error: unknown) {
    message.className = 'status error'; message.setAttribute('role', 'alert')
    message.textContent = error instanceof DeviceEditError
      ? `${error.message} ${error.paths.map(path => JSON.stringify(path)).join('; ')}` : String(error)
  }
  function read(): DeviceDraft {
    return { input: structuredClone(draft), selection: structuredClone(selection),
      dirty: !exactInputEqual(baseline, draft), incomplete: [...pending.values()].map(item => [...item.path]) }
  }
  function apply() {
    if (disposed || pending.size) return false
    options.onApply(structuredClone(draft))
    status.textContent = 'Draft sent to preview. Read the preparation result or field errors below; execution remains disabled.'
    return true
  }
  function discard() {
    if (disposed) return
    draft = structuredClone(baseline); pending.clear(); message.textContent = ''; connectionsOpen = false
    render()
  }
  function updateStatus() {
    card.dataset.state = pending.size ? 'incomplete' : exactInputEqual(baseline, draft) ? 'unchanged' : 'draft'
    status.textContent = pending.size ? `Finish incomplete edits before applying: ${[...pending.values()].map(item => JSON.stringify(item.path)).join('; ')}`
      : card.dataset.state === 'draft' ? 'Draft changed. Apply to inspect the resolved values and field errors, or discard to restore the supplied input.'
        : 'Supplied input unchanged. Apply to inspect its preparation.'
    applyButton.disabled = pending.size > 0
    for (const button of fields.querySelectorAll<HTMLButtonElement>('[data-structural]')) button.disabled = pending.size > 0 || button.dataset.edge === 'true'
  }
  function renderNavigation() {
    const cells: Cell[] = draft.schema_version === 'solarlab.device-preparation.v1' ? ['device'] : ['top_cell', 'bottom_cell']
    if (!cells.includes(selection.cell)) selection = selections(draft, cells[0])[0]
    const available = selections(draft, selection.cell)
    if (!available.some(item => exactInputEqual(item, selection))) selection = available[0]
    cell.replaceChildren(...cells.map(key => new Option(key === 'device' ? 'Device' : key === 'top_cell' ? 'Top cell' : 'Bottom cell', key)))
    cell.value = selection.cell; cell.hidden = cells.length === 1
    entity.replaceChildren(...available.map(item => {
      if (item.kind === 'settings') return new Option('Device settings', JSON.stringify(item))
      const found = selectedEntity(draft, item), value = found.value!
      const label = `${item.kind === 'layer' ? 'Layer' : 'Material'} ${item.id}: ${Reflect.get(value, 'name')}${item.occurrence ? ` (duplicate ${item.occurrence + 1})` : ''}`
      return new Option(label, JSON.stringify(item))
    }))
    entity.value = JSON.stringify(selection)
  }
  function fieldControl(name: string, parameterField = false) {
    const schema = fieldSchema(selection, parameterField), metadata = schema.properties![name]
    const selected = selectedEntity(draft, selection)
    const owner = parameterField && selected.value ? Reflect.get(selected.value, 'parameters') : selected.value
    const path = [...selected.path, ...(parameterField ? ['parameters'] : []), name]
    const target = structuredClone(selection), key = JSON.stringify([target, parameterField, name])
    const saved = pending.get(key)
    const initial = saved ? { kind: 'value' as const, value: saved.edit.text } : supplied(owner, name)
    const container = element('div'); container.dataset.field = name; container.dataset.area = parameterField ? 'parameters' : 'basic'
    const control = mountScalar(container, name, metadata, schema.required?.includes(name) ?? false, initial, edit => {
      if (disposed) return
      message.textContent = ''
      if (edit.kind === 'incomplete') pending.set(key, { edit, path })
      else {
        try { draft = writeField(draft, target, name, edit, parameterField); pending.delete(key) }
        catch (error) { showError(error); return }
      }
      renderNavigation(); updateStatus(); refreshReferenceNotices()
    })
    controls.push(control)
    // Preserve pre-existing invalid values without normalizing them on mount.
    if (saved) pending.set(key, { edit: saved.edit, path })
    if (name === 'material' && selection.kind === 'layer') {
      const input = container.querySelector('input')
      if (input) {
        const list = element('datalist'); list.id = `material-choices-${++nextId}`
        for (const material of cellInput(draft, selection.cell).materials ?? []) list.append(new Option(material.name, material.id))
        input.setAttribute('list', list.id); container.append(list)
      }
    }
    return container
  }
  function retain(label: string, owner: object, omittedKeys: string[]) {
    const detail = element('details'); detail.className = 'device-settings'; detail.append(element('summary', label))
    const list = element('ul')
    for (const [key, value] of Object.entries(owner)) if (!omittedKeys.includes(key)) {
      const state = value === null ? 'explicit null' : Array.isArray(value) ? `${value.length} supplied records`
        : typeof value === 'object' ? 'supplied object' : text(value)
      list.append(element('li', `${key}: ${state}; retained`))
    }
    detail.append(list.childElementCount ? list : element('p', 'No additional fields supplied. Omitted fields remain omitted.'))
    return detail
  }
  function refreshReferenceNotices() {
    const notices = topology.querySelector('[data-role=reference-notices]')
    if (!notices) return
    const device = cellInput(draft, selection.cell)
    const items: HTMLElement[] = []
    const ids = device.layers.map(layer => layer.id)
    for (const [index, layer] of device.layers.entries()) {
      if (ids.filter(id => id === layer.id).length > 1) items.push(element('li', `layers[${index}].id: duplicate ${layer.id}`))
      if (layer.material != null && !(device.materials ?? []).some(material => material.id === layer.material)) items.push(element('li', `layers[${index}].material: ${layer.material} is not a supplied material`))
    }
    for (const ref of layerReferences(device)) if (!ids.includes(ref.value)) items.push(element('li', `${JSON.stringify(ref.path)}: ${ref.value} is not a supplied layer`))
    for (const [index, item] of (device.interfaces ?? []).entries()) {
      if (ids.indexOf(item.left) < 0 || ids[ids.indexOf(item.left) + 1] !== item.right) items.push(element('li', `interfaces[${index}]: ${item.left} → ${item.right} is not a directed adjacent pair in this order`))
    }
    notices.replaceChildren(...items)
  }
  function renderConnections() {
    const device = cellInput(draft, selection.cell), detail = element('details'); detail.open = connectionsOpen
    detail.dataset.section = 'connections'; detail.className = 'device-settings'
    detail.append(element('summary', 'Layer order and supplied connections'))
    const order = element('p', device.layers.map(layer => layer.id).join(' → ')); order.style.overflowWrap = 'anywhere'; order.dataset.role = 'layer-order'
    detail.append(order, element('p', 'Moving a layer retains every supplied interface and contact. Correct endpoints explicitly and apply to see authoritative topology errors.'))
    const notices = element('ul'); notices.dataset.role = 'reference-notices'
    const ids = device.layers.map(layer => layer.id)
    for (const collection of ['interfaces', 'contacts'] as const) {
      const items = device[collection]
      if (!items?.length) { detail.append(element('p', `${collection}: ${items ? 'empty supplied list' : 'omitted'}; retained`)); continue }
      items.forEach((item, index) => {
        const group = element('div'); group.className = 'form-group'; group.dataset.connection = `${collection}:${index}`
        group.append(element('strong', `${collection === 'interfaces' ? 'Interface' : 'Contact'} ${item.id}`))
        if (collection === 'contacts') group.append(element('span', ` (${Reflect.get(item, 'side')})`))
        for (const field of collection === 'interfaces' ? ['left', 'right'] as const : ['layer'] as const) {
          const label = element('label', field === 'left' ? 'Left layer' : field === 'right' ? 'Right layer' : 'Contact layer')
          const input = element('input'); input.type = 'text'; input.value = Reflect.get(item, field); input.dataset.endpoint = field
          input.setAttribute('aria-label', `${item.id}: ${label.textContent}`)
          const list = element('datalist'); list.id = `connection-choices-${++nextId}`
          for (const id of ids) list.append(new Option(id, id))
          input.setAttribute('list', list.id); label.append(input)
          const selectedCell = selection.cell
          listen(input, 'input', () => { draft = writeConnection(draft, selectedCell, collection, index, field, input.value); updateStatus(); refreshReferenceNotices() })
          group.append(label, list)
        }
        group.append(element('p', 'Other supplied connection values are retained.'))
        detail.append(group)
      })
    }
    detail.append(notices); topology.replaceChildren(detail); refreshReferenceNotices()
  }
  function render() {
    for (const remove of renderListeners.splice(0)) remove()
    for (const control of controls.splice(0)) control.dispose()
    rendering = true
    renderNavigation(); fields.replaceChildren(); retained.replaceChildren()
    const selected = selectedEntity(draft, selection), device = cellInput(draft, selection.cell)
    if (selection.kind !== 'settings') {
      fields.append(fieldControl('name'))
      if (selection.kind === 'layer') {
        fields.append(fieldControl('role'), fieldControl('thickness'), fieldControl('material'))
        const identity = element('details'); identity.className = 'device-settings'; identity.append(element('summary', 'Change layer ID or order'))
        const id = element('input'); id.type = 'text'; id.value = selection.id; id.dataset.role = 'layer-id'; id.setAttribute('aria-label', 'New layer ID')
        const rename = button('Change ID and update references', 'rename-layer', () => {
          if (pending.size) return
          const result = renameLayer(draft, selection, id.value); draft = result.input; selection = result.selection
          connectionsOpen = true; render()
          message.className = 'status'; message.setAttribute('role', 'status')
          message.textContent = `Changed references: ${result.updatedPaths.map(path => JSON.stringify(path)).join('; ') || 'none'}`
        }); rename.dataset.structural = 'true'
        identity.append(id, rename)
        for (const direction of [-1, 1] as const) {
          const move = button(direction < 0 ? 'Move layer earlier' : 'Move layer later', direction < 0 ? 'move-earlier' : 'move-later', () => {
            if (pending.size) return
            const result = moveLayer(draft, selection, direction); draft = result.input; selection = result.selection
            connectionsOpen = true; render()
          })
          move.dataset.structural = 'true'; move.dataset.edge = String(selected.index + direction < 0 || selected.index + direction >= device.layers.length)
          identity.append(move)
        }
        fields.append(identity)
      } else fields.append(element('p', `Material ID: ${selection.id}. Editing its name keeps this reference unchanged.`))
    }
    const parameterField = selection.kind !== 'settings', schema = fieldSchema(selection, parameterField)
    const keys = Object.keys(schema.properties ?? {}).filter(key => scalarField(schema.properties![key]))
    const suppliedOwner = parameterField && selected.value ? Reflect.get(selected.value, 'parameters') : selected.value
    const ordered = [...new Set([...Object.keys(suppliedOwner ?? {}).filter(key => keys.includes(key)), ...keys])]
    if (!ordered.includes(parameter)) parameter = ordered[0]
    const label = element('label', parameterField ? 'Parameter override' : 'Device setting'), picker = element('select')
    picker.dataset.role = 'parameter'; picker.setAttribute('aria-label', label.textContent!)
    for (const key of ordered) picker.add(new Option(`${schema.properties![key].title ?? key} (${key})${suppliedOwner && Object.hasOwn(suppliedOwner, key) ? ' — supplied' : ' — omitted'}`, key))
    picker.value = parameter
    const parameterRoot = element('div'); parameterRoot.append(fieldControl(parameter, parameterField))
    listen(picker, 'change', () => { parameter = picker.value; render() })
    fields.append(label, picker, parameterRoot)
    if (selected.value) retained.append(retain('Other supplied fields retained', selected.value,
      selection.kind === 'layer' ? ['id', 'name', 'role', 'thickness', 'material'] : selection.kind === 'material' ? ['id', 'name'] : []))
    retained.append(retain('Device data retained', device, ['layers', 'materials', 'settings']))
    if (draft.schema_version === 'solarlab.tandem-preparation.v1') retained.append(retain('Tandem data retained', draft, ['top_cell', 'bottom_cell']))
    renderConnections(); updateStatus(); rendering = false
  }
  listen(cell, 'change', () => { selection = selections(draft, cell.value as Cell)[0]; message.textContent = ''; render() })
  listen(entity, 'change', () => { selection = JSON.parse(entity.value) as Selection; message.textContent = ''; render() })
  render()
  return {
    read, apply, discard,
    setInput(input) { if (!disposed) { baseline = structuredClone(input); draft = structuredClone(input); pending.clear(); message.textContent = ''; render() } },
    dispose() {
      if (disposed) return
      disposed = true
      for (const control of controls.splice(0)) control.dispose()
      for (const remove of listeners.splice(0)) remove()
      for (const remove of renderListeners.splice(0)) remove()
      pending.clear(); card.remove()
    },
  }
}
