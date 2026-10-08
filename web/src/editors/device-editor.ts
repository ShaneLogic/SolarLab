import { mountQuantityInput } from './quantity-input'
import {
  allowsNull, cellInput, DeviceEditError, exactInputEqual, fieldSchema,
  layerReferences, movedRow, moveLayer, renameConnection, renameLayer, renameMaterial, scalarField, schemaVariants, selectedEntity,
  selections, writeConnection, writeField,
} from './device-edits'
import type { Cell, EditorInput, FieldEdit, FieldSchema, FieldValue, Path, RowMutation, Selection } from './device-edits'
import { changeDefectRows, changeProfileRows, clearInterfaceDefect, defectArray, defectSections, isDefectSelection, isStructuredSelection, nestedValue,
  renameDefect, setDefectSection, writeDefectField, writeDefectRow } from './defect-edits'
import type { DefectSection } from './defect-edits'
import { createRecord, layerParent, recordKind, removeRecord, setLayerDocument } from './inventory-edits'
import type { NewRecordKind } from './inventory-edits'

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
  const minimumLength = metadata.minLength ?? Math.max(0, ...variants.map(item => item.minLength ?? 0))
  const input = boolean || choices.length ? element('select') : element('input')
  input.id = id; input.dataset.role = 'scalar-value'
  if (input instanceof HTMLInputElement) { input.type = 'text'; input.spellcheck = false }
  else {
    input.add(new Option('Choose a value', ''))
    for (const value of choices.length ? [...new Set(choices)] : [true, false]) input.add(new Option(text(value), text(value)))
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
    if (input.value.length < minimumLength || input instanceof HTMLSelectElement && !input.value) return incomplete('Enter or choose a value.')
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
  let lastApplied: EditorInput | undefined
  let selection: Selection = selections(draft, draft.schema_version === 'solarlab.device-preparation.v1' ? 'device' : 'top_cell')[0]
  let disposed = false, parameter = '', defectSection = 'defect', connectionsOpen = false
  let newKind: NewRecordKind = 'layer', newId = '', newTemplate = '', creationOpen = false
  const pending = new Map<string, { edit: Extract<FieldEdit, { kind: 'incomplete' }>; path: Path }>()
  const controls: { dispose(): void }[] = []
  const listeners: (() => void)[] = []
  const renderListeners: (() => void)[] = []
  let rendering = false
  const card = element('section'); card.className = 'card'; card.dataset.editor = 'device'
  const status = element('p'); status.className = 'status'; status.setAttribute('role', 'status'); status.dataset.role = 'draft-status'
  const message = element('p'); message.className = 'status error'; message.setAttribute('role', 'alert'); message.dataset.role = 'edit-error'
  const cell = element('select'); cell.dataset.role = 'cell'; cell.setAttribute('aria-label', 'Device or tandem cell')
  const entity = element('select'); entity.dataset.role = 'entity'; entity.setAttribute('aria-label', 'Layer, material, connection, defect or settings')
  const fields = element('div'), topology = element('div'), retained = element('div'), creation = element('div')
  const applyButton = button('Apply draft to preview', 'apply', apply)
  const discardButton = button('Discard draft', 'discard', discard)
  const actions = element('div'); actions.className = 'actions'; actions.style.flexWrap = 'wrap'; actions.append(applyButton, discardButton)
  card.append(element('h3', 'Device configuration draft'),
    element('p', 'Edit a draft of the supplied input. The preview shows the last applied values and their origins. Applying checks preparation; it does not run a simulation.'),
    cell, entity, status, message, actions, creation, fields, topology, retained)
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
    lastApplied = structuredClone(draft)
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
    card.dataset.previewRelation = lastApplied && exactInputEqual(lastApplied, draft) ? 'matches-last-submission' : 'not-submitted'
    status.textContent = pending.size ? `Finish incomplete edits before applying: ${[...pending.values()].map(item => JSON.stringify(item.path)).join('; ')}`
      : card.dataset.previewRelation === 'matches-last-submission' ? 'Draft matches the last input sent to preview. Read its preparation result or field errors below; execution remains disabled.'
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
      if (item.kind === 'metastable_document' || item.kind === 'metastable_preparation') return new Option(`${item.kind === 'metastable_document' ? 'Metastable inventory' : 'Metastable preparation'} — layer ${item.id}${item.occurrence ? ` (duplicate ${item.occurrence + 1})` : ''}`, JSON.stringify(item))
      const found = selectedEntity(draft, item), value = found.value!
      if (item.kind === 'interface_defect') return new Option(`Interface defect ${item.id} — interface ${item.parent.id}${item.parent.occurrence ? ` (duplicate ${item.parent.occurrence + 1})` : ''}`, JSON.stringify(item))
      if (item.kind === 'bulk_defect' || item.kind === 'metastable_defect') return new Option(`${item.kind === 'bulk_defect' ? 'Bulk' : 'Metastable'} defect ${item.id}${item.occurrence ? ` (duplicate ${item.occurrence + 1})` : ''}${Reflect.get(value, 'name') == null ? '' : `: ${text(Reflect.get(value, 'name'))}`} — layer ${item.parent.id}${item.parent.occurrence ? ` (duplicate ${item.parent.occurrence + 1})` : ''}`, JSON.stringify(item))
      const title = item.kind === 'layer' ? 'Layer' : item.kind === 'material' ? 'Material' : item.kind === 'interface' ? 'Interface' : 'Contact'
      const name = Reflect.get(value, 'name')
      const label = `${title} ${item.id}${name === undefined ? '' : `: ${text(name)}`}${item.occurrence ? ` (duplicate ${item.occurrence + 1})` : ''}`
      return new Option(label, JSON.stringify(item))
    }))
    entity.value = JSON.stringify(selection)
  }
  function fieldControl(name: string, parameterField = false, section?: DefectSection) {
    const schema = section?.schema ?? fieldSchema(selection, parameterField), metadata = schema.properties![name]
    const selected = selectedEntity(draft, selection)
    const owner = section ? nestedValue(selected.value, section.path)
      : parameterField && selected.value ? Reflect.get(selected.value, 'parameters') : selected.value
    const path = [...selected.path, ...(section?.path ?? (parameterField ? ['parameters'] : [])), name]
    const target = structuredClone(selection), key = JSON.stringify(path)
    const saved = pending.get(key)
    const initial = saved ? { kind: 'value' as const, value: saved.edit.text } : supplied(owner, name)
    const container = element('div'); container.dataset.field = name; container.dataset.area = section?.key ?? (parameterField ? 'parameters' : 'basic')
    const control = mountScalar(container, name, metadata, schema.required?.includes(name) ?? false, initial, edit => {
      if (disposed) return
      message.textContent = ''
      if (edit.kind === 'incomplete') pending.set(key, { edit, path })
      else {
        try {
          draft = section && isStructuredSelection(target) ? writeDefectField(draft, target, section.key, name, edit)
            : writeField(draft, target, name, edit, parameterField)
          pending.delete(key)
        }
        catch (error) { showError(error); return }
      }
      renderNavigation(); updateStatus(); refreshReferenceNotices(); refreshParameterPresence()
    })
    controls.push(control)
    // Preserve pre-existing invalid values without normalizing them on mount.
    if (saved) pending.set(key, { edit: saved.edit, path })
    const initialEdit = control.read()
    if (initialEdit.kind === 'incomplete') pending.set(key, { edit: initialEdit, path })
    if (name === 'material' && selection.kind === 'layer') {
      const input = container.querySelector('input')
      if (input) {
        const list = element('datalist'); list.id = `material-choices-${++nextId}`
        for (const material of cellInput(draft, selection.cell).materials ?? []) list.append(new Option(material.name, material.id))
        input.setAttribute('list', list.id); container.append(list)
      }
    }
    if ((selection.kind === 'contact' && name === 'layer') || (selection.kind === 'interface' && (name === 'left' || name === 'right'))) {
      const input = container.querySelector('input'), list = element('datalist'); list.id = `layer-choices-${++nextId}`
      for (const layer of cellInput(draft, selection.cell).layers) list.append(new Option(layer.name, layer.id))
      if (input) { input.setAttribute('list', list.id); container.append(list) }
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
  function refreshParameterPresence() {
    if (selection.kind !== 'settings' && selection.kind !== 'layer' && selection.kind !== 'material') return
    const picker = fields.querySelector<HTMLSelectElement>('[data-role=parameter]')
    if (!picker) return
    const selected = selectedEntity(draft, selection), parameterField = selection.kind !== 'settings'
    const owner = parameterField && selected.value ? Reflect.get(selected.value, 'parameters') : selected.value
    const schema = fieldSchema(selection, parameterField)
    for (const option of picker.options) option.textContent = `${schema.properties![option.value].title ?? option.value} (${option.value})${owner && Object.hasOwn(owner, option.value) ? ' — supplied' : ' — omitted'}`
  }
  function refreshReferenceNotices() {
    const notices = topology.querySelector('[data-role=reference-notices]')
    if (!notices) return
    const device = cellInput(draft, selection.cell)
    const items: HTMLElement[] = []
    const ids = device.layers.map(layer => layer.id)
    for (const collection of ['interfaces', 'contacts'] as const) {
      const records = device[collection] ?? []
      for (const [index, record] of records.entries()) if (records.filter(item => item.id === record.id).length > 1) items.push(element('li', `${collection}[${index}].id: duplicate ${record.id}`))
    }
    for (const [index, layer] of device.layers.entries()) {
      if (ids.filter(id => id === layer.id).length > 1) items.push(element('li', `layers[${index}].id: duplicate ${layer.id}`))
      if (layer.material != null && !(device.materials ?? []).some(material => material.id === layer.material)) items.push(element('li', `layers[${index}].material: ${layer.material} is not a supplied material`))
      const defects = layer.bulk_defects ?? []
      for (const [offset, defect] of defects.entries()) if (defects.filter(item => item.id === defect.id).length > 1) items.push(element('li', `layers[${index}].bulk_defects[${offset}].id: duplicate ${defect.id}`))
      const metastable = layer.metastable_document?.metastable_defects ?? []
      for (const [offset, defect] of metastable.entries()) if (metastable.filter(item => item.id === defect.id).length > 1) items.push(element('li', `layers[${index}].metastable_document.metastable_defects[${offset}].id: duplicate ${defect.id}`))
      for (const [offset, metadata] of (layer.scaps_defect_metadata ?? []).entries()) if (!defects.some(defect => defect.id === metadata.defect_id)) items.push(element('li', `layers[${index}].scaps_defect_metadata[${offset}].defect_id: ${metadata.defect_id} is not a supplied defect`))
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
        if ((selection.kind === 'interface' && collection === 'interfaces' || selection.kind === 'contact' && collection === 'contacts')
          && selectedEntity(draft, selection).index === index) {
          group.append(element('p', 'This connection is selected for editing above.'))
          detail.append(group); return
        }
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
    renderNavigation(); fields.replaceChildren(); retained.replaceChildren(); renderCreation()
    const selected = selectedEntity(draft, selection), device = cellInput(draft, selection.cell)
    if (isStructuredSelection(selection)) {
      renderDefect()
    } else if (selection.kind === 'contact' || selection.kind === 'interface') {
      const schema = fieldSchema(selection, false)
      for (const key of Object.keys(schema.properties ?? {})) if (key !== 'id' && scalarField(schema.properties![key])) fields.append(fieldControl(key))
      identityControl('connection')
      if (selection.kind === 'interface') {
        const item = device.interfaces![selected.index]
        const presence = item.defect === null ? 'explicit null' : Object.hasOwn(item, 'defect') ? 'supplied' : 'omitted'
        fields.append(element('p', `Interface defect: ${presence}. Clearing or removing it is an explicit draft edit.`))
        if (item.defect) fields.append(button('Edit supplied interface defect', 'select-interface-defect', () => {
          if (selection.kind !== 'interface') return
          selection = { cell: selection.cell, kind: 'interface_defect', parent: { id: selection.id, occurrence: selection.occurrence }, id: item.defect!.id }
          defectSection = 'defect'; render()
        }))
        for (const mode of ['null', 'omit'] as const) fields.append(button(mode === 'null' ? 'Clear interface defect (null)' : 'Remove interface defect override', `defect-${mode}`, () => {
          const path = [...selectedEntity(draft, selection).path, 'defect']
          draft = clearInterfaceDefect(draft, selection, mode)
          clearPending(path); render()
        }))
      }
    } else if (selection.kind !== 'settings') {
      fields.append(fieldControl('name'))
      if (selection.kind === 'layer') {
        fields.append(fieldControl('role'), fieldControl('thickness'), fieldControl('material'), fieldControl('parameterization'))
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
        const inventory = element('details'); inventory.className = 'device-settings'; inventory.dataset.section = 'inventory'
        inventory.append(element('summary', 'Defect inventory declarations'),
          element('p', 'Model and schema version remain explicit. Editing a distribution or profile never chooses a new model or version for you.'),
          fieldControl('defect_model'), fieldControl('defect_schema_version'))
        const selectedLayer = structuredClone(selection)
        for (const item of selections(draft, selection.cell)) if (item.kind === 'bulk_defect'
          && item.parent.id === selection.id && item.parent.occurrence === selection.occurrence) {
          inventory.append(button(`Edit bulk defect ${item.id}${item.occurrence ? ` (duplicate ${item.occurrence + 1})` : ''}`, 'select-bulk-defect', () => { selection = item; defectSection = 'defect'; render() }))
        }
        if (!(device.layers[selected.index].bulk_defects?.length)) inventory.append(element('p', `No bulk defects are supplied for ${selectedLayer.id}.`))
        fields.append(inventory)
        fields.append(layerDocuments())
      } else {
        fields.append(element('p', `Material ID: ${selection.id}. Editing its name keeps this reference unchanged.`))
        identityControl('material')
      }
    }
    if (selection.kind === 'layer' || selection.kind === 'material' || selection.kind === 'settings') {
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
    }
    if (selected.value) retained.append(retain('Other supplied fields retained', selected.value,
      selection.kind === 'layer' ? ['id', 'name', 'role', 'thickness', 'material'] : selection.kind === 'material' ? ['id', 'name'] : []))
    retained.append(retain('Device data retained', device, ['layers', 'materials', 'settings']))
    if (draft.schema_version === 'solarlab.tandem-preparation.v1') retained.append(retain('Tandem data retained', draft, ['top_cell', 'bottom_cell']))
    renderConnections(); updateStatus(); rendering = false
  }

  function clearPending(path: Path) {
    for (const [key, item] of pending) if (path.every((part, index) => item.path[index] === part)) pending.delete(key)
  }
  function movePending(path: Path, action: RowMutation, copiedFrom?: Path, copyTo?: Path) {
    const before = [...pending.values()]
    pending.clear()
    for (const item of before) {
      let nextPath = [...item.path]
      if (path.every((part, index) => item.path[index] === part) && typeof item.path[path.length] === 'number') {
        const index = movedRow(item.path[path.length] as number, action)
        if (index === undefined) continue
        nextPath[path.length] = index
      }
      pending.set(JSON.stringify(nextPath), { edit: item.edit, path: nextPath })
    }
    if (copiedFrom && copyTo) for (const item of before) if (copiedFrom.every((part, index) => item.path[index] === part)) {
      const nextPath = [...copyTo, ...item.path.slice(copiedFrom.length)]
      pending.set(JSON.stringify(nextPath), { edit: { ...item.edit }, path: nextPath })
    }
  }
  function changeRows(section: DefectSection, action: RowMutation, value?: unknown, copiedIndex?: number) {
    if (!isStructuredSelection(selection)) return
    const { path } = defectArray(draft, selection, section.key)
    const current = defectSections(draft, selection).find(item => item.key === defectSection)?.row
    draft = changeDefectRows(draft, selection, section.key, action, value)
    movePending(path, action, copiedIndex === undefined ? undefined : [...path, copiedIndex],
      action.kind === 'insert' ? [...path, action.index] : undefined)
    if (current?.arrayKey === section.key) {
      const index = movedRow(current.index, action)
      defectSection = defectSections(draft, selection).find(item => item.row?.arrayKey === section.key && item.row.index === index)?.key ?? section.key
    }
    if (action.kind === 'insert' && copiedIndex === undefined && section.array!.kind === 'kinetics') {
      defectSection = defectSections(draft, selection).find(item => item.row?.arrayKey === section.key && item.row.index === action.index)!.key
    }
    render()
  }
  function rowActions(section: DefectSection, index: number) {
    const actions = element('div'); actions.className = 'actions'; actions.style.flexWrap = 'wrap'
    if (!isStructuredSelection(selection)) return actions
    const { rows } = defectArray(draft, selection, section.key)
    actions.dataset.arrayKey = section.key; actions.dataset.row = String(index)
    const empty = () => section.array!.kind === 'kinetics' ? {} : ''
    actions.append(button('Insert empty row before', 'row-insert', () => changeRows(section, { kind: 'insert', index }, empty())),
      button('Copy this row', 'row-copy', () => {
        if (isStructuredSelection(selection)) changeRows(section, { kind: 'insert', index: index + 1 }, defectArray(draft, selection, section.key).rows[index], index)
      }),
      button('Remove this row', 'row-remove', () => changeRows(section, { kind: 'remove', index })))
    for (const direction of [-1, 1] as const) {
      const move = button(direction < 0 ? 'Move row earlier' : 'Move row later', direction < 0 ? 'row-earlier' : 'row-later',
        () => changeRows(section, { kind: 'move', from: index, to: index + direction }))
      move.disabled = index + direction < 0 || index + direction >= rows.length
      actions.append(move)
    }
    return actions
  }
  function renderArray(section: DefectSection) {
    if (!isStructuredSelection(selection)) return
    const target = structuredClone(selection)
    let array: ReturnType<typeof defectArray>
    try { array = defectArray(draft, target, section.key) }
    catch (error) { fields.append(element('p', String(error))); return }
    const list = element('div'); list.dataset.role = 'defect-array'; list.dataset.arrayKey = section.key
    for (const [index, value] of array.rows.entries()) {
      const row = element('div'); row.className = 'form-group'; row.dataset.row = String(index)
      if (section.array!.kind === 'value') {
        const path = [...array.path, index], key = JSON.stringify(path), saved = pending.get(key)
        const root = element('div'); root.dataset.role = 'array-value'
        const metadata = { ...section.array!.item, title: `${section.label}: row ${index + 1}` }
        const control = mountScalar(root, section.key, metadata, true,
          { kind: 'value', value: saved ? saved.edit.text : value as Extract<FieldValue, { kind: 'value' }>['value'] }, edit => {
            if (disposed) return
            if (edit.kind === 'incomplete') pending.set(key, { edit, path })
            else {
              try { draft = writeDefectRow(draft, target, section.key, index, edit); pending.delete(key) }
              catch (error) { showError(error); return }
            }
            updateStatus()
          })
        controls.push(control)
        const initial = control.read()
        if (initial.kind === 'incomplete') pending.set(key, { edit: initial, path })
        row.append(root)
      } else {
        row.append(element('strong', `Transition ${index + 1}`), button('Edit transition kinetics', 'edit-transition', () => {
          if (!isStructuredSelection(selection)) return
          defectSection = defectSections(draft, selection).find(item => item.row?.arrayKey === section.key && item.row.index === index)!.key
          render()
        }))
        if (value !== null && typeof value === 'object') row.append(retain('Supplied transition fields', value, []))
        else row.append(element('p', `Unsupported row retained: ${text(value)}`))
      }
      row.append(rowActions(section, index)); list.append(row)
    }
    if (!array.rows.length) list.append(element('p', 'Explicit empty array. Related arrays remain unchanged.'))
    list.append(button('Add empty row', 'row-append', () => changeRows(section, { kind: 'insert', index: array.rows.length }, section.array!.kind === 'kinetics' ? {} : '')))
    fields.append(list)
  }
  function removeSelectedRecord() {
    const result = removeRecord(draft, selection)
    if (result.mutation) movePending(result.path, result.mutation)
    else clearPending(result.path)
    draft = result.input; selection = result.selection; connectionsOpen = true; render()
  }
  function renderCreation() {
    const detail = element('details'); detail.className = 'device-settings'; detail.open = creationOpen
    detail.dataset.section = 'record-creation'; detail.append(element('summary', 'Create or remove records'),
      element('p', 'Create an empty declaration or explicitly copy a record in this cell. New IDs and topology are checked when you apply; references are never removed automatically.'))
    listen(detail, 'toggle', () => { creationOpen = detail.open })
    const kinds: [NewRecordKind, string][] = [['layer', 'Layer'], ['material', 'Named material'], ['contact', 'Contact'], ['interface', 'Interface'],
      ['bulk_defect', 'Ordinary bulk defect'], ['multivalent_defect', 'Multivalent bulk defect'], ['metastable_defect', 'Metastable defect'], ['interface_defect', 'Interface defect']]
    const kind = element('select'); kind.dataset.role = 'new-record-kind'; kind.setAttribute('aria-label', 'New record type')
    for (const [value, label] of kinds) kind.add(new Option(label, value))
    kind.value = newKind
    listen(kind, 'change', () => { newKind = kind.value as NewRecordKind; newTemplate = ''; render() })
    const template = element('select'); template.dataset.role = 'record-template'; template.setAttribute('aria-label', 'Start from an empty declaration or a supplied record')
    template.add(new Option('Empty declaration — enter required fields', ''))
    for (const item of selections(draft, selection.cell)) if (recordKind(draft, item) === newKind) {
      const value = JSON.stringify(item), label = [...entity.options].find(option => option.value === value)?.textContent
      template.add(new Option(`Copy ${label ?? ('id' in item ? item.id : '')}`, value))
    }
    if (![...template.options].some(item => item.value === newTemplate)) newTemplate = ''
    template.value = newTemplate; listen(template, 'change', () => { newTemplate = template.value })
    const id = element('input'); id.type = 'text'; id.value = newId; id.dataset.role = 'new-record-id'; id.setAttribute('aria-label', 'ID for the new record')
    listen(id, 'input', () => { newId = id.value })
    detail.append(element('label', 'New record type'), kind, element('label', 'Start from'), template, element('label', 'New record ID'), id,
      button('Create record', 'create-record', () => {
        const result = createRecord(draft, selection, newKind, newId, newTemplate ? JSON.parse(newTemplate) as Selection : undefined)
        if (result.mutation?.kind === 'insert') movePending(result.path, result.mutation, result.copiedFrom, [...result.path, result.mutation.index])
        draft = result.input; selection = result.selection; newId = ''; connectionsOpen = true; defectSection = 'defect'; render()
      }))
    if (selection.kind !== 'settings' && selection.kind !== 'metastable_document' && selection.kind !== 'metastable_preparation') detail.append(button('Remove selected record', 'remove-record', removeSelectedRecord))
    detail.append(element('p', 'Bulk and metastable defects are created on the selected layer. Start a metastable inventory first if it is absent. An interface can hold one defect; select its target interface explicitly.'))
    creation.replaceChildren(detail)
  }
  function layerDocuments() {
    const detail = element('details'); detail.className = 'device-settings'; detail.dataset.section = 'metastable-documents'
    detail.append(element('summary', 'Metastable inventory and preparation'))
    const parent = layerParent(selection)
    if (!parent) return detail
    const layer = selectedEntity(draft, { cell: selection.cell, kind: 'layer', ...parent })
    for (const key of ['metastable_document', 'metastable_preparation'] as const) {
      const title = key === 'metastable_document' ? 'Metastable inventory' : 'Metastable preparation', value = Reflect.get(layer.value!, key)
      const group = element('div'); group.className = 'form-group'; group.dataset.document = key
      group.append(element('p', `${title}: ${value === null ? 'explicit null' : value === undefined ? 'omitted' : 'supplied'}.`))
      if (value && typeof value === 'object') group.append(button(`Edit ${title.toLowerCase()}`, `edit-${key}`, () => { selection = { cell: selection.cell, kind: key, ...parent }; render() }))
      else group.append(button(`Start empty ${title.toLowerCase()}`, `start-${key}`, () => {
        draft = setLayerDocument(draft, selection.cell, parent, key, 'empty'); selection = { cell: selection.cell, kind: key, ...parent }; render()
      }))
      for (const mode of ['null', 'omit'] as const) group.append(button(mode === 'null' ? `Clear ${title.toLowerCase()} (null)` : `Remove ${title.toLowerCase()} override`, `${mode}-${key}`, () => {
        draft = setLayerDocument(draft, selection.cell, parent, key, mode); clearPending([...layer.path, key])
        selection = { cell: selection.cell, kind: 'layer', ...parent }; render()
      }))
      detail.append(group)
    }
    detail.append(element('p', 'Document tags, donor/acceptor configurations and preparation controls are explicit declarations. This editor never prepares a metastable state.'))
    return detail
  }
  function identityControl(kind: 'connection' | 'defect' | 'material') {
    if (selection.kind === 'settings' || selection.kind === 'layer' || selection.kind === 'metastable_document' || selection.kind === 'metastable_preparation') return
    const detail = element('details'); detail.className = 'device-settings'; detail.append(element('summary', `Change ${kind} ID`))
    const input = element('input'); input.type = 'text'; input.value = selection.id; input.dataset.role = 'item-id'; input.setAttribute('aria-label', `New ${kind} ID`)
    const change = button('Change ID', 'rename-item', () => {
      if (pending.size) return
      const result = isDefectSelection(selection) ? renameDefect(draft, selection, input.value)
        : selection.kind === 'material' ? renameMaterial(draft, selection, input.value) : renameConnection(draft, selection, input.value)
      draft = result.input; selection = result.selection; render()
      message.className = 'status'; message.setAttribute('role', 'status'); message.textContent = 'ID changed in this draft. Apply to check references and duplicate IDs.'
    })
    change.dataset.structural = 'true'; detail.append(input, change); fields.append(detail)
  }
  function renderDefect() {
    if (!isStructuredSelection(selection)) return
    const sections = defectSections(draft, selection), found = selectedEntity(draft, selection)
    // Empty declarations keep all required scalar fields incomplete, including
    // those in another section. Nothing is materialized from schema defaults.
    for (const section of sections) {
      if (section.array) continue
      const owner = nestedValue(found.value, section.path)
      if (!owner) continue
      for (const name of section.schema.required ?? []) {
        const metadata = section.schema.properties?.[name]
        if (name === 'id' || !metadata || !scalarField(metadata) || Object.hasOwn(owner, name)) continue
        const path = [...found.path, ...section.path, name], key = JSON.stringify(path)
        if (!pending.has(key)) pending.set(key, { path, edit: { kind: 'incomplete', text: '', message: 'Complete this required declaration.' } })
      }
    }
    if (!sections.some(section => section.key === defectSection)) defectSection = sections[0].key
    const picker = element('select'); picker.dataset.role = 'defect-section'; picker.setAttribute('aria-label', 'Defect section')
    for (const section of sections) picker.add(new Option(section.label, section.key))
    picker.value = defectSection
    listen(picker, 'change', () => { defectSection = picker.value; render() })
    const title = selection.kind === 'metastable_document' ? `Metastable inventory for ${selection.id}`
      : selection.kind === 'metastable_preparation' ? `Metastable preparation for ${selection.id}`
        : `${selection.kind === 'bulk_defect' ? 'Bulk' : selection.kind === 'metastable_defect' ? 'Metastable' : 'Interface'} defect ${selection.id}`
    fields.append(element('h4', title), picker)
    const section = sections.find(item => item.key === defectSection)!
    fields.append(element('p', section.note))
    if (section.array) renderArray(section)
    else if (nestedValue(found.value, section.path)) {
      for (const name of Object.keys(section.schema.properties ?? {})) if (name !== 'id' && scalarField(section.schema.properties![name])) fields.append(fieldControl(name, false, section))
    } else fields.append(element('p', 'This section is not supplied. Its current value is retained for authoritative validation.'))
    if (section.row) {
      const array = sections.find(item => item.key === section.row!.arrayKey)!
      fields.append(rowActions(array, section.row.index))
    }
    if (section.optional) {
      const value = found.value && Reflect.get(found.value, section.optional)
      fields.append(element('p', `Section state: ${value === null ? 'explicit null' : found.value && Object.hasOwn(found.value, section.optional) ? 'supplied' : 'omitted'}.`))
      if (!nestedValue(found.value, section.path)) fields.append(button('Start empty section', 'section-empty', () => {
        if (!isDefectSelection(selection)) return
        draft = setDefectSection(draft, selection, section.key, 'empty'); render()
      }))
      for (const mode of ['null', 'omit'] as const) fields.append(button(mode === 'null' ? 'Clear section (null)' : 'Remove section override', `section-${mode}`, () => {
        if (!isDefectSelection(selection)) return
        const path = [...selectedEntity(draft, selection).path, ...section.path]
        draft = setDefectSection(draft, selection, section.key, mode); clearPending(path); render()
      }))
    }
    const profile = nestedValue(found.value, ['spatial_profile']), knots = profile && Reflect.get(profile, 'knots')
    if (selection.kind === 'bulk_defect' && Array.isArray(knots) && (section.key === 'spatial_profile' || section.key.startsWith('knot:'))) {
      const add = button('Add empty knot', 'add-knot', () => {
        if (!isDefectSelection(selection) || pending.size) return
        draft = changeProfileRows(draft, selection, { kind: 'append', knot: { position_fraction: '', density_multiplier: '' } })
        defectSection = `knot:${knots.length}`; render()
      }); add.dataset.structural = 'true'; fields.append(add)
      if (section.key.startsWith('knot:')) {
        const remove = button('Remove selected knot', 'remove-knot', () => {
          if (!isDefectSelection(selection) || pending.size) return
          draft = changeProfileRows(draft, selection, { kind: 'remove', index: section.path.at(-1) as number })
          clearPending([...found.path, 'spatial_profile', 'knots']); defectSection = 'spatial_profile'; render()
        }); remove.dataset.structural = 'true'; fields.append(remove)
      }
      fields.append(element('p', `${knots.length} supplied knots. Adding a row leaves its two values blank; no interpolation or normalization is performed.`))
    }
    if (isDefectSelection(selection)) identityControl('defect')
    else fields.append(layerDocuments())
    if (selection.kind === 'bulk_defect') {
      const remove = button('Remove selected bulk defect', 'remove-bulk-defect', () => {
        removeSelectedRecord()
      }); fields.append(remove)
      fields.append(element('p', 'Removal keeps all partner metadata and references. Any resulting inconsistency is shown by the preview.'))
    }
  }
  listen(cell, 'change', () => { selection = selections(draft, cell.value as Cell)[0]; message.textContent = ''; render() })
  listen(entity, 'change', () => { selection = JSON.parse(entity.value) as Selection; message.textContent = ''; render() })
  render()
  return {
    read, apply, discard,
    setInput(input) { if (!disposed) { baseline = structuredClone(input); draft = structuredClone(input); lastApplied = undefined; pending.clear(); message.textContent = ''; render() } },
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
