/** One-dimensional J-V input and history declarations. Preparation never runs a sweep. */
import type { DeviceInput, JVExperimentInput } from '../generated/configuration-inputs'
import { configurationSchema } from '../generated/configuration-schema'
import { jvExperimentInputFromPreview } from '../configuration-client'
import { mountConfigurationPreviewPanel } from '../panels/configuration-preview'
import { mountDeviceEditor, mountScalar } from './device-editor'
import { allowsNull, exactInputEqual, movedRow, mutateRows } from './device-edits'
import type { FieldEdit, FieldSchema, FieldValue, Path, RowMutation } from './device-edits'

export interface JVExperimentDraft { input: JVExperimentInput; dirty: boolean; incomplete: Path[] }
export interface JVExperimentEditor {
  read(): JVExperimentDraft
  setInput(input: JVExperimentInput): void
  apply(): boolean
  discard(): void
  dispose(): void
}
type DraftEdit = Exclude<FieldEdit, { kind: 'value' }> | { kind: 'value'; value: unknown }
const schema: FieldSchema = configurationSchema.dto_schemas.JVExperimentInput.schema
const definitions = schema.$defs!
function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string) {
  const node = document.createElement(tag)
  if (text !== undefined) node.textContent = text
  return node
}
function at(value: unknown, path: Path): unknown {
  for (const key of path) {
    if (!value || typeof value !== 'object') return undefined
    value = Reflect.get(value, key)
  }
  return value
}
function object(value: unknown): value is Record<string, unknown> { return value !== null && typeof value === 'object' && !Array.isArray(value) }
function under(path: Path, prefix: Path) { return prefix.every((key, index) => path[index] === key) }
function shown(value: unknown) { return Object.is(value, -0) ? '-0' : String(value) }

/** Device edits and experiment edits are joined only as copied input. Resolved
 * settings, generated histories and defaults are never copied into the draft. */
export function mountJVExperimentEditor(root: HTMLElement, options: {
  input: JVExperimentInput; onApply(input: JVExperimentInput): void
}): JVExperimentEditor {
  let baseline = structuredClone(options.input), draft = structuredClone(options.input)
  let disposed = false, section = 'controls', lastApplied: JVExperimentInput | undefined
  const pending = new Map<string, { path: Path; edit: Extract<FieldEdit, { kind: 'incomplete' }> }>()
  const selectedRows = new Map<string, number>(), controls: { dispose(): void }[] = []
  const card = element('section'); card.className = 'card'; card.dataset.panel = 'jv-experiment-editor'
  card.style.minWidth = '0'; card.style.overflowWrap = 'anywhere'
  const title = element('h3'), status = element('p'), fields = element('div')
  status.setAttribute('role', 'status'); status.dataset.role = 'experiment-draft-status'
  fields.dataset.role = 'experiment-fields'
  const navigation = element('select'); navigation.setAttribute('aria-label', 'J-V experiment section'); navigation.dataset.role = 'experiment-section'
  const deviceDetails = element('details'), deviceHost = element('div'); deviceHost.dataset.role = 'experiment-device'
  deviceDetails.append(element('summary', 'Device layers, materials and models'), deviceHost)
  const applyButton = element('button', 'Apply experiment to preview'), discardButton = element('button', 'Discard experiment changes')
  applyButton.type = discardButton.type = 'button'; applyButton.className = discardButton.className = 'btn btn-ghost'
  applyButton.dataset.action = 'apply-experiment'; discardButton.dataset.action = 'discard-experiment'
  const actions = element('div'); actions.className = 'toolbar'; actions.append(applyButton, discardButton)
  card.append(title, element('p', 'Edit the supplied request and apply it to inspect declared settings and state history. Preparation creates no physical state, J–V curve or executable admission.'),
    status, actions, deviceDetails, navigation, fields)
  root.replaceChildren(card)
  const device = mountDeviceEditor(deviceHost, { input: draft.device, onApply: () => { apply() } })

  function read(): JVExperimentDraft {
    const child = device.read(), input = { ...structuredClone(draft), device: child.input as DeviceInput }
    return { input, dirty: !exactInputEqual(input, baseline),
      incomplete: [...pending.values()].map(item => [...item.path]).concat(child.incomplete.map(path => ['device', ...path])) }
  }
  function updateStatus() {
    if (disposed) return
    const current = read(), applied = !!lastApplied && exactInputEqual(current.input, lastApplied)
    card.dataset.state = current.incomplete.length ? 'incomplete' : current.dirty ? 'draft' : 'unchanged'
    card.dataset.previewRelation = applied ? 'matches-last-submission' : 'not-submitted'
    status.textContent = current.incomplete.length ? `Finish incomplete edits: ${current.incomplete.map(path => JSON.stringify(path)).join('; ')}`
      : applied ? 'Draft matches the last submitted input. Read the preparation result or validation errors below; execution remains disabled.'
        : current.dirty ? 'Draft changed. The preview still describes the last submitted input.'
          : 'Supplied input unchanged. Effective values and field origins appear in the preview.'
    applyButton.disabled = current.incomplete.length > 0
  }
  function write(path: Path, edit: DraftEdit) {
    const key = JSON.stringify(path)
    if (edit.kind === 'incomplete') pending.set(key, { path, edit })
    else {
      const owner = at(draft, path.slice(0, -1))
      if (!owner || typeof owner !== 'object') throw new Error('Create this declaration before editing its fields.')
      if (edit.kind === 'omit') Reflect.deleteProperty(owner, path.at(-1)!)
      else Reflect.set(owner, path.at(-1)!, structuredClone(edit.value))
      pending.delete(key)
    }
    updateStatus()
  }
  function forget(path: Path) { for (const [key, entry] of pending) if (under(entry.path, path)) pending.delete(key) }
  function button(label: string, action: string, callback: () => void) {
    const result = element('button', label); result.type = 'button'; result.className = 'btn btn-ghost'; result.dataset.action = action
    result.addEventListener('click', () => { if (!disposed) callback() }); return result
  }
  function scalar(host: HTMLElement, path: Path, metadata: FieldSchema, required: boolean) {
    const value = at(draft, path), key = JSON.stringify(path), saved = pending.get(key)
    const box = element('div'); box.dataset.path = key; box.dataset.field = String(path.at(-1)); host.append(box)
    if (metadata.type === 'null') {
      box.append(element('p', `${metadata.title ?? path.at(-1)}: ${value === null ? 'explicit null' : value === undefined ? 'required null is missing' : 'supplied value retained; this field permits null only'}`))
      if (value !== null) box.append(button('Set explicit null', 'set-null', () => { write(path, { kind: 'value', value: null }); render() }))
      if (value === undefined) pending.set(key, { path, edit: { kind: 'incomplete', text: '', message: 'Set the required null field.' } })
      return
    }
    const scalarValue = value === null || ['string', 'number', 'boolean'].includes(typeof value)
    const initial: FieldValue = saved ? { kind: 'value', value: saved.edit.text }
      : scalarValue ? { kind: 'value', value: value as string | number | boolean | null } : { kind: 'omit' }
    const control = mountScalar(box, String(path.at(-1)), metadata, required, initial, edit => { if (!disposed) write(path, edit) })
    controls.push(control)
    if (value !== undefined && !scalarValue) box.append(element('p', 'Supplied non-scalar data is retained until explicitly replaced.'))
    const edit = control.read()
    if (edit.kind === 'incomplete') pending.set(key, { path, edit })
    else if (saved) pending.set(key, saved)
  }
  function scalarFields(host: HTMLElement, path: Path, model: FieldSchema, except: readonly string[] = []) {
    const grid = element('div'); grid.style.display = 'grid'; grid.style.gap = '12px'
    grid.style.gridTemplateColumns = 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))'
    for (const [name, metadata] of Object.entries(model.properties ?? {})) {
      if (!except.includes(name)) scalar(grid, [...path, name], metadata, !!model.required?.includes(name))
    }
    host.append(grid)
    const value = at(draft, path)
    if (object(value)) {
      const retained = Object.keys(value).filter(name => !Object.hasOwn(model.properties ?? {}, name))
      if (retained.length) host.append(element('p', `Additional supplied fields retained for validation: ${retained.join(', ')}`))
    }
  }
  function presence(host: HTMLElement, path: Path, metadata: FieldSchema, required: boolean, empty: object) {
    const value = at(draft, path), mode = element('select'), box = element('div')
    box.dataset.path = JSON.stringify(path); mode.dataset.role = 'declaration-source'
    mode.setAttribute('aria-label', `${metadata.title ?? path.at(-1)}: declaration`)
    mode.add(new Option('Explicit declaration', 'value'))
    if (!required) mode.add(new Option('Omit / inherit', 'omit'))
    if (allowsNull(metadata)) mode.add(new Option('Explicit null', 'null'))
    mode.value = value === undefined ? 'omit' : value === null ? 'null' : 'value'
    if (required && value === undefined) {
      mode.add(new Option('Required declaration missing', 'missing')); mode.value = 'missing'
      pending.set(JSON.stringify(path), { path, edit: { kind: 'incomplete', text: '', message: 'Create the required declaration.' } })
    }
    mode.addEventListener('change', () => {
      if (disposed || mode.value === 'missing') return
      forget(path)
      write(path, mode.value === 'omit' ? { kind: 'omit' } : { kind: 'value', value: mode.value === 'null' ? null : at(draft, path) ?? empty })
      render()
    })
    box.append(element('p', metadata.title ?? String(path.at(-1))), mode); host.append(box)
    if (value === undefined || value === null) return false
    if (Array.isArray(empty) ? !Array.isArray(value) : !object(value)) {
      box.append(element('p', 'Supplied value is retained. Replace it explicitly to edit this declaration.'),
        button('Replace with empty declaration', 'replace-declaration', () => { forget(path); write(path, { kind: 'value', value: empty }); render() }))
      return false
    }
    return true
  }
  function mutate(path: Path, action: RowMutation, inserted?: unknown) {
    const values = at(draft, path); if (!Array.isArray(values)) return
    const key = JSON.stringify(path), selected = selectedRows.get(key) ?? 0
    const next = new Map<string, { path: Path; edit: Extract<FieldEdit, { kind: 'incomplete' }> }>()
    for (const entry of pending.values()) {
      if (under(entry.path, path) && typeof entry.path[path.length] === 'number') {
        const moved = movedRow(entry.path[path.length] as number, action)
        if (moved === undefined) continue
        const shifted = [...entry.path]; shifted[path.length] = moved
        next.set(JSON.stringify(shifted), { ...entry, path: shifted })
      } else next.set(JSON.stringify(entry.path), entry)
    }
    mutateRows(values, action, path, inserted)
    selectedRows.set(key, action.kind === 'insert' ? action.index : movedRow(selected, action) ?? Math.min(selected, values.length - 1))
    pending.clear(); for (const [pendingKey, entry] of next) pending.set(pendingKey, entry)
    render()
  }
  function array(host: HTMLElement, path: Path, metadata: FieldSchema, rowSchema?: FieldSchema) {
    const key = JSON.stringify(path), box = element('section'); box.dataset.array = key; box.className = 'device-settings'; host.append(box)
    if (!presence(box, path, metadata, true, [])) return
    const values = at(draft, path) as unknown[]
    if (values.length) {
      const index = Math.max(0, Math.min(selectedRows.get(key) ?? 0, values.length - 1)); selectedRows.set(key, index)
      const selection = element('select'); selection.dataset.role = 'row-selection'; selection.setAttribute('aria-label', `${metadata.title ?? path.at(-1)}: row`)
      values.forEach((value, row) => selection.add(new Option(`${row + 1}: ${object(value) ? shown(value.phase ?? 'Unnamed phase') : shown(value)}`, String(row))))
      selection.value = String(index)
      selection.addEventListener('change', () => { if (!disposed) { selectedRows.set(key, selection.selectedIndex); render() } })
      const row = element('div'); row.className = 'card'; row.dataset.row = String(index)
      const actions = element('div'); actions.className = 'toolbar'
      const up = button('Move earlier', 'move-up', () => mutate(path, { kind: 'move', from: index, to: index - 1 }))
      const down = button('Move later', 'move-down', () => mutate(path, { kind: 'move', from: index, to: index + 1 }))
      up.disabled = index === 0; down.disabled = index === values.length - 1
      actions.append(up, down, button('Copy stored row', 'copy-row', () => mutate(path, { kind: 'insert', index: index + 1 }, values[index])),
        button('Remove row', 'remove-row', () => mutate(path, { kind: 'remove', index })))
      row.append(actions)
      if (rowSchema && object(values[index])) scalarFields(row, [...path, index], rowSchema)
      else if (rowSchema) row.append(element('p', 'Supplied non-object row is retained for validation. Remove it or add an empty phase explicitly.'))
      else scalar(row, [...path, index], { ...metadata.items, unit: metadata.item_unit, title: 'Voltage sample' }, true)
      box.append(selection, row)
    }
    box.append(button(rowSchema ? 'Add empty phase' : 'Add empty voltage sample', 'add-row', () => mutate(path, { kind: 'insert', index: values.length }, rowSchema ? {} : '')),
      element('p', 'Order is preserved exactly. Copy uses the stored row; unfinished text stays with its original row.'))
  }
  function protocol(host: HTMLElement) {
    const path: Path = ['experiment', 'experiment_protocol'], model = definitions.JVProtocolInput
    if (!presence(host, path, definitions.JVInput.properties!.experiment_protocol, false, {})) return
    host.append(element('p', 'Supplied history must agree with the request. Holds, forward/reverse order and sampling are checked by preparation; edits never reset or create a physical state.'))
    if (section === 'history') array(host, [...path, 'illumination_history'], model.properties!.illumination_history, definitions.JVIlluminationStepInput)
    else if (section === 'sampling') {
      if (presence(host, [...path, 'sampling'], model.properties!.sampling, true, {})) {
        scalarFields(host, [...path, 'sampling'], definitions.JVSamplingInput, ['values'])
        array(host, [...path, 'sampling', 'values'], definitions.JVSamplingInput.properties!.values)
      }
    } else {
      scalarFields(host, path, model, ['illumination_history', 'scan', 'sampling', 'dc_settle'])
      for (const [name, definition] of [['scan', 'JVScanInput'], ['dc_settle', 'JVDCSettleInput']]) {
        if (presence(host, [...path, name], model.properties![name], true, {})) scalarFields(host, [...path, name], definitions[definition])
      }
    }
  }
  function render() {
    for (const control of controls.splice(0)) control.dispose()
    fields.replaceChildren()
    const jv = draft.experiment.kind === 'jv'
    title.textContent = jv ? 'One-dimensional J–V preparation' : 'Dark J–V preparation'
    const sections = jv ? [['controls', 'Request and driver'], ['waveform', 'Continuous waveform and holds'],
      ['tolerance', 'Waveform numerical controls'], ['protocol', 'Supplied protocol and scan'], ['history', 'Supplied illumination order'], ['sampling', 'Supplied voltage samples']]
      : [['controls', 'Dark sweep request']]
    navigation.replaceChildren(...sections.map(([value, label]) => new Option(label, value)))
    if (!sections.some(([value]) => value === section)) section = 'controls'
    navigation.value = section
    if (section === 'controls') {
      scalar(fields, ['id'], schema.properties!.id, true)
      fields.append(element('p', jv
        ? 'Choose the request API whose inherited controls you intend to use. Omitted or null maximum voltage awaits operating contact potential. A steady driver remains a separate declaration and does not become a forward/reverse state history.'
        : 'The existing dark branch uses a transient sweep and exposes a forward curve with a diode fit. It accepts no waveform or illumination override.'))
      scalarFields(fields, ['experiment'], definitions[jv ? 'JVInput' : 'DarkJVInput'], ['kind', 'waveform', 'waveform_controls', 'experiment_protocol'])
    } else if (section === 'waveform' || section === 'tolerance') {
      const waveform = section === 'waveform', name = waveform ? 'waveform' : 'waveform_controls'
      fields.append(element('p', waveform
        ? 'Declare dark seed and prebias holds, forward ramp, turnaround and reverse ramp. Generation null uses device optics; explicit zero remains zero. All required waveform values are supplied explicitly.'
        : 'These tolerances belong to the continuous waveform. Omission or null inherits the source-bound waveform controls; archived numerical settings must be supplied explicitly.'))
      if (presence(fields, ['experiment', name], definitions.JVInput.properties![name], false, {})) {
        scalarFields(fields, ['experiment', name], definitions[waveform ? 'JVWaveformInput' : 'JVWaveformControlsInput'])
      }
    } else protocol(fields)
    updateStatus()
  }
  function apply() {
    const current = read()
    if (disposed || current.incomplete.length) return false
    lastApplied = structuredClone(current.input); options.onApply(structuredClone(current.input)); updateStatus(); return true
  }
  function discard() { if (!disposed) { draft = structuredClone(baseline); pending.clear(); device.setInput(draft.device); render() } }
  const navigate = () => { if (!disposed) { section = navigation.value; render() } }
  navigation.addEventListener('change', navigate); applyButton.addEventListener('click', apply); discardButton.addEventListener('click', discard)
  for (const event of ['input', 'change', 'click']) deviceHost.addEventListener(event, updateStatus)
  render()
  return { read, apply, discard,
    setInput(input) { if (!disposed) { baseline = structuredClone(input); draft = structuredClone(input); pending.clear(); selectedRows.clear(); lastApplied = undefined; device.setInput(draft.device); render() } },
    dispose() {
      if (disposed) return
      disposed = true
      for (const control of controls.splice(0)) control.dispose()
      navigation.removeEventListener('change', navigate); applyButton.removeEventListener('click', apply); discardButton.removeEventListener('click', discard)
      for (const event of ['input', 'change', 'click']) deviceHost.removeEventListener(event, updateStatus)
      device.dispose(); pending.clear(); card.remove()
    },
  }
}

export interface JVExperimentPreview extends JVExperimentEditor {
  readonly ready: Promise<void>
  select(input: JVExperimentInput, endpoint?: string | URL): Promise<void>
  reopen(originalResponseJson: string): Promise<void>
}
export function mountJVExperimentPreview(root: HTMLElement, options: { input: JVExperimentInput; endpoint: string | URL }): JVExperimentPreview {
  let endpoint = options.endpoint, disposed = false
  const editorHost = element('div'), previewHost = element('div'), reopen = element('details')
  const source = element('textarea'), error = element('p'), open = element('button', 'Reopen supplied input')
  source.setAttribute('aria-label', 'Original experiment preview JSON'); source.rows = 6; source.style.width = '100%'
  open.type = 'button'; open.className = 'btn btn-ghost'; open.dataset.action = 'reopen-experiment'; error.setAttribute('role', 'alert')
  reopen.append(element('summary', 'Reopen an exported preview'), source, open, error)
  root.replaceChildren(editorHost, previewHost, reopen)
  const panel = mountConfigurationPreviewPanel(previewHost, { endpoint, kind: 'experiment', input: options.input })
  let ready = panel.ready
  const editor = mountJVExperimentEditor(editorHost, { input: options.input, onApply(input) { ready = panel.select({ endpoint, kind: 'experiment', input }) } })
  async function select(input: JVExperimentInput, nextEndpoint = endpoint) {
    if (disposed) return
    endpoint = nextEndpoint; editor.setInput(input); ready = panel.select({ endpoint, kind: 'experiment', input }); await ready
  }
  async function reopenInput(json: string) { if (!disposed) await select(jvExperimentInputFromPreview(json)) }
  const clicked = () => { error.textContent = ''; void reopenInput(source.value).catch(cause => { if (!disposed) error.textContent = String(cause) }) }
  open.addEventListener('click', clicked)
  return { read: editor.read, apply: editor.apply, discard: editor.discard, setInput: input => { void select(input) },
    get ready() { return ready }, select, reopen: reopenInput,
    dispose() { if (!disposed) { disposed = true; open.removeEventListener('click', clicked); editor.dispose(); panel.dispose(); root.replaceChildren() } },
  }
}
