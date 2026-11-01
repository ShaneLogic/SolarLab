/** Experiment-owned geometry/protocol controls composed with the device editor.
 * Generated metadata describes input; only the server decides its semantics. */
import type { DeviceInput, SpatialExperimentInput } from '../generated/configuration-inputs'
import { configurationSchema } from '../generated/configuration-schema'
import { spatialExperimentInputFromPreview } from '../configuration-client'
import { mountConfigurationPreviewPanel } from '../panels/configuration-preview'
import { mountDeviceEditor, mountScalar } from './device-editor'
import type { FieldEdit, FieldValue, FieldSchema, Path, RowMutation } from './device-edits'
import { allowsNull, exactInputEqual, movedRow, mutateRows } from './device-edits'

export interface SpatialExperimentDraft {
  input: SpatialExperimentInput
  dirty: boolean
  incomplete: Path[]
}
export interface SpatialExperimentEditor {
  read(): SpatialExperimentDraft
  setInput(input: SpatialExperimentInput): void
  apply(): boolean
  discard(): void
  dispose(): void
}
type Schema = FieldSchema & { readonly item_input_unit?: string }
type DraftEdit = Exclude<FieldEdit, { kind: 'value' }> | { kind: 'value'; value: unknown }
const definitions: Readonly<Record<string, Schema>> = configurationSchema.dto_schemas.SpatialExperimentInput.schema.$defs
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
function under(path: Path, prefix: Path) { return prefix.every((key, index) => path[index] === key) }

/** One copied experiment draft; the accepted device editor owns its nested draft.
 * Reading/applying joins the two without recreating either from effective data. */
export function mountSpatialExperimentEditor(root: HTMLElement, options: {
  input: SpatialExperimentInput; onApply(input: SpatialExperimentInput): void
}): SpatialExperimentEditor {
  let baseline = structuredClone(options.input), draft = structuredClone(options.input)
  let disposed = false, section = 'controls', lastApplied: SpatialExperimentInput | undefined
  const pending = new Map<string, { path: Path; edit: Extract<FieldEdit, { kind: 'incomplete' }> }>()
  const controls: { dispose(): void }[] = []
  const card = element('section'); card.className = 'card'; card.dataset.panel = 'spatial-experiment-editor'
  card.style.minWidth = '0'; card.style.overflowWrap = 'anywhere'
  const title = element('h3'), status = element('p'), fields = element('div')
  status.setAttribute('role', 'status'); status.dataset.role = 'experiment-draft-status'
  fields.dataset.role = 'experiment-fields'
  const navigation = element('select'); navigation.setAttribute('aria-label', 'Experiment section')
  navigation.dataset.role = 'experiment-section'
  const deviceDetails = element('details'), deviceHost = element('div'); deviceHost.dataset.role = 'experiment-device'
  deviceDetails.append(element('summary', 'Device layers, materials and models'), deviceHost)
  const applyButton = element('button', 'Apply experiment to preview'), discardButton = element('button', 'Discard experiment changes')
  applyButton.type = discardButton.type = 'button'; applyButton.className = discardButton.className = 'btn btn-ghost'
  applyButton.dataset.action = 'apply-experiment'; discardButton.dataset.action = 'discard-experiment'
  const actions = element('div'); actions.className = 'toolbar'; actions.append(applyButton, discardButton)
  card.append(title, element('p', 'Edit declared controls, then apply to inspect the preparation result. No mesh, initial state or simulation is created.'),
    status, actions, deviceDetails, navigation, fields)
  root.replaceChildren(card)
  const device = mountDeviceEditor(deviceHost, { input: draft.device, onApply: () => { apply() } })

  function read(): SpatialExperimentDraft {
    const child = device.read()
    const input = { ...structuredClone(draft), device: child.input as DeviceInput }
    return { input, dirty: !exactInputEqual(input, baseline),
      incomplete: [...pending.values()].map(item => [...item.path]).concat(child.incomplete.map(path => ['device', ...path])) }
  }
  function updateStatus() {
    if (disposed) return
    const current = read()
    card.dataset.state = current.incomplete.length ? 'incomplete' : current.dirty ? 'draft' : 'unchanged'
    const applied = !!lastApplied && exactInputEqual(current.input, lastApplied)
    card.dataset.previewRelation = applied ? 'matches-last-submission' : 'not-submitted'
    status.textContent = current.incomplete.length ? `Finish incomplete edits: ${current.incomplete.map(path => JSON.stringify(path)).join('; ')}`
      : applied ? 'Draft matches the last submitted input. Read the preparation result or validation errors below; execution remains disabled.'
        : current.dirty ? 'Draft changed. The preview still describes the last submitted input.'
          : 'Supplied input unchanged. Effective values appear in the preview; they are not copied into this input.'
    applyButton.disabled = current.incomplete.length > 0
  }
  function write(path: Path, value: DraftEdit) {
    const key = JSON.stringify(path)
    if (value.kind === 'incomplete') pending.set(key, { path, edit: value })
    else {
      const owner = at(draft, path.slice(0, -1))
      if (!owner || typeof owner !== 'object') throw new Error('Create this declaration before editing its fields.')
      if (value.kind === 'omit') Reflect.deleteProperty(owner, path.at(-1)!)
      else Reflect.set(owner, path.at(-1)!, structuredClone(value.value))
      pending.delete(key)
    }
    updateStatus()
  }
  function forget(path: Path) {
    for (const [key, value] of pending) if (under(value.path, path)) pending.delete(key)
  }
  function scalar(host: HTMLElement, path: Path, metadata: Schema, required: boolean) {
    const value = at(draft, path), key = JSON.stringify(path), saved = pending.get(key)
    const box = element('div'); box.dataset.path = key; box.dataset.field = String(path.at(-1))
    const scalarValue = value === null || ['string', 'number', 'boolean'].includes(typeof value)
    const initial: FieldValue = saved ? { kind: 'value', value: saved.edit.text }
      : scalarValue ? { kind: 'value', value: value as string | number | boolean | null } : { kind: 'omit' }
    const control = mountScalar(box, String(path.at(-1)), metadata, required, initial, edit => { if (!disposed) write(path, edit) })
    controls.push(control)
    if (value !== undefined && !scalarValue) box.append(element('p', 'Supplied non-scalar data is retained until explicitly replaced.'))
    const edit = control.read()
    if (edit.kind === 'incomplete') pending.set(key, { path, edit })
    else if (saved) pending.set(key, saved)
    host.append(box)
  }
  function scalarFields(host: HTMLElement, path: Path, schema: Schema, except: readonly string[] = []) {
    const grid = element('div'); grid.style.display = 'grid'; grid.style.gap = '12px'
    grid.style.gridTemplateColumns = 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))'
    for (const [name, metadata] of Object.entries(schema.properties ?? {})) {
      if (!except.includes(name)) scalar(grid, [...path, name], metadata, !!schema.required?.includes(name))
    }
    host.append(grid)
  }
  function button(label: string, action: string, callback: () => void) {
    const button = element('button', label); button.type = 'button'; button.className = 'btn btn-ghost'; button.dataset.action = action
    button.addEventListener('click', () => { if (!disposed) callback() })
    return button
  }
  function presence(host: HTMLElement, path: Path, metadata: Schema, required: boolean, empty: object) {
    const value = at(draft, path), box = element('div'), mode = element('select')
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
    return value !== undefined && value !== null
  }
  function mutate(path: Path, action: RowMutation, inserted?: unknown) {
    const values = at(draft, path)
    if (!Array.isArray(values)) return
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
    pending.clear(); for (const [key, entry] of next) pending.set(key, entry)
    render()
  }
  function array(host: HTMLElement, path: Path, metadata: Schema, required: boolean,
                 rowSchema?: Schema, rowFields?: (host: HTMLElement, path: Path) => void) {
    const box = element('section'); box.dataset.array = JSON.stringify(path); box.className = 'device-settings'
    host.append(box)
    if (!presence(box, path, metadata, required, [])) return
    const values = at(draft, path)
    if (!Array.isArray(values)) { box.append(element('p', 'Supplied value is retained. Choose an explicit array to replace it.')); return }
    if (metadata.item_input_unit) box.append(element('p', `Bare numbers use ${metadata.item_input_unit}; unit strings remain unchanged. Effective values use ${metadata.item_unit}.`))
    values.forEach((value, index) => {
      const row = element('div'); row.dataset.row = String(index); row.className = 'card'
      row.append(element('h4', `${metadata.title ?? path.at(-1)} — row ${index + 1}`))
      const actions = element('div'); actions.className = 'toolbar'
      const move = (to: number) => mutate(path, { kind: 'move', from: index, to })
      const up = button('Move up', 'move-up', () => move(index - 1)), down = button('Move down', 'move-down', () => move(index + 1))
      up.disabled = index === 0; down.disabled = index === values.length - 1
      actions.append(up, down, button('Copy stored row', 'copy-row', () => mutate(path, { kind: 'insert', index: index + 1 }, value)),
        button('Remove row', 'remove-row', () => mutate(path, { kind: 'remove', index })))
      row.append(actions)
      if (rowFields) rowFields(row, [...path, index])
      else scalar(row, [...path, index], rowSchema ?? { ...metadata.items, unit: metadata.item_unit, title: `Value ${index + 1}` }, true)
      box.append(row)
    })
    box.append(button(rowFields ? 'Add empty row' : 'Add empty value', 'add-row', () => mutate(path, { kind: 'insert', index: values.length }, rowFields ? {} : '')))
  }
  function protocol(host: HTMLElement) {
    const path: Path = ['experiment', 'jv_2d_protocol'], metadata = definitions.JV2DInput.properties!.jv_2d_protocol
    if (!presence(host, path, metadata, false, {})) return
    host.append(element('p', 'These are supplied protocol declarations. Coordinate and schedule arrays are not generated or bound to an executable state. Fixed-voltage sampling currently permits a zero voltage rate only.'))
    const schema = definitions.JV2DProtocolInput
    scalarFields(host, path, schema, ['voltage_values_V', 'x_coordinates_m', 'y_coordinates_m', 'grain_boundaries', 'solver_atol'])
    for (const name of ['voltage_values_V', 'x_coordinates_m', 'y_coordinates_m']) array(host, [...path, name], schema.properties![name], true)
    const tolerance = element('section'); host.append(tolerance)
    if (presence(tolerance, [...path, 'solver_atol'], schema.properties!.solver_atol, true, {})) scalarFields(tolerance, [...path, 'solver_atol'], definitions.JV2DAtolInput)
    array(host, [...path, 'grain_boundaries'], schema.properties!.grain_boundaries, true, undefined,
      (row, rowPath) => scalarFields(row, rowPath, definitions.JV2DGrainProtocolInput))
  }
  function microstructure(host: HTMLElement) {
    const path: Path = ['experiment', 'microstructure']
    if (!presence(host, path, definitions.JV2DInput.properties!.microstructure, false, {})) return
    host.append(element('p', 'The current backend inherits device grain bands for an omitted, null or empty object override. An explicit empty grain array clears them. Geometry bounds and boundary support are checked by preparation.'))
    array(host, [...path, 'grain_boundaries'], definitions.MicrostructureInput.properties!.grain_boundaries, false, undefined,
      (row, rowPath) => {
        scalarFields(row, rowPath, definitions.GrainBoundaryInput, ['layer_ids'])
        array(row, [...rowPath, 'layer_ids'], definitions.GrainBoundaryInput.properties!.layer_ids, true)
      })
  }
  function render() {
    for (const control of controls.splice(0)) control.dispose()
    fields.replaceChildren()
    const jv = draft.experiment.kind === 'jv_2d'
    title.textContent = jv ? 'Two-dimensional J–V preparation' : 'Grain-size sweep preparation'
    navigation.replaceChildren(...(jv ? [['controls', 'Geometry and sweep controls'], ['tolerance', 'Componentwise tolerance policy'],
      ['microstructure', 'Microstructure override'], ['protocol', 'Supplied execution protocol']] : [['controls', 'Grain sizes and sweep controls']])
      .map(([value, label]) => new Option(label, value)))
    if (![...navigation.options].some(option => option.value === section)) section = 'controls'
    navigation.value = section
    if (section === 'protocol') protocol(fields)
    else if (section === 'microstructure') microstructure(fields)
    else if (section === 'tolerance') {
      const path = ['experiment', 'componentwise_atol']
      if (presence(fields, path, definitions.JV2DInput.properties!.componentwise_atol, false, {})) scalarFields(fields, path, definitions.ComponentwiseAtolInput)
      fields.append(element('p', 'Scalar atol and an explicit componentwise policy are alternative declarations. Choosing one does not silently clear the other; conflicts appear in preparation.'))
    } else {
      const schema = definitions[jv ? 'JV2DInput' : 'GrainSweepInput']
      fields.append(element('p', 'Nx counts intervals; the declared node count is Nx + 1. The current caller uses uniform lateral spacing and Ny_per_layer for every electrical layer. The retained device electrical-grid document is not applied by this experiment.'))
      scalarFields(fields, ['experiment'], schema, ['kind', 'microstructure', 'componentwise_atol', 'jv_2d_protocol', 'grain_sizes_nm', 'grain_sizes'])
      if (!jv) for (const name of ['grain_sizes_nm', 'grain_sizes']) array(fields, ['experiment', name], schema.properties![name], false)
    }
    updateStatus()
  }
  function apply() {
    const current = read()
    if (disposed || current.incomplete.length) return false
    lastApplied = structuredClone(current.input); options.onApply(structuredClone(current.input)); updateStatus(); return true
  }
  function discard() {
    if (disposed) return
    draft = structuredClone(baseline); pending.clear(); device.setInput(draft.device); render()
  }
  const navigate = () => { if (!disposed) { section = navigation.value; render() } }
  navigation.addEventListener('change', navigate); applyButton.addEventListener('click', apply); discardButton.addEventListener('click', discard)
  for (const event of ['input', 'change', 'click']) deviceHost.addEventListener(event, updateStatus)
  render()
  return { read, apply, discard,
    setInput(input) { if (!disposed) { baseline = structuredClone(input); draft = structuredClone(input); pending.clear(); lastApplied = undefined; device.setInput(draft.device); render() } },
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

export interface SpatialExperimentPreview extends SpatialExperimentEditor {
  readonly ready: Promise<void>
  select(input: SpatialExperimentInput, endpoint?: string | URL): Promise<void>
  reopen(originalResponseJson: string): Promise<void>
}

/** Public preparation surface, using the accepted client/panel and exact JSON. */
export function mountSpatialExperimentPreview(root: HTMLElement, options: {
  input: SpatialExperimentInput; endpoint: string | URL
}): SpatialExperimentPreview {
  let endpoint = options.endpoint, disposed = false
  const editorHost = element('div'), previewHost = element('div'), reopen = element('details')
  const source = element('textarea'), error = element('p'), open = element('button', 'Reopen supplied input')
  source.setAttribute('aria-label', 'Original experiment preview JSON'); source.rows = 6; source.style.width = '100%'
  open.type = 'button'; open.className = 'btn btn-ghost'; open.dataset.action = 'reopen-experiment'
  error.setAttribute('role', 'alert')
  reopen.append(element('summary', 'Reopen an exported preview'), source, open, error)
  root.replaceChildren(editorHost, previewHost, reopen)
  const panel = mountConfigurationPreviewPanel(previewHost, { endpoint, kind: 'experiment', input: options.input })
  let ready = panel.ready
  const editor = mountSpatialExperimentEditor(editorHost, { input: options.input,
    onApply(input) { ready = panel.select({ endpoint, kind: 'experiment', input }) } })
  async function select(input: SpatialExperimentInput, nextEndpoint = endpoint) {
    if (disposed) return
    endpoint = nextEndpoint; editor.setInput(input)
    ready = panel.select({ endpoint, kind: 'experiment', input }); await ready
  }
  async function reopenInput(json: string) { if (!disposed) await select(spatialExperimentInputFromPreview(json)) }
  const clicked = () => { error.textContent = ''; void reopenInput(source.value).catch(cause => { if (!disposed) error.textContent = String(cause) }) }
  open.addEventListener('click', clicked)
  return { read: editor.read, apply: editor.apply, discard: editor.discard, setInput: input => { void select(input) },
    get ready() { return ready }, select, reopen: reopenInput,
    dispose() { if (!disposed) { disposed = true; open.removeEventListener('click', clicked); editor.dispose(); panel.dispose(); root.replaceChildren() } },
  }
}
