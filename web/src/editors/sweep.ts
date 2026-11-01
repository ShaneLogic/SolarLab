/** Explicit serial sweep declarations composed with the existing input editors. */
import type { DeviceInput, JVExperimentInput, SweepInput, SweepInputDefinitions } from '../generated/configuration-inputs'
import { configurationSchema } from '../generated/configuration-schema'
import { sweepInputFromPreview, sweepPointInputFromPreview } from '../configuration-client'
import { mountConfigurationPreviewPanel } from '../panels/configuration-preview'
import type { ConfigurationPreviewPanel } from '../panels/configuration-preview'
import { mountDeviceEditor, mountScalar } from './device-editor'
import { mountJVExperimentEditor } from './jv-experiment'
import { exactInputEqual, movedRow, mutateRows, schemaVariants } from './device-edits'
import type { FieldEdit, FieldSchema, FieldValue, Path, RowMutation } from './device-edits'

type Target = SweepInputDefinitions.SweepTargetInput
type Base = DeviceInput | JVExperimentInput
type TargetChoice = { target: Target; label: string; metadata: FieldSchema; required: boolean }
const definitions: Readonly<Record<string, FieldSchema>> = configurationSchema.dto_schemas.SweepInput.schema.$defs
const sweepSchema: FieldSchema = configurationSchema.dto_schemas.SweepInput.schema
function node<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string) {
  const value = document.createElement(tag); if (text !== undefined) value.textContent = text; return value
}
function display(value: unknown): string { return Object.is(value, -0) ? '-0' : value === null ? 'null' : String(value) }
function object(value: unknown): value is Record<string, unknown> { return value !== null && typeof value === 'object' && !Array.isArray(value) }
function at(value: unknown, path: Path): unknown { for (const key of path) { if (!value || typeof value !== 'object') return undefined; value = Reflect.get(value, key) } return value }
function under(path: Path, prefix: Path) { return prefix.every((key, index) => path[index] === key) }

/** Field lists and units come from generated DTOs. Array indexes are only UI
 * positions; the transmitted target uses the supplied stable instance IDs. */
export function sweepTargets(base: Base): TargetChoice[] {
  const device = base.schema_version === 'solarlab.device-preparation.v1' ? base : base.device, choices: TargetChoice[] = []
  function fields(model: FieldSchema, target: Omit<Target, 'parameter'>, label: string, prefix: string[] = []) {
    for (const [name, metadata] of Object.entries(model.properties ?? {})) {
      if (['id', 'kind', 'schema_version', 'request_api'].includes(name)
        || schemaVariants(metadata).some(value => value.$ref || ['array', 'object'].includes(value.type ?? ''))) continue
      choices.push({ target: { ...target, parameter: [...prefix, name] as Target['parameter'] },
        label: `${label} / ${[...prefix, name].join('.')}`, metadata, required: !!model.required?.includes(name) })
    }
  }
  fields(definitions.DeviceSettingsInput, { family: 'setting' }, 'Device settings')
  if (base.schema_version === 'solarlab.experiment-preparation.v1') fields(definitions[base.experiment.kind === 'jv' ? 'JVInput' : 'DarkJVInput'], { family: 'jv' }, 'J–V controls')
  for (const layer of device.layers) {
    const duplicates = device.layers.filter(item => item.id === layer.id).length > 1 ? ' (duplicate ID)' : ''
    fields(definitions.FullParameterInput, { family: 'layer_parameter', owner_id: layer.id }, `${layer.name} [${layer.id}]${duplicates}`)
    for (const defect of layer.bulk_defects ?? []) {
      if ('distribution' in defect) fields(definitions.DistributionInput, { family: 'bulk_defect', owner_id: layer.id, local_id: defect.id }, `${layer.id} / ${defect.name ?? defect.id} [${defect.id}]`, ['distribution'])
      if ('energy_level' in defect && defect.energy_level) fields(definitions.EnergyLevelInput, { family: 'bulk_defect', owner_id: layer.id, local_id: defect.id }, `${layer.id} / ${defect.name ?? defect.id} [${defect.id}]`, ['energy_level'])
    }
  }
  for (const material of device.materials ?? []) fields(definitions.FullParameterInput, { family: 'material_parameter', owner_id: material.id }, `Material ${material.name ?? material.id} [${material.id}]`)
  for (const item of device.interfaces ?? []) if (item.defect) fields(definitions.InterfaceDefectInput,
    { family: 'interface_defect', owner_id: item.id, local_id: item.defect.id }, `${item.left} → ${item.right} [${item.id}/${item.defect.id}]`)
  for (const etl of device.layers.filter(item => item.role === 'ETL')) for (const pvk of device.layers.filter(item => item.role === 'absorber')) {
    choices.push({ target: { family: 'cbo', owner_id: etl.id, reference_id: pvk.id, parameter: ['chi'] },
      label: `CBO: ${pvk.id} affinity − ${etl.id} affinity`, metadata: { ...definitions.FullParameterInput.properties!.chi, title: 'Conduction-band offset' }, required: true })
  }
  return choices
}

export interface SweepDraft { input: SweepInput; dirty: boolean; incomplete: Path[] }
export interface SweepEditor { read(): SweepDraft; setInput(input: SweepInput): void; apply(): boolean; discard(): void; dispose(): void }

export function mountSweepEditor(root: HTMLElement, options: { input: SweepInput; onApply(input: SweepInput): void }): SweepEditor {
  let baseline = structuredClone(options.input), draft = structuredClone(options.input), disposed = false
  let axisIndex = 0, coordinateIndex = 0, lastApplied: SweepInput | undefined
  const pending = new Map<string, { path: Path; edit: Extract<FieldEdit, { kind: 'incomplete' }> }>(), controls: { dispose(): void }[] = []
  const card = node('section'); card.className = 'card'; card.dataset.panel = 'sweep-editor'; card.style.minWidth = '0'; card.style.overflowWrap = 'anywhere'
  const status = node('p'), fields = node('div'), navigation = node('select'), actions = node('div'), baseDetails = node('details'), baseHost = node('div')
  status.setAttribute('role', 'status'); status.dataset.role = 'sweep-draft-status'; actions.className = 'toolbar'
  navigation.setAttribute('aria-label', 'Sweep axis'); navigation.dataset.role = 'sweep-axis'
  baseDetails.append(node('summary', 'Base device and J–V input'), baseHost)
  const applyButton = button('Apply sweep to preview', 'apply-sweep', () => { apply() }), discardButton = button('Discard sweep changes', 'discard-sweep', discard)
  actions.append(applyButton, discardButton)
  card.append(node('h3', 'Serial sweep preparation'), node('p', 'Declare one or two axes and inspect each prepared point. No numerical scan, point reuse or execution qualification is performed.'), status, actions, baseDetails, navigation, fields)
  root.replaceChildren(card)
  let child: { read(): { input: Base; incomplete: Path[] }; dispose(): void }
  function mountBase() {
    child?.dispose()
    const editor = draft.base.schema_version === 'solarlab.device-preparation.v1'
      ? mountDeviceEditor(baseHost, { input: draft.base, onApply: () => { apply() } })
      : mountJVExperimentEditor(baseHost, { input: draft.base, onApply: () => { apply() } })
    child = { read: () => { const value = editor.read(); return { input: value.input as Base, incomplete: value.incomplete } }, dispose: editor.dispose }
  }
  mountBase()
  function read(): SweepDraft {
    const nested = child.read(), input = { ...structuredClone(draft), base: nested.input }
    return { input, dirty: !exactInputEqual(input, baseline), incomplete: [...pending.values()].map(item => [...item.path]).concat(nested.incomplete.map(path => ['base', ...path])) }
  }
  function updateStatus() {
    if (disposed) return
    const value = read(), applied = !!lastApplied && exactInputEqual(value.input, lastApplied)
    for (const [index, option] of [...navigation.options].entries()) option.textContent = `${index + 1}: ${draft.axes[index]?.id || 'Unnamed axis'}`
    const coordinateSelect = fields.querySelector<HTMLSelectElement>('[data-role=sweep-coordinate]')
    if (coordinateSelect) for (const [index, option] of [...coordinateSelect.options].entries()) option.textContent = coordinateLabel(index)
    card.dataset.state = value.incomplete.length ? 'incomplete' : value.dirty ? 'draft' : 'unchanged'
    card.dataset.previewRelation = applied ? 'matches-last-submission' : 'not-submitted'
    status.textContent = value.incomplete.length ? `Finish incomplete edits: ${value.incomplete.map(path => JSON.stringify(path)).join('; ')}`
      : applied ? 'Draft matches the last submission. Each point may still have validation errors; all points remain non-executable.'
        : value.dirty ? 'Draft changed. The preview still describes the last submitted sweep.' : 'Supplied sweep unchanged. Omission inherits through the resolver.'
    applyButton.disabled = value.incomplete.length > 0
  }
  function coordinateLabel(index: number) {
    const value = draft.axes[axisIndex]?.coordinates[index], edit = pending.get(JSON.stringify(['axes', axisIndex, 'coordinates', index]))
    return `${index + 1}: ${edit ? `${edit.edit.text || '(empty)'} (incomplete)` : value?.kind === 'omit' ? 'inherit / omit' : display(value?.value)}`
  }
  function button(label: string, action: string, callback: () => void) {
    const value = node('button', label); value.type = 'button'; value.className = 'btn btn-ghost'; value.dataset.action = action
    value.addEventListener('click', () => { if (!disposed) callback() }); return value
  }
  function write(path: Path, value: FieldEdit) {
    const key = JSON.stringify(path)
    if (value.kind === 'incomplete') pending.set(key, { path, edit: value })
    else {
      const owner = at(draft, path.slice(0, -1)); if (!owner || typeof owner !== 'object') throw Error('Create the selected declaration first.')
      if (value.kind === 'omit') Reflect.deleteProperty(owner, path.at(-1)!)
      else Reflect.set(owner, path.at(-1)!, value.value)
      pending.delete(key)
    }
    updateStatus()
  }
  function scalar(host: HTMLElement, path: Path, metadata: FieldSchema, required: boolean, initial: FieldValue,
                  changed: (edit: FieldEdit) => void = edit => write(path, edit)) {
    const key = JSON.stringify(path), saved = pending.get(key), box = node('div'); box.dataset.path = key
    const control = mountScalar(box, String(path.at(-1)), metadata, required, saved ? { kind: 'value', value: saved.edit.text } : initial,
      edit => { if (!disposed) { if (edit.kind === 'incomplete') pending.set(key, { path, edit }); else pending.delete(key); changed(edit) } })
    controls.push(control); host.append(box)
    const edit = control.read(); if (edit.kind === 'incomplete') pending.set(key, { path, edit }); else if (saved) pending.set(key, saved)
  }
  function mutate(path: Path, action: RowMutation, inserted?: unknown) {
    const values = at(draft, path); if (!Array.isArray(values)) return
    const entries = [...pending.values()]; pending.clear()
    for (const entry of entries) {
      if (under(entry.path, path) && typeof entry.path[path.length] === 'number') {
        const moved = movedRow(entry.path[path.length] as number, action); if (moved === undefined) continue
        entry.path = [...entry.path]; entry.path[path.length] = moved
      }
      pending.set(JSON.stringify(entry.path), entry)
    }
    mutateRows(values, action, path, inserted)
    const selected = path.length === 1 ? axisIndex : coordinateIndex
    const next = action.kind === 'insert' ? action.index : movedRow(selected, action) ?? Math.min(selected, values.length - 1)
    if (path.length === 1) { axisIndex = Math.max(0, next); coordinateIndex = 0 } else coordinateIndex = Math.max(0, next)
    render()
  }
  function rowActions(host: HTMLElement, path: Path, index: number, value: unknown, count: number, noun: string) {
    const bar = node('div'); bar.className = 'toolbar'
    const up = button(`Move ${noun} earlier`, `move-${noun}-up`, () => mutate(path, { kind: 'move', from: index, to: index - 1 }))
    const down = button(`Move ${noun} later`, `move-${noun}-down`, () => mutate(path, { kind: 'move', from: index, to: index + 1 }))
    up.disabled = index === 0; down.disabled = index === count - 1
    bar.append(up, down, button(`Copy ${noun}`, `copy-${noun}`, () => mutate(path, { kind: 'insert', index: index + 1 }, value)),
      button(`Remove ${noun}`, `remove-${noun}`, () => mutate(path, { kind: 'remove', index })))
    host.append(bar)
  }
  function render() {
    for (const control of controls.splice(0)) control.dispose()
    fields.replaceChildren(); navigation.replaceChildren(...draft.axes.map((axis, index) => new Option(`${index + 1}: ${axis.id || 'Unnamed axis'}`, String(index))))
    axisIndex = Math.max(0, Math.min(axisIndex, draft.axes.length - 1)); navigation.value = String(axisIndex)
    scalar(fields, ['id'], sweepSchema.properties!.id, true, { kind: 'value', value: draft.id })
    scalar(fields, ['reference_id'], sweepSchema.properties!.reference_id, false,
      Object.hasOwn(draft, 'reference_id') ? { kind: 'value', value: draft.reference_id ?? null } : { kind: 'omit' })
    const add = button('Add empty axis', 'add-axis', () => mutate(['axes'], { kind: 'insert', index: draft.axes.length }, { id: '', target: {}, coordinates: [] }))
    add.disabled = draft.axes.length >= 2; fields.append(add)
    const axis = draft.axes[axisIndex]
    if (!axis) { updateStatus(); return }
    rowActions(fields, ['axes'], axisIndex, axis, draft.axes.length, 'axis')
    scalar(fields, ['axes', axisIndex, 'id'], definitions.SweepAxisInput.properties!.id, true, { kind: 'value', value: axis.id })
    const choices = sweepTargets(child.read().input), targetSelect = node('select'), path: Path = ['axes', axisIndex, 'target'], key = JSON.stringify(path)
    targetSelect.setAttribute('aria-label', 'Stable parameter target'); targetSelect.dataset.role = 'sweep-target'
    targetSelect.add(new Option('Choose a supplied parameter instance', ''))
    choices.forEach((choice, index) => targetSelect.add(new Option(choice.label, String(index))))
    const selected = choices.findIndex(choice => exactInputEqual(choice.target, axis.target))
    if (selected === -1 && Object.keys(axis.target).length) {
      targetSelect.add(new Option(`Unavailable supplied target: ${JSON.stringify(axis.target)}`, 'retained')); targetSelect.value = 'retained'
    } else targetSelect.value = selected < 0 ? '' : String(selected)
    if (selected < 0 && !Object.keys(axis.target).length) pending.set(key, { path, edit: { kind: 'incomplete', text: '', message: 'Select a target.' } })
    targetSelect.addEventListener('change', () => {
      if (disposed || targetSelect.value === '' || targetSelect.value === 'retained') return
      axis.target = structuredClone(choices[targetSelect.selectedIndex - 1].target); pending.delete(key); render()
    })
    fields.append(node('p', 'Parameter instance (stable IDs)'), targetSelect)
    const choice = choices[selected]
    if (choice) fields.append(node('p', `Target: ${JSON.stringify(axis.target)}${choice.target.family === 'cbo' ? ' — effective absorber affinity minus the declared offset; no V_bi adjustment.' : ''}`))
    const coordinates = node('select'); coordinates.dataset.role = 'sweep-coordinate'; coordinates.setAttribute('aria-label', 'Declared coordinate')
    axis.coordinates.forEach((_value, index) => coordinates.add(new Option(coordinateLabel(index), String(index))))
    coordinateIndex = Math.max(0, Math.min(coordinateIndex, axis.coordinates.length - 1)); coordinates.value = String(coordinateIndex)
    coordinates.addEventListener('change', () => { if (!disposed) { coordinateIndex = coordinates.selectedIndex; render() } })
    fields.append(node('p', 'Ordered coordinates'), coordinates)
    const valuesPath: Path = ['axes', axisIndex, 'coordinates'], coordinate = axis.coordinates[coordinateIndex]
    if (coordinate) {
      rowActions(fields, valuesPath, coordinateIndex, coordinate, axis.coordinates.length, 'coordinate')
      if (choice) {
        const valuePath = [...valuesPath, coordinateIndex]
        scalar(fields, valuePath, { ...choice.metadata, title: 'Coordinate value' }, choice.required,
          coordinate.kind === 'omit' ? { kind: 'omit' } : { kind: 'value', value: coordinate.value ?? null }, edit => {
            if (edit.kind !== 'incomplete') {
              if (edit.kind === 'omit') { coordinate.kind = 'omit'; Reflect.deleteProperty(coordinate, 'value') }
              else { coordinate.kind = 'value'; coordinate.value = edit.value }
            }
            updateStatus()
          })
      } else fields.append(node('p', 'Coordinate words are retained while the target is unavailable; select a supported target to edit them.'))
    }
    fields.append(button('Add empty coordinate', 'add-coordinate', () => mutate(valuesPath, { kind: 'insert', index: axis.coordinates.length }, { kind: 'value', value: '' })),
      node('p', 'One axis declares its coordinates; two axes declare their Cartesian product in the supplied order. Equivalent duplicate coordinates and resource-limit overflow are reported without truncation.'))
    updateStatus()
  }
  function apply() { const value = read(); if (disposed || value.incomplete.length) return false; lastApplied = structuredClone(value.input); options.onApply(structuredClone(value.input)); updateStatus(); return true }
  function discard() { if (!disposed) { draft = structuredClone(baseline); pending.clear(); mountBase(); render() } }
  const navigate = () => { if (!disposed) { axisIndex = navigation.selectedIndex; coordinateIndex = 0; render() } }
  navigation.addEventListener('change', navigate)
  for (const event of ['input', 'change', 'click']) baseHost.addEventListener(event, updateStatus)
  render()
  return { read, apply, discard,
    setInput(input) { if (!disposed) { baseline = structuredClone(input); draft = structuredClone(input); pending.clear(); axisIndex = coordinateIndex = 0; lastApplied = undefined; mountBase(); render() } },
    dispose() { if (!disposed) { disposed = true; for (const item of controls.splice(0)) item.dispose(); child.dispose(); navigation.removeEventListener('change', navigate); for (const event of ['input', 'change', 'click']) baseHost.removeEventListener(event, updateStatus); pending.clear(); card.remove() } },
  }
}

export interface SweepPreview extends SweepEditor { readonly ready: Promise<void>; select(input: SweepInput): Promise<void>; reopen(json: string): Promise<void>; openPoint(id: string): Promise<void> }
export function mountSweepPreview(root: HTMLElement, options: { input: SweepInput; endpoints: { sweep: string | URL; device: string | URL; experiment: string | URL } }): SweepPreview {
  let disposed = false, generation = 0, selectedPoint = '', pointPanel: ConfigurationPreviewPanel | undefined
  const editorHost = node('div'), previewHost = node('div'), pointHost = node('div'), pointView = node('section'), points = node('select'), detail = node('p'), values = node('p'), error = node('p'), referenceNote = node('p')
  previewHost.dataset.role = 'sweep-preview'; pointHost.dataset.role = 'point-preview'
  pointView.className = 'card'; pointView.dataset.panel = 'sweep-points'; points.setAttribute('aria-label', 'Prepared point'); points.dataset.role = 'prepared-point'
  const open = node('button', 'Open selected point input'); open.type = 'button'; open.className = 'btn btn-ghost'; open.dataset.action = 'open-point'
  values.dataset.role = 'point-effective-values'; error.setAttribute('role', 'alert')
  referenceNote.dataset.role = 'sweep-reference-interpretation'; referenceNote.hidden = true
  pointView.append(node('h3', 'Points from the last submitted sweep'), referenceNote, points, detail, values, open, error, pointHost)
  const source = node('textarea'), reopenDetails = node('details'), reopenButton = node('button', 'Reopen sweep input')
  source.rows = 6; source.style.width = '100%'; source.setAttribute('aria-label', 'Original sweep preview JSON'); reopenButton.type = 'button'; reopenButton.dataset.action = 'reopen-sweep'; reopenButton.className = 'btn btn-ghost'
  reopenDetails.append(node('summary', 'Reopen an exported sweep'), source, reopenButton)
  root.replaceChildren(editorHost, pointView, previewHost, reopenDetails)
  const panel = mountConfigurationPreviewPanel(previewHost, { endpoint: options.endpoints.sweep, kind: 'sweep', input: options.input })
  function records() { const value = panel.document?.value.resolved.points; return Array.isArray(value) ? value.filter(object) : [] }
  function showPoint() {
    const value = records().find(point => point.id === selectedPoint)
    detail.textContent = value ? `${value.status}; simulation: ${value.simulation_status}; reference: ${object(value.reference) ? value.reference.status : 'unavailable'}${object(value.reference) && value.reference.reason ? ' — ' + value.reference.reason : ''}` : 'Apply a sweep to inspect its points.'
    if (object(value?.reference) && value.reference.baseline_matches === false) detail.textContent += ' — Base input differs from the reference declaration; coverage is not a qualified comparison.'
    if (Array.isArray(value?.field_errors)) detail.textContent += ' — ' + value.field_errors.filter(object).map(field => `${JSON.stringify(field.loc)}: ${field.msg}`).join('; ')
    values.textContent = object(value?.effective_targets) ? Object.entries(value.effective_targets).filter(([, item]) => object(item))
      .map(([name, item]) => {
        const field = item as Record<string, unknown>
        if (Object.hasOwn(field, 'offset_eV')) return `${name}: ${display(field.offset_eV)} eV; ETL electron affinity: ${display(field.value)} eV; absorber electron affinity: ${display(field.reference_affinity_eV)} eV; origin: ${field.origin}`
        return `${name}: ${Object.hasOwn(field, 'value') ? display(field.value) : 'named-material consumers retained'}; origin: ${field.origin}`
      }).join(' · ') : ''
    open.disabled = !value || !object(value.applied_input)
  }
  function refreshed() {
    const scope = panel.document?.value.resolved.reference_scope, interpretation = object(scope) && object(scope.interpretation) ? scope.interpretation : undefined
    referenceNote.hidden = !interpretation?.interface_density_choice
    if (interpretation?.interface_density_choice) {
      const axes = Array.isArray(interpretation.interface_density_axes) ? interpretation.interface_density_axes.filter(object) : []
      const units = axes.map(axis => `${axis.axis_id}: reported ${axis.reported_unit}; declared ${axis.input_unit}; resolved ${axis.resolved_unit}.`).join(' ')
      referenceNote.textContent = `SCAPS interface-density channel: ${interpretation.interface_density_choice}. ${units || 'This sweep has no interface-density axis.'} This is the historical areal-input hypothesis, with no thickness conversion and no verified native SCAPS input.`
    } else referenceNote.textContent = ''
    const values = records(); if (!values.some(point => point.id === selectedPoint)) selectedPoint = String(values[0]?.id ?? '')
    points.replaceChildren(...values.map(point => new Option(`${object(point.coordinates) ? Object.entries(point.coordinates).map(([name, value]) => `${name}=${object(value) && value.kind === 'omit' ? 'inherit' : object(value) ? display(value.value) : display(value)}`).join(', ') : point.id} — ${point.status}`, String(point.id))))
    points.value = selectedPoint; showPoint()
  }
  let ready = panel.ready.then(refreshed)
  async function submit(input: SweepInput) {
    const current = ++generation; pointPanel?.dispose(); pointPanel = undefined; error.textContent = ''
    await panel.select({ endpoint: options.endpoints.sweep, kind: 'sweep', input })
    if (!disposed && current === generation) refreshed()
  }
  const editor = mountSweepEditor(editorHost, { input: options.input, onApply(input) { ready = submit(input) } })
  async function select(input: SweepInput) { if (!disposed) { editor.setInput(input); ready = submit(input); await ready } }
  async function reopen(json: string) { if (!disposed) await select(sweepInputFromPreview(json)) }
  async function openPoint(id: string) {
    if (disposed || !panel.document) return
    const input = sweepPointInputFromPreview(panel.document.json, id); selectedPoint = id; points.value = id; showPoint()
    pointPanel?.dispose()
    pointPanel = input.schema_version === 'solarlab.device-preparation.v1'
      ? mountConfigurationPreviewPanel(pointHost, { endpoint: options.endpoints.device, kind: 'device', input })
      : mountConfigurationPreviewPanel(pointHost, { endpoint: options.endpoints.experiment, kind: 'experiment', input })
    await pointPanel.ready
  }
  const changed = () => { if (!disposed) { selectedPoint = points.value; showPoint() } }
  const opened = () => { error.textContent = ''; ready = openPoint(selectedPoint).catch(cause => { if (!disposed) error.textContent = String(cause) }) }
  const reopened = () => { error.textContent = ''; ready = reopen(source.value).catch(cause => { if (!disposed) error.textContent = String(cause) }) }
  points.addEventListener('change', changed); open.addEventListener('click', opened); reopenButton.addEventListener('click', reopened)
  return { read: editor.read, apply: editor.apply, discard: editor.discard, setInput: input => { void select(input) }, select, reopen, openPoint,
    get ready() { return ready },
    dispose() { if (!disposed) { disposed = true; generation++; editor.dispose(); panel.dispose(); pointPanel?.dispose(); points.removeEventListener('change', changed); open.removeEventListener('click', opened); reopenButton.removeEventListener('click', reopened); root.replaceChildren() } },
  }
}
