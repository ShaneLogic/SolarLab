import {
  cellInput, DeviceEditError, mutateRows, selectedEntity, selections,
} from './device-edits'
import type { Cell, EditorInput, ItemIdentity, Path, RowMutation, Selection } from './device-edits'

export type NewRecordKind = 'layer' | 'material' | 'contact' | 'interface'
  | 'bulk_defect' | 'multivalent_defect' | 'metastable_defect' | 'interface_defect'

export function canRemoveRecord(selection: Selection) {
  return ['layer', 'material', 'contact', 'interface', 'bulk_defect', 'metastable_defect', 'interface_defect'].includes(selection.kind)
}

export function recordKind(input: EditorInput, selection: Selection): NewRecordKind | undefined {
  if (selection.kind !== 'layer' && selection.kind !== 'material' && selection.kind !== 'contact' && selection.kind !== 'interface'
    && selection.kind !== 'bulk_defect' && selection.kind !== 'metastable_defect' && selection.kind !== 'interface_defect') return undefined
  if (selection.kind !== 'bulk_defect') return selection.kind
  const item = selectedEntity(input, selection).value!
  if (Object.hasOwn(item, 'configuration') && !Object.hasOwn(item, 'distribution')) return 'multivalent_defect'
  return Object.hasOwn(item, 'distribution') && !Object.hasOwn(item, 'configuration') ? 'bulk_defect' : undefined
}

export function layerParent(selection: Selection): ItemIdentity | undefined {
  if (selection.kind === 'layer' || selection.kind === 'metastable_document' || selection.kind === 'metastable_preparation') return { id: selection.id, occurrence: selection.occurrence }
  if (selection.kind === 'bulk_defect' || selection.kind === 'metastable_defect') return selection.parent
}

export function interfaceParent(selection: Selection): ItemIdentity | undefined {
  if (selection.kind === 'interface') return { id: selection.id, occurrence: selection.occurrence }
  if (selection.kind === 'interface_defect') return selection.parent
}

function emptyConfiguration() {
  return { charge_states_e: [], state_degeneracies: [], energy_levels: { correlation_energies_eV: [] }, transition_kinetics: [] }
}

/** Empty declarations contain structure only. Required physical values, branch
 * choices and protocol tags stay absent until the user explicitly enters them. */
function emptyRecord(kind: NewRecordKind, id: string): Record<string, unknown> {
  switch (kind) {
    case 'layer': case 'contact': case 'interface': return { id }
    case 'material': return { id, parameters: {} }
    case 'interface_defect': return { id, kinetics: {} }
    case 'bulk_defect': return { id, distribution: {}, kinetics: {} }
    case 'multivalent_defect': return { id, configuration: emptyConfiguration() }
    case 'metastable_defect': return { id, donor_configuration: emptyConfiguration(), acceptor_configuration: emptyConfiguration(), conversion_kinetics: {} }
  }
}

export interface RecordChange<T extends EditorInput> {
  input: T
  selection: Selection
  path: Path
  mutation?: RowMutation
  copiedFrom?: Path
}

export function createRecord<T extends EditorInput>(input: T, target: Selection, kind: NewRecordKind,
  id: string, template?: Selection): RecordChange<T> {
  if (!id.length) throw new DeviceEditError('Enter an explicit ID for the new record.', [['id']])
  if (template && (template.cell !== target.cell || recordKind(input, template) !== kind)) throw new DeviceEditError('Select a matching template in this cell.', [selectedEntity(input, template).path])
  const next = structuredClone(input), device = cellInput(next, target.cell)
  const prefix: Path = target.cell === 'device' ? [] : [target.cell]
  const source = template && selectedEntity(input, template), item = source ? structuredClone(source.value!) : emptyRecord(kind, id)
  Reflect.set(item, 'id', id)
  // This reference belongs to the copied interface defect itself. References
  // outside a copied record are never added, deleted or redirected implicitly.
  if (source && kind === 'interface_defect') {
    const metadata = Reflect.get(item, 'partner_metadata')
    if (metadata && metadata.defect_id === Reflect.get(source.value!, 'id')) metadata.defect_id = id
  }
  const parent = layerParent(target)
  let owner: object = device, key: string, path: Path
  if (kind === 'interface_defect') {
    const identity = interfaceParent(target)
    if (!identity) throw new DeviceEditError('Select the interface that will receive the defect.', [[...prefix, 'interfaces']])
    const selected = selectedEntity(next, { cell: target.cell, kind: 'interface', ...identity })
    const parentInterface = selected.value!
    if (Reflect.get(parentInterface, 'defect') != null) throw new DeviceEditError('This interface already has a defect; remove it explicitly before creating another.', [[...selected.path, 'defect']])
    Reflect.set(parentInterface, 'defect', item)
    return { input: next, selection: { cell: target.cell, kind: 'interface_defect', parent: identity, id },
      path: [...selected.path, 'defect'], copiedFrom: source?.path }
  }
  if (kind === 'bulk_defect' || kind === 'multivalent_defect' || kind === 'metastable_defect') {
    if (!parent) throw new DeviceEditError('Select the layer that will receive the defect.', [[...prefix, 'layers']])
    const layer = selectedEntity(next, { cell: target.cell, kind: 'layer', ...parent })
    if (kind === 'metastable_defect') {
      owner = Reflect.get(layer.value!, 'metastable_document')
      if (!owner || !Array.isArray(Reflect.get(owner, 'metastable_defects'))) throw new DeviceEditError('Start a metastable inventory declaration on this layer first.', [[...layer.path, 'metastable_document']])
      key = 'metastable_defects'; path = [...layer.path, 'metastable_document', key]
    } else { owner = layer.value!; key = 'bulk_defects'; path = [...layer.path, key] }
  } else {
    key = kind === 'layer' ? 'layers' : kind === 'material' ? 'materials' : kind === 'contact' ? 'contacts' : 'interfaces'
    path = [...prefix, key]
  }
  const original = Reflect.get(owner, key)
  if (original !== undefined && !Array.isArray(original)) throw new DeviceEditError('The supplied inventory is not a list; it was retained unchanged.', [path])
  const rows: object[] = original ?? []
  const index = rows.length, mutation = { kind: 'insert' as const, index }
  mutateRows(rows, mutation, path, item); Reflect.set(owner, key, rows)
  const occurrence = rows.slice(0, index).filter(value => Reflect.get(value, 'id') === id).length
  const selection: Selection = kind === 'bulk_defect' || kind === 'multivalent_defect' || kind === 'metastable_defect'
    ? { cell: target.cell, kind: kind === 'metastable_defect' ? kind : 'bulk_defect', parent: parent!, id, occurrence }
    : { cell: target.cell, kind, id, occurrence }
  return { input: next, selection, path, mutation, copiedFrom: source?.path }
}

export function removeRecord<T extends EditorInput>(input: T, selection: Selection): RecordChange<T> {
  if (!canRemoveRecord(selection)) throw new DeviceEditError('Select a layer, material, contact, interface or defect record.', [selectedEntity(input, selection).path])
  const next = structuredClone(input), found = selectedEntity(next, selection)
  if (selection.kind === 'interface_defect') {
    const parent: Selection = { cell: selection.cell, kind: 'interface', ...selection.parent }
    Reflect.deleteProperty(selectedEntity(next, parent).value!, 'defect')
    return { input: next, selection: parent, path: found.path }
  }
  const path = found.path.slice(0, -1), mutation = { kind: 'remove' as const, index: found.index }
  let owner: unknown = next
  for (const part of path) owner = Reflect.get(owner as object, part)
  if (!Array.isArray(owner)) throw new DeviceEditError('The selected inventory is not a supplied list.', [path])
  mutateRows(owner, mutation, path)
  const parent: Selection | undefined = selection.kind === 'bulk_defect' || selection.kind === 'metastable_defect'
    ? { cell: selection.cell, kind: 'layer', ...selection.parent } : undefined
  return { input: next, selection: parent ?? selections(next, selection.cell)[0], path, mutation }
}

export function setLayerDocument<T extends EditorInput>(input: T, cell: Cell, parent: ItemIdentity,
  key: 'metastable_document' | 'metastable_preparation', mode: 'empty' | 'null' | 'omit'): T {
  const next = structuredClone(input), layer = selectedEntity(next, { cell, kind: 'layer', ...parent })
  if (mode === 'empty') {
    if (Reflect.get(layer.value!, key) != null) throw new DeviceEditError('The document is already supplied; edit it or explicitly remove it first.', [[...layer.path, key]])
    Reflect.set(layer.value!, key, key === 'metastable_document' ? { metastable_defects: [] } : { numerics: {} })
  } else if (mode === 'null') Reflect.set(layer.value!, key, null)
  else Reflect.deleteProperty(layer.value!, key)
  return next
}
