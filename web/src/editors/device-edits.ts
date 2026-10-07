import { configurationSchema } from '../generated/configuration-schema'
import type { DeviceInput, TandemInput } from '../generated/configuration-inputs'

export type EditorInput = DeviceInput | TandemInput
export type Cell = 'device' | 'top_cell' | 'bottom_cell'
export type ItemIdentity = { id: string; occurrence: number }
export type DefectSelection = { cell: Cell; kind: 'interface_defect'; parent: ItemIdentity; id: string }
  | { cell: Cell; kind: 'bulk_defect'; parent: ItemIdentity; id: string; occurrence: number }
export type Selection = { cell: Cell } & (
  { kind: 'settings' } | ({ kind: 'layer' | 'material' | 'interface' | 'contact' } & ItemIdentity)
) | DefectSelection
export type FieldValue = { kind: 'omit' } | { kind: 'value'; value: string | number | boolean | null }
export type FieldEdit = FieldValue | { kind: 'incomplete'; text: string; message: string }
export type Path = (string | number)[]

/** Only the schema features needed to present supplied scalar fields and audit
 * a layer-ID rewrite. This is not a replacement for Python validation. */
export interface FieldSchema {
  readonly title?: string
  readonly type?: string
  readonly unit?: string
  readonly nullable?: boolean
  readonly minLength?: number
  readonly enum?: readonly (string | number | boolean | null)[]
  readonly const?: string | number | boolean | null
  readonly anyOf?: readonly FieldSchema[]
  readonly $ref?: string
  readonly properties?: Readonly<Record<string, FieldSchema>>
  readonly required?: readonly string[]
  readonly items?: FieldSchema
  readonly $defs?: Readonly<Record<string, FieldSchema>>
}

export const fieldSchemas: Record<'layer' | 'material' | 'parameters' | 'settings' | 'interface' | 'contact' | 'interface_defect' | 'bulk_defect', FieldSchema> = {
  layer: configurationSchema.dto_schemas.FullLayerInput.schema,
  material: configurationSchema.dto_schemas.NamedMaterialInput.schema,
  parameters: configurationSchema.dto_schemas.FullParameterInput.schema,
  settings: configurationSchema.dto_schemas.DeviceSettingsInput.schema,
  interface: configurationSchema.dto_schemas.InterfaceInput.schema,
  contact: configurationSchema.dto_schemas.ContactInput.schema,
  interface_defect: configurationSchema.dto_schemas.DeviceInput.schema.$defs.InterfaceDefectInput,
  bulk_defect: configurationSchema.dto_schemas.BulkDefectInput.schema,
}

export class DeviceEditError extends Error {
  readonly paths: Path[]
  constructor(message: string, paths: Path[]) {
    super(message)
    this.name = 'DeviceEditError'
    this.paths = paths
  }
}

export function cellInput(input: EditorInput, cell: Cell): DeviceInput {
  if (input.schema_version === 'solarlab.device-preparation.v1' && cell === 'device') return input
  if (input.schema_version === 'solarlab.tandem-preparation.v1' && cell !== 'device') return input[cell]
  throw new DeviceEditError('Select a cell supplied by this input.', [[cell]])
}

export function selections(input: EditorInput, cell: Cell): Selection[] {
  const device = cellInput(input, cell)
  const entries = (kind: 'layer' | 'material' | 'interface' | 'contact', items: readonly { id: string }[]) => items.map((item, index) => ({
    cell, kind, id: item.id, occurrence: items.slice(0, index).filter(other => other.id === item.id).length,
  } as Selection))
  const interfaces = entries('interface', device.interfaces ?? []).flatMap((selection, index) => {
    if (selection.kind !== 'interface') return []
    const defect = device.interfaces![index].defect
    return [selection, ...(defect && typeof defect === 'object' ? [{ cell, kind: 'interface_defect' as const,
      parent: { id: selection.id, occurrence: selection.occurrence }, id: defect.id }] : [])]
  })
  const layers = entries('layer', device.layers).flatMap((selection, index) => {
    if (selection.kind !== 'layer') return []
    const defects = device.layers[index].bulk_defects ?? []
    return [selection, ...defects.map((defect, offset) => ({ cell, kind: 'bulk_defect' as const,
      parent: { id: selection.id, occurrence: selection.occurrence }, id: defect.id,
      occurrence: defects.slice(0, offset).filter(item => item.id === defect.id).length }))]
  })
  return [...layers, ...entries('material', device.materials ?? []),
    ...entries('contact', device.contacts ?? []), ...interfaces, { cell, kind: 'settings' }]
}

export function selectedEntity(input: EditorInput, selection: Selection): { value: object | undefined; path: Path; index: number } {
  const device = cellInput(input, selection.cell)
  const prefix: Path = selection.cell === 'device' ? [] : [selection.cell]
  if (selection.kind === 'settings') return { value: device.settings, path: [...prefix, 'settings'], index: -1 }
  if (selection.kind === 'interface_defect') {
    const parent = selectedEntity(input, { cell: selection.cell, kind: 'interface', ...selection.parent })
    const defect = device.interfaces![parent.index].defect
    if (!defect || defect.id !== selection.id) throw new DeviceEditError('The selected interface defect is no longer supplied.', [[...parent.path, 'defect']])
    return { value: defect, path: [...parent.path, 'defect'], index: parent.index }
  }
  if (selection.kind === 'bulk_defect') {
    const parent = selectedEntity(input, { cell: selection.cell, kind: 'layer', ...selection.parent })
    const defects = device.layers[parent.index].bulk_defects ?? []
    let occurrence = 0
    const index = defects.findIndex(item => item.id === selection.id && occurrence++ === selection.occurrence)
    if (index < 0) throw new DeviceEditError('The selected bulk defect is no longer supplied.', [[...parent.path, 'bulk_defects']])
    return { value: defects[index], path: [...parent.path, 'bulk_defects', index], index }
  }
  const collection = selection.kind === 'layer' ? 'layers' : selection.kind === 'material' ? 'materials'
    : selection.kind === 'interface' ? 'interfaces' : 'contacts'
  const items = device[collection] ?? []
  let occurrence = 0
  const index = items.findIndex(item => item.id === selection.id && occurrence++ === selection.occurrence)
  if (index < 0) throw new DeviceEditError('The selected item is no longer supplied.', [[...prefix, collection]])
  return { value: items[index], path: [...prefix, collection, index], index }
}

export function schemaVariants(schema: FieldSchema): readonly FieldSchema[] {
  return schema.anyOf ? schema.anyOf.flatMap(schemaVariants) : [schema]
}

export function allowsNull(schema: FieldSchema) {
  return schema.nullable === true || schemaVariants(schema).some(item => item.type === 'null')
}

export function scalarField(schema: FieldSchema) {
  return schemaVariants(schema).every(item => !item.$ref && ['string', 'integer', 'number', 'boolean', 'null'].includes(item.type ?? ''))
}

export function fieldSchema(selection: Selection, parameter: boolean) {
  return fieldSchemas[parameter ? 'parameters' : selection.kind]
}

/** Preserve every unselected value and container. Omitting a missing override
 * never creates its parent; removing a field from a supplied {} keeps that {}. */
export function writeField<T extends EditorInput>(input: T, selection: Selection, field: string,
  edit: FieldValue, parameter = false): T {
  const schema = fieldSchema(selection, parameter), metadata = schema.properties?.[field]
  const selected = selectedEntity(input, selection)
  const path = [...selected.path, ...(parameter ? ['parameters'] : []), field]
  const basic = selection.kind === 'layer' ? ['name', 'role', 'thickness', 'material', 'defect_model', 'defect_schema_version']
    : selection.kind === 'interface' || selection.kind === 'contact' ? Object.keys(schema.properties ?? {}).filter(key => key !== 'id') : ['name']
  if (parameter && selection.kind !== 'layer' && selection.kind !== 'material') throw new DeviceEditError('Select a layer or material parameter.', [path])
  if (!metadata || !scalarField(metadata) || (selection.kind !== 'settings' && !parameter && !basic.includes(field))) {
    throw new DeviceEditError('This field has no scalar editor.', [path])
  }
  if (edit.kind === 'omit' && schema.required?.includes(field)) {
    throw new DeviceEditError('This required field cannot be omitted.', [path])
  }
  if (edit.kind === 'value' && edit.value === null && !allowsNull(metadata)) {
    throw new DeviceEditError('This field does not accept null.', [path])
  }
  const next = structuredClone(input), device = cellInput(next, selection.cell)
  let owner: object | undefined
  if (selection.kind === 'settings') {
    if (edit.kind === 'value' && !device.settings) device.settings = {}
    owner = device.settings
  } else {
    const entity = selectedEntity(next, selection).value!
    if (parameter) {
      if (edit.kind === 'value' && !Reflect.get(entity, 'parameters')) Reflect.set(entity, 'parameters', {})
      owner = Reflect.get(entity, 'parameters')
    } else owner = entity
  }
  if (owner) {
    if (edit.kind === 'omit') Reflect.deleteProperty(owner, field)
    else Reflect.set(owner, field, edit.value)
  }
  return next
}

/** Explicit typed layer references in the current DeviceInput contract. Names,
 * nested defect IDs and resource IDs are not layer references. */
export function layerReferences(device: DeviceInput): { path: Path; value: string }[] {
  return [
    ...(device.interfaces ?? []).flatMap((item, index) => [
      { path: ['interfaces', index, 'left'], value: item.left },
      { path: ['interfaces', index, 'right'], value: item.right },
    ]),
    ...(device.contacts ?? []).map((item, index) => ({ path: ['contacts', index, 'layer'], value: item.layer })),
    ...(device.electrical_grid ?? []).map((item, index) => ({ path: ['electrical_grid', index, 'layer'], value: item.layer })),
    ...(device.grain_boundaries ?? []).flatMap((item, index) => item.layer_ids.map((value, offset) => ({
      path: ['grain_boundaries', index, 'layer_ids', offset], value,
    }))),
  ]
}

function setPath(target: object, path: Path, value: string) {
  let owner = target
  for (const key of path.slice(0, -1)) owner = Reflect.get(owner, key)
  Reflect.set(owner, path.at(-1)!, value)
}

/** Connections have no typed external ID references in this input schema.
 * Renaming the selected occurrence never changes its layer bindings or defect. */
export function renameConnection<T extends EditorInput>(input: T, selection: Selection, id: string) {
  const found = selectedEntity(input, selection)
  if (selection.kind !== 'interface' && selection.kind !== 'contact') throw new DeviceEditError('Select a contact or interface.', [found.path])
  const next = structuredClone(input), device = cellInput(next, selection.cell)
  const prefix: Path = selection.cell === 'device' ? [] : [selection.cell]
  const unknown = unknownPaths(device, [configurationSchema.dto_schemas.DeviceInput.schema], configurationSchema.dto_schemas.DeviceInput.schema, prefix)
  if (id !== selection.id && unknown.length) throw new DeviceEditError('Cannot guarantee references in unrecognized fields; the ID was not changed.', unknown)
  const collection = selection.kind === 'interface' ? 'interfaces' : 'contacts'
  device[collection]![found.index].id = id
  return { input: next, selection: { ...selection, id,
    occurrence: device[collection]!.slice(0, found.index).filter(item => item.id === id).length } }
}

// Reject unknown/opaque input branches before an identity rewrite. The schema
// has no general reference annotations for extensions; do not guess their IDs.
function unknownPaths(value: unknown, schemas: readonly FieldSchema[], root: FieldSchema, path: Path): Path[] {
  const expanded = schemas.flatMap(schema => schema.$ref
    ? [root.$defs?.[schema.$ref.split('/').at(-1)!] ?? {}] : schemaVariants(schema))
  if (expanded.some(schema => schema.$ref || schema.anyOf)) return unknownPaths(value, expanded, root, path)
  if (value === null || typeof value !== 'object') return []
  if (Array.isArray(value)) {
    const items = expanded.flatMap(schema => schema.items ? [schema.items] : [])
    return items.length ? value.flatMap((item, index) => unknownPaths(item, items, root, [...path, index])) : [path]
  }
  const objects = expanded.filter(schema => schema.properties)
  if (!objects.length) return [path]
  return Object.entries(value).flatMap(([key, item]) => {
    const children = objects.flatMap(schema => schema.properties?.[key] ? [schema.properties[key]] : [])
    return children.length ? unknownPaths(item, children, root, [...path, key]) : [[...path, key]]
  })
}

export function requireKnownIdentityScope(input: EditorInput, cell: Cell) {
  const schema = configurationSchema.dto_schemas.DeviceInput.schema
  const paths = unknownPaths(cellInput(input, cell), [schema], schema, cell === 'device' ? [] : [cell])
  if (paths.length) throw new DeviceEditError('Cannot guarantee references in unrecognized fields; the ID was not changed.', paths)
}

export function renameLayer<T extends EditorInput>(input: T, selection: Selection, id: string) {
  const selected = selectedEntity(input, selection)
  if (selection.kind !== 'layer') throw new DeviceEditError('Select a layer to change its ID.', [selected.path])
  const next = structuredClone(input), device = cellInput(next, selection.cell)
  if (id === selection.id) return { input: next, selection, updatedPaths: [] as Path[] }
  const prefix: Path = selection.cell === 'device' ? [] : [selection.cell]
  const unknown = unknownPaths(device, [configurationSchema.dto_schemas.DeviceInput.schema],
    configurationSchema.dto_schemas.DeviceInput.schema, prefix)
  if (unknown.length) throw new DeviceEditError('Cannot guarantee references in unrecognized fields; the ID was not changed.', unknown)
  const references = layerReferences(device).filter(item => item.value === selection.id)
  const duplicates = device.layers.flatMap((item, index) => item.id === selection.id ? [[...prefix, 'layers', index, 'id']] : [])
  if (duplicates.length !== 1) throw new DeviceEditError('This ID is ambiguous; discard the duplicate-ID draft before rewriting references.',
    [...duplicates, ...references.map(item => [...prefix, ...item.path])])
  device.layers[selected.index].id = id
  for (const item of references) setPath(device, item.path, id)
  return { input: next, selection: { ...selection, id,
    occurrence: device.layers.slice(0, selected.index).filter(item => item.id === id).length },
  updatedPaths: [[...selected.path, 'id'], ...references.map(item => [...prefix, ...item.path])] }
}

export function moveLayer<T extends EditorInput>(input: T, selection: Selection, direction: -1 | 1) {
  const selected = selectedEntity(input, selection)
  if (selection.kind !== 'layer') throw new DeviceEditError('Select a layer to move.', [selected.path])
  const next = structuredClone(input), device = cellInput(next, selection.cell)
  const destination = selected.index + direction
  if (destination < 0 || destination >= device.layers.length) return { input: next, selection }
  const [layer] = device.layers.splice(selected.index, 1)
  device.layers.splice(destination, 0, layer)
  return { input: next, selection: { ...selection,
    occurrence: device.layers.slice(0, destination).filter(item => item.id === selection.id).length } }
}

export function writeConnection<T extends EditorInput>(input: T, cell: Cell,
  collection: 'interfaces' | 'contacts', index: number, field: 'left' | 'right' | 'layer', value: string): T {
  const next = structuredClone(input), owner = cellInput(next, cell)[collection]?.[index]
  if (!owner || !(collection === 'interfaces' ? ['left', 'right'] : ['layer']).includes(field)) {
    throw new DeviceEditError('Select a supplied connection endpoint.', [[cell, collection, index, field]])
  }
  Reflect.set(owner, field, value)
  return next
}

export function exactInputEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true
  if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object' || Array.isArray(a) !== Array.isArray(b)) return false
  const keys = Object.keys(a), other = Object.keys(b)
  return keys.length === other.length && keys.every((key, index) => key === other[index]
    && exactInputEqual(Reflect.get(a, key), Reflect.get(b, key)))
}
