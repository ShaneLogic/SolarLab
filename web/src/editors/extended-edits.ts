import { configurationSchema } from '../generated/configuration-schema'
import {
  allowsNull, cellInput, DeviceEditError, mutateRows, scalarField, selectedEntity,
} from './device-edits'
import type { EditorInput, ExtendedSelection, FieldSchema, FieldValue, Path, RowMutation, Selection } from './device-edits'
import { nestedValue } from './defect-edits'

const deviceSchema = configurationSchema.dto_schemas.DeviceInput.schema
const definitions = deviceSchema.$defs
const tandemSchema = configurationSchema.dto_schemas.TandemInput.schema

export interface ExtendedSection {
  key: string
  label: string
  path: Path
  schema: FieldSchema
  note: string
  fields?: readonly string[]
  document?: { nullable: boolean; required: boolean; empty: object }
  array?: { item: FieldSchema; empty: unknown; rowPrefix?: string }
  row?: { arrayKey: string; index: number }
}

export function isExtendedSelection(selection: Selection): selection is ExtendedSelection {
  return selection.kind === 'layer_optics' || selection.kind === 'material_optics' || selection.kind === 'tunnelling'
    || selection.kind === 'microstructure' || selection.kind === 'optical_stack' || selection.kind === 'resources'
}

function optional(property: FieldSchema, empty: object = {}, required = false) {
  return { nullable: allowsNull(property), required, empty }
}

/** These are the named preparation DTO sections, not a recursive schema form.
 * All field types/units come from generated metadata; no defaults are copied. */
export function extendedSections(input: EditorInput, selection: ExtendedSelection): ExtendedSection[] {
  const found = selectedEntity(input, selection), owner = found.value!
  if (selection.kind === 'resources') return [{ key: 'resources', label: 'Optical and generation resources', path: [], schema: deviceSchema,
    fields: ['spectrum', 'fixed_generation'], note: 'Resource names are resolved by the trusted library. A name is not a server file path. Layer optical resources and refractive-index overrides remain in each layer’s parameter controls.' }]
  if (selection.kind === 'layer_optics' || selection.kind === 'material_optics') return [{
    key: 'cigs', label: 'CIGS graded optics', path: ['cigs_graded_optics'], schema: definitions.CigsOpticsInput,
    document: optional(definitions.FullLayerInput.properties.cigs_graded_optics),
    note: 'GGI endpoints follow the layer’s electrical grading coordinate. Omitted layer optics inherit the named material; null clears that inheritance. Omitted model, slice and quadrature controls use the supplied catalog. Activation also requires explicit device settings and electrical grading; the preview reports conflicts.',
  }]
  if (selection.kind === 'tunnelling') {
    const channels = [
      ['band_to_band', 'Band-to-band', definitions.BandToBandInput],
      ['intraband', 'Intraband', definitions.IntrabandInput],
      ['interface_defect_assisted', 'Interface defect assisted', definitions.DefectAssistedInput],
      ['contact', 'Contact', definitions.ContactTunnellingInput],
    ] as const
    return [{ key: 'tunnelling', label: 'Tunnelling declaration', path: ['tunnelling_channels'], schema: definitions.TunnellingInput,
      document: optional(deviceSchema.properties.tunnelling_channels),
      note: 'Each channel is independent. Starting an empty declaration leaves all options to the trusted catalog; it does not enable a channel or execution.' },
    ...channels.map(([key, label, schema]) => ({ key, label: `${label} channel`, path: ['tunnelling_channels', key], schema,
      document: optional(definitions.TunnellingInput.properties[key]),
      note: 'Omit scalar overrides to inherit the supplied channel settings; scalar null is unsupported. Removing or clearing a channel restores its catalog settings. Declared enabled channels still require the coupling qualifications reported by the preview.' }))]
  }
  const list = (key: string, label: string, path: Path, schema: FieldSchema, item: FieldSchema, empty: unknown, rowPrefix?: string, required = false): ExtendedSection => ({
    key, label, path, schema, document: optional(schema, [], required), array: { item, empty, rowPrefix },
    note: 'Rows stay in the displayed order. Add an empty declaration or explicitly copy a row; removal never deletes linked data or changes another row.',
  })
  const rows = (key: string, prefix: string, label: string, path: Path, schema: FieldSchema, note: string): ExtendedSection[] => {
    const parent = nestedValue(owner, path.slice(0, -1)), values = parent && Reflect.get(parent, path.at(-1)!)
    return Array.isArray(values) ? values.map((value, index) => ({ key: `${prefix}:${index}`,
      label: `${label} ${index + 1}${value && typeof value === 'object' ? ` — ${String(value.id ?? value.layer ?? 'unnamed')}` : ' — unsupported row retained'}`,
      path: [...path, index], schema, row: { arrayKey: key, index }, note })) : []
  }
  if (selection.kind === 'optical_stack') return [
    { key: 'stack', label: 'Tandem optical model', path: [], schema: tandemSchema, fields: ['junction_model', 'light_direction'],
      note: 'The junction optical layers and back reflector belong to the tandem, separate from either cell’s physical layers. Resource names refer to the trusted optical library; IDs are unique across the junction stack and reflector.' },
    list('junction', 'Junction optical layers', ['junction_stack'], tandemSchema.properties.junction_stack, tandemSchema.$defs.OpticalLayerInput, {}, 'optical', true),
    ...rows('junction', 'optical', 'Optical layer', ['junction_stack'], tandemSchema.$defs.OpticalLayerInput,
      'Edit this optical layer only. Changing its display name keeps the ID; an explicit ID edit never renames a physical layer. Incoherence and optical material are explicit declarations.'),
    { key: 'reflector', label: 'Back reflector', path: ['back_reflector'], schema: tandemSchema.$defs.OpticalLayerInput,
      document: optional(tandemSchema.properties.back_reflector),
      note: 'For the supported top-first illumination, this optional rear optical layer is behind the bottom cell: top cell → junction stack → bottom cell → back reflector. Clear or omit it explicitly; the two physical cells remain unchanged.' },
  ]
  const grains = rows('grains', 'grain', 'Grain boundary', ['grain_boundaries'], definitions.GrainBoundaryInput,
    'Position and width are physical lengths; lifetimes apply to the finite-width band. Explicit layer IDs choose its electrical layers. The resolver checks geometry and overlaps; no coordinates are normalized.')
  return [
    list('grains', 'Grain boundaries', ['grain_boundaries'], deviceSchema.properties.grain_boundaries, definitions.GrainBoundaryInput, { layer_ids: [] }, 'grain'),
    ...grains.flatMap(row => [row, { ...list(`grain-layers:${row.row!.index}`, `Layer references — ${row.label}`, [...row.path, 'layer_ids'],
      definitions.GrainBoundaryInput.properties.layer_ids, definitions.GrainBoundaryInput.properties.layer_ids.items, '', undefined, true), row: row.row }]),
    list('grid', 'Electrical grid allocation', ['electrical_grid'], deviceSchema.properties.electrical_grid, definitions.GridLayerInput, {}, 'grid-row'),
    ...rows('grid', 'grid-row', 'Grid row', ['electrical_grid'], definitions.GridLayerInput,
      'Each row references a physical electrical layer. A supplied grid must cover every electrical layer exactly once. Weights and alpha stay as entered; no mesh is generated.'),
  ]
}

export function extendedSection(input: EditorInput, selection: ExtendedSelection, key: string) {
  const selected = selectedEntity(input, selection), section = extendedSections(input, selection).find(item => item.key === key)
  if (!section) throw new DeviceEditError('Select a supported declaration section.', [selected.path])
  return { selected, section, path: [...selected.path, ...section.path] }
}

export function writeExtendedField<T extends EditorInput>(input: T, selection: ExtendedSelection, key: string, field: string, edit: FieldValue): T {
  const { section, path } = extendedSection(input, selection, key), metadata = section.schema.properties?.[field]
  if (!metadata || !scalarField(metadata) || section.fields && !section.fields.includes(field)) throw new DeviceEditError('This declaration has no scalar editor for this field.', [[...path, field]])
  if (edit.kind === 'omit' && section.schema.required?.includes(field)) throw new DeviceEditError('This required field cannot be omitted.', [[...path, field]])
  if (edit.kind === 'value' && edit.value === null && !allowsNull(metadata)) throw new DeviceEditError('This field does not accept null; omit an optional override to inherit.', [[...path, field]])
  const next = structuredClone(input), owner = nestedValue(selectedEntity(next, selection).value, section.path)
  if (!owner) throw new DeviceEditError('Start this declaration explicitly before editing its fields.', [path])
  if (edit.kind === 'omit') Reflect.deleteProperty(owner, field)
  else Reflect.set(owner, field, edit.value)
  return next
}

export function setExtendedSection<T extends EditorInput>(input: T, selection: ExtendedSelection, key: string,
  mode: 'empty' | 'null' | 'omit', copy?: { selection: ExtendedSelection; key: string }): T {
  const { section, path } = extendedSection(input, selection, key)
  if (!section.document) throw new DeviceEditError('This section cannot be replaced.', [path])
  if (mode === 'null' && !section.document.nullable || mode === 'omit' && section.document.required) throw new DeviceEditError('This section does not permit that presence state.', [path])
  const next = structuredClone(input), owner = nestedValue(selectedEntity(next, selection).value, section.path.slice(0, -1))
  if (!owner) throw new DeviceEditError('Start the containing declaration first.', [path.slice(0, -1)])
  const field = section.path.at(-1)!
  if (mode === 'omit') Reflect.deleteProperty(owner, field)
  else if (mode === 'null') Reflect.set(owner, field, null)
  else {
    if (Reflect.get(owner, field) != null) throw new DeviceEditError('The supplied section is retained; remove it explicitly before replacing it.', [path])
    let value = section.document.empty
    if (copy) {
      const source = extendedSection(input, copy.selection, copy.key)
      if (source.section.schema !== section.schema || selection.cell !== copy.selection.cell) throw new DeviceEditError('Select a matching declaration in this cell.', [source.path])
      const supplied = nestedValue(source.selected.value, source.section.path)
      if (!supplied) throw new DeviceEditError('The selected template is not a supplied object.', [source.path])
      value = supplied
    }
    Reflect.set(owner, field, structuredClone(value))
  }
  return next
}

export function extendedArray(input: EditorInput, selection: ExtendedSelection, key: string) {
  const found = extendedSection(input, selection, key), parent = nestedValue(found.selected.value, found.section.path.slice(0, -1))
  const rows: unknown = parent && Reflect.get(parent, found.section.path.at(-1)!)
  if (!found.section.array || !Array.isArray(rows)) throw new DeviceEditError('Start a supplied row list before editing it.', [found.path])
  return { ...found, rows }
}

export function changeExtendedRows<T extends EditorInput>(input: T, selection: ExtendedSelection, key: string, action: RowMutation, copyIndex?: number): T {
  const next = structuredClone(input), { section, rows, path } = extendedArray(next, selection, key)
  if (copyIndex !== undefined && (!Number.isSafeInteger(copyIndex) || copyIndex < 0 || copyIndex >= rows.length)) throw new DeviceEditError('The copied row is no longer supplied.', [path])
  mutateRows(rows, action, path, copyIndex === undefined ? section.array!.empty : rows[copyIndex])
  return next
}

export function writeExtendedRow<T extends EditorInput>(input: T, selection: ExtendedSelection, key: string, index: number, edit: FieldValue): T {
  const next = structuredClone(input), { section, rows, path } = extendedArray(next, selection, key)
  if (section.array!.rowPrefix || !Number.isSafeInteger(index) || index < 0 || index >= rows.length) throw new DeviceEditError('Select a scalar reference row.', [path])
  if (edit.kind === 'omit' || edit.value === null && !allowsNull(section.array!.item)) throw new DeviceEditError('Enter a reference or explicitly remove this row.', [[...path, index]])
  rows[index] = edit.value
  return next
}

export function extendedReferenceNotices(input: EditorInput, selection: Selection): { path: Path; message: string }[] {
  const device = cellInput(input, selection.cell), prefix: Path = selection.cell === 'device' ? [] : [selection.cell]
  const notices: { path: Path; message: string }[] = []
  for (const [index, grain] of (device.grain_boundaries ?? []).entries()) {
    if (device.grain_boundaries!.filter(item => item.id === grain.id).length > 1) notices.push({ path: [...prefix, 'grain_boundaries', index, 'id'], message: `Duplicate ID ${grain.id}` })
    for (const [offset, id] of grain.layer_ids.entries()) if (!device.layers.some(layer => layer.id === id && layer.role !== 'substrate') || grain.layer_ids.filter(value => value === id).length > 1) notices.push({
      path: [...prefix, 'grain_boundaries', index, 'layer_ids', offset], message: `${id} must reference one existing electrical layer without duplicates`,
    })
  }
  for (const [index, row] of (device.electrical_grid ?? []).entries()) if (!device.layers.some(layer => layer.id === row.layer && layer.role !== 'substrate') || device.electrical_grid!.filter(item => item.layer === row.layer).length > 1) notices.push({
    path: [...prefix, 'electrical_grid', index, 'layer'], message: `${row.layer} must reference one existing electrical layer exactly once`,
  })
  if (input.schema_version === 'solarlab.tandem-preparation.v1') {
    const rows = [...input.junction_stack.map((value, index) => ({ value, path: ['junction_stack', index] as Path })),
      ...(input.back_reflector ? [{ value: input.back_reflector, path: ['back_reflector'] as Path }] : [])]
    for (const row of rows) if (rows.filter(item => item.value.id === row.value.id).length > 1) notices.push({ path: [...row.path, 'id'], message: `Duplicate optical-layer ID ${row.value.id}` })
  }
  return notices
}
