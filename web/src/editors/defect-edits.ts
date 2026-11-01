import { configurationSchema } from '../generated/configuration-schema'
import type { BulkDefectInput } from '../generated/configuration-inputs'
import {
  allowsNull, cellInput, DeviceEditError, mutateRows, requireKnownIdentityScope, scalarField, selectedEntity,
} from './device-edits'
import type { DefectSelection, EditorInput, FieldSchema, FieldValue, LayerDocumentSelection, Path, RowMutation, Selection } from './device-edits'

const definitions = configurationSchema.dto_schemas.DeviceInput.schema.$defs

export interface DefectSection {
  key: string
  label: string
  path: Path
  schema: FieldSchema
  note: string
  optional?: 'energy_level' | 'spatial_profile'
  array?: { item: FieldSchema; kind: 'value' | 'kinetics' }
  row?: { arrayKey: string; index: number }
}

export type StructuredSelection = DefectSelection | LayerDocumentSelection

export function isDefectSelection(selection: Selection): selection is DefectSelection {
  return selection.kind === 'interface_defect' || selection.kind === 'bulk_defect' || selection.kind === 'metastable_defect'
}

export function isStructuredSelection(selection: Selection): selection is StructuredSelection {
  return isDefectSelection(selection) || selection.kind === 'metastable_document' || selection.kind === 'metastable_preparation'
}

export function nestedValue(owner: object | undefined, path: Path): object | undefined {
  let value: unknown = owner
  for (const key of path) value = value !== null && typeof value === 'object' ? Reflect.get(value, key) : undefined
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value : undefined
}

/** Explicit sections of the prepared defect DTOs; no recursive form compiler
 * or schema-default materialization. Unsupported branches remain in the draft. */
export function defectSections(input: EditorInput, selection: StructuredSelection): DefectSection[] {
  if (selection.kind === 'metastable_document') return [{ key: 'document', label: 'Metastable inventory declaration', path: [], schema: definitions.MetastableDocumentInput,
    note: 'This document declares a frozen metastable model and its inventory. It contains no prepared populations or simulated states.' }]
  if (selection.kind === 'metastable_preparation') return [
    { key: 'preparation', label: 'Metastable preparation declaration', path: [], schema: definitions.MetastablePreparationInput,
      note: 'These are preparation and measurement declarations only. No stationary state or measurement is evaluated by this editor.' },
    { key: 'numerics', label: 'Preparation numerical declarations', path: ['numerics'], schema: definitions.MetastableNumericsInput,
      note: 'Edit the supplied numerical declarations independently. No preparation or solver is run.' },
  ]
  if (selection.kind === 'interface_defect') return [
    { key: 'defect', label: 'Interface defect', path: [], schema: definitions.InterfaceDefectInput,
      note: 'Interface density is integrated per area (m^-2). Trap depth and its energy reference are supplied together.' },
    { key: 'kinetics', label: 'Capture kinetics', path: ['kinetics'], schema: definitions.KineticsInput,
      note: 'Supplied capture cross sections and thermal velocities. No kinetics branch or retained metadata is replaced.' },
  ]
  const defect = selectedEntity(input, selection).value!
  if (selection.kind === 'metastable_defect') return [
    { key: 'defect', label: 'Metastable defect', path: [], schema: definitions.MetastableDefectInput,
      note: 'Density is integrated per volume (m^-3). Conversion state indices refer to the independently ordered donor and acceptor arrays.' },
    ...multivalentSections(defect, ['donor_configuration'], 'donor:', 'Donor'),
    ...multivalentSections(defect, ['acceptor_configuration'], 'acceptor:', 'Acceptor'),
    { key: 'conversion', label: 'Conversion kinetics', path: ['conversion_kinetics'], schema: definitions.MetastableConversionInput,
      note: 'Choose the capture paths explicitly. Barrier and rate declarations remain independent; the resolver checks detailed balance.' },
  ]
  if (Object.hasOwn(defect, 'configuration') && !Object.hasOwn(defect, 'distribution')) {
    return [
      { key: 'defect', label: 'Multivalent bulk defect', path: [], schema: definitions.MultivalentDefectInput,
        note: 'Multivalent inventory retained in its supplied form. Density is integrated per volume (m^-3).' },
      ...multivalentSections(defect, ['configuration'], '', 'Multivalent'),
    ]
  }
  if (!Object.hasOwn(defect, 'distribution') || Object.hasOwn(defect, 'configuration')) return [
    { key: 'retained', label: 'Unsupported defect data retained', path: [], schema: {},
      note: 'This record is not an unambiguous supported defect shape. It remains unchanged for authoritative validation; no branch data is discarded.' },
  ]
  const profile = nestedValue(defect, ['spatial_profile']), knots = profile && Reflect.get(profile, 'knots')
  return [
    { key: 'defect', label: 'Bulk defect', path: [], schema: definitions.BulkDefectInput,
      note: 'Charge transition and neutral reference are separate declarations. Applying the draft checks their agreement.' },
    { key: 'distribution', label: 'Energy distribution and density', path: ['distribution'], schema: definitions.DistributionInput,
      note: 'Bulk density is integrated per volume (m^-3). Choose the Gaussian width convention explicitly; changing kind never clears width or support data.' },
    { key: 'energy_level', label: 'Referenced energy level', path: ['energy_level'], schema: definitions.EnergyLevelInput, optional: 'energy_level',
      note: 'Declare one energy authority: the distribution center or a referenced energy level. Switching authority requires explicit edits; conflicting fields are retained.' },
    { key: 'kinetics', label: 'Capture kinetics', path: ['kinetics'], schema: definitions.KineticsInput,
      note: 'Supplied capture cross sections and thermal velocities. Other branch data and tags are retained.' },
    { key: 'spatial_profile', label: 'Spatial profile', path: ['spatial_profile'], schema: definitions.SpatialProfileInput, optional: 'spatial_profile',
      note: 'Profile coordinates, interpolation and normalization are explicit. Knot edits are never sorted, integrated or normalized by the editor.' },
    ...(Array.isArray(knots) ? knots.map((_item, index) => ({ key: `knot:${index}`, label: `Spatial knot ${index + 1}`,
      path: ['spatial_profile', 'knots', index], schema: definitions.SpatialKnotInput,
      note: 'Position and multiplier are the supplied values. Invalid order, endpoints or layer average remain a draft for resolver validation.' })) : []),
  ]
}

function multivalentSections(defect: object, path: Path, prefix: string, label: string): DefectSection[] {
  const configuration = nestedValue(defect, path), kinetics = configuration && Reflect.get(configuration, 'transition_kinetics')
  const array = (key: string, title: string, relative: Path, schema: FieldSchema, kind: 'value' | 'kinetics'): DefectSection => ({
    key: prefix + key, label: `${label} ${title}`, path: [...path, ...relative], schema,
    array: { kind, item: schema.item_unit === undefined ? schema.items! : { ...schema.items, unit: schema.item_unit } },
    note: 'Rows are independent and ordered. Insert, remove or move only the chosen row; related arrays and conversion indices are not rewritten.',
  })
  return [
    { key: prefix + 'configuration', label: `${label} configuration`, path, schema: definitions.MultivalentConfigurationInput,
      note: 'Changing family or convention never regenerates charge states, degeneracies, energies or kinetics. The resolver checks consistency.' },
    array('charges', 'charge states', ['charge_states_e'], definitions.MultivalentConfigurationInput.properties.charge_states_e, 'value'),
    array('degeneracies', 'state degeneracies', ['state_degeneracies'], definitions.MultivalentConfigurationInput.properties.state_degeneracies, 'value'),
    { key: prefix + 'energy_levels', label: `${label} transition energies`, path: [...path, 'energy_levels'], schema: definitions.MultivalentEnergyInput,
      note: 'The first transition and ordered signed correlations are separate inputs. No cumulative energy is written back to the input.' },
    array('correlations', 'correlation energies', ['energy_levels', 'correlation_energies_eV'], definitions.MultivalentEnergyInput.properties.correlation_energies_eV, 'value'),
    array('transitions', 'transition kinetics rows', ['transition_kinetics'], definitions.MultivalentConfigurationInput.properties.transition_kinetics, 'kinetics'),
    ...(Array.isArray(kinetics) ? kinetics.map((_item, index) => ({ key: `${prefix}transition:${index}`, label: `${label} transition ${index + 1} kinetics`,
      path: [...path, 'transition_kinetics', index], schema: definitions.KineticsInput,
      row: { arrayKey: prefix + 'transitions', index },
      note: 'Edit this supplied transition only. Other rows, charge states and conversion indices remain unchanged.' })) : []),
  ]
}

export function defectArray(input: EditorInput, selection: StructuredSelection, key: string) {
  const section = defectSections(input, selection).find(item => item.key === key && item.array)
  if (!section) throw new DeviceEditError('Select a supported ordered defect array.', [selectedEntity(input, selection).path])
  const selected = selectedEntity(input, selection), owner = nestedValue(selected.value, section.path.slice(0, -1))
  const rows: unknown = owner && Reflect.get(owner, section.path.at(-1)!)
  if (!Array.isArray(rows)) throw new DeviceEditError('The selected array is not supplied.', [[...selected.path, ...section.path]])
  return { section, rows, path: [...selected.path, ...section.path] }
}

export function writeDefectRow<T extends EditorInput>(input: T, selection: StructuredSelection, key: string, index: number, edit: FieldValue): T {
  const next = structuredClone(input), array = defectArray(next, selection, key)
  if (array.section.array!.kind !== 'value' || !Number.isSafeInteger(index) || index < 0 || index >= array.rows.length) throw new DeviceEditError('Select a supplied scalar row.', [array.path])
  if (edit.kind === 'omit' || edit.value === null && !allowsNull(array.section.array!.item)) throw new DeviceEditError('Rows require an explicit value; remove a row explicitly.', [[...array.path, index]])
  array.rows[index] = edit.value
  return next
}

export function changeDefectRows<T extends EditorInput>(input: T, selection: StructuredSelection, key: string, action: RowMutation, value?: unknown): T {
  const next = structuredClone(input), array = defectArray(next, selection, key)
  mutateRows(array.rows, action, array.path, value)
  return next
}

export function writeDefectField<T extends EditorInput>(input: T, selection: StructuredSelection,
  sectionKey: string, field: string, edit: FieldValue): T {
  const section = defectSections(input, selection).find(item => item.key === sectionKey)
  const selected = selectedEntity(input, selection), metadata = section?.schema.properties?.[field]
  const path = [...selected.path, ...(section?.path ?? []), field]
  if (!section || !metadata || !scalarField(metadata) || field === 'id') {
    throw new DeviceEditError('This field has no scalar defect editor.', [path])
  }
  if (edit.kind === 'omit' && section.schema.required?.includes(field)) throw new DeviceEditError('This required field cannot be omitted.', [path])
  if (edit.kind === 'value' && edit.value === null && !allowsNull(metadata)) throw new DeviceEditError('This field does not accept null.', [path])
  const next = structuredClone(input), owner = nestedValue(selectedEntity(next, selection).value, section.path)
  if (!owner) throw new DeviceEditError('The selected defect section is not supplied.', [[...selected.path, ...section.path]])
  if (edit.kind === 'omit') Reflect.deleteProperty(owner, field)
  else Reflect.set(owner, field, edit.value)
  return next
}

export function renameDefect<T extends EditorInput>(input: T, selection: DefectSelection, id: string) {
  const found = selectedEntity(input, selection), next = structuredClone(input)
  if (id !== selection.id) requireKnownIdentityScope(input, selection.cell)
  const updatedPaths: Path[] = []
  if (selection.kind === 'metastable_defect') {
    const parent = selectedEntity(next, { cell: selection.cell, kind: 'layer', ...selection.parent })
    const defects = cellInput(next, selection.cell).layers[parent.index].metastable_document!.metastable_defects
    defects[found.index].id = id
    return { input: next, selection: { ...selection, id, occurrence: defects.slice(0, found.index).filter(item => item.id === id).length },
      updatedPaths: id === selection.id ? [] : [[...found.path, 'id']] }
  }
  if (selection.kind === 'bulk_defect') {
    const parent = selectedEntity(next, { cell: selection.cell, kind: 'layer', ...selection.parent })
    const layer = cellInput(next, selection.cell).layers[parent.index], defects = layer.bulk_defects!
    const references = (layer.scaps_defect_metadata ?? []).flatMap((item, index) => item.defect_id === selection.id ? [index] : [])
    if (id !== selection.id && references.length && defects.filter(item => item.id === selection.id).length !== 1) {
      throw new DeviceEditError('The defect ID has ambiguous partner references; edit or discard the duplicate draft explicitly.',
        references.map(index => [...parent.path, 'scaps_defect_metadata', index, 'defect_id']))
    }
    if (id !== selection.id) {
      defects[found.index].id = id; updatedPaths.push([...found.path, 'id'])
      for (const index of references) { layer.scaps_defect_metadata![index].defect_id = id; updatedPaths.push([...parent.path, 'scaps_defect_metadata', index, 'defect_id']) }
    }
    return { input: next, selection: { ...selection, id, occurrence: defects.slice(0, found.index).filter(item => item.id === id).length }, updatedPaths }
  }
  const defect = cellInput(next, selection.cell).interfaces![found.index].defect!
  if (id !== selection.id) {
    defect.id = id; updatedPaths.push([...found.path, 'id'])
    if (defect.partner_metadata?.defect_id === selection.id) {
      defect.partner_metadata.defect_id = id; updatedPaths.push([...found.path, 'partner_metadata', 'defect_id'])
    }
  }
  return { input: next, selection: { ...selection, id }, updatedPaths }
}

/** Optional object creation starts empty: required declarations must be entered
 * explicitly. Clear and omission never stand in for one another. */
export function setDefectSection<T extends EditorInput>(input: T, selection: DefectSelection,
  sectionKey: string, mode: 'empty' | 'null' | 'omit'): T {
  const section = defectSections(input, selection).find(item => item.key === sectionKey)
  const selected = selectedEntity(input, selection)
  if (!section?.optional || selection.kind !== 'bulk_defect') throw new DeviceEditError('Select an optional defect section.', [selected.path])
  const next = structuredClone(input), owner = selectedEntity(next, selection).value!
  if (mode === 'empty') {
    if (nestedValue(owner, [section.optional])) throw new DeviceEditError('This section is already supplied; edit it or explicitly clear it first.', [[...selected.path, section.optional]])
    Reflect.set(owner, section.optional, section.optional === 'spatial_profile' ? { knots: [] } : {})
  } else if (mode === 'null') Reflect.set(owner, section.optional, null)
  else Reflect.deleteProperty(owner, section.optional)
  return next
}

type Knot = NonNullable<BulkDefectInput['spatial_profile']>['knots'][number]
export function changeProfileRows<T extends EditorInput>(input: T, selection: DefectSelection,
  action: { kind: 'append'; knot: Knot } | { kind: 'remove'; index: number }): T {
  const next = structuredClone(input), selected = selectedEntity(next, selection)
  if (selection.kind !== 'bulk_defect' || !defectSections(input, selection).some(item => item.key === 'spatial_profile')) throw new DeviceEditError('Select an ordinary bulk-defect profile.', [selected.path])
  const profile = nestedValue(selected.value, ['spatial_profile']), knots = profile && Reflect.get(profile, 'knots')
  if (!Array.isArray(knots)) throw new DeviceEditError('A supplied spatial-profile knot list is required.', [[...selected.path, 'spatial_profile', 'knots']])
  if (action.kind === 'append') knots.push(structuredClone(action.knot))
  else {
    if (!Number.isSafeInteger(action.index) || action.index < 0 || action.index >= knots.length) throw new DeviceEditError('The selected knot is no longer supplied.', [[...selected.path, 'spatial_profile', 'knots', action.index]])
    knots.splice(action.index, 1)
  }
  return next
}

export function removeBulkDefect<T extends EditorInput>(input: T, selection: DefectSelection): T {
  const next = structuredClone(input)
  if (selection.kind !== 'bulk_defect') throw new DeviceEditError('Select a supplied bulk defect.', [[selection.cell]])
  const found = selectedEntity(next, selection), parent = selectedEntity(next, { cell: selection.cell, kind: 'layer', ...selection.parent })
  // Preserve metadata references; a dangling partner reference is an explicit
  // invalid draft, never silently deleted to make validation succeed.
  cellInput(next, selection.cell).layers[parent.index].bulk_defects!.splice(found.index, 1)
  return next
}

/** Clear/remove only by an explicit user action; no automatic inventory repair. */
export function clearInterfaceDefect<T extends EditorInput>(input: T, selection: Selection, mode: 'null' | 'omit'): T {
  if (selection.kind !== 'interface') throw new DeviceEditError('Select an interface.', [[selection.cell, 'interfaces']])
  const next = structuredClone(input), found = selectedEntity(next, selection)
  const item = cellInput(next, selection.cell).interfaces![found.index]
  if (mode === 'null') item.defect = null
  else delete item.defect
  return next
}
