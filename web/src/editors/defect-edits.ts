import { configurationSchema } from '../generated/configuration-schema'
import type { BulkDefectInput } from '../generated/configuration-inputs'
import {
  allowsNull, cellInput, DeviceEditError, requireKnownIdentityScope, scalarField, selectedEntity,
} from './device-edits'
import type { DefectSelection, EditorInput, FieldSchema, FieldValue, Path, Selection } from './device-edits'

const definitions = configurationSchema.dto_schemas.DeviceInput.schema.$defs

export interface DefectSection {
  key: string
  label: string
  path: Path
  schema: FieldSchema
  note: string
  optional?: 'energy_level' | 'spatial_profile'
}

export function isDefectSelection(selection: Selection): selection is DefectSelection {
  return selection.kind === 'interface_defect' || selection.kind === 'bulk_defect'
}

export function nestedValue(owner: object | undefined, path: Path): object | undefined {
  let value: unknown = owner
  for (const key of path) value = value !== null && typeof value === 'object' ? Reflect.get(value, key) : undefined
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value : undefined
}

/** Explicit sections of the prepared defect DTOs; no recursive form compiler
 * or schema-default materialization. Unsupported branches remain in the draft. */
export function defectSections(input: EditorInput, selection: DefectSelection): DefectSection[] {
  if (selection.kind === 'interface_defect') return [
    { key: 'defect', label: 'Interface defect', path: [], schema: definitions.InterfaceDefectInput,
      note: 'Interface density is integrated per area (m^-2). Trap depth and its energy reference are supplied together.' },
    { key: 'kinetics', label: 'Capture kinetics', path: ['kinetics'], schema: definitions.KineticsInput,
      note: 'Supplied capture cross sections and thermal velocities. No kinetics branch or retained metadata is replaced.' },
  ]
  const defect = selectedEntity(input, selection).value!
  if (Object.hasOwn(defect, 'configuration') && !Object.hasOwn(defect, 'distribution')) {
    const configuration = nestedValue(defect, ['configuration']), kinetics = configuration && Reflect.get(configuration, 'transition_kinetics')
    return [
      { key: 'defect', label: 'Multivalent bulk defect', path: [], schema: definitions.MultivalentDefectInput,
        note: 'Multivalent inventory retained in its supplied form. Density is integrated per volume (m^-3).' },
      { key: 'configuration', label: 'Multivalent configuration', path: ['configuration'], schema: definitions.MultivalentConfigurationInput,
        note: 'Changing family or convention does not regenerate charge states, degeneracies or transition data. The resolver checks consistency.' },
      { key: 'energy_levels', label: 'Transition energies', path: ['configuration', 'energy_levels'], schema: definitions.MultivalentEnergyInput,
        note: 'The supplied correlation-energy array and ordered transition convention are retained.' },
      ...(Array.isArray(kinetics) ? kinetics.map((_item, index) => ({ key: `transition:${index}`, label: `Transition ${index + 1} kinetics`,
        path: ['configuration', 'transition_kinetics', index], schema: definitions.KineticsInput,
        note: 'Edit this supplied transition only. Other transitions and configuration fields are retained.' })) : []),
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

export function writeDefectField<T extends EditorInput>(input: T, selection: DefectSelection,
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
