import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { configurationSchemaBytes, webcrypto } from '../test-helpers/schema-fixture.mjs'
import {
  configurationSchema,
  configurationSchemaSha256,
} from '../generated/configuration-schema'
import type {
  DeviceInput,
  SweepInput,
  TandemInput,
} from '../generated/configuration-inputs'
import {
  previewDeviceConfiguration,
  previewTandemConfiguration,
} from '../configuration-client'
import {
  collectReferenceDiagnostics,
  exactInputEqual,
  moveLayer,
  renameConnection,
  renameLayer,
  renameMaterial,
  writeField,
} from '../editors/device-edits'
import {
  changeProfileRows,
  removeBulkDefect,
  renameDefect,
  setDefectSection,
  writeDefectField,
} from '../editors/defect-edits'
import {
  changeExtendedRows,
  setExtendedSection,
  writeExtendedField,
} from '../editors/extended-edits'
import { mountDeviceEditor } from '../editors/device-editor'
import { collectSweepReferenceDiagnostics, mountSweepEditor } from '../editors/sweep'
import { mountConfigurationWorkstation } from '../panels/configuration-workstation'
import {
  renderHandshakeBanner,
  verifyWorkstationHandshake,
} from './schema-handshake'

let SCHEMA_JSON_BYTES: ArrayBuffer
const DUMMY_HASH_A = 'a'.repeat(64)
const DUMMY_HASH_B = 'b'.repeat(64)
const DUMMY_HASH_C = 'c'.repeat(64)

async function sha256Hex(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(text)
  const digest = await webcrypto.subtle.digest('SHA-256', bytes)
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('')
}

beforeAll(() => {
  SCHEMA_JSON_BYTES = configurationSchemaBytes()
})

beforeEach(() => {
  vi.stubGlobal('crypto', webcrypto)
})

async function buildCatalogResponse(overrides?: {
  registryHeader?: string
  jobSchemaHeader?: string
  etag?: string
  bodyJobSchema?: string
  status?: number
}): Promise<Response> {
  const status = overrides?.status ?? 200
  if (status !== 200) {
    return new Response(JSON.stringify({ detail: 'mismatch' }), {
      status,
      headers: { 'Content-Type': 'application/json' },
    })
  }
  const jobSchemaSha = overrides?.bodyJobSchema ?? DUMMY_HASH_B
  const payload = JSON.stringify({
    schema: 'solarlab.execution-catalog.v1',
    job_schema_sha256: jobSchemaSha,
    model_registry_sha256: DUMMY_HASH_C,
    bindings: [
      { experiment_id: 'jv', solver_family: 'steady_1d' },
      { experiment_id: 'tpv', solver_family: 'transient_1d' },
    ],
  })
  const computedDigest = await sha256Hex(payload)
  const registryHeader = overrides?.registryHeader ?? computedDigest
  const jobSchemaHeader = overrides?.jobSchemaHeader ?? jobSchemaSha
  const etag = overrides?.etag ?? `"${registryHeader}"`
  return new Response(payload, {
    status: 200,
    headers: {
      'Content-Type': 'application/json',
      'X-Solarlab-Registry': registryHeader,
      'X-Solarlab-Job-Schema': jobSchemaHeader,
      ETag: etag,
    },
  })
}

afterEach(() => {
  vi.unstubAllGlobals()
  document.body.replaceChildren()
})

describe('P06-07: startup schema/registry digest handshake', () => {
  it('verifies matching /configuration-schema and /experiments SHA-256 digests and hides alert banner', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: string | URL) => {
        const url = String(input)
        if (url.includes('configuration-schema')) {
          return new Response(SCHEMA_JSON_BYTES, {
            status: 200,
            headers: {
              'Content-Type': 'application/json',
              'X-Solarlab-Configuration-Schema': configurationSchemaSha256,
            },
          })
        }
        return buildCatalogResponse()
      }),
    )

    const result = await verifyWorkstationHandshake({
      schemaEndpoint: 'http://test.local/configuration-schema',
      experimentsEndpoint: 'http://test.local/experiments',
    })
    expect(result.status).toBe('verified')
    if (result.status === 'verified') {
      expect(result.configurationSchemaSha256).toBe(configurationSchemaSha256)
      expect(result.metadata.schema_version).toBe(configurationSchema.schema_version)
      expect(result.catalog?.bindingsCount).toBe(2)
      expect(result.catalog?.bindingIds).toEqual(['jv', 'tpv'])
    }

    const bannerHost = document.createElement('div')
    renderHandshakeBanner(bannerHost, result)
    expect(bannerHost.hidden).toBe(true)
    expect(bannerHost.querySelector('[data-test="schema-handshake-alert"]')).toBeNull()
  })

  it('surfaces an explicit [role="alert"] banner when /configuration-schema digest disagrees with client DTOs', async () => {
    const tamperedBytes = new TextEncoder().encode(
      JSON.stringify({ ...configurationSchema, tampered_extra: true }),
    )
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: string | URL) => {
        const url = String(input)
        if (url.includes('configuration-schema')) {
          return new Response(tamperedBytes, {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          })
        }
        return buildCatalogResponse()
      }),
    )

    const result = await verifyWorkstationHandshake({
      schemaEndpoint: 'http://test.local/configuration-schema',
      experimentsEndpoint: 'http://test.local/experiments',
    })
    expect(result.status).toBe('mismatch')
    if (result.status === 'mismatch') {
      expect(result.code).toBe('identity_mismatch')
    }

    const bannerHost = document.createElement('div')
    renderHandshakeBanner(bannerHost, result)
    expect(bannerHost.hidden).toBe(false)
    const alert = bannerHost.querySelector<HTMLElement>('[data-test="schema-handshake-alert"]')
    expect(alert).not.toBeNull()
    expect(alert?.getAttribute('role')).toBe('alert')
    expect(alert?.textContent).toContain('Schema / registry version mismatch')
    expect(alert?.textContent).toContain('Silent field dropping or coercion is refused')
  })

  it('surfaces an explicit [role="alert"] banner when /experiments X-Solarlab-Registry or X-Solarlab-Job-Schema mismatches', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: string | URL) => {
        const url = String(input)
        if (url.includes('configuration-schema')) {
          return new Response(SCHEMA_JSON_BYTES, {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          })
        }
        return buildCatalogResponse({ registryHeader: DUMMY_HASH_A })
      }),
    )

    const result = await verifyWorkstationHandshake({
      schemaEndpoint: 'http://test.local/configuration-schema',
      experimentsEndpoint: 'http://test.local/experiments',
    })
    expect(result.status).toBe('mismatch')
    if (result.status === 'mismatch') {
      expect(result.code).toBe('catalog_mismatch')
    }

    const bannerHost = document.createElement('div')
    renderHandshakeBanner(bannerHost, result)
    const alert = bannerHost.querySelector<HTMLElement>('[data-test="schema-handshake-alert"]')
    expect(alert?.getAttribute('role')).toBe('alert')
    expect(alert?.textContent).toContain('X-Solarlab-Registry')
  })
})

describe('P06-07: stable-ID add/remove/rename/reorder and dangling-reference diagnostics', () => {
  function sampleDevice(): DeviceInput {
    return {
      schema_version: 'solarlab.device-preparation.v1',
      id: 'dev-stable',
      source_format: 'canonical',
      materials: [
        { id: 'mat_pvk', name: 'MAPbI3', parameters: { Eg: '1.6 eV' } },
      ],
      layers: [
        { id: 'htl', name: 'HTL', role: 'HTL', thickness: '40 nm' },
        {
          id: 'pvk',
          name: 'Absorber',
          role: 'absorber',
          thickness: '500 nm',
          material: 'mat_pvk',
          bulk_defects: [
            {
              id: 'trap_1',
              charge_transition: 'acceptor',
              neutral_reference: 'empty',
              degeneracy: 1,
              distribution: {
                kind: 'single_level',
                normalization: 'integrated_total',
                total_density_m3: '1e21 m^-3',
              },
              energy_level: { reference: 'below_conduction_band', value_eV: '0.8 eV' },
              kinetics: {
                sigma_n_m2: '1e-19 m^2',
                sigma_p_m2: '1e-19 m^2',
                thermal_velocity_n_m_s: '1e5 m/s',
                thermal_velocity_p_m_s: '1e5 m/s',
              },
            },
          ],
          scaps_defect_metadata: [
            {
              defect_id: 'trap_1',
              distribution: 'single',
              energy_reference: 'below_conduction_band',
              trap_depth_eV: '0.8 eV',
            },
          ],
        },
        { id: 'etl', name: 'ETL', role: 'ETL', thickness: '30 nm' },
      ],
      interfaces: [
        {
          id: 'if_htl_pvk',
          left: 'htl',
          right: 'pvk',
          defect: {
            id: 'if_trap_1',
            calibration_factor: 1,
            iface_state_calibration_factor: 1,
            energy_reference: 'below_conduction_band',
            total_density_m2: '1e14 m^-2',
            trap_depth_eV: '0.3 eV',
            kinetics: {
              sigma_n_m2: '1e-19 m^2',
              sigma_p_m2: '1e-19 m^2',
              thermal_velocity_n_m_s: '1e5 m/s',
              thermal_velocity_p_m_s: '1e5 m/s',
            },
            partner_metadata: {
              defect_id: 'if_trap_1',
              distribution: 'single',
              energy_reference: 'below_conduction_band',
              trap_depth_eV: '0.3 eV',
            },
          },
        },
        { id: 'if_pvk_etl', left: 'pvk', right: 'etl' },
      ],
      contacts: [
        { id: 'c_left', side: 'left', layer: 'htl' },
        { id: 'c_right', side: 'right', layer: 'etl' },
      ],
      grain_boundaries: [
        {
          id: 'gb_main',
          x_position: '250 nm',
          width: '10 nm',
          tau_n: '1e-9 s',
          tau_p: '1e-9 s',
          layer_ids: ['pvk'],
        },
      ],
      electrical_grid: [
        { layer: 'htl', interval_weight: 1, alpha: 1 },
        { layer: 'pvk', interval_weight: 4, alpha: 1 },
        { layer: 'etl', interval_weight: 1, alpha: 1 },
      ],
    }
  }

  it('propagates layer, material, interface, contact, and defect renames across typed references cleanly', () => {
    let doc = sampleDevice()
    expect(collectReferenceDiagnostics(doc)).toEqual([])

    doc = renameLayer(doc, { cell: 'device', kind: 'layer', id: 'pvk', occurrence: 0 }, 'absorber_main').input
    expect(doc.interfaces?.[0].right).toBe('absorber_main')
    expect(doc.interfaces?.[1].left).toBe('absorber_main')
    expect(doc.grain_boundaries?.[0].layer_ids).toEqual(['absorber_main'])
    expect(doc.electrical_grid?.[1].layer).toBe('absorber_main')
    expect(collectReferenceDiagnostics(doc)).toEqual([])

    doc = renameMaterial(doc, { cell: 'device', kind: 'material', id: 'mat_pvk', occurrence: 0 }, 'mat_fapbi3').input
    expect(doc.layers[1].material).toBe('mat_fapbi3')

    doc = renameConnection(doc, { cell: 'device', kind: 'interface', id: 'if_htl_pvk', occurrence: 0 }, 'if_front').input
    doc = renameConnection(doc, { cell: 'device', kind: 'contact', id: 'c_left', occurrence: 0 }, 'anode').input
    expect(doc.interfaces?.[0].id).toBe('if_front')
    expect(doc.contacts?.[0].id).toBe('anode')

    doc = renameDefect(
      doc,
      {
        cell: 'device',
        kind: 'bulk_defect',
        parent: { id: 'absorber_main', occurrence: 0 },
        id: 'trap_1',
        occurrence: 0,
      },
      'deep_acceptor',
    ).input
    expect(doc.layers[1].scaps_defect_metadata?.[0].defect_id).toBe('deep_acceptor')

    doc = renameDefect(
      doc,
      {
        cell: 'device',
        kind: 'interface_defect',
        parent: { id: 'if_front', occurrence: 0 },
        id: 'if_trap_1',
      },
      'front_recombination_center',
    ).input
    expect(doc.interfaces?.[0].defect?.partner_metadata?.defect_id).toBe('front_recombination_center')
    expect(collectReferenceDiagnostics(doc)).toEqual([])
  })

  it('detects dangling references, adjacency breakage on layer reorder, duplicate material IDs, duplicate contact sides, and dangling sweep targets', () => {
    const base = sampleDevice()
    const reordered = moveLayer(base, { cell: 'device', kind: 'layer', id: 'pvk', occurrence: 0 }, 1).input
    const reorderDiag = collectReferenceDiagnostics(reordered)
    expect(reorderDiag.some(d => d.code === 'topology_order')).toBe(true)

    const removedDefect = removeBulkDefect(base, {
      cell: 'device',
      kind: 'bulk_defect',
      parent: { id: 'pvk', occurrence: 0 },
      id: 'trap_1',
      occurrence: 0,
    })
    const defectDiag = collectReferenceDiagnostics(removedDefect)
    expect(
      defectDiag.some(
        d => d.code === 'dangling_reference' && d.message.includes('scaps_defect_metadata'),
      ),
    ).toBe(true)

    const brokenDevice: DeviceInput = {
      ...base,
      materials: [
        { id: 'dup_mat', name: 'M1', parameters: {} },
        { id: 'dup_mat', name: 'M2', parameters: {} },
      ],
      layers: base.layers.map((l, idx) => (idx === 0 ? { ...l, material: 'missing_mat' } : l)),
      contacts: [
        { id: 'c1', side: 'left', layer: 'htl' },
        { id: 'c2', side: 'left', layer: 'ghost_layer' },
      ],
    }
    const brokenDiag = collectReferenceDiagnostics(brokenDevice)
    expect(brokenDiag.some(d => d.code === 'duplicate_id' && d.message.includes('dup_mat'))).toBe(true)
    expect(brokenDiag.some(d => d.code === 'dangling_reference' && d.message.includes('missing_mat'))).toBe(true)
    expect(brokenDiag.some(d => d.code === 'duplicate_contact_side')).toBe(true)
    expect(brokenDiag.some(d => d.code === 'dangling_reference' && d.message.includes('ghost_layer'))).toBe(true)

    const editorHost = document.createElement('div')
    document.body.appendChild(editorHost)
    const editor = mountDeviceEditor(editorHost, { input: brokenDevice, onApply: () => {} })
    const noticesText = editorHost.querySelector('[data-role="reference-notices"]')?.textContent ?? ''
    expect(noticesText).toContain('dup_mat')
    expect(noticesText).toContain('duplicate left contact')
    editor.dispose()

    const sweepInput: SweepInput = {
      schema_version: 'solarlab.sweep-preparation.v1',
      id: 'sweep-dangling',
      base: sampleDevice(),
      axes: [
        {
          id: 'ax_1',
          target: { family: 'layer_parameter', owner_id: 'deleted_layer', parameter: ['mu_n'] },
          coordinates: [{ kind: 'omit' }],
        },
      ],
    }
    expect(collectSweepReferenceDiagnostics(sweepInput)).toHaveLength(1)
    const sweepHost = document.createElement('div')
    document.body.appendChild(sweepHost)
    const sweepEditor = mountSweepEditor(sweepHost, { input: sweepInput, onApply: () => {} })
    const sweepNotice = sweepHost.querySelector('[data-role="sweep-reference-notices"]')?.textContent ?? ''
    expect(sweepNotice).toContain('deleted_layer')
    sweepEditor.dispose()
  })
})

describe('P06-07: lossless round-trip across all 4 special editors (untouched defaults, explicit null, units, -0.0)', () => {
  it('round-trips spatial defect profiles, composition grading, n-k optical curves, and 2T tandem stacks without coercing defaults, null, units, or -0.0', async () => {
    const capturedBodies: Array<{ url: string; body: string }> = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: string | URL, init?: RequestInit) => {
        const url = String(input)
        const body = String(init?.body ?? '')
        capturedBodies.push({ url, body })
        const isTandem = url.includes('/tandem')
        const kind = isTandem ? 'tandem' : 'device'
        const responseJson =
          `{"schema":"solarlab.configuration-preview.v1","kind":"${kind}",` +
          `"status":"prepared_pending_dependencies","can_execute":false,` +
          `"input":${body},` +
          `"resolved":{"schema":"solarlab.resolved-${kind}-preparation.v1","id":${JSON.stringify(isTandem ? 'tandem-special' : 'device-special')},"can_execute":false},` +
          `"identity":{"scope":"configuration_content","content_sha256":"${DUMMY_HASH_A}","default_catalog_sha256":"${DUMMY_HASH_B}","resource_library_sha256":"${DUMMY_HASH_C}"}}`
        return new Response(responseJson, {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      }),
    )

    // 1 + 2 + 3: Spatial defect profile + Composition grading + n-k optical resources on DeviceInput
    let device: DeviceInput = {
      schema_version: 'solarlab.device-preparation.v1',
      id: 'device-special',
      source_format: 'canonical',
      spectrum: 'AM1.5G',
      layers: [
        {
          id: 'cigs_abs',
          name: 'Graded CIGS',
          role: 'absorber',
          thickness: '2.0 um',
          parameters: {
            optical_material: 'CIGS_nk_table',
            n_optical: 2.85,
            incoherent: false,
            Eg: '1.15 eV',
            Eg_back: '1.35 eV',
            chi: '4.5 eV',
            chi_back: '4.3 eV',
            grading_profile: 'linear',
          },
          cigs_graded_optics: null,
          bulk_defects: [
            {
              id: 'graded_trap',
              charge_transition: 'acceptor',
              neutral_reference: 'empty',
              degeneracy: 1,
              distribution: {
                kind: 'gaussian',
                normalization: 'integrated_total',
                total_density_m3: '2.5e21 m^-3',
                center_eV_above_vb: '0.65 eV',
                width_eV: '0.05 eV',
                width_convention: 'gaussian_standard_deviation',
              },
              kinetics: {
                sigma_n_m2: '1e-19 m^2',
                sigma_p_m2: '5e-20 m^2',
                thermal_velocity_n_m_s: '1e5 m/s',
                thermal_velocity_p_m_s: '1e5 m/s',
              },
              spatial_profile: {
                coordinate: 'normalized_layer_coordinate',
                density_normalization: 'layer_average_unity',
                interpolation: 'piecewise_linear',
                knots: [
                  { position_fraction: -0.0, density_multiplier: 2.0 },
                  { position_fraction: 1.0, density_multiplier: 0.5 },
                ],
              },
            },
          ],
        },
      ],
    }

    // Mutate via the special editor helpers and verify untouched defaults stay omitted
    const defectSel = {
      cell: 'device' as const,
      kind: 'bulk_defect' as const,
      parent: { id: 'cigs_abs', occurrence: 0 },
      id: 'graded_trap',
      occurrence: 0,
    }
    device = changeProfileRows(device, defectSel, {
      kind: 'append',
      knot: { position_fraction: 0.5, density_multiplier: 1.25 },
    })
    device = writeDefectField(device, defectSel, 'spatial_profile', 'interpolation', {
      kind: 'value',
      value: 'piecewise_linear',
    })
    device = setDefectSection(device, defectSel, 'energy_level', 'null')

    // Toggle layer_optics cigs_graded_optics from explicit null -> omitted -> explicit object -> explicit null
    const opticsSel = { cell: 'device' as const, kind: 'layer_optics' as const, id: 'cigs_abs', occurrence: 0 }
    const omittedOptics = setExtendedSection(device, opticsSel, 'cigs', 'omit')
    expect(Object.hasOwn(omittedOptics.layers[0], 'cigs_graded_optics')).toBe(false)
    const emptyOptics = setExtendedSection(omittedOptics, opticsSel, 'cigs', 'empty')
    const editedOptics = writeExtendedField(emptyOptics, opticsSel, 'cigs', 'ggi_front', {
      kind: 'value',
      value: -0.0,
    })
    expect(Object.is(editedOptics.layers[0].cigs_graded_optics?.ggi_front, -0)).toBe(true)
    const restoredNullOptics = setExtendedSection(editedOptics, opticsSel, 'cigs', 'null')
    expect(restoredNullOptics.layers[0].cigs_graded_optics).toBeNull()

    // Edit optical curve resource via resources & layer parameters
    device = writeExtendedField(
      restoredNullOptics,
      { cell: 'device', kind: 'resources' },
      'resources',
      'spectrum',
      { kind: 'value', value: 'AM1.5G_tabular_nk' },
    )
    device = writeField(
      device,
      { cell: 'device', kind: 'layer', id: 'cigs_abs', occurrence: 0 },
      'Eg_back',
      { kind: 'value', value: '1.40 eV' },
      true,
    )

    const devicePreview = await previewDeviceConfiguration(
      'http://test.local/configuration/preview/device',
      device,
    )
    expect(capturedBodies[0].body).toContain('"position_fraction":-0.0')
    expect(capturedBodies[0].body).toContain('"cigs_graded_optics":null')
    expect(capturedBodies[0].body).toContain('"energy_level":null')
    expect(capturedBodies[0].body).toContain('"Eg_back":"1.40 eV"')
    expect(capturedBodies[0].body).toContain('"optical_material":"CIGS_nk_table"')
    expect(capturedBodies[0].body).not.toContain('"contacts"')
    expect(exactInputEqual(devicePreview.value.input, device)).toBe(true)

    // 4: 2T Tandem stack special editor round-trip
    let tandem: TandemInput = {
      schema_version: 'solarlab.tandem-preparation.v1',
      id: 'tandem-special',
      source_schema_version: 1,
      device_type: 'tandem_2T_monolithic',
      top_cell: {
        schema_version: 'solarlab.device-preparation.v1',
        id: 'top-pvk',
        source_format: 'canonical',
        settings: { V_bi: -0.0 },
        layers: [{ id: 'top_abs', name: 'Wide-Eg PVK', role: 'absorber', thickness: '400 nm' }],
      },
      bottom_cell: device,
      top_cell_reference: 'top.yaml',
      bottom_cell_reference: 'bot.yaml',
      junction_model: 'ideal_ohmic',
      light_direction: 'top_first',
      junction_stack: [
        {
          id: 'iz_ito',
          name: 'ITO Recombination Layer',
          thickness: '15 nm',
          optical_material: 'ITO_nk',
          incoherent: false,
        },
      ],
      benchmark: null,
      back_reflector: null,
    }

    const stackSel = { cell: 'top_cell' as const, kind: 'optical_stack' as const }
    tandem = changeExtendedRows(tandem, stackSel, 'junction', { kind: 'insert', index: 1 }, 0)
    tandem.junction_stack[1].id = 'iz_sn_ox'
    tandem = writeExtendedField(tandem, stackSel, 'optical:1', 'thickness', {
      kind: 'value',
      value: '10 nm',
    })

    const tandemPreview = await previewTandemConfiguration(
      'http://test.local/configuration/preview/tandem',
      tandem,
    )
    expect(capturedBodies[1].body).toContain('"V_bi":-0.0')
    expect(capturedBodies[1].body).toContain('"back_reflector":null')
    expect(capturedBodies[1].body).toContain('"benchmark":null')
    expect(capturedBodies[1].body).toContain('"optical_material":"ITO_nk"')
    expect(exactInputEqual(tandemPreview.value.input, tandem)).toBe(true)
  })

  it('mounts ConfigurationWorkstation across all 5 preparation modes and blocks preview on schema mismatch', async () => {
    const previewCalls: string[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: string | URL) => {
        const url = String(input)
        if (url.includes('configuration-schema')) {
          return new Response(new TextEncoder().encode('{"schema_version":"tampered"}'), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          })
        }
        if (url.includes('/experiments')) {
          return buildCatalogResponse()
        }
        previewCalls.push(url)
        return new Response('{}', { status: 200 })
      }),
    )

    const root = document.createElement('div')
    document.body.appendChild(root)
    const ws = mountConfigurationWorkstation(root, new URL('http://test.local/'))
    const handshake = await ws.ready
    expect(handshake.status).toBe('mismatch')
    expect(root.querySelector('[data-test="schema-handshake-alert"]')).not.toBeNull()

    for (const mode of ['tandem', 'jv', 'spatial', 'sweep', 'device'] as const) {
      ws.selectMode(mode)
      expect(
        root.querySelector<HTMLSelectElement>('[data-role="configuration-mode"]')?.value,
      ).toBe(mode)
    }

    const applyBtn = root.querySelector<HTMLButtonElement>('[data-action="apply-device"]')
    applyBtn?.click()
    expect(previewCalls).toEqual([])
    ws.dispose()
  })
})
