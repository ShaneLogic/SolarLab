/**
 * V2 Configuration Preparation & Lossless DTO Editor Surface (P06-07).
 *
 * Connects the generated schema digest handshake (`verifyWorkstationHandshake`)
 * with the stable-ID lossless editors (`mountDeviceEditor`, `mountJVExperimentEditor`,
 * `mountSpatialExperimentEditor`, `mountSweepEditor`) and `mountConfigurationPreviewPanel`.
 */
import type {
  DeviceInput,
  JVExperimentInput,
  SpatialExperimentInput,
  SweepInput,
  TandemInput,
} from '../generated/configuration-inputs'
import { mountDeviceEditor } from '../editors/device-editor'
import { mountJVExperimentEditor } from '../editors/jv-experiment'
import { mountSpatialExperimentEditor } from '../editors/spatial-experiment'
import { mountSweepEditor } from '../editors/sweep'
import { mountConfigurationPreviewPanel, type ConfigurationPreviewPanel } from './configuration-preview'
import {
  renderHandshakeBanner,
  verifyWorkstationHandshake,
  type WorkstationHandshakeOptions,
  type WorkstationHandshakeResult,
} from '../workstation/schema-handshake'

export type ConfigurationEditorMode = 'device' | 'tandem' | 'jv' | 'spatial' | 'sweep'

export interface ConfigurationWorkstationHandle {
  readonly ready: Promise<WorkstationHandshakeResult>
  selectMode(mode: ConfigurationEditorMode): void
  dispose(): void
}

function defaultDeviceInput(id = 'device-v2'): DeviceInput {
  return {
    schema_version: 'solarlab.device-preparation.v1',
    id,
    source_format: 'canonical',
    layers: [
      { id: 'htl', name: 'HTL', role: 'HTL', thickness: '40 nm' },
      { id: 'absorber', name: 'Absorber', role: 'absorber', thickness: '500 nm' },
      { id: 'etl', name: 'ETL', role: 'ETL', thickness: '30 nm' },
    ],
    interfaces: [
      { id: 'htl_absorber', left: 'htl', right: 'absorber' },
      { id: 'absorber_etl', left: 'absorber', right: 'etl' },
    ],
    contacts: [
      { id: 'front_contact', side: 'left', layer: 'htl' },
      { id: 'back_contact', side: 'right', layer: 'etl' },
    ],
  }
}

function defaultTandemInput(): TandemInput {
  return {
    schema_version: 'solarlab.tandem-preparation.v1',
    id: 'tandem-v2',
    source_schema_version: 1,
    device_type: 'tandem_2T_monolithic',
    top_cell: defaultDeviceInput('top-cell'),
    bottom_cell: defaultDeviceInput('bottom-cell'),
    top_cell_reference: 'top_cell.yaml',
    bottom_cell_reference: 'bottom_cell.yaml',
    junction_model: 'ideal_ohmic',
    light_direction: 'top_first',
    junction_stack: [
      {
        id: 'recombination_ito',
        name: 'Recombination ITO',
        thickness: '20 nm',
        optical_material: 'ITO',
        incoherent: false,
      },
    ],
    benchmark: null,
    back_reflector: null,
  }
}

function defaultJVExperimentInput(): JVExperimentInput {
  return {
    schema_version: 'solarlab.experiment-preparation.v1',
    id: 'jv-v2',
    device: defaultDeviceInput('jv-device'),
    experiment: { kind: 'jv' },
  }
}

function defaultSpatialExperimentInput(): SpatialExperimentInput {
  return {
    schema_version: 'solarlab.experiment-preparation.v1',
    id: 'spatial-v2',
    device: defaultDeviceInput('spatial-device'),
    experiment: { kind: 'jv_2d' },
  }
}

function defaultSweepInput(): SweepInput {
  return {
    schema_version: 'solarlab.sweep-preparation.v1',
    id: 'sweep-v2',
    base: defaultDeviceInput('sweep-device'),
    axes: [
      {
        id: 'absorber_mobility',
        target: { family: 'layer_parameter', owner_id: 'absorber', parameter: ['mu_n'] },
        coordinates: [{ kind: 'omit' }],
      },
    ],
  }
}

export function mountConfigurationWorkstation(
  root: HTMLElement,
  serviceBase: URL,
  handshakeOptions?: WorkstationHandshakeOptions,
): ConfigurationWorkstationHandle {
  let disposed = false
  let activeEditor: { dispose(): void } | null = null
  let previewPanel: ConfigurationPreviewPanel | null = null

  const bannerHost = document.createElement('div')
  bannerHost.dataset.role = 'configuration-handshake-host'

  const toolbar = document.createElement('div')
  toolbar.className = 'toolbar'
  toolbar.style.marginBottom = '12px'
  const modeLabel = document.createElement('label')
  modeLabel.textContent = 'Preparation document '
  const modeSelect = document.createElement('select')
  modeSelect.dataset.role = 'configuration-mode'
  modeSelect.setAttribute('aria-label', 'Preparation document')
  for (const [value, label] of [
    ['device', 'Single-junction device (DeviceInput)'],
    ['tandem', '2T monolithic tandem (TandemInput)'],
    ['jv', '1D J–V experiment (JVExperimentInput)'],
    ['spatial', '2D spatial experiment (SpatialExperimentInput)'],
    ['sweep', 'Serial parameter sweep (SweepInput)'],
  ] as const) {
    modeSelect.add(new Option(label, value))
  }
  modeLabel.appendChild(modeSelect)
  toolbar.appendChild(modeLabel)

  const layout = document.createElement('div')
  layout.style.cssText = 'display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;align-items:start;'
  const editorHost = document.createElement('div')
  editorHost.dataset.role = 'configuration-editor-host'
  const previewHost = document.createElement('div')
  previewHost.dataset.role = 'configuration-preview-host'
  layout.append(editorHost, previewHost)

  root.replaceChildren(bannerHost, toolbar, layout)

  let handshakeBlocked = false

  function previewEndpoint(path: string): URL {
    return new URL(path, serviceBase)
  }

  function showPreview(
    selection: Parameters<ConfigurationPreviewPanel['select']>[0],
  ): void {
    if (disposed || handshakeBlocked) return
    if (!previewPanel) {
      previewPanel = mountConfigurationPreviewPanel(previewHost, selection)
    } else {
      void previewPanel.select(selection)
    }
  }

  function selectMode(mode: ConfigurationEditorMode): void {
    if (disposed) return
    modeSelect.value = mode
    activeEditor?.dispose()
    activeEditor = null
    editorHost.replaceChildren()

    if (mode === 'device') {
      activeEditor = mountDeviceEditor(editorHost, {
        input: defaultDeviceInput(),
        onApply: input =>
          showPreview({
            endpoint: previewEndpoint('configuration-preview/device'),
            kind: 'device',
            input: input as DeviceInput,
          }),
      })
    } else if (mode === 'tandem') {
      activeEditor = mountDeviceEditor(editorHost, {
        input: defaultTandemInput(),
        onApply: input =>
          showPreview({
            endpoint: previewEndpoint('configuration-preview/tandem'),
            kind: 'tandem',
            input: input as TandemInput,
          }),
      })
    } else if (mode === 'jv') {
      activeEditor = mountJVExperimentEditor(editorHost, {
        input: defaultJVExperimentInput(),
        onApply: input =>
          showPreview({
            endpoint: previewEndpoint('configuration-preview/experiment'),
            kind: 'experiment',
            input,
          }),
      })
    } else if (mode === 'spatial') {
      activeEditor = mountSpatialExperimentEditor(editorHost, {
        input: defaultSpatialExperimentInput(),
        onApply: input =>
          showPreview({
            endpoint: previewEndpoint('configuration-preview/experiment'),
            kind: 'experiment',
            input,
          }),
      })
    } else {
      activeEditor = mountSweepEditor(editorHost, {
        input: defaultSweepInput(),
        onApply: input =>
          showPreview({
            endpoint: previewEndpoint('configuration-preview/sweep'),
            kind: 'sweep',
            input,
          }),
      })
    }
  }

  modeSelect.addEventListener('change', () => {
    selectMode(modeSelect.value as ConfigurationEditorMode)
  })

  selectMode('device')

  const ready = verifyWorkstationHandshake(
    handshakeOptions ?? {
      schemaEndpoint: new URL('configuration-schema', serviceBase),
      experimentsEndpoint: new URL('experiments', serviceBase),
    },
  ).then(result => {
    if (!disposed) {
      handshakeBlocked = result.status === 'mismatch' || result.status === 'error'
      renderHandshakeBanner(bannerHost, result)
    }
    return result
  })

  return {
    ready,
    selectMode,
    dispose() {
      if (disposed) return
      disposed = true
      activeEditor?.dispose()
      previewPanel?.dispose()
      root.replaceChildren()
    },
  }
}
