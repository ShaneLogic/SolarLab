import referenceJson from '../../tests/fixtures/ScapsBeerLambertDeviceV1.json'
import { beforeEach, expect, it } from 'vitest'
import { renderDeviceEditor, readDeviceEditor } from './config-editor'
import type { DeviceConfig } from './types'

const reference = referenceJson as unknown as DeviceConfig

beforeEach(() => { document.body.innerHTML = '<div id="editor"></div>' })

function withoutInertDefaults(config: DeviceConfig) {
  const copy = structuredClone(config)
  const device = copy.device as Record<string, unknown>
  for (const key of ['band_grading', 'flat_band_contacts', 'interface_plane_closure', 'interface_plane_projection', 'interface_tunneling']) {
    if (device[key] === false) delete device[key]
  }
  if (!device.interface_tunneling && device.tunnel_mass_eff === .2) delete device.tunnel_mass_eff
  for (const layer of copy.layers) {
    if (layer.Eg_back == null && layer.chi_back == null) {
      for (const key of ['grading_profile', 'grading_direction', 'grading_bowing', 'grading_N_mult']) {
        delete (layer as unknown as Record<string, unknown>)[key]
      }
    }
  }
  return copy
}

it('submits every normalized report parameter without display-rounding or model loss', () => {
  renderDeviceEditor(document.querySelector('#editor')!, reference, 'fast')
  const result = readDeviceEditor(reference)
  // The editor omits inactive grading/tunnelling defaults. Everything that
  // determines the optical and electrical calculation must be unchanged.
  expect(withoutInertDefaults(result)).toEqual(withoutInertDefaults(reference))
})
