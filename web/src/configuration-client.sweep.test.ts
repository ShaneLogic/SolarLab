// @vitest-environment node
import { afterEach, expect, it, vi } from 'vitest'
import type { SweepInput } from './generated/configuration-inputs'
import { configurationSchemaSha256 } from './generated/configuration-schema'
import { previewSweepConfiguration, sweepInputFromPreview, sweepPointInputFromPreview } from './configuration-client'

const raw: SweepInput = { schema_version: 'solarlab.sweep-preparation.v1', id: 'sweep',
  base: { schema_version: 'solarlab.device-preparation.v1', id: 'base', source_format: 'canonical', layers: [] },
  axes: [{ id: 'voltage', target: { family: 'setting', parameter: ['phi_left'] }, coordinates: [{ kind: 'value', value: -0 }, { kind: 'omit' }] }] }
function json() {
  return JSON.stringify({ schema: 'solarlab.configuration-preview.v1', kind: 'sweep', status: 'prepared_pending_dependencies', can_execute: false, input: raw,
    resolved: { schema: 'solarlab.resolved-sweep-preparation.v1', id: raw.id, can_execute: false, points: [{ id: 'point:1', can_execute: false, applied_input: { ...raw.base, settings: { phi_left: -0 } } }] },
    identity: { scope: 'configuration_content', content_sha256: 'a'.repeat(64), default_catalog_sha256: 'b'.repeat(64), resource_library_sha256: 'c'.repeat(64), configuration_schema_sha256: configurationSchemaSha256 } })
    .replace('"value":0', '"value":-0.0').replace('"phi_left":0', '"phi_left":-0.0')
}
afterEach(() => { vi.unstubAllGlobals() })
it('posts exact sweep words and validates the new schema-bound kind', async () => {
  const original = json(), fetcher = vi.fn().mockResolvedValue(new Response(original, { headers: { 'Content-Type': 'application/json' } }))
  vi.stubGlobal('fetch', fetcher)
  const doc = await previewSweepConfiguration('/sweep', raw)
  expect(fetcher.mock.calls[0][1].headers['X-Solarlab-Configuration-Schema']).toBe(configurationSchemaSha256)
  expect(fetcher.mock.calls[0][1].body).toContain('"value":-0.0')
  expect(doc.json).toBe(original)
  expect(sweepInputFromPreview(original)).toStrictEqual(raw)
  expect(Object.is((sweepPointInputFromPreview(original, 'point:1') as typeof raw.base & { settings: { phi_left: number } }).settings.phi_left, -0)).toBe(true)
})
it('keeps mismatched schemas, unavailable points and duplicate point identities visible', () => {
  expect(() => sweepInputFromPreview(json().replace(configurationSchemaSha256, '0'.repeat(64)))).toThrow('schema identity')
  expect(() => sweepPointInputFromPreview(json(), 'missing')).toThrow('Select one')
  const value = JSON.parse(json()); value.resolved.points.push(value.resolved.points[0])
  expect(() => sweepPointInputFromPreview(JSON.stringify(value), 'point:1')).toThrow('Select one')
  value.resolved.points = [{ id: 'point:1', can_execute: false, applied_input: null }]
  expect(() => sweepPointInputFromPreview(JSON.stringify(value), 'point:1')).toThrow('applied input')
})
