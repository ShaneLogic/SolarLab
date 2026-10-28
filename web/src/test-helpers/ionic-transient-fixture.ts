/** Small synthetic wire/view fixtures. Actual computation evidence uses saved HTTP jobs. */
import type { IonicCatalog, IonicProtocol, IonicResult } from '../ionic-transient-client'
import { IONIC_EXPERIMENT, IONIC_SCOPE } from '../ionic-transient-client'
import type { FloatArray } from '../steady-jv-client'

export const registry = 'a'.repeat(64), schema = 'b'.repeat(64), prepared = 'c'.repeat(64)
const budget = { max_iterations: 50, max_residual_evaluations: 500, max_jacobian_evaluations: 50, max_backtracks: 24 }
export const protocol: IonicProtocol = {
  times_s: [0, .25, .5, .75, 1], voltage_V: [.002, .0025, .003, .0035, .004], generation_scale: [0, 0, 0, 0, 0], intervals_per_layer: [12, 12, 12],
  controls: { initial_steady: { budget: { ...budget, max_iterations: 60, max_residual_evaluations: 600, max_jacobian_evaluations: 60 }, inventory_row_node: 12, wall_s: 30 },
    transient: { newton_budget: { ...budget }, max_steps: 16, wall_s: 30 } },
}
const bounds = {
  samples: { min: 2, max: 17 }, layers: { min: 1, max: 8 }, intervals_per_layer: { min: 2, max: 64 },
  times_s: { min: 0, max: 1 }, voltage_V: { min: -.02, max: .02 }, generation_scale: { min: 0, max: 0 },
  max_steps: { min: 1, max: 16 }, wall_s: { min: 0, max: 30 }, initial_wall_s: { min: 0, max: 30 }, inventory_row_node: { min: 0, max: 64 },
  max_iterations: { min: 1, max: 50 }, max_residual_evaluations: { min: 1, max: 500 }, max_jacobian_evaluations: { min: 1, max: 50 }, max_backtracks: { min: 0, max: 24 },
  initial_max_iterations: { min: 1, max: 60 }, initial_max_residual_evaluations: { min: 1, max: 600 }, initial_max_jacobian_evaluations: { min: 1, max: 60 }, initial_max_backtracks: { min: 0, max: 24 },
}
export const catalog: IonicCatalog = { registry, schema, ceiling: { wall_seconds: 30, rss_bytes: 1_610_612_736, output_bytes: 100_663_296 },
  routes: [{ id: IONIC_EXPERIMENT, version: 1, devices: [{ id: 'research_1d_shared', preparedSha256: prepared }], domain: 'absorber', policy: {}, bounds }] }

export function catalogDocument() {
  const item = (min: number, max: number) => ({ minimum: min, maximum: max, type: 'number' })
  const array = (min: number, max: number, minItems = 2, maxItems = 17) => ({ type: 'array', minItems, maxItems, items: item(min, max) })
  const closed = (properties: Record<string, unknown>) => ({ additionalProperties: false, type: 'object', properties, required: Object.keys(properties) })
  const b = closed({ max_iterations: item(1, 60), max_residual_evaluations: item(1, 600), max_jacobian_evaluations: item(1, 60), max_backtracks: item(0, 24) })
  return { schema: 'solarlab.execution-catalog.v1', job_schema_sha256: schema, resource_ceiling: { ...catalog.ceiling, calls: null, accepted_steps: null }, bindings: [{
    experiment_id: IONIC_EXPERIMENT, version: 1, driver: IONIC_EXPERIMENT, engine: 'solarlab', scope: IONIC_SCOPE,
    input_schema: { ...closed({ device_id: { type: 'string' }, times_s: array(0, 1), voltage_V: array(-.02, .02), generation_scale: array(0, 0),
      intervals_per_layer: array(2, 64, 1, 8), controls: { $ref: '#/$defs/Controls' } }), $defs: {
      Controls: closed({ initial_steady: { $ref: '#/$defs/Initial' }, transient: { $ref: '#/$defs/Time' } }), Budget: b,
      Initial: closed({ budget: { $ref: '#/$defs/Budget' }, inventory_row_node: item(0, 64), wall_s: item(0, 30) }),
      Time: closed({ newton_budget: { $ref: '#/$defs/Budget' }, max_steps: item(1, 16), wall_s: item(0, 30) }),
    } }, preparation_context: { devices: [{ device_id: 'research_1d_shared', prepared_sha256: prepared }],
      ion_domain_endpoints: ['blocking', 'blocking'], ion_domain_layer: 'absorber', fixed_numerical_policy: {} },
  }] }
}

export function resultFixture(failed = false, count = failed ? 2 : 5): IonicResult {
  const times = protocol.times_s.slice(0, count), x = [0, 1e-8, 3e-8, 4e-8], xi = x.slice(1, 3), steps = Math.max(0, count - 1)
  const vector = (values: number[]): FloatArray => ({ shape: [values.length], values })
  const rows = (c: number, n: number, offset: number): FloatArray => ({ shape: [c, n], values: Array.from({ length: c * n }, (_, i) => offset + i) })
  const arrays: Record<string, FloatArray> = { requested_times_s: vector(protocol.times_s), requested_voltage_V: vector(protocol.voltage_V),
    requested_generation_scale: vector(protocol.generation_scale), accepted_times_s: vector(times), nodes_m: vector(x),
    ion_nodes_m: vector(xi), ion_node_indices: vector([1, 2]), ion_cell_measures_m: vector([1e-8, 1e-8]),
    potential_V: rows(count, 4, 1), electron_density_m3: rows(count, 4, 10), hole_density_m3: rows(count, 4, 20),
    positive_ion_density_m3: rows(count, 2, 30), negative_ion_density_m3: rows(count, 2, 40),
    positive_ion_inventory_m2: vector(times.map(() => 60)), negative_ion_inventory_m2: vector(times.map(() => 80)),
    positive_ion_cell_inventory_m2: rows(count, 2, 1), negative_ion_cell_inventory_m2: rows(count, 2, 2), state_coordinates: rows(count, 16, 0) }
  for (const [i, kind] of ['carrier_conduction', 'ion_conduction', 'conduction', 'displacement', 'total'].entries()) arrays[kind + '_current_density_A_m2'] = rows(steps, 5, -20 + i)
  for (const kind of ['conduction', 'displacement', 'total']) arrays['terminal_' + kind + '_current_density_A_m2'] = rows(steps, 2, -10)
  const attempts = Array.from({ length: steps + (failed && count ? 1 : 0) }, (_, i) => ({ start_s: protocol.times_s[i], end_s: protocol.times_s[i + 1], accepted: i < steps,
    termination: i < steps ? 'converged' : 'budget_exhausted' }))
  return { state: failed ? 'failed' : 'succeeded', arrays, storedArrays: {}, units: {}, species: ['positive', 'negative'], attempts,
    summary: { physical_certificate_status: 'not_evaluated', reason: failed ? 'budget_exhausted' : null },
    worker: { registered_device_id: 'research_1d_shared', scientific_qualification: false },
    diagnosticsJson: '{"failed_nonfinite":{"$float":"NaN"}}', initialJson: '{}', originalJson: '{}', manifestJson: '{}' }
}

export function ionicNpy(array: FloatArray, integer = false, endian: '<' | '>' = '<'): Uint8Array {
  const shape = array.shape.join(', ') + (array.shape.length === 1 ? ',' : '')
  const header = `{'descr': '${endian}${integer ? 'i' : 'f'}8', 'fortran_order': False, 'shape': (${shape}), }`
  const pad = 64 - (10 + header.length + 1) % 64, bytes = new Uint8Array(10 + header.length + pad + 1 + array.values.length * 8)
  bytes.set([147, 78, 85, 77, 80, 89, 1, 0]); const view = new DataView(bytes.buffer)
  view.setUint16(8, header.length + pad + 1, true); bytes.set(new TextEncoder().encode(header + ' '.repeat(pad) + '\n'), 10)
  array.values.forEach((v, i) => integer ? view.setBigInt64(bytes.length - array.values.length * 8 + i * 8, BigInt(v), endian === '<')
    : view.setFloat64(bytes.length - array.values.length * 8 + i * 8, v, endian === '<'))
  return bytes
}
