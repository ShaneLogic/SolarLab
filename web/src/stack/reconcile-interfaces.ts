import type { LayerConfig } from '../types'

type Pair = readonly [number, number]

/**
 * Reconcile the interfaces array after a layer mutation.
 *
 * Invariant: ``result.length === Math.max(0, newLayers.length - 1)``.
 *
 * Surviving adjacent pairs (same `(left.name, right.name)` in old AND new)
 * keep their values. New pairs default to `[0, 0]`. Order matches the new
 * layer order.
 */
export function reconcileInterfaces(
  oldLayers: ReadonlyArray<LayerConfig>,
  newLayers: ReadonlyArray<LayerConfig>,
  oldInterfaces: ReadonlyArray<Pair>,
): Array<[number, number]> {
  const map = new Map<string, Pair>()
  for (let i = 0; i < oldInterfaces.length; i++) {
    const left = oldLayers[i]?.name
    const right = oldLayers[i + 1]?.name
    if (left != null && right != null) {
      map.set(`${left}\u0000${right}`, oldInterfaces[i])
    }
  }
  const result: Array<[number, number]> = []
  for (let i = 0; i < newLayers.length - 1; i++) {
    const key = `${newLayers[i].name}\u0000${newLayers[i + 1].name}`
    const pair = map.get(key)
    result.push(pair != null ? [pair[0], pair[1]] : [0, 0])
  }
  return result
}
