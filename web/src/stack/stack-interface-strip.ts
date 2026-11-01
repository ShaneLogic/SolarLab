type Pair = readonly [number, number]

function fmtSrv(v: number): string {
  if (!Number.isFinite(v) || v === 0) return '0'
  if (v >= 1) return v.toExponential(1)
  return v.toExponential(1)
}

/**
 * Render the inline interface row that sits between two adjacent layer
 * cards in the visualizer. Click events are delegated to the visualizer
 * via the data-action attribute.
 */
export function renderInterfaceStrip(
  ifaceIdx: number,
  pair: Pair,
  isDefaultZero: boolean,
): string {
  const cls = isDefaultZero ? 'iface-strip is-default' : 'iface-strip'
  const label = isDefaultZero
    ? '◆ uses default 0 m/s'
    : `◆ <i>v</i><sub>n</sub>=${fmtSrv(pair[0])} <i>v</i><sub>p</sub>=${fmtSrv(pair[1])} m/s`
  return `
    <button class="${cls}"
            data-action="edit-iface"
            data-iface-idx="${ifaceIdx}"
            type="button"
            aria-label="Edit interface ${ifaceIdx + 1}">
      ${label}
    </button>`
}
