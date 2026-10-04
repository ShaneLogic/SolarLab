import { getScapsReferenceCatalog, getScapsReferencePoint } from './api'
import type { DeviceConfig } from './types'

export function mountScapsReferenceControls(root: HTMLElement, onLoad: (config: DeviceConfig) => void) {
  root.innerHTML = `<details>
    <summary>Report parameters</summary>
    <div class="form-grid">
      <label class="form-group"><span>Reference sweep</span><select data-reference="sweep"></select></label>
      <label class="form-group"><span data-reference="unit">Value</span><input type="number" step="any" data-reference="value"></label>
    </div>
    <div class="actions"><button class="btn btn-ghost" data-reference="load" title="Restore the report baseline and apply this parameter point">Load point</button><span role="status" data-reference="status"></span></div>
  </details>`
  const sweep = root.querySelector<HTMLSelectElement>('[data-reference="sweep"]')!
  const value = root.querySelector<HTMLInputElement>('[data-reference="value"]')!
  const button = root.querySelector<HTMLButtonElement>('[data-reference="load"]')!
  const status = root.querySelector<HTMLElement>('[data-reference="status"]')!
  let catalog: Awaited<ReturnType<typeof getScapsReferenceCatalog>> | null = null
  let loading = false
  const syncValue = () => {
    const entry = catalog?.sweeps[sweep.value]
    if (!entry) return
    root.querySelector<HTMLElement>('[data-reference="unit"]')!.textContent = `Value (${entry.unit})`
    value.min = String(Math.min(...entry.points.map(p => p.x)))
    value.max = String(Math.max(...entry.points.map(p => p.x)))
    value.value = String(entry.points[0].x)
  }
  sweep.addEventListener('change', syncValue)
  button.addEventListener('click', async () => {
    if (!catalog || value.value.trim() === '' || !value.checkValidity()) {
      value.reportValidity()
      status.textContent = 'Enter a value within the reference interval.'
      return
    }
    button.disabled = true
    status.textContent = 'Loading...'
    try {
      const config = await getScapsReferencePoint(sweep.value, Number(value.value))
      onLoad(config)
      status.textContent = config.simulation_hints?.reference_point?.scope === 'high_doping_extrapolation'
        ? 'Loaded (high-doping extrapolation)' : 'Loaded'
    } catch (error) {
      status.textContent = (error as Error).message
    } finally {
      button.disabled = false
    }
  })
  return {
    update(config: DeviceConfig) {
      root.hidden = config.simulation_hints?.reference_study !== 'scaps_beer_lambert_v1'
      if (root.hidden || catalog || loading) return
      loading = true
      button.disabled = true
      void getScapsReferenceCatalog().then(data => {
        catalog = data
        for (const [id, entry] of Object.entries(data.sweeps)) {
          sweep.add(new Option(entry.title, id))
        }
        syncValue()
        const point = config.simulation_hints?.reference_point
        if (point && data.sweeps[point.sweep]) {
          sweep.value = point.sweep
          syncValue()
          value.value = String(point.value)
        }
      }).catch(error => {
        status.textContent = (error as Error).message
      }).finally(() => { loading = false; button.disabled = catalog === null })
    },
  }
}
