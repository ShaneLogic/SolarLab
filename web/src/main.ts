import './style.css'

const app = document.querySelector<HTMLDivElement>('#app')!

async function mount() {
  const query = new URL(location.href).searchParams
  if (query.get('view') === 'transient') {
    const { mountTransientPanel } = await import('./panels/transient')
    app.innerHTML = `<header class="workstation-header"><div><h1>SolarLab</h1><p>Ion-free transient research</p></div>
      <a class="btn btn-ghost" href="?view=steady-jv">Steady J–V</a>
      <a class="btn btn-ghost" href="?">Workspace / legacy experiments</a></header><main id="transient"></main>`
    const host = app.querySelector<HTMLElement>('#transient')!
    host.style.cssText = 'max-width:1300px;margin:auto;padding:20px;overflow:auto;max-height:calc(100vh - 90px)'
    const initial = query.get('storeId') && query.get('runId') && query.get('attemptId')
      ? { storeId: query.get('storeId')!, runId: query.get('runId')!, attemptId: query.get('attemptId')! } : undefined
    const panel = mountTransientPanel(host, new URL('/steady-service/', location.origin), initial)
    window.addEventListener('pagehide', () => panel.dispose(), { once: true })
    await panel.ready
  } else if (query.get('view') === 'steady-jv') {
    const { mountSteadyJVPanel } = await import('./panels/steady-jv')
    app.innerHTML = `<header class="workstation-header"><div><h1>SolarLab</h1><p>Steady DC research</p></div>
      <a class="btn btn-ghost" href="?view=transient">Transient · new core</a>
      <a class="btn btn-ghost" href="?">Workspace / legacy experiments</a></header><main id="steady-dc"></main>`
    const host = app.querySelector<HTMLElement>('#steady-dc')!
    host.style.cssText = 'max-width:1300px;margin:auto;padding:20px;overflow:auto;max-height:calc(100vh - 90px)'
    const panel = mountSteadyJVPanel(host, new URL('/steady-service/', location.origin))
    window.addEventListener('pagehide', () => panel.dispose(), { once: true })
    await panel.ready
  } else {
    const { mountWorkstation } = await import('./workstation/shell')
    await mountWorkstation(app)
  }
}
void mount().catch(error => {
  app.replaceChildren()
  const message = document.createElement('p'); message.textContent = `Unable to open workspace: ${String(error)}`
  const steady = document.createElement('a'); steady.href = '?view=steady-jv'; steady.textContent = 'Open steady J–V (new core)'
  const transient = document.createElement('a'); transient.href = '?view=transient'; transient.textContent = 'Open transient (new core)'
  app.append(message, steady, transient)
  console.error(error)
})
