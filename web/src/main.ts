import './style.css'

const app = document.querySelector<HTMLDivElement>('#app')!

async function mount() {
  const query = new URL(location.href).searchParams
  if (query.get('view') === 'steady-2d') {
    const { mountSteady2DPanel } = await import('./panels/steady-2d')
    app.innerHTML = `<header class="workstation-header"><div><h1>SolarLab</h1><p>2D steady research</p></div>
      <a class="btn btn-ghost" href="?view=steady-jv">Steady J–V</a>
      <a class="btn btn-ghost" href="?view=transient">Ion-free transient</a>
      <a class="btn btn-ghost" href="?view=ionic-transient">Ionic transient</a>
      <a class="btn btn-ghost" href="?view=ionic-jv">Ionic J–V</a>
      <a class="btn btn-ghost" href="?">Workspace / legacy experiments</a></header><main id="steady-2d"></main>`
    const host = app.querySelector<HTMLElement>('#steady-2d')!
    host.style.cssText = 'max-width:1300px;margin:auto;padding:20px;overflow:auto;max-height:calc(100vh - 90px)'
    const initial = query.get('storeId') && query.get('runId') && query.get('attemptId')
      ? { storeId: query.get('storeId')!, runId: query.get('runId')!, attemptId: query.get('attemptId')! } : undefined
    const panel = mountSteady2DPanel(host, new URL('/steady-service/', location.origin), initial)
    window.addEventListener('pagehide', () => panel.dispose(), { once: true })
    await panel.ready
  } else if (query.get('view') === 'ionic-jv') {
    const { mountIonicJVPanel } = await import('./panels/ionic-jv')
    app.innerHTML = `<header class="workstation-header"><div><h1>SolarLab</h1><p>Illuminated ionic steady J–V</p></div>
      <a class="btn btn-ghost" href="?view=steady-2d">2D steady</a>
      <a class="btn btn-ghost" href="?view=steady-jv">Steady J–V</a>
      <a class="btn btn-ghost" href="?view=transient">Ion-free transient</a>
      <a class="btn btn-ghost" href="?view=ionic-transient">Ionic transient</a>
      <a class="btn btn-ghost" href="?">Workspace / legacy experiments</a></header><main id="ionic-jv"></main>`
    const host = app.querySelector<HTMLElement>('#ionic-jv')!
    host.style.cssText = 'max-width:1300px;margin:auto;padding:20px;overflow:auto;max-height:calc(100vh - 90px)'
    const initial = query.get('storeId') && query.get('runId') && query.get('attemptId')
      ? { storeId: query.get('storeId')!, runId: query.get('runId')!, attemptId: query.get('attemptId')! } : undefined
    const panel = mountIonicJVPanel(host, new URL('/steady-service/', location.origin), initial)
    window.addEventListener('pagehide', () => panel.dispose(), { once: true })
    await panel.ready
  } else if (query.get('view') === 'ionic-transient') {
    const { mountIonicTransientPanel } = await import('./panels/ionic-transient')
    app.innerHTML = `<header class="workstation-header"><div><h1>SolarLab</h1><p>Ionic transient research</p></div>
      <a class="btn btn-ghost" href="?view=steady-2d">2D steady</a>
      <a class="btn btn-ghost" href="?view=steady-jv">Steady J–V</a>
      <a class="btn btn-ghost" href="?view=transient">Ion-free transient</a>
      <a class="btn btn-ghost" href="?view=ionic-jv">Ionic J–V</a>
      <a class="btn btn-ghost" href="?">Workspace / legacy experiments</a></header><main id="ionic-transient"></main>`
    const host = app.querySelector<HTMLElement>('#ionic-transient')!
    host.style.cssText = 'max-width:1300px;margin:auto;padding:20px;overflow:auto;max-height:calc(100vh - 90px)'
    const initial = query.get('storeId') && query.get('runId') && query.get('attemptId')
      ? { storeId: query.get('storeId')!, runId: query.get('runId')!, attemptId: query.get('attemptId')! } : undefined
    const panel = mountIonicTransientPanel(host, new URL('/steady-service/', location.origin), initial)
    window.addEventListener('pagehide', () => panel.dispose(), { once: true })
    await panel.ready
  } else if (query.get('view') === 'transient') {
    const { mountTransientPanel } = await import('./panels/transient')
    app.innerHTML = `<header class="workstation-header"><div><h1>SolarLab</h1><p>Ion-free transient research</p></div>
      <a class="btn btn-ghost" href="?view=steady-2d">2D steady</a>
      <a class="btn btn-ghost" href="?view=steady-jv">Steady J–V</a>
      <a class="btn btn-ghost" href="?view=ionic-transient">Ionic transient</a>
      <a class="btn btn-ghost" href="?view=ionic-jv">Ionic J–V</a>
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
      <a class="btn btn-ghost" href="?view=steady-2d">2D steady</a>
      <a class="btn btn-ghost" href="?view=transient">Transient · new core</a>
      <a class="btn btn-ghost" href="?view=ionic-transient">Ionic transient</a>
      <a class="btn btn-ghost" href="?view=ionic-jv">Ionic J–V</a>
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
  const ionic = document.createElement('a'); ionic.href = '?view=ionic-transient'; ionic.textContent = 'Open ionic transient (new core)'
  const ionicJV = document.createElement('a'); ionicJV.href = '?view=ionic-jv'; ionicJV.textContent = 'Open ionic J–V (new core)'
  const steady2D = document.createElement('a'); steady2D.href = '?view=steady-2d'; steady2D.textContent = 'Open 2D steady (new core)'
  app.append(message, steady, transient, ionic, ionicJV, steady2D)
  console.error(error)
})
