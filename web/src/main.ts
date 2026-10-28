import './style.css'

const app = document.querySelector<HTMLDivElement>('#app')!

async function mount() {
  if (new URL(location.href).searchParams.get('view') === 'steady-jv') {
    const { mountSteadyJVPanel } = await import('./panels/steady-jv')
    app.innerHTML = `<header class="workstation-header"><div><h1>SolarLab</h1><p>Steady DC research</p></div>
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
  app.append(message, steady)
  console.error(error)
})
