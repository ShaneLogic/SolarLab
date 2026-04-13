import './style.css'
import { mountWorkstation } from './workstation/shell'

const app = document.querySelector<HTMLDivElement>('#app')!
void mountWorkstation(app).catch(e => {
  app.innerHTML = `<div class="error-card" style="padding:20px;">
    Failed to mount workstation: ${(e as Error).message}
  </div>`
  console.error(e)
})
