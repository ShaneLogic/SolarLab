export interface ConsoleHandle {
  /** Show the mode, with its supported physics in a tooltip. */
  setPhysics(tierLabel: string, summary: string): void
  /** Show a short status, with optional details in a tooltip. */
  log(message: string, details?: string): void
}

export function mountConsole(container: HTMLElement): ConsoleHandle {
  container.classList.add('solver-console')
  container.innerHTML = `
    <span class="console-physics" id="console-physics" title="No active device">
      <span class="console-dot"></span>
      <span class="console-tier">IDLE</span>
    </span>
    <span class="console-log" id="console-log"></span>`

  const tierEl = container.querySelector<HTMLElement>('.console-tier')!
  const physicsEl = container.querySelector<HTMLElement>('.console-physics')!
  const logEl = container.querySelector<HTMLElement>('#console-log')!

  return {
    setPhysics(tierLabel, summary) {
      tierEl.textContent = tierLabel
      physicsEl.title = `${tierLabel} mode capabilities: ${summary}`
    },
    log(message, details = message) {
      logEl.textContent = message
      logEl.title = details
    },
  }
}
