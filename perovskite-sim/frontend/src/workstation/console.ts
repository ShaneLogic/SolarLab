export interface ConsoleHandle {
  /** Set the left-most "● FULL  band offsets · …" indicator. */
  setPhysics(tierLabel: string, summary: string): void
  /** Replace the right-hand scrolling log line. */
  log(message: string): void
}

export function mountConsole(container: HTMLElement): ConsoleHandle {
  container.classList.add('solver-console')
  container.innerHTML = `
    <span class="console-physics" id="console-physics">
      <span class="console-dot"></span>
      <span class="console-tier">IDLE</span>
      <span class="console-summary">(no active device)</span>
    </span>
    <span class="console-log" id="console-log"></span>`

  const tierEl = container.querySelector<HTMLElement>('.console-tier')!
  const summaryEl = container.querySelector<HTMLElement>('.console-summary')!
  const logEl = container.querySelector<HTMLElement>('#console-log')!

  return {
    setPhysics(tierLabel, summary) {
      tierEl.textContent = tierLabel
      summaryEl.textContent = summary
    },
    log(message) {
      logEl.textContent = message
    },
  }
}
