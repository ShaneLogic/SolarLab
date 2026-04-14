export function tutorialHTML(): string {
  return `
  <div class="card doc-card">
    <h3>Getting Started</h3>
    <div class="doc-body">
      <p>This simulator solves the 1D drift-diffusion equations coupled with Poisson's equation and mobile-ion transport to reproduce the electrical behaviour of thin-film solar cells.</p>

      <h4>1. Choose a device</h4>
      <p>Use the <b>Config</b> dropdown at the top of any experiment tab. The preset library includes:</p>
      <ul>
        <li><code>ionmonger_benchmark</code> — Courtier et al. 2019, MAPbI<sub>3</sub> n-i-p reference</li>
        <li><code>nip_MAPbI3</code> / <code>pin_MAPbI3</code> — perovskite stacks with HTL / absorber / ETL</li>
        <li><code>cigs_baseline</code> — ZnO / CdS / CIGS heterostructure, no mobile ions</li>
        <li><code>cSi_homojunction</code> — crystalline-Si n<sup>+</sup>/p wafer</li>
      </ul>
      <p>Click <b>Reset</b> to reload the preset, or edit the per-layer parameters directly — each layer expands to reveal Geometry, Transport, Recombination, and Ion/Optics groups.</p>

      <h4>2. Pick a physics tier</h4>
      <p>The <b>Mode</b> dropdown in the Device group selects which physics upgrades are active (see the <b>Algorithm</b> tab for the equations):</p>
      <ul>
        <li><b>Full</b> (default) — all Phase 1–4 upgrades: band-offset <i>V</i><sub>bi,eff</sub> + Richardson–Dushman thermionic emission at heterointerfaces, transfer-matrix optics when <code>optical_material</code> is set, dual-species ion migration when <i>D</i><sub>ion,−</sub> is provided, position-dependent trap profile, and temperature scaling of <i>V</i><sub>T</sub>, <i>μ</i>(<i>T</i>), <i>n</i><sub>i</sub>(<i>T</i>), <i>D</i><sub>ion</sub>(<i>T</i>).</li>
        <li><b>Fast</b> — Beer–Lambert optics, no thermionic emission cap. Use this for quick parameter sweeps where coherent optics are not needed.</li>
        <li><b>Legacy</b> — IonMonger-compatible ceiling: <i>T</i> pinned to 300 K, no TE, Beer–Lambert, single ion species, uniform bulk <i>τ</i>. Use this to reproduce the <code>ionmonger_benchmark</code> reference numbers exactly.</li>
      </ul>
      <p>The <i>T</i> field next to Mode sets the device temperature (K). It only affects the solution when Mode = <b>Full</b>; Fast and Legacy clamp to 300 K internally.</p>

      <h4>3. Run an experiment</h4>
      <ul>
        <li><b>J–V Sweep</b> — forward + reverse scan at constant voltage rate. Returns V<sub>oc</sub>, J<sub>sc</sub>, FF, PCE and the hysteresis index.</li>
        <li><b>Impedance</b> — small-signal AC analysis across a frequency range. Produces Nyquist and Bode plots from lock-in extraction of the transient response.</li>
        <li><b>Degradation</b> — long-time evolution under illumination; periodic frozen-ion J–V snapshots track metric drift.</li>
      </ul>

      <h4>4. Read the results</h4>
      <ul>
        <li><b>J–V:</b> compare forward (V: 0→V<sub>max</sub>) and reverse (V<sub>max</sub>→0) curves. Non-zero hysteresis index (HI) indicates slow ionic rearrangement.</li>
        <li><b>Impedance:</b> low-frequency intercept = R<sub>s</sub>+R<sub>rec</sub>; semicircle diameter = recombination resistance; additional low-f arc reveals ionic capacitance.</li>
        <li><b>Degradation:</b> snapshot metrics vs time — V<sub>oc</sub> decay implies growing non-radiative recombination; FF loss implies transport or interface deterioration.</li>
      </ul>

      <h4>Optical generation: TMM vs Beer&ndash;Lambert</h4>
      <p>Generation of electron&ndash;hole pairs <i>G</i>(<i>x</i>) is the source term that drives the drift&ndash;diffusion equations. The simulator supports two optical models:</p>
      <ul>
        <li><b>Beer&ndash;Lambert</b> (default on Legacy and Fast tiers): <i>G</i>(<i>x</i>) = <i>α</i> Φ e<sup>&minus;<i>αx</i></sup>. Simple and fast, but ignores reflection at layer interfaces and wavelength dependence. Typically overestimates <i>J</i><sub>SC</sub> by 5&ndash;15 %.</li>
        <li><b>Transfer-matrix method</b> (Full tier, active whenever <code>optical_material</code> is set on layers): solves Maxwell's equations across the coherent layer stack at each wavelength of the AM1.5G spectrum and integrates. Captures interference fringes, front-surface reflection, and wavelength-dependent absorption.</li>
      </ul>
      <p>To activate TMM, switch to <b>Full</b> tier and pick a preset whose name ends in <code>_tmm</code>, or set the <code>optical_material</code> field on every optical layer of your custom device.</p>

      <h4>Tips</h4>
      <ul>
        <li>Increase <i>N</i><sub>grid</sub> for smoother curves and better convergence near <i>V</i><sub>bi</sub>.</li>
        <li>Lower the scan rate <i>v</i><sub>rate</sub> to suppress hysteresis caused purely by scan speed.</li>
        <li>For non-perovskite materials set <i>D</i><sub>ion</sub> = 0 in every layer — the ion equations stay well-posed but integrate nothing.</li>
        <li>Tighten the tolerances <i>r</i><sub>tol</sub> / <i>a</i><sub>tol</sub> only if the curve shows unphysical kinks. The solver's Radau step cap already guards against the near-singular-Jacobian failure mode at flat-band.</li>
        <li>To reproduce IonMonger numbers exactly, switch Mode to <b>Legacy</b>. <i>V</i><sub>OC</sub> on Full is typically ≈ 0.1 V higher than Legacy because the thermionic emission cap reshapes collection at the HTL/absorber interface.</li>
        <li>Temperature coefficients (d<i>V</i><sub>OC</sub>/d<i>T</i> &lt; 0) only show up on <b>Full</b> — Fast and Legacy ignore the <i>T</i> field.</li>
      </ul>
    </div>
  </div>`
}

export async function mountTutorialPanel(el: HTMLElement): Promise<void> {
  el.innerHTML = tutorialHTML()
}
