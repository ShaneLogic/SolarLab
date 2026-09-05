export function tutorialHTML(): string {
  return `
  <div class="card doc-card">
    <h3>Getting Started</h3>
    <div class="doc-body">
      <p>This simulator solves the 1D drift-diffusion equations coupled with Poisson's equation and mobile-ion transport to reproduce the electrical behaviour of thin-film solar cells.</p>

      <h4>1. Choose a device</h4>
      <p>The research presets cover two studies:</p>
      <ul>
        <li><b>SCAPS parity - Reference v2</b> — the defect-corrected SCAPS partner device, with mobile ions disabled. Studies band offsets, doping, thickness and recombination trends. The reference mode is <b>Fast</b>; selecting this device does not imply that every SCAPS parity target has passed.</li>
        <li><b>Calado 2016 - Ion hysteresis</b> — the Fig. 1f toy device for ionic redistribution, contact recombination and scan-rate-dependent J–V hysteresis. The reference mode is <b>Legacy</b>, with <i>D</i><sub>c</sub> = 2.585 × 10<sup>−18</sup> m²/s and <i>c</i><sub>0</sub> = 10<sup>25</sup> m⁻³ in the absorber.</li>
      </ul>
      <p>For a Calado trend study, a useful starting point is a transient J–V sweep at <b>0.04 V/s</b> up to <b>1.2 V</b>. Change one of <i>D</i><sub>c</sub>, <i>c</i><sub>0</sub>, scan rate or contact SRH at a time. The built-in 0 → V<sub>max</sub> → 0 sweep differs from the paper's −1 → +1.2 V protocol with a 3 s hold; quantitative paper comparison uses the dedicated Calado driver. The forward-branch mismatch remains open.</p>
      <p>Click <b>Reset</b> to reload the preset, or edit the per-layer parameters directly — each layer expands to reveal Geometry, Transport, Recombination, and Ion/Optics groups.</p>

      <h4>2. Pick a physics tier</h4>
      <p>The <b>Mode</b> dropdown in the Device group selects which physics upgrades are active (see the <b>Algorithm</b> tab for the equations):</p>
      <ul>
        <li><b>Full</b> — every configured upgrade, including self-consistent radiative reabsorption, field-dependent mobility, and explicit finite-rate Robin contacts.</li>
        <li><b>Fast</b> — thermionic emission, TMM optics, dual-species ions, position-dependent traps, temperature scaling and photon recycling remain active; only the three per-RHS upgrades reserved for Full are omitted.</li>
        <li><b>Legacy</b> — <i>T</i> pinned to 300 K, no TE, Beer–Lambert, single ion species, uniform bulk <i>τ</i>. This is the reference mode for the Calado 2016 study.</li>
      </ul>
      <p>The <i>T</i> field next to Mode sets the device temperature (K) in <b>Fast</b> and <b>Full</b>. Legacy clamps the model to 300 K.</p>

      <h4>3. Run an experiment</h4>
      <ul>
        <li><b>J–V Sweep</b> — forward + reverse scan at constant voltage rate. Returns V<sub>oc</sub>, J<sub>sc</sub>, FF, PCE and the hysteresis index. Supports <b>dark mode</b> (G = 0) for diode characterisation, <b>current decomposition</b> (J<sub>n</sub>, J<sub>p</sub>, J<sub>ion</sub>, J<sub>disp</sub>), and <b>spatial profile export</b> at every bias point.</li>
        <li><b>Impedance</b> — small-signal AC analysis across a frequency range. Produces Nyquist and Bode plots from lock-in extraction of the transient response.</li>
        <li><b>Degradation</b> — long-time evolution under illumination; periodic frozen-ion J–V snapshots track metric drift.</li>
        <li><b>Transient Photovoltage (TPV)</b> — the device is equilibrated at open circuit under steady illumination, then a small light pulse is applied. The voltage transient V(t) decays back to V<sub>oc</sub> as excess carriers recombine. Fitted mono-exponential lifetime &tau; encodes the dominant recombination rate.</li>
      </ul>

      <h4>4. Read the results</h4>
      <ul>
        <li><b>J–V:</b> compare forward (V: 0&rarr;V<sub>max</sub>) and reverse (V<sub>max</sub>&rarr;0) curves. Non-zero hysteresis index (HI) indicates slow ionic rearrangement. In <b>dark mode</b>, the curve shows the diode injection characteristic (no photocurrent). Use <b>current decomposition</b> to identify which carrier species dominates at each bias.</li>
        <li><b>Impedance:</b> arcs and intercepts are protocol-dependent observables. Attribute a low-frequency branch to ions only after the DC state, frequency window, perturbation amplitude, cycle history and grid have passed their reported checks.</li>
        <li><b>Degradation:</b> snapshot metrics vs time &mdash; V<sub>oc</sub> decay implies growing non-radiative recombination; FF loss implies transport or interface deterioration.</li>
        <li><b>TPV:</b> the decay time &tau; is the effective carrier lifetime at open circuit. Shorter &tau; indicates faster recombination. Compare across device configurations or degradation states to track recombination evolution.</li>
      </ul>

      <h4>Characterisation experiments (Python API)</h4>
      <p>Four higher-level wrappers on top of the solver deliver the fits that experimentalists read off of their measured data. They currently run from a notebook or script, not from this UI:</p>
      <ul>
        <li><b>Dark J&ndash;V fit</b> &mdash; <code>run_dark_jv(stack, V_max, n_points)</code> runs a G=0 forward sweep and extracts diode ideality <i>n</i> and saturation current <i>J</i><sub>0</sub> from the log|<i>J</i>| vs <i>V</i> slope. Auto-selects the exponential-regime window (rejects sub-turn-on leakage and high-<i>V</i> series-resistance roll-off).</li>
        <li><b>Suns&ndash;V<sub>oc</sub></b> &mdash; <code>run_suns_voc(stack, suns_levels)</code> sweeps light intensity, bisects for <i>V</i><sub>oc</sub>(<i>X</i>) at each level, and builds a Sinton pseudo-JV curve immune to series resistance. Reports <b>pseudo-FF</b> and the <i>V</i><sub>oc</sub>-vs-ln(<i>X</i>) slope used as a recombination-ideality proxy.</li>
        <li><b>EQE / IPCE</b> &mdash; <code>compute_eqe(stack, wavelengths_nm)</code> runs a monochromatic TMM + drift&ndash;diffusion at each wavelength and returns EQE(&lambda;). Integrating against AM1.5G gives <i>J</i><sub>sc</sub>; cross-checked against the full-spectrum sweep to within ~25 %. Requires a TMM-enabled preset (<code>optical_material</code> on the absorber), such as the SCAPS reference.</li>
        <li><b>Mott&ndash;Schottky C&ndash;V</b> &mdash; <code>run_mott_schottky(stack, V_range, frequency)</code> runs a dark C&ndash;V sweep (<code>illuminated=False</code> on the impedance path) and fits 1/<i>C</i><sup>2</sup> vs <i>V</i> to extract the apparent built-in voltage <i>V</i><sub>bi,app</sub> (intercept with the p&ndash;n thermal correction) and the net ionised density <i>N</i><sub>eff</sub> (slope). Includes an adaptive window selector that rejects the fully-depleted and injection tails.</li>
      </ul>
      <p>See the <code>README.md</code> under <em>Phase 2 Characterisation Experiments</em> for signatures and the physics of each fit.</p>

      <h4>Optical generation: TMM vs Beer&ndash;Lambert</h4>
      <p>Generation of electron&ndash;hole pairs <i>G</i>(<i>x</i>) is the source term that drives the drift&ndash;diffusion equations. The simulator supports two optical models:</p>
      <ul>
        <li><b>Beer&ndash;Lambert</b> (Legacy, or the fallback when no optical material is configured): <i>G</i>(<i>x</i>) = <i>α</i> Φ e<sup>&minus;<i>αx</i></sup>. Simple and fast, but ignores reflection at layer interfaces and wavelength dependence. Typically overestimates <i>J</i><sub>SC</sub> by 5&ndash;15 %.</li>
        <li><b>Transfer-matrix method</b> (Fast / Full, active whenever <code>optical_material</code> is set on layers): solves Maxwell's equations across the coherent layer stack at each wavelength of the AM1.5G spectrum and integrates. Captures interference fringes, front-surface reflection, and wavelength-dependent absorption.</li>
      </ul>
      <p>The SCAPS reference supplies TMM optical materials and uses <b>Fast</b>. The Calado reference uses weak Beer&ndash;Lambert absorption in <b>Legacy</b>; changing its optics changes the comparison protocol.</p>

      <h4>Custom Stacks</h4>
      <p>
        In <b>Full</b> tier the Device pane shows your stack as a vertical
        cross-section. You can:
      </p>
      <ul>
        <li><b>Add</b> a layer with <em>＋ Add layer…</em> or any <em>+</em>
            between layers — pick a starter from the template library
            (TiO<sub>2</sub> ETL, spiro HTL, Ag back contact, …) or start blank.</li>
        <li><b>Reorder</b> by dragging a layer's <em>⋮⋮</em> handle, or by
            clicking the ↑↓ buttons on hover (keyboard-accessible).</li>
        <li><b>Delete</b> with the <em>✕</em> button on hover.</li>
        <li><b>Edit</b> any field by clicking a layer to select it and using
            the detail editor on the right.</li>
        <li><b>Save</b> as a named user preset via <em>Save as…</em>; it lands
            in <code>configs/user/</code> and appears under <em>User presets</em>
            in the dropdown.</li>
        <li><b>Export</b> the current device as YAML via <em>↓ YAML</em> for
            sharing outside the app.</li>
      </ul>
      <p>
        Both n-i-p (<code>ETL / absorber / HTL</code>) and p-i-n
        (<code>HTL / absorber / ETL</code>) orientations are supported — the
        simulator derives the built-in voltage from the stack itself.
      </p>

      <h4>Tips</h4>
      <ul>
        <li>Increase <i>N</i><sub>grid</sub> for smoother curves and better convergence near <i>V</i><sub>bi</sub>.</li>
        <li>Lower the scan rate <i>v</i><sub>rate</sub> to suppress hysteresis caused purely by scan speed.</li>
        <li>The J&ndash;V sweep's <i>V</i><sub>max</sub> field is the upper voltage of the forward leg; if your stack's <i>V</i><sub>OC</sub> exceeds it the curve never crosses <i>J</i> = 0 and the reported <i>V</i><sub>OC</sub> will be clipped to <i>V</i><sub>max</sub>. The default 1.4&nbsp;V covers MAPbI<sub>3</sub>-like stacks; the Python API picks <code>max(<i>V</i><sub>bi,&nbsp;eff</sub>&times;1.3, 1.4&nbsp;V)</code> automatically when called with <code>V_max=None</code>.</li>
        <li>To exclude mobile ions, set <i>D</i><sub>c</sub> = <i>D</i><sub>a</sub> = 0 in every layer. Initial reference concentrations are <i>c</i><sub>0</sub> for positive ions and <i>a</i><sub>0</sub> for negative ions, in m⁻³.</li>
        <li>Tighten the tolerances <i>r</i><sub>tol</sub> / <i>a</i><sub>tol</sub> only if the curve shows unphysical kinks. The solver's Radau step cap already guards against the near-singular-Jacobian failure mode at flat-band.</li>
        <li>Keep the reference mode fixed during a parameter sweep: <b>Fast</b> for SCAPS parity and <b>Legacy</b> for Calado 2016. Changing the tier changes the physical assumptions.</li>
        <li>Temperature coefficients (d<i>V</i><sub>OC</sub>/d<i>T</i> &lt; 0) appear in <b>Fast</b> and <b>Full</b>; Legacy ignores the <i>T</i> field.</li>
      </ul>
    </div>
  </div>`
}

export async function mountTutorialPanel(el: HTMLElement): Promise<void> {
  el.innerHTML = tutorialHTML()
}
