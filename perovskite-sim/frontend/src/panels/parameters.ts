export function parametersHTML(): string {
  return `
  <div class="card doc-card">
    <h3>Physical Quantities &amp; Parameters</h3>
    <div class="doc-body">
      <p>Every field in the layer editor maps directly to a term in the drift-diffusion + Poisson + ion-transport system. All values are in SI units unless marked otherwise. Click any reference link to open the original source.</p>

      <h4>Device-level</h4>
      <table class="param-table">
        <thead><tr><th>Symbol</th><th>Name</th><th>Unit</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr>
            <td>Mode</td><td>Physics tier</td><td>—</td>
            <td><b>Full</b> enables every Phase 1–4 upgrade; <b>Fast</b> drops TMM optics and the thermionic-emission cap; <b>Legacy</b> is IonMonger-compatible (<i>T</i> pinned to 300 K, single ion, uniform <i>τ</i>, Beer–Lambert, no TE). See the Algorithm tab for the per-flag gating table.</td>
          </tr>
          <tr>
            <td><i>T</i></td><td>Device temperature</td><td>K</td>
            <td>Lattice temperature. In <b>Full</b> mode it sets <i>V</i><sub>T</sub> = <i>k</i><sub>B</sub><i>T</i>/<i>q</i> and rescales <i>μ</i>(<i>T</i>) ∝ <i>T</i><sup>−3/2</sup>, <i>n</i><sub>i</sub>(<i>T</i>) (Boltzmann + DOS), and <i>D</i><sub>ion</sub>(<i>T</i>) (Arrhenius). Ignored in Fast/Legacy (clamped to 300 K).</td>
          </tr>
          <tr>
            <td><i>V</i><sub>bi</sub></td><td>Built-in voltage</td><td>V</td>
            <td>Manual built-in potential used as the Poisson Dirichlet boundary. In <b>Full</b> mode, a separate <i>V</i><sub>bi,eff</sub> is also derived from the heterostructure via <i>χ</i>/<i>E</i><sub>g</sub>/doping and used to set the voltage-sweep range. See Algorithm tab.</td>
          </tr>
          <tr>
            <td><i>Φ</i></td><td>Photon flux</td><td>m⁻²·s⁻¹</td>
            <td>Above-bandgap incident photon flux used by the Beer–Lambert generation profile <i>G</i>(<i>x</i>) = <i>α</i> <i>Φ</i> e<sup>−<i>αx</i></sup>. For AM1.5G at Eg ≈ 1.6 eV this is ≈ 1.4 × 10<sup>21</sup> m⁻²·s⁻¹. Unused when TMM optics are active (any layer sets <code>optical_material</code>).</td>
          </tr>
          <tr>
            <td><i>v</i><sub>n</sub>, <i>v</i><sub>p</sub></td><td>Interface SRV</td><td>m·s⁻¹</td>
            <td>Surface-recombination velocities (electrons / holes) at each heterointerface. High SRV on the minority-carrier side represents a blocking contact losing the wrong carrier type.</td>
          </tr>
        </tbody>
      </table>

      <h4>Geometry &amp; Electrostatics (per layer)</h4>
      <table class="param-table">
        <thead><tr><th>Symbol</th><th>Name</th><th>Unit</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td>Thickness</td><td>Layer thickness</td><td>m</td><td>Physical extent of the layer along the 1D coordinate <i>x</i>.</td></tr>
          <tr><td><i>ε</i><sub>r</sub></td><td>Relative permittivity</td><td>—</td><td>Static dielectric constant entering Poisson's equation. Harmonic-mean averaged at heterointerfaces.</td></tr>
          <tr><td><i>χ</i></td><td>Electron affinity</td><td>eV</td><td>Energy from vacuum level to conduction-band minimum. Together with <i>E</i><sub>g</sub> it sets the band alignment (CBO / VBO) between layers and — in <b>Full</b> mode — the thermionic-emission cap at heterointerfaces where |Δ<i>E</i><sub>c</sub>| or |Δ<i>E</i><sub>v</sub>| &gt; 0.05 eV.</td></tr>
          <tr><td><i>E</i><sub>g</sub></td><td>Band gap</td><td>eV</td><td>Energy difference between conduction- and valence-band edges. Drives the thermal <i>n</i><sub>i</sub>, the radiative recombination rate, and (with <i>χ</i>) the built-in <i>V</i><sub>bi,eff</sub>.</td></tr>
        </tbody>
      </table>

      <h4>Transport (per layer)</h4>
      <table class="param-table">
        <thead><tr><th>Symbol</th><th>Name</th><th>Unit</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td><i>μ</i><sub>n</sub>, <i>μ</i><sub>p</sub></td><td>Carrier mobilities</td><td>m²·V⁻¹·s⁻¹</td><td>Electron / hole mobilities at the reference temperature (300 K). Diffusion coefficients follow from the Einstein relation <i>D</i> = <i>μ</i> <i>k</i><sub>B</sub><i>T</i>/<i>q</i>. In <b>Full</b> mode they are rescaled as <i>μ</i>(<i>T</i>) = <i>μ</i><sub>300</sub>·(<i>T</i>/300)<sup>−3/2</sup>.</td></tr>
          <tr><td><i>n</i><sub>i</sub></td><td>Intrinsic carrier density</td><td>m⁻³</td><td>Thermal equilibrium density for an undoped layer, <i>n</i><sub>i</sub><sup>2</sup> = <i>N</i><sub>C</sub> <i>N</i><sub>V</sub> e<sup>−<i>E</i><sub>g</sub>/<i>k</i><sub>B</sub><i>T</i></sup>. Sets mass-action law <i>np</i> = <i>n</i><sub>i</sub><sup>2</sup> at equilibrium. Rescaled with <i>T</i> in Full mode using either the identity form (no <i>N</i><sub>C,V</sub>) or the explicit DOS form when available.</td></tr>
          <tr><td><i>N</i><sub>D</sub>, <i>N</i><sub>A</sub></td><td>Donor / acceptor doping</td><td>m⁻³</td><td>Ionised dopant densities (assumed fully ionised). Enter the Poisson space-charge term directly.</td></tr>
        </tbody>
      </table>

      <h4>Recombination (per layer)</h4>
      <table class="param-table">
        <thead><tr><th>Symbol</th><th>Name</th><th>Unit</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td><i>τ</i><sub>n</sub>, <i>τ</i><sub>p</sub></td><td>SRH lifetimes (bulk)</td><td>s</td><td>Shockley–Read–Hall minority-carrier lifetimes used as the bulk value. In <b>Full</b> mode the absorber trap density becomes position-dependent, <i>N</i><sub>t</sub>(<i>x</i>) = <i>N</i><sub>t,bulk</sub> + (<i>N</i><sub>t,iface</sub> − <i>N</i><sub>t,bulk</sub>)·[e<sup>−<i>d</i><sub>L</sub>/<i>L</i><sub>d</sub></sup> + e<sup>−<i>d</i><sub>R</sub>/<i>L</i><sub>d</sub></sup>], and <i>τ</i>(<i>x</i>) is inverse-scaled to N<sub>t</sub>(<i>x</i>). Legacy/Fast keep <i>τ</i> uniform.</td></tr>
          <tr><td><i>n</i><sub>1</sub>, <i>p</i><sub>1</sub></td><td>SRH trap references</td><td>m⁻³</td><td>Occupancy reference densities for the SRH denominator. Set to <i>n</i><sub>i</sub> for mid-gap traps.</td></tr>
          <tr><td><i>B</i><sub>rad</sub></td><td>Radiative coefficient</td><td>m³·s⁻¹</td><td>Bimolecular radiative rate: <i>R</i><sub>rad</sub> = <i>B</i><sub>rad</sub> (<i>np</i> − <i>n</i><sub>i</sub><sup>2</sup>). Non-zero mainly in direct-gap materials like GaAs or MAPbI<sub>3</sub>.</td></tr>
          <tr><td><i>C</i><sub>n</sub>, <i>C</i><sub>p</sub></td><td>Auger coefficients</td><td>m⁶·s⁻¹</td><td>Three-particle recombination, dominant at high injection or heavy doping. Key for c-Si emitters.</td></tr>
        </tbody>
      </table>

      <h4>Ions &amp; Optics (per layer)</h4>
      <table class="param-table">
        <thead><tr><th>Symbol</th><th>Name</th><th>Unit</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td><i>D</i><sub>ion</sub></td><td>Ion diffusion coefficient (cations / +)</td><td>m²·s⁻¹</td><td>Mobility of the positive mobile ionic species (iodide vacancy in perovskites). Set to 0 in inorganic layers. Rescaled with temperature as <i>D</i><sub>ion</sub>(<i>T</i>) = <i>D</i><sub>ion,300</sub>·exp[−<i>E</i><sub>a</sub>/<i>k</i><sub>B</sub>·(1/<i>T</i> − 1/300)] in Full mode.</td></tr>
          <tr><td><i>D</i><sub>ion,−</sub></td><td>Ion diffusion coefficient (anions / −)</td><td>m²·s⁻¹</td><td>Optional second mobile species with reversed drift direction. Only active in <b>Full</b> mode when set per layer; Fast and Legacy skip the dual-ion flux entirely.</td></tr>
          <tr><td><i>P</i><sub>lim</sub></td><td>Site density</td><td>m⁻³</td><td>Maximum ion vacancy density — saturates the Blakemore flux via the steric factor (1 − <i>P</i>/<i>P</i><sub>lim</sub>).</td></tr>
          <tr><td><i>P</i><sub>0</sub></td><td>Background ion density</td><td>m⁻³</td><td>Reference ion density used in the Poisson charge term (<i>P</i> − <i>P</i><sub>0</sub>); ensures charge neutrality at rest.</td></tr>
          <tr><td><i>α</i></td><td>Optical absorption</td><td>m⁻¹</td><td>Beer–Lambert coefficient. Set to 0 in non-absorbing window / buffer layers. Only used when no layer defines <code>optical_material</code> (TMM off) or when Mode is Fast/Legacy.</td></tr>
          <tr><td><code>optical_material</code></td><td>TMM n,k key</td><td>string | null</td><td>Identifier matching a CSV in <code>perovskite_sim/data/nk/</code>. When set (and Mode = <b>Full</b>), the layer participates in the TMM stack and the solver replaces Beer&ndash;Lambert with the Pettersson/Burkhard transfer-matrix method against AM1.5G. When <code>null</code>, the layer is invisible to TMM and the absorber falls back to Beer&ndash;Lambert. Available keys: <code>MAPbI3</code>, <code>TiO2</code>, <code>spiro_OMeTAD</code>, <code>FTO</code>, <code>ITO</code>, <code>SnO2</code>, <code>C60</code>, <code>PCBM</code>, <code>PEDOT_PSS</code>, <code>Ag</code>, <code>Au</code>, <code>glass</code>. <code>n_optical</code> provides a constant-<i>n</i> fallback when no CSV is available.</td></tr>
          <tr><td><code>incoherent</code></td><td>Incoherent-layer flag</td><td>bool</td><td>Marks a layer as optically incoherent (e.g. a glass substrate &gt; 100 &micro;m). The layer uses bulk Beer&ndash;Lambert locally to avoid spurious sub-nanometer interference fringes from the matrix product. Must be the first layer in the stack. Default: <code>false</code>.</td></tr>
          <tr><td><code>role: substrate</code></td><td>Optical-only role</td><td>role value</td><td>Marks a layer as optical-only: it is included in the TMM stack but excluded from the drift-diffusion grid and boundary conditions. A substrate layer must be the first layer, must set <code>incoherent: true</code>, and must set an <code>optical_material</code>.</td></tr>
        </tbody>
      </table>

      <h4>Mode-gated physics summary</h4>
      <p>The table below shows which physics upgrades are active in each tier. See the <b>Algorithm</b> tab for the full equations.</p>
      <table class="param-table mode-table">
        <thead><tr><th>Upgrade</th><th>Legacy</th><th>Fast</th><th>Full</th></tr></thead>
        <tbody>
          <tr><td>Band-offset contact BCs (<i>V</i><sub>bi,eff</sub>)</td><td>—</td><td>—</td><td>✓</td></tr>
          <tr><td>Thermionic-emission flux cap (Richardson–Dushman)</td><td>—</td><td>—</td><td>✓</td></tr>
          <tr><td>Transfer-matrix optics (coherent n,k)</td><td>—</td><td>—</td><td>✓</td></tr>
          <tr><td>Dual-species ion migration</td><td>—</td><td>—</td><td>✓</td></tr>
          <tr><td>Position-dependent trap profile</td><td>—</td><td>—</td><td>✓</td></tr>
          <tr><td>Temperature scaling (<i>V</i><sub>T</sub>, <i>μ</i>, <i>n</i><sub>i</sub>, <i>D</i><sub>ion</sub>)</td><td>—</td><td>—</td><td>✓</td></tr>
          <tr><td>Beer–Lambert generation</td><td>✓</td><td>✓</td><td>fallback</td></tr>
          <tr><td>Single-species ion flux</td><td>✓</td><td>✓</td><td>fallback</td></tr>
        </tbody>
      </table>

      <h4>Experiment controls — J–V Sweep</h4>
      <table class="param-table">
        <thead><tr><th>Symbol</th><th>Name</th><th>Unit</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td><i>N</i><sub>grid</sub></td><td>Spatial nodes</td><td>—</td><td>Number of tanh-clustered finite-element nodes across the whole stack. Higher values smooth J(V) near <i>V</i><sub>bi</sub> but raise Radau cost roughly linearly.</td></tr>
          <tr><td>V sample points</td><td>Bias samples per leg</td><td>—</td><td>Number of voltage stops on each of the forward (0 → <i>V</i><sub>max</sub>) and reverse (<i>V</i><sub>max</sub> → 0) legs. Result is 2·<i>N</i> current values plus the hysteresis index.</td></tr>
          <tr><td><i>v</i><sub>rate</sub></td><td>Scan rate</td><td>V·s⁻¹</td><td>Voltage ramp speed. Determines the time window each Radau sub-interval must integrate; small rates suppress scan-speed hysteresis, fast rates expose it.</td></tr>
          <tr><td><i>V</i><sub>max</sub></td><td>Upper bias</td><td>V</td><td>Forward endpoint. Should exceed the expected <i>V</i><sub>OC</sub> by ~50 mV so the curve crosses <i>J</i> = 0.</td></tr>
        </tbody>
      </table>

      <h4>Experiment controls — Impedance</h4>
      <table class="param-table">
        <thead><tr><th>Symbol</th><th>Name</th><th>Unit</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td><i>N</i><sub>grid</sub></td><td>Spatial nodes</td><td>—</td><td>As above; impedance is usually fine at 40 because the small-signal excitation stays near a fixed operating point.</td></tr>
          <tr><td><i>V</i><sub>dc</sub></td><td>Bias point</td><td>V</td><td>DC operating voltage around which the AC excitation oscillates. Near <i>V</i><sub>OC</sub> reveals recombination resistance; under bright short-circuit reveals transport.</td></tr>
          <tr><td><i>N</i><sub>f</sub></td><td>Frequency points</td><td>—</td><td>Number of log-spaced frequencies between <i>f</i><sub>min</sub> and <i>f</i><sub>max</sub>.</td></tr>
          <tr><td><i>f</i><sub>min</sub>, <i>f</i><sub>max</sub></td><td>Frequency range</td><td>Hz</td><td>Lock-in analysis window. Ionic arcs appear below ~10<sup>2</sup> Hz; electronic arcs dominate above ~10<sup>3</sup> Hz.</td></tr>
        </tbody>
      </table>

      <h4>Experiment controls — Degradation</h4>
      <table class="param-table">
        <thead><tr><th>Symbol</th><th>Name</th><th>Unit</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td><i>N</i><sub>grid</sub></td><td>Spatial nodes</td><td>—</td><td>As above. Long transients are sensitive to node count near heterointerfaces — raise to 60 if snapshot metrics look noisy.</td></tr>
          <tr><td><i>V</i><sub>bias</sub></td><td>Stress voltage</td><td>V</td><td>Applied bias during the aging transient. Maximum-power-point (~<i>V</i><sub>OC</sub>·0.8) is the conventional stress condition.</td></tr>
          <tr><td><i>t</i><sub>end</sub></td><td>Total stress time</td><td>s</td><td>End of the aging window. Snapshot J–V scans are injected at log-spaced probe times.</td></tr>
          <tr><td>N snapshots</td><td>Probe count</td><td>—</td><td>Number of frozen-ion snapshot J–V measurements taken between <i>t</i> = 0 and <i>t</i><sub>end</sub>, used to plot metric decay vs time.</td></tr>
        </tbody>
      </table>

      <h4>Output metrics</h4>
      <table class="param-table">
        <thead><tr><th>Symbol</th><th>Name</th><th>Unit</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td><i>V</i><sub>OC</sub></td><td>Open-circuit voltage</td><td>V</td><td>Bias at which <i>J</i> = 0 under illumination. Upper bound set by the quasi-Fermi-level splitting in the absorber.</td></tr>
          <tr><td><i>J</i><sub>SC</sub></td><td>Short-circuit current density</td><td>mA·cm⁻²</td><td>Current at <i>V</i> = 0. Set by photogeneration minus collection losses.</td></tr>
          <tr><td>FF</td><td>Fill factor</td><td>—</td><td>(<i>V</i><sub>mp</sub> <i>J</i><sub>mp</sub>) / (<i>V</i><sub>OC</sub> <i>J</i><sub>SC</sub>). Measures how "square" the J–V curve is; degraded by series resistance, shunts, and recombination.</td></tr>
          <tr><td>PCE</td><td>Power conversion efficiency</td><td>%</td><td>(<i>V</i><sub>OC</sub> <i>J</i><sub>SC</sub> · FF) / <i>P</i><sub>in</sub>, with <i>P</i><sub>in</sub> = 1000 W·m⁻² for AM1.5G.</td></tr>
          <tr><td>HI</td><td>Hysteresis index</td><td>—</td><td>(PCE<sub>rev</sub> − PCE<sub>fwd</sub>) / PCE<sub>rev</sub>. Non-zero values indicate slow ionic rearrangement during the scan.</td></tr>
          <tr><td><i>Z</i>(<i>ω</i>)</td><td>Small-signal impedance</td><td>Ω·m²</td><td>Ratio of AC voltage to AC current density at frequency <i>ω</i> = 2π<i>f</i>. Real part is resistive loss, imaginary part is reactive storage.</td></tr>
        </tbody>
      </table>

      <h4>Key references</h4>
      <ul class="ref-list">
        <li>
          Courtier, Richardson, Foster (2018). <b>A fast and robust numerical scheme for solving models of charge carrier transport and ion vacancy motion in perovskite solar cells.</b> Applied Mathematical Modelling.
          <a href="https://doi.org/10.1016/j.apm.2018.06.051" target="_blank" rel="noopener">doi:10.1016/j.apm.2018.06.051</a> · IonMonger code: <a href="https://github.com/PerovskiteSCModelling/IonMonger" target="_blank" rel="noopener">github.com/PerovskiteSCModelling/IonMonger</a>
        </li>
        <li>
          Calado et al. (2022). <b>Driftfusion: an open source code for simulating ordered semiconductor devices with mixed ionic-electronic conducting materials.</b> Journal of Computational Electronics.
          <a href="https://doi.org/10.1007/s10825-021-01827-z" target="_blank" rel="noopener">doi:10.1007/s10825-021-01827-z</a> · <a href="https://github.com/barnesgroupICL/Driftfusion" target="_blank" rel="noopener">github.com/barnesgroupICL/Driftfusion</a>
        </li>
        <li>
          Scharfetter, Gummel (1969). <b>Large-signal analysis of a silicon Read diode oscillator.</b> IEEE Transactions on Electron Devices. The original exponential-fit finite-volume flux used by this simulator.
          <a href="https://doi.org/10.1109/T-ED.1969.16566" target="_blank" rel="noopener">doi:10.1109/T-ED.1969.16566</a>
        </li>
        <li>
          Selberherr, S. (1984). <b>Analysis and Simulation of Semiconductor Devices.</b> Springer. Canonical reference for the drift-diffusion system, discretisation, and boundary conditions.
          <a href="https://doi.org/10.1007/978-3-7091-8752-4" target="_blank" rel="noopener">doi:10.1007/978-3-7091-8752-4</a>
        </li>
        <li>
          Sze, Ng (2006). <b>Physics of Semiconductor Devices</b>, 3rd ed., Wiley.  Reference textbook for SRH/radiative/Auger recombination formulas and c-Si / CIGS / GaAs material parameters.
          <a href="https://doi.org/10.1002/0470068329" target="_blank" rel="noopener">doi:10.1002/0470068329</a>
        </li>
        <li>
          Hairer, Wanner (1996). <b>Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems.</b> Springer. Describes the Radau IIA implicit Runge–Kutta method used by <code>scipy.integrate.solve_ivp</code>.
          <a href="https://doi.org/10.1007/978-3-642-05221-7" target="_blank" rel="noopener">doi:10.1007/978-3-642-05221-7</a>
        </li>
        <li>
          Blakemore (1962). <b>The parameters of partially degenerate semiconductors.</b> Proc. Phys. Soc. 79, 1127.  Origin of the steric (site-limiting) factor in the ion-vacancy flux.
          <a href="https://doi.org/10.1088/0370-1328/79/6/303" target="_blank" rel="noopener">doi:10.1088/0370-1328/79/6/303</a>
        </li>
        <li>
          NREL Solar Spectra (AM0 / AM1.5G / AM1.5D reference spectra).
          <a href="https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html" target="_blank" rel="noopener">nrel.gov/grid/solar-resource/spectra-am1.5.html</a>
        </li>
      </ul>
    </div>
  </div>`
}

export async function mountParametersPanel(el: HTMLElement): Promise<void> {
  el.innerHTML = parametersHTML()
}
