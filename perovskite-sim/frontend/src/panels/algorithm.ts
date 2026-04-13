export function algorithmHTML(): string {
  return `
  <div class="card doc-card">
    <h3>Algorithm &amp; Physics</h3>
    <div class="doc-body">
      <h4>Governing equations</h4>
      <p>On each layer the simulator solves Poisson's equation coupled to electron, hole, and mobile-ion continuity:</p>
      <pre class="eqn">
∂/∂<i>x</i> ( <i>ε</i><sub>0</sub> <i>ε</i><sub>r</sub>(<i>x</i>) · ∂<i>φ</i>/∂<i>x</i> )  =  − <i>q</i> [ <i>p</i> − <i>n</i> + (<i>P</i> − <i>P</i><sub>0</sub>) − <i>N</i><sub>A</sub> + <i>N</i><sub>D</sub> ]

∂<i>n</i>/∂<i>t</i>  =  (1/<i>q</i>) ∂<i>J</i><sub>n</sub>/∂<i>x</i>  +  <i>G</i> − <i>R</i>
∂<i>p</i>/∂<i>t</i>  = − (1/<i>q</i>) ∂<i>J</i><sub>p</sub>/∂<i>x</i>  +  <i>G</i> − <i>R</i>
∂<i>P</i>/∂<i>t</i>  = − ∂<i>F</i><sub>ion</sub>/∂<i>x</i>

<i>J</i><sub>n</sub> =  <i>q</i> <i>μ</i><sub>n</sub> <i>n</i> <i>E</i>  +  <i>q</i> <i>D</i><sub>n</sub> ∂<i>n</i>/∂<i>x</i>
<i>J</i><sub>p</sub> =  <i>q</i> <i>μ</i><sub>p</sub> <i>p</i> <i>E</i>  −  <i>q</i> <i>D</i><sub>p</sub> ∂<i>p</i>/∂<i>x</i></pre>

      <h4>Spatial discretization — Scharfetter–Gummel</h4>
      <p>Each interior face uses the exponential-fit flux that is exact for a piecewise-constant electric field over the cell:</p>
      <pre class="eqn">
<i>J</i><sub>n, f</sub>  =  (<i>q</i> <i>D</i><sub>n</sub> / Δ<i>x</i>) · [ <i>n</i><sub>R</sub> <i>B</i>(<i>ξ</i>)  −  <i>n</i><sub>L</sub> <i>B</i>(−<i>ξ</i>) ]

<i>B</i>(<i>ξ</i>)  =  <i>ξ</i> / (<i>e</i><sup><i>ξ</i></sup> − 1)    ,    <i>ξ</i>  =  <i>q</i> (<i>φ</i><sub>R</sub> − <i>φ</i><sub>L</sub>) / (<i>k</i><sub>B</sub> <i>T</i>)</pre>
      <p>The Bernoulli function B is evaluated in series/asymptotic branches to avoid catastrophic cancellation. The grid is a <b>tanh-clustered</b> multilayer mesh that refines toward heterointerfaces, where carrier density gradients are steep.</p>

      <h4>Poisson step</h4>
      <p>Poisson is a sparse <b>tridiagonal</b> system on the non-uniform grid with a harmonic-mean face permittivity and Dirichlet BCs that absorb (<i>V</i><sub>bi</sub> − <i>V</i><sub>app</sub>). It is solved in one pass per RHS evaluation — no inner Newton loop is needed.</p>

      <h4>Recombination</h4>
      <pre class="eqn">
<i>R</i>  =  <i>R</i><sub>SRH</sub>  +  <i>R</i><sub>rad</sub>  +  <i>R</i><sub>Auger</sub>  +  <i>R</i><sub>iface</sub>

<i>R</i><sub>SRH</sub>   =  (<i>n</i><i>p</i> − <i>n</i><sub>i</sub><sup>2</sup>) / [ <i>τ</i><sub>p</sub>(<i>n</i> + <i>n</i><sub>1</sub>)  +  <i>τ</i><sub>n</sub>(<i>p</i> + <i>p</i><sub>1</sub>) ]
<i>R</i><sub>rad</sub>   =  <i>B</i> (<i>n</i><i>p</i> − <i>n</i><sub>i</sub><sup>2</sup>)
<i>R</i><sub>Auger</sub> =  (<i>C</i><sub>n</sub> <i>n</i>  +  <i>C</i><sub>p</sub> <i>p</i>) · (<i>n</i><i>p</i> − <i>n</i><sub>i</sub><sup>2</sup>)
<i>R</i><sub>iface</sub> =  (<i>n</i><i>p</i> − <i>n</i><sub>i</sub><sup>2</sup>) / [ (<i>p</i> + <i>p</i><sub>1</sub>)/<i>v</i><sub>n</sub>  +  (<i>n</i> + <i>n</i><sub>1</sub>)/<i>v</i><sub>p</sub> ]</pre>

      <h4>Ion migration</h4>
      <p>Mobile ions follow a steric (Blakemore) drift-diffusion flux that saturates at the density-of-site limit <i>P</i><sub>lim</sub>:</p>
      <pre class="eqn">
<i>F</i><sub>ion</sub>  =  − <i>D</i><sub>ion</sub> [ ∂<i>P</i>/∂<i>x</i>  +  (<i>q</i> / <i>k</i><sub>B</sub><i>T</i>) <i>P</i> (1 − <i>P</i>/<i>P</i><sub>lim</sub>) ∂<i>φ</i>/∂<i>x</i> ]</pre>
      <p>Boundary conditions are zero-flux at the contacts (ions are confined to the absorber).</p>

      <h4>Time integration — Method of Lines</h4>
      <p>The semi-discretised system is cast as a stiff ODE d<i>y</i>/d<i>t</i> = <i>f</i>(<i>t</i>, <i>y</i>) with state vector <i>y</i> = (<i>n</i>, <i>p</i>, <i>P</i>). We integrate with SciPy's <b>Radau</b> implicit Runge–Kutta method. Near flat-band (<i>V</i> ≈ <i>V</i><sub>bi</sub>) the Jacobian becomes nearly singular and Radau's adaptive estimator can under-report the LTE, so we <b>cap Δ<i>t</i><sub>max</sub> = Δ<i>t</i> / 20</b> on every sub-interval. This eliminates unphysical branch jumps without touching the computed solution values.</p>

      <h4>Measurement protocols</h4>
      <ul>
        <li><b>J–V sweep:</b> forward then reverse, using the previous steady state as the next initial condition — this preserves true ionic memory and produces the hysteresis loop from first principles.</li>
        <li><b>Impedance:</b> at each frequency we integrate three AC cycles, extract <i>V</i>(<i>t</i>) and <i>J</i>(<i>t</i>), and apply a <b>lock-in amplifier</b> (multiply by sin / cos at <i>f</i>, low-pass) to recover amplitude and phase. Displacement current <i>J</i><sub>disp</sub> = <i>ε</i><sub>0</sub> <i>ε</i><sub>r</sub> (∂<i>E</i> / ∂<i>t</i>) is added at the measurement plane.</li>
        <li><b>Degradation snapshot J–V:</b> at each probe time the stack is <b>frozen</b> (<i>D</i><sub>ion</sub> → 0 copy) and a short settle integration is run at each <i>V</i><sub>probe</sub>. This measures the instantaneous electronic response under the current ionic configuration, with no averaging over ion drift.</li>
      </ul>

      <h4>Physics tiers &mdash; Full vs Fast vs Legacy</h4>
      <p>The <b>Mode</b> selector in the device panel gates every physics upgrade below. <b>Full</b> is the default and enables everything the configuration supplies; <b>Legacy</b> disables the upgrades to reproduce the IonMonger-compatible baseline for benchmarking; <b>Fast</b> presently aliases Legacy and reserves the API for future optimisations (e.g. pre-integrated TMM &rarr; effective <i>α</i>).</p>

      <table class="param-table mode-table">
        <thead>
          <tr><th>Feature</th><th>Legacy</th><th>Fast</th><th>Full</th></tr>
        </thead>
        <tbody>
          <tr><td>Contact BCs from band offsets</td><td>Doping only</td><td>Band-offset</td><td>Band-offset</td></tr>
          <tr><td>Thermionic-emission flux cap at heterointerfaces</td><td>&mdash;</td><td>&mdash;</td><td>&#10003;</td></tr>
          <tr><td>Optical generation</td><td>Beer&ndash;Lambert</td><td>Beer&ndash;Lambert</td><td>Transfer-matrix (TMM)</td></tr>
          <tr><td>Mobile ions</td><td>Single species</td><td>Single species</td><td>Dual species (if configured)</td></tr>
          <tr><td>SRH trap density <i>N</i><sub>t</sub>(<i>x</i>)</td><td>Uniform <i>τ</i></td><td>Uniform <i>τ</i></td><td>Position-dependent</td></tr>
          <tr><td>Temperature scaling (<i>V</i><sub>T</sub>, <i>μ</i>, <i>n</i><sub>i</sub>, <i>D</i><sub>ion</sub>)</td><td>Fixed 300 K</td><td>Fixed 300 K</td><td>Uses device <i>T</i></td></tr>
        </tbody>
      </table>

      <h4>1. Band-offset contacts &amp; thermionic emission (Full only)</h4>
      <p>Each layer carries an electron affinity <i>χ</i> and band gap <i>E</i><sub>g</sub>. The built-in potential is derived from the Fermi-level difference across the stack:</p>
      <pre class="eqn">
<i>V</i><sub>bi, eff</sub>  =  <i>E</i><sub>F, left</sub> &minus; <i>E</i><sub>F, right</sub></pre>
      <p>At internal heterointerfaces where |Δ<i>E</i><sub>c</sub>| or |Δ<i>E</i><sub>v</sub>| exceeds 0.05 eV, the Scharfetter&ndash;Gummel flux is capped to the Richardson&ndash;Dushman thermionic-emission limit:</p>
      <pre class="eqn">
<i>J</i><sub>TE</sub>  =  <i>A</i><sup>&lowast;</sup> <i>T</i><sup>2</sup> &middot; exp(&minus; <i>q</i> Δ<i>E</i> / <i>k</i><sub>B</sub><i>T</i>)</pre>
      <p>This prevents SG from overestimating injection when the band discontinuity is resolved in a single cell.</p>

      <h4>2. Transfer-matrix optical generation (Full only)</h4>
      <p>Instead of the scalar <i>G</i>(<i>x</i>) = <i>α</i> Φ exp(&minus;<i>α</i><i>x</i>) law, Full mode solves the coherent thin-film transfer-matrix problem at 200 wavelengths against the AM1.5G spectrum:</p>
      <pre class="eqn">
<i>a</i>(<i>x</i>, <i>&lambda;</i>)  =  (4<i>&pi;</i> <i>n</i>(<i>&lambda;</i>) <i>k</i>(<i>&lambda;</i>)) / (<i>&lambda;</i> <i>n</i><sub>ambient</sub>) &middot; |<i>E</i>(<i>x</i>, <i>&lambda;</i>)|<sup>2</sup>

<i>G</i>(<i>x</i>)  =  &int; <i>a</i>(<i>x</i>, <i>&lambda;</i>) &middot; Φ<sub>AM1.5G</sub>(<i>&lambda;</i>) / (<i>h</i><i>c</i>/<i>&lambda;</i>) d<i>&lambda;</i></pre>
      <p>The <i>n</i>/<i>n</i><sub>ambient</sub> Poynting correction guarantees <i>R</i> + <i>T</i> + <i>A</i> = 1 to within 2 %. <i>n</i>, <i>k</i> come from layer-specific CSV tables when <code>optical_material</code> is set, otherwise from a scalar refractive index fallback.</p>

      <h4>3. Dual-species ion migration (Full only)</h4>
      <p>Perovskites support both positive vacancies (e.g. <i>V</i><sub>I</sub><sup>+</sup>) and negative mobile species (interstitials or methylammonium vacancies). Full mode integrates a second continuity equation with sign-reversed drift:</p>
      <pre class="eqn">
<i>F</i><sub>ion, &minus;</sub>  =  &minus; <i>D</i><sub>ion, &minus;</sub> [ &part;<i>P</i><sub>&minus;</sub>/&part;<i>x</i>  &minus;  (<i>q</i> / <i>k</i><sub>B</sub><i>T</i>) <i>P</i><sub>&minus;</sub> (1 &minus; <i>P</i><sub>&minus;</sub>/<i>P</i><sub>lim, &minus;</sub>) &part;<i>&phi;</i>/&part;<i>x</i> ]</pre>
      <p>The state vector grows from 3<i>N</i> = (<i>n</i>, <i>p</i>, <i>P</i><sub>+</sub>) to 4<i>N</i> = (<i>n</i>, <i>p</i>, <i>P</i><sub>+</sub>, <i>P</i><sub>&minus;</sub>) automatically whenever <i>D</i><sub>ion,&minus;</sub> &gt; 0 in the YAML.</p>

      <h4>4. Position-dependent traps &amp; temperature scaling (Full only)</h4>
      <p>When a layer sets <code>trap_N_t_interface</code>, <code>trap_N_t_bulk</code>, and <code>trap_decay_length</code>, the SRH lifetime decays exponentially from both layer boundaries:</p>
      <pre class="eqn">
<i>N</i><sub>t</sub>(<i>x</i>)  =  <i>N</i><sub>t, bulk</sub>  +  (<i>N</i><sub>t, iface</sub> &minus; <i>N</i><sub>t, bulk</sub>) &middot; [ <i>e</i><sup>&minus;<i>d</i><sub>L</sub>/<i>L</i><sub>d</sub></sup>  +  <i>e</i><sup>&minus;<i>d</i><sub>R</sub>/<i>L</i><sub>d</sub></sup> ]

<i>&tau;</i>(<i>x</i>)  =  <i>&tau;</i><sub>bulk</sub> &middot; <i>N</i><sub>t, bulk</sub> / <i>N</i><sub>t</sub>(<i>x</i>)</pre>
      <p>The device temperature <i>T</i> (editable in the Device panel) drives every temperature-sensitive parameter:</p>
      <pre class="eqn">
<i>V</i><sub>T</sub>(<i>T</i>)      =  <i>k</i><sub>B</sub><i>T</i> / <i>q</i>
<i>&mu;</i>(<i>T</i>)        =  <i>&mu;</i><sub>300</sub> &middot; (<i>T</i> / 300)<sup>&gamma;</sup>        ,  &gamma; = &minus;1.5  (acoustic phonons)
<i>n</i><sub>i</sub>(<i>T</i>)       &prop;  <i>T</i><sup>3/2</sup> &middot; exp(&minus; <i>E</i><sub>g</sub> / 2 <i>k</i><sub>B</sub><i>T</i>)
<i>D</i><sub>ion</sub>(<i>T</i>)    =  <i>D</i><sub>ion, 300</sub> &middot; exp(&minus; <i>E</i><sub>a</sub> / <i>k</i><sub>B</sub> &middot; (1/<i>T</i> &minus; 1/300))</pre>
      <p>The resulting dV<sub>oc</sub>/d<i>T</i> is negative on the order of &minus;1 to &minus;3 mV/K on the IonMonger benchmark, consistent with experiment.</p>

      <h4>What is <i>not</i> done</h4>
      <ul>
        <li>No clipping, smoothing, or monotone envelope on any output.</li>
        <li>No empirical ideality-factor or shunt-resistance fits &mdash; FF comes directly from the computed <i>J</i>(<i>V</i>).</li>
        <li>No reduced-order model. Every data point is a full PDE solve.</li>
      </ul>
    </div>
  </div>`
}

export async function mountAlgorithmPanel(el: HTMLElement): Promise<void> {
  el.innerHTML = algorithmHTML()
}
