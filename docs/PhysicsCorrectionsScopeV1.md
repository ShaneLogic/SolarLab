# Prospective Physics Corrections Scope (V1)

**Task ID:** `P08-01` (Wave 3 Lead-In — Prospective Physics Questions, Falsification Citations & Executable Counterexamples)
**Baseline Branch / Commit:** `devel` @ `0b03de5028e2124e743483671de890278dd038de`
**Executable Witness Suite:** `tests/refactor/test_p08_counterexamples.py`

---

## 1. Governance, Commit Isolation & Evidence Boundaries

Stages `P03` through `P06` migrate the SolarLab simulator under two strict non-physics change classes (`ArchitectureRefactorExecutionV2.md` §1.2):

- **`M`-class (Mechanical / Bit-Identical):** Packaging, typed configuration, process supervision, and legacy channel reproduction with zero mathematical change.
- **`N`-class (Numerical Conditioning & Representation):** Positive log-activity carrier coordinates $(z_n, z_p)$, physical-inventory Backward Euler (`S_old * expm1(log_ratio) / dt`), cancellation-free Scharfetter–Gummel electrochemical steps, and sparse analytical Jacobians (`AssembledDC1DProblem`, `AssembledIonic1DProblem`, 2D sparse operators) preserving the underlying continuum physical laws.

All prospective **physical** corrections (`P`-class) are quarantined into Stage `P08` (`P08-02` through `P08-05`, gated by `G8`) under five binding rules:

1. **One-Commit-Per-Physics-Fix Isolation (`V1` §12 `P08-01`–`P08-05`, `V2` §1.2):** Every `P`-class correction must land in its own isolated commit with a distinct model or protocol version token (`v1` $\to$ `v2` or a newly named model identifier). Multiple physical defaults must **never** be flipped in a single commit, and `P`-class changes must never be bundled with `M`/`N` refactoring.
2. **Retention of Historical Channels & Failed Verdicts (`AGENTS.md`, `V1` §19.35):** Pre-`P08` channels (`legacy_mol_integer_endpoint_v1`, `legacy_twod_half_endpoint_v1`, nodal harmonic face permittivity, legacy density-weighted thermionic cap, monochromatic `beer_lambert`, coherent `tmm`, and summed-potential hole SG differencing) remain selectable for bit-reproducible historical replay. Historical failed or `inconsistent` runs retain their original status and citations permanently; a new `P08` model version creates a new evidence channel rather than relabeling old runs.
3. **`RS` (`R1`) Pre-`P08` Pinning (`V2` Decision `D20`):** Under `ArchitectureRefactorExecutionV2.md` `D20`, Stage `P08` depends on `G4`/`G5` and is decoupled from `RS-G7`. The `R1` research lane (`one_dimensional_mechanism_r1`) remains pinned to pre-`P08` model versions (`physical_boundary_two_sided_v1` and pre-`P08` constitutive tokens) unless a post-`P08` research protocol version is separately specified and admitted.
4. **Permanent Falsification Enforcement:** Scientifically falsified closures (`interface_shared_occupancy` and bulk-node Gaussian $E_t$ occupancy integration) remain permanently refused pre-solve via `CapabilityError`. Improved nonlinear/time-integrator convergence on the new kernel is **never** grounds to re-enable a falsified physical law.
5. **Source-Bound Certificate Re-Verification (`P08-05` / `G8`):** Any `P`-class change invalidates stale source-bound reference certificates for affected consumers until re-verified under `P08-05`, keeping internal numerical convergence, calibrated external-solver reproduction (`SCAPS`, `IonMonger`, `Driftfusion`), and experimental validation strictly separate (`perovskite-sim/CLAUDE.md` 2026-08-14 audit).

---

## 2. Master `P08` Scope Matrix

| Item ID | Target Physics / Arithmetic Question | Change Class | `RS` (`R1`) Impact (`V2` `D20`) | Executable Witness in `tests/refactor/test_p08_counterexamples.py` |
| --- | --- | --- | --- | --- |
| `P08-02a` | 1D/2D non-uniform mesh & heterointerface distance-weighted series dielectric capacitance vs nodal harmonic / arithmetic face means | `P` | Pinned to pre-`P08` (`physical_boundary_two_sided_v1` already uses `series_face_capacitance` at split planes) | `test_p08_02_nonuniform_series_capacitor_dielectric_witness` |
| `P08-02b` | Endpoint half-cell vs full-cell control-volume geometry (`sum(w) == L`) coupled across storage, sources, Poisson, sheet traps & ports | `P` | Pinned to pre-`P08` (`physical_boundary_two_sided_v1` already has `sum(w) == L`) | `test_p08_02_endpoint_control_volume_and_hole_step_rounding_witness` |
| `P08-02c` | `hole_separate_increment` (`BandStepsV1`) `4.0107e-4 A/m^2` binary64 differencing disposition & internal vs PV terminal current sign conventions | `N`/`P` boundary (versioned opt-in / isolated commit) | Pinned to pre-`P08` | `test_p08_02_endpoint_control_volume_and_hole_step_rounding_witness` |
| `P08-03a` | Heterointerface unequal-DOS ($N_{C,L} \neq N_{C,R}$, $m^*_L \neq m^*_R$) reciprocal Bethe/Richardson TE detailed balance on **bare TE** + deep forward-bias conditioning | `P` | Pinned to pre-`P08` | `test_p08_03_unequal_dos_bare_te_equilibrium_and_unit_scaling_witness` |
| `P08-03b` | Fixed `5 meV` compatible-reservoir contact thermodynamic gate vs permanent `6.025 meV` SCAPS `inconsistent` historical label (`F08`) | Invariant guard (no gate relaxation) | Unchanged (`5 meV` gate enforced) | `test_p08_03_contact_gate_and_falsified_shared_occupancy_refusals` |
| `P08-03c` | Permanent falsification & pre-solve `CapabilityError` refusal of `interface_shared_occupancy` (1D & 2D) and bulk-node Gaussian $E_t$ occupancy | Falsified / permanently rejected | Refused | `test_p08_03_contact_gate_and_falsified_shared_occupancy_refusals` |
| `P08-04a` | Spectral control-volume Beer–Lambert (`archive/wip-beer-lambert` `658baa84`) + optical photon budget + positive log-activity transient readiness | `P` | Pinned to pre-`P08` | `test_p08_04_spectral_beer_lambert_photon_budget_and_positive_log_activity_witness` |
| `P08-04b` | Explicit `optical_model` selection (`beer_lambert` \| `beer_lambert_spectral` \| `tmm`) & external generation profile-file loader (`not_available` / net-new scope) | Scope boundary | Unchanged | `test_p08_04_spectral_beer_lambert_photon_budget_and_positive_log_activity_witness` |
| Deferred | `F14` switching-surface kinks (`R_tot=0`, hard TE `min`, Poole–Frenkel `\|E\|=0`), coupled time-domain PDE $T(x,t)$, arbitrary plotting | Out of `P08` / fail-closed guards retained | Unchanged | Documented in §6 |

---

## 3. `P08-02` — Electrostatics, Control-Volume Geometry, Current Sign Conventions & `hole_separate_increment`

### 3.1 `P08-02a`: Non-Uniform Mesh & Heterointerface Face Permittivity

- **Old Semantics (`devel` @ `0b03de5`):**
  - `PoissonOperator1D.from_harmonic_nodes` (`src/solarlab/physics/electrostatics/poisson.py` L67–80) computes unweighted nodal harmonic face permittivity $\tilde{\varepsilon}_{i+1/2} = 2\varepsilon_{r,i}\varepsilon_{r,i+1}/(\varepsilon_{r,i}+\varepsilon_{r,i+1})$ and $C_{i+1/2} = \varepsilon_0 \tilde{\varepsilon}_{i+1/2}/h_i$ for `INTEGER_ENDPOINT_V1` (`legacy_mol_integer_endpoint_v1`), `HALF_ENDPOINT_V1` (`legacy_twod_half_endpoint_v1`), and extruded 2D stack faces in `CompiledDevice2D` (`src/solarlab/equations/dc_2d.py` L148–149).
  - On shared-node grids (`plane.shared_node = k`), `RIGHT_LAYER_WINS` (`src/solarlab/mesh/geometry.py` L312, L325) sets $\varepsilon_r[k] = \varepsilon_R$, polluting the left-layer face $[x_{k-1}, x_k]$ with $\varepsilon_R$.
  - Only `TWO_SIDED_TRACE_V1` and `PHYSICAL_BOUNDARY_TWO_SIDED_V1` replace split interface faces with `series_face_capacitance(d_L, eps_L, d_R, eps_R)` (`poisson.py` L109–114).
- **Defect / Counterexample Evidence:**
  1. *Non-uniform split face ($d_L \neq d_R$, $\varepsilon_L \neq \varepsilon_R$):* Exact Gauss-law continuity $D = -\varepsilon_0 \varepsilon_r \partial_x \phi = \text{const}$ across a planar dielectric boundary splitting face $h = d_L + d_R$ yields the exact series capacitance

     $$
     C_{\mathrm{exact}} = \frac{\varepsilon_0}{d_L/\varepsilon_L + d_R/\varepsilon_R}.
     $$

     Unweighted `harmonic_face_mean` assumes $d_L = d_R = h/2$, producing an $O(1)$ capacitance and potential error whenever mesh clustering makes $d_L \neq d_R$ across a high-contrast heterointerface (e.g. organic HTL $\varepsilon_1 = 3.0$ vs perovskite $\varepsilon_2 = 25.0$). Arithmetic averaging $0.5(\varepsilon_L + \varepsilon_R)/h$ treats series dielectrics as a parallel capacitor and fails even when $d_L = d_R$.
  2. *Shared-node `RIGHT_LAYER_WINS` interface shift:* When node $x_k = X_{\text{if}}$ sits on the interface, $[x_{k-1}, x_k]$ lies 100% inside material $L$ ($\varepsilon_L$) and $[x_k, x_{k+1}]$ lies 100% inside material $R$ ($\varepsilon_R$). Setting $\varepsilon_r[k] = \varepsilon_R$ gives $C_{k-1/2} = \varepsilon_0 \frac{2\varepsilon_L\varepsilon_R}{\varepsilon_L+\varepsilon_R} / dx_L$, shifting the effective dielectric boundary by $dx_L / 2$ into layer $L$ across the two adjacent faces:

     $$
     C_{k-1/2}^{-1} + C_{k+1/2}^{-1} = \frac{dx_L / 2}{\varepsilon_0 \varepsilon_L} + \frac{dx_L / 2 + dx_R}{\varepsilon_0 \varepsilon_R} \neq \frac{dx_L}{\varepsilon_0 \varepsilon_L} + \frac{dx_R}{\varepsilon_0 \varepsilon_R}.
     $$
- **New Target Semantics:**
  - Every 1D and 2D electrostatic face (stack and lateral) uses exact sub-face material-extent series capacitance $C_f = \varepsilon_0 / (d_{f,L}/\varepsilon_L + d_{f,R}/\varepsilon_R)$. Within a homogeneous layer ($\varepsilon_L = \varepsilon_R = \varepsilon$), this reduces identically to $\varepsilon_0 \varepsilon / h_f$; on a shared-node interface $x_k = X_{\text{if}}$, $[x_{k-1}, x_k]$ uses $\varepsilon_L$ over $dx_L$ and $[x_k, x_{k+1}]$ uses $\varepsilon_R$ over $dx_R$.
- **Allowed Scope:** `src/solarlab/discretization/face_average.py`, `src/solarlab/physics/electrostatics/`, `src/solarlab/device/compile_1d.py`, `src/solarlab/device/compile_2d.py`, `src/solarlab/equations/dc_2d.py`, and small-signal/displacement-current observers reading `PoissonOperator1D`.
- **Falsification & Non-Regression Boundaries:** Must preserve exact homogeneous-medium reduction ($\varepsilon_L = \varepsilon_R \implies C_f = \varepsilon_0 \varepsilon / h_f$ to 0–1 ULP), exact Gauss identity $D_R - D_L = \sigma_{\text{sheet}}$, and keep `INTEGER_ENDPOINT_V1` / `HALF_ENDPOINT_V1` historical constructors available for `M`/`N` parity baselines.
- **Affected Consumers:** 1D/2D DC steady-state, transient displacement current $J_{\text{disp}} = \partial_t D$, Mott–Schottky / geometric capacitance $C(\omega)$, and ionic J–V electrostatic screening.
- **Independent Acceptance Criteria & Expected Direction of Change:**
  - Two-layer charge-free capacitor ($\varepsilon_1 \neq \varepsilon_2$) reproduces analytical $\phi(x)$, $D$, and $C_{\text{geom}} = (\sum_\ell L_\ell / (\varepsilon_0 \varepsilon_\ell))^{-1}$ to $\le 10^{-14}$ relative error on arbitrary non-uniform 1D and 2D meshes.
  - Expected physical shift: high-contrast transport/absorber stacks ($\varepsilon_{\text{TL}} \approx 3\text{–}9$, $\varepsilon_{\text{PVK}} \approx 25$) see small, mesh-convergent corrections to high-frequency geometric capacitance $C_{\text{hf}}$ and interfacial voltage partition, vanishing as $O(\Delta x_{\text{if}} / L)$ under refinement.
- **Versioning & Commit Isolation Rule:** Single `P`-class commit introducing versioned distance-weighted/layer-true Poisson geometry without simultaneously altering endpoint cell measures, TE normalization, or optical generation.

---

### 3.2 `P08-02b`: Endpoint Control-Volume Geometry (Half-Cell vs Full-Cell)

- **Old Semantics (`devel` @ `0b03de5`):**
  - Four named historical 1D geometries coexist in `src/solarlab/mesh/geometry.py`:
    - `INTEGER_ENDPOINT_V1` (`legacy_mol_integer_endpoint_v1`): $w_0 = dx_0$, $w_{N-1} = dx_{N-1}$, so $\sum_{i=0}^{N-1} w_i = L + \frac{1}{2}(dx_0 + dx_{N-1}) > L$ (`geometric_cells=False`).
    - `TWO_SIDED_TRACE_V1` (`qf_two_sided_trace_integer_endpoint_v1`): true interior interface planes, but integer contact widths ($\sum w_i > L$).
    - `HALF_ENDPOINT_V1` (`legacy_twod_half_endpoint_v1`) and `PHYSICAL_BOUNDARY_TWO_SIDED_V1`: $w_0 = dx_0 / 2$, $w_{N-1} = dx_{N-1} / 2$, so $\sum_{i=0}^{N-1} w_i = L$ (`geometric_cells=True`).
- **Defect / Counterexample Evidence:**
  - Under `INTEGER_ENDPOINT_V1`, `beer_lambert_absorbed_photon_flux` (`src/solarlab/physics/generation/beer_lambert.py` L190–201) integrates over geometric dual faces $[x_0, x_0 + dx_0/2]$ of width $dx_0/2$, then divides by $w_0 = dx_0$, halving nodal $G_0$ and $G_{N-1}$ while conserving $\sum_i w_i G_i$. Conversely, any spatially uniform density $\rho_0$ integrated with `cell_measures_m` yields $\sum_i w_i \rho_0 = \rho_0 [L + \frac{1}{2}(dx_0 + dx_{N-1})] > \rho_0 L$, and `contact_displacement` (`poisson.py` L172–183) extends Gauss's law across `widths[0]` and `widths[-1]` using `cell_measures_m`. Mixing half-cell measures in one operator with full-cell measures in a coupled operator breaks exact terminal Maxwell / particle conservation.
- **New Target Semantics:**
  - Physical control volumes satisfy $w_i = b_{i+1} - b_i$ everywhere, including contact half-cells ($b_0 = x_0$, $b_1 = (x_0 + x_1)/2$, $\sum_{i=0}^{N-1} w_i = L$) and true material-interface planes, bound **simultaneously** across carrier storage, mobile-ion conservation, Beer–Lambert / TMM / photon-recycling generation, Poisson/contact Gauss extension, sheet-trap areal conversion, and 1D/2D terminal port accounting.
- **Allowed Scope:** `src/solarlab/mesh/`, `src/solarlab/device/compile_1d.py`, `src/solarlab/device/compile_ionic_1d.py`, and coupled `equations/` bindings.
- **Falsification & Non-Regression Boundaries:** Must never change `w_0, w_{N-1}` in generation or storage alone while leaving `contact_displacement` or `step_accounting` on `INTEGER_ENDPOINT_V1`; global particle and Maxwell total-current conservation in `step_accounting` (`backward_euler.py` L220–226) must hold to machine precision ($\le 10^{-13}$ relative).
- **Affected Consumers:** 1D DC/transient/AC/ionic solvers, contact recombination/extraction accounting, and 1D-to-2D column consistency.
- **Independent Acceptance Criteria & Expected Direction of Change:**
  - Uniform-column 2D (`Nx >= 2`, `HALF_ENDPOINT_V1`) and physical-endpoint 1D (`Ny` matching `N`) agree to solver tolerance on dark/illuminated $J(V)$, inventory, and terminal displacement current without endpoint-measure discrepancy.
- **Versioning & Commit Isolation Rule:** Isolated `P`-class commit switching default compiled 1D geometry to a named physical half-endpoint geometry token while keeping `INTEGER_ENDPOINT_V1` explicitly selectable for legacy parity tests.

---

### 3.3 `P08-02c`: `hole_separate_increment` (`BandStepsV1`) `4.0107e-4 A/m^2` Rounding Disposition & Current Sign Conventions

- **Old Semantics (`devel` @ `0b03de5`):**
  - `BandStructure.transport_potentials(phi)` (`src/solarlab/physics/transport/drift_diffusion.py` L46; `electrochemical_flux.py` L256) forms nodal hole potential $\psi_{p,i} = \phi_i + \chi_{t,i} + E_{g,t,i}$ ($\approx 5.4\text{–}6.0\text{ V}$) first and then differences $\psi_{p,i+1} - \psi_{p,i}$.
- **Defect / Counterexample Evidence (`ArchitectureRefactorExecutionV1.md` §12 L784, §19.35 L1621):**
  - Because $\chi_t + E_{g,t} \approx 5.5\text{ eV}$ has $\mathrm{ulp}(5.5) = 2^{-50} \approx 8.88\times 10^{-16}\text{ V}$, adding $\chi_t + E_{g,t}$ prior to subtracting adjacent nodes quantizes small intra-layer electrostatic steps $\Delta\phi = \phi_{i+1} - \phi_i$. At heavily doped transport-layer faces ($p \sim 10^{24}\text{ m}^{-3}$, $\Delta x \sim 1\text{ nm}$), that 1-ULP quantization perturbs $J_p$ by $O(10^{-4})\text{ A/m}^2$.
  - On the saved SCAPS fixed-state native residual screens (`V1` §19.35 L1621), native summed-potential differencing gave `8.79892088904392e-5 A/m^2` on `Final61` and `4.010734989424907e-4 A/m^2` on `Final121` (exceeding the exact $1/3000\text{ A/m}^2 = 3.3333333333333335\times 10^{-4}\text{ A/m}^2$ screen on `Final121`), whereas `BandStepsV1` (`hole_separate_increment`, forming $(\phi_{i+1}-\phi_i) + (\chi_{t,i+1}-\chi_{t,i}) + (E_{g,t,i+1}-E_{g,t,i})$ before summation in `FullRhs01`) reduced carrier-balance sums to `2.9697029217142876e-5 A/m^2` (`61`) and `1.9735726377304774e-5 A/m^2` (`121`).
- **Explicit `N` vs `P` Disposition & Sign Convention Rule:**
  1. Over $\mathbb{R}$, separate-increment differencing is algebraically identical to summed-potential differencing (it is a floating-point conditioning improvement, not a new continuum physical law), **but** because it changes bit-level face currents by up to $4.01\times 10^{-4}\text{ A/m}^2$ on fine grids and moves `Final121` across the historical $1/3000\text{ A/m}^2$ screen, it must **never** be applied silently inside an unversioned edit or used to relabel the historical `Final121` native failure (`V1` §19.35). If adopted in `P08-02`, separate-increment step formation $(\Delta\phi) + (\Delta\chi_t) + (\Delta E_{g,t})$ must be isolated in its own versioned commit with explicit Jacobian consistency (`d step / d phi = +/- 1 / V_T` unchanged) and historical `Final121` failure preserved verbatim.
  2. **Current Sign Convention:** All internal 1D/2D operators keep `+x` / `+y` coordinate orientation (`F_n, F_p` particle fluxes along `+x`, $J_n = -q F_n$, $J_p = +q F_p$, $D = -C \Delta\phi$, $J_{\text{disp}} = \partial_t D$). Photovoltaic sign conventions ($J_{\text{PV}} > 0$ at short circuit / power generation quadrant via `voltage_polarity`) belong exclusively to experiment readout/reporting layers and must never flip internal conservation signs.

---

## 4. `P08-03` — Heterointerface Thermionic Detailed Balance, Contact `5 meV` Gate & Falsified Closures

### 4.1 `P08-03a`: Unequal-DOS / Effective-Mass / Physical-Barrier Thermionic Detailed Balance

- **Old Semantics (`devel` @ `0b03de5`):**
  - Deduplicated 1D TE (`compile_1d.py` L367–398; `thermionic_emission_bound_m2_s` in `src/solarlab/physics/interface_transport/thermionic.py` L173–214 with `dos_norm_m3=None`) evaluates left-layer `A_star` only and computes

    $$
    J_{\mathrm{TE,legacy}} = A^* T^2 \left(n_L e^{-\max(\Delta E_t, 0)/V_T} - n_R e^{-\max(-\Delta E_t, 0)/V_T}\right),
    $$

    which has unphysical units $\text{A}\cdot\text{m}^{-5}$ ($\sim 10^{28}\text{–}10^{35}$ numerically in SI, rendering $\min(|J_{\text{SG}}|, |J_{\text{TE}}|)$ nearly inert and violating length-unit invariance).
  - Single-DOS `settings.te_physical_norm` (`ArchiveDigest.md` X3; `perovskite-sim/CLAUDE.md` F02) divides by a single $N_{\text{dos}}$ to form $v_R = A^* T^2 / (q N_{\text{dos}})$ [$\text{m/s}$], and is currently refused with `CapabilityError` in `compile_device_1d` (`compile_1d.py` L147).
  - Two-sided trace (`thermionic.py` L438–615 `fermi_dirac_richardson_flux` & `fermi_richardson_flux`; `compile_1d.py` L589–594, L670–698) already resolves per-side $N_{C,L}, N_{C,R}, N_{V,L}, N_{V,R}$, true physical band offsets, and reciprocal $\tilde{A}^* = \min(A^*_L, A^*_R)$.
- **Defect / Counterexample Evidence:**
  1. *Unit-system non-invariance:* Rescaling densities and DOS together ($n \to \lambda n$, $N_C \to \lambda N_C$, e.g. $\text{m}^{-3} \leftrightarrow \text{cm}^{-3}$ with $\lambda = 10^{-6}$) changes `thermionic_emission_bound_m2_s(..., dos_norm_m3=None)` by $\lambda$, whereas physical emission flux $J_{\text{TE}} \propto (n / N_C)$ is strictly scale-invariant.
  2. *Failure of single-DOS / non-reciprocal $A^*$ on unequal-DOS heterojunctions ($N_{C,L} \neq N_{C,R}$, $m^*_L \neq m^*_R$):* At thermal equilibrium ($E_{F,L} = E_{F,R}$), Boltzmann carrier densities across physical conduction-band step $\Delta E_{\text{step}} = \Delta E_{C,\text{phys}} - (\phi_R - \phi_L)$ satisfy

     $$
     \frac{n_L}{N_{C,L}} e^{-\max(\Delta E_{\mathrm{step}}, 0)/V_T} = \frac{n_R}{N_{C,R}} e^{-\max(-\Delta E_{\mathrm{step}}, 0)/V_T}.
     $$
     Evaluating bare TE with physical $\Delta E_{\text{step}}$ and a **single** shared divisor $N_{\text{dos}} = N_{C,L}$ yields a nonzero spurious equilibrium current $\propto (1 - N_{C,R}/N_{C,L})$. Likewise, using side-specific $A^*_L \neq A^*_R$ separately on the forward and reverse legs yields a spurious equilibrium current $\propto (A^*_L - A^*_R)$ because transverse momentum conservation across a specular planar barrier restricts both directions to the smaller phase-space cone $\tilde{A}^* = \min(A^*_L, A^*_R)$ (Bethe/Richardson reciprocity).
  3. *Historical conditioning failures (`ArchiveDigest.md` X3):* Four prior legacy attempts to flip `te_physical_norm` on by default failed solver conditioning in deep forward injection (`V_max = 3.0 V` non-finite density Newton/Radau steps).
- **New Target Semantics:**
  - Interface thermionic transport separates: (1) true physical electrostatic + band-edge barrier $\Delta E_{\text{step}}$, (2) per-side DOS activities $n_L/N_{C,L}, n_R/N_{C,R}$ (or degenerate supply $F_1(\eta - \Delta E_{\text{bar}}/V_T)$), (3) reciprocal effective-mass Richardson constant $\tilde{A}^* = \min(A^*_L, A^*_R)$ (or explicit SCAPS effective-velocity variant `scaps_thermionic_flux` when reproducing SCAPS), (4) explicit transmission $\mathcal{T}$, and (5) cancellation-free electrochemical difference $\Delta E_F / k_B T$.
- **Allowed Scope:** `src/solarlab/physics/interface_transport/`, `src/solarlab/device/compile_1d.py`, `src/solarlab/equations/dc_1d.py`, `src/solarlab/equations/assembly/`, and `solvers/nonlinear/` forward-bias continuation guards if required.
- **Falsification & Non-Regression Boundaries:**
  - Must be verified on **bare TE** (direct cross-plane TE kernel without SG `min` masking hiding equilibrium or Jacobian defects).
  - Must converge without `NaN`/`Inf` across both photovoltaic ($0 \to V_{oc}$) and deep forward-injection ($V > V_{bi}$, including the historical `V_max = 3.0 V` stress regime or explicit physical contact/series-resistance regime).
- **Affected Consumers:** 1D/2D heterojunction J–V ($V_{oc}$, $FF$, S-kinks across CBO/VBO sweeps), dark diode ideality, C–V / impedance across barriers, and transient open-circuit photovoltage decay.
- **Independent Acceptance Criteria & Expected Direction of Change:**
  - Bare TE equilibrium current $|J_{\text{TE}}(V=0, G=0)| \le 10^{-15} \times J_{\text{one-way}}$ across arbitrary $N_{C,L}/N_{C,R} \in [10^{-2}, 10^2]$ and $m^*_L \neq m^*_R$.
  - Analytical Jacobian wrt $(\phi, z_n, z_p)$ matches central finite differences to $\le 10^{-8}$ relative Frobenius norm.
  - Expected physical direction: activating physical emission-velocity TE lowers the TE ceiling from inert $\sim 10^{10}\text{ A/m}^2$ to physical $\sim 10^3\text{–}10^5\text{ A/m}^2$, increasing extraction resistance / S-shape sensitivity when $|\Delta E_C|$ or $|\Delta E_V|$ is large.
- **Versioning & Commit Isolation Rule:** Dedicated `P`-class commit with an explicit TE normalization/reciprocity model version; legacy `dos_norm_m3=None` retained only under explicit legacy preset selection.

---

### 4.2 `P08-03b`: Contact Thermodynamic `5 meV` Gate & Historical `6.025 meV` SCAPS Label (`F08`)

- **Current & Target Semantics (Invariant Guard — No `P08` Relaxation Permitted):**
  - `assess_contact_thermodynamics` (`src/solarlab/physics/contacts/thermodynamics.py` L43, L137–228) enforces `CONTACT_THERMODYNAMIC_TOLERANCE_EV = 5.0e-3 eV` on the four contact quasi-Fermi levels $(E_{Fn,L}, E_{Fp,L}, E_{Fn,R}, E_{Fp,R})$ and any declared metal work functions; callers may tighten `tolerance_eV` below `5 meV` but attempting `tolerance_eV > 5.0e-3` raises `ValueError`.
  - Compatible `semiconductor_work_function` contacts satisfy span $< 10^{-12}\text{ eV}$ (`status="certified"`).
  - On historical `scaps_mirror_v2` (`legacy_manual` $V_{bi} = 1.300\text{ V}$ vs band-derived $1.2939750419\dots\text{ V}$; `ArchitectureRefactorReviewEvidenceV1.json` L143–157; `test_contact_thermodynamics_label.py`), the $6.024958093130\times 10^{-3}\text{ eV}$ equilibrium QFL span remains permanently `status="inconsistent"` with `historical_label="legacy_manual_v_bi_contact_offset"` and fails `require_contact_thermodynamic_certificate` with `ContactThermodynamicError`. Stage `P08` must **never** widen the `5 meV` gate to `7 meV` to pass manual-offset presets.

---

### 4.3 `P08-03c`: Permanent Falsification of `interface_shared_occupancy` & Bulk-Node Gaussian $E_t$

- **Falsification Record (`perovskite-sim/CLAUDE.md` L86–89; `Specs/ArchitectureRefactorV1.md` L846–847; `ArchiveDigest.md` L176):**
  - `interface_shared_occupancy`: **Falsified** — collapses the SCAPS CBO $V_{oc}$ sweep trend from $734\text{ mV}$ to $203\text{ mV}$ because algebraic trap occupancy evaluated directly on adjacent bulk-node densities ignores electrostatic/band projection to the interface plane.
  - Bulk-node Gaussian $E_t$ defect integration: **Pre-falsified** by the same probe lineage — bulk-node carrier sampling is $E_t$-blind by $6\text{–}13$ orders of magnitude in heterointerface depletion spikes.
  - `interface_two_sided` (legacy additive mirror channel): Measured exact no-op across 1D sweeps (distinct from `TWO_SIDED_TRACE_V1` QSS trace physics and clamp-passive `two_sided_cross_node` 2D sheet SRH).
- **Mandatory Boundary:** `compile_device_1d`, `compile_device_2d`, and `compile_ionic_device_2d` / `build_two_sided_interface_srh_2d` must continue raising `CapabilityError` before any solve whenever `settings.interface_shared_occupancy=True`. It is permanently excluded from `P08`.

---

## 5. `P08-04` — Spectral Control-Volume Beer–Lambert, Optical Photon Budget & Positive-Density Transient Readiness

### 5.1 `P08-04a`: Spectral Beer–Lambert Re-Introduction (`archive/wip-beer-lambert`)

- **Old Semantics (`devel` @ `0b03de5`):**
  - `src/solarlab/physics/generation/beer_lambert.py` (`BeerLambertBinding`) supports monochromatic scalar or piecewise-constant/nodal $\alpha$ [$\text{m}^{-1}$] with scalar photon flux $\Phi$ [$\text{m}^{-2}\text{s}^{-1}$].
  - `src/solarlab/device/optical_generation.py` (`bind_optical_generation_1d`) binds coherent normal-incidence spectral TMM (`OpticalStackBinding`), requiring optical $(n(\lambda), k(\lambda))$ or constant $(n_{\text{opt}}, \alpha)$ on every layer.
  - Wavelength-resolved single-pass Beer–Lambert (`beer_lambert_spectral`) was parked at Git tag `archive/wip-beer-lambert` (`658baa84c5106ad3a4c9946d18a0c2860d9a4bff`, 2026-10-05; `BeerLambertReferenceProtocolV1.md`, `perovskite_sim/physics/beer_lambert_spectral.py`).
- **Defect / Counterexample & Historical Evidence:**
  1. *Spectral vs scalar Beer–Lambert:* For dispersive semiconductors where $\alpha(\lambda) = 4\pi k(\lambda)/\lambda$ spans $\sim 10^5\text{–}2\times 10^7\text{ m}^{-1}$ across `300–1000 nm`, monochromatic Beer–Lambert using wavelength-averaged $\bar{\alpha}$ overpredicts total absorption by Jensen's inequality ($1 - e^{-\bar{\alpha} L} > \int w_\lambda (1 - e^{-\alpha(\lambda) L})\,d\lambda$), while any effective scalar $\alpha_{\text{eff}}$ tuned to match total absorbed photons $\int G\,dx$ severely underpredicts short-wavelength front-surface peak generation $G(0^+)$.
  2. *Optical photon budget anchors (`Specs/ArchitectureRefactorV1.md` L880–883; `ArchitectureRefactorExecutionV1.md` L800):*
     - On `scaps_mirror_v2`, control-volume spectral Beer–Lambert yields mesh-independent ($N=30$ and $N=60$) absorbed photon current $q \int G\,dx = 260.14\text{ A/m}^2$, aligning with SCAPS single-pass $J_{sc} \approx 262.8\text{ A/m}^2$, whereas coherent TMM reflectance/interference lowers absorbed photon current to $235.06\text{ A/m}^2$.
     - On `scaps_mirror`, true electrical-stack TMM absorbed photon budget is $219.01\text{ A/m}^2$, demonstrating that the historical $231.7\text{ A/m}^2$ baseline over-collected photons relative to the electrical-stack TMM budget.
  3. *Legacy density-Radau negative-$p$ collapse vs new-kernel positivity (`Specs/ArchitectureRefactorV1.md` L297, L879–885):*
     - On `perovskite_sim` (`658baa84`), switching `scaps_mirror_v2` at $N_{\text{grid}} = 60$ to spectral Beer–Lambert ($q \int G\,dx = 260.14\text{ A/m}^2$, $\sim 10\%$ higher front peak $G$) drove linear-density `solve_ivp(Radau)` to negative hole density ($p < 0$) at absorber-front `node 20`, collapsing step size and hanging `tests/integration/test_interface_flux_balance.py` past `600 s` inside `_dense_num_jac`.
     - On `solarlab`'s new kernel, `State1D` parameterizes carriers via log-activities $n = n_{\text{ref}}\exp(\Delta\phi/V_T + z_n) > 0$, $p = p_{\text{ref}}\exp(-\Delta\phi/V_T - z_p) > 0$, and `BackwardEuler1DStep` integrates physical inventories via `S_old * expm1(log_ratio) / dt` with `generation_inventory` positive logarithmic seeding (`log1p(dt * G / density)`), structurally excluding $p \le 0$ domain violations.
- **New Target Semantics:**
  - Introduce immutable `SpectralBeerLambertBinding` in `src/solarlab/physics/generation/` and wire explicit optical-model selection (`beer_lambert` | `beer_lambert_spectral` | `tmm`) in `config/` + `device/`:
    - Per-layer $\alpha_\ell(\lambda) = 4\pi k_\ell(\lambda)/\lambda \ge 0$ integrated exactly over each cell's true optical-depth increment $\Delta\tau_{i,\lambda}$ via $\exp(-\tau_{\text{left}, i,\lambda}) \cdot (-\mathrm{expm1}(-\Delta\tau_{i,\lambda}))$, followed by explicit wavelength quadrature over the declared `SpectralPhotonFlux` resource.
    - Optical-only front `substrate` layers attenuate $\tau_{\text{left}}(\lambda)$ without generating electrical carriers (`electrical=False`), with closed spectral photon budget ($\text{attenuated\_substrate} + \text{absorbed\_electrical} + \text{transmitted} = \Phi_{\text{incident}}$ to `rtol <= 1e-12`).
- **Allowed Scope:** `src/solarlab/physics/generation/`, `src/solarlab/device/optical_generation.py`, `src/solarlab/device/optical_generation_2d.py`, `src/solarlab/device/compile_1d.py`, `src/solarlab/device/compile_2d.py`, and typed optical-model selector fields in `src/solarlab/config/` + `src/solarlab/device/inputs.py`.
- **Falsification & Non-Regression Boundaries:**
  - Must **not** relabel historical TMM runs (`2026-07-02` `FormalVerify.pdf`) or uniform-generation literature benchmarks as spectral Beer–Lambert (`BeerLambertReferenceProtocolV1.md`).
  - Must pass the $N_{\text{grid}} = 60$ high-front-generation heterojunction transient / steady-state stress case cleanly in $< 5\text{ s}$ with strict $n > 0, p > 0$ and zero step collapse.
- **Affected Consumers:** 1D/2D steady & transient J–V, `scaps_mirror_v2` reference studies, Suns-$V_{oc}$, TPV/TPC, and EQE/spectral-response pipelines that opt into single-pass optics.
- **Independent Acceptance Criteria & Expected Direction of Change:**
  - Mesh-independent electrical photon conservation $\sum_i w_i G_i = \int \Phi_\lambda e^{-\tau_{\text{sub}}(\lambda)} (1 - e^{-\tau_{\text{elec}}(\lambda)})\,d\lambda$ to `rtol <= 1e-13` across $N=15, 30, 60, 120$ non-uniform grids.
  - Expected physical direction: removing air/glass/TCO Fresnel reflection & Fabry–Pérot interference increases `scaps_mirror_v2` absorbed photon current from $235.06\text{ A/m}^2$ (TMM) to $260.14\text{ A/m}^2$ (spectral BL), shifting $J_{sc}$ upward toward single-pass SCAPS ($\sim 262.8\text{ A/m}^2$).
- **Versioning & Commit Isolation Rule:** Isolated `P`-class commit adding `SpectralBeerLambertBinding` (`beer_lambert_spectral_v1`) and explicit `optical_model` selection; monochromatic `beer_lambert` and coherent `tmm` presets remain unchanged unless explicitly version-bumped in a separate preset commit.

---

### 5.2 `P08-04b`: External Generation Profile-File Loader Demarcation

- **Scope Boundary (`ArchitectureRefactorExecutionV1.md` §12 `P08-04` L801–802):**
  - `FixedGenerationProfile` (`src/solarlab/physics/generation/fixed.py`) binds node-aligned fixed generation resources already admitted in typed device declarations.
  - Arbitrary external user-file spatial/spectral generation profile loaders (e.g. SCAPS `.gen` file parsing, arbitrary non-matching mesh interpolation, and external photon-budget certification) were **absent** in legacy `perovskite_sim` and remain explicitly `not_available` during the architecture refactor unless commissioned as net-new post-refactor scope. They must not be smuggled in as an unreviewed migration patch.

---

## 6. Deferred, Non-Smooth & Out-of-Scope Dispositions

1. **`F14` Switching-Surface Non-Smoothness (`R_tot = 0`, Hard TE `min`, Poole–Frenkel `|E| = 0`):**
   - Piecewise kinks at dark radiative/recombination floors (`R_tot == 0`), hard thermionic flux capping (`te_softness == 0.0`, $\min(|J_{\text{SG}}|, |J_{\text{TE}}|)$), and Poole–Frenkel field-assisted mobility ($\exp(\gamma_{\text{PF}}\sqrt{|E|})$ at $|E| = 0$) have discontinuous or singular classical derivatives on their switching manifolds.
   - Current disposition (`P04-12` / `solve_nonlocal_small_signal_1d` / `FieldMobilityCapabilityError`): AC small-signal linearization fails closed with `CapabilityError` on uncertified non-smooth switching surfaces rather than silently returning one-sided subgradients; smooth $C^2$ `compact_support` / `legacy_logistic` TE regularization (`te_softness > 0`) remains explicit opt-in. Any future smooth replacement of Poole–Frenkel near $E=0$ or default `te_softness > 0` requires its own `P`-class commit and convergence/AC verification.
2. **Coupled Time-Domain Electrothermal PDE $T(x, t)$:**
   - Static isothermal temperature scaling (`T`, `mu_T_gamma`, `B_rad_T_gamma`, `varshni_alpha`/`varshni_beta`, `E_a_ion`) is bound at compile time (`time_domain_temperature_qualified = False`). Dynamic self-heating / coupled heat-equation PDEs $c_p \partial_t T - \nabla\cdot(\kappa\nabla T) = Q_{\text{Joule}} + Q_{\text{rec}}$ are out of `P08` scope.
3. **Arbitrary Ad-Hoc Plotting / Unbound Post-Processing:**
   - Core `solarlab` produces typed immutable results and reproducible `RunStore` bundles; ad-hoc plotting scripts remain outside `src/solarlab/`.
4. **`RS` (`R1`) Research Lane Pinning (`V2` `D20`):**
   - `R1` execution and long-window `G2-R` / `RS-G7` qualification remain pinned to pre-`P08` model versions so prospective `P08` physics updates cannot invalidate or confound historical `R1` long-window evidence.

---

## 7. `P08-05` — Unified Opt-In Compiler Wiring, `v1 -> v2` Version Mapping & Gate `G8` Closeout

### 7.1 Explicit `v1 -> v2` Model & Compiler Mapping (`P08_MODEL_VERSION_MAPPINGS`)

`src/solarlab/physics/presets.py` (`P08_MODEL_VERSION_MAPPINGS`, `P08CompilerOptions`, `build_p08_model_registry`, `build_p08_preset`) and `src/solarlab/device/compile_1d.py` (`p08_identity_inputs`) bind every Stage `P08` physics correction to an explicit model-version upgrade (`v1 -> v2`) while keeping all five compilers (`compile_device_1d`, `compile_ionic_device_1d`, `compile_multiple_ionic_device_1d`, `compile_device_2d`, `compile_ionic_device_2d`) **0-ULP / 0-byte manifest-digest invariant** under default `v1`:

| Item | Family | Pre-`P08` Default (`v1` `ModelRef` & Token) | Prospective Opt-In (`v2` `ModelRef` & Compiler Keyword) | Semantic Delta |
| --- | --- | --- | --- | --- |
| `P08-02a` | `electrostatics_face_permittivity` | `electrostatics.face_permittivity@v1` (`nodal_harmonic_dielectric_v1`) | `electrostatics.face_permittivity@v2` (`dielectric_rule="layer_series_capacitance_v2"`) | Replaces unweighted nodal harmonic face permittivity and shared-node right-layer bleed with exact sub-face material-extent series capacitance $C_f = \varepsilon_0 / (d_L/\varepsilon_L + d_R/\varepsilon_R)$ and 2D cell-parallel lateral permittivity. |
| `P08-02b` | `control_volume_geometry` | `mesh.endpoint_control_volume@v1` (`legacy_mol_integer_endpoint_v1` in 1D; `legacy_twod_half_endpoint_v1` in 2D) | `mesh.endpoint_control_volume@v2` (`geometry_name="physical_half_endpoint_v2"` in 1D; `stack_geometry="physical_half_endpoint_v2"` in 2D) | Binds contact half-cells ($w_0 = dx_0/2, w_{N-1} = dx_{N-1}/2$, $\sum_i w_i = L$) simultaneously across storage, mobile ions, optical generation, Poisson contact displacement, and terminal ports. |
| `P08-02c` | `band_step_differencing` | `transport.band_step_differencing@v1` (`legacy_summed_potential_v1`) | `transport.band_step_differencing@v2` (`band_step_mode="hole_separate_increment_v1"`) | Forms $(\Delta\phi + \Delta\chi_t) + \Delta E_{g,t}$ per face before summation in 1D/2D SG transport so $\sim 5.5\text{ eV}$ valence-band offsets do not quantize sub-mV electrostatic steps in binary64. |
| `P08-03` | `interface_thermionic` | `interface_transport.thermionic_emission@v1` (`legacy_unnormalized_thermionic_v1`) | `prospective_interface_p08@v2` / `prospective_interface_p08_compact_support@v2` (`thermionic_mode="reciprocal_physical_thermionic_v2"`, `combination_mode="harmonic_series" \| "compact_support"`) | Separates unfolded physical band-edge barriers ($\chi_{\text{phys}}, E_{g,\text{phys}}$), per-side DOS activities ($c_L/N_L, c_R/N_R$), reciprocal $\tilde{A}^* = \min(A^*_L, A^*_R)$, transmission $\mathcal{T}$, and cancellation-free `expm1` electrochemical drive across both `deduplicated` and `two_sided_trace` (`statistics="maxwell_boltzmann"`) 1D/2D heterointerfaces. |
| `P08-04` | `optical_generation` | `monochromatic_beer_lambert_1d@v1` (`"beer_lambert"` / coherent `"tmm"` per device declaration) | `spectral_control_volume_beer_lambert@v2` (`optical_model="beer_lambert_spectral"`) | Integrates wavelength-resolved $\alpha_\ell(\lambda) = 4\pi k_\ell(\lambda)/\lambda$ over exact physical control-volume optical depths with closed front-substrate + electrical + transmitted photon budget (`rtol <= 1e-12`). |

### 7.2 Identity Separation, Certificate Non-Reuse & Historical Disable Tracking

1. **Zero Drift on Default `v1`:** Calling any of the five compilers with default arguments (or `P08CompilerOptions()`) appends zero new manifest keys and preserves bit-for-bit identical `compiled.identity` SHA-256 digests, numerical residuals, and sparse Jacobians relative to the Wave 2 `N`-baseline (`refactor/wave2-n-baseline` @ `582f019ef5743d43546219590a5ae094ca300594`).
2. **Deterministic `compiled.identity` & `H_physics` Invalidation on `v2`:** Activating any `P08` `v2` selector deterministically mutates both:
   - `compiled.identity` (and `interface_model.identity_sha256` / `CompiledIonicDevice{1D,2D}.identity`), preventing `solve_transient`, `solve_impedance`, `solve_ionic_transient`, and `solve_ionic_impedance` from reusing a `v1` steady-state operating point or certificate on a `v2`-compiled device (`ValueError`); and
   - `physics_identity(p08_identity_inputs(prepared, implementation, options=...)).sha256` (`H_physics`), while `build_p08_preset(P08CompilerOptions(), sources)` records all five `v2` `ModelRef`s in `historical_disables` so explicit `select_preset(..., overrides=...)` surfaces exact `reenabled_historical_mechanisms` provenance.

### 7.3 Gate `G8` Closeout Disposition Matrix

| `G8` Verification Requirement | Disposition | Executable Evidence / Enforcement |
| --- | --- | --- |
| One-commit-per-physics-fix isolation (`P08-01`..`P08-05`) | **Satisfied** | Separate isolated commits on `wave3/ws1-p08-05-compiler-wiring`: `P08-01` (`dab6e1c`), `P08-02a` (`8b00294`), `P08-02b` (`6ace4d0`), `P08-02c` (`521a9fb`), `P08-03` (`b9bec98`), `P08-04` (`958d0cc`), and `P08-05` compiler/registry closeout. |
| Default `v1` 0-byte manifest & numerical parity with `582f019e` | **Satisfied** | `tests/integration/test_p08_05_compiler_wiring.py` pins exact `582f019e` SHA-256 hex digests across 1D (`deduplicated` & `two_sided_trace`), single/dual ionic 1D, 2D, and ionic 2D compilers. |
| End-to-end `v2` 1D/2D DC, BE transient conservation, 0-ULP `Assembled` parity & spatial refinement | **Satisfied** | `test_p08_02_refinement_and_parity.py`, `test_p08_03_heterojunction_detailed_balance.py`, `test_spectral_beer_lambert_stress.py`, and `test_p08_05_compiler_wiring.py`. |
| Preservation of historical `v1` verdicts & failed criteria (`AGENTS.md`) | **Enforced** | `scaps_mirror_v2` `6.024958093130e-3 eV` `legacy_manual_v_bi_contact_offset` remains `inconsistent` (`CONTACT_THERMODYNAMIC_TOLERANCE_EV = 5.0e-3` unrelaxed); historical `Final121` (`4.010735e-4 > 1/3000 A/m^2`), `CND2 V4` strict-scope (`not_passed`), and `R1 V53` step-309 (`failed`) remain unoverwritten. |
| Permanent refusal of falsified closures (`interface_shared_occupancy`, bulk-node Gaussian $E_t$) | **Enforced** | `compile_device_1d`, `compile_device_2d`, and `compile_ionic_device_2d` raise `CapabilityError` pre-solve (`test_p08_counterexamples.py`, `test_p08_03_thermionic_and_contacts.py`, `test_p08_05_compiler_wiring.py`). |
| Decoupled `RS` (`R1`) lane pre-`P08` baseline pinning (`V2` `D20`) | **Enforced** | `RS` remains pinned to `refactor/wave2-n-baseline` @ `582f019ef5743d43546219590a5ae094ca300594`; `solarlab_research.cbo_scan.enforce_pre_p08_cbo_pinning` rejects all `P08_V2_MODEL_REFS` (`InterfaceCBOScanError`) unless `allow_prospective_p08_models=True`. |
| Deferred out-of-`P08` scope (`F14` kinks, dynamic $T(x,t)$ PDE, external `.gen` file loader) | **Demarcated (not counted as `passed`)** | Explicitly documented in §5.2 & §6; non-smooth AC switching manifolds fail closed via `CapabilityError`. |
