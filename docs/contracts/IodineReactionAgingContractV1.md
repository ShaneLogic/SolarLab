# Iodine Reaction–Diffusion–Electrostatics Aging Contract (V1)

## 1. Scope and Provenance

This specification defines the continuum **iodine-chemistry-driven perovskite aging model** (`solarlab.iodine_aging.v1`) implemented on SolarLab's shared `SystemLayout` / `Assembler` / `StorageProblem` architecture.

- **Primary literature reference**: S. Bitton and N. Tessler, *"Perovskite ionics – elucidating degradation mechanisms in perovskite solar cells via device modelling and iodine chemistry"*, *Energy & Environmental Science*, **16**(6), 2621–2628 (2023), DOI `10.1039/d3ee00881a` (Supplementary Information Eqs. 1–6, Table S1, Figures S1–S2).
- **Qualification boundary**: Engineering implementation and analytical/conservation/parity verification under this contract do **not** constitute full experimental qualification (`scientific_qualification = False` remains explicit on all run manifests).

---

## 2. Dynamic Species, Stoichiometric Matrix, and Exact Conservation Laws

The coupled 1D system resolves electrostatic potential $\phi(x, t)$ together with five mobile particle populations:

| Index | Block / Species ID | Symbol | Charge Number $z_s$ | Iodine Atom Count $m_{\mathrm{I},s}$ |
|:-----:|:-------------------|:------:|:-------------------:|:------------------------------------:|
| 0 | `electron` | $n$ | $-1$ | $0$ |
| 1 | `hole` | $p$ | $+1$ | $0$ |
| 2 | `iodide` | $c_-$ ($\mathrm{I}_i^-$) | $-1$ | $1$ |
| 3 | `iodine_atom` | $c_0$ ($\mathrm{I}_i^0$) | $0$ | $1$ |
| 4 | `iodine_molecule` | $c_2$ ($\mathrm{I}_2$) | $0$ | $2$ |

Five bulk chemical reaction channels $\mathbf{R} = [R_a, R_b, R_c, R_d, R_e]^{\top}$ ($\mathrm{m}^{-3}\;\mathrm{s}^{-1}$) act through the $5 \times 5$ integer stoichiometric matrix $\boldsymbol{\nu}$:

$$
\boldsymbol{\nu} =
\begin{bmatrix}
0 & 0 & -1 & 0 & -2 \\
-1 & +1 & 0 & 0 & 0 \\
-1 & +1 & +1 & 0 & +2 \\
+1 & -1 & -1 & -2 & 0 \\
0 & 0 & 0 & +1 & -1
\end{bmatrix},
\qquad
\mathbf{S}_{\mathrm{chem}} = \boldsymbol{\nu}\,\mathbf{R} =
\begin{bmatrix}
-R_c - 2R_e \\
-R_a + R_b \\
-R_a + R_b + R_c + 2R_e \\
+R_a - R_b - R_c - 2R_d \\
+R_d - R_e
\end{bmatrix}.
$$

Every column $r \in \{a, b, c, d, e\}$ strictly satisfies the two nullspace invariants:

1. **Microscopic Charge Conservation**: $\mathbf{z}^{\top}\boldsymbol{\nu} = \mathbf{0}^{\top}$, so $q(-S_n + S_p - S_-) \equiv 0$ identically at every node and trial state.
2. **Elemental Iodine Conservation**: $\mathbf{m}_{\mathrm{I}}^{\top}\boldsymbol{\nu} = \mathbf{0}^{\top}\;\Longrightarrow\; S_- + S_0 + 2S_2 \equiv 0$ vanishes identically at every node and trial state.

Any candidate `StoichiometricNetwork` violating either nullspace condition is rejected at construction time with `ContractError`.

---

## 3. Constitutive Kinetics and Thermodynamic Detailed Balance

### 3.1 Reaction Rate Modes

Two constitutive closures are admitted via `thermodynamic_mode`:

1. **`"wegscheider_detailed_balance"` (Default)**:
   Enforces Wegscheider cyclic detailed balance across the $(a, b, c)$ monomer redox cycle and the $(a, b, d, e)$ dimer redox cycle using local intrinsic carrier density squared $n_i^2 = N_C N_V \exp(-E_g / V_T)$, equilibrium $\mathrm{I}^0/\mathrm{I}^-$ electron density $n_{1,\mathrm{I}} = k_a n_i^2 / k_b$ (when $k_b > 0$), and dimer dissociation constant $K_{\mathrm{diss}} = k_{-d}/k_d > 0$:

$$
\begin{aligned}
R_a &= k_a c_- p, \\
R_b &= k_b c_0, \\
R_c &= k_c \left(c_0 n - n_{1,\mathrm{I}} c_-\right), \\
R_d &= k_d \left(c_0^2 - K_{\mathrm{diss}} c_2\right).
\end{aligned}
$$

   For molecular iodine reduction ($\mathrm{I}_2 + 2e^- \rightleftharpoons 2\mathrm{I}^-$), two kinetic orders are supported via `reaction_e_order`:
   - **`"bimolecular_n"` (Default, Bitton & Tessler Table S1 Smoluchowski closure)**:
     Derived from the two-step radical-anion mechanism $\mathrm{I}_2 + e^- \rightleftharpoons \mathrm{I}_2^{\bullet -}$ followed by $\mathrm{I}_2^{\bullet -} + e^- \rightleftharpoons 2\mathrm{I}^-$ with radical crossover density $n_{e,\ast} > 0$:

$$
R_e = \frac{k_e}{n + n_{e,\ast}}\left(c_2 n^2 - \frac{n_{1,\mathrm{I}}^2}{K_{\mathrm{diss}}}\,c_-^2\right),
\qquad
\frac{\partial R_e}{\partial n} = k_e\,\frac{c_2 n (n + 2 n_{e,\ast}) + (n_{1,\mathrm{I}}^2 / K_{\mathrm{diss}})\,c_-^2}{(n + n_{e,\ast})^2} \ge 0.
$$

     For $n \gg n_{e,\ast}$, $R_e \to k_e c_2 n$ with $k_e = 4\pi D_e R_{AB}$ ($\mathrm{m}^3\;\mathrm{s}^{-1}$); at dark thermal equilibrium ($n p = n_i^2$, $c_0 = (k_a/k_b) p c_-$, $c_2 = c_0^2 / K_{\mathrm{diss}}$), $R_a - R_b = R_c = R_d = R_e = 0$ identically without any $1/n$ singularity as $n \to 0$.
   - **`"termolecular_n2"`**:
     $R_e = k_{e,3}\bigl(c_2 n^2 - (n_{1,\mathrm{I}}^2 / K_{\mathrm{diss}})\;c_-^2\bigr)$ with $k_{e,3}$ in $\mathrm{m}^6\;\mathrm{s}^{-1}$.

2. **`"irreversible_mass_action"` (Direct Sentaurus comparison toggle)**:
   Evaluates $R_a = k_a c_- p$, $R_b = k_b c_0$, $R_c = k_c c_0 n$, $R_d = k_d c_0^2$, and $R_e = k_e c_2 n$ (or $k_{e,3} c_2 n^2$).

### 3.2 Smoluchowski Diffusion-Limited Rate Factory

For diffusion-controlled reactions with capture radius $R_{AB} > 0$, rate constants follow $k_{AB} = 4\pi (D_A + D_B) R_{AB}$ in SI units ($\mathrm{m}^3\;\mathrm{s}^{-1}$), where $D_{e,h} = \mu_{e,h} V_T$, $D_{AA} = 2 D_0$ for homodimerization $2\mathrm{I}^0 \to \mathrm{I}_2$, and explicit layer-by-layer rate overrides are preserved verbatim when reproducing Table S1.

---

## 4. Coordinate Regularization and Multi-Species Transport

### 4.1 Dimensionless Fischer–Burmeister Positive Map

In `full_stack_permeable` stacks (e.g., `HBL / Perovskite / EBL`), blocking layers start at zero initial iodine density (`Initial I- = 0`). To avoid the singular zero storage tangent of pure logarithmic coordinates while strictly guaranteeing $c_j > 0$ without `exp` overflow, dilute species use the smooth dimensionless Fischer–Burmeister map with scale $c_{\mathrm{scale},j} > 0$ and regularization ratio $\eta_j \in (0, 1]$:

$$
r(u) = \mathrm{hypot}(u, 2\eta_j),
\qquad
\psi(u) =
\begin{cases}
\dfrac{u + r(u)}{2}, & u \ge 0, \\[8pt]
\dfrac{2\eta_j^2}{r(u) - u}, & u < 0,
\end{cases}
\qquad
c_j(u) = c_{\mathrm{scale},j}\,\psi(u) > 0.
$$

Its tangent and cancellation-free BDF storage secant are strictly positive and subtraction-free:

$$
\frac{\mathrm{d}c_j}{\mathrm{d}u} = c_{\mathrm{scale},j}\,\frac{\psi(u)}{r(u)} > 0,
\qquad
c_j(u_2) - c_j(u_1) = c_{\mathrm{scale},j}\,(u_2 - u_1)\,\frac{\psi(u_1) + \psi(u_2)}{r(u_1) + r(u_2)}.
$$

When finite-site steric exclusion is declared for $\mathrm{I}^-$ (`iodide_site_limit_m3`), $c_-$ switches to the bounded logistic map $c_- \in (0, c_{\mathrm{lim}})$ and uses `solarlab.physics.ions` steric Scharfetter–Gummel fluxes.

### 4.2 Exponential-Fitted Face Fluxes and Heterointerface Partitioning

Across interior face $f$ with spacing $h_f > 0$, harmonic face diffusivity $D_{j,f}$, valence $z_j \in \{-1, 0\}$, and relative partition ratio $\kappa_{j,f} = K_{\mathrm{part},j,R} / K_{\mathrm{part},j,L} > 0$ (default $\kappa_{j,f} = 1$), the particle flux is:

$$
\xi_{j,f} = \frac{z_j (\phi_R - \phi_L)}{V_T} - \ln \kappa_{j,f},
\qquad
\Gamma_{j,f} = \frac{D_{j,f}}{h_f}\Bigl[B(\xi_{j,f})\,c_{j,L} - B(-\xi_{j,f})\,c_{j,R}\Bigr],
$$

where $B(\xi) = \xi / (e^\xi - 1)$ is the stabilized Bernoulli function. For $z_0 = z_2 = 0$ and $\kappa_{j,f} = 1$, this reduces identically to Fickian diffusion $\Gamma_{j,f} = D_{j,f}(c_{j,L} - c_{j,R})/h_f$.

---

## 5. Domain Topologies, Boundary Closures, and Accounting Gates

1. **Domain Topologies (`domain_mode`)**:
   - `"absorber_blocking"`: Chemical `NodeSet("chem")` is embedded on the absorber layer; neutral and charged iodine species have zero flux into transport layers.
   - `"full_stack_permeable"`: Chemical `NodeSet("chem")` spans the full electrical stack (`HBL / Perovskite / EBL`) with piecewise layer diffusivities, reaction rates, and optional partition factors.
2. **Outer Boundary Closures**:
   - **Volatile $\mathrm{I}_2$ surface escape**: Outward boundary flux $J_{\mathrm{loss},2} = k_{\mathrm{loss}} c_2 \ge 0$ ($\mathrm{m}^{-2}\;\mathrm{s}^{-1}$) on `"left"` and/or `"right"` ports. Closed steady-state inventory constraints are admitted only when $k_{\mathrm{loss,left}} = k_{\mathrm{loss,right}} = 0$; open-boundary escape requires transient integration.
   - **Charge-conserving contact reaction (`contact_reaction_mode`)**:
     - `"none"` (blocking to $\mathrm{I}^-$ and $\mathrm{I}^0$);
     - `"hole_oxidation_to_I0"`: surface reaction $\mathrm{I}^- + h^+ \to \mathrm{I}^0$ with rate $J_{\mathrm{ct}} = k_{\mathrm{ct}} c_- p$ removing equal $\mathrm{I}^-$ and $h^+$ fluxes while injecting $+J_{\mathrm{ct}}$ into $\mathrm{I}^0$ (zero net charge creation, zero iodine atom loss);
     - `"immobilized_product_sheet"`: surface reaction $\mathrm{I}^- + h^+ \to \mathrm{I}_{\mathrm{surf}}^0(\text{fixed})$ removing equal $\mathrm{I}^-$ and $h^+$ fluxes (zero net charge error) and accumulating trapped surface iodine inventory in the step ledger.
   Non-conserving bare $\mathrm{I}^-$ sink boundaries are prohibited.
3. **Mandatory Step Accounting Invariants**:
   Every transient step tracks in-device elemental iodine $M_{\mathrm{I}}(t) = \sum_i w_i (c_{-,i} + c_{0,i} + 2c_{2,i})$ plus cumulative escaped/immobilized boundary iodine $M_{\mathrm{I,bound}}(t)$, enforcing relative conservation drift $\lvert M_{\mathrm{I}}(t) + M_{\mathrm{I,bound}}(t) - M_{\mathrm{I}}(0) \rvert / M_{\mathrm{I}}(0) \le 10^{-12}$.
