# Multivalent and metastable defect contract

## Status and capability boundary

This document describes the first D7 checkpoint. It adds:

- a canonical version-4 bulk-defect document for one physical defect with
  coupled charge states;
- a solver-independent stationary master-equation closure with analytic local
  density tangents;
- a canonical metastable donor/acceptor configuration definition;
- a replayable initial-working-point and frozen-measurement protocol.

This checkpoint does **not** connect version 4 to `MaterialParams`, Poisson,
continuity, contacts, QF/DC, AC, transient solvers, backend, or frontend. It
does not claim SCAPS numerical parity. Existing v1-v3 explicit-defect and
default effective-lifetime paths are unchanged.

The current capability label is:

```text
multivalent/metastable canonical contract and pure-local stationary closure;
production execution fail-closed and not yet wired
```

## Primary model sources

The contract is based on the public SCAPS model descriptions rather than an
independent-SRH approximation:

- SCAPS Manual 2016, sections 3.6.2, 3.7, and 4.3:
  [`../../docs/manual/SCAPSManual2016.pdf`](../../docs/manual/SCAPSManual2016.pdf)
- K. Decock, S. Khelifi, and M. Burgelman, "Modelling multivalent defects in
  thin film solar cells," Thin Solid Films 519 (2011) 7481-7484,
  [doi:10.1016/j.tsf.2010.12.039](https://doi.org/10.1016/j.tsf.2010.12.039)
- K. Decock, P. Zabierowski, and M. Burgelman, "Modeling metastabilities in
  chalcopyrite-based thin film solar cells," J. Appl. Phys. 111 (2012)
  043703,
  [doi:10.1063/1.3686651](https://doi.org/10.1063/1.3686651)

SCAPS permits up to four transitions (five states) per defect, with charge in
the range -3 to +3. States are ordered from the most positive to the most
negative. The special families are represented literally:

| Family | `charge_states_e` |
|---|---|
| `single_donor` | `[+1, 0]` |
| `single_acceptor` | `[0, -1]` |
| `double_donor` | `[+2, +1, 0]` |
| `double_acceptor` | `[0, -1, -2]` |
| `amphoteric` | `[+1, 0, -1]` |
| `custom_multilevel` | any 2-5 consecutive states inside `[-3, +3]` |

Two separately normalized monovalent species are not equivalent: they each
own a full defect density. Version 4 instead owns one total density and
requires

$$
\sum_{s=0}^{H} P_s = 1, \qquad N_s=N_tP_s,
\qquad q_{s+1}=q_s-1.
$$

## Canonical version-4 schema

`perovskite_sim.models.multivalent_defects` defines
`solarlab-explicit-bulk-defects-v4`. A document retains the familiar top-level
keys:

```json
{
  "schema_version": "solarlab-explicit-bulk-defects-v4",
  "defect_model": "explicit_quasi_steady",
  "bulk_defects": []
}
```

Each `MultivalentBulkDefectSpecies` contains:

- a unique name;
- one shared `total_density_m3`;
- a `MultivalentDefectConfiguration` containing the state family, ordered
  integer charges, state degeneracies, transition energies, and one
  `BulkDefectKinetics` per adjacent transition.

Transition energies use an above-valence-band reference and are encoded as
the first transition plus signed correlation energies:

$$
E_{t,0}=E_{\mathrm{first}}, \qquad
E_{t,s}=E_{t,s-1}+U_s.
$$

This representation supports positive and negative correlation energy while
requiring every resolved level to remain inside the material band gap.
SCAPS's two degeneracy choices are explicit:

$$
g_s={H \choose s} \quad\text{or}\quad g_s=1.
$$

`explicit` also accepts declared positive state degeneracies. No degeneracy is
silently folded into a temperature-dependent transition energy.

The focused version-4 fixture has document hash:

```text
3f29bf380c710b03834b9ef50ad4b95d403352e743345046f55922e6283f7e81
```

## Stationary master equation

For transition `s -> s+1`, define capture constants
`c_n = sigma_n*v_th,n` and `c_p = sigma_p*v_th,p`. With the local valence band
as the zero of energy,

$$
n_{1,s}=N_C\exp\left[-\frac{E_g-E_{t,s}}{V_T}\right],\qquad
p_{1,s}=N_V\exp\left[-\frac{E_{t,s}}{V_T}\right].
$$

Detailed balance including adjacent state degeneracy gives

$$
e_{n,s}=c_{n,s}n_{1,s}\frac{g_s}{g_{s+1}},\qquad
e_{p,s}=c_{p,s}p_{1,s}\frac{g_{s+1}}{g_s}.
$$

The total rates toward the more negative and more positive states are

$$
a_s=c_{n,s}n+e_{p,s},\qquad
b_s=e_{n,s}+c_{p,s}p.
$$

The implementation constructs the full column-conservative generator `K` and
checks `K P = 0`. The stationary probabilities use the stable recurrence

$$
w_0=1,\qquad
w_s=\prod_{x=0}^{s-1}\frac{a_x}{b_x},\qquad
P_s=\frac{w_s}{\sum_x w_x},
$$

evaluated in log space. This avoids subtracting two nearly equal dominant
state densities.

The transition recombination rate uses the stable shared-population form

$$
U_s=N_t(P_s+P_{s+1})
\frac{c_{n,s}c_{p,s}(np-n_i^2)}
{c_{n,s}n+c_{p,s}p+e_{n,s}+e_{p,s}}.
$$

Defect charge is computed from the same probabilities:

$$
\rho_{\mathrm{def}}=qN_t\sum_s q_sP_s.
$$

No clipping, state renormalization after the solve, or independent-SRH density
duplication is used.

## Analytic tangent

Let `l_s = log(w_s)`. The implementation differentiates the recurrence,

$$
\frac{\partial l_s}{\partial n}
=\sum_{x<s}\frac{c_{n,x}}{a_x},\qquad
\frac{\partial l_s}{\partial p}
=-\sum_{x<s}\frac{c_{p,x}}{b_x},
$$

and then the normalized probabilities,

$$
\frac{\partial P_s}{\partial z}
=P_s\left(\frac{\partial l_s}{\partial z}
-\sum_xP_x\frac{\partial l_x}{\partial z}\right).
$$

This is the recurrence form of the implicit-function tangent. Tests also
evaluate the matrix identity

$$
K\frac{\partial P}{\partial z}
+\frac{\partial K}{\partial z}P=0,
\qquad \sum_s\frac{\partial P_s}{\partial z}=0.
$$

Charge and recombination tangents, including the fixed-quasi-Fermi potential
direction, are derived from these same probabilities.

## Metastable definition and preparation

`solarlab-metastable-bulk-defects-v1` represents a metastable defect as two
conventional configurations. Each configuration may itself be multivalent.
The selected donor and acceptor conversion states must differ by exactly two
elementary charges. A single `total_density_m3` belongs to the metastable pair;
configuration densities are not independent inputs.

`MetastableConversionKinetics` records:

- the thermodynamic transition energy;
- EC, EE, HC, and HE activation barriers;
- the electron-side pathway (`double_electron_capture` or
  `electron_capture_plus_hole_emission`);
- the hole-side pathway (`double_hole_capture` or
  `hole_capture_plus_electron_emission`);
- electron/hole capture constants and phonon frequency.

With `E_V=0` and `E_C=E_g`, the resolved barriers must obey detailed balance.
For example,

$$
\Delta E_{EE}=\begin{cases}
\Delta E_{EC}+2(E_g-E_{TR}), & \text{double EC},\\
\Delta E_{EC}+E_g-2E_{TR}, & \text{EC+HE},
\end{cases}
$$

and

$$
\Delta E_{HE}=\begin{cases}
\Delta E_{HC}+2E_{TR}, & \text{double HC},\\
\Delta E_{HC}+2E_{TR}-E_g, & \text{HC+EE}.
\end{cases}
$$

The frozen fixture uses the published CuInSe2 `(VSe-VCu)` values and has hash:

```text
101ab83c111fbeb5c1ac89e4479360721255fd8d0aa4694ba3e8269d5c43900b
```

`solarlab-metastable-preparation-v1` separates preparation from measurement.
It records temperature, voltage, illumination, continuation steps, stationary
infinite-time preparation, nonlinear controls, measurement temperature, the
bound measurement-protocol SHA-256, and the mandatory freeze stage. A clamped
iterate can only seed a final unclamped refinement. The fixture hash is:

```text
7bba03a617e8420bdfd974b004e0422ab36debad66a0e6ff8085f7274de14ea6
```

Version 1 rejects finite-time metastable conversion and rejects any protocol
that allows configuration fractions to update during the measurement. Those
paths require a later fully dynamic state and independent conservation tests.

## Focused evidence

The first checkpoint tests cover:

- all predefined charge families and custom five-state inputs;
- exact-key parsing, round-trip, and frozen SHA-256 identities;
- signed correlation energy and band-gap bounds;
- state probability nonnegativity and normalization;
- detailed balance and equilibrium zero recombination;
- acceptor and donor single-transition limits against the D2 closure;
- analytic centered-difference and IFT tangent identities;
- electron-hole mirror symmetry;
- metastable conversion-state compatibility and barrier detailed balance;
- preparation hash sensitivity and frozen-measurement fail-closed behavior.

Verified on the checkpoint worktree:

```text
focused v4 schema + local closure: 29 passed
all unit models + physics:          994 passed, 1 deselected
complete Python suite:              3418 passed, 2 skipped,
                                    267 deselected, 12 warnings
Ruff format/check, compileall, and git diff --check: passed
```

The 12 complete-suite warnings are the pre-existing NumPy `trapz`
deprecations. No production module imports version 4 at this checkpoint, so
the 29 added tests account for the increase from the D6-E4 full-suite baseline
of 3389 passing tests.

## Required next checkpoints

1. Compile version 4 into a node-local production model and aggregate multiple
   multivalent species without losing one-density-per-physical-defect ownership.
2. Couple the same charge/recombination/tangent closure to contact neutrality,
   Poisson, continuity, and QF/DC, then certify grid/energy/tolerance refinement.
3. Add the initial-working-point stationary metastable outer solve, immutable
   frozen configuration densities, and protocol-bound measurement replay.
4. Add AC and fully dynamic metastable state only after DC conservation and
   frozen-history tests pass.
5. Import a real SCAPS multivalent/metastable deck and directly exported
   charge-state/recombination profiles before claiming external parity.
