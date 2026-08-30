# Multivalent and metastable defect contract

## Status and capability boundary

This document describes the D7-E0 contract checkpoint and the D7-E1
production wiring checkpoint.

D7-E0 added:

- a canonical version-4 bulk-defect document for one physical defect with
  coupled charge states;
- a solver-independent stationary master-equation closure with analytic local
  density tangents;
- a canonical metastable donor/acceptor configuration definition;
- a replayable initial-working-point and frozen-measurement protocol.

D7-E1 (2026-08-31, commit `f158cdd`) wires version 4 into the guarded
QF/DC production lane:

- generic v4 layer parsing in `models/config_loader.material_params_from_dict`
  (shared by YAML and backend inline devices) and v4 document dispatch inside
  `MaterialParams`;
- `physics/multivalent_defect_device.py`: disjoint uniform-layer region
  compilation (`MultivalentBulkDefectModel`), multi-species aggregation that
  keeps one shared total density and one normalized charge-state probability
  per physical defect, full-grid evaluation, and MB contact charge neutrality
  solved on the same master-equation closure;
- `solver/mol.py`: `_compile_multivalent_bulk_defects` under the
  `EXPLICIT_DEFECT_CHARGE_QF_DC` closure only, with a uniform-layer gate and
  the `ni^2 = Nc*Nv*exp(-Eg/Vt)` consistency gate; ordinary MoL assembly
  keeps failing closed without `phi_frozen`;
- one closure for every consumer: contact work functions
  (`physics/contacts.build_semiconductor_contact_state`), bulk Poisson charge
  and the fixed-QF tangent (`_bulk_space_charge_and_tangent`, transport
  seed), the continuity recombination source
  (`physics/continuity.py` -> `physics/recombination.py` mixed dispatch), and
  the certified diagnostics on `QuasiFermiSteadyStateResult`
  (`multivalent_bulk_defect_diagnostics`).

The current capability label is:

```text
multivalent stationary uniform-layer bulk QF/DC (dark/illuminated point and
J-V sweep) internally certified; metastable, dynamic occupancy, AC/transient,
interface-plane, mobile-ion combination, graded v4 layers, backend/frontend,
numerical-refinement certificate, and SCAPS parity all fail closed / open
```

Fail-closed routes verified by tests: ordinary `build_material_arrays`
without the QF/DC closure, `assemble_rhs` without `phi_frozen`,
`interface_boundary=True`, mobile ions in a v4 layer, the dynamic/ion
`_QuasiFermiSystem` entry points, and the D5/D6 device AC / transient lanes
(via `_require_supported(allow_multivalent_bulk_defects=False)`).

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
deprecations. No production module imported version 4 at the D7-E0
checkpoint, so the 29 added tests account for the increase from the D6-E4
full-suite baseline of 3389 passing tests.

## D7-E1 verification evidence (2026-08-30)

New tests: `tests/unit/physics/test_multivalent_defect_device.py` (8),
`tests/unit/models/test_multivalent_defect_loader.py` (7), and
`tests/integration/test_multivalent_explicit_defects_qf.py` (29). The
integration file pins, on the certified QF/DC lane: fail-closed default
build and contact modes, one shared compiled model with recombination
dispatch equality, the fixed-QF Poisson tangent against a centered
difference (pure v4, multi-species, and mixed v1+v4 stacks), certified dark
equilibria for double-donor / double-acceptor / amphoteric families with
charge-sign checks, certified illuminated biased points and J-V sweeps
carrying the new aggregate diagnostics, doped and dopant-profiled v4 layers
(the per-doping-pair neutrality seed loop), a neutral-v1-beside-v4 stack
(the only partition that reaches the neutral branch of the mixed
multivalent recombination dispatch), SCAPS-binomial degeneracies propagated
through contact neutrality and the QF solve, graded-v4 fail-closed,
document-mismatch and intrinsic-product rejections, the converse guard that
a v1-v3 qf_dc build leaves the new cache `None`, the neutral/monovalent
exclusivity invariant under a multivalent model, and the interface / ion /
dynamic / AC / transient / QF-impedance fail-closed routes. The unit device
file additionally pins multi-region row-offset bookkeeping against
independent single-region evaluations.

The physically decisive gate is the device-level single-transition limit:
a v4 `single_donor` / `single_acceptor` species solved through the full
QF/DC route matches the certified v1 monovalent route to 2.3e-14 relative
in terminal current and carrier state. The assertion is pinned two orders
looser (1e-10) than the measured agreement because the two sides are
independent Newton solves each certified only to a ~1e-10 residual; a lane
that dropped the shared master equation moves these by O(1).

Verified on the D7-E1 worktree:

```text
focused v4 schema + closure + device + loader + QF integration: 75 passed
unit physics + unit solver suites:       1373 passed, 1 skipped
complete default Python suite:           3462 passed, 2 skipped,
                                          267 deselected, 12 warnings (555 s)
Ruff (owned files) identical to the HEAD baseline, compileall,
git diff --check: passed
```

3462 = the 3418 D7-E0 baseline + the 44 new D7-E1 tests, with zero
failures. The complete-suite run uses an off-OneDrive copy of the working
tree; that copy must include `.git` and keep the directory name
`perovskite-sim`, or `test_p0_patch_and_frozen_files_match_manifest` and
`test_run_l0_runs_from_package_root` fail on the copy environment rather
than on any code defect.

Two adversarial review rounds over the D7-E1 diff (4 reviewer dimensions,
18 independent refutation passes) confirmed nine findings, all closed here.
Five were composition holes that would have given one physical defect two
different carrier states, or run different physics silently:

- `flat_band_metal_contacts` overwrote the contact reservoirs that ARE the
  root of the defect charge-neutrality closure, without re-solving it; the
  measured normalized neutrality residual went from 6e-15 to 1.0037 while the
  solve still reported `certified=True`. Now refused on this lane (which also
  closes the same latent hole on the monovalent lane).
- `het_recomb_despike` fed the closure blended interface densities for
  recombination while Poisson used the true ones; now refused.
- `carrier_statistics_transport='research_recombination_off'` zeroed the
  multivalent recombination while keeping its charge and tangent; the pair is
  now refused.
- The structured-DAE and analytic-reaction lanes forwarded only the neutral
  inventory, substituting effective-lifetime SRH for a compiled v4 model
  instead of failing closed; `require_neutral_only_defect_inventory` now
  guards all four Jacobian builders, the DAE capability validator, and the
  analytic-reaction node rate.
- The neutral/monovalent exclusivity invariant was bypassed by the new
  multivalent dispatch; restored in `_multivalent_mixed_inputs` and the
  scalar node path.

The remainder were coverage and reporting defects: the multivalent
derivative dispatch was unreachable and had never executed (now pinned by a
finite-difference test), `chi_back`-graded layers slipped the uniform-layer
gate (electron affinity added to it), off-region state probabilities were
published as zeros (now NaN, with the certificate re-deriving normalization
from the published array on owned nodes), and the binomial-degeneracy,
multi-species, multi-region, doped-layer, neutral-plus-v4,
graded-fail-closed, converse-guard and QF-impedance cases were untested.

## Required next checkpoints

1. ~~Compile version 4 into a node-local production model and aggregate
   multiple multivalent species without losing one-density-per-physical-defect
   ownership.~~ Done at D7-E1 for uniform layers; graded v4 layers still fail
   closed pending a separately certified node-local compiler.
2. ~~Couple the same charge/recombination/tangent closure to contact
   neutrality, Poisson, continuity, and QF/DC~~ (done at D7-E1), then certify
   grid/energy/tolerance refinement and a real SCAPS charge-state profile
   export (D7-E2).
3. Add the initial-working-point stationary metastable outer solve, immutable
   frozen configuration densities, and protocol-bound measurement replay.
4. Add AC and fully dynamic metastable state only after DC conservation and
   frozen-history tests pass.
5. Import a real SCAPS multivalent/metastable deck and directly exported
   charge-state/recombination profiles before claiming external parity.
