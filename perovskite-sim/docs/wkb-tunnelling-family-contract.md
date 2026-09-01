# WKB tunnelling family contract (D8)

## Status and capability boundary

This document covers **D8-P0** (audit of the existing tunnelling surface),
**D8-E0** (canonical four-channel contract plus the local WKB physics),
**D8-E1** (production wiring into the guarded QF/DC lane),
**D8-E2** (the registered refinement certificate) and **D8-E2R** (the drive
convention audit, which retracts two of D8-E1/E2's headline claims).

> **RETRACTED, 2026-09-01 — read this before quoting any magnitude below.**
> Two claims in the D8-E1/E2 sections are artifacts of a level-vs-potential
> convention error and are **withdrawn**: "the channel carries ~19-20 % of the
> terminal current", and "equilibrium net flux is exactly zero by
> reciprocity". See [Retraction (D8-E2R)](#retraction-d8-e2r). The convergence
> results, the mesh order, the injection identity and the D8-E0 unit-level
> reciprocity claim are unaffected.

The current capability label is:

```text
four independently switchable WKB tunnelling channels, each anchored to its
own barrier, wired into the guarded QF/DC lane so an enabled channel adds a
face current to the certified residual, with the intraband electron channel
carrying a registered 3x3x3 grid/tolerance/energy-order refinement
certificate; every other solver route fails closed; no SCAPS comparison has
been made and no channel magnitude is validated against any reference
```

Nothing in this checkpoint changes a shipped number: the canonical document
defaults every channel to disabled, `DeviceStack.tunnelling_channels` defaults
to `None`, and no shipped config sets the key — so every existing result is
bit-identical, pinned by
`test_an_all_disabled_document_is_bit_identical_to_no_document`.

## What already existed, and why it is not enough (D8-P0)

`physics/tunneling.py` provides `tfe_gamma`: a static Padovani-Stratton
thermionic-field-emission factor

```text
E_00 = (q h_bar / 2) sqrt(N_iface / (m* eps_s));  E_0 = E_00 coth(E_00 / V_T)
delta_tun = |dE| (1 - V_T / E_0);                 Gamma = exp(delta_tun / V_T)
```

folded into the per-face Richardson constants at TE-capped interface faces
(`A_star_n_node[f] *= tfe_gamma(...)`, `solver/mol.py`). The audit established:

- it is **one dimensionless number per (face, carrier)**, built once from
  geometry and doping — no transmission spectrum, no direction, no energy
  integration, no field or bias dependence, no trap level, no occupancy
  coupling and no contact reach;
- it only raises a **magnitude ceiling** that a direction-preserving
  magnitude-min applies against the Scharfetter-Gummel flux, so it can never
  exceed the drift-diffusion flux;
- it is default OFF, no shipped config enables it, and the repository's own
  SCAPS test asserts it must stay off *because* "SCAPS uses WKB tunnelling and
  the two forms are not comparable".

The D8 exit condition — "do not use one scalar enhancement to claim parity for
four tunnelling channels" — therefore rules out extending it. D8 is a new
contract; nothing in it rescales `A*`.

Two statements in `perovskite-sim/CLAUDE.md` about the existing model were
found **stale** by the audit and are recorded here rather than silently left
in place: `A_star_*` is no longer consumed only by `physics/continuity.py`
(there are four more consumers, three of which take `min(A*[left], A*[right])`
and so discard the enhancement), and the TE cap no longer reads the same
folded `chi`/`Eg` that the enhancement is sized from. The design note
`tasks/wtwrzoz92.output` referenced there does not exist in the tree.

## The four channels (D8-E0)

`models/tunneling_channels.py` defines
`solarlab-wkb-tunnelling-channels-v1`. Each channel is a separate frozen
dataclass with its own enable flag, effective mass, energy quadrature order
and declared units, so a frozen comparison can toggle them one at a time and
no channel can be switched on implicitly by another:

| channel | physics | needs |
|---|---|---|
| `band_to_band` | Zener tunnelling across the gap under a field | reduced effective mass, minimum field |
| `intraband` | tunnelling through a band-edge spike, one band | per-carrier effective mass, carrier selector |
| `interface_defect_assisted` | two-step tunnelling via an interface trap | an **explicit** interface occupancy |
| `contact` | field emission through a Schottky barrier | barrier height, contact side |

A disabled channel **raises** rather than returning zero, so "switched off" is
never silently indistinguishable from "physically vanishing".

`interface_defect_assisted` carries `requires_explicit_occupancy: True` which
cannot be waived. The audit found that the default transient path never
materialises an interface occupancy — it evaluates the lumped SRH quotient
with the occupancy algebraically eliminated — so this channel is only
admissible on the `two_sided_trace` / QF lane where the occupancy is a real
variable. That is the roadmap's own precondition, enforced in the schema.

## Shared WKB core

`physics/wkb_tunneling.py` computes transmission from an actual barrier
profile:

```text
S(E) = ∫_{U(x) > E} kappa dx,   kappa = sqrt(2 m* (U - E)) / h_bar,   T = exp(-2 S)
```

**Reciprocity is structural.** `reciprocal_net_flux` is the only sanctioned
way to turn a transmission into a current:

```text
J = C ∫ T(E) [ f_left(E) - f_right(E) ] dE
```

One transmission drives both directions, so equal occupations give an
identically zero integrand at every energy. "Zero net tunnelling current at
zero bias" is therefore a property of the construction, not a numerical
cancellation — and every channel measures **exactly 0.0**, not a small
residual.

### Validity

`wkb_validity` reports the action and the local-wavelength gradient.
`valid` gates on the **action alone**. The local-wavelength condition
`|d(1/kappa)/dx| << 1` fails as any smooth barrier approaches its turning
point, however large the action is, because `1/kappa` diverges there; that is
the textbook Airy region already absorbed into the `exp(-2S)` normalisation.
Gating on it would reject every physical barrier while carrying no
information, so it is reported and not gated. This is stated explicitly
because the alternative — quietly tuning the threshold until real barriers
pass — would have hidden the same fact.

## Measured evidence (D8-E0)

| property | result |
|---|---|
| action vs closed-form triangular barrier | converges at **order 1.50**, the expected O(h^3/2) for a square-root turning point |
| effective-mass scaling | doubling `m*` scales the action by exactly sqrt(2) to 1e-12 |
| energy dependence | matches `(U0 - E)^{3/2} / U0` to 5e-5 across four energies |
| monotone decay | strict in barrier width, height and effective mass |
| thin/shallow guard | action 4.8e-4 reported as not meaningful |
| zero bias, all four channels | net flux **exactly 0.0** |
| trap-assisted stationary point | occupancy residual **7.1e-17** at its stationary occupancy, 0.499 off it |

The trap-assisted residual is reported as an **equivalent occupancy offset**
rather than a bare rate: `d(rate)/df` is of order 1e30 there, so representing
the stationary occupancy in double precision alone perturbs the net rate by
~1e14. Comparing that rate against zero would measure floating-point spacing,
not physics.

Tests: 29 (`tests/unit/physics/test_wkb_tunneling.py`,
`tests/unit/physics/test_tunneling_channels.py`).

## Device wiring (D8-E1)

`DeviceStack.tunnelling_channels` carries the document; `config_loader` parses
it; `build_material_arrays` compiles only the **static** part (which faces,
which contacts, the masses, the quadrature orders). The transmission itself
cannot be cached — a tunnelling barrier is `E_C(x) = -(phi(x) + chi(x))` and
`phi` is solved for — so `physics/tunneling_channel_device.py` evaluates every
enabled channel per residual call from the live potential.

### Two corrections the wiring forced

Both were found by running the channels on a real device grid, and both are
cases where the D8-E0 local formulation was not wrong so much as **not yet a
device statement**.

**1. A channel integrates its own barrier, not the forbidden set.** A device
grid holds several barriers at once — each heterojunction spike, the band
bending at each contact — so the classically forbidden set at a given energy
is disconnected. Integrating all of it merges unrelated barriers into one
fictitious path: wrong, and silently plausible. `forbidden_run` /
`windowed_wkb_action` extract the connected run containing the channel's own
face, and `local_barrier_window` takes the energy window from the local
feature rather than the device endpoints. Measured on the two separated barriers
of the test's own `_two_barriers()` fixture at `E = 0.20`: `T = 1.2509e-1` at
the first face and `1.0442e-2` at the second, where a whole-grid integral
would have reported one merged barrier at both. The test pins the *ordering*
(`0 < T_high < T_low < 1`) rather than these values.

**2. The driving force is the local quasi-Fermi drop, not the applied bias.**
A channel is one conduction path across one barrier, so it is driven by the
quasi-Fermi drop across *that* barrier — the same drop the Scharfetter-Gummel
flux on the same face sees, which is what makes the two additive rather than
double-counted. Reading the contact levels instead inflated the flux by orders
of magnitude and made it *grow* with bias. The corrected behaviour is the
opposite: raising the bias flattens the junction, the local drop shrinks, and
the tunnelling flux falls.

**Superseded during D8-E2 — "local" means across the BARRIER, not across one
grid cell.** D8-E1 first read the drop between the two nodes of the anchor
face. That is a difference taken across a single cell, so it shrinks with the
mesh: the convergence sweep measured the channel flux **halving on every grid
doubling** (`-1.09e9 -> -3.52e8 -> -1.44e8 m^-2 s^-1` at 35 / 69 / 137 nodes),
i.e. the current was an artifact of the discretisation rather than a property
of the barrier. The drop across the barrier converges over the same ladder
(`2.44e-4 -> 2.22e-4 -> 2.18e-4 eV`).

The occupations are therefore read at each energy's own **turning points** —
where `U(x) = E`, which is where the carrier actually leaves and re-enters the
allowed region. Those sit at fixed physical positions, so the flux converges.
Reading them at the nearest *node* still costs `O(h)` in position and dragged
the whole channel to first-order convergence; interpolating the crossing
recovers order **1.5**, which is the O(h^3/2) the square-root turning point
imposes and exactly the order D8-E0 already measured for the action itself.

### The band-to-band channel uses the two-band exponent

The single-band exponent has no meaning for Zener tunnelling: the particle
leaves the valence band where `E = E_V` and enters the conduction band where
`E = E_C`, so its two turning points sit on **different bands**. Using `E_C`
alone puts both turning points on the same edge and integrates the wrong
region. `two_band_decay_constant_per_m` implements the Kane form

```text
kappa(x) = sqrt(2 m_r (E_C - E)(E - E_V) / E_g) / h_bar
```

which vanishes at both turning points. Under a uniform field it integrates in
closed form to `S = pi sqrt(2 m_r) Eg^{3/2} / (8 h_bar q F)`, i.e. the textbook
`T = exp(-pi sqrt(m_r) Eg^{3/2} / (2 sqrt(2) h_bar q F))` — verified
symbolically, and the quadrature reproduces it to **1.4e-6**. So the D8-E0
limitation "the Kane prefactor is not applied and is not claimed" is now
retired: it is applied, and pinned against its closed form.

### Fail-closed surface

| route | behaviour |
|---|---|
| ordinary MoL `assemble_rhs` | raises — the guard is **unconditional**, not gated on `phi_frozen`; a frozen potential excuses a missing Poisson charge, never a missing current |
| QF/DC source assembly | the source mat is transport-free by construction and has the channels **stripped**, so the unconditional guard above holds without an exemption |
| interface-bound channel, no heterointerface | raises |
| defect-assisted channel, no explicit occupancy | raises — the schema flag cannot be waived |
| zero contact barrier height | raises at document construction |

### Where the injection goes, and why the order matters

The interface plane zeroes its own face current so its reservoir transfer is
not double-counted in the divergence. Tunnelling is a separate physical path
**through** the barrier rather than into the trap plane, so it is injected
*after* that zeroing. Injecting before it silently deletes the whole channel
on any interface-bound stack — the diagnostics still report a perfectly good
flux and the terminal current does not move at all. That is exactly why
`test_an_enabled_channel_changes_the_certified_terminal_current` asserts on
the solved current rather than on the diagnostics.

### The inert family must not re-address the tree

Introducing `DeviceStack.tunnelling_channels` moved the frozen semantic
SHA-256 of **every** shipped config, including ones with nothing to do with
tunnelling — caught by `test_v1_shipped_device_semantic_hashes_remain_frozen`
and `test_matrix_covers_and_loads_every_shipped_config`. `reproducibility.py`
now drops the field from the digest, following the same rule the existing
inert-capability exclusions use.

The rule is on whether a channel is **enabled**, not on whether the key is
present. An all-disabled document compiles away entirely and is measurably
bit-identical to omitting it, so hashing it differently would content-address
a distinction the solver cannot make. An enabled family stays recursively
content-addressed in full, and the invariant is pinned both ways by
`test_an_inert_family_does_not_move_a_configs_semantic_hash`.

### Measured evidence (D8-E1)

| property | result |
|---|---|
| Kane quadrature vs closed-form Zener exponent | **1.4e-6** relative |
| disabled family vs no document | state arrays **bit-identical**, current identical |
| zero bias, wired lane | net flux exactly 0.0 — ~~by reciprocity~~ **RETRACTED: by float64 saturation; see D8-E2R** |
| enabled channel at 0.2 V dark | net flux `-1.95e10 m^-2 s^-1` = `-3.13e-9 A/m^2`. ~~19 % of the terminal current~~ **RETRACTED — 6.2e-7 with the true level; see D8-E2R.** The terminal current shifts by 1.2e-5 relative |
| local vs contact driving | flux **falls** >10x from 0.2 V to 0.5 V |
| mesh convergence of the flux | order **1.5**, 2.5 % change over the last doubling (273 nodes, energy order 384) |
| energy-quadrature convergence | 3.4e-3 relative change from order 192 to 384 |
| injected face current vs reported flux | agree to **1e-12** |
| inert family vs no family | identical semantic SHA-256; an enabled family differs |

Tests: 35 added (`tests/unit/physics/test_wkb_local_windows.py` 12,
`tests/integration/test_tunnelling_channels_device.py` 23); the 19 D8-E0
channel tests were re-anchored to `anchor_face` and still pass.

## Declared limitations

- Channel prefactors are supply-function estimates, not fitted to any
  reference. The tests assert transmission behaviour, reciprocity and the
  Kane exponent; **absolute channel magnitudes are not validated** against any
  external solver or measurement.
- The wiring is on the **guarded QF/DC lane only**. The transient MoL driver,
  the algebraic lanes and the 2D solver all fail closed rather than carrying
  the channels, so no shipped J-V is affected in either direction.
- Interface-bound channels bind to `interface_faces[0]`. A stack with several
  heterointerfaces runs the channel on the first one only; the other
  interfaces are not covered, and this is a scope limit rather than a physical
  statement about them.
- The channel flux converges at order 1.5 in the mesh and needs an energy
  quadrature order of ~192 to reach a 1e-2 relative change. Both are measured
  here but a **registered** refinement lane with frozen gates is D8-E2.

## Numerical certificate (D8-E2)

Lane `wkb-tunnelling-channel-qf-dc-v1`, config
`configs/wkb_tunnelling_intraband_spike.yaml`, executor
`perovskite_sim/validation/tunnelling_channel_refinement.py`.

### Three axes on a two-axis runner

`MatrixPoint` carries exactly `(grid, tolerance_factor)` and
`ConvergenceCheck.dimension` is a closed literal, enforced at five sites that
43 other lanes depend on. The energy quadrature order is therefore swept
**inside** each cell and reported as quality metrics — the same shape the four
existing energy-distributed lanes use. Grid `[24, 48, 96]` intervals per
electrical layer, tolerance `[1.0, 0.1, 0.01]`, energy order
`[96, 192, 384]`.

### The observable is the channel, not the terminal current

Enabling the channel moves the terminal current by ~1.2e-5 relative while the
channel itself reports ~20 % of it — **that 20 % is retracted (D8-E2R): with
the true quasi-Fermi level it is 6.2e-7**. The reason for choosing the channel
over the terminal current is unaffected and if anything strengthened: the
tunnelling path is in parallel with the drift-diffusion flux on the same face,
so the terminal current barely moves either way. A terminal-current gate would
therefore pass whether or not the channel worked, and the registered
observables are the channel's own net flux and maximum action. Both are grid-dependent, so the shared
grid/tolerance convergence check has something real to measure.

### What the gates are, and where their numbers come from

The exact identities are gated as **exact**, not with a tolerance — a
threshold there would hide a sign or bookkeeping error:

| gate | value | meaning |
|---|---|---|
| `equilibrium_net_flux_m2_s` | `le 0.0` | reciprocity survives the device wiring |
| `equilibrium_face_current_A_m2` | `le 0.0` | and reaches the face array as zero |
| `face_current_injection_relative_error` | `le 1e-12` | the injected current IS the reported flux |
| `injected_face_count` | `eq 1` | on exactly one face |
| `disabled_family_reports_nothing` | `eq 1` | a disabled family produces no diagnostics |
| `minimum_transmission_below_unity` | `eq 1` | the barrier actually blocks something |

The convergence gates are sized from the measured order rather than fitted:
`max_energy_flux_relative_change le 0.05` against a measured 2.3e-2, and the
observable limits at 0.1 against a measured 2.5e-2 over the last grid pair at
order 1.5. `residual_over_solver_limit le 1.0` is deliberately expressed as a
ratio against the solver's **own** accepted limit rather than a number chosen
here, so the gate cannot drift away from what `certified` means.

### Measured evidence (D8-E2)

Certificate `12f3a50cb9a6501a5551210ba97653fa0fb33dc52d2a086911883931007932cd`,
run `c983b07f8ec83e4d395edbbb472bc707b37b4e6a20c96b68a0dc436bf41bdce8`,
**status `certified`** — 9/9 cells, 156 checks, zero failed, zero unconverged
dimensions, 359 s.

| convergence (last pair) | grid | tolerance | limit |
|---|---|---|---|
| channel net flux | 4.11e-3 | 0.0 | 0.1 |
| channel face current | 4.11e-3 | 0.0 | 0.1 |
| channel maximum action | 3.84e-3 | 0.0 | 0.05 |
| dark terminal current | 1.93e-4 | 2.66e-10 | 0.02 |
| dark potential profile | 8.58e-5 | 0.0 | 0.002 |

The exact identities hold **exactly in all nine cells**: equilibrium net flux
0.0, equilibrium face current 0.0, injection relative error 0.0, injected face
count 1. Channel flux stays at 20.1-20.5 % of the terminal current across the
matrix; energy-order convergence spans 9.4e-3 to 2.3e-2 against the 0.05 gate.

### Why the action and not the transmission

The first certified run reported `intraband_electron_minimum_transmission` at
**0.1039 against a 0.1 limit** — the one failure. That is not a tolerance to
loosen: `T = exp(-2S)` with `S ~ 14` here, so a relative error in the exponent
is amplified by `2S ~ 28` in `T`. The action's own convergence is 3.84e-3, and
`28 x 3.84e-3 = 0.107`, which reproduces the observed 0.1039. Gating the raw
transmission would therefore have measured the exponential's amplification of
a converged exponent. The registered observable is the action; `T` is kept in
the cell provenance.

### Two defects this checkpoint surfaced

Registering the lane is what found both; neither was visible from D8-E1's own
tests.

1. **The mesh divergence** described above — the channel flux halved on every
   grid doubling because the drive was read across one grid cell. A
   convergence lane is exactly the instrument that catches this, and it would
   have certified a meaningless number had it been registered first.
2. **The backend inline path dropped the channel document.**
   `backend/main.py:stack_from_dict` did not carry `tunnelling_channels`, so
   an inline run from the editor silently lost the physics the YAML path
   applies — the drift class this repository has hit repeatedly. Caught by
   `test_standard_yaml_and_inline_backend_have_identical_semantics`, which
   compares the semantic hash across both paths for every shipped config.

## Retraction (D8-E2R)

### What was wrong

`experiments/quasi_fermi_steady_state.py` builds its solver variable as

```text
qfn0 = V_T*ln(n0) - (phi0 + chi)
```

Since `E_C = -(phi + chi)` and `E_Fn = E_C + V_T*ln(n/N_C)`, that is

```text
qfn0 = E_Fn + V_T*ln(N_C)
```

This is correct **for the solver**, which only ever uses `diff(qfn0)/V_T`
where the constant cancels. The channel wiring misread it: the channels feed
it to a Fermi-Dirac occupation as an absolute level. Measured on the lane
config, the offset is `V_T*ln(N_C)` = **1.428634 eV**, reproduced to twelve
decimals, and it puts the level **0.83 eV above** `E_C` where the true
quasi-Fermi level sits **0.60 eV below** it.

### Why it is not a defensible convention

Under Maxwell-Boltzmann the offset multiplies the occupation by exactly `N_C`
at every level, so "DOS folded into the level" would cancel against a `1/N_C`
supply prefactor and the scheme would be self-consistent. Measured ratios at
three levels: `1.000000e+24` each time. Under Fermi-Dirac the same offset
saturates the occupation at 1, so the ratio is `3.5e11 / 5.2e6 / 48.9` at
those same three levels — level-dependent, and nothing can cancel it. The
channels use Fermi-Dirac. Pinned by
`test_the_offset_is_only_a_constant_factor_under_boltzmann`.

### What the two retracted claims should have said

| retracted | measured with the true level |
|---|---|
| "channel carries ~19-20 % of the terminal current" | **6.2e-7** of it |
| "equilibrium net flux exactly 0.0 by reciprocity" | exactly 0.0 **by float64 saturation** |

The second is the more serious. The registered gate
`equilibrium_net_flux_m2_s: le 0.0` was documented here as "gated as exact,
not with a tolerance — a threshold there would hide a sign or bookkeeping
error." Measured, it passes because both Fermi factors round to the **same
double** near 1.0: the residual equilibrium quasi-Fermi gradient is 3.96e-13
eV, which perturbs an occupation of ~1 by ~1e-20, below the ~2.2e-16 ulp
there. Read at the true level the occupation is ~3e-12, the ulp is ~1e-27,
and the same gradient resolves to `max|f_L - f_R| = 9.37e-26` — non-zero.

So the gate could not fail, and **a gate that cannot fail is not evidence**.
It hid precisely the class of error its own docstring named. The negative
control that proves it discriminates is
`test_the_equilibrium_zero_is_saturation_not_reciprocity`.

### What survives

- The D8-E0 **unit-level** reciprocity claim. That test passes one occupation
  array as both sides, so the integrand is identically zero by construction —
  a property of `reciprocal_net_flux`, not of any device state.
- Mesh convergence at order 1.5, the energy-order convergence, and the
  injection identity (`face current == -Q * flux` to 1e-12, on one face).
  Those hold for any consistent level.
- Every fail-closed route and the disabled-family bit-identity.

### Why the convention was not simply flipped here

Correcting the level fails three registered gates at once
(`channel_flux_fraction_of_terminal_current: ge 0.01` against a corrected
6.2e-7; `equilibrium_net_flux_m2_s: le 0.0`; and the flux observables' own
limits), which flips the lane `certified -> failed` and cascades: the
benchmark `wkb-tunnelling-channel-internal-closure` loses `status: pass`, and
`configs/wkb_tunnelling_intraband_spike.yaml` (`status: partial`) then has no
physical benchmark evidence. Re-registering is a v2-lane project whose entry
condition is a config where the **corrected** channel clears `ge 0.01` — no
such config exists yet, and finding one is physics design rather than
plumbing. Publishing the envelope and leaving the failure visible is what
§13 of the roadmap prescribes, and is what this checkpoint does.

### One free fix that was in scope

The hole branch negated `qfp` before handing it over. `qfp0` is already in the
hole particle-energy convention that matches the `-E_V` barrier
(`V_T*ln(p) + (phi + chi + Eg)` = `-E_V + V_T*ln(p)`), so negating it put the
drive **12.9 eV below** its own barrier instead of 0.72 eV above, and the flux
underflowed to ~1e-204. Removing the negation moves it to 7.32e8 m^-2 s^-1.
The electron branch — and therefore the certified lane — is unchanged. This
makes the two branches consistent with each other; it does not make either
correct, since both still carry the level offset above.

## Required next checkpoints

1. **D8-E3** — the frozen channel-by-channel SCAPS comparison the roadmap
   asks for, which needs a real SCAPS deck and raw export that the repository
   still does not have. **Blocked on external input**, not on work here.
