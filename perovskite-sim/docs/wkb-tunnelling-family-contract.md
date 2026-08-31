# WKB tunnelling family contract (D8)

## Status and capability boundary

This document covers **D8-P0** (audit of the existing tunnelling surface),
**D8-E0** (canonical four-channel contract plus the local WKB physics) and
**D8-E1** (production wiring into the guarded QF/DC lane).

The current capability label is:

```text
four independently switchable WKB tunnelling channels, each anchored to its
own barrier, wired into the guarded QF/DC lane so an enabled channel adds a
face current to the certified residual; every other solver route fails
closed; no SCAPS comparison has been made and no channel magnitude is
validated against any reference
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
opposite and is pinned as such: raising the bias flattens the junction, the
local drop shrinks, and the tunnelling flux falls by more than 10× between
0.2 V and 0.5 V.

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
| zero bias, wired lane | net flux **exactly 0.0**, face currents all exactly zero |
| enabled channel at 0.2 V dark | net flux `-1.086e9 m^-2 s^-1` = `-1.74e-10 A/m^2`, shifts the certified terminal current by 6.9e-7 relative |
| local vs contact driving | flux **falls** >10x from 0.2 V to 0.5 V |
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
- No refinement certificate exists yet, so grid, tolerance and
  energy-quadrature-order convergence of the channel currents is **unmeasured**
  (that is D8-E2).

## Required next checkpoints

1. **D8-E2** — a registered grid/tolerance/energy-order refinement lane for
   the channels, in the shape of the D7-E2 lane.
2. **D8-E3** — the frozen channel-by-channel SCAPS comparison the roadmap
   asks for, which needs a real SCAPS deck and raw export that the repository
   still does not have.
