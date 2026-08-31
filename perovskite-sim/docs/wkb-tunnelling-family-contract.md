# WKB tunnelling family contract (D8)

## Status and capability boundary

This document covers **D8-P0** (audit of the existing tunnelling surface) and
**D8-E0** (canonical four-channel contract plus the local WKB physics).

The current capability label is:

```text
four independently switchable WKB tunnelling channels with transmission,
reciprocity, validity diagnostics and unit tests, all LOCAL; no channel is
wired into any solver, no device-level current is produced, and no SCAPS
comparison has been made
```

Nothing in this checkpoint changes a shipped number: the canonical document
defaults every channel to disabled, and no solver imports the channels yet.

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

## Declared limitations

- The band-to-band channel uses the single-band (parabolic) WKB exponent with
  a reduced effective mass. The two-band Kane dispersion differs in the
  numerical prefactor; that correction is **not** applied and is not claimed.
- Channel prefactors are supply-function estimates, not fitted to any
  reference. The tests assert transmission behaviour and reciprocity;
  **absolute channel magnitudes are not validated** here.
- Every channel is local to the structure handed to it. **No solver wiring
  exists at this checkpoint**, so no device J-V is affected.

## Required next checkpoints

1. **D8-E1** — wire the channels into the guarded QF/DC lane: barrier
   extraction from `chi_phys` / `Eg_phys` and the live `phi`, per-interface
   geometry from the two-sided trace, and the defect-assisted channel bound to
   the explicit interface occupancy. Every other route fails closed.
2. **D8-E2** — a registered grid/tolerance/energy-order refinement lane for
   the channels, in the shape of the D7-E2 lane.
3. **D8-E3** — the frozen channel-by-channel SCAPS comparison the roadmap
   asks for, which needs a real SCAPS deck and raw export that the repository
   still does not have.
