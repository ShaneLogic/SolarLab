# Dynamic two-sided interface-defect device AC

## Scope

`run_interface_defect_device_impedance()` is the research-only D5-E2b lane for
one microscopic defect population at each physical two-sided interface. It is
not wired to the production impedance API or frontend. Bulk explicit defects,
mobile ions, selective contacts, field-dependent mobility, photon recycling,
and non-two-sided interface topologies remain fail-closed in this lane.

Bulk/interface and defect/ion combinations are intentionally deferred to
D5-E2c. A certificate from this adapter must not be reused for those models.

## Dynamic state and charge

Each physical interface has one shared electron occupancy `f`, represented by
a logit increment so every finite nonlinear probe satisfies `0 < f < 1`. The
conserved areal population and equilibrium-referenced sheet charge are

```text
N_occ = N_t f
Delta sigma_t = -q N_t (f - f_eq).
```

Both acceptor-like `Q=-qN_t f` and donor-like `Q=+qN_t(1-f)` populations give
the same equilibrium-referenced increment. No separate trap-character sign is
therefore accepted by this closure.

For the canonical local order `[n_L, p_L, n_R, p_R]`, the four capture legs are

```text
C_n,s = S_n [n_s (1-f) - n1_s f]
C_p,s = S_p [p_s f - p1_s (1-f)],       s in {L,R},
```

where the microscopic binding is exact: `S_n,p = sigma_n,p v_th N_t`. The trap
storage equation is

```text
d(N_t f)/dt = C_n,L + C_n,R - C_p,L - C_p,R.
```

At finite frequency the summed electron and hole captures need not be equal;
their difference is the trap-storage current. Treating them as a single
quasi-steady recombination rate would remove the physical relaxation branch.

## Self-consistent device coupling

For every carrier/occupancy/voltage probe, SolarLab performs all of the
following on the same nonlinear state:

1. Adds `Delta sigma_t` to the global finite-volume Poisson residual.
2. Re-solves the left/right electrostatic traces at that fixed sheet charge.
3. Re-solves all four zero-volume carrier traces at the supplied occupancy.
4. Drains the four resolved capture fluxes from their adjacent carrier control
   volumes through the existing conservative right-first material adapter.
5. Reports electron and hole conduction on every device face and derives
   displacement charge from the same solved electrostatic field.

The fixed-occupancy local capture Jacobian is analytic. The device small-signal
operator uses the shared central-difference engine

```text
(j omega M - A) delta u = (b - j omega m_V) delta V
```

with interior electron/hole QF increments and one interface-occupancy logit per
physical interface. Cancellation-safe DC QF face drops are retained directly;
the AC path does not reconstruct them by subtracting large absolute potentials.

## Certificate

The result is certified only when all of these independent checks pass:

- dark-reference grid, stack, state, document, capture-velocity, and density
  hashes remain exact;
- a supplied DC state is re-evaluated on the live AC current operator;
- fixed occupancy at the DC QSS value embeds the charged D4 operator;
- every local carrier and Gauss residual remains within its declared bound;
- left/right capture response and occupied storage satisfy the frequency-domain
  trap balance;
- electron, hole, and displacement admittances close on every device face;
- the complex solve has finite reciprocal condition and componentwise backward
  error;
- three decreasing finite-difference levels converge;
- the low-frequency result approaches the charged QSS operator and the
  high-frequency result approaches the frozen-occupancy operator;
- the requested frequency grid brackets every actual device-level interface
  relaxation corner with the declared margins and sampling density.

An insufficient frequency window can be inspected with
`require_certificate=False`, but the returned result stays uncertified and
records `trap_frequency_window_incomplete`.

## D5-E2b reference evidence

The real two-layer, one-interface integration case uses 45 logarithmic
frequencies from `1e-8` to `1e14 Hz` and three finite-difference levels. Its
interface relaxation corner is `2.687163 Hz`. The observed maxima are:

| Gate | Value |
|---|---:|
| live DC operator match | `1.191e-16` |
| DC normalized residual | `1.284e-9` |
| QSS embedding error | `1.562e-26` |
| local interface residual | `1.086e-14` |
| normalized Gauss residual | `1.723e-20` |
| local trap-balance error | `5.347e-12` |
| all-face admittance spread | `4.744e-6` |
| linear backward error | `2.131e-16` |
| minimum reciprocal condition | `4.773e-2` |
| finite-difference refinement change | `2.491e-8` |
| low-frequency QSS-limit error | `2.077e-8` |
| high-frequency frozen-limit error | `3.167e-20` |

This evidence establishes the narrow internal interface-only AC closure. It is
not external SCAPS validation, experimental validation, or a production API
claim.
