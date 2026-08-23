# Ion-aware structured Jacobian comparison

Status: `INTERNAL_TESTED_ANALYTIC_TWO_SIDED_INTERFACE_REACTION` as of
2026-08-23.
This is a validation path for the ion-aware impedance reference engine.
Poisson, Scharfetter-Gummel transport, local bulk SRH/radiative/Auger
recombination, defect-free single-node interface SRH, clamp-inactive cross-node
`InterfaceDefect` SRH, its smooth unclipped Boltzmann projection, and finite-rate
outer selective contacts are analytic. The algebraic shared-occupancy
cross-interface SRH branch is also analytic on its positive-density,
clamp-inactive slice, as is the additive two-sided mirror pair.
The carrier transport tangent also includes the production Caughey-Thomas and
Poole-Frenkel field-mobility chain at differentiable operating points.
The QSS interface closure is not analytic. The separate QF
`two_sided_trace` zero-volume topology is outside this MoL comparison. This is
not yet a fully analytic production Jacobian, a registered numerical
certificate, or external validation.

## Purpose

`perovskite_sim.experiments.ion_aware_structured_jacobian` tests the global
chain rule in the eliminated-Poisson formulation before a production sparse
Jacobian is introduced. Protocol v10 retains six independently constructed
objects:

1. the existing nonlinear callback, which re-solves Poisson at every finite-
   difference stencil;
2. a frozen-potential finite-difference operator, used as a block-local
   transport reference;
3. a transport-structured operator with exact discrete Poisson sensitivities,
   analytic carrier/ion SG face currents, analytic CT/PF mobility tangents,
   and matching conservative flux-divergence rate rows;
4. a bulk-reaction hybrid, which additionally replaces local bulk
   SRH/radiative/Auger rate derivatives;
5. an interface-reaction hybrid, which also replaces defect-free single-node,
   clamp-inactive cross-node, smooth unclipped projected, and positive-density
   shared-occupancy/additive-two-sided interface SRH;
6. the final hybrid operator, which replaces finite-rate outer-contact rate
   rows while retaining frozen-potential finite differences only for explicitly
   unsupported interface closures.

All paths use the same certified DC state, state layout, voltage convention,
material cache, frequencies, current decomposition, and adaptive per-column
log-density stencil. Existing transient, DC, and current APIs continue to use
their original numerical paths; the analytic derivatives are opt-in here.

## Exact Poisson block

For each active log-density coordinate `u_j`, the charge derivative is

```text
d rho / d u_j = s_j * q * y_dc,j
```

at that coordinate's node, with `s_j = -1, +1, +1, -1` for electrons,
holes, positive ions, and negative ions. The code solves the already-factored
finite-volume Poisson matrix for every charge derivative. The voltage
derivative uses zero bulk charge and the exact right-boundary derivative
`-junction_polarity`.

The certificate directly evaluates the componentwise backward error of every
state and voltage sensitivity solve. Poisson therefore does not pass merely
because the final impedance happens to agree.

## Adaptive column stencil

A single log-density step is not numerically meaningful across this device.
On the N61 IonMonger state, maximum potential sensitivity spans roughly
`1e-52` to `2.5e2 V` per unit log increment. The comparison protocol chooses

```text
h_j = clip(target_potential_step / max(abs(d phi / d u_j)), h_min, h_max)
```

with defaults `target_potential_step=1e-9 V`, `h_max=1e-3`, and `h_min` bound
to the final impedance refinement level. For the default reference this is
`1e-5 * 0.25 = 2.5e-6`, not the coarse first level. The full-Poisson and
structured callbacks are both expressed in
the same scaled coordinates, so each operator column uses the same physical
stencil. The maximum ion step is checked against the site-occupancy limit
before linearization.

## Analytic transport block

`bernoulli_derivative` uses the cancellation-free power series near zero and
an `exp(-x)` form at large positive argument. Electron and hole face-current
Jacobians include direct left/right density derivatives and both potential
derivatives, with the fixed band-edge offsets retained in the SG argument.

The ion block covers both implemented steric laws:

- the legacy whole-flux steric multiplier, including its concentration
  derivative;
- the physical diffusion-only lattice-gas chemical potential;
- positive and negative charge signs;
- single-ion, distinct-site dual-ion, and shared-site dual-ion cross
  derivatives.

Each face derivative is chained through the exact Poisson sensitivity and the
per-column log-density scale. The same analytic face-current correction is
inserted into the corresponding electron, hole, positive-ion, or negative-ion
continuity divergence. Generation, selective-contact rows, and unsupported
interface closures remain the independently evaluated finite-difference
remainder at this stage; the contact block is replaced separately below.
This paired replacement is required: replacing current alone produced a
measured low-frequency all-face spread of `6.66e-1`; replacing its matching
conservative divergence reduced it below `4e-7` on N13/N61/N91.

The analytic lane fails closed for an active or near-switching thermionic cap,
a smoothed thermionic cap, exclusive interface transport, incomplete dual-ion
or field-mobility arrays, a non-differentiable field-mobility face, or an
active face on a steric clipping kink. A zero-diffusivity structural face is
allowed because every derivative there is identically zero.

## Analytic field-mobility chain

The production ordering is Poole-Frenkel first and Caughey-Thomas second:

```text
mu_pf = mu_0 exp(clip(gamma sqrt(abs(E)), -80, 80))
r = mu_pf abs(E) / v_sat
mu = mu_pf (1 + r^beta)^(-1/beta)
```

For an active CT channel define `w = r^beta / (1 + r^beta)`. Away from a
constitutive kink the exact signed-field tangent is

```text
dmu/dE = mu [(1 - w) d(log(mu_pf))/dE - w sign(E)/abs(E)].
```

Inside the unclipped hard PF branch,
`d(log(mu_pf))/dE = gamma sign(E) / (2 sqrt(abs(E)))`. The reusable physics
helper also differentiates the opt-in compact PF zero-field regularization;
the ion-aware impedance lane itself evaluates the historical hard expression.
The returned mobility is always evaluated by the production
`apply_field_mobility` function, so the certificate validates the tangent and
does not duplicate the constitutive value path.

Because each SG face flux is linear in mobility, the extra potential chain is

```text
dJ/dphi_left  += (J/mu) (dmu/dE) / dx
dJ/dphi_right -= (J/mu) (dmu/dE) / dx.
```

These terms then pass through the exact eliminated-Poisson state and voltage
sensitivities and enter both terminal current and the matching conservative
continuity divergence. A zero-mobility face has an exact zero flux/tangent and
is handled without division by zero.

An independent adaptive central stencil calls `apply_field_mobility` directly
on every face and species. Its electron and hole derivatives are separate
certificate quantities with a `5e-6` relative-error limit. The hard PF law is
not differentiable at `E=0`; CT is not differentiable there for `beta <= 1`;
and the exact PF clipping surfaces are also kinks. Exact contact with any of
those surfaces fails closed. For PF and non-smooth CT zero-field surfaces,
the global state and voltage stencils must additionally satisfy
`abs(delta E) / abs(E) <= 0.1`; if the reference minimum step cannot meet that
condition, the comparison is rejected instead of using a secant as a tangent.
The independent local stencil also rejects any PF clipping-surface crossing.

## Analytic bulk-reaction block

The production bulk sink is

```text
R = R_SRH + B (n p - ni^2) + (C_n n + C_p p) (n p - ni^2)
R_SRH = (n p - ni^2) / D
D = tau_p (n + n1) + tau_n (p + p1)
```

The physics helper returns the production rate and exact local derivatives. For
SRH it evaluates

```text
dR_SRH/dn = (p - R_SRH tau_p) / D
dR_SRH/dp = (n - R_SRH tau_n) / D
```

and adds the direct radiative and Auger derivatives. Each carrier coordinate is
the scaled log-density tangent `delta n = n h_j` or `delta p = p h_j`. The same
local `-delta R` is inserted into the electron and hole continuity rows at that
node. Ion columns and the direct voltage derivative are exactly zero.

An independent local central stencil is retained as validation evidence and to
remove the corresponding block from the composite frozen-potential remainder;
the final rate matrix contains the analytic tangent. The expected `sinh(h)/h`
truncation difference is visible in the comparison and bounded explicitly.

The lane fails closed when the SRH denominator is non-finite or non-positive,
when self-consistent radiative reabsorption makes the source nonlocal, or when
heterojunction recombination de-spiking introduces a cross-node geometric-mean
tangent. These branches remain valid in the ordinary solver and finite-
difference reference; they are only outside this analytic lane.

## Analytic interface-reaction block

For a defect-free interface, the electron and hole evaluation nodes both equal
the interface control-volume node. For a declared `InterfaceDefect`, the
production cross-carrier path instead samples electrons from `idx + 1` and
holes from `idx - 1`, while applying both continuity losses at `idx`. In either
case the unclamped production surface sink is

```text
R_s = (n p - ni^2) / D_s
D_s = (n + n1) / v_p + (p + p1) / v_n
R_vol = R_s / dx_cell
```

For the cross-node defect path, `ni^2` in this notation is the cached detailed-
balance reference `n_R,eq * p_L,eq`; it is not the local intrinsic-density
square at `idx`.

The exact density derivatives are

```text
dR_s/dn = (p - R_s / v_p) / D_s
dR_s/dp = (n - R_s / v_n) / D_s
```

and the assembled log-coordinate columns are
`-(dR_s/dn) n h_j / dx_cell` or
`-(dR_s/dp) p h_j / dx_cell` in both carrier continuity rows. For the cross-
node path, those columns belong to `n[idx + 1]` and `p[idx - 1]`; the two target
rows remain `n[idx]` and `p[idx]`. Without Boltzmann projection, ion columns and
the direct voltage derivative are zero. A blocked electron or hole capture
channel retains the production limit `R_s = 0` with a zero tangent only for the
defect-free local path.

The production solver defaults to `R_s = max(R_s_raw, 0)` for cross-node
defects. Protocol v10 certifies only the differentiable, clamp-inactive slice.
It requires `SOLARLAB_IFACE_ALLOW_GEN != 1`, `R_s_raw > 0` at the operating
point, and `R_s_raw > 0` at both sides of every active log-density central
stencil. A negative or exactly zero operating rate, or any stencil that crosses
zero, fails closed instead of receiving a zero or one-sided derivative. The
linearization report records the electron/hole evaluation nodes, the cross-node
interface indices, and the minimum raw surface rate over the operating point
and accepted stencils. This is a branch certificate, not a smoothing of the
production clamp.

When `interface_plane_projection=True`, the certified cross-node slice also
uses the exact production Boltzmann projection before SRH evaluation. With
`R=idx+1`, `L=idx-1`, and

```text
a_n = (phi_idx - phi_R) / V_T
a_p = (phi_L - phi_idx) / V_T
n_tilde = n_R exp(a_n)
p_tilde = p_L exp(a_p)
ni_tilde^2 = ni_eff^2 exp(a_n + a_p),
```

the joint `ni_eff^2` projection preserves the detailed-balance numerator:
`n_tilde*p_tilde-ni_tilde^2 = exp(a_n+a_p)*(n_R*p_L-ni_eff^2)`.
For any state or voltage direction `q`, the analytic surface-rate tangent is

```text
dR_s/dq = R_n n_tilde (dlog(n_R)/dq + da_n/dq)
         + R_p p_tilde (dlog(p_L)/dq + da_p/dq)
         - ni_tilde^2 (da_n/dq + da_p/dq) / D_s,
```

where `da_n/dq=(dphi_idx/dq-dphi_R/dq)/V_T` and
`da_p/dq=(dphi_L/dq-dphi_idx/dq)/V_T`. The exact eliminated-Poisson
sensitivities therefore introduce physically required global carrier/ion
columns and a direct voltage derivative; both sink rows still receive
`-dR_s/(dx_cell dq)`.

This projection tangent is admitted only while production's exponent cap is
strictly inactive at the operating point and at both sides of every declared
state and voltage stencil. Touching or crossing `|a|=40` fails closed. The
report records projected interface indices and the minimum remaining exponent
margin. This certifies the smooth interior branch; it does not differentiate
the hard cap itself.

When `interface_shared_occupancy=True`, production evaluates that algebraic
branch before projection, QSS, or the two-sided mirror pair. At a declared
defect it forms

```text
n_sum = n_L + n_R
p_sum = p_L + p_R
ref_sum = (n_L,eq + n_R,eq) (p_L,eq + p_R,eq)
n1_sum = n1_L + n1_R
p1_sum = p1_L + p1_R
R_s = (n_sum p_sum - ref_sum)
      / ((n_sum + n1_sum)/v_p + (p_sum + p1_sum)/v_n).
```

The four production inputs are `n[idx-1]`, `n[idx+1]`, `p[idx-1]`, and
`p[idx+1]`; both losses remain at `idx`. Consequently each electron sample has
the same density derivative `dR_s/dn_sum`, each hole sample has
`dR_s/dp_sum`, and its scaled log-coordinate column multiplies that derivative
by the individual sampled density and `h_j`. This branch has no potential or
direct voltage dependence, so ion columns and its voltage forcing are exactly
zero. If projection is also enabled, shared occupancy wins exactly as it does
in production; no projected derivative is invented.

Production floors each of the four raw sampled densities at zero. Protocol v10
accepts only strictly positive operating densities and positive values on both
sides of every log-density stencil, so those floors are inactive. It also
applies the same positive raw-rate clamp gates as the ordinary cross-node
branch. The report records shared-occupancy indices, the minimum individual
density-floor margin, and the minimum accepted raw-rate margin. Nonpositive
density inputs, incomplete per-side trap/equilibrium arrays, or a clamp-active
or clamp-crossing state fail closed.

When `interface_two_sided=True` and shared occupancy is inactive, production
first evaluates the ordinary cross-node pair A and then adds the mirror pair

```text
R_A = SRH(n_R, p_L; ni_A^2 = n_R,eq p_L,eq)
R_B = SRH(n_L, p_R; ni_B^2 = n_L,eq p_R,eq).
```

Both rates use the same `n1`, `p1`, calibrated capture velocities, interface
control-volume width, and two sink rows. Pair B contributes direct columns at
`n[idx-1]` and `p[idx+1]`. Production floors those two minority densities at
zero and applies pair B only for `R_B>0`; protocol v10 therefore requires
strictly positive densities and a strictly positive raw pair-B rate at the
operating point and every state/voltage stencil. Pair A retains its own
independent clamp certificate.

With Boltzmann projection, pair B uses

```text
a_n,B = (phi_idx - phi_L) / V_T
a_p,B = (phi_R - phi_idx) / V_T
ni_B,projected^2 = ni_B^2 exp(a_n,B + a_p,B),
```

and the same exact projected-SRH chain rule as pair A, including global
carrier/ion columns and direct voltage forcing. The production exponent-cap
gate is checked for both pairs. Independent central and complex-step state and
voltage objects validate the sum before it replaces the finite-difference
remainder. If shared occupancy is also enabled, its earlier production branch
continues before pair B; the analytic lane preserves that precedence and
reports no two-sided contribution.

This additive mirror closure must not be confused with the separately
implemented QF `two_sided_trace` control-volume topology. The latter changes
the grid and locally eliminates a zero-volume interface system; it is neither
executed nor certified by this MoL impedance lane.

Two independent numerical objects are retained for both state and voltage
directions. The ordinary double-precision central stencil is subtracted from
the composite frozen-potential operator so the same nonlinear block is
replaced. A separate complex-step object validates the analytic formula,
projection chain, and topology/volume assembly without the subtraction
cancellation seen when a large surface rate is differenced at the N91 minimum
step. The certificate compares the analytic matrix/vector to this complex-step
reference, while the final global rate and impedance gates still compare
against the full nonlinear central-difference operator.

This interface slice fails closed for clamp-active or clamp-crossing defect
states, a projection-cap-active or cap-crossing stencil, the allow-generation
escape branch, a mismatch between declared defects and material sampling, the
QSS local root solve, dynamic interface-plane states, exclusive interface
transport, non-aligned interface arrays, invalid dual-cell widths, or
non-finite/nonphysical inputs. Those models remain available in the ordinary
solver; protocol v10 does not certify their tangent.

## Analytic selective-contact block

For every configured finite-rate outer contact, the production Robin currents
are

```text
J_n,L = +q S_n,L (n - n_eq)    J_n,R = -q S_n,R (n - n_eq)
J_p,L = -q S_p,L (p - p_eq)    J_p,R = +q S_p,R (p - p_eq)
```

Combining these signs with the electron and hole continuity divergences gives
the same density derivative for all four carrier/side channels,

```text
d(rate_contact) / d(density) = -S / dx_cell.
```

The scaled log-coordinate entry is therefore
`-S * density * h_j / dx_cell` in the matching boundary rate row. Contact
reservoir densities and surface velocities are cached constants in the
production model, so ion columns and the direct voltage derivative are exactly
zero. Dirichlet carrier/side channels have no dynamic coordinate and contribute
no contact tangent; an explicit `S=0` blocking channel retains a dynamic
coordinate with an exact zero contact tangent.

The independently constructed central matrix calls the production
`selective_contact_flux` for both perturbation signs and retains its full
carrier/side sign chain. Its expected difference from the analytic log tangent
is the bounded `sinh(h_j)/h_j` truncation factor. Non-finite or negative `S`, a
non-positive active boundary density/reservoir, invalid control-volume width,
or an active contact absent from the state layout fails closed. This block does
not certify heterointerface thermionic caps; those remain under the transport
capability gate. Contact thermodynamic compatibility is also a separate
certificate axis.

## Comparison contract

The mass block uses the exact affine tangent of `y_dc * exp(u)`. The rate rows
are scaled by operating storage, matching the frequency solver. Columns are
grouped separately as electron, hole, positive ion, and negative ion so a
fast ionic block cannot hide a carrier error.

Every column must pass either its declared self-relative error gate or a
`1e-6` absolute-error gate normalized to that species group's dominant
column. Columns smaller than `1e-4` of the group scale are additionally
listed in `bounded_weak_columns`; columns whose relative comparison is noisy
but whose group-normalized error is bounded are listed in
`absolute_bounded_columns`. A column that satisfies neither gate appears in
`failed_columns` and fails the certificate. Final impedance magnitude and
phase are independently compared with the converged three-level reference
response.

Default gates include:

| Quantity | Limit |
|---|---:|
| Poisson componentwise backward error | `2e-12` |
| mass column error | `2e-7` |
| group-normalized absolute column error | `1e-6` |
| storage-voltage derivative error | `1e-12` |
| storage-scaled rate column error | `5e-5` |
| total conduction column error | `1e-4` |
| displacement-charge column error | `1e-5` |
| named current-component column error | `1e-4` |
| analytic SG transport column error | `5e-6` |
| analytic SG transport voltage error | `5e-6` |
| analytic electron/hole `dmu/dE` error | `5e-6` |
| non-smooth field-stencil fraction | `0.1` |
| analytic bulk-reaction column error | `5e-6` |
| analytic local/cross-node/projected interface reaction column error | `5e-6` |
| analytic selective-contact column error | `5e-6` |
| impedance magnitude error | `1e-4` |
| impedance phase error | `1e-3 deg` |

Unknown protocol fields, a mismatched impedance protocol hash, invalid step
bounds, an occupancy-crossing stencil, a failed reference certificate, or a
failed comparison gate all fail closed. Diagnostic mode returns the complete
failed evidence without promoting it.

## Current evidence

The real N13 single-ion and symmetric dual-ion integrations pass. The N61
single-ion probe uses 138 dynamic coordinates. Its adaptive steps range from
`2.5e-6` to `1e-3`, with a median `4.13e-5`. A three-frequency probe spanning
`1e-4` to `1e6 Hz` observed:

| Check | N61 observed | Limit |
|---|---:|---:|
| Poisson backward error | `4.16e-13` | `2e-12` |
| storage-scaled rate column error | `1.67e-7` | `5e-5` |
| analytic bulk-reaction column error | `1.67e-7` | `5e-6` |
| analytic local-interface reaction error | `5.87e-11` | `5e-6` |
| conduction self-relative error | `2.70e-4` | dual gate |
| conduction group-normalized error | `3.37e-8` | `1e-6` |
| analytic transport self-relative error | `1.90e-4` | dual gate |
| analytic transport group-normalized error | `3.00e-8` | `1e-6` |
| displacement group-normalized error | `2.24e-9` | `1e-6` |
| impedance magnitude error | `1.18e-8` | `1e-4` |
| impedance phase error | `1.44e-7 deg` | `1e-3 deg` |
| all-face admittance spread | `2.53e-7` | reference protocol |

The structured comparison itself took `0.39 s` in the recorded single-thread
N61 probe after the DC state was available. This is validation evidence, not a
production performance claim; the code still assembles dense matrices and
retains finite-difference work for unsupported-interface reactions.

An N91 probe also passes the dual column gate. Its Poisson backward error is
`1.05e-12`; the largest self-relative analytic hole-current discrepancy is
`1.22e-3`, while its error normalized to the dominant hole block is
`2.45e-7`. The classification is retained as absolute-bounded weak-column
evidence rather than reported as false high-relative-accuracy evidence. Its
analytic bulk-reaction error is `1.67e-7`, analytic local-interface error is
`1.30e-10`, impedance magnitude error is `6.85e-8`, and all-face spread is
`3.78e-7`.

An independently settled N13 IonMonger variant with all four outer Robin
channels active (`S_n,L=1e-3`, `S_p,L=1e3`, `S_n,R=1e3`, `S_p,R=1e-3 m/s`)
also passes. Its analytic-contact error is `1.67e-7`, full storage-scaled rate
error is `2.48e-6`, impedance magnitude error is `2.36e-8`, impedance phase
error is `1.91e-7 deg`, all-face spread is `2.84e-8`, and linear-system
backward error is `1.74e-16`. This is a numerical boundary-operator test, not a
claim that those four demonstration velocities are experimentally calibrated.

A real four-node IonMonger closure activates PF holes in the HTL
(`gamma_p=3e-4 (V/m)^-0.5`) and beta-2 CT electrons and holes in the absorber
(`v_sat=1e5 m/s`). Its final log-density reference step is `1e-7`; the maximum
PF state and voltage field fractions are `1.43e-3` and `4.31e-5`. Electron and
hole local `dmu/dE` errors are `2.92e-9` and `1.39e-12`, the full storage-scaled
rate error is `2.51e-6`, impedance magnitude error is `7.53e-8`, phase error is
`1.34e-7 deg`, and all-face spread is `1.36e-9`. A finer-grid HTL example with
near-zero PF faces is deliberately rejected when no finite-difference step is
both locally differentiable and numerically resolvable. This is constitutive
and operator-assembly evidence, not calibration of the demonstration mobility
parameters.

The structured unit layer is `23 passed`. The real single-ion, symmetric
dual-ion, N61, N91, active-selective-contact, active-field-mobility, and active
cross-node-defect integration layer is `10 passed`. In the unprojected
IonMonger defect
case the minimum raw cross-node rate over the accepted stencils is
`9.2599e12 m^-2 s^-1`; the analytic-to-complex-step interface error is
`8.56e-11`, the full rate error is `4.10e-6`, and the impedance magnitude and
phase errors are `2.40e-8` and `1.84e-7 deg`. The projected N13 variant has a
minimum raw-rate margin `4.0500e12 m^-2 s^-1` and exponent-cap margin `36.8276`.
Its analytic-to-complex-step interface state error is `2.72e-10`, voltage
error is `3.92e-16`, full rate/voltage errors are `4.10e-6` and `3.09e-9`, and
the impedance magnitude/phase errors are `3.07e-8` and `2.29e-7 deg`.
The shared-occupancy N13 variant has minimum raw-rate and density-floor margins
of `9.2599e12 m^-2 s^-1` and `8.8197e11 m^-3`. Its analytic-to-complex-step
interface error is `1.21e-11`, full rate/voltage errors are `4.10e-6` and
`1.64e-9`, and impedance magnitude/phase errors are `3.90e-8` and
`2.56e-7 deg`. The projected additive-two-sided N13 variant has a pair-B rate
of `2.4439e7 m^-2 s^-1`, minimum pair-B raw-rate and density-floor margins of
`2.4371e7 m^-2 s^-1` and `8.8198e11 m^-3`, and projection-cap margin
`36.8276`. Its analytic-to-complex-step interface error is `1.14e-10`, full
rate/voltage errors are `4.10e-6` and `6.62e-9`, and impedance magnitude/phase
errors are `3.07e-8` and `2.02e-7 deg`. The repository-wide suite is `2079
passed, 2 skipped, 263 deselected`; the focused ion-aware/interface domain is
`103 passed, 2 deselected`. These are internal numerical checks, not external
validation.

## Remaining work

1. Extend the interface tangent to QSS only after its implicit and non-smooth
   branch semantics have a differentiable contract.
2. Replace the remaining frozen-potential reaction differences block by block,
   preserving these per-column comparisons.
3. Introduce sparse or matrix-free assembly only after analytic parity passes.
4. Complete frequency-window coverage and transient lock-in cross-checks
   before routing ion-aware impedance through public backend or frontend APIs.

Contact thermodynamics, external IonMonger or Driftfusion comparison, and
experimental impedance validation remain separate evidence axes.
