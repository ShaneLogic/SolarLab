# Interface Defect Transient Increment Contract

## Physical and Numerical Scope

This contract applies to `interface_defect_transient.py` and its
`interface_defect_ion_transient.py` extension. Both retain backward-Euler
time integration of bulk carriers and interface occupancy, coupled to
algebraic bulk and two-sided interface electrostatics. The extension also
advances its declared positive and negative mobile-ion populations.

The physical reference for interface charge, material parameters, ion
backgrounds, site capacities and contact conditions is fixed throughout a
trace. Changing a numerical coordinate origin must not change any of these
physical quantities. No new kinetic coefficient or acceptance limit is
introduced here.

## Coordinates Relative to Each Accepted Step

Each Newton solve starts from zero increments about the preceding accepted
state. For dimensionless carrier quasi-Fermi increments `u_n`, `u_p` and
potential increment `u_phi = delta_phi / V_T`, the interior densities obey

```text
n_new = n_previous * exp(u_n + u_phi)
p_new = p_previous * exp(u_p - u_phi)
P_new = P_previous * exp(u_P)
f_new = logistic(logit(f_previous) + u_f)
```

Carrier and ion storage differences are evaluated with `expm1`; the
interface occupancy difference uses `logit_occupancy_increment`. Local
interface potentials and log trace densities use the corresponding previous
state as their numerical reference. Contact values are still determined by
the imposed voltage and the resolved physical contact model.

`rebase` creates a shallow copy of the numerical system and a separate
previous-state record with zero local coordinates. Shared immutable
material data and physical charge references remain fixed. The original
system supplies storage, Poisson and interface error scales to `_solve_step`;
rebasing the densities therefore does not reset the declared accuracy
reference. The integration trace retains cumulative coordinates for its
history, while current and storage differences use the two states expressed
in the same local coordinate system.

This avoids subtracting two order-one accumulated coordinates to recover a
change near `1e-16`. Resolving that difference matters when displacement
current divides a small electric-field change by a sub-nanosecond time step.

## Electrostatic Residuals

Let `L` be the existing conservative Poisson operator, `W` its control-volume
weights, and `B` the existing interface-sheet allocation. The nonlinear
residual is evaluated as

```text
R_phi,new = R_phi,previous + L delta_phi + W delta_rho + B delta_sigma
delta_rho = q (delta_p - delta_n + delta_P_positive - delta_P_negative)
delta_sigma = -q N_t delta_f
```

`L delta_phi` is formed by taking the difference of the face fluxes
`C_face * diff(delta_phi)`. The previous residual is retained; the method
does not silently discard an initial electrostatic error. The analogous
interface increment includes the potential-jump constraint and

```text
delta_R_Gauss = C_left  (delta_phi_trace,left  - delta_phi_bulk,left)
              + C_right (delta_phi_trace,right - delta_phi_bulk,right)
              - delta_sigma.
```

These are algebraic rearrangements of the same nonlinear equations. The
analytic sparse Jacobian keeps its original physical derivatives. The
physical density, potential and interface state are still reconstructed on
every evaluation for transport and reaction rates.

Each accepted state retains a separate direct Poisson residual computed
from its reconstructed absolute fields. The reported maximum Poisson
residual covers both forms. Interface diagnostics also retain the directly
evaluated electrostatic trace residuals, and the full transient is compared
against the independently eliminated electrostatic operator. Agreement of
the increment equations alone is not sufficient for certification.

## Conservation and Acceptance

Conduction currents use the accepted physical carrier, trap and ion state.
Displacement current uses the potential increment from that same time step,
including both interface traces. Its time denominator is the same `dt` as
the storage equations. Particle inventories, integrated charge and every
face's total current retain their existing independent checks.

The historical grid-eight defect-dominated case must pass its original
4/8/16 nested substep calculation, nonlinear residual limit, `2e-6` all-face
and two-sided interface current limits, and inventory constraints. The
finite-difference Jacobian comparison and independent eliminated-operator
comparison remain active. Other registered cases retain their frozen
thresholds and physical histories.

The optional bounded nonmonotone line-search policy remains available. A
successful calculation is not required to exercise it. The repaired
grid-eight test requires convergence under the default monotone policy and
the same physical result when the bounded option is enabled. Historical
iteration-27 failures remain in the foundation-stage evidence archive;
requiring the old numerical failure is no longer a current test assertion.

The original dark two-sided interface and declared mobile-ion scopes remain
in force. Numerical conservation and time refinement do not establish
experimental trap lifetimes, ion migration rates or general chemical
degradation kinetics.
