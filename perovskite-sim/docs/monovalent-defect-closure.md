# Monovalent Bulk-Defect Local Closure

Status: DEF-2 solver-independent constitutive slice. This module is tested but
is not connected to production Poisson, contacts, J-V, impedance, or transient
state equations. Charged `explicit_quasi_steady` device execution therefore
continues to fail closed until DEF-3.

## 1. Scope

`perovskite_sim.physics.defect_closure` consumes the canonical DEF-0
`BulkDefectSpecies` inventory and evaluates a local Maxwell-Boltzmann,
single-level, quasi-steady closure. One call produces, from the same occupancy:

- per-species and total SRH recombination;
- acceptor, donor, or neutral defect charge;
- analytic carrier-density derivatives;
- the potential-direction derivative at fixed quasi-Fermi levels;
- immutable, JSON-compatible diagnostics and a content identity.

The closure does not mutate its inputs, clip occupancy, solve a global state,
or add charge to a production residual.

## 2. Local equations

For species `i`,

```text
c_n = sigma_n * v_th,n
c_p = sigma_p * v_th,p
n1  = N_C * exp(-(E_g - E_t) / V_T)
p1  = N_V * exp(-E_t / V_T)

D   = c_n * (n + n1) + c_p * (p + p1)
f   = (c_n * n + c_p * p1) / D
R   = N_t * c_n * c_p * (n*p - n_i^2) / D
```

The signed charge-number density is

```text
acceptor: -N_t * f       (empty neutral, filled negative)
donor:     N_t * (1-f)  (empty positive, filled neutral)
neutral:   0
```

and `rho_def = q * N_charge`. Multiple species are evaluated independently and
then summed. A neutral species uses the same recombination equation but its
charge and every charge derivative are exactly zero.

## 3. Analytic tangent contract

The public result records these local partial derivatives:

```text
dR/dn, dR/dp                         [s^-1]
df/dn, df/dp                         [m^3]
drho/dn, drho/dp                     [C]
```

It also records a directional derivative used by a quasi-Fermi/Poisson
linearization. Holding electron and hole quasi-Fermi levels fixed gives

```text
dn/dphi =  n / V_T
dp/dphi = -p / V_T

d rho/dphi|QF = (d rho/dn * n - d rho/dp * p) / V_T
d R/dphi|QF   = (d R/dn   * n - d R/dp   * p) / V_T
```

This is not an explicit trap-energy derivative. At fixed `n` and `p`, a level
referenced to the local band edge has `partial rho_def / partial phi = 0` in
this closure. DEF-3 must use the derivative matching its chosen global
coordinates and must not mix these meanings.

## 4. Supported input

DEF-2 accepts:

- one or more uniquely named canonical species;
- `single_level` distributions with integrated `total_density_m3`;
- `neutral`, `acceptor`, or `donor` charge transitions;
- independent electron/hole cross sections and thermal velocities;
- zero on one capture leg, which gives zero recombination but a finite
  occupancy controlled by the remaining capture/emission pair;
- finite non-negative carrier densities and positive `T`, `E_g`, `N_C`, `N_V`.

The following remain fail closed:

- unresolved charge transition or unnamed/duplicate species;
- Gaussian or other energy distributions;
- non-unit degeneracy (it is retained by the schema but not ignored);
- both capture legs equal to zero, because charged occupancy is then
  kinetically undefined;
- Fermi-Dirac trap kinetics, multivalent states, metastable transitions,
  spatial grading, or dynamic occupancy.

## 5. Result and identity

`MonovalentDefectClosureResult` stores thermodynamic context, `n1/p1`, capture
coefficients, kinetic denominators, occupancy, occupied density, signed charge,
recombination, all analytic derivatives, and totals. Every NumPy array is a
read-only copy.

`closure_identity_sha256` covers:

- closure version and Maxwell-Boltzmann statistics label;
- temperature, band gap, and effective DOS;
- every canonical species field in declared order.

It identifies the local constitutive document, not a complete device,
protocol, grid, or numerical certificate.

## 6. Verification boundary

The focused tests cover closed-form acceptor/donor charge, agreement with the
existing charged-trap primitive, the DEF-1 neutral limit, equilibrium detailed
balance, centered differences for every tangent, multi-species totals,
`N_t -> 0`, one-leg capture limits, band-edge levels, 180--420 K extreme
states, immutable serialization, and fail-closed inputs.

Passing these tests establishes a local constitutive closure only. Production
charged-defect capability additionally requires, in DEF-3, a shared contact
neutrality reference, Poisson and continuity coupling, a structured global
Jacobian, dark/light/bias integration tests, and numerical certification.
