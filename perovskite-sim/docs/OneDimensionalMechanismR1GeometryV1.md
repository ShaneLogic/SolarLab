# One-Dimensional Mechanism R1 Geometry V1

This contract is consumed by `experiments/one_dimensional_mechanism_r1.py`,
`scripts/run_one_dimensional_mechanism_r1.py` and the R1 geometry tests.
It is the R1-0 implementation contract, not a certificate for all of R1.
The preregistered study specification is archived as
`plans/Specs/OneDimensionalMechanismR1StudySpecV1.md`, SHA-256
`f140505c3a7ec7c52fe38c46f36d53b876a87c2ee50512568cb62671b05c6a73`.
The parent is `5b744b023ac708e50b6035616ca6f07cee3c9eb1`.

## Geometry and Conservation

The explicit identity is `physical_boundary_two_sided_v1`.
Control-volume faces are physical contacts, same-material node midpoints,
and the true material interface. The four-node widths are 25/75/75/25 nm.
The legacy material builder and public R0 protocol retain full endpoint widths.

For the positive-ion population, `w*dP/dt = F_left - F_right`.
The same `w` enters DC inventory, transient rate and analytic tangent, AC
rate/storage pairing, and all inventory and charge integrals. Storage rows
use density units and divide flux differences by w, which is algebraically
equivalent to area-density storage. The carrier divergence has the opposite
orientation to particle flux; the R1 ion Jacobian explicitly uses that sign.
No material density, site capacity or trap population is rescaled by mesh.
The finite-site diffusion-only law is unchanged: at theta=0.5 its
zero-field linear diffusivity is 2D.

Poisson uses the physical interior volumes and exact series face capacitance.
The sheet remains a zero-thickness population. Its left/right allocation is
the Schur elimination of the two-sided local Gauss equation.

Full charge is `sum(rho*w) + sigma`, including endpoint populations and
fixed compensation. The older interior integral retains its own name.
Physical displacement at the contacts is `D_first-rho_0*w_0` and
`D_last+rho_last*w_last`. Pinned carrier endpoints have zero storage:
pair recombination supplies opposite electron/hole contact-current corrections.
External ion flux is zero; internal ionic current is not metal-contact current.
Contact conduction plus displacement, both interface traces and every ordinary
face are checked without projecting the currents onto a common value.
All archived currents explicitly labelled contact/internal in the R1 record
are along +x; inherited engine and AC observables use junction-polarity
conjugate current.

## Precision

The incremental storage, logit and expm1 calculations remain in force.
A voltage change has a known harmonic dielectric potential lift.
R1 removes that exact solution from the Newton unknowns, shifts carrier QF
coordinates so physical populations are unchanged by the lift, and solves the
remaining small increments. The harmonic lift has zero Poisson divergence
and zero interface Gauss jump analytically. Previous residuals are retained;
independently reconstructed absolute Poisson and eliminated-operator checks
remain mandatory. Displacement observation restores the harmonic contribution.
This is a coordinate transformation, not removal of an initial charge error.

The R1-0 short test uses 0/10 ns/1 us/100 us at 0/5/5/5 mV, nested 1/2/4
backward-Euler steps. It is a geometry/coupling test. The first finite-step
current includes voltage charging and is not an instantaneous regular current.
Separate 0-/0+ records and the full long-time protocol remain R1-1/R1-2 work.
The nonlinear factor only scales the ten fields named in study section 10.
Iteration limits remain 100 Newton / 40 line search / 2 bounded nonmonotone.

## Fixed Reference

GEO-01..06 must pass before preparation. Independently solve charge-off,
dark zero-bias DC at 32/64/128 intervals, then double to at most 512 if needed.
The finest adjacent occupancy difference must be at most 1e-8.
`ReferenceBindingV1` binds the selected f_ref, all rung values, geometry,
microscopic document hashes and complete base-stack identity. Target meshes
independently solve with this fixed f_ref; they do not zero their sheet charge.
This R1-0 binding admits the base parameters only. Controlled rate changes and
common-state import are not yet exposed as an R1-1 API.

The preparation CLI requires a successful `geometry` completion record, not
merely a successful run from another stage. Its manifest must cover the
geometry arrays, JUnit/log evidence, input, resolved stack, execution contract,
source manifest and completion record; each listed file is verified. The input,
stack, physical grids, contract and solver/test source must match the current
preparation. Every required GEO-01..06 case on 4/16/32/64 grids must be present
in the JUnit record, without failures, errors, skips or duplicate cases.
Preparation saves the verified prerequisite hashes in
`GeometryPrerequisiteV1.json`. Changed numerical source or protocol requires
fresh geometry evidence; archived historical evidence is not overwritten.

The shared DC solver first retains its original warm-start solve. If its
normalized residual still exceeds the unchanged acceptance limit, it may make
one retry in increments about the last physical state. A zero increment avoids
the vanishing initial trust radius caused by roundoff-sized dark carrier
coordinates. The retry uses only the remaining function-evaluation budget;
the certificate counts both attempts and retains all original physical gates.
Accepted original solves are not retried. Four-interval coupled DC, short
transient and AC regressions exercise this recovery without claiming spatial
convergence or extending the formal 16/32/64 study axis.

## Gates and Evidence

GEO-01/02: positive, nonoverlapping volumes; layer width and nominal inventory
relative errors <=1e-12 on 4/16/32/64 intervals.
GEO-03: arbitrary-flux telescoping and production-rate volume agreement.
GEO-04: exact 2D tangent and second-order cosine decay convergence.
GEO-05: quadratic constant-charge oracle, contact Gauss law, and a separate
nonquadratic second-order spatial convergence test.
GEO-06: dielectric series capacitance, continuous trace potential and sheet jump.
The full Gauss absolute tolerance is `q*1e15*1e-10 C/m2`.

Every saved coupled step is checked: inventory drift <=1e-10, charge continuity
<=1e-10 with the original normalization, contact/internal total spread <=2e-6.
Original local capture/Gauss, Jacobian, eliminated-operator, time refinement,
and 1e-14 current-decomposition gates remain active.
The AC geometry check uses 45 frequencies from 1e-3 to 1e8 Hz and three
finite-difference factors 1/.5/.25. It retains all legacy AC gates and adds
`V_T*abs(sum(w*P_hat_per_V))/1e15 <= 1e-10`.
AC observations include physical contacts and both interface traces.
No A-D mechanism controls or time/frequency agreement are certified here.

Run from `perovskite-sim` using a single-thread-observable BLAS environment:

```sh
python scripts/run_one_dimensional_mechanism_r1.py list
python scripts/run_one_dimensional_mechanism_r1.py geometry --output-dir outputs/one_dimensional_mechanism_r1/geometry_v1
python scripts/run_one_dimensional_mechanism_r1.py prepare --geometry-evidence outputs/one_dimensional_mechanism_r1/geometry_v1 --output-dir outputs/one_dimensional_mechanism_r1/preparation_v1
python scripts/run_one_dimensional_mechanism_r1.py coupled --intervals 64 --nonlinear-factor .1 --reference outputs/one_dimensional_mechanism_r1/preparation_v1/ReferenceBindingV1.json --output-dir outputs/one_dimensional_mechanism_r1/coupled_v1
python scripts/run_one_dimensional_mechanism_r1.py ac --intervals 64 --reference outputs/one_dimensional_mechanism_r1/preparation_v1/ReferenceBindingV1.json --output-dir outputs/one_dimensional_mechanism_r1/ac_v1
python scripts/run_one_dimensional_mechanism_r1.py verify --output-dir outputs/one_dimensional_mechanism_r1/ac_v1
```

Use a fresh directory per execution. Outputs include strict JSON, accepted
substeps, failures, source ZIP/manifest/diff, environment and byte manifests.
Completed evidence is archived under `results/OneDimensionalMechanism/R1V1/`.
Internal numerical conservation is not real-material or experimental validation.
