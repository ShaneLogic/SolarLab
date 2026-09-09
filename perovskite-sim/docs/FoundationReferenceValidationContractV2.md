# Foundation Reference Validation Contract V2

## Fixed Physics and Accuracy

This extends the unchanged physical p-i-n input, 300 K temperature,
monochromatic photon/power definitions and acceptance limits in
[FoundationReferenceValidationContractV1.md](FoundationReferenceValidationContractV1.md).
The active steady reference is `foundation-physical-homojunction-resolved-v2`.
No material coefficient, current normalization or accuracy limit changes.

The earlier `foundation-physical-homojunction-steady-v1` remains registered
with its uniform 44/88/176 grids and original sampling. Both environments
completed all nine valid records, but 14 spatial profile checks failed.
Its terminal observables and solver-tolerance checks passed. It remains
partial evidence and must not be described as a converged internal state.

## Resolved Interfaces

The new meshes have 88/176/352 total intervals, still allocated in the
3:16:3 layer ratio. The existing multilayer tanh generator is used with an
absorber concentration parameter of 3. For each adjacent layer, a scalar
root solve chooses the tanh parameter giving the same physical first/last
cell spacing as the absorber. Physical layer boundaries remain grid nodes.
All solved tanh parameters and actual coordinates are recorded.

The first 88-interval illuminated state uses the density initializer followed
by the full QF solve. Finer meshes prolong that certified coarse QF state
and independently solve the fine-grid QF equations. The physical gates,
tolerance factor and final observations all belong to the target mesh.

This resolves abrupt doping and optical boundaries without changing their
physical locations or values. Equal spacings on both sides of each internal
boundary also make the absorber's dual-cell volume sum equal its specified
400 nm thickness. A uniform ionic population therefore has the same total
particle number across these meshes. Fine numerical cells describe the
continuum equations; they do not provide an atomistic material model.

## Physical Profile Reconstruction

Linearly interpolating `ln(n)` between an ideal contact and its first
interior point is inappropriate under finite diffusion current. The pinned
minority concentration can be below `1 m-3` while the first interior value
is many orders larger. The density is nearly linear very close to that
contact; its logarithm is not. The v1 comparison consequently reported log
differences near 15 at a fixed position even though terminal currents agree.

The v2 carrier sampling uses the same local steady drift-diffusion solution
that defines the Scharfetter-Gummel face flux. For a fractional position `s`
inside a face interval and the signed transport-potential difference `xi`,

```text
w(s) = expm1(s*xi) / expm1(xi) = s * B(xi) / B(s*xi)
c(s) = (1-w(s))*c_left + w(s)*c_right
```

`B` is the existing stable Bernoulli function. Electrons use the positive
transport-potential difference and holes its negative. At zero field this
is linear density interpolation; at zero current it gives the correct
Boltzmann exponential. Independent tests verify both limits and constant
current under subdivision. The result supplies physical quasi-Fermi levels
and recombination through their existing constitutive functions.

The v1 re-evaluation with this reconstruction alone still leaves about
3.3 percent equilibrium log-density variation between 88 and 176 intervals.
It also leaves short-circuit profile differences near an abrupt optical
boundary. The local mesh refinement is therefore required independently of
correcting the sampling method.

Each layer retains its 17 uniform interior comparison positions and adds
fixed distances of 0.05, 0.1, 0.2, 0.5, 1, 2, 5 and 10 nm from both
boundaries. The same 99 physical positions are used across the complete
matrix. These are mathematical continuum-profile checks, including near
ideal contacts; microscopic contact structure is outside this model.

## Evidence and Remaining Scope

The 3 by 3 spatial/tolerance matrix still contains the independent 20/10/5 mV
sampling study, current root, maximum-power search, and all four operating
states. Every observable and physical quality limit is identical to v1.
Convergence of the terminal quantities alone does not pass the profile gates.

Raw nodal states, signed currents, generation and recombination are retained
alongside comparison positions and sampling rules. Source, input, environment
and protocol identities bind each record. The first partial v1 evidence is
retained separately. Dynamic, dimensional, chemical and external-validation
boundaries remain those in the foundation plan and v1 physical contract.

## Mobile-Ion Reference History

The same p-i-n carrier material is prepared at its ion-free dark equilibrium.
At `t=0`, a uniform positive population `1e22 m-3` is present in the absorber,
with diffusion coefficient `1e-12 m2/s` and capacity `1e26 m-3`. The dual-ion
variant additionally has a negative population `0.7e22 m-3`, diffusivity
`0.5e-12 m2/s` and capacity `1e26 m-3`. Each population starts equal to its
fixed compensating background, so the initial Poisson solution is identical.
Uniform ions in the initial built-in field are a declared prepared state,
not a claim of full ionic equilibrium. These accelerated diffusivities are
synthetic numerical inputs, not measured material migration coefficients.

The fixed illumination switches on at `t=0`. Voltage rises linearly from
0 to 0.6 V over `1e-4 s`, then holds until `2e-3 s`. The ramp endpoint is an
explicit integration boundary. Both species use the physical diffusion-only
steric model with separate site capacities. The result retains this whole
history, including the initial response and final redistribution.

The initial 88/176/352 balanced-tanh attempt preserves charge and resolves
the later ionic history, but fails the carrier profile gate immediately
after the ideal light step. An additional 704-grid run reduces the same
error, confirming unresolved early diffusion rather than time-converged
ionic failure. These earlier results remain incomplete evidence.

The resolved mobile sequence is 176/352/704 total intervals, distributed
in the ratio 3:5:3 across the three fixed physical layers. The absorber uses
a tanh parameter of 3. Each outer layer uses uniform interior intervals
between two short end intervals that match the absorber's interface spacing.
The grid retains exact layer boundaries and absorber volume, while adding
resolution along the earliest minority-carrier diffusion fronts. Every
previous time sample, physical parameter and accuracy bound remains present.
Internal step ceilings are 20/10/5 microseconds. Relative tolerances are
`1e-4`, `1e-5`, `1e-6`, with the same factors applied to the predeclared
componentwise absolute tolerance reference. One-dimensional state profiles
are compared at 17 fixed interior positions per layer. Carriers use the SG
reconstruction; mobile-ion profiles use linear concentration reconstruction
inside the absorber. The finest pair must satisfy 1 mV potential error,
1 percent carrier log-density/ionic reference-density error and 0.5 percent
of `q*Phi` for the current waveform, in addition to `1e-10` per-species
inventory drift. The instantaneous all-face total current uses the RHS
charge rate and differentiated Poisson boundary voltage at the same time.
