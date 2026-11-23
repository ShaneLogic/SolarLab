# Two-sided interface topology: Stage 4.2

## Status

Stage 4.2 connects the tested local heterointerface control element in
`perovskite_sim.physics.two_sided_interface` to the full quasi-Fermi solver,
CBO scan API, main CLI, and sensitivity CLI. It is selected only with
`interface_topology="two_sided_trace"`; the existing `deduplicated_qss`
topology remains the default.

The new path has an internally grid-converged Jsc development envelope for
the declared protocol. It has not passed independent SCAPS validation, so the
result must not be described as an externally validated physical threshold.

The two-sided grid removes each shared material-boundary node. Strict left and
right bulk nodes bracket the physical interface, their exact distances define
the two SG half-fluxes, and the interface face uses the exact dielectric series
capacitance. Carrier control-volume widths are corrected without changing the
total device volume.

## Local model

For one interface, the local unknowns are the electrostatic traces
`phi_L`, `phi_R` and carrier traces `(n_L, p_L, n_R, p_R)`.

The electrostatic constraints are

```text
phi_R - phi_L = Delta_phi_interface

eps_L (phi_L - phi_bulk,L) / h_L
+ eps_R (phi_R - phi_bulk,R) / h_R = sigma_interface
```

Thus `phi_L = phi_R` when no interface dipole is declared; separate traces do
not imply an arbitrary discontinuity. Fixed sheet charge and an explicit
dipole are supported independently.

Each carrier trace receives a one-material SG half-flux over its exact
distance `h_L` or `h_R`. Cross-interface exchange uses reciprocal 3D
Fermi-Dirac Richardson supplies,

```text
Phi_TE = transmission * A* T^2 / q
         * [F_1(eta_L - barrier_L/kT)
            - F_1(eta_R - barrier_R/kT)].
```

A shared trap occupancy supplies signed, side-resolved electron and hole
capture. The four zero-volume balances are solved in log-density variables.
Their analytic Jacobian differentiates the same tabulated `F_1/2`
constitutive law used to invert density to reduced chemical potential.

## Constitutive acceptance gates

The local element must retain all of the following before global dispatch:

1. Potential-jump and Gauss-law residuals close to floating-point precision.
2. Common-quasi-Fermi heterojunction states have zero net transport.
3. Cross-interface exchange is pairwise conservative.
4. Shared-trap total electron capture equals total hole capture.
5. The analytic log-density Jacobian agrees with central finite differences.
6. A homojunction mirror reverses current and swaps left/right trace states.
7. Exact left/right distances act only on their own SG half-flux.
8. A deduplicated boundary node is identified but never used as a bulk
   reservoir, because its interface distance is zero.

## Integration status

1. Complete: the interface state is statically condensed and replaces the
   ordinary carrier face. No parallel SG path remains.
2. Complete for the current finite-difference outer Newton: every perturbed
   residual evaluation re-eliminates the local state, so the implicit local
   response is included. The analytic local Jacobian is tested. An explicitly
   assembled global implicit-function Jacobian remains a performance and
   robustness optimization, not a claim made by this stage.
3. Complete: terminal contacts are unchanged, and the historical solver path
   is retained unless the new topology is explicitly selected.
4. Complete: local constitutive tests, full-device dark equilibrium, Poisson,
   electron/hole continuity, total-current spread, warm-start rejection, and
   illuminated CBO scan certificates pass.
5. Complete for Jsc under the protocol below: nominal N_grid values 40, 50,
   and 60 have actual interval counts 37, 46, and 58 after removing two shared
   interface nodes. Their union envelope is 0.382421875-0.389453125 eV,
   7.03125 meV wide, below the 10 meV gate. Successive midpoint shifts are
   4.296875 and 2.34375 meV, with contraction ratio 0.545.
6. Not complete: the existing SCAPS data lack the required provenance and
   dense onset coverage. At N_grid=60, the sparse comparison also has maximum
   normalized Jsc error 0.38849, above the 0.05 gate.

## Legacy de-spike policy

`het_recomb_despike` corrects bulk recombination at the old shared boundary
node. That node does not exist in `two_sided_trace`, so applying the correction
to either adjacent bulk node would change its meaning. The API therefore
rejects a nonzero value. The CLI requires the explicit
`--disable-legacy-heterojunction-despike` flag and records the input value,
effective value, and explicit-disable decision in schema 1.4 output.

## Reproducible Jsc grid gate

```bash
python scripts/run_interface_cbo_scan.py \
  --config configs/scaps_mirror_v2.yaml \
  --out outputs/interface-cbo/scan-two-sided-fd-grid-40-50-60.json \
  --delta-min 0 --delta-max 0.5 --delta-step 0.25 \
  --grid-ladder 40 50 60 \
  --short-circuit-only \
  --interface-topology two_sided_trace \
  --interface-transport-model fermi_dirac_richardson \
  --disable-legacy-heterojunction-despike
```

This command certifies internal numerical convergence for Jsc only. FF and PCE
still require complete voltage sweeps and voltage-grid convergence. External
SCAPS validation still requires a dense direct export, source deck, populated
parameter manifest, and content hashes.
