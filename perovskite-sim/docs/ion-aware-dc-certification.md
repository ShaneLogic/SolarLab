# 1D ion-aware DC certification closure

Implementation and evidence date: 2026-08-20

This checkpoint closes step 1 of the Phase 2 ion-aware impedance roadmap: a
finite-time mobile-ion state is no longer treated as DC merely because the
integrator returned successfully. The new opt-in lane promotes a state only
after independent carrier, ion, current, conservation, positivity and
protocol gates pass at consecutive endpoints.

This is an internal numerical DC certificate. It is not an IonMonger external
validation and it does not certify the legacy contact deck thermodynamically.

## Problem closed

The historical transient impedance preconditioner used a fixed 1 ms state. On
the registered IonMonger stack at `V_dc = 0.9 V`, N30 and one-sun generation,
the new independent evaluator finds at 1 ms:

- ion area residual `9.656e-4 A/m2`;
- maximum per-species ionic face current `4.827e-4 A/m2`;
- a non-positive terminal hole-density entry.

The registered DC limits are `1e-6 A/m2` for both ionic quantities, and every
active terminal density must be above zero. The 1 ms state is therefore
explicitly rejected as DC. At N30, the 64 s and 128 s endpoints both pass;
the 128 s endpoint has carrier area residual `4.590e-5 A/m2`, ion area
residual `3.110e-16 A/m2`, ionic face current `1.164e-16 A/m2`, all-face DC
current spread `3.123e-5 A/m2`, and ion inventory drift `2.716e-13`.

The endpoint sequence is not assumed monotone. For example, the N30 10 s
state still fails carrier/current gates. Requiring two consecutive accepted
endpoints prevents one accidental low-residual sample from ending the
preparation early.

## Public research API

```python
from perovskite_sim.experiments.ion_aware_dc import (
    build_ion_aware_dc_protocol,
    solve_ion_aware_dc,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid

x = build_electrical_grid(stack, 90)
protocol = build_ion_aware_dc_protocol(
    stack,
    V_dc=0.9,
    illuminated=True,
)
result = solve_ion_aware_dc(x, stack, protocol)

assert result.numerically_certified
assert result.steps[-2].accepted_for_closure
assert result.steps[-1].accepted_for_closure
```

The API is explicit and opt-in. Existing J-V and impedance defaults are not
rerouted by this checkpoint.

`IonAwareDCProtocol` records the fixed bias, effective solver temperature,
dark/light source, initial-state source, blocking ion boundary, ordered
settle-endpoint ladder, consecutive-pass count and every physical acceptance
threshold. It has strict JSON round-trip and canonical SHA-256 behavior.
When a caller supplies `y0`, the protocol must carry the canonical packed-state
SHA-256 and the solver verifies it before integration. Thus two different
user states cannot share one claimed history hash.

The registered execution protocol separately records numerical controls:
grid source, componentwise `atol` policy and refinement factor, `rtol`,
`max_nfev`, and the ordered `Radau -> BDF` recovery ladder. Physical and
numerical provenance are not conflated.

## Independent state certificate

For solver dual-cell widths `w_i`, the endpoint evaluator recomputes the full
method-of-lines RHS and reports separate area residuals:

```text
r_n = q sum_i w_i |dn_i/dt|
r_p = q sum_i w_i |dp_i/dt|
r_P+ = q sum_i w_i |dP_i+/dt|
r_P- = q sum_i w_i |dP_i-/dt|
```

It also evaluates:

- positive and negative ionic face currents separately, so dual-species
  cancellation cannot manufacture a DC pass;
- conduction plus ionic all-face current and its peak-to-peak spread;
- positive and negative ion inventories with the exact discrete dual-cell
  invariant, plus terminal ion centroid;
- terminal electron, hole and active-ion minima;
- shared or separate ion-site occupancy;
- non-finite RHS/trial counts and exact SRH denominator diagnostics;
- the contact thermodynamic assessment as an independent evidence axis.

Every accepted or failed method attempt retains its method, message,
diagnostics and `nfev/njev/nlu`. Exhausting the method ladder raises an error
that still carries the target endpoint and all attempt evidence. Negative
implicit trial densities are observed rather than clipped; non-finite trials,
non-positive terminal active densities and failed independent endpoint gates
cannot pass.

Dynamic interface-state blocks fail closed in v1 because their electrostatic
charge is not yet coupled self-consistently. Ion-free stacks are also rejected
because they belong to the existing QF DC/frequency lanes.

## Registered refinement evidence

Both lanes use tolerance factors `1`, `0.1`, `0.01`, the same physical and
quality thresholds, and content-addressed artifacts under the ignored local
`outputs/numerical-refinement/` tree.

| Lane | Grid ladder | Status | Completed | Run ID | Certificate SHA-256 |
|---|---|---|---:|---|---|
| `ionmonger-ion-aware-dc-v1` | N30/60/90 | `partial` | 9/9 | `314c4b8cdaecb31b64180a204cf0bdc541f2779fa07d5b32d4c4dc59d90665f1` | `13716fd0ce4d2a588819f450c25061706489ffa736bedb654556d7c876eecfec` |
| `ionmonger-ion-aware-dc-resolved-v2` | N60/90/120 | `certified` | 9/9 | `ba1e7b8dbcb0b16695d03c4626555bc98458ca851fc4aad9611425f55663fe1f` | `fcc84e55b8e6138e9b52c4abb49260f623e747a08b9b2a64af215c63b8ea51e9` |

The v1 matrix is complete and every per-cell quality gate passes. Its sole
failure is the N60 to N90 maximum-site-occupancy relative change:
`0.0154895 > 0.01`. The current-density grid change is `4.483e-4` relative
and the ion-centroid grid change is `1.812e-4` absolute, both passing.

The versioned resolved-v2 lane preserves the v1 failure and does not tune its
threshold. On N90 to N120 it records:

| Observable | Comparison | Observed | Limit |
|---|---|---:|---:|
| DC current density | relative | `1.059e-4` | `5e-3` |
| maximum site occupancy | relative | `7.046e-3` | `1e-2` |
| positive-ion centroid | absolute | `7.747e-5` | `2e-3` |

Across all nine resolved-v2 cells, the worst carrier area residual is
`9.118e-4 A/m2`, worst DC current spread is `2.183e-4 A/m2`, worst ion area
residual is `1.029e-14 A/m2`, worst per-species ionic face current is
`5.080e-15 A/m2`, and worst inventory drift is `2.193e-13`. All non-finite
counts are zero and all terminal positivity, occupancy, consecutive-pass and
diagnostic-completeness gates pass.

Both matrices bind commit `b55d1e5d03fe573397aa9cfcfe65f5caa68dc361`,
source fingerprint
`58db6ba3c01d37880043080c3601612b908ef1eb64a5fbc82510e3ce11cfff86`,
execution protocol SHA-256
`d0687b54df6291cf2ad17efc157f0921a1c99fc2f1135069fe6233794f831a07`,
macOS 26.6.2, Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. OpenBLAS, OMP and
VECLIB thread counts are fixed to one; MKL is unset.

## Verification

Verification after the final source-bound matrix runs produced:

- `86 passed, 4 deselected` for the focused DC, J-V current, ion migration,
  refinement executor/certificate and conservation suite;
- `1999 passed, 2 skipped, 263 deselected` for the repository default Python
  suite in one run;
- `29` configs, `8` numerical refinement lanes and all frozen P0 checks passing
  in `scripts/verify_reproducibility.py --json`;
- isolated subprocess tests proving both repository CLIs prefer their own
  checkout over a foreign `PYTHONPATH` package;
- scoped Ruff, critical J-V Ruff, `compileall` and `git diff --check` passing.

The full suite emitted only existing NumPy `trapz` deprecation warnings. No
frontend files changed in this checkpoint.

## Evidence boundary and next work

All nine cells report `numerically_certified=true`, but the IonMonger deck
lacks endpoint effective-density-of-states data. Its contact assessment is
therefore `compatible_unverified`, every
`thermodynamically_certified` flag is false, and the combined physical
certificate remains false. This distinction is intentional.

The result closes only the DC-state prerequisite for ion-aware impedance.
The subsequent opt-in reference engine now implements the mobile-ion storage
matrix, state/voltage central-finite-difference linearization and decomposed
frequency-domain solve. It is `INTERNAL_TESTED_REFERENCE`, not a registered
grid/frequency certificate. Frequency coverage assessment, structured
Jacobian comparison and transient lock-in remain open. See
[ion-aware-impedance-reference-engine.md](ion-aware-impedance-reference-engine.md).
