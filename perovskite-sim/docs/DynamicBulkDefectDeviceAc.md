# Dynamic bulk-defect device AC

## Status and scope

`run_bulk_defect_device_impedance` is the research-only D5-E2a adapter for
one-dimensional, monovalent explicit bulk defects. Its fixed capability label
is:

```text
research_bulk_dynamic_defect_device_ac_only
```

It supports the canonical single-level, energy-distributed, and spatially
graded bulk-defect documents already accepted by the certified QF/DC path. It
does not yet support two-sided dynamic interface defects, mobile ions, the
combined defect/ion state space, the production impedance API, or the
frontend. Those remain D5-E2b, D5-E2c, and D5-E3 work.

## Dynamic state

For every selected physical node, defect source, and energy quadrature node,
the compiled layout stores one occupancy `f` and the rates

```text
C_n = N_t c_n [(1-f)n - f n1]
C_p = N_t c_p [f p - (1-f)p1]
N_t df/dt = C_n - C_p
lambda = c_n(n+n1) + c_p(p+p1).
```

The numerical coordinate is a logit increment about the residual-certified DC
occupancy. This keeps `0 < f < 1` without clipping. The evaluator uses an exact
increment expansion about the QSS reference and retains the `dn*df` and
`dp*df` terms. It is therefore a cancellation-safe rewrite of the nonlinear
kinetics, not a linear constitutive approximation.

For acceptor and donor transitions, the differential electrostatic charge has
the same sign:

```text
delta rho_t = -q N_t delta f.
```

Neutral transitions contribute capture and storage but no dynamic charge.

## Device coupling

The device state is

```text
[interior electron QF increments,
 interior hole QF increments,
 explicit trap logit increments].
```

The small-signal engine linearizes the following coupled device quantities:

- electron and hole carrier densities as storage rows;
- occupied trap populations `N_t f` as independent storage rows;
- electron capture `C_n` and hole capture `C_p` in their respective
  continuity equations;
- explicit trap charge in the eliminated Poisson solve;
- electron conduction, hole conduction, and displacement current at every
  device face.

The explicit dynamic capture terms replace the legacy lifetime SRH sink at
every dynamic-defect node. This prevents the effective-lifetime and explicit
defect models from being counted twice. At the QSS reference, the dynamic
device residual is required to reproduce the existing certified QF/DC
operator.

The solved frequency-domain equation remains the shared engine contract

```text
(i*omega*M - A) u_hat = (b - i*omega*m_V) V_hat,
```

with the `exp(+i*omega*t)` phasor convention and responses normalized per
applied volt.

## Current and storage evidence

The returned all-face admittance is decomposed as

```text
Y_total(x_face) = Y_n(x_face) + Y_p(x_face) + Y_displacement(x_face).
```

Carrier storage, charged-trap storage, occupied trap population, occupancy,
and state-resolved electron/hole capture responses are returned separately.
Energy quadrature nodes remain visible in the result, but the local trap
balance certificate first aggregates them by physical defect source and device
control volume. Quadrature nodes are numerical integration points; the
source/control-volume population is the conserved physical quantity.

## Certificate gates

The D5-E2a result is certified only when all of the following pass:

- the supplied or internally solved QF/DC operating point is certified;
- the DC defect-model identity and distributed quadrature order match the AC
  material, and the state passes fresh normalized-residual, electron/hole
  continuity, face-current-spread, and Poisson checks on the current device
  operator;
- the dynamic QSS embedding normalized error is at most `1e-10`;
- the grouped local trap balance error is at most `1e-4`;
- the all-face admittance spread is at most `5e-4`;
- the componentwise linear-solve backward error is at most `1e-10`;
- the three-level finite-difference refinement change is at most `2e-3`;
- the low-frequency result approaches the QSS device reference within `3e-2`;
- the high-frequency result approaches the frozen-occupancy device reference
  within `3e-2`;
- the requested frequencies bracket every compiled relaxation corner with the
  configured branch margin and sampling-gap limit.

`require_certificate=True` fails closed and attaches the finite partial result
to `BulkDefectDeviceACCertificationError`. A missing explicit-defect model,
incompatible model identity, incomplete interior-node coverage, mismatched or
stale DC state, or unsupported coupled physics fails before a certified result
can be returned. A copied `certified=True` flag is not accepted as a substitute
for re-evaluation on the current operator.

## Verification boundary

The D5-E2a checkpoint exercises real device solves for single-level,
energy-distributed, and spatially graded defects. It also covers an incomplete
frequency window and a no-defect capability rejection. The implementation
preserves the pre-existing QF/DC path and its results when the new research
adapter is not called.

These tests establish internal bulk-defect device-AC closure. They are not an
external SCAPS validation, an ion/defect combined certificate, or a production
API claim.
