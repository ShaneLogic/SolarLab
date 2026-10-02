# One-Dimensional Mechanism R1 Admittance V1

This contract covers the pure numerical module
`perovskite_sim/experiments/one_dimensional_mechanism_r1_admittance.py`.
It implements the reconstruction in the archived
`plans/Specs/OneDimensionalMechanismR1StudySpecV1.md`, sections 6, 7.2, 7.3
and 10.2. It does not amend the study's acceptance thresholds or certify a
device, source revision, reference state, linear range or frequency band.

## Inputs and sign convention

`reconstruct_admittance` accepts named SI quantities:

| Input | Meaning | Units |
|---|---|---|
| `frequency_Hz` | Nonnegative strictly increasing frequencies | Hz |
| `time_s` | Strictly increasing positive regular-current sample times | s |
| `regular_current_A_m2` | Signed voltage-conjugate current, excluding the mathematical impulse | A/m² |
| `step_amplitude_V` | Signed nonzero voltage step | V |
| `baseline_current_A_m2` | Independent zero-bias DC current | A/m² |
| `dc_conductance_S_m2` | Independent differential DC conductance, `G0` | S/m² |
| `impulse_charge_C_m2` | Signed charge in the initial ideal-step impulse | C/m² |

The current is `j = junction_polarity * J_x`, with the study's coordinate and
voltage convention. The phasor convention is `exp(+i omega t)`, with
`omega = 2 pi frequency_Hz`. No absolute value is applied to current, charge,
conductance or reconstructed admittance. A positive ideal capacitance has
positive imaginary admittance. Negative imaginary components are not clipped.
The caller excludes `0-`, `0+` state records and the impulse from the positive
regular-current sample vector. The impulse is supplied separately.

The input shape must identify one current per time; implicit broadcasting of
current samples is rejected. Nonfinite inputs, ambiguous scalar types such as
booleans or strings, duplicate coordinates and nonfinite derived arithmetic
raise. Input arrays are not modified; result arrays are copied and read-only.

The pure module accepts negative steps for sign tests. The R1 experiment's
allowed positive amplitude ladder remains the responsibility of its study
entry point. The module cannot recognize an otherwise plausible number
entered in the wrong units; independent analytic tests pin the conversion.

## Reconstruction

For amplitude `a`, the sampled normalized regular residual is

```text
r(t_i) = (j_reg(t_i) - j_dc(0)) / a - G0                 [S/m²]
C_imp = Q_imp / a                                      [F/m²]
I_window(omega) = integral(t_first..T, r_lin(t) exp(-i omega t) dt)
                                                      [F/m²]
Y_window(omega) = G0 + i omega C_imp + i omega I_window  [S/m²]
```

`r_lin` is linear in physical time between each adjacent pair of supplied
samples, including on a logarithmically spaced output grid. Each interval is
mapped to `[0,1]`; SciPy `integrate.quad` with `cos` and `sin` weights computes
the oscillatory real and imaginary integrals. The phase at the interval start
is retained. Summation uses `math.fsum` for the two components. The numerical
method follows the installed SciPy API documentation for finite weighted
integration (`qawoe`); SciPy describes `abserr` as an error estimate, not a
rigorous enclosure. Full integration diagnostics are inspected, and any
nonconvergence message or nonfinite/negative error estimate raises
`R1AdmittanceIntegrationError`.

The requested absolute quadrature tolerance is in **F/m²**. It is allocated
across the two components and all time intervals; the relative tolerance
also applies. The result records both requested tolerances and the actual
summed quadrature error estimate, multiplied by `abs(omega)` into S/m².

Only `[t_first,T]` is integrated. The point estimate contains neither an
invented early interval nor an extrapolated infinite tail. `G0` is never
estimated from the last sample. At zero frequency the returned point estimate
is the supplied `G0` exactly. An unverified or unconverged `G0` cannot become
verified through this operation.

## Error accounting

`R1AdmittanceErrors` supplies seven independent sources. Every field defaults
to `None`, which means **unknown**, not zero. Every unknown source is reported
in `unknown_error_sources` and contributes infinity at every requested
frequency, including zero. A caller must explicitly justify any zero bound.

| Result component | Caller input and propagation into S/m² |
|---|---|
| `early_omission` | `abs(omega)` times a bound on `integral(0..t_first, abs(r) dt)` in F/m² |
| `tail_omission` | `abs(omega)` times a bound on `integral(T..infinity, abs(r) dt)` in F/m² |
| `interpolation` | `abs(omega)` times a bound on `integral(t_first..T, abs(r-r_lin) dt)` in F/m² |
| `sample_current` | `abs(omega)/abs(a)` times the trapezoidal integral of the nonnegative nodal current error envelope in A/m² |
| `baseline_current` | `abs(omega)*(T-t_first)/abs(a)` times the baseline current error in A/m² |
| `dc_conductance` | `(1+abs(omega)*(T-t_first))` times the `G0` error in S/m² |
| `impulse_charge` | `abs(omega)/abs(a)` times impulse charge error in C/m² |
| `quadrature_estimate` | `abs(omega)` times the sum of absolute quadrature estimates in F/m² |
| `floating_point_estimate` | Explicit engineering estimate for subtraction/normalization, phase arguments, complex products and summation |

The DC bound includes both the separate `G0` term and the occurrence of `-G0`
inside the finite-window residual; the two errors are conservatively combined.
Baseline and nodal errors are kept separate. The nodal envelope describes
error in the linear interpolant caused by erroneous node values. Interpolation
uncertainty additionally describes the true curve between exact nodes. These
sources must not be substituted for one another or silently counted as zero.

`linear_interpolation_integral_bound` accepts an independently justified bound
`M_i >= abs(r''(t))` throughout each time interval. It returns
`sum(M_i * h_i**3 / 12)` in F/m², from the linear interpolation remainder.
It never infers curvature from samples. A scalar supplies the same bound on
all intervals; an array must have one value per interval. The helper makes
analytic checks possible and does not establish a curvature bound for a
device trajectory.

The floating-point estimate scales with the absolute normalization operands,
the integrated operand scale and `omega*T`. Large absolute phases can have
large uncertainty even if oscillatory cancellation makes the computed result
small. This is not interval arithmetic or a proof of a rounding-error bound.

`total_error_estimate_S_m2` sums all nine components without cancellation.
The components use absolute complex-modulus budgets and can conservatively
be compared with either real- or imaginary-component tolerance. Because the
total includes estimated quadrature and floating-point errors, even complete
caller bounds produce a **conditional numerical error estimate**, not a strict
mathematical certificate. Time, voltage and frequency coordinate uncertainties,
finite-amplitude nonlinearity and the validity of the steady preparation are
outside this total; they need separate study checks.

## Analytic checks and limitations

The independent unit suite uses a pure resistor, a pure capacitor and

```text
Y = G0 + i omega Cinf + i omega DeltaC / (1 + i omega tau)
r(t) = (DeltaC/tau) exp(-t/tau).
```

It checks charge normalization, Fourier sign, units, low/high frequency limits,
positive dissipation for the passive analytic example, missing early/tail
intervals, finite-window agreement, error propagation, tolerance tightening
and malformed/nonfinite inputs. Deliberate wrong-sign, missing-impulse,
missing-DC, voltage-unit and Fourier-kernel faults fail the analytic oracles.
The capacitor check separately distinguishes Hz from rad/s. Quadrature failure
injection must raise rather than produce a successful numerical result.

Passing these tests establishes the pure reconstruction's tested behavior. It
does not establish the same-model DC derivative, device passivity, amplitude
linearity, spatial/time/nonlinear convergence, a closed observation window or
agreement with the coupled AC solver. The result's scope is permanently
`finite_window_reconstruction_only`; there is no `passed` or valid-band field.

Before the R1 study labels any frequency as dual-domain consistent, sections
7.2 and 10 still require all amplitude, independent numerical-axis, early-time,
`T` versus `10T`, same-model DC-tail, quadrature-tightening and AC comparisons.
The reconstructed real and imaginary errors and the reconstruction's own
error budget each need the separate `1e-8 S/m² + 1%` component criterion.
Two finite windows do not prove that a hidden slower mode is absent. No
single-exponential extrapolation or claimed infinite-tail bound is derived
from agreement between them.
