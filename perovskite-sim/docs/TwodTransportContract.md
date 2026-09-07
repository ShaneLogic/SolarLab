# 2D Transport Contract

This note preserves the transport constraints from the retired Stage A/B
designs against the current implementation. It is a model-scope and test
reference, not a new numerical certificate or external validation result.

## Geometry And Scope

The 2D implementation uses a rectilinear tensor grid: `x` is lateral, `y`
follows the electrical layer stack, and node arrays have shape `(Ny, Nx)`.
The default carrier state is `(n, p)`, with each block flattened in C order;
ionic densities remain fixed Poisson background fields on that path.
Poisson and Scharfetter-Gummel transport have separate 2D implementations,
while material construction and constitutive primitives reuse the 1D code.
Incident optical generation is extruded from the 1D layer calculation.
This does not implement textured-device Maxwell optics, cell-scale contacts,
module wiring, arbitrary 3D geometry, or externally validated tandem-2D physics.

Empty microstructures support the historical periodic-x and Neumann-x paths.
A non-empty finite-width vertical grain boundary requires Neumann-x: its
physical width is mapped by control-volume overlap, and the SRH rates are
mixed as `R_bulk + f * (R_GB - R_bulk)`. Averaging lifetimes first is not an
equivalent operation. The thin-band estimate `S_GB ~ width / tau_GB` is only
a dimensional conversion, not a microscopic surface-defect closure. The
[grain-boundary area contract](TwodGrainBoundaryAreaClosure.md) defines
the geometry, rate mixing, rejection rules, and evidence limits.

Separate opt-in research paths cover
[one mobile positive ion](TwodMobileIonTransient.md),
[complete terminal current](TwodMobileIonCurrent.md), and
[two-sided cross-node interface SRH](TwodTwoSidedInterfaceSrh.md).
Their [explicit J-V protocol](TwodJvExecutionProtocol.md) and
[combined numerical certificate](TwodCombinedNumericalCertificate.md)
have their own restrictions. In particular, that combined certificate does
not cover field mobility, photon recycling, periodic-x, dual ions, or dynamic
interface occupancy. It does not establish arbitrary composability of all
available 2D features.

## Uniform Device Parity

The historical carrier-only parity case in
[`test_twod_validation.py`](../tests/regression/test_twod_validation.py)
uses a laterally uniform device with no grain boundaries, frozen ions,
matched transport-axis grids, periodic-x, and relaxed carrier states. These
are the six implemented checks in `test_twod_uniform_matches_1d_within_tolerance`:

| Quantity | Implemented condition |
|---|---|
| Open-circuit voltage | `abs(Voc_2D - Voc_1D) <= 1e-4 V` |
| Short-circuit current | `abs(Jsc_2D - Jsc_1D) / abs(Jsc_1D) <= 5e-4` |
| Fill factor | `abs(FF_2D - FF_1D) <= 1e-3` |
| Lateral electron-density variation | `max(abs(n - n[:, [0]])) / max(abs(n[:, 0])) <= 1e-9` |
| Interior carrier-current divergence | `max(abs(div(J_n + J_p))) <= 100 * abs(Jsc_2D) / min(diff(y))` |
| Contact potentials | `phi[0, :] = 0` and `phi[-1, :] = V_bi - V_snapshot`, using `np.allclose` with `atol=1e-10` and `1e-6`, respectively, and its default relative tolerance |

The divergence check is a loose scale-dependent bound. The last check
checks contact potentials, not a Poisson residual norm. Neither should be
reported as the stronger residual certification proposed by the early
design. Field-mobility and reabsorption parity cases in the same file have
their own tolerances; the table above is not a universal 2D acceptance gate.

The old `configs/twod/` and related example presets were removed on
2026-09-05. These historical full-device tests require their original inputs
and cannot run unchanged in this checkout. New studies must specify new
inputs, protocols, and acceptance criteria before reporting validation;
substituting either current research preset does not preserve old evidence.

## Field Dependent Mobility

[`recompute_d_eff_2d`](../perovskite_sim/twod/field_mobility_2d.py)
uses the field component normal to each face:

```text
E_x_face = -(phi[:, 1:] - phi[:, :-1]) / diff(x)
E_y_face = -(phi[1:, :] - phi[:-1, :]) / diff(y)
mu_base_face = harmonic_mean(D_node) / V_T
D_eff_face = apply_field_mobility(mu_base_face, abs(E_normal), ...) * V_T
```

Electron and hole transport use the existing Caughey-Thomas and Poole-Frenkel
primitive separately. Node diffusion coefficients use harmonic face means;
`v_sat`, `ct_beta`, and `pf_gamma` use arithmetic face means. A harmonic mean
for the latter would spuriously turn off a face when one adjacent value is
zero. For periodic-x, the wrap face uses columns `0` and `-1`, with
`dx_wrap = 0.5 * (diff(x)[0] + diff(x)[-1])` and matching wrap parameters.

The helper is called from the RHS only when `has_field_mobility` is true.
Otherwise the existing constant-diffusion face path is used. Activation
requires the simulation tier to permit field mobility and a layer with a
positive saturation velocity or Poole-Frenkel coefficient. Shapes and the
zero-parameter Einstein limit are covered by
[`test_field_mobility_2d.py`](../tests/unit/twod/test_field_mobility_2d.py).

Using total `sqrt(E_x**2 + E_y**2)` at each face would require cross-axis
interpolation and separate validation; it is not the current closure.
Per-grain mobility patterning and new temperature-coupled mobility laws
beyond the existing temperature scaling are also outside this contract.

## Radiative Reabsorption

[`recompute_g_with_rad_2d`](../perovskite_sim/twod/radiative_reabsorption_2d.py)
uses net emission, including the equilibrium subtraction absent from the
early design:

```text
R_tot = integral_absorber integral_x B_rad * (n * p - ni_sq) dx dy
A_abs = physical_absorber_thickness * lateral_length
G_rad = R_tot * (1 - P_esc) / A_abs
G_out[absorber_rows, :] = G_optical[absorber_rows, :] + G_rad
```

The implementation applies trapezoidal integration over `y` first and then
`x`. `R_tot` has units `m^-1 s^-1`; `G_rad` has units `m^-3 s^-1` and is
uniform over the absorber area. A nonpositive integral, nonpositive area,
`P_esc >= 1`, or fewer than two nodes along either integration axis produces
no added source for that absorber. At `n * p = ni_sq`, recycled generation
vanishes. The helper returns a new array and never mutates cached
`G_optical`; the disabled RHS path uses the original generation cache.

Activation requires both photon recycling and radiative reabsorption to be
enabled by the tier, plus the absorber escape data from the 1D material
builder. The FULL tier permits this composition; FAST and LEGACY do not
enable this per-RHS feedback. Input shapes, equilibrium cancellation,
nonmutation, and the laterally uniform 1D limit are covered by
[`test_radiative_reabsorption_2d.py`](../tests/unit/twod/test_radiative_reabsorption_2d.py).

On an illuminated voltage-step failure, the
[J-V driver](../perovskite_sim/twod/experiments/jv_sweep_2d.py) can retry with
the same net-emission source frozen at the entry state of that step. This is
a step-local lagged approximation; it does not certify fully self-consistent
feedback throughout the retry. A failed retry continues through the driver's
bounded time-subdivision recovery and ultimately raises on exhaustion.

Optical-profile-weighted redistribution, lateral photon transport, and
validated per-grain optical heterogeneity are not supplied by this uniform
area closure. Existing synthetic or laterally uniform tests do not establish
those capabilities or agreement with measured grain-size trends.
