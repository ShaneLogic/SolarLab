# Composition-graded CIGS optics

## Scope

P4.4 adds an explicit, research-only composition-dependent optical model for
one graded CIGS absorber. It does not change the historical CIGS presets or
reinterpret their scalar `alpha`. The new path is active only when all three
conditions hold:

```yaml
device:
  band_grading: true
  graded_optics: true

layers:
  - role: absorber
    Eg_back: 1.40
    cigs_graded_optics:
      model: minoura_2015
      ggi_front: 0.225
      ggi_back: 0.600
      cgi: 0.900
      slices: 25
      kk_quadrature_order: 192
```

The block is absorber-only and cannot coexist with `optical_material` or
`n_optical` on that layer. Unknown fields, GGI outside `[0,1]`, CGI outside
`[0.75,1]`, unsupported models, non-integer slice/quadrature controls, a
missing electrical back endpoint, or activation in `legacy` mode fail closed.
With `graded_optics` absent or false, a carried block is inert and historical
semantic hashes and numerical paths remain unchanged.

## Constitutive model

The production `n(lambda), k(lambda)` path follows the composition model of
[Minoura et al. (2015)](https://doi.org/10.1063/1.4921300). The six published
reference spectra are represented by their Table II Tauc-Lorentz oscillator
parameters. Ga composition shifts the critical-point energies; Cu composition
interpolates the CGI anchor spectra. The implementation first constructs the
shifted/interpolated imaginary dielectric function and then evaluates

```text
epsilon_1(E) = epsilon_inf
             + (2/pi) P integral [E' epsilon_2(E')/(E'^2-E^2)] dE'.

n = sqrt[(|epsilon| + epsilon_1)/2]
k = sqrt[(|epsilon| - epsilon_1)/2]
```

The principal value uses pole subtraction plus a controlled Gauss-Legendre
quadrature. The public wavelength domain is the source model's 0.7-6.5 eV
range. Returning `n,k` from one Kramers-Kronig-related dielectric function is
the causality contract; independently interpolating `n` and `k` is not used.

The near-edge absorption expression from
[Carron et al. (2018)](https://doi.org/10.1080/14686996.2018.1458579) is an
independent benchmark, not the source of the production `k`. Its GGI/CGI
band-gap, Urbach, parabolic, and high-energy branches are implemented through
2.5 eV. The public CGI domain is conservatively restricted to `[0.75,1]`,
where the two source models overlap their declared device regime.

## Shared composition coordinate

At each CIGS optical slice centre, `physics/optical_stack.py` calls the same
`grading_coordinate` used for electrical `Eg(x)` and `chi(x)`, including
profile, direction, and characteristic length. It then maps that coordinate
to

```text
GGI(x) = [1-y(x)] GGI_front + y(x) GGI_back.
```

Here `GGI_front` and `GGI_back` name the `y=0` and `y=1` material endpoints.
With `grading_direction: back_to_front`, the shared coordinate reverses their
physical-face placement for optics and electrical bands together.

The adapter expands only the physical absorber into TMM slices and retains a
mapping from each physical layer to its expanded optical slice range. The
mapping is consumed by build-once MoL generation, photon recycling, EQE, EL,
and tandem optics. Non-CIGS layers retain their existing tabulated or scalar
fallback. A uniform composition produces the same reflectance for one slice
and many adjacent identical slices to floating-point roundoff.

## Numerical and physical gates

The `cigs-graded-optics-v1` lane freezes a 3x3 matrix:

- CIGS optical slices: `8/16/32`;
- Kramers-Kronig orders: `96/192/384`, encoded as inverse factors
  `1/0.5/0.25`;
- fixed 32-interval electrical mesh per physical layer and 100 AM1.5G
  wavelengths from 300 to 1000 nm;
- observables: absorbed photon flux, generation centroid, mean reflectance,
  and the normalized generation profile;
- every-cell gates: finite causal `n,k`, exact opt-in topology, default-off
  inertness, non-negative photon budget, bounded reflectance, uniform limit,
  optical/electrical endpoint-gap consistency, and a three-composition Carron
  comparison.

The terminal limits are `0.1%` relative for absorbed photon flux, `0.1%`
absolute for centroid and mean reflectance, and `0.5%` absolute `L_inf` for
the normalized generation profile. Carron comparisons use front, midpoint,
and back GGI, with 151 energy points per composition beginning at least
0.15 eV above the Carron gap. The maximum composition-wise median relative
difference must be at most `8%`; the Minoura/Carron ratio must remain within
`[0.60,1.15]`. Electrical and Carron optical endpoint gaps must agree within
`10 meV`.

The photon-conserving TMM cell integral remains the production observable:
`sum(G_i dx_i)` is bounded by incident photons on every electrical mesh. The
registry lane refines optical slices and the Kramers-Kronig integral; it does
not use electrical mesh refinement to manufacture photon conservation.

## Evidence boundary

This capability supports a composition-resolved CIGS absorber optics claim.
It is not a calibrated device model. The shipped ZnO and CdS layers still use
nominal scalar optical fallbacks; CGI is spatially uniform; the source models
do not establish the actual composition, roughness, texture, parasitic loss,
or thickness of a measured cell. The lane contains no transport solve and
therefore cannot certify J-V, PCE improvement, external SCAPS/Setfos parity,
or experimental validity.

At the implementation checkpoint the source-clean 3x3 certificate is pending.
Its run ID, certificate hash, terminal differences, and worst quality values
will be recorded here only after the implementation is committed and the
matrix is executed from that clean source identity.
