# Band-gap narrowing closure

SolarLab exposes one opt-in empirical band-gap-narrowing (BGN) law for the
restricted high-doping silicon equilibrium research lane. The default remains
`band_gap_narrowing_model: off`; no existing configuration changes physics or
semantic identity unless the model is explicitly activated.

## Constitutive law

The implemented Slotboom-de Graaff form is

```text
N_I       = N_A + N_D
u         = ln(N_I / N_ref)
Delta E_g = E_ref [u + sqrt(u^2 + C)]
```

where `N_A` and `N_D` are chemical dopant densities, not the temperature- and
potential-dependent ionized fractions. The frozen silicon research parameters
are:

```text
E_ref = 0.009 eV
N_ref = 1.0e23 m^-3 = 1.0e17 cm^-3
C     = 0.5
```

These values and the equal conduction/valence edge partition follow the common
silicon Slotboom parameterization documented by
[TU Wien](https://www.iue.tuwien.ac.at/phd/palankovski/node39.html). The model
form is also documented in the
[COMSOL Semiconductor Module](https://doc.comsol.com/6.3/doc/com.comsol.help.semicond/semicond_ug_semiconductor.6.60.html);
the original empirical work is
[Slotboom and de Graaff (1976)](https://doi.org/10.1016/0038-1101(76)90043-5).

For `u < 0`, the implementation evaluates the algebraically equivalent form

```text
C / [sqrt(u^2 + C) - u]
```

and computes `u` as `ln(N_I) - ln(N_ref)`. This avoids cancellation and ratio
underflow at extremely low but finite densities. Zero chemical doping returns
exactly zero narrowing.

## Band-edge convention

With conduction-band fraction `alpha_c`, SolarLab applies

```text
chi_eff = chi + alpha_c Delta E_g
Eg_eff  = Eg  - Delta E_g
```

so the conduction edge moves downward by `alpha_c Delta E_g` and the valence
edge moves upward by `(1-alpha_c) Delta E_g`. The configured fraction must lie
in `[0, 1]`, and `Delta E_g` must remain smaller than the temperature-adjusted
base gap.

The same effective affinity and gap feed:

- semiconductor contact charge-neutrality and work-function construction;
- Poisson carrier statistics in the restricted PN equilibrium solver;
- generalized Scharfetter-Gummel carrier fluxes;
- the intrinsic product, scaled as `ni^2 exp(Delta E_g/V_T)`; and
- midgap-fixed SRH reference densities, each scaled by
  `exp(Delta E_g/(2 V_T))`.

The registered research lane disables bulk recombination, so the last two
quantities are closure checks rather than a degenerate-recombination claim.

## Capability boundary

An active BGN model requires
`built_in_potential_mode: semiconductor_work_function`. The default production
material assembly rejects it. The only enabled bulk route is the existing
`research_degenerate_recombination_off` homojunction equilibrium slice, which
also requires one common carrier-statistics, dopant-ionization, and BGN law on
all electrical layers.

The combined frozen configuration is
`configs/csi_incomplete_ionization_bgn_pn_research.yaml`. Its registered lane,
`incomplete-ionization-bgn-temperature-equilibrium-v1`, scans 100-300 K over a
3x3 grid/tolerance matrix and records the BGN value, effective gap, ionized
fractions, normalized peak field, integrated charge width, and charge balance.

This closure does not model impurity bands, the Mott transition, alternate BGN
parameterizations, dopant kinetics, degenerate SRH/Auger/radiative
recombination, biased transport, heterojunctions, or production experiments.
Passing the registered matrix establishes internal numerical consistency only;
it is not external Sentaurus/PC1D validation or experimental validation.

## Internal numerical certificate

The source-clean 3x3 matrix at commit `d34ef7f` completed all nine cells with
zero failed, missing, or reused cells:

```text
run ID       bc654a0c76f2d13cdbf64256160fc22b8c7079388aa4babd76998e866e3c3557
certificate  bc1285a3ef9d42e1e6346aedad33ad3871b5597a7b409f4ea002bf029f024ce6
protocol     58c95444ae1d35ad819b53a238cc266c18699d984cb700ee28a89db76b992e3a
```

At the terminal grid/tolerance cell, `Delta E_g` is `0.0216460 eV` across the
frozen temperature scan while the Varshni-adjusted effective gap decreases
from `1.141408 eV` at 100 K to `1.102354 eV` at 300 K. Acceptor and donor
ionized fractions increase from `0.09027/0.19690` to `0.68718/0.90058`.

Terminal grid differences are `2.1641e-4` for normalized integrated charge
width, `1.1777e-3` for normalized peak field, and `4.9062e-4` for space-charge
balance. Corresponding terminal tolerance differences are `1.8514e-7`,
`5.6964e-8`, and `2.7499e-7`. Across the full matrix, maximum normalized
Poisson residual, carrier rate, relative face current, and charge-balance
error are `7.3089e-9`, `1.7262e-13`, `2.1785e-13`, and `1.9930e-3`; no solve
required more than nine Newton iterations.
