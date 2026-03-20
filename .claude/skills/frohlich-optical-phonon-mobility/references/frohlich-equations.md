# Detailed Fröhlich Interaction Equations

## Polarization Calculation (Eq. 17.25)

The polarization due to lattice vibration is:

```
P = (d_r × e_c) / V_o
```

where:
- P = polarization
- d_r = change in interatomic distance during vibration
- e_c = Callen effective charge
- V_o = unit lattice cell volume

This polarization creates an induced electric field proportional to P.

## Low-Temperature Mobility (T < θ, Eq. 17.26)

Using the variational method of Howarth and Sondheimer (1953):

```
μ_n,ion,opt = (2 × m_n × e × α_c × k × θ) × exp(θ/T)
```

where:
- μ_n,ion,opt = electron mobility from ionic optical phonon scattering
- m_n = effective mass of electrons
- e = elementary charge
- α_c = coupling constant
- k = Boltzmann constant
- θ = Debye temperature (related to LO phonon energy: ℏω_LO = kθ)
- T = temperature

## Numerical Expression (Eq. 17.27)

```
μ_n,ion,opt = C × (m_n/m_0)^{-1/2} × (α_c)^{-1} × (θ) × exp(θ/T)
```

where C is a material-specific constant.

## High-Temperature Approximation (T ≥ θ, Eq. 17.28)

From Seeger (1973):

```
μ_n,ion,opt = A × (m_n/m_0)^{-1/2} × (α_c)^{-1} × (θ/T)^(1/2) × (exp(θ/T) - 1)^(-1)
```

where A is a constant specific to the material system.

## Screened Potential (Zawadzki and Szymanska, 1971)

The Yukawa-type screened potential modifies the scattering rate:

```
μ_screened = μ_unscreened / F_op
```

At high temperatures:
```
F_op ≈ 1 (screening parameter)
```

For InSb at room temperature with n = 10^19 cm⁻³:
```
μ_screened ≈ 3.5 × μ_unscreened
```

## Degenerate Semiconductor Case (Eq. 17.30)

When Fermi level is in the conduction band:

```
μ = f(n, α_c, θ, T, E_F)
```

The mobility becomes explicitly dependent on electron density n through the Fermi energy E_F.

## Key Relationships

### Coupling Constant (Eq. 17.6)

```
α_c = (e^2 / ℏ) × (m_n/2ℏω_LO)^(1/2) × (1/ε_∞ - 1/ε_s)
```

where:
- ε_∞ = high-frequency dielectric constant
- ε_s = static dielectric constant

### Debye Temperature and LO Phonon Energy

```
ℏω_LO = kθ
```

The Debye temperature θ characterizes the LO phonon energy at q=0 (Brillouin zone center).

## Behavior Summary

| Regime | Temperature Dependence | Physical Interpretation |
|--------|----------------------|------------------------|
| T << θ | exp(θ/T) | Optical phonon freeze-out |
| T ≈ θ | Transition | Both emission and absorption |
| T >> θ | (θ/T)^(1/2) | Classical limit |

## Material Examples

| Material | θ (K) | α_c | Typical μ at 300K |
|----------|------|-----|-------------------|
| GaAs | ~360 | ~0.07 | ~8000 cm²/V·s |
| InSb | ~200 | ~0.02 | ~78000 cm²/V·s |
| InP | ~420 | ~0.12 | ~5400 cm²/V·s |