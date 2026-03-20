---
name: rutherford-scattering-and-ionized-impurity-scattering
description: Calculate scattering cross-sections and carrier mobility when Coulomb interactions with charged particles or ionized impurities dominate. Use this for analyzing carrier transport in semiconductors with charged defects, ionized dopants, or when classical Rutherford scattering theory applies.
---

# Rutherford Scattering and Ionized Impurity Scattering

## When to Use
- Coulomb-driven interactions (protons, alpha-particles, charged defects)
- Carrier transport in semiconductors with ionized impurities
- Calculating mobility limited by charged defect scattering
- High impurity densities or low temperatures where phonon scattering is reduced

## Fundamental Rutherford Scattering

### Basic Formula
For Coulomb-driven interactions:
```
σ ∝ 1 / R²
```
Where R is the distance of closest approach. The cross-section is angle-dependent (θ).

### Contrast with Hard-Sphere Collisions
- Rutherford: angle-dependent, Coulomb forces
- Hard-sphere: angle-independent (neutrons, fast ions)

## Ionic Defect Scattering in Semiconductors

### Differential Cross-Section
```
dσ/dΩ = (z²e⁴ε_st⁻²m_n²)/(4v_rms⁴sin⁴(θ/2))
```
Key variables:
- z: ionic charge number
- ε_st: static dielectric constant
- m_n: carrier effective mass
- v_rms: root-mean-square carrier velocity
- θ: scattering angle

### Scattering Parameter β
```
β = (ze²)/(4πε_stm_nv_rms²)
```

### Typical Magnitude
- Ionic scattering: ~10⁻¹³ cm²
- Neutral defect scattering: ~10⁻¹⁵ cm² (~100× smaller)

## Mobility Models

### Conwell-Weisskopf Model (Classical)
**When to use:** Low-field conditions, classical approximation

**Mobility formula:**
```
μ ∝ T^(3/2) / (N_t × Z²)
```

**Key features:**
- Cutoff distance: d_max = 1/(2×N_t^(1/3)) (average distance between centers)
- Coulomb radius: r_c = e²Z/(4πεε_0kT)
- Scattering radius: r_i = r_c × √(c_c)
- Mobility increases with T^(3/2) (faster electrons less effectively scattered)

### Brooks-Herring Model (Screened)
**When to use:** Screened potential, partial compensation, carrier density effects

**Key differences:**
- Uses screened Yukawa potential with Debye length λ_D as cutoff
- Screening length: λ_D ∝ 1/√n
- Accounts for partial compensation: n replaced by n×(n+N_a)/[1-(n+N_a)/N_d]
- Simple relation: λ_D ≈ d_max

**Mobility formula:**
```
μ ∝ T^(3/2) / N_eff (with screening corrections)
```

## Physical Interpretation
- Scattering occurs when carrier penetrates within Coulomb radius r_c
- Associated with charge trapping at Coulomb-attractive centers
- For inelastic events, carrier must reach r < r_c to dissipate energy
- Higher carrier densities reduce cross-section via screening