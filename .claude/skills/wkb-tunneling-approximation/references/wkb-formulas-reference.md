# WKB Tunneling Formulas Reference

## Variable Definitions

| Variable | Type | Description |
|----------|------|-------------|
| Te | Dimensionless | Transmission probability |
| DeltaE | Energy | Barrier height or energy difference |
| Eg | Energy | Bandgap energy |
| F | Electric Field | Applied electric field |
| m | Mass | Effective mass of tunneling particle |
| hbar | Action | Reduced Planck constant (ħ) |
| e | Charge | Elementary charge |
| x | Position | Spatial coordinate |

## Detailed Formulas

### General WKB Expression

```
Te = exp(-2 * ∫k(x)dx)

where:
k(x) = √(2m * ΔE(x)) / ħ
```

### 1. Triangular Barrier

**Wavevector:**
```
k(x) = √(2m * (ΔE - eFx)) / ħ
```

**Transmission Probability:**
```
Te = exp(-4 * √(2m) * ΔE^(3/2) / (3 * ħ * eF))
```

**Integration limits:** From x=0 to x=ΔE/eF (where barrier goes to zero)

### 2. Parabolic Barrier

**Physical interpretation:**
- Barrier lowering from triangular case due to image force effects
- Effective barrier height reduced from Eg to Emax

**Transmission Probability:**
```
Te = exp(-π * ΔE² / (2 * ħ * eF))
```

**Key relationship:**
- Exponent is reduced by factor 3π/16 ≈ 0.59 compared to triangular barrier
- Superlinear dependence: doubling ΔE requires 2^(3/2) ≈ 2.83× field for same Te

### 3. Band-to-Band Tunneling

**Formula:**
```
Te = exp(-π * Eg² / (2 * ħ * eF))
```

**Requirements:**
- Very high fields (> 10^6 V/cm) for significant current
- Exception: narrow gap semiconductors (InSb, Ge, etc.)

**Applications:**
- Zener breakdown
- Tunnel diodes
- Esaki diodes

### 4. Parabolic Barrier with Overlapping Fields

**Scenario:** Combined Coulomb potential and external electric field

**Wavevector:**
```
k(x) = √(2m * (ΔE² - (eFx)²)) / ħ
```

**Transmission Probability:**
```
Te = exp(-π * ΔE² / (4 * ħ * eF))
```

**Integration limits:** From x=-ΔE/eF to x=+ΔE/eF

## Mathematical Derivations

### Triangular Barrier Integration

```
∫₀^(ΔE/eF) √(2m * (ΔE - eFx)) / ħ dx

Let u = ΔE - eFx, du = -eF dx

= (1/eF) ∫₀^ΔE √(2m * u) / ħ du
= (√(2m) / (ħ * eF)) * (2/3) * ΔE^(3/2)
= (2√(2m) * ΔE^(3/2)) / (3ħ * eF)

Te = exp(-2 * integral) = exp(-4√(2m) * ΔE^(3/2) / (3ħ * eF))
```

### Parabolic Barrier Integration

For parabolic barrier: ΔE(x) = √(ΔE² - (eFx)²)

```
∫^(-ΔE/eF)_(ΔE/eF) √(2m * √(ΔE² - (eFx)²)) / ħ dx

Using substitution and trigonometric integration:
= π * ΔE² / (4ħ * eF)

Te = exp(-2 * integral) = exp(-π * ΔE² / (2ħ * eF))
```

## Example Calculations

### Example 1: Triangular Barrier in Silicon

Given:
- ΔE = 1.1 eV (Si bandgap)
- F = 10^6 V/cm = 10^8 V/m
- m* = 0.26 * m₀ (electron effective mass)
- ħ = 1.055 × 10^(-34) J·s
- e = 1.602 × 10^(-19) C

Calculate Te using triangular barrier formula.

### Example 2: Band-to-Band Tunneling Comparison

Compare tunneling probability for:
- Silicon (Eg = 1.1 eV)
- Germanium (Eg = 0.66 eV)
- Indium Antimonide (Eg = 0.17 eV)

at same electric field F = 10^6 V/cm

## Limitations and Assumptions

1. **Pre-exponential factor**: Neglected (assumed ≈ 1), actual values may differ by orders of magnitude
2. **Effective mass**: Assumed constant, may vary with energy
3. **Barrier shape**: Idealized triangular/parabolic approximations
4. **Temperature effects**: Not included in basic WKB treatment
5. **Multi-dimensional effects**: 1D approximation only

## Related Concepts

- Fowler-Nordheim tunneling (triangular barrier with field emission)
- Direct tunneling vs. trap-assisted tunneling
- Image force barrier lowering (Schottky effect)
- Resonant tunneling in double-barrier structures