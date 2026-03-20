# Haug's Formula for Auger Lifetime

## Complete Formula

The quantum-mechanical expression for Auger-limited electron lifetime:

```
τ_A = [2.4 × 10⁻³¹ × (εr/m*)² × (1 + m*/m₀) × exp(ΔE/kT)] / (n² × I₁² × I₂²)
```

## Parameter Definitions

| Symbol | Description | Units |
|--------|-------------|-------|
| εr | Relative permittivity | dimensionless |
| m* | Effective mass | m₀ |
| m₀ | Free electron mass | 9.109 × 10⁻³¹ kg |
| ΔE | Energy dissipated in Auger process | eV |
| Eg | Band gap energy | eV |
| I₁, I₂ | Overlap integrals | dimensionless |
| n | Electron density | cm⁻³ |
| kT | Thermal energy | eV |

## Energy Dissipation Term

```
ΔE = [(2m* + mp)/(m* + mp)] × Eg
```

Where mp is the hole effective mass.

## Overlap Integrals

Typical estimation values:
- I₁ ≈ 1 (Bloch function overlap)
- I₂ ≈ 0.1 (momentum conservation factor)
- Combined: I₁² × I₂² ≈ 0.01

## Quantum-Mechanical Derivation

The Auger matrix element M involves:

1. **Bloch functions**: Φ for initial and final states
2. **Screened Coulomb potential**:
   ```
   V(r) = (e²/4πε₀εr) × (exp(-qλ)/q)
   ```

Where:
- λ = electron screening factor
- q = |k' - k| = momentum transfer

## Material-Specific Examples

### InSb (Narrow Gap)
- Eg = 0.17 eV
- εr ≈ 17
- m*/m₀ ≈ 0.014
- B ≈ 10⁻²⁶ cm⁶s⁻¹
- Auger dominant at room temperature

### Silicon (Indirect Gap)
- Eg = 1.12 eV
- εr ≈ 11.7
- m*/m₀ ≈ 0.26
- B ≈ 10⁻³¹ cm⁶s⁻¹
- Auger only at very high injection

## Temperature Dependence

The exponential term exp(ΔE/kT) shows:
- Higher temperature → shorter lifetime
- Stronger effect in wider gap materials
- Activation energy related to band gap