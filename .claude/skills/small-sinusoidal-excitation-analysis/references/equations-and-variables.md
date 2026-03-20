# Equations and Variables Reference

## Core Equations

### Generation Rate
```
g(t) = g₀ + g₁ exp(iωt)
```

### Electron Density Response
```
n(t) = n₀ + n₁ exp(iωt)
```

### Coefficient A (Response Amplitude)
```
A = n₁/g₁ = 1 / (1/τₙ + iω(1 + dnₜ/dn))
```

### Coefficient B (Trap Density Derivative)
```
B = dnₜ/dn
```

### Time Constant Ratio (ωτ << 1)
```
|j_ac| / |j_dc| = g₁ / g₀ × (τ / τₙ)
```

### Carrier Mobility
```
μₙ = j_ac / (2ω e g₀ n₀)
```

## Variable Definitions

| Variable | Type | Description |
|----------|------|-------------|
| g(t) | function | Total generation rate (bias + modulation) |
| g₀ | rate | Constant bias generation rate |
| g₁ | rate | Amplitude of modulated generation |
| ω | frequency | Angular frequency of modulation |
| n(t) | function | Total electron density |
| n₀ | density | Steady-state electron density |
| n₁ | density | Amplitude of electron density modulation |
| τ | time | Carrier lifetime |
| τₙ | time | Electron recombination time constant |
| dnₜ/dn | dimensionless | Derivative of trap density with respect to electron density |
| j_ac | current density | AC component of photocurrent |
| j_dc | current density | DC component of photocurrent |
| μₙ | mobility | Electron carrier mobility |
| e | charge | Elementary charge |

## Equation References
- Eq 23.39: Generation rate formulation
- Eq 23.40: Electron density response
- Eq 23.43: Coefficient definitions
- Eq 23.45: Mobility calculation

## Special Cases

### Trap Saturation Condition
When light intensity is high enough to fill all traps:
- dnₜ/dn → 0
- τₙ = τ
- Simplifies mobility calculation

### Quasi-Fermi Level Between Trap Levels
When quasi-Fermi level falls between trap energy levels:
- dnₜ/dn << 1
- τₙ ≈ τ
- Mobility calculation valid