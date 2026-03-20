# Derivation and Examples

## Derivation from Fermi-Dirac Statistics

### Fermi-Dirac Distribution
The probability that an energy state E is occupied:
```
f(E) = 1 / [exp((E - EF)/kT) + 1]
```

### Boltzmann Approximation
When E - EF >> kT:
```
f(E) ≈ exp[-(E - EF)/kT]
```

### Electron Density Integration
```
n = ∫ Nc(E) × f(E) dE
```

For parabolic bands:
```
n = Nc × exp[-(Ec - EF)/kT]
```

## Worked Example: Silicon at 300K

### Given:
- mn* = 1.08 × m0 (density-of-states effective mass)
- T = 300 K
- kT = 0.0259 eV

### Calculate Nc:
```
Nc = 2 × (2π × 1.08 × 9.11×10^-31 × 1.38×10^-23 × 300 / (6.626×10^-34)²)^(3/2)
Nc ≈ 2.8 × 10^19 cm^-3
```

### For n-type Si with EF - Ec = -0.1 eV:
```
n = 2.8 × 10^19 × exp(0.1/0.0259)
n ≈ 2.8 × 10^19 × 0.021
n ≈ 5.9 × 10^17 cm^-3
```

## Validity Range

| Condition | Validity |
|-----------|----------|
| Ec - EF > 3kT | Boltzmann valid |
| Ec - EF ≈ kT | Moderate accuracy |
| Ec - EF < kT | Degenerate, use Fermi-Dirac |

## Space-Charge Region Application

In the space-charge region:
- ψ varies with position
- n varies according to local potential
- ψn remains constant in equilibrium

This allows calculation of carrier density profiles in:
- p-n junctions
- Schottky barriers
- MOS structures