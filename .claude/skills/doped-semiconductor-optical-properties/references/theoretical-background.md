# Theoretical Background

## Burstein-Moss Effect Theory

### Fermi Level Position
In a degenerate semiconductor:
```
EF = Ec + (h²/2mn*)(3π²n)^(2/3)
```

### Absorption Edge Shift
The shift in absorption edge:
```
ΔE = EF - Ec = (h²/2mn*)(3π²n)^(2/3)
```

### Conditions for Significant Shift
1. Small effective mass → large shift
2. High carrier concentration
3. Low temperature (reduced thermal smearing)

## Band Tailing Theory

### Kane Model
For heavily doped semiconductors, the band tail width:
```
E0 = (e²/4πεε₀) × (4πND/3)^(1/3)
```
Where ND is donor concentration.

### Halperin-Lax Model
More accurate for deeper tail states:
- Accounts for potential fluctuations
- Includes correlation effects

## Density of States in Heavily Doped Materials

### Parabolic Band (Undoped)
```
N(E) = (1/2π²) × (2m*/h²)^(3/2) × √(E - Ec)
```

### With Band Tailing
```
N(E) = N_parabolic(E) + N_tail(E)
```

The tail contribution:
- Exponential decay into gap
- Material dependent width

## Optical Transition Matrix Elements

### k-Selection Rule (Ordered Crystal)
- Momentum conserved
- Vertical transitions in k-space

### No k-Selection (Disordered)
- Energy conservation only
- Transitions between all states with appropriate energy

## Worked Examples

### Example 1: n-GaAs Burstein-Moss Shift
Given:
- mn* = 0.067 m₀
- n = 10^18 cm^-3
- T = 300 K

Calculate:
```
Nc = 2 × (2π × 0.067 × 9.11×10^-31 × 1.38×10^-23 × 300 / (6.626×10^-34)²)^(3/2)
Nc ≈ 4.7 × 10^17 cm^-3

EF - Ec ≈ kT × ln(n/Nc) ≈ 0.026 × ln(2.1) ≈ 0.019 eV
```

### Example 2: Absorption Edge Shift in InSb
Given:
- mn* = 0.014 m₀ (very low!)
- n = 10^18 cm^-3

The Burstein-Moss shift is much larger due to low mass:
```
EF - Ec ≈ 0.15 eV (significant shift)
```

## References
- Abram et al. (1978) - Absorption coefficient theory
- Burstein (1954) - Original observation
- Moss (1954) - Independent discovery