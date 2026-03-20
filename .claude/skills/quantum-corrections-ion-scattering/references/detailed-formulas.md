# Detailed Formulas for Quantum Corrections

## Born Approximation Correction (δ_B)

### Low Temperature (T < 100K)

δ_B = 2Q(β) * const

Where:
- Q(β) is a slowly varying function with range 0.2 < Q < 0.8
- The constant depends on material parameters

### High Temperature (T > 100K)

δ_B ∝ √(n/T^(3/2))

Where:
- n is the ion density
- T is the temperature

## Multiple Scattering Correction (δ_m)

Estimated by Raymond et al. (1977) for conditions where:
- Mean free path ≈ screening length
- Coherent scattering from multiple ions occurs

## Dressing Effect Correction (δ_d)

δ_d ≈ (0.3 to 0.5) × δ_m

This correction accounts for the chemical individuality of scattering centers through:
1. Electron wave function of the scattering ion (small effect)
2. Stress field surrounding impurities of different sizes (dominant mechanism)

## Material-Specific Observations

### InSb doped with Se or Te
- Demchuk and Tsidilkovskii (1977) observed significant defect individuality at high impurity densities

### Si or Ge doped with As or Sb
- Morimoto and Tani (1962) reported similar effects

## Key References

- Moore (1967): Established importance of Born correction at T < 100K
- Raymond et al. (1977): Multiple scattering correction estimation
- Morgan (1972): Proposed stress field mechanism over central cell potential corrections
- Demchuk and Tsidilkovskii (1977): InSb doping studies
- Morimoto and Tani (1962): Si/Ge doping studies

## Typical Parameter Ranges

| Parameter | Range |
|-----------|-------|
| Ion density (n) | 10^16 to 10^19 cm^-3 |
| δ_B | 0.1 to 1 |
| Q(β) | 0.2 to 0.8 |
| δ_d / δ_m | 0.3 to 0.5 |