---
name: cuinse2-cigs-material-properties
description: Calculate material properties, doping behavior, and band gap energy for CuInSe2, CuGaSe2, and Cu(InGa)Se2 chalcopyrite semiconductors. Use when determining conductivity type from composition/annealing conditions or computing band gap for alloy compositions.
---

# CuInSe2/CIGS Material Properties and Band Gap Calculation

## When to Use
- Determining doping type (p-type vs n-type) for CuInSe2 or CIGS materials
- Calculating band gap energy for Cu(In1-xGax)Se2 alloys
- Accessing fundamental material constants for chalcopyrite semiconductors
- Planning processing conditions (annealing atmosphere, composition)

## Material Constants (CuInSe2)
| Property | Value |
|----------|-------|
| Crystal Structure | Chalcopyrite |
| Lattice Constant a | 5.78 Å |
| Lattice Constant c | 11.62 Å |
| Band Gap (Eg) | 1.04 eV |
| Melting Temperature | 986 °C |
| Density | 5.75 g/cm³ |
| Thermal Expansion (a-axis) | 11.23×10⁻⁶ /K |
| Thermal Expansion (c-axis) | 7.90×10⁻⁶ /K |
| Electron Effective Mass | 0.08 m₀ |
| Heavy Hole Effective Mass | 0.72 m₀ |
| Light Hole Effective Mass | 0.09 m₀ |

## Doping Type Determination

### For CuInSe2:
1. **Check copper composition:**
   - If excess Cu present → ALWAYS p-type

2. **If indium-rich:**
   - Doping type depends on Selenium (Se) content
   - Can be p-type or n-type

3. **Apply annealing effects:**
   - High pressure Se atmosphere → n-type converts to p-type
   - Low Se pressure → p-type converts to n-type

### For CuGaSe2:
- ALWAYS p-type (regardless of composition or annealing)

### Mobility Ranges:
- Hole mobility (Epitaxial): Up to 200 cm²/Vs
- Electron mobility (Crystals): 90 to 900 cm²/Vs
- Solar Cell practical values: 5–20 cm²/Vs

## Band Gap Calculation for Cu(In1-xGax)Se2

Use the empirical quadratic equation:
```
Eg(x) = (1-x)·Eg(CuInSe2) + x·Eg(CuGaSe2) - b·x·(1-x)
```

**Parameters:**
- Bowing parameter b = 0.264
- Eg(CuInSe2) = 1.035 eV
- Eg(CuGaSe2) = 1.68 eV
- x = Ga fraction (0 ≤ x ≤ 1)

**Constraints:**
- Valid only for thin films with In,Ga alloy
- Composition range: 0 ≤ x ≤ 1

See `references/band-gap-calculation-details.md` for detailed calculation examples.