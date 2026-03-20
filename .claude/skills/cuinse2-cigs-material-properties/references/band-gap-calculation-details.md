# Band Gap Calculation Details

## Full Calculation Example

For x = 0.5 (50% Ga composition):

```
Eg(0.5) = 0.5 × 1.035 + 0.5 × 1.68 - 0.264 × 0.5 × 0.5
Eg(0.5) = 0.5175 + 0.84 - 0.066
Eg(0.5) = 1.2915 eV
```

## Composition Range Behavior

| Ga Fraction (x) | Material | Approximate Band Gap |
|-----------------|----------|---------------------|
| 0.0 | Pure CuInSe2 | 1.035 eV |
| 0.25 | Cu(In0.75Ga0.25)Se2 | ~1.15 eV |
| 0.50 | Cu(In0.5Ga0.5)Se2 | ~1.29 eV |
| 0.75 | Cu(In0.25Ga0.75)Se2 | ~1.46 eV |
| 1.0 | Pure CuGaSe2 | 1.68 eV |

## Physical Interpretation

The quadratic term (-b·x·(1-x)) represents the bowing effect due to:
- Lattice mismatch between CuInSe2 and CuGaSe2
- Changes in atomic ordering
- Variation in bond lengths and angles

This bowing parameter causes the actual band gap to be lower than a simple linear interpolation would predict.