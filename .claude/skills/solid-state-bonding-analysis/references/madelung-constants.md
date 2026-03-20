# Madelung Constants for Common Crystal Structures

| Crystal Structure | Madelung Constant (A) |
|------------------|----------------------|
| NaCl | 1.7476 |
| CsCl | 1.7627 |
| Zinc-blende | 1.6381 |
| Wurtzite | 1.6410 |
| CaF₂ | 5.0388 |
| Cu₂O | 4.1155 |
| TiO₂ (Rutile) | 4.8160 |

## Calculation Method

The Madelung constant sums over all neighbors:
```
A = Σ[(number of equidistant neighbors) / (distance in lattice units)]
```

- Series alternates signs based on charge polarity
- Direct summation converges slowly
- Use Ewald's method (theta-function method) for efficient numerical evaluation

## Example: NaCl Lattice Energy

Given:
- A = 1.7476
- re = rNa+ + rCl- = 2.8 Å
- m = 9

Theoretical lattice binding energy: H₀(NaCl) = 7.948 eV
Experimental value: 7.934 eV