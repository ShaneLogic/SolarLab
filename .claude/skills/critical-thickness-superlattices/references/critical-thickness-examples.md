# Critical Thickness Reference Values and Examples

## Reference Values

### Lattice Mismatch vs. Critical Length
| Lattice Mismatch | Critical Length (approximate) |
|------------------|-------------------------------|
| 4%               | ~25Å                          |

Note: Critical length decreases as lattice mismatch increases.

## Example Material Systems

### Si/Ge System
- Silicon (Si): Lattice constant ≈ 5.431 Å
- Germanium (Ge): Lattice constant ≈ 5.658 Å
- Lattice mismatch: ~4.2%
- Critical length: ~25Å

### GaAs/InAs System
- Gallium Arsenide (GaAs): Lattice constant ≈ 5.653 Å
- Indium Arsenide (InAs): Lattice constant ≈ 6.058 Å
- Lattice mismatch: ~7.2%
- Critical length: Significantly less than 25Å (decreases with higher mismatch)

## Calculation Example

For Si/Ge superlattice:
1. Lattice mismatch = 4.2%
2. Critical length ≈ 25Å (from reference)
3. If planning 20Å layers: 20Å < 25Å → **Safe**, dislocation-free growth possible
4. If planning 30Å layers: 30Å > 25Å → **Unsafe**, dislocations will form

## Edge Cases and Considerations

- Temperature effects: Critical length can vary with deposition temperature
- Strain relaxation: Multiple layers may behave differently than single layers
- Surface energy effects: Very thin layers (< 5-10Å) may have additional considerations