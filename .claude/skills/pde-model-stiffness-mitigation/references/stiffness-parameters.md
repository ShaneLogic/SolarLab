# Dimensionless Parameter Estimates

## Typical Device Configuration

- 600 nm thick methylammonium lead tri-iodide perovskite absorber
- Sandwiched between Titania ETL and Spiro HTL

## Parameter Values (Equation 18)

1. **λ (Lambda)**: Debye length / layer width
   - Value: ≈ 10^(-3)
   - Physical meaning: Characteristic electrostatic screening relative to device thickness

2. **ν (Nu)**: Timescale ratio (carrier motion / ion vacancy motion)
   - Value: ≈ 10^(-8)
   - Physical meaning: How much faster electronic carriers move compared to ions

3. **δ (Delta)**: Carrier concentration / vacancy concentration
   - Value: ≈ 10^(-2)
   - Physical meaning: Relative abundance of electronic vs ionic charge carriers

## Implications for Numerical Solution

### Small ν (≈ 10^(-8))
- Large disparity in timescales
- Electronic processes are extremely fast
- Ionic processes are extremely slow
- **Necessitates**: Adaptive timestep methods

### Small λ (≈ 10^(-3))
- Characteristic of electrochemical problems
- Causes stiffness due to rapid changes in narrow Debye layers
- **Necessitates**: Non-uniform meshing

### Large Potential Drop
- φ_bi - φ(t) can be significant
- Exacerbates stiffness across Debye layers

## Exponential Stiffness Example

Concentrations in Debye layers are Boltzmann distributed:

n ∝ exp(-qφ/kT)

For a 0.5V potential drop:
- Dimensionless potential: 0.5 / V_T ≈ 20 (at room temperature)
- Resulting change: exp(20) ≈ 4.85 × 10^8

This means concentrations can change by a factor of ~10^9 over a width of ~10^-3 dimensionless units.

## Conclusion

The problem is classified as "extremely stiff" due to:
1. Exponential sensitivity to potential variations
2. Disparate timescales spanning 8 orders of magnitude
3. Narrow boundary layers requiring high spatial resolution

These characteristics lead to large condition numbers in the discretized system and significant round-off errors in standard numerical methods.