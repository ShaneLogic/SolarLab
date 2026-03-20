# Detailed Scattering Model References

## Gas-Kinetic Model Limitations
The gas-kinetic model assumes:
- Random distribution of scattering centers
- Well-defined scattering cross-sections
- Straight-line paths between collisions

These simplifications often lead to overestimation of tolerable defect densities in real semiconductors.

## Scattering Angle Distributions

### Small-Angle Scattering
- Common in ionized impurity scattering
- Coulomb interactions favor small deflections
- Multiple collisions required for momentum randomization
- Results in τm >> τsc

### Large-Angle Scattering
- Common in acoustic phonon scattering at low temperatures
- Neutral impurity scattering
- Results in τm ≈ τsc

## Energy Relaxation Details

### Acoustic Phonon Energy Loss
Using equipartition for acoustic phonons:
```
Energy loss per collision ≈ (m / M*) × (kT)
```
Where M* is the effective phonon mass.

For typical semiconductors at room temperature:
- m ≈ 0.1 me (electron mass)
- M* ≈ 10⁻²⁰ kg
- Energy fraction lost ~ 10⁻³

### Optical Phonon Emission
- Threshold energy: ħω₀ (optical phonon energy)
- When electron energy > ħω₀, optical phonon emission becomes dominant
- At room temperature: τE ≈ τsc
- Energy loss per event: ~30-60 meV (typical optical phonon energy)