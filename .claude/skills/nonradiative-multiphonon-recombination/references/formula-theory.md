# Nonradiative Multiphonon Recombination - Theory Reference

## Complete Formula

### Haug's Formula (Eq. 22.18)

The transition probability for band-to-band recombination via multiphonon emission:

```
Pcv = exp[-S(2nph + 1)] × (nph + 1)⁻S × (hv₁/hω₀)
```

### Phonon Occupation Number

Bose-Einstein distribution for phonons:

```
nph = 1 / [exp(hω₀/kT) - 1]
```

## Physical Interpretation

### Huang-Rhys Factor (S)

The Huang-Rhys factor represents the average number of phonons emitted during the transition:
- Larger S → Stronger electron-phonon coupling
- More phonons involved in the transition

### Temperature Dependence Mechanism

1. Phonon occupation (nph) increases with temperature
2. Thermal energy assists multiphonon processes
3. Exponential suppression factor becomes less effective

## Comparison: Radiative vs Nonradiative

| Property | Nonradiative | Radiative |
|----------|--------------|-----------|
| Temperature dependence | Strong (exponential) | Weak (minor) |
| Dominance regime | High T | Low T |
| Energy dissipation mechanism | Phonons | Photons |

## Typical Values

### Energy Scales
- Bandgap transitions: ~1 eV
- Phonon energies: ~10-50 meV
- Phonons for 1 eV dissipation: ~20-100
- Typical value: ~30 phonons for 1 eV bandgap

### Constraints
- Rare transitions for tightly bound centers
- High rates may indicate alternative mechanisms

## Alternative Mechanisms

When predictions don't match observations:

1. **Deep centers with large lattice relaxation**
   - Significant atomic rearrangement
   - Enhanced multiphonon efficiency

2. **Auger-like processes**
   - Energy transfer to another carrier
   - Carrier acceleration near recombination centers

3. **Cascade capture**
   - Sequential phonon emission
   - Multiple excited state transitions