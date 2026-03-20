---
name: assisted-tunneling-mechanisms
description: Calculate tunneling probabilities modified by phonon, trap, or photon assistance when standard tunneling is insufficient or specific energy exchange processes occur. Use this for indirect band-gap materials, defect-assisted transport, or optical field-enhanced tunneling.
---

# Assisted Tunneling Mechanisms

## When to Use
- Indirect band-gap materials requiring momentum change
- High defect density enabling trap-assisted tunneling
- Optical excitation combined with electric fields
- When bias reaches phonon/photon energy thresholds

## Mechanism Selection

### Phonon Assistance
**Conditions:** Indirect gap materials

**Modified tunneling probability:**
```
Te_phonon ~ exp(-2∫k(x)dx ± ℏω/(eFa))
```
Where:
- ℏω: phonon energy
- F: electric field
- a: characteristic length

**Key characteristics:**
- Probability reduced by factor ~10⁻³ vs direct tunneling
- Current increases measurably when Vr reaches phonon energy
- Provides momentum for indirect transitions

### Trap Assistance (Two-Step Tunneling)
**Conditions:** High defect density near interface

**Process:**
1. Tunnel from band into trap state
2. Tunnel from trap to opposite band

**Total probability:**
```
1/Te_total = 1/Te_1 + 1/Te_2
```
Where Te_1 and Te_2 are individual step probabilities.

**Requirements:**
- Defect centers close to interface
- Sufficient trap density
- Appropriate trap energy levels

### Photon Assistance (Franz-Keldysh Effect)
**Conditions:** Optical excitation with applied electric field

**Process:** Photon provides most energy, field provides small additional energy enabling tunneling

**Absorption edge shift:**
```
ΔE_opt = constant × (eFℏ)^(2/3) / (2m)^(1/3)
```

**Example:** For 10 meV shift (100 Å at 2 eV gap):
- Required field: ~50 kV/cm

**Key characteristics:**
- Shifts absorption edge to lower energies
- Exponent remains constant (~1)
- Enables photon absorption below nominal bandgap

## Comparison of Mechanisms
| Mechanism | Energy Exchange | Probability Factor | Typical Conditions |
|-----------|----------------|-------------------|-------------------|
| Phonon | Momentum + small energy | ~10⁻³ | Indirect gap, Vr ≈ ℏω |
| Trap | Stepwise via defect | Depends on trap density | High defect density |
| Photon | Large optical + small field | Field-dependent | Optical illumination |

## Application Notes
- Phonon and photon assistance provide momentum for indirect transitions
- Trap assistance enables tunneling through otherwise forbidden barriers
- Multiple mechanisms can operate simultaneously
- Dominant mechanism depends on material properties and experimental conditions