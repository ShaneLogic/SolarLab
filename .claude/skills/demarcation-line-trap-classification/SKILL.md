---
name: demarcation-line-trap-classification
description: Classify localized states in semiconductor band gaps as electron traps, hole traps, or recombination centers by calculating demarcation lines based on quasi-Fermi levels, capture cross-sections, and temperature. Use this skill when analyzing trap behavior, determining recombination efficiency, or characterizing defect states in semiconductor materials.
---

# Demarcation Line Trap Classification

Classify band gap centers as electron traps, hole traps, or recombination centers using the demarcation line method.

## When to Use

Use this skill when:
- Analyzing localized states in semiconductor band gaps
- Determining whether a defect acts primarily as a trap or recombination center
- Studying recombination mechanisms in steady-state conditions
- Characterizing defect behavior in n-type or p-type materials

## Prerequisites

Before calculating demarcation lines, ensure you have:
- Electron quasi-Fermi level (EFn) and hole quasi-Fermi level (EFp)
- Electron capture cross-section (sn) and hole capture cross-section (sp)
- Temperature (T)
- Effective density of states (Nc for conduction band, Nv for valence band)
- Band gap energy (Ec - Ev)

## Core Procedure

### Step 1: Calculate Correction Terms

Calculate the correction terms δi and δj based on capture cross-section ratios:

```
δi = kT × ln[(sn/sp) × (Nc/Nv) × exp(-(Ec - Ev)/2kT)]
δj = kT × ln[(sp/sn) × (Nv/Nc) × exp(-(Ec - Ev)/2kT)]
```

Where:
- k = Boltzmann constant (8.617 × 10⁻⁵ eV/K)
- T = Temperature in Kelvin
- sn = Electron capture cross-section (cm²)
- sp = Hole capture cross-section (cm²)
- Nc = Effective density of states in conduction band
- Nv = Effective density of states in valence band
- Ec - Ev = Band gap energy

### Step 2: Calculate Demarcation Lines

Compute the electron and hole demarcation lines:

```
EDn = EFn - δi
EDp = EFp + δj
```

Where:
- EDn = Electron demarcation line energy
- EDp = Hole demarcation line energy
- EFn = Electron quasi-Fermi level
- EFp = Hole quasi-Fermi level

### Step 3: Classify the Center

Apply classification rules based on the center's energy position:

| Center Position | Classification |
|----------------|----------------|
| Above EDn (closer to conduction band) | Electron trap |
| Below EDp (closer to valence band) | Hole trap |
| Between EDn and EDp (middle of band gap) | Recombination center |

## Key Considerations

- **Steady-state assumption**: This method applies under steady-state conditions
- **Capture cross-section dependence**: The sn/sp ratio can change with center occupancy
- **Material type**: n-type materials typically show a wider range of electron traps, while p-type materials show more hole traps
- **Temperature sensitivity**: δ values vary with temperature, affecting demarcation line positions

## Common Values

- Capture cross-sections typically range from 10⁻¹² to 10⁻²² cm²
- At room temperature, δ values can vary by up to 0.6 eV
- Neutral centers often have sn ~ 10⁻¹⁶ cm², becoming negatively charged after electron capture (sp ~ 10⁻¹⁴ cm²)

## Output

The skill provides:
- Electron demarcation line energy (EDn)
- Hole demarcation line energy (EDp)
- Classification of the center (electron trap, hole trap, or recombination center)

For detailed examples, derivation, and edge cases, see `references/demarcation-line-details.md`.