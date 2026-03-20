---
name: Nonradiative Multiphonon Recombination
description: Calculate transition probability for nonradiative multiphonon recombination in semiconductors. Use when analyzing band-to-band recombination, tightly bound centers, or multiphonon emission processes. Trigger when user asks about temperature dependence of recombination rates, Huang-Rhys factor calculations, or comparing radiative vs nonradiative transitions.
---

# Nonradiative Multiphonon Recombination Probability

## When to Use

Apply this skill when:
- Calculating transition probability for band-to-band recombination via multiphonon emission
- Analyzing tightly bound centers with multiphonon processes
- Evaluating temperature dependence of nonradiative recombination
- Comparing radiative vs nonradiative transition rates

## Prerequisites

Gather the following parameters before calculation:
- **hv₁**: Energy to be dissipated (electron energy)
- **hω₀**: Relevant phonon energy
- **S**: Huang-Rhys factor (average phonons emitted)
- **T**: Temperature

## Procedure

### Step 1: Calculate Phonon Occupation Number

```
nph = 1 / [exp(hω₀/kT) - 1]
```

where k is the Boltzmann constant.

### Step 2: Apply Haug's Formula

Calculate transition probability Pcv using Eq. 22.18:

```
Pcv = exp[-S(2nph + 1)] × (nph + 1)⁻S × (hv₁/hω₀)
```

### Step 3: Interpret the Results

Analyze the behavior:
- **Temperature effect**: Pcv increases EXPONENTIALLY with temperature
- **Energy effect**: Pcv DECREASES with increasing energy dissipation (hv₁)
- **Dominance**: Nonradiative transitions PREDOMINATE at higher temperatures

### Step 4: Compare with Radiative Transitions

Radiative transitions show only minor temperature dependence. This contrast explains why nonradiative processes dominate at elevated temperatures.

### Step 5: Validate Against Typical Values

- For 1 eV bandgap transition: typically ~30 phonons emitted
- Such transitions are comparatively rare for tightly bound centers

## Output

Return transition probability Pcv (dimensionless, typically very small).

## Troubleshooting

If observed nonradiative rates exceed predictions at elevated temperatures, consider:
- Deep centers with large lattice relaxation
- Acceleration of free carriers near recombination centers taking up energy

## Key Variables

| Variable | Description |
|----------|-------------|
| Pcv | Transition probability |
| S | Huang-Rhys factor (dimensionless) |
| hv₁ | Energy to dissipate |
| hω₀ | Phonon energy |
| nph | Phonon occupation number |
| T | Temperature |