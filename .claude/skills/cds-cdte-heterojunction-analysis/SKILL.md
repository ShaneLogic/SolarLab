---
name: CdS/CdTe Heterojunction Analysis
description: Analyze electric field distribution, field quenching mechanisms, and photoconductivity effects in CdS/CdTe heterojunction solar cells. Use when modeling CdS-based junctions, understanding field-dependent carrier behavior, or analyzing copper-doped CdS performance under optical excitation.
---

# CdS/CdTe Heterojunction Analysis

## When to Use
- Analyzing field distribution in CdS side of CdS/CdTe junctions
- Modeling field quenching effects in copper-doped CdS
- Understanding photoconductivity changes under bias
- Predicting band alignment changes at heterojunction interface
- Analyzing Schottky barrier behavior in CdS

## Prerequisites
- Trap density in CdS (typically ~10^17 cm^-3)
- Copper doping concentration (if applicable)
- Bias conditions
- Optical excitation level

## Procedure

### 1. Calculate Linear Field Distribution

For first approximation in CdS:
- Field increases linearly from bulk value to junction interface
- Slope relates directly to trap density
- Similar to Schottky barrier behavior

Field profile:
```
F(x) = F_bulk + (F_max - F_bulk) * (x/d)
```

Where:
- F_bulk: Field in bulk CdS
- F_max: Maximum field at junction interface
- d: CdS layer thickness

### 2. Determine Bias-Dependent Field Values

| Bias Condition | Typical Field at Junction |
|----------------|---------------------------|
| Forward bias | ~20 kV/cm |
| Near open circuit | Rapidly increasing |
| Reverse bias | Can reach tunneling values |

### 3. Identify Field Quenching Onset

**Critical threshold**: F ≥ 23 kV/cm

At this field:
- Barrier lowering δE = 2kT achieved
- Holes released from slow copper recombination centers
- Field quenching process initiates

### 4. Model Field Quenching Cascade

When field reaches threshold:

1. **Initiation** (at 23 kV/cm):
   - Frenkel-Poole excitation releases trapped holes
   - Holes freed from Coulomb-attractive copper centers

2. **Cascade effect**:
   - Released holes increase valence band hole density
   - Holes captured by fast recombination centers
   - Enhanced recombination with conduction electrons

3. **Progressive quenching** (above 50 kV/cm):
   - Electron density markedly reduced
   - Photoconductivity quenched

### 5. Analyze Band Alignment Changes

Field quenching consequences:
- Distance between Fermi level and conduction band increases
- Conduction band at interface moves up with changing bias
- Band discontinuity develops (apparent electron affinity change)
- Measured work function change: ~0.25 eV at Au contact

### 6. Predict Junction Behavior

**Forward bias to open circuit**:
- Conduction bands connected initially
- Gradual separation develops
- Field quenching begins affecting carrier density

**Reverse bias**:
- Significant band separation
- Field quenching pronounced
- Electron density substantially reduced at interface

## Output
- Field distribution profile F(x)
- Field quenching threshold status
- Modified electron density at interface
- Band alignment changes

## Constraints
- Requires Coulomb-attractive hole centers for field quenching
- Linear approximation does not account for field quenching at high fields
- Density of copper centers must be sufficient for observable effect

## Key Parameters

| Parameter | Typical Value |
|-----------|---------------|
| Trap density | 10^17 cm^-3 |
| Field quenching onset | 23 kV/cm |
| Marked quenching | >50 kV/cm |
| Work function change | 0.25 eV |
| Photoconductive electron density | ~10^18 cm^-3 (under optical excitation) |