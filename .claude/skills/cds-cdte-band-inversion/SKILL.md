---
name: CdS/CdTe Band Inversion Analysis
description: Analyze and model p-type inversion at CdS/CdTe interfaces under strong field quenching conditions. Use this skill when working with CdS/CdTe solar cell band diagrams, investigating leakage current prevention through band disconnection, or modeling interface dipole moment changes under photoconductivity effects.
---

# CdS/CdTe Band Inversion Analysis

## When to Use
- Modeling CdS/CdTe interface behavior under field quenching
- Analyzing band structure changes to prevent electron back-diffusion
- Investigating dipole moment changes at semiconductor interfaces
- Designing leakage current mitigation strategies in thin-film solar cells

## Prerequisites
- Strong field quenching condition present
- Open circuit condition available
- CdS/CdTe interface band diagram accessible

## Procedure

### 1. Induce Strong Field Quenching
- Apply conditions that cause CdS layer to invert to p-type near the interface
- Verify the inversion region has formed

### 2. Analyze Fermi Level Shift
- In the p-type region, the Fermi level moves closer to the valence band
- Document the shift magnitude relative to the original position

### 3. Model Band Bending (Open Circuit Condition)
- **Constraint**: Fermi level must remain horizontal (constant)
- Both valence and conduction bands curve upward in the field-quenched region
- Verify band continuity across the interface

### 4. Evaluate Band Disconnection
- Compare forward bias state vs. quenched state:
  - **Forward Bias**: CdS conduction band connects to CdTe conduction band
  - **Quenched State**: CdS conduction band disconnects from CdTe conduction band
- Assess the effectiveness of electron back-diffusion limitation

### 5. Calculate Dipole Moment Change
- Determine the band offset from the dipole moment shift
- Reference experimental evidence: Schottky barrier dipole moments can change with photoconductivity

## Output
- Modified band diagram showing band disconnection
- Analysis of leakage prevention effectiveness
- Dipole moment change quantification

## Constraints
- Fermi level must remain horizontal throughout the analysis
- Field quenching strength must be sufficient for inversion