---
name: critical-thickness-superlattices
description: Determine if layer thickness is safe for ultrathin superlattice formation when depositing alternating layers with substantial lattice mismatch. Use when designing heteroepitaxial structures, superlattices, or quantum wells where preventing dislocation formation is critical for material quality.
---

# Critical Thickness for Ultrathin Superlattices

## When to Use
- Designing or analyzing ultrathin superlattices with alternating layers of different materials
- Depositing heteroepitaxial layers with substantial lattice mismatch
- Evaluating whether a proposed layer thickness will result in dislocation-free growth
- Working with material pairs like Si/Ge, GaAs/InAs, or other lattice-mismatched systems

## Required Information
Before applying this skill, determine:
- Lattice constants of both materials
- Thickness planned for each layer

## Procedure

### 1. Calculate Lattice Mismatch
Determine the lattice mismatch percentage between the two materials based on their lattice constants.

### 2. Identify Critical Length
Find the critical length (maximum safe thickness) for the specific lattice mismatch. The critical length decreases as lattice mismatch increases.

### 3. Compare Layer Thickness to Critical Length
Check if each layer's thickness is below the critical length:
- If thickness < critical length: Safe - can form dislocation-free superlattice
- If thickness ≥ critical length: Unsafe - risk of dislocation formation

## Variables
- **Lattice Mismatch**: Percentage difference in lattice constants between materials
- **Critical Length**: Maximum thickness before dislocation formation occurs

## Output Interpretation
- **Safe**: Layer thickness is below critical length; ultrathin superlattice or artificial compound formation is possible without dislocations
- **Unsafe**: Layer thickness exceeds or equals critical length; dislocations will form, compromising material quality

## Notes
- As lattice mismatch increases, the critical length decreases significantly
- This principle enables the creation of artificial new compounds not found in nature