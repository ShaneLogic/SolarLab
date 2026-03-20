---
name: magnetic-induction-transport-theory
description: Apply mathematical treatment for magnetic induction in transport analysis. Use when analyzing Hall effect, magnetoresistance, or transport in magnetic fields where the Boltzmann equation needs to account for magnetic field effects.
---

# Magnetic Induction Transport Theory

Use this skill when analyzing transport phenomena in the presence of magnetic fields, particularly when:
- Calculating Hall effect or magnetoresistance
- Solving the Boltzmann equation with magnetic induction
- Determining distribution function deformation due to magnetic fields

## Determine Perturbation Level

1. Check if magnetic induction B is small enough for perturbation approximation
   - Small B: First-order terms sufficient
   - Large B: Higher-order terms needed (observable magnetoelectric effects)

2. Verify prerequisites:
   - Electric field present
   - Thermal gradients may be present
   - Magnetic induction B applied

## Apply Mathematical Treatment

1. For small perturbations:
   - Consider only first-order terms
   - Include term proportional to gradient of δf (perturbation to distribution function)
   - Solve resulting equation by iteration (Eq. 16.5)

2. For spherical equi-energy surfaces near conduction band bottom:
   - Express electron velocity: v = ℏk/mn
   - Apply Eq. (16.6) with abbreviations from Eq. (16.7)

3. Expand distribution function:
   - f = f(r, k, T(r), B, t)
   - Account for electric, thermal, AND magnetic field influences

## Interpret Results

The deformed distribution function causes changes in transport properties:
- Electrical currents modified
- Thermal currents modified
- Interacting fields produce observable magnetotransport effects

## Key Variables

- **B**: Magnetic induction field (vector)
- **δf**: Perturbation to distribution function (function)
- **v**: Electron velocity (vector)
- **k**: Wave vector (vector)
- **mn**: Effective electron mass (scalar)

## References

See `references/magnetic-induction-details.md` for detailed derivations and equations.