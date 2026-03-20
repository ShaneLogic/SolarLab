---
name: deep-level-center-modeling
description: Apply square well potential models to analyze and classify deep level centers (transition metal impurities, vacancies) with strong core potentials and localized wavefunctions when hydrogenic effective mass approximation is insufficient.
---

# Deep Level Center Modeling

## When to Use

Apply this skill when:

- Defect has a strong core potential and short-range potential
- Wavefunction of ground state remains localized close to defect core
- Defect connects to BOTH conduction and valence bands (extending through entire Brillouin zone)
- Defect does not follow one specific band under perturbation (alloying/pressure)
- Hydrogenic effective mass approximation fails to describe the defect

## Core Workflow

### 1. Identify Deep Center Characteristics

Confirm the defect exhibits these fundamental properties:
- Ground state wavefunction is highly localized near the defect core
- Energy levels are connected to both conduction and valence bands
- Behavior is not tied to a single band under alloying or pressure perturbations
- Defect is best described by a short-range potential model

### 2. Apply Square Well Potential Model

Use a 1D rectangular well model with parameters:
- **V0**: Depth of the potential well (in energy units)
- **a**: Half-width of the potential well (in length units)
- **nq**: Quantum number

Key relationships:
- Eigenstate energies increase quadratically with nq: E ~ nq²
- This contrasts with hydrogen-like defects where eigenstates decrease as 1/nq²
- Use electron rest mass (electron is close to center)

### 3. Evaluate Coulomb Tail Effects

For charged deep centers:
- The long-range Coulomb potential tail dominates higher excited states
- Higher excited states become hydrogen-like (shallow-like)
- Ground state shows deep level behavior when short-range potential is large (>10 eV)
- Expected spectrum: One or several deep levels, followed by hydrogen-like shallow levels near bands

### 4. Assess Core Potential Depth

Use atomic electronegativity as an indicator for:
- The depth of the square well
- The strength of the core potential

## Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| V0 | Energy | Depth of the potential well |
| a | Length | Half-width of the potential well |
| nq | Integer | Quantum number |

## Output Interpretation

The skill produces:
- Eigenstate energy values calculated from the square well model
- Classification of defect as deep vs shallow based on localization and potential characteristics
- Energy level spectrum structure (deep levels followed by shallow hydrogen-like levels for charged centers)