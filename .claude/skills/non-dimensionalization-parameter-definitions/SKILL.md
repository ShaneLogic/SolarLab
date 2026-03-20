---
name: non-dimensionalization-parameter-definitions
description: Defines and applies non-dimensionalization parameters for physical systems involving charged species and carrier diffusivity. Use when analyzing electrochemical systems, plasma physics, or any system requiring dimensionless parameters based on Debye length and carrier diffusivity.
---

# Non-dimensionalization Parameter Definitions

## When to Use
Apply this skill when:
- Analyzing physical systems with charged species
- Working with systems characterized by carrier diffusivity
- Converting dimensional models to dimensionless form
- Characterizing timescales of electronic and ionic motion

## Prerequisites
Before applying this skill, identify:
- The most populous charged species in the system
- The typical carrier diffusivity value

## Procedure

### 1. Define Scaling Parameters

**Typical Carrier Diffusivity (D̂)**
- Define the typical carrier diffusivity parameter, denoted as D̂
- This represents the characteristic diffusion rate in the system

**Debye Length (Ld)**
- Define the Debye length parameter, denoted as Ld
- Calculate Ld based on the **most populous charged species** present in the system
- Note: The Debye length calculation is species-dependent

### 2. Apply Non-dimensionalization

Use the defined parameters (D̂, Ld) to generate dimensionless quantities for the system.

### 3. Interpret Dimensionless Parameters

**General Rule**: Most dimensionless parameters are self-evident from their definitions.

**Specific Parameter - ν**:
- ν is defined as the ratio of the timescales for electronic and ionic motion
- This parameter characterizes the relative speed of electronic vs ionic processes

## Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| D̂ | Physical Parameter | Typical carrier diffusivity |
| Ld | Physical Parameter | Debye length (based on most populous charged species) |
| ν | Dimensionless Parameter | Ratio of timescales for electronic and ionic motion |

## Important Notes
- The choice of dominant species for Debye length calculation significantly impacts results
- Common dominant species include ion vacancies, electrons, or ions depending on the system