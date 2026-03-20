---
name: memory-erasing-collision-fraction
description: Use when calculating the fraction of scattering events that result in total angle randomization (memory-erasing collisions). Apply this formula to determine what proportion of all scatterings actually randomize the particle's direction, as referenced in Equation (17.38).
---

# Memory-Erasing Collision Fraction Calculation

## When to Use
- Calculating events that result in total angle randomization
- Determining the fraction of collisions that erase particle "memory" of initial direction
- Analyzing collision statistics in particle systems
- Evaluating randomization efficiency in scattering processes

## Core Procedure

### Define the Objective
Identify the number of collisions that "totally randomize the angle after collision"—these are defined as **memory-erasing collisions**.

### Apply the Formula
**Fraction = (1 - cosθ)**

Where:
- **θ**: Scattering angle of the collision
- **Fraction**: Proportion of all scatterings that are memory-erasing

### Interpret the Result
Only this fraction of all scattering events actually counts as memory-erasing collisions. The remaining fraction preserves some memory of the initial direction.

## Key Notes
- This applies specifically to collisions that completely randomize the angle
- Reference: Equation (17.38)
- The formula yields a value between 0 and 1
- When θ = 180° (cosθ = -1), Fraction = 2 (theoretical maximum)
- When θ = 90° (cosθ = 0), Fraction = 1 (complete randomization)