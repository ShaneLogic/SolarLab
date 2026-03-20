---
name: energy-transfer-particle-collisions
description: Calculate energy transfer from impacting particles to lattice atoms and analyze collision trajectories. Use when analyzing radiation damage, calculating maximum energy transfer in collisions, or studying particle-lattice interactions in semiconductor materials.
---

# Energy Transfer in Particle Collisions

## When to Use
- Particle impacts lattice with sufficient energy/momentum
- Calculating radiation damage from particle bombardment
- Analyzing energy transfer to lattice atoms
- Studying collision trajectories and defect formation
- Working with impacting electrons, nuclei, or ions

## Prerequisites
- Sufficiently large energy and momentum of impacting particle
- Does NOT apply if energy/momentum is too low to create defects

## Calculate Maximum Transferred Energy (E_max)

### Formula
```
E_max = (4 × E_i × M_i × M) / (M_i + M)²
```

### Where
- E_i = Energy of impacting particle
- M_i = Mass of impacting particle
- M = Mass of lattice atom

## Calculate Angle-Dependent Energy Transfer

### Formula (Eq. 11.8)
```
E = E_max × sin²(θ/2)
```

### Where
- θ = Scattering angle in center-of-mass reference frame

## Determine Collision Type

Based on the angle and alignment of the collision:

### Head-on Collision
- θ = π
- The impacting particle is reflected back

### Focusing Collision
- Impacting angle almost aligned with low index crystallographic direction
- Low index direction = closely packed atoms
- Target atom moves closer to low index direction
- Process continues with more alignment of forward motion
- Continues until crystal defect, opposite surface reached
- At surface: reflection, radiation damage, or rejection occurs

### Large Angle Collision
- Angle with low index direction is larger
- Target ion moves nearly perpendicular to path of incident particle

## Key Variables
| Variable | Description |
|----------|-------------|
| E_max | Maximum transferred energy |
| E_i | Energy of impacting particle |
| M_i | Mass of impacting particle |
| M | Mass of lattice atom |
| θ | Scattering angle in center-of-mass reference frame |
| E | Energy transferred (angle-dependent) |