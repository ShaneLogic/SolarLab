---
name: drift-diffusion-current-analysis
description: Analyze and calculate drift and diffusion current components in semiconductor junctions. Use this skill when examining individual current mechanisms in bulk semiconductor regions or at junction interfaces, particularly when distinguishing between field-driven drift current and gradient-driven diffusion current.
---

# Drift and Diffusion Current Analysis

## When to Use
Apply this skill when:
- Analyzing individual current components (drift vs diffusion) in a semiconductor junction
- Calculating current density in bulk semiconductor regions
- Examining current behavior at doping interfaces
- Verifying equilibrium conditions in unbiased junctions
- Investigating how applied bias affects current components

## Prerequisites
Before applying this skill, ensure you have:
- Electric field profile (dψ/dx) for the region
- Carrier density gradient information
- Doping profile of the junction
- Applied bias voltage (if any)

## Core Workflow

### 1. Calculate Drift Current in Bulk Regions

For regions 1 and 2 (bulk semiconductor regions):

```
jn,drift = σn * dψ/dx
```

Where:
- `jn,drift` = drift component of electron current density
- `σn` = electron conductivity
- `dψ/dx` = gradient of electrostatic electron potential

**Action**: Compute the drift current density using the electric field profile and material conductivity.

### 2. Analyze Diffusion Current at Doping Interface

At the doping interface between regions:

- Diffusion current becomes very large (typically ~10^4 A/cm²)
- Diffusion current changes little with applied bias
- Magnitude is driven by carrier density gradient

**Action**: Identify the interface location and evaluate the diffusion current magnitude based on the doping gradient.

### 3. Compare Drift and Diffusion Distributions

Observe the relationship:

- Drift current distribution matches diffusion current in shape
- Drift current is shifted parallel to the current axis
- The shift results in a constant net current

**Action**: Plot or compare the spatial distributions of both current components to verify their relationship.

### 4. Verify Equilibrium Condition

For zero bias (V = 0):

- Total current jn = 0
- Drift and diffusion currents are equal in magnitude
- Currents cancel each other out: jn,drift = -jn,diffusion

**Action**: Confirm that at equilibrium, the drift and diffusion components sum to zero.

## Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| jn,drift | Current Density | Drift component of current |
| jn,diffusion | Current Density | Diffusion component of current |
| ψ | Potential | Electrostatic electron potential |
| dψ/dx | Field | Electric field (potential gradient) |
| σn | Conductivity | Electron conductivity |

## Output Interpretation

The analysis yields:
- Magnitude of drift current in bulk regions
- Magnitude of diffusion current at the interface
- Spatial distribution of both components
- Net current (sum of components)
- Verification of equilibrium condition

## Common Patterns

- **Bulk regions**: Drift current dominates due to electric field
- **Interface region**: Diffusion current peaks due to sharp doping gradient
- **Applied bias**: Primarily affects the balance between components, not individual magnitudes
- **Equilibrium**: Perfect cancellation of drift and diffusion