---
name: inhomogeneous-material-transport
description: Select and apply extended transport equations for semiconductor devices with position-dependent material parameters (inhomogeneities). Use this when material properties such as doping, bandgap, or mobility vary based on position within the device structure, where standard transport equations are insufficient.
---

# Inhomogeneous Material Transport Analysis

## When to Use
- Material parameters vary with position in the device structure
- Standard transport equations are insufficient
- Doping profiles are non-uniform
- Bandgap grading exists
- Composition varies spatially (e.g., alloy grading)

## Execution Procedure

### 1. Assess Device Material Inhomogeneity

Determine if material parameters change based on position within the device structure:
- Check doping profiles
- Verify bandgap uniformity
- Examine mobility variations
- Identify composition gradients

### 2. Identify Inhomogeneity Condition

The presence of position-dependent parameters indicates 'inhomogeneities'.

### 3. Select Extended Transport Equations

**DO NOT use standard transport equations** for inhomogeneous materials.

**Instead:** Select a set of appropriately extended transport equations capable of dealing with such inhomogeneities.

### 4. Verify Equation Capability

Ensure the chosen equations:
- Explicitly account for position-dependent parameters
- Can handle gradient terms arising from inhomogeneities
- Are formulated for non-uniform materials

## Key Considerations

### Standard vs Extended Equations

**Standard transport equations** assume:
- Uniform material properties
- Position-independent parameters

**Extended transport equations** include:
- Additional terms for parameter gradients
- Position-dependent coefficients
- Modified boundary conditions

### Common Inhomogeneity Types

1. **Doping gradients**: Nd(x) or Na(x) vary with position
2. **Bandgap grading**: Eg(x) changes (e.g., in heterostructures)
3. **Mobility variations**: μ(x) depends on position
4. **Alloy composition**: Changing material composition along device

## Verification Step

After selecting extended equations, verify:
- All inhomogeneity terms are included
- Boundary conditions account for parameter variations
- Numerical stability for large gradients