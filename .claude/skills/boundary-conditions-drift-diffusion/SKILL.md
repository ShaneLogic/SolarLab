---
name: boundary-conditions-drift-diffusion
description: Apply appropriate boundary conditions for drift-diffusion simulations, specifically Dirichlet conditions for infinite ion reservoirs at system boundaries.
---

# Boundary Conditions for Drift-Diffusion Simulations

Use this skill when you need to:
- Define boundary conditions for semiconductor device simulations
- Model systems with infinite ion reservoirs
- Apply Dirichlet boundary conditions to maintain constant ion density
- Set up electrolyte or electrode boundary conditions

## Dirichlet Boundary Condition for Infinite Ion Reservoir

**When to apply:** System boundary has an infinite reservoir of ions

**Applicable Objects:**
- System boundaries (electrodes)
- Electrolytes
- Ion reservoirs

**Procedure:**
1. Identify the system boundary within the physical model
2. Determine if an 'infinite reservoir of ions' exists at this boundary
   - Example: An electrolyte is a typical example
3. If infinite reservoir condition is met:
   - Impose a Dirichlet boundary condition
   - Define this condition to maintain constant ion density

**Key Concepts:**

**Dirichlet Boundary Condition:**
- Specifies the value of a variable at the boundary
- In this case: maintains constant ion density
- Alternative to other boundary condition types

**Infinite Reservoir:**
- Source of ions at the boundary
- Considered infinite in capacity
- Acts as a fixed reference for ion concentration

**Output:**
- Constant ion density enforced at the boundary
- System can exchange ions without changing reservoir concentration