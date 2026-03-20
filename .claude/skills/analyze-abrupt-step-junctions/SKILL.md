---
name: analyze-abrupt-step-junctions
description: Use this skill when analyzing the electrical behavior of one-carrier abrupt step-junctions, such as nn+-junctions or semiconductor barriers. It applies when you need to model the physics of these junctions using governing transport and Poisson equations to determine electron density, electric field, and potential distributions.
---

# Analyze Abrupt Step-Junctions

Use this workflow to model the electrical behavior of one-carrier abrupt step-junctions. Since these systems involve non-linear differential equations that cannot be solved in closed form, numerical integration is required.

## Procedure

1. **Establish the Governing Equations**
   Define the system of three simultaneous non-linear differential equations that determine the electrical behavior:
   - Transport/Potential Relation
   - Poisson's Equation
   - Current Continuity Equation
   
   Refer to `governing-equations.md` for the specific mathematical forms and the expanded set of four equations (Eqs. 25.6 - 25.9).

2. **Define Boundary Conditions**
   Apply boundary conditions assuming the "thick slices" scenario (far from the interface):
   - Set electron density ($n$) and electric field ($F$) to constant values at the boundaries.
   - Approximate charged donor density ($p_d$) as equal to total donor density ($N_d$): $p_{d1} \approx N_{d1}$ and $p_{d2} \approx N_{d2}$.
   - Ensure shallow, fully ionized donor assumptions are met.
   
   *Note: Six boundary conditions are required for physically meaningful solutions.*

3. **Perform Numerical Integration**
   - Use a numerical solver to integrate the system.
   - Do not attempt closed-form integration, as it is impossible for this non-linear system.

4. **Extract Results**
   Generate and analyze the following solution curves across the junction:
   - Electron density distribution $n(x)$
   - Electric field distribution $F(x)$
   - Potential distribution