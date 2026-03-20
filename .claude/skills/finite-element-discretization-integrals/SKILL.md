---
name: finite-element-discretization-integrals
description: Calculate finite element discretization integrals (A.1, A.2, A.3) for converting governing equations into a system of ODEs. Use when performing spatial discretization using the finite element approach for drift-diffusion models or similar PDE systems requiring matrix assembly.
---

# Finite Element Discretization Integrals

## When to Use
Apply this skill when:
- Performing spatial discretization of governing equations using the finite element method
- Converting PDEs (specifically Eqs. 11-16) into a system of ordinary differential equations
- Assembling discretization matrices for numerical solution
- Working with drift-diffusion models that require integral evaluation over basis functions

## Procedure

### 1. Define Basis Functions
- Use φ_i(x) as the basis "hat" or "tent" function (see Eq. 19 in references)
- Apply prime notation (') to denote derivatives with respect to x
- Set index range: i, j, k range from 0 to N

### 2. Calculate Required Integrals
Compute the following three integral types for the discretization matrix:

- **Integral A.1**: Required for primary matrix components
- **Integral A.2**: Required for secondary matrix components
- **Integral A.3**: Required for coupling terms and additional matrix elements

Refer to `references/integral-forms.md` for the specific mathematical expressions and evaluation methods.

### 3. Assemble Matrix Components
- Organize computed integrals into the system matrix structure
- Ensure proper index mapping for i, j, k
- Verify matrix properties match the governing equations

## Key Variables
- **φ_i(x)**: Basis "hat" function - piecewise linear interpolation function
- **φ_i'(x)**: Derivative of basis function with respect to x
- **i, j, k**: Mesh indices (range: 0 to N)
- **N**: Number of mesh intervals

## Output
Matrix components for the system of ODEs resulting from spatial discretization.