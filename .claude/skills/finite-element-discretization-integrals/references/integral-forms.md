# Integral Forms Reference

## Basis Function Definition (Eq. 19)

The "hat" or "tent" function φ_i(x) is a piecewise linear basis function defined on a mesh.

**Key properties:**
- Continuous across element boundaries
- Local support: non-zero only on adjacent elements
- φ_i(x_j) = δ_{ij} (equals 1 at node i, 0 at other nodes)
- Derivative φ_i'(x) is piecewise constant

*Note: The complete mathematical expression is defined in Eq. 19 of the source material.*

## Required Integrals for Discretization

The following integrals form the core components of the discretization matrix:

### Integral A.1
- **Purpose**: Primary matrix component
- **Description**: Specific integral form required for the main system matrix
- **Reference**: Depicted in source figures

### Integral A.2
- **Purpose**: Secondary matrix component
- **Description**: Specific integral form required for additional matrix elements
- **Reference**: Depicted in source figures

### Integral A.3
- **Purpose**: Coupling terms
- **Description**: Specific integral form handling variable coupling
- **Reference**: Depicted in source figures

## Governing Equations

These integrals apply to the following governing equations:
- Eq. 11
- Eq. 12
- Eq. 13
- Eq. 14
- Eq. 15
- Eq. 16

## Constraints
- These integral forms are specific to the drift-diffusion model presented
- Requires completed spatial discretization step
- Basis function definition must be established before integral evaluation

## Evaluation Notes
- Integrals are essential components of the discretization matrix assembly
- The specific mathematical expressions are provided in the source figures
- Proper handling of boundary conditions is required during evaluation