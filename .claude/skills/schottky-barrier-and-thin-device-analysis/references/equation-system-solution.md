### Numerical Solution Strategy

#### Boundary Conditions
Required six conditions:
- nb: electron density at boundary
- Fb: electric field at boundary  
- ψb: potential at boundary
- pb: hole density at boundary
- jnb: electron current at boundary
- jpb: hole current at boundary

#### Iteration Approach
1. Assume initial carrier distributions
2. Solve transport equations for currents
3. Update charge density
4. Solve Poisson equation for field
5. Update potentials and carrier densities
6. Repeat until convergence

#### Mixed Condition Strategy
- Some boundaries specify density (Dirichlet)
- Some specify current (Neumann)
- Use shooting method or relaxation

### Debye Length Calculation
```
LD = √(εkT/(q²n10))
```
Critical for assessing ambipolar approximation validity.

### Current Density Reference Values
- Main electron current: ~40 A/cm²
- GR current contribution: ~20 μA/cm²
- Hole bulk current: negligible (~10⁻⁵ × jni)