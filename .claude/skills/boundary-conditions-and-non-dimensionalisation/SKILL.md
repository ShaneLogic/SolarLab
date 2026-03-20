---
name: boundary-conditions-and-non-dimensionalisation
description: Define boundary conditions for PDE drift-diffusion models and perform non-dimensionalisation of variables for numerical stability. Use when setting up the solution space, scaling variables for numerical solution, or establishing initial/boundary conditions for ion vacancy and charge carrier transport models.
---

# Boundary Conditions and Non-dimensionalisation

Use this skill when defining the solution space for PDE drift-diffusion models or scaling variables for numerical stability in perovskite solar cell simulations.

## Apply Boundary Conditions

### At x = 0 (ETL interface):
1. **Anion vacancies**: No flux condition - F_P = 0
2. **Potential**: Specified as φ = V_ap (applied voltage)
3. **Electron concentration**: n = n_b (determined by band offsets)
4. **Hole current**: j_p = R_l (surface recombination rate at left interface)

### At x = b (HTL interface):
1. **Anion vacancies**: No flux condition - F_P = 0
2. **Potential**: Specified as φ = V_bi (built-in potential)
3. **Hole concentration**: p = p_b (determined by band offsets)
4. **Electron current**: j_n = -R_r (surface recombination rate at right interface)

## Define Initial Conditions

Set initial profiles for:
- n(x,0) = n_i(x) - electron density
- p(x,0) = p_i(x) - hole density
- P(x,0) = P_i(x) - vacancy density

## Perform Non-dimensionalisation (Scaling)

### Spatial Scaling
- x* = x / b
- Where b = perovskite layer width

### Time Scaling
- t* = t / τ_ion
- Where τ_ion = L_d * b / D_+ (characteristic timescale for ion motion)

### Density Scaling
- n* = n / λ_0
- p* = p / λ_0
- P* = P / N_0
- Where λ_0 = F_ph * b (characteristic density from photon flux)

### Potential Scaling
- φ* = φ / V_T
- Where V_T = thermal voltage

### Key Length Scale
- Debye length: L_d = sqrt(ε * V_T / (2 * q^2 * N_0))

## Key Variables

- **V_ap**: Applied voltage
- **V_bi**: Built-in potential
- **R_l, R_r**: Surface recombination rates (left/right interfaces)
- **τ_ion**: Characteristic timescale for ion motion
- **L_d**: Debye length

## Output

A complete set of constraints defining the system edges and normalized variables for numerical solution.