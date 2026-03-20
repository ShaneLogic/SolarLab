---
name: carrier-statistics-modeling
description: Apply appropriate statistical models (Boltzmann, Fermi-Dirac, Gaussian) to calculate carrier densities and current densities in semiconductor materials. Use when simulating charge transport in perovskite solar cells, modeling transport layers (ETL/HTL), or working with organic vs inorganic materials where non-Boltzmann statistics may be required.
---

# Carrier Statistics Modeling

Use this skill when calculating carrier densities (n, p) and current densities (j_n, j_p) in semiconductor devices, particularly when:
- Simulating perovskite solar cells with organic transport layers
- Material properties suggest Boltzmann approximation may be invalid
- High doping concentrations or strong degeneracy conditions exist
- Comparing ordered (crystalline) vs disordered (organic) materials

## General Carrier Density Calculation

For any statistical model, calculate carrier density using the statistical integral:

```
n = g_c * S((E_fn - E_c) / k_B T)
p = g_v * S((E_v - E_fp) / k_B T)
```

Where:
- `S` is the statistical integral (model-dependent)
- `g_c`, `g_v` are effective density of states
- `E_fn`, `E_fp` are quasi-Fermi levels
- `E_c`, `E_v` are band edge energies
- Account for band bending: `E_c,v = const - q * φ`

## Select Statistical Model

### 1. Boltzmann Approximation (Default)
- **When to use**: Low carrier densities, quasi-Fermi levels > 3kT from band edge
- **Statistical integral**: `S(xi) = exp(xi)`
- **Inverse**: `S^{-1}(y) = ln(y)`

### 2. Parabolic Band Model (Fermi-Dirac)
- **When to use**: Ordered crystalline/inorganic semiconductors with high doping
- **Statistical integral**: Fermi-Dirac integral `F_1/2(xi)`
- **Formula**: `F(xi) = (2/√π) ∫₀^∞ √(η)/(1 + exp(η - xi)) dη`
- **Boltzmann valid for**: `xi < -3` or `n < 2 * g_c`

### 3. Gaussian Band Model
- **When to use**: Organic/disordered materials, transport layers with hopping transport
- **Statistical integral**: Gauss-Fermi integral `G_s(xi)`
- **Reference energies**: LUMO (E_L), HOMO (E_H) - no defined band edges
- **Disorder parameter**: `s` (dimensionless), `σ = s * k_B T`
- **Boltzmann form**: `S(xi) = exp(xi + s²/2)`

## Configure Non-Boltzmann Statistics

When simulating with non-Boltzmann statistics, specify parameters in parameters.m:

```matlab
% ETL parameters
SE = 'F12';      % Statistical integral for ETL
SEinv = 'F12inv'; % Inverse statistical integral

% HTL parameters  
SH = 'G';        % Statistical integral for HTL
SHinv = 'Ginv';  % Inverse statistical integral
```

**Default behavior**: If not specified, uses Boltzmann approximation (`exp`/`ln`)

## Calculate Current Density

Use the generalized drift-diffusion equation:

```
j_n = μ_n * k_B T * d/dx[ n * S^{-1}(n/g_c) - (q/k_B T) * dφ/dx ]
j_p = μ_p * k_B T * d/dx[ p * S^{-1}(p/g_v) + (q/k_B T) * dφ/dx ]
```

In Boltzmann limit, this reduces to standard form:
```
j_n = q * μ_n * n * E + k_B T * μ_n * dn/dx
```

## Apply Transport Layer Boundary Conditions

At ETL and HTL interfaces, apply continuity conditions with equilibrium ratios:

```
Carrier density continuity: n|_x=0- = n|_x=0+
Current density continuity: j_n|_x=0- = j_n|_x=0+
```

Calculate equilibrium ratios to handle non-Boltzmann statistics across interfaces.