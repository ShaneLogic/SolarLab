---
name: depletion-width-collection-model
description: Calculate voltage-dependent collection efficiency in pn-junction solar cells using depletion width modeling. Use when modeling carrier collection in pn-homojunction or CdTe-based solar cells where material parameters (acceptor density, built-in potential, absorption coefficient, diffusion length) are known or can be approximated.
---

# Voltage-Dependent Depletion Width Collection Model

## When to Use

Apply this skill when:
- Modeling pn-homojunction solar cells with voltage-dependent depletion width
- Analyzing CdTe-based solar cells with known material parameters
- Calculating collection efficiency that varies with applied voltage
- Need to relate I-V characteristics to carrier collection physics

## Prerequisites

Ensure the following parameters are available:
- **NA**: Acceptor density (cm⁻³)
- **VB**: Built-in potential (V)
- **VD**: Diffusion voltage (V)
- **α**: Effective optical absorption constant (cm⁻¹)
- **L**: Diffusion length (cm)

## Execution Steps

### 1. Calculate Voltage-Dependent Depletion Width

Compute the depletion width as a function of applied voltage:

```
W(V) = [2εε₀(VD - V)/eNA]^(1/2)
```

Where:
- ε = relative permittivity of the material
- ε₀ = vacuum permittivity
- e = elementary charge

### 2. Calculate Field Distribution

Model the electric field in the depletion region:

```
F(x) = (2eNA/εε₀)^(1/2) × (VB - V)^(1/2) × [1 - (2x/W)]
```

### 3. Calculate Basic Collection Efficiency

Compute the collection efficiency accounting for absorption:

```
ηc(V) = 1 - exp(-αW(V)/(1 + αL))
```

### 4. Calculate Photo-Electric Collection Efficiency

For complete collection efficiency including field-dependent effects:

```
ηc(V) = (1 - exp(-αW(V)/(1 + αL))) × (μF₀(V)/(S + μF₀(V)))
```

Where:
- μ = carrier mobility
- F₀(V) = field at the junction interface
- S = surface recombination velocity

### 5. Parameter Extraction for CdTe Cells

When working with CdTe solar cells:
- Use weighted CdTe absorption coefficient for α approximation
- Extract parameters from measured I-V characteristics under sunlight
- Four fitting parameters remain: α, NA, L, VD

## Output

Returns collection efficiency values that account for:
- Optical absorption in the depletion region
- Carrier diffusion from quasi-neutral regions
- Voltage-dependent depletion width modulation
- Field-dependent carrier collection at interfaces