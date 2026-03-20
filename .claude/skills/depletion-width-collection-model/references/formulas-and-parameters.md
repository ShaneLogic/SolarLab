# Depletion Width Collection Model - Reference

## Complete Formula Set

### Equation 35.2: Field Distribution

The electric field distribution in the depletion region:

```
F(x) = (2eNA/εε₀)^(1/2) × (VB - V)^(1/2) × [1 - (2x/W)]
```

**Parameters:**
| Symbol | Description | Units |
|--------|-------------|-------|
| F(x) | Electric field at position x | V/cm |
| NA | Acceptor density | cm⁻³ |
| VB | Built-in potential | V |
| V | Applied voltage | V |
| W | Depletion width | cm |
| ε | Relative permittivity | dimensionless |
| ε₀ | Vacuum permittivity (8.854×10⁻¹⁴) | F/cm |
| e | Elementary charge (1.602×10⁻¹⁹) | C |

### Equation 35.3: Voltage-Dependent Depletion Width

```
W(V) = [2εε₀(VD - V)/eNA]^(1/2)
```

**Parameters:**
| Symbol | Description | Units |
|--------|-------------|-------|
| W(V) | Depletion width at voltage V | cm |
| VD | Diffusion voltage | V |
| V | Applied voltage | V |

### Equation 35.4: Collection Efficiency

Basic form:
```
ηc(V) = 1 - exp(-αW(V)/(1 + αL))
```

Extended form with field-dependent collection:
```
ηc(V) = (1 - exp(-αW(V)/(1 + αL))) × (μF₀(V)/(S + μF₀(V)))
```

**Parameters:**
| Symbol | Description | Units |
|--------|-------------|-------|
| ηc(V) | Collection efficiency at voltage V | dimensionless |
| α | Effective optical absorption constant | cm⁻¹ |
| L | Diffusion length | cm |
| μ | Carrier mobility | cm²/(V·s) |
| F₀(V) | Field at junction interface | V/cm |
| S | Surface recombination velocity | cm/s |

## CdTe Material Parameters

Typical values for CdTe solar cells:

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| NA | 10¹⁴ - 10¹⁶ cm⁻³ | Acceptor density in p-type region |
| VD | 0.8 - 1.2 V | Diffusion voltage |
| α | 10³ - 10⁵ cm⁻¹ | Weighted absorption coefficient |
| L | 0.1 - 1.0 μm | Minority carrier diffusion length |
| ε | 10.2 | Relative permittivity of CdTe |

## Physical Interpretation

### Depletion Width Dependence

The depletion width increases as reverse bias increases (V decreases):
- At V = 0: Maximum depletion width under short-circuit
- At V = VD: Depletion width approaches zero (forward bias limit)

### Collection Efficiency Components

1. **Absorption term**: `1 - exp(-αW(V))`
   - Represents carriers generated within depletion region
   - Higher α or W increases collection

2. **Diffusion term**: `1/(1 + αL)` factor
   - Accounts for carriers diffusing from quasi-neutral region
   - Longer L improves collection

3. **Field-dependent term**: `μF₀(V)/(S + μF₀(V))`
   - Models competition between drift collection and surface recombination
   - Higher field improves collection

## Parameter Extraction Method

When parameters are unknown, extract from I-V measurements:

1. Measure I-V characteristic under known illumination
2. Fit the voltage-dependent collection model to the data
3. Optimize four parameters: α, NA, L, VD
4. Validate by comparing modeled and measured quantum efficiency

## Assumptions and Limitations

- Field distribution approximated as simple pn-homojunction
- Single-sided abrupt junction assumed
- Constant doping in each region
- No series resistance effects in collection model
- Valid for voltages below breakdown and above flat-band condition