---
name: parasitic-resistances-modeling
description: Incorporate parasitic resistances and band offsets into perovskite solar cell simulations. Use when fitting experimental J-V curves, modeling non-ideal device behavior, accounting for contact resistances, or adjusting for circuit losses in device performance.
---

# Parasitic Resistances Modeling

Use this skill when:
- Fitting simulation results to experimental J-V curves
- Modeling device non-idealities (series resistance, shunt leakage)
- Accounting for contact resistances in metal electrodes
- Adjusting for workfunction mismatches at interfaces
- Analyzing fill factor losses due to resistive effects
- Comparing ideal vs real device performance

## Circuit Embedding Model

The drift-diffusion charge transport model is embedded within an equivalent circuit to account for parasitic effects:

```
               Rs (series)
    V_app ─────/\/\/\─────┬───────────────────
                          │
                   ┌──────┴──────┐
                   │  PSC Model  │
                   └──────┬──────┘
                          │
                   ┌──────┴──────┐
                   │    Rp       │
                   └─────────────┘
```

## Required Parameters

Configure parasitic resistances in parameters file:

```matlab
% Parasitic Resistances
Rs = 10;      % Series resistance [Ohms], Default: 0
Rp = 1000;    % Parallel/shunt resistance [Ohms], Default: Inf
Acell = 1;    % Cell area [cm²], Default: 1

% Band Offsets (Workfunctions)
Ect = -4.0;   % Cathode workfunction [eV], Default: EfE
Ean = -5.0;   % Anode workfunction [eV], Default: EHH
```

**Default behavior:** Ideal device with `Rs = 0`, `Rp = Inf`, perfect band alignment

## Modify Current Density Calculation

Update total current density to include parallel resistor contribution:

```
J(t) = V_p / R_p + j_n + j_p - ∂/∂t(ε ∂φ/∂x)
```

Where:
- `J(t)`: Total current density [A/m²]
- `V_p`: Potential difference across parallel resistor [V]
- `j_n`, `j_p`: Electron and hole current densities [A/m²]
- `ε`: Permittivity, `φ`: Electric potential
- Displacement current term: `∂/∂t(ε ∂φ/∂x)`

**Note:** IonMonger calculates J(t) at perovskite layer midpoint to minimize numerical error

## Update Boundary Conditions

Modify boundary condition at HTL/metal contact (x = b_H):

```
φ|_{x=b_H} = [V_bi - V(t) - R_s(A*J(t) - V(t)/R_p)] / (R_s + R_p)
```

Where:
- `V_bi`: Built-in voltage [V]
- `V(t)`: Applied voltage [V]
- `A`: Cell area [m²] (convert from cm²)
- `R_s`, `R_p`: Series and parallel resistances [Ω]
- `J(t)`: Total current density [A/m²]

**Assumption:** Displacement current is negligible at the boundary

## Band Offset Effects

Workfunction parameters modify interface energetics:

- `Ect` (Cathode): Adjusts ETL/cathode alignment
- `Ean` (Anode): Adjusts HTL/anode alignment
- Affects built-in potential: `V_bi = (Ean - Ect) / q`
- Influences carrier injection barriers

## Impact on Device Performance

### Series Resistance (Rs)
- **Effect**: Reduces fill factor, lowers short-circuit current at high Rs
- **Sources**: Contact resistance, bulk resistance, lead resistance
- **J-V signature**: Flattening of curve at high current
- **Optimization**: Minimize through good contacts, thick electrodes

### Parallel Resistance (Rp)  
- **Effect**: Reduces open-circuit voltage, fill factor
- **Sources**: Shunt paths, pinholes, edge leakage
- **J-V signature**: Soft turn-on, slope at low voltage
- **Optimization**: Improve film quality, passivation

### Band Offsets
- **Effect**: Changes built-in voltage, carrier extraction efficiency
- **Sources**: Workfunction mismatch, interface dipoles
- **J-V signature**: Shifts V_OC, changes fill factor
- **Optimization**: Select appropriate electrode materials

## Fitting Procedure

When fitting to experimental data:

1. **Start with ideal parameters**: `Rs = 0`, `Rp = Inf`
2. **Match V_OC**: Adjust built-in voltage via workfunctions
3. **Match J_SC**: Adjust generation, recombination
4. **Match fill factor**: Tune Rs and Rp
5. **Iterate**: Small adjustments to all parameters

## Common Values

| Parameter | Typical Range | Typical Sources |
|-----------|---------------|-----------------|
| Rs | 1 - 100 Ω | Contact resistance, TCO resistance |
| Rp | 100 - 10000 Ω·cm² | Shunt paths, leakage |
| Ect | -3.5 to -5.0 eV | ITO, FTO, metal contacts |
| Ean | -4.5 to -6.0 eV | Metal oxides, organic HTLs |

## Validation

Check that:
1. Current density remains physical (no sign errors)
2. Voltage range appropriate for device
3. Parasitic effects don't dominate ideal behavior
4. Units consistent (convert A to cm² as needed)
5. Boundary conditions properly implemented