---
name: ion-migration-modeling
description: Model ion migration in perovskite layers including steric effects and nonlinear diffusion. Use when simulating mobile ion behavior, high ion vacancy density regimes, hysteresis in J-V curves, impedance spectra showing ionic features, or when P approaches the maximum packing density.
---

# Ion Migration Modeling

Use this skill when:
- Simulating perovskite solar cells with mobile ion migration
- Modeling high ion vacancy density regimes (P close to P_lim)
- Investigating hysteresis effects in J-V curves
- Analyzing impedance spectra with low-frequency ionic features
- Preventing unphysical ion densities exceeding packing limits
- Using PNP systems where standard assumptions fail

## Standard PNP Limitations

The standard Poisson-Nernst-Planck (PNP) model assumes:
- Dilute ion distribution: `P << P_lim`
- Linear flux (no steric effects)
- Unlimited ion capacity

These assumptions fail at high ion densities typical in perovskites.

## Enable Steric Effects

Configure steric effects in parameters file:

```matlab
% Steric effects configuration
NonlinearFP = 'Diffusion';  % Form of ion vacancy flux: 'Diffusion' or 'Drift'
Plim = 1e27;               % Maximum vacancy density [m^-3]
```

**Default behavior:** If Plim not specified, uses standard linear flux (no steric effects)

## Nonlinear Diffusion Flux Model

When steric effects enabled and NonlinearFP = 'Diffusion', use the modified ion flux:

```
F_P = -D_I (∂P/∂x) [1 / (1 - P/P_lim)] + (qP / k_B T)(∂φ/∂x)
```

Where:
- `F_P`: Ion vacancy flux [m⁻² s⁻¹]
- `D_I`: Constant diffusion coefficient [m² s⁻¹]
- `P`: Ion vacancy density [m⁻³]
- `P_lim`: Density of anion sites (maximum vacancy density) [m⁻³]
- `φ`: Electric potential [V]
- `q`: Elementary charge [C]
- `k_B`: Boltzmann constant [J/K]
- `T`: Temperature [K]

## Key Behaviors

### Enhanced Diffusion
- The term `[1 / (1 - P/P_lim)]` diverges as `P → P_lim`
- Diffusion is "enhanced" when vacancy density approaches maximum
- Prevents unphysical accumulation beyond P_lim

### Blakemore Model Connection
- This formulation results from employing a Blakemore model
- Equivalent to Fermi-Dirac integral of order -1
- Physically accounts for finite size/packing of ions

### Limiting Cases

**Low density (P << P_lim):**
```
[1 / (1 - P/P_lim)] → 1
F_P → -D_I (∂P/∂x) + (qP / k_B T)(∂φ/∂x)
```
Recovers standard PNP system

**Maximum density (P_lim → ∞):**
```
[1 / (1 - P/P_lim)] → 1
```
Standard PNP system (unlimited capacity)

## Dynamic Behavior

- **Steady state**: Same as Modified Drift model
- **Dynamics**: Different transient behavior
- **Important for**: Hysteresis, impedance spectra, time-dependent response

## Implementation Steps

1. **Check density regime**:
   - Calculate `P/P_lim` ratio
   - If approaching 1, steric effects become significant

2. **Select flux form**:
   - Use 'Diffusion' for enhanced diffusion model
   - Or 'Drift' for alternative steric formulation

3. **Set P_lim**:
   - Based on material crystal structure
   - Typical values: ~10²⁶ - 10²⁸ m⁻³ for perovskites

4. **Monitor density**:
   - Ensure P never exceeds P_lim
   - Check for numerical stability at high P

## When Steric Effects Matter

Steric effects become important when:

**High ion density:**
- High vacancy concentrations
- Strong ion accumulation at interfaces
- `P/P_lim > 0.5` indicates significant steric effects

**Specific regimes:**
- During device operation (ion migration active)
- Near interfaces (ion accumulation)
- Under bias (field-driven ion redistribution)
- Impedance at low frequencies

**Material properties:**
- Materials with limited vacancy sites
- High ion mobility leading to accumulation
- Small P_lim values (tight packing)

## Parameters and Units

| Parameter | Symbol | Unit | Typical Range |
|-----------|--------|------|---------------|
| Ion density | P | m⁻³ | 10²⁴ - 10²⁷ |
| Max density | P_lim | m⁻³ | 10²⁶ - 10²⁸ |
| Diffusion coeff | D_I | m²/s | 10⁻¹⁴ - 10⁻¹² |
| Temperature | T | K | 280 - 320 |

## Verification

Always verify:
1. Physical constraint: `P ≤ P_lim` never violated
2. Flux remains finite (check denominator)
3. Recovers standard PNP at low densities
4. Reasonable ion mobilities
5. Consistent with experimental hysteresis behavior