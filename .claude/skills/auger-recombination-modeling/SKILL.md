---
name: auger-recombination-modeling
description: Calculate Auger recombination rates and incorporate into bulk recombination models for perovskite solar cells. Use when simulating high carrier density conditions, analyzing high-injection regimes, or requiring accurate recombination modeling beyond SRH and radiative mechanisms.
---

# Auger Recombination Modeling

Use this skill when:
- Simulating perovskite solar cells under high illumination intensity
- Modeling high carrier density conditions where three-particle processes matter
- Fitting experimental J-V curves that show Auger-dominated losses
- Comparing different recombination mechanisms (SRH, radiative, Auger)
- Investigating non-ideal device behavior at high voltages

## Auger Recombination Fundamentals

Auger recombination is a three-particle process where:
- An electron and hole recombine
- The excess energy is transferred to a third carrier (electron or hole)
- Rate scales with the cube of carrier density

**Prerequisites:**
- Carrier densities n and p
- Auger coefficients A_n (electron-dominated) and A_p (hole-dominated)
- Intrinsic carrier density n_i

## Calculate Auger Recombination Rate

Use the Auger recombination formula:

```
R_Auger = (A_n * n + A_p * p) * (n * p - n_i²)
```

Where:
- `R_Auger`: Auger recombination rate [m⁻³ s⁻¹]
- `A_n`: Electron Auger coefficient [m⁶ s⁻¹]
- `A_p`: Hole Auger coefficient [m⁻¹ s⁻¹]
- `n`: Electron density [m⁻³]
- `p`: Hole density [m⁻³]
- `n_i`: Intrinsic carrier density [m⁻³]

**Critical constraint:** This formulation ensures `R_Auger = 0` when `n * p = n_i²` (thermal equilibrium)

## Total Bulk Recombination Rate

Combine all recombination mechanisms:

```
R(n,p) = R_SRH + R_rad + R_Auger
```

Where:
- `R_SRH`: Shockley-Read-Hall (trap-assisted) recombination
- `R_rad`: Radiative/bimolecular recombination
- `R_Auger`: Auger recombination

## Configure Auger Parameters

Enable Auger recombination in simulation parameters:

```matlab
% Auger coefficients
Augn = 1e-42;  % Electron-dominated Auger rate [m^6 s^-1]
Augp = 1e-42;  % Hole-dominated Auger rate [m^6 s^-1]

% Default values (if not specified)
% Augn = 0     % Auger disabled
% Augp = 0
```

## Advanced Generation Modeling

When enabling Auger recombination, also consider spectral generation:

```matlab
% Generation rate G(x,t) - supports spectrum
% Default: Eq. (31) - single wavelength
% Specify custom G for realistic solar spectra
G = @(x,t) custom_generation_profile(x,t);
```

**Optional parameters for advanced modeling:**
- Immobile ion distributions: `DI = 0` (default) or specify ion diffusion coefficient
- Wavelength-dependent absorption profiles
- Time-varying illumination conditions

## When Auger Matters

Auger recombination becomes significant when:

**High carrier densities:**
- Strong illumination (full sun or concentrated)
- High injection conditions
- `n * p ≫ n_i²`

**Material properties:**
- High Auger coefficients (material-dependent)
- Small bandgap materials
- Certain perovskite compositions

**Device operation:**
- Near open-circuit voltage
- Under high forward bias
- In high-efficiency devices where other losses minimized

## Implementation Notes

- Auger rate is added to continuity equations for electrons and holes
- In IonMonger, modify parameters file with Augn and Augp
- Default model assumes simple generation and no Auger if parameters omitted
- Auger coefficients typically in range 10⁻⁴¹ to 10⁻⁴⁴ m⁶ s⁻¹ for perovskites

## Verification

Always verify:
1. `R_Auger = 0` at thermal equilibrium (`n*p = n_i²`)
2. Positive recombination rate under non-equilibrium
3. Reasonable magnitude compared to other recombination mechanisms
4. Units consistency (m⁻³ s⁻¹ for all R terms)