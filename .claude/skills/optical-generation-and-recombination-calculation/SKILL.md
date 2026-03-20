---
name: optical-generation-and-recombination-calculation
description: Calculate optical generation rates, bulk recombination rates, and interfacial recombination rates for perovskite solar cell simulations. Use when modeling carrier dynamics and losses in drift-diffusion simulations of semiconductor devices, specifically PSCs.
---

# Optical Generation and Recombination Calculation

## When to Use
- Setting up generation rate profiles for optical excitation in PSC simulations
- Calculating carrier recombination losses in the bulk material
- Computing interfacial recombination at perovskite/transport layer boundaries
- Modeling carrier lifetime and efficiency limitations in semiconductor devices

## Optical Generation Rate

Apply when wavelength-dependent generation profile (Equation 32) is NOT used.

**Procedure:**
1. Determine the simplified generation rate G(x,t) instead of using the full wavelength-dependent model
2. Apply Equation 63: G(x,t) = Is(t) Fph alpha exp(-alpha(2 + l x - 2))
3. Ensure all variables are defined for the specific device simulation

**Key Variables:**
- Is(t): Time-dependent incident light intensity
- Fph: Photon flux factor
- alpha: Absorption coefficient
- l: Device thickness or characteristic length
- x: Position within device

## Bulk Recombination Rate

Apply when calculating carrier losses within the bulk volume of the perovskite material.

**Procedure:**
1. Identify carrier concentrations (n, p) and intrinsic carrier density (ni)
2. Obtain material recombination coefficients
3. Apply Equation 64 to calculate R_bulk
4. Result: Recombination rate in units of cm^-3 s^-1

## Interfacial Recombination Rates

Apply when calculating recombination at device interfaces (perovskite/HTL or perovskite/ETL).

**Procedure:**
1. Determine perovskite carrier concentrations adjacent to each interface (n+, p+)
2. Obtain intrinsic carrier density (ni), surface recombination velocities, and electric field (E)
3. Apply Equations 65 and 66 to calculate interfacial recombination rates R_l(n+, p+)
4. These rates are equivalent to IonMonger formulations but coded in terms of perovskite carrier concentrations

**Important:** These rates apply specifically to interfaces and are expressed in terms of perovskite-side carrier concentrations (see Footnote 4 in source).