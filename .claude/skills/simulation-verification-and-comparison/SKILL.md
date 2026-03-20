---
name: simulation-verification-and-comparison
description: Compare simulation results between Driftfusion and other simulators (ASA, IonMonger), analyze discrepancies in J-V characteristics, and configure simulations for fair comparison.
---

# Simulation Verification and Comparison

Use this skill when you need to:
- Compare Driftfusion results with ASA or IonMonger simulators
- Analyze discrepancies in J-V characteristics between simulators
- Configure simulations for fair comparison
- Identify variance sources and apply mitigation strategies

## ASA Comparison Configuration

**When to use:** Comparing Driftfusion results with ASA tool

**Configuration Steps:**

1. **Discretization Setup:**
   - Set linear grid spacing for ASA: 1 nm
   - Set interface thickness in Driftfusion: 1 nm

2. **Optical Model:**
   - Select Beer-Lambert option (without back contact reflection)
   - Use identical optical constant and photon flux density spectrum data
   - Alternative: Insert generation profile from ASA into Driftfusion parameters

3. **J-V Scan Settings:**
   - Scan range: $V_{app} = 0$ to 1.3 V
   - Scan rate: $k_{scan} = 10^{-10}$ V/s
   - Reason: Minimizes displacement current for fair comparison with steady-state ASA solver

## Simulator Discrepancy Analysis

**When to use:** Comparing simulation results for devices with varying conduction band properties

**Calculation:**
$$\text{Percentage Difference} = 100 \times \frac{J_{ASA} - J_{DF}}{J_{ASA}}$$

**Analysis Guidelines:**

**Parameter Set 1 (PS1):**
- Expected difference: ~1% for $J > 10^{-12}$ A cm⁻²
- Halving active layer thickness has minimal impact

**Parameter Set 2 (PS2):**
- Expected difference: Up to ~5% for $J > 10^{-12}$ mA cm⁻²
- Root causes:
  - Electron density change > 7 orders of magnitude at absorber-ETL interface
  - eDOS transition: $N_{CB} = 10^{18}$ to $10^{20}$ cm⁻³
  - Conduction band energy change: 0.3 eV

**Mitigation:** Use uniform eDOS ($N_{CB} = 10^{18}$ cm⁻³) across all layers

## Three-Layer Device Methodology Comparison

**IonMonger Approach:**
- Abrupt interfaces
- Solves 8 variables simultaneously
- Only holes in HTL, only electrons in ETL
- Boundary conditions evaluate interfacial recombination at same grid point

**Driftfusion Approach:**
- Discrete interlayer interface approach
- Solves 4 variables simultaneously
- All carriers resolved in all regions
- Ionic carrier mobility = 0 in HTL, ETL, and interfaces
- Ionic charge compensated by static background charged density

## Variance Sources and Mitigation

**Primary Variance Sources:**
- Treatment of electronic currents across interfaces
- Spatial mesh differences
- Ionic carrier density calculation (Driftfusion: all layers; IonMonger: not all)

**Interfacial Recombination Errors:**
- Volumetric surface recombination scheme introduces errors
- Surface carrier density differences (electron density at active layer-HTL interface)
- Errors increase with energetic barriers (0.4 to 0.8 eV)

**Mitigation Strategies:**
1. Increase interface thickness
2. Use more interface mesh points

**Trade-off:** Increased thickness sacrifices consistency with analytical models using abrupt interfaces, but provides greater flexibility.