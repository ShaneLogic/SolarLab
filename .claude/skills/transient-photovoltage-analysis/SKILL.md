---
name: transient-photovoltage-analysis
description: Simulate and analyze transient photovoltage (TPV) response using analytical models and Driftfusion configuration for field-free single layer devices under small perturbation conditions.
---

# Transient Photovoltage Analysis

Use this skill when you need to:
- Verify time-dependence of simulation solutions
- Analyze TPV decay characteristics
- Configure Driftfusion for TPV simulations
- Compare analytical TPV models with numerical results

## TPV Analytical Model

**Applicability:** Field-free single layer devices at open circuit

**Prerequisites:**
- Small perturbation condition: $\Delta n \ll n_{OC}$
- Steady-state carrier density ($n_{OC}$) known

**Key Relations:**

**Voltage Change:**
$$\Delta V_{OC}(t) = \frac{kT}{q} \cdot \frac{\Delta n(t)}{n_{OC}}$$

**Carrier Density Change:**

**During pulse ($t < t_{pulse}$):**
$$\Delta n(t) = \frac{\Delta g}{k_{TPV}} \cdot (1 - e^{-k_{TPV} t})$$

**After pulse ($t \geq t_{pulse}$):**
$$\Delta n(t) \text{ follows decay kinetics determined by } k_{TPV}$$

**Variables:**
- $k_{TPV}$: Decay rate constant of TPV signal
- $t_{pulse}$: Laser pulse length
- $\Delta g$: Additional uniform volumetric generation rate from pulse
- $n_{OC}$: Steady-state open circuit carrier density

## TPV Simulation Configuration

**Device Geometry:**
- 100 nm field-free single layer
- Bandgap: $E_g = 1.6$ eV (example)

**Boundary Conditions:**
- Zero flux density for electronic carriers
- Represents perfect blocking contacts
- **Note:** If electric field is present, these conditions do not represent open circuit; use high series resistance ($R_s$) or mirrored cell approach instead

**Optical Generation:**
- Constant uniform volumetric generation rate: $g_0 = 1.89 \times 10^{21}$ cm⁻³ s⁻¹
- Based on integrated photon flux for AM1.5G spectrum
- Step function absorption model

**Recombination Settings:**
- Band-to-band coefficient: $B = 10^{-10}$ cm³ s⁻¹
- SRH recombination: Switched off

**Perturbation Parameters:**
- Pulse length: $t_{pulse} = 1$ μs
- Pulse intensity: 20% of bias light intensity

**Prerequisites:**
- AM1.5 spectrum data available
- Bandgap defined