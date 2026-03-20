---
name: semiconductor-physics-fundamentals
description: Apply fundamental semiconductor physics concepts including Fermi energy positioning, depletion approximation for p-n junctions, and current density relations to analyze device behavior and validate simulation results.
---

# Semiconductor Physics Fundamentals

Use this skill when you need to:
- Determine Fermi energy position in different semiconductor types at equilibrium
- Apply depletion approximation for analytical p-n junction solutions
- Calculate current density parameters (J0, JSC) for validation
- Relate simulation parameters to analytical outputs

## Equilibrium Fermi Energy Position

Determine Fermi energy ($E_{F0}$) position based on semiconductor type:

**Conditions:**
- Equilibrium conditions only
- Voltage $V = 0$

**Position Rules:**
1. **Intrinsic Semiconductor:** $E_{F0}$ lies close to the middle of the band gap
2. **P-type Semiconductor:** $E_{F0}$ shifts toward the valence band (acceptors take electrons)
3. **N-type Semiconductor:** $E_{F0}$ shifts toward the conduction band (donors provide electrons)

**Notation:** Subscript '0' denotes equilibrium values (e.g., $E_{F0}$, $E_{CB,0}$)

## Depletion Approximation for p-n Junctions

Apply depletion approximation for analytical validation:

**Assumptions:**
- Space charge density approximated as step function
- Transport and recombination neglected in depletion region
- Diffusion length significantly greater than device thickness ($L_{n,p} \gg d$)

**Key Calculations:**
- Depletion widths for n-type ($w_n$) and p-type ($w_p$) regions
- Current using Shockley diode equation: $J = J_{SC} - J_0 \cdot (\exp(qV/kT) - 1)$

## Current Density Relations

**Dark Saturation Current Density ($J_0$):**
$$J_0 = q \left[ \frac{n_i^2 D_n}{N_A L_n} + \frac{n_i^2 D_p}{N_D L_p} \right]$$

Where $L_n = \sqrt{\tau_n D_n}$ and $L_p = \sqrt{\tau_p D_p}$

**Short Circuit Current Density ($J_{SC,max}$):**
$$J_{SC,max} = q \int_{E_g}^{\infty} \phi_0(E_\gamma) \cdot \eta(E_\gamma) \, dE_\gamma$$

For perfectly absorbing semiconductor ($\eta=1$ for $E_\gamma \geq E_g$):
$$g_0 = \frac{j_{SC,max}}{d_{DR}}$$

See `references/current-density-formulas.md` for detailed derivations and variable definitions.