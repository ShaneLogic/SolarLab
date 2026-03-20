---
name: equilibrium-steady-state-classification
description: Determine whether a semiconductor system is in thermal equilibrium or steady state based on external excitation conditions. Use to correctly apply appropriate statistical models, Fermi level configurations, and recombination analysis.
---

# Thermal Equilibrium vs Steady State Classification

Classify the thermodynamic state of a semiconductor system to determine appropriate analysis methods and statistical models.

## When to Use
- Analyzing semiconductor state for any type of calculation
- Setting up transport or recombination equations
- Determining which Fermi level model applies
- Identifying appropriate demarcation line configuration

## Decision Procedure

### Check for External Excitation

**IF** semiconductor is at constant temperature **WITHOUT** external excitation (optical or electrical):

**THEN** classify as **Thermal Equilibrium**

**ELSE IF** external excitation is present (light or electric field):

**THEN** classify as **Steady State** (non-equilibrium)

## Thermal Equilibrium Characteristics

- **Temperature**: Constant, no external excitation
- **Carrier Balance**: Generation and recombination rates balanced (detailed balance) in every volume element
- **Transport**: No net transport of carriers (currents j_n and j_p vanish)
- **Fermi Level**: Single Fermi level E_F uniquely describes carrier distribution
- **Quasi-Fermi Levels**: Collapse (E_Fn = E_Fp = E_F)
- **Demarcation Lines**: Coincide (E_Dn = E_Dp = E_D)
- **Recombination Centers**: No recombination center range exists

## Steady State Characteristics

- **Excitation**: Non-thermal carrier generation present (light or electric field)
- **Deviations**: Stationary (constant over time)
- **Fermi Energy**: Splits into two quasi-Fermi levels (E_Fn and E_Fp)
- **Demarcation Lines**: Two sets appear for each defect type
- **Center Behavior**: Levels previously acting as traps may now act as recombination centers
- **Balance**: Net carrier flow through centers balanced by other transitions (e.g., optical band-to-band generation)
- **Populations**: Time-independent carrier populations maintained

## Output
- System classification: Thermal Equilibrium OR Steady State
- Appropriate statistical model specification
- Fermi level configuration (single vs quasi-Fermi levels)
- Demarcation line configuration
- Recombination center behavior prediction