---
name: carrier-recombination-trapping
description: Analyze carrier recombination and trapping mechanisms in semiconductors. Use when calculating carrier lifetimes, determining recombination currents, analyzing defect states and trap behavior, or evaluating how excess carriers return to thermal equilibrium.
---

# Carrier Recombination and Trapping

## When to Use
- Calculating carrier lifetimes in semiconductors
- Determining recombination current components
- Analyzing defect states and trap behavior
- Evaluating excess carrier return to equilibrium
- Modeling semiconductor device performance

## Recombination Mechanisms

### Radiative Recombination
- Direct band-to-band recombination with photon emission
- Important in direct bandgap semiconductors
- Efficiency depends on overlap of electron and hole wavefunctions

### Nonradiative Recombination
- **Geminate recombination**: Initial electron-hole pair recombination
- **Plasmon-induced recombination**: Energy transferred to plasmon modes
- **Phonon-assisted Auger transitions**: Three-particle process
- **Impact ionization**: Reverse process generating multiple carriers

### Shockley-Read-Hall (SRH) Theory

The SRH model describes recombination through defect states in the bandgap:

**Key Concepts**:
- Hall-Shockley-Read centers: Defect states facilitating recombination
- Schottky-Read-Hall centers: Alternative terminology for same mechanism
- Shockley-Frank-Read approximation: Simplified SRH model

**SRH Model**: Provides framework for calculating recombination rates via trap states

## Trapping Mechanisms

### Trap Types
- **Hole traps**: Capture and temporarily hold holes
- **Electron traps**: Capture and temporarily hold electrons
- **Shallow traps**: Located near band edges, easily thermally released
- **Isoelectronic traps**: Neutral impurities with same valence as host

### Trap Effects
- Influence carrier mobility and lifetime
- Can cause persistent photoconductivity
- Affect device switching characteristics
- May be beneficial or detrimental depending on application

## Recombination Currents

### Current Components
- **Generation/recombination current**: Combined GR processes
- **Generation current**: Net carrier generation in depletion regions
- **Recombination current**: Net carrier recombination
- **Recombination leakage current**: Parasitic recombination path
- **Pure generation current**: Generation without recombination component

### Dynamic Effects
- **Recombination overshoot**: Transient excess recombination
- **Maximum GR-currents**: Upper limit of GR current magnitude

## Recombination Centers

### Center Characteristics
- **Recombination center density**: Concentration of active recombination sites
- **Center annihilation**: Removal or deactivation of recombination centers

### Center Types
- **Recombination centers**: General term for defect-mediated recombination sites
- Specific center types identified by energy level and capture cross-section

## Analysis Approach

1. **Identify dominant mechanism**: Determine radiative vs nonradiative vs SRH dominance
2. **Characterize trap states**: Determine energy levels, capture cross-sections, densities
3. **Calculate lifetimes**: Use appropriate model for carrier lifetime
4. **Determine current components**: Separate generation, recombination, and leakage currents
5. **Evaluate temperature dependence**: Analyze how mechanisms vary with temperature

## Key Parameters
- **Carrier lifetime**: Average time before recombination
- **Capture cross-section**: Probability of carrier capture by trap
- **Trap density**: Concentration of trapping/recombination centers
- **Thermal velocity**: Carrier velocity affecting capture rate

## Applications
- Solar cell efficiency optimization
- LED design and performance
- Photodetector response time
- Transistor switching characteristics
- Radiation damage analysis