---
name: deep-center-recombination-analysis
description: Analyze recombination mechanisms at deep centers using configuration coordinate diagrams. Use when determining whether recombination is radiative or nonradiative, calculating temperature-dependent capture cross sections, or evaluating electron-lattice coupling strength.
---

# Deep Center Recombination Analysis

## When to Use
- Determining recombination mechanism (radiative vs nonradiative) at deep centers
- Calculating temperature-dependent capture cross sections for deep traps
- Analyzing defects with strong electron-lattice coupling
- Evaluating multiphonon emission processes
- Predicting luminescence efficiency

## Dexter-Klick-Russell Rule: Determine Recombination Type

1. **Identify key energies**:
   - En: Optical excitation energy
   - EB: Crossover energy (intersection of upper and lower curves)

2. **Apply the rule**:
   - **IF En > EB**: Radiative recombination occurs
     - Electron recombines from upper minimum to lower curve
     - Indicates relatively weak electron-lattice coupling
   - **IF En < EB**: Nonradiative recombination occurs
     - Electron crosses over to lower curve
     - Reaches ground state via multiphonon emission
     - Indicates strong electron-lattice coupling

3. **Thermal activation check**:
   - For nonradiative case, determine if thermal activation is required
   - Nonthermal portion accomplished by tunneling

## Nonradiative Recombination: Multiphonon Model

### Mechanism Overview
1. Strong coupling between electron eigenfunctions and lattice oscillations
2. Defect level moves in band gap with atom vibrations
3. Large vibration moves level into conduction band to accept electron
4. Violent vibration follows capture, relaxing with multiphonon emission
5. Large relaxation can convert level from upper to lower band gap half (hole trap)

### Calculate Temperature-Dependent Capture Cross Section

**Step 1**: Determine activation energy from configuration coordinate diagram
- EB1: Activation energy for electron trapping
- EB2: Activation energy for hole trapping

**Step 2**: Apply capture cross section formula
```
σ = σ_∞ × exp(-EB / kT)
```

**Step 3**: Calculate pre-exponential factor σ_∞ (if needed)
```
σ_∞ = s_∞ × √(v_c × S / v_t)
```

Where:
- s_∞: Detailed balance factor (similar to unrelaxed trap equation)
- v_c: Number of equivalent valleys in conduction band
- v_t: Degeneracy of deep trap level
- S: Number of phonons emitted (Huang-Rhys factor)
- wr: Defect eigenfrequency of breathing mode

### Luminescence Efficiency Calculation
```
Ratio = (En - EB) / (hω)
```
This ratio determines luminescence efficiency as function of electron-lattice coupling parameter A.

## Output Specification
Provides:
- Recombination type (radiative or nonradiative)
- Temperature-dependent capture cross section σ
- Assessment of electron-lattice coupling strength
- Luminescence efficiency prediction