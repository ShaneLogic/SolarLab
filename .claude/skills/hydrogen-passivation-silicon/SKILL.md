---
name: hydrogen-passivation-silicon
description: Analyze hydrogen behavior, charge state, and passivation effects in silicon wafers during device fabrication. Use when determining hydrogen configuration based on doping type (p-type vs n-type) or evaluating passivation of deep centers.
---

# Hydrogen Passivation in Silicon

## When to Use
- Processing silicon wafers with hydrogen impurities for device fabrication
- Determining hydrogen charge state and lattice position based on doping type
- Evaluating passivation effectiveness for deep centers
- Analyzing conductivity reduction due to hydrogen incorporation

## Determine Hydrogen Configuration

### For p-type Silicon:
- **Charge state**: H+
- **Position**: Bond-center (B) site
- **Lattice relaxation**: 0.4 Å in the bond direction
- **Effect**: Strong passivation of deep centers, significant conductivity reduction

### For n-type Silicon:
- **Charge state**: H0 or H-
  - H0: Finds shallow minima near C-site or T-site (eases diffusion)
  - H-: Stable at T-site (requires high electron density)
- **Electrical property**: Deep donor acting as a negative-U center
- **Effect**: Weak passivation of deep centers, minimal conductivity reduction

## General Hydrogen Behavior

1. **Diffusion**: Hydrogen diffuses easily through silicon lattice
2. **Passivation mechanism**: Attaches to dangling bonds, passivating many deep centers
3. **Conductivity impact**: Reduces conductivity (strongly in p-type, weakly in n-type)
4. **Incorporation behavior**: Resembles self-interstitials, depends on Fermi level

## Output Specification
Provides:
- Hydrogen charge state (H+/H0/H-)
- Lattice site (B/C/T)
- Degree of conductivity reduction
- Passivation effectiveness for deep centers