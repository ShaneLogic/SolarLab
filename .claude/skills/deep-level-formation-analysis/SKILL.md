---
name: deep-level-formation-analysis
description: Predicts energy level positions and character (deep, shallow, or resonant) of defect centers in semiconductors. Use when modeling substitutional impurities, analyzing vacancy-related defects, determining whether impurity-induced states are deep (vacancy-like) or shallow (hydrogen-like), or predicting charge density distributions around defect sites in compound semiconductors.
---

# Deep Level Formation Analysis

## When to Use This Skill

Apply this skill when:
- Modeling energy levels of deep centers caused by substitutional impurities
- Analyzing vacancy-related defect states in semiconductors
- Determining whether an impurity will create deep or shallow levels
- Predicting charge density distributions around defect sites

## Prerequisites

- Understanding of vacancy states (A1 and T2 symmetry)
- Knowledge of impurity potential magnitude (V0)
- Familiarity with band structure concepts

## Core Procedure

### Step 1: Identify the Defect Configuration

Determine the impurity type and its substitutional site in the host semiconductor lattice.

### Step 2: Assess Impurity Potential (V0)

Evaluate the depth of the impurity binding potential:
- **Lower energy orbitals** (compared to host): Small shift expected, impurity-like hyper-deep bonding level in/below valence band, vacancy-like antibonding level in gap
- **Higher energy orbitals**: Levels lie within conduction band, slightly above vacancy states

### Step 3: Determine State Character

Based on V0 magnitude:

**Deep Well (large |V0|):**
- Produces deep level center
- Vacancy-like charge density distribution
- States localized near defect site

**Weakening Potential (decreasing V0):**
- Transition toward hydrogen-like spread out distribution
- States become more extended

**Intersection with Conduction Band:**
- Effective mass approximation applies
- Results in shallow (hydrogen-like) levels

### Step 4: Analyze Level Positions

Apply the interaction mechanism:
- s and p orbitals interact with A1 and T2 vacancy states
- Bonding states merge with valence band
- Antibonding states may appear in the gap or merge with conduction band

### Step 5: Evaluate Spectrum Compression

Consider that two-band influence compresses the spectrum:
- |∂E/∂V| decreases rapidly as |V0| increases
- Deeper potentials lead to less sensitivity to potential variations

## Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| V0 | Energy | Depth of the impurity binding potential |
| ∂E/∂V | Rate | Rate of change of energy state with respect to potential |

## Output Interpretation

The analysis predicts one of three outcomes:

1. **Deep Level**: Vacancy-like, localized charge distribution
2. **Shallow Level**: Hydrogen-like, spread out distribution (effective mass regime)
3. **Resonant State**: Level merged with band states

## Constraints

- Derived from Green's function calculations using semi-empirical tight-binding Hamiltonians
- Vacancy states represent limiting cases for donor/acceptor-like states