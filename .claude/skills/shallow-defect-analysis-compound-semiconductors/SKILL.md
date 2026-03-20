---
name: shallow-defect-analysis-compound-semiconductors
description: Analyze shallow donor and acceptor defects in compound semiconductors by considering ionicity effects, site-dependent behavior, and lattice interactions. Use when analyzing defect energy levels, comparing ground state vs excited state behavior, or evaluating differences between anion and cation site incorporation in III-V and II-VI compounds.
---

# Shallow Defect Analysis in Compound Semiconductors

## When to Use This Skill
Apply this skill when:
- Analyzing shallow donors or acceptors in compound semiconductors (III-V, II-VI)
- Comparing defect behavior between elemental and compound semiconductors
- Evaluating energy level differences due to ionicity or site incorporation
- Assessing whether hydrogenic models apply to your defect system
- Investigating ground state vs excited state behavior of shallow defects

## Key Complexity Factors

Compound semiconductors introduce two critical complexities not present in elemental semiconductors:

1. **Lattice Ionicity**: The electron or hole interacts with alternatingly charged ions in the lattice
2. **Site Dependence**: Defects can incorporate on either anion or cation sites, leading to different behaviors

## Analysis Procedure

### Step 1: Identify Material and Defect Type
- Determine the compound semiconductor type (III-V, II-VI, etc.)
- Identify whether the defect is a shallow donor or acceptor
- Note the incorporation site (anion or cation)

### Step 2: Assess Ionicity Impact
- Recognize that even small degrees of ionicity have non-negligible effects
- Compare ground-state energies for different site incorporations (e.g., GaP:ZnGa vs GaP:SiP)
- Note that site-dependent screening alone cannot fully explain energy differences

### Step 3: Evaluate Model Applicability
- **For ground states**: Standard hydrogenic models often fail due to ionicity and site effects
- **For excited states**: Modified hydrogenic effective mass approximation works reasonably well if the m_e/m_p ratio is sufficiently large

### Step 4: Apply Appropriate Model
- For III-V and II-VI compounds with large m_e/m_p ratios, use modified hydrogenic effective mass approximation for excited states
- For ground states, consider more sophisticated models that account for ionicity

## Applicable Materials
- III-V compounds (e.g., GaP, GaAs, InP)
- II-VI compounds
- Other compound semiconductors with alternating charged ions

## Limitations
- Standard hydrogenic models are inadequate for ground state analysis
- Site effects cannot be explained by screening differences alone
- Results are material-specific and depend on ionicity degree