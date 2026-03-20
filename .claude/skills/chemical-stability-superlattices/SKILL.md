---
name: chemical-stability-superlattices
description: Evaluates the long-term chemical stability of semiconductor superlattices by analyzing material type and lattice mismatch. Use this when assessing whether a superlattice structure will degrade through alloy formation or segregation, particularly for isovalent systems (e.g., GaAs-AlAs), mismatched systems (e.g., Si-Ge, GaAs-InAs), or low-mismatch systems.
---

# Chemical Stability of Superlattices

## When to Use
Use this skill when:
- Analyzing the long-term stability of a superlattice structure
- Evaluating potential degradation mechanisms in semiconductor heterostructures
- Comparing stability between different material pairings
- Assessing whether alloy formation will occur over time

## Procedure

### 1. Classify Material Type
Determine which category the superlattice falls into:
- **Case A**: Isovalent semiconductors (e.g., GaAs-AlAs)
- **Case B**: Semiconductors with large lattice mismatch (e.g., Si-Ge, GaAs-InAs)
- **Case C**: Low lattice mismatch systems

### 2. Evaluate Stability Based on Classification

#### For Case A (Isovalent Semiconductors):
- **Status**: Chemically unstable with respect to segregation
- **Mechanism**: Alloy formation is the dominant degradation mechanism because it does not require nucleation
- **Temperature Effect**: Recrystallization is usually frozen-in at room temperature
- **Example Result**: Ga1-xAlxAs alloy will form over time

#### For Case C (Low Lattice Mismatch):
- **Status**: Unstable with respect to alloy formation
- **Mechanism**: Similar to isovalent systems, alloy formation drives degradation

#### For Case B (Large Lattice Mismatch):
- **Status**: More stable than ultrathin superlattices
- **Reason**: Alloy formation energy lies above that for ultrathin superlattices
- **Relative Comparison**: Enhanced thermodynamic stability compared to thin-layer structures

### 3. Check for Spontaneous Ordering
Consider whether the system exhibits spontaneous ordering:
- Some ultrathin superlattices can grow spontaneously as ordered compounds
- Example: (GaAs)1(AlAs)1 near 840K without artificial layer-by-layer deposition
- This indicates a natural tendency toward ordered compound formation under specific conditions

## Key Constraints
- **Temperature dependence**: Stability predictions assume room temperature where recrystallization is frozen
- **Time dependence**: Evaluation is for long-term stability, not immediate structural integrity

## Output
Provide a prediction of chemical stability with the following components:
- Stability status (Stable vs. Unstable/Prone to Alloying)
- Dominant degradation mechanism (alloy formation, segregation)
- Specific material considerations
- Relative stability comparison when applicable