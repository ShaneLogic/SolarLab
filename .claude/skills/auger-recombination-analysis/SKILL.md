---
name: auger-recombination-analysis
description: Calculate Auger recombination rates and carrier lifetimes in semiconductors. Use this skill when analyzing high carrier density scenarios (e.g., heavily doped materials, high injection conditions), narrow gap semiconductors (Eg < 0.35 eV), or when determining dominant recombination mechanisms at elevated carrier concentrations.
---

# Auger Recombination Analysis

## When to Use This Skill

Apply Auger recombination analysis when:
- Working with narrow gap semiconductors (Eg < 0.35 eV) at room temperature
- Analyzing high carrier density conditions (>10¹⁷ cm⁻³)
- Evaluating recombination in heavily doped materials
- Determining carrier lifetime limiting mechanisms
- Designing optoelectronic devices where Auger losses are critical

## Core Workflow

### Step 1: Assess Auger Relevance

Check if Auger recombination is significant:
- **Band gap criterion**: Auger dominates for Eg < 0.35 eV at room temperature
- **Carrier density criterion**: Auger becomes dominant at high injection levels
- **Material type**: Critical for InSb, HgCdTe, and similar narrow gap materials

### Step 2: Calculate Basic Recombination Rate

Use the fundamental Auger recombination formula:

```
R_Auger = B × n² × p
```

Where:
- B = Auger coefficient (typically 10⁻³⁰ to 10⁻²² cm⁶s⁻¹)
- n = electron density (cm⁻³)
- p = hole density (cm⁻³)

### Step 3: Calculate Carrier Lifetime

For electron lifetime limited by Auger:

```
τ_A = 1/(B × n²)
```

**Lifetime scaling with density:**
- Low densities: τ independent of n
- Medium densities: τ ∝ 1/n
- High densities: τ ∝ 1/n² (Auger-dominated regime)

### Step 4: Detailed Quantum-Mechanical Calculation

For precise calculations, use Haug's formula (see references/haug-formula.md):

```
τ_A = [2.4 × 10⁻³¹ × (εr/m*)² × (1 + m*/m₀) × exp(ΔE/kT)] / (n² × I₁² × I₂²)
```

Where ΔE = [(2m* + mp)/(m* + mp)] × Eg

### Step 5: Interpret Results

**Band gap dependence:**
- τ_A increases rapidly with increasing Eg
- For Eg > 0.35 eV: τ_A typically reaches 10⁻⁶ s (Auger negligible)
- Narrow gap materials: Auger is intrinsic and unavoidable at room temperature

**High doping effects:**
- Heavy doping creates sufficient carrier densities for Auger activation
- Auger can dominate even in wider gap semiconductors under high doping

## Quick Reference Values

| Material Type | Typical B (cm⁶s⁻¹) | Critical Density |
|---------------|---------------------|------------------|
| Narrow gap (InSb) | 10⁻²⁶ to 10⁻²² | >10¹⁶ cm⁻³ |
| Medium gap (Si) | 10⁻³¹ to 10⁻³⁰ | >10¹⁸ cm⁻³ |
| Wide gap (GaAs) | 10⁻³⁰ to 10⁻²⁹ | >10¹⁸ cm⁻³ |

## Output Format

Provide results as:
- Recombination rate in cm⁻³s⁻¹
- Carrier lifetime in seconds
- Dominant recombination regime identification
- Comparison with other recombination mechanisms if data available