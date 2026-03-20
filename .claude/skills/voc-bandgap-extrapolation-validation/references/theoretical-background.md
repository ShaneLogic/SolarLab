# Theoretical Background

## Ideal Heterojunction Rule

For an ideal heterojunction solar cell, the open-circuit voltage (Voc) follows the relationship:

```
Voc → EG (as T → 0K)
```

Where:
- Voc = Open circuit voltage (V)
- EG = Bandgap of absorber material (eV)
- T = Temperature (K)

## Equation Reference
Equation 36.1 from Thompson et al. (2006) formalizes this relationship for CdS/CdTe heterojunctions.

## Historical Context

### Early Theories
Early discussions attributed CdS benefits in CdTe solar cells to:
- Reduction of refractive index mismatch
- Decreased sunlight reflection at the surface

### Limitation of Early Theories
The optical explanation could NOT account for:
- Significant improvement in open circuit voltage
- Fundamental junction quality differences

### Bandgap Extrapolation as Fundamental Explanation
The Voc bandgap extrapolation method provides a more fundamental explanation:
- Demonstrates CdS enables proper junction formation
- Shows electrical quality improvement beyond optical effects
- Validates the heterojunction behavior

## Literature Reference
Thompson et al. (2006) - Figure 36.1 illustrates the difference between CdS-covered and uncovered CdTe solar cells:
- Lower part: CdS-covered cell showing Voc extrapolation to EG
- Upper part: Uncovered cell showing deviation from EG

## Variables

| Variable | Type | Description |
|----------|------|-------------|
| Voc | float | Open circuit voltage (V) |
| EG | float | Bandgap of absorber material (eV) |
| T | float | Temperature (K) |

## Domain Tags
- Voc_analysis
- bandgap
- heterojunction_quality
- junction_validation

## Importance Level
Medium - This is a specialized validation technique for CdS/CdTe heterojunction characterization.