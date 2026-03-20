# Parameter Reference for Eq. (18)

## Realistic Parameter Estimates
Eq. (18) provides the following realistic parameters for model configuration:

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Recombination coefficient | γ | 2.4 | s⁻¹ (normalized) |
| Bulk recombination rate | R(n,p) | γp | cm⁻³s⁻¹ |
| Left surface recombination | R_l(p) | 0 | cm⁻²s⁻¹ |
| Right surface recombination | R_r(n) | 0 | cm⁻²s⁻¹ |

## Implementation Notes

### Code Implementation Example
```python
def simplified_recombination(p, gamma=2.4):
    """
    Calculate simplified bulk recombination rate.
    
    Args:
        p: Hole concentration (cm^-3)
        gamma: Recombination coefficient (default: 2.4)
    
    Returns:
        R: Recombination rate (cm^-3 s^-1)
    """
    return gamma * p

def surface_recombination_left(p):
    """Left surface recombination (set to zero)."""
    return 0.0

def surface_recombination_right(n):
    """Right surface recombination (set to zero)."""
    return 0.0
```

### Units Consistency
Ensure all parameters are in consistent units before applying the simplification. The recombination coefficient γ should be expressed in appropriate time units matching your simulation time step.