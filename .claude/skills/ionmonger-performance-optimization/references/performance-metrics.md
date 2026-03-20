# Detailed Performance Metrics

## Benchmark Conditions

**Impedance Spectrum Test:**
- 60 sample frequencies
- Same parameter sets for both versions
- Identical hardware configuration

**Results:**
| Version | Time | Speedup |
|---------|------|---------|
| IonMonger 1.0 | 61s | 1.0x (baseline) |
| IonMonger 2.0 | 31s | 1.97x (~2x) |

## Solver Information

**ode15s:**
- MATLAB's solver for stiff differential equations
- Uses variable-step, variable-order methods
- Numerical differentiation vs Analytic Jacobian
- Backward differentiation formulas (BDF)

## Mathematical Context

**DAE System Structure:**
```
F(t, y, y', p) = 0
```

**Jacobian Definition:**
```
J = ∂F/∂y + α ∂F/∂y'
```

Where α is parameter from numerical integration scheme.

## Additional Benefits

**Reduced Computational Cost:**
- Fewer function evaluations
- Less time spent on Jacobian approximations
- More efficient step size selection

**Improved Accuracy:**
- No truncation error from numerical differentiation
- Exact derivatives preserve mathematical structure
- Better convergence to true solution

**Parameter Set Robustness:**
- Tested across wide range of parameters
- Consistent performance improvement
- No regressions observed