---
name: numerical-scheme-convergence-verification
description: Validate the spatial accuracy of numerical schemes by computing time-averaged errors against a highly refined reference solution. Use when verifying second-order convergence (slope of 2 on log-log plot) for finite element or finite difference discretizations in PDE simulations.
---

# Numerical Scheme Convergence Verification

## When to Use This Skill

Use this skill when:
- Validating a numerical scheme's spatial accuracy
- Verifying second-order convergence for PDE discretizations
- Quantifying error metrics for FE or FD schemes
- Assessing grid refinement effects on solution quality

## Prerequisites

- Non-uniform spatial grid (e.g., tanh grid)
- Reference solution on a highly refined grid
- Exact solution or sufficient grid resolution for reference

## Verification Procedure

### 1. Define Reference Solution

Establish a reference solution using a highly refined grid:
- **N_M = 3200** subintervals (fixed reference resolution)

### 2. Calculate Absolute Error

For a scalar quantity v computed on N+1 grid points (N subintervals):

```
E(N) = |v(N) - v(N_M)|
```

Where:
- `v(N)` = variable value computed on grid with N subintervals
- `v(N_M)` = variable value on reference grid
- `E(N)` = absolute error at final time T

### 3. Compute Time-Averaged Error

Define time-averaged error `E_bar` by averaging absolute error over M time points:

**Parameters:**
- **M = 100** time points (fixed)
- Time interval: (1/M, 1)
- **Exclude the first time point** from averaging

### 4. Verify Convergence Order

**Plot E_bar versus N** on a log-log scale:
- Expect slope of 2 for second-order spatial convergence
- Error should decrease proportionally to 1/N²

**Interpretation:**
- Slope ≈ 2: Scheme achieves expected second-order convergence
- Slope < 2: Reduced convergence (check implementation)
- Slope > 2: Possible superconvergence or numerical artifact

## Variables

| Variable | Type | Description |
|----------|------|-------------|
| N | Integer | Number of subintervals in test grid |
| N_M | Integer | Reference grid subintervals (3200) |
| v(N) | Scalar | Computed variable on grid N |
| E(N) | Scalar | Absolute error at time T |
| E_bar | Scalar | Time-averaged error |
| M | Integer | Time points for averaging (100) |

## Limitations

- Requires exact solution or highly refined reference grid
- Does not account for interpolation errors
- Accuracy depends on reference grid resolution

## See Also

- `psc-numerical-method-selection` - for method selection criteria