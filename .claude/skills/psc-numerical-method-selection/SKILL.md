---
name: psc-numerical-method-selection
description: Select the optimal numerical scheme for perovskite solar cell (PSC) simulation based on comprehensive performance benchmarks. Use when determining which discretization approach provides the best balance of accuracy, stability, and computational efficiency for realistic charge transport models in metal halide PSCs.
---

# PSC Numerical Method Selection

## When to Use This Skill

Use this skill when:
- Selecting a numerical method for PSC simulation
- Comparing Finite Element, Finite Difference, and MATLAB pdepe approaches
- Optimizing computational efficiency while maintaining accuracy
- Setting up simulation parameters for charge transport models

## Selection Procedure

### 1. Compare Available Methods

All three methods (FE, FD, pdepe) use:
- Method of Lines for spatial discretization
- MATLAB ode15s for temporal integration

**Key differences arise from spatial discretization approaches.**

### 2. Select Grid Type

**Prefer the tanh grid over the Chebyshev grid.**
- Tanh grid shows significant decrease in error size across all methods
- Error improvement is method-independent

### 3. Apply Performance Ranking

**Method accuracy ranking (highest to lowest):**
1. Finite Element (FE) scheme
2. MATLAB pdepe
3. Finite Difference (FD) scheme

**Computational efficiency:**
- FE scheme requires ~50 times less processing time than FD scheme and pdepe
- This holds when comparing against FD with quadruple precision

### 4. Configure Stability Parameters

**Set temporal tolerance (RelTol):**
- Maintain RelTol < 10^-3 for stability
- Methods become unstable if RelTol > 10^-3
- RelTol values of 10^-4 and 10^-5 yield overlapping results (time stability confirmed)

**Runtime definition:**
- Measure time during simulation call only
- Exclude one-off set-up time from performance metrics

### 5. Final Selection

**Choose: Finite Element scheme on a tanh grid with RelTol ≤ 10^-4**

This configuration provides:
- Highest accuracy among tested methods
- 50x speedup compared to alternatives
- Stable temporal integration
- Robust performance for realistic charge transport models

## See Also

- `matlab-pdepe-limitations-avoidance` - for scenarios where pdepe is unsuitable
- `numerical-scheme-convergence-verification` - for validating spatial accuracy