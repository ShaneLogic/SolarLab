---
name: chebfun-pde-solver-evaluation
description: Evaluates the viability of using Chebfun v5.7.0 with the pde15s solver for PSC (Photoelectrochemical) models. Use this skill when modeling PSC systems where stiffness parameters (lambda, nu) may be below 0.25 or when Shockley-Read-Hall (SRH) recombination nonlinearity is present.
---

# Chebfun PDE Solver Evaluation

## When to Use
Use this skill when:
- Implementing PSC models with Chebfun v5.7.0 using the pde15s solver
- System exhibits stiffness (lambda or nu parameters may be < 0.25)
- SRH (Shockley-Read-Hall) recombination is included in the model
- Evaluating solver performance or troubleshooting failures

## Evaluation Procedure

### Step 1: Check Stiffness Parameters

Examine the stiffness parameters in your PSC model:

1. Identify the values of `lambda` and `nu`
2. Compare each to the threshold of 0.25
3. If either parameter is **below 0.25**, Chebfun v5.7.0 will fail

### Step 2: Assess Nonlinearity

Determine if SRH recombination is present:

1. Check if the model includes SRH (Shockley-Read-Hall) recombination rate
2. The SRH term introduces significant nonlinearity
3. Chebfun cannot handle this nonlinearity directly

### Step 3: Determine Solver Viability

Based on Steps 1 and 2:

| Condition | Chebfun Viability |
|-----------|------------------|
| lambda/nu ≥ 0.25 AND no SRH | **Viable** |
| lambda/nu < 0.25 | **Will fail** |
| SRH recombination present | **Will fail** |
| Both conditions present | **Will fail** |

### Step 4: Consider Workarounds

If Chebfun fails, evaluate these alternatives:

**Manual timestep (backward Euler)**:
- Can handle extreme parameter values (lambda/nu < 0.25)
- Still fails with SRH recombination nonlinearity

**Predictor-corrector strategy**:
- Remedies the SRH issue by linearizing the recombination rate
- Trade-off: Extremely high computational overhead
- May take impractical amounts of time (days)

### Step 5: Make Recommendation

If either condition triggers failure:
1. Avoid using Chebfun unless impractical runtimes are acceptable
2. Consider alternative solvers or numerical methods
3. If proceeding with predictor-corrector, ensure sufficient computational resources

## Key Thresholds

- **Stiffness threshold**: lambda = 0.25, nu = 0.25
- **Version constraint**: Specific to Chebfun v5.7.0
- **Nonlinearity**: SRH recombination causes solver failure