---
name: chebfun-statistical-analysis
description: Compute statistical measures including 2-norm, mean, standard deviation, and variance for chebfun objects. Use when you need to analyze the statistical properties of functions defined over intervals, compute function norms, or perform quantitative analysis of continuous data.
---

# Chebfun Statistical Analysis

## Norm Computation

### 2-Norm (Default)
```matlab
f_norm = norm(f)
```

The 2-norm is computed as:
```
||f|| = sqrt(∫ |f(x)|² dx)
```

### Examples
```matlab
% Smooth function
f = chebfun('sin(pi*theta)')
norm_f = norm(f)  % Returns 1

% Non-smooth function with splitting
g = chebfun('sign(sin(pi*theta))', 'splitting', 'on')
norm_g = norm(g)  % Returns sqrt(2)
```

## Mean Computation

```matlab
f_mean = mean(f)
```

The mean is computed based on the integral of the function over its domain.

## Standard Deviation

```matlab
f_std = std(f)
```

## Variance

```matlab
f_var = var(f)
```

## Calculation Basis

All statistical commands (mean, std, var) are based on integrals of the function over its region of definition.

## When to Use

- **norm(f)**: Compute L² norm of a function
- **mean(f)**: Compute average value over domain
- **std(f)**: Compute standard deviation
- **var(f)**: Compute variance

## Notes

- Calculations are exact to machine precision for smooth functions
- For non-smooth functions, ensure 'splitting' is on for accurate results
- All computations use the underlying Chebyshev representation