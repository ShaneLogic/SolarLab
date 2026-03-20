---
name: rational-approximation-methods
description: Compute rational function approximations p/q of type (m,n) for a given function f. Use when approximating functions with rational functions, analyzing poles, or when polynomial approximations are insufficient.
---

# Rational Approximation Methods

## When to Use
Use this skill when:
- Computing a rational approximant p/q for a function f
- Analyzing poles of a function
- Polynomial approximations are insufficient or inefficient
- Need best approximations with equioscillating error

## Prerequisites
- Function f to approximate
- Degree m (numerator polynomial, max degree)
- Degree n (denominator polynomial, max degree)

## Method Selection

Select the appropriate method based on your requirements:

| Method | Command | Best For | Notes |
|--------|---------|----------|-------|
| Chebyshev-Pade | `chebpade` | Functions on an interval | Analogue of Pade approximation |
| Rational Interpolation | `ratinterp` | Fast, robust interpolation | Through m+n+1 Chebyshev points |
| Best Approximant | `remez` | Minimax approximation | Can be fragile for rational functions |
| Caratheodory-Fejer | `cf` | Smooth functions | More robust than remez |

## Execution

### 1. Chebyshev-Pade Approximant (`chebpade`)
Computes rational function where Chebyshev series of p/q matches f as far as possible.

**Default**: Clenshaw-Lord approximation
**Variation**: Use 'aehly' flag for linearized Maehly approximation

### 2. Rational Interpolation (`ratinterp`)
Computes type (m,n) interpolant through m+n+1 Chebyshev points.

**Characteristics**: Often faster and more robust than chebpade
**Option**: Can specify a different set of points

### 3. Best Approximant (`remez`)
Computes best rational approximants with equioscillating error.

**Constraint**: Can be somewhat fragile for rational functions

### 4. Caratheodory-Fejer Approximant (`cf`)
Computes robust approximants for smooth functions.

**Advantage**: More robust than remez for smooth functions

## Pole Analysis
After computing the rational approximation, analyze the roots of the denominator q to approximate the poles of the original function f.

## Output
Rational function r = p./q approximating f