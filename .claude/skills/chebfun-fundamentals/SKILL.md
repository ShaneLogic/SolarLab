---
name: chebfun-fundamentals
description: Understand what chebfuns are, their philosophy, and how to construct them. Use when creating numerical representations of functions on intervals.
---

# Chebfun Fundamentals

## What is a Chebfun?

A **chebfun** is a numerical representation of a function of one variable defined on a finite interval [a, b]. It achieves for functions what floating-point arithmetic achieves for numbers—rapid computation where each operation is exact apart from a very small relative rounding error.

**Core Philosophy**: "Feel symbolic but run at the speed of numerics"

The syntax for chebfuns is almost exactly the same as MATLAB syntax for vectors, with familiar commands overloaded in natural ways.

## Construction Syntax

### Basic Constructor
```matlab
f = chebfun('cos(20*x)')              % String expression on default domain
f = chebfun(@(x) besselj(0,x))        % Anonymous function handle
```

### Specifying Domain
```matlab
f = chebfun('cos(20*x)', [0, 100])    % Custom domain [a, b]
f = chebfun(@(x) sin(x), [0, 2*pi])   % Function handle with domain
```

### Default Domain
- If no domain is specified, defaults to `[-1, 1]` (mimics Chebyshev polynomials)
- Explicitly pass domain interval for functions on other intervals

## Internal Representation

Chebfuns are represented by polynomial interpolation in Chebyshev points (or equivalently, expansions in Chebyshev polynomials).

**Chebyshev Points Formula** (for interval [-1, 1]):
```
x_j = cos(j*pi/N) for j = 0, ..., N
```

- Points are clustered near endpoints (denser at -1 and 1)
- For other intervals, points are obtained by linear scaling
- Number of points (length) is determined adaptively
  - Simple functions: 20-30 points
  - Complicated functions: 1000 or 1,000,000+ points
- Evaluation uses barycentric formula (stable even for millions of points)

## Accuracy Goal

Adaptive procedures aim to represent each function to roughly **machine precision** (about 15 digits of relative accuracy).

## Use When

- Creating numerical representations of mathematical functions
- Need high-precision function approximation on an interval
- Working with functions that may be smooth or piecewise smooth
- Preparing functions for operations like integration, differentiation, root-finding