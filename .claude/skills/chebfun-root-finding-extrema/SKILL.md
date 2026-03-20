---
name: chebfun-root-finding-extrema
description: Find all zeros (roots) of a chebfun object using the Boyd-Battles method, and identify local minima and maxima without explicitly computing derivatives. Use when you need to locate all function zeros in a domain, find intersections of curves, or identify local extrema of smooth and non-smooth functions.
---

# Chebfun Root Finding and Extrema

## Global Root Finding

Find all zeros of a chebfun in its region of definition.

### Basic Usage
```matlab
roots_vector = roots(f)
```

### Algorithm
- Uses method due to Boyd and Battles
- For chebfun degree > 50, recursively breaks into smaller pieces
- On each piece, zeros found as eigenvalues of colleague matrix (analogue of companion matrix for Chebyshev polynomials)

### Handling Discontinuities

**Default behavior:** Includes roots at jumps for piecewise functions.

**Omit jump roots:**
```matlab
genuine_roots = roots(f, 'nojump')
```

### Applications

Find intersections of curves:
```matlabnf = chebfun('x')
g = chebfun('cos(x)')
intersection_points = roots(f - g)
```

## Local Extrema Finding

Find local minima and maxima without explicitly computing derivatives.

### Basic Usage
```matlab
% Find local minima
[min_points, min_values] = min(f, 'local')

% Find local maxima
[max_points, max_values] = max(f, 'local')
```

### Return Values
The command returns:
- Interior local extrema points
- Endpoints of the function domain

### Smooth Functions
Extrema located by finding zeros of the derivative.

### Non-smooth Functions
Non-smooth extrema identified at points where:
- Derivative "jumps from one sign to the other"
- Corresponds to "zeros" of the derivative at discontinuities

### Example
```matlab
f = chebfun('sin(x) + 0.5*cos(3*x)', [0, 2*pi])
[local_mins, min_vals] = min(f, 'local')
[local_maxs, max_vals] = max(f, 'local')
plot(f)
hold on
plot(local_mins, min_vals, 'ro')
plot(local_maxs, max_vals, 'go')
```

## When to Use

- **roots(f)**: Find all zeros of a function in its domain
- **roots(f, 'nojump')**: Find only genuine roots, excluding jump discontinuities
- **min/max(f, 'local')**: Find local extrema without derivative computation
- **Curve intersections**: Compute roots of f - g