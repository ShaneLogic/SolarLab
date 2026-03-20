---
name: spectral-algorithms
description: Configure spectral discretization methods and algorithmic preferences for high-accuracy solving. Use when solving high-order differential equations or when default accuracy is insufficient.
---

# Spectral Algorithms

## Chebyshev Grid Kind

### Second Kind (Default)
- Points: `cos(j*pi/n)` for `0 <= j <= n`
- Implemented in: `chebtech2`
- Query: `t = chebkind` (returns 2)

### First Kind
- Points: `cos((j+1/2)*pi/(n+1))` for `0 <= j <= n`
- Implemented in: `chebtech1`

**Switching to First Kind**:
```matlab
chebkind(1);
% or
chebfunpref.setDefaults('tech', @chebtech1);
```

## Discretization for Differential Equations

### Default: Rectangular Collocation (Driscoll-Hale)
- Uses Chebyshev spectral methods on automatically chosen grids
- Discretizes operator as rectangular matrix mapping to fewer grid points
- Grid sequence: 33, 65, ... until convergence
- Set with: `cheboppref.setDefaults('discretization', 'colloc2')`

**Accuracy Limitations**:
- 2nd-order equations: loses 2-3 digits
- 4th-order problems: loses 5-6 digits
- Due to ill-conditioning of discretization matrices

### Alternative: Ultraspherical (Olver-Townsend)
- Uses different bases of ultraspherical polynomials based on operator order
- Results in better conditioned and sparser matrices
- Often yields higher accuracy (closer to machine precision)
- Set with: `cheboppref.setDefaults('discretization', 'ultraspherical')`

### First Kind Collocation
```matlab
cheboppref.setDefaults('discretization', 'colloc1');
```

## When to Switch Algorithms

**Use Ultraspherical for**:
- High-order differential equations (4th order or higher)
- Problems where default accuracy is insufficient
- Well-conditioned computations are critical

**Stick with Rectangular for**:
- Low-order equations (1st or 2nd order)
- Problems where default accuracy is acceptable
- Maximum compatibility and speed

## Chebfun2 Preferences

For functions on a rectangle:

### MaxRank Parameter
Factory default: 512

Determines maximum rank of low-rank approximation.

```matlab
chebfunpref.setDefaults({'cheb2Prefs', 'maxRank'}, value);
```

## Use When

- Solving high-order differential equations
- Default discretization accuracy is insufficient
- Need better conditioned matrices
- Working with Chebfun2 for 2D functions
- Specific spectral method requirements