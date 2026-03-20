---
name: chebfun-resolution-strategies
description: Configure sampling, splitting, and resolution preferences for chebfuns. Use when constructing chebfuns for functions with varying complexity, spikes, or piecewise smooth behavior.
---

# Chebfun Resolution Strategies

## Splitting Mode

Chebfuns can represent functions as either:
- **Splitting OFF**: Single global polynomial (one smooth piece)
- **Splitting ON**: Concatenation of smooth pieces (for piecewise smooth functions)

### When to Use Each Mode

**Splitting OFF** (recommended for smooth functions):
- More efficient for smooth functions over large intervals
- Example: `sin(x)` over `[0, 1000]` as single polynomial
- Set: `chebfunpref.setDefaults('splitting', 'off')`

**Splitting ON** (for piecewise smooth or non-smooth functions):
- Handles functions with discontinuities, corners, or varying behavior
- Automatically subdivides intervals at detected features
- Example: Functions with absolute values or piecewise definitions

## Sampling Preferences

### minSamples Parameter
Factory default: 17

Defines minimum starting points for adaptive sampling:
- Sequence: 17, 33, 65... (values are 2^k + 1)
- Non-power-of-2 values are rounded UP to next such value

**Process**:
1. Sample function at `minSamples` points
2. Compute Chebyshev expansion coefficients
3. If tail is negligible, stop
4. Otherwise, double grid size (minus 1) and repeat

**When to Increase**:
- Functions with narrow spikes that might be missed (e.g., `exp(-30*(x-0.47)^2)`)
- Need to capture sharp features between grid points

### resampling Parameter
Factory default: 'off'

- 'off': Reuses values from nested Chebyshev grids (more efficient)
- 'on': Forces recomputation at every grid step

**When to Use 'on'**:
- Functions that depend on grid size itself (e.g., `length(x)*sin(15*x)`)
- Working with Chebops (where matrices change with grid size)

## Length Limits

### splitLength (Splitting ON Mode)
Factory default: 129 (configured as 160)

- If polynomial segment of `splitLength` is insufficient, interval is subdivided
- Can increase to allow longer segments before splitting (up to 513)

### maxLength (Splitting OFF Mode)
Factory default: 2^16 + 1 = 65537

- Defines "giving-up point" for global polynomial representation
- If function cannot be resolved within `maxLength` points, constructor fails with warning

**Adjustment Guidelines**:
- Lower (e.g., 50): Examine low-degree interpolants
- Raise significantly (millions): Resolve high-degree smooth functions (e.g., `|x|^(5/4)`)

## Piecewise Smooth Functions

Many functions of interest are not globally smooth but piecewise smooth:
- Chebfun represents these as **concatenations of smooth pieces**
- Each smooth piece has its own polynomial representation
- Structure allows handling non-smooth functions by breaking them into manageable smooth segments

## Use When

- Constructing chebfuns for functions with complex behavior
- Dealing with piecewise smooth functions
- Capturing narrow spikes or sharp features
- Optimizing performance for smooth vs. non-smooth functions
- Troubleshooting construction failures or poor resolution