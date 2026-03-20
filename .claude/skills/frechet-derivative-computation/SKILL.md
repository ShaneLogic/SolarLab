---
name: frechet-derivative-computation
description: Compute Fréchet derivatives of nonlinear operators using automatic differentiation with adchebfun objects. Use when determining how one variable depends on another in a sequence of operations or performing sensitivity analysis. Requires Chebfun Version 5+.
---

# Computing Fréchet Derivatives via Automatic Differentiation

## Overview

Fréchet derivatives represent the linearization of nonlinear operators. In Chebfun v5+, use automatic differentiation (AD) with `adchebfun` objects to compute these derivatives.

## Procedure

### Step 1: Select Basis Variable

Choose the independent variable for derivative computations:

```matlab
% Define basis variable on domain
u = chebfun(@(x) x.^3, [0, 1]);
```

### Step 2: Convert to adchebfun

Enable automatic differentiation:

```matlab
% Convert to adchebfun
u_ad = adchebfun(u);
```

### Step 3: Define Dependent Variables

Redefine dependent variables using the adchebfun to propagate derivative information:

```matlab
% Dependent variable v = u^2
v = u_ad.^2;

% Another dependent variable w = v + dv/dx
w = v + diff(v);
```

### Step 4: Access Derivative Operator

```matlab
% Get Fréchet derivative dv/du
dvdu = v.jacobian;

% Get Fréchet derivative dw/du
dwdu = w.jacobian;
```

### Step 5: Interpret and Verify

```matlab
% Apply derivative operator to test function
test_fun = chebfun(@(x) x, [0, 1]);
result = dvdu * test_fun;

% Manual calculation for comparison
% If v = u^2 and u = x^3, then v = x^6
% dv/du = 2u = 2x^3
expected = 2 * u_ad .* test_fun;
```

## Understanding the Result

### Linear Operator Structure

The result is a linear `chebop` representing the Fréchet derivative:

```matlab
% Example: dv/du for v = u^2
% Result is a multiplier operator: 2u

class(dvdu)  % 'chebop'
```

### Common Operator Types

1. **Multiplier Operator**: `M(f) = m(x) * f(x)`
2. **Identity Operator**: `I(f) = f(x)`
3. **Differentiation Operator**: `D(f) = f'(x)`
4. **Combinations**: `I + D`, `M1 + M2*D`, etc.

### Example Interpretations

```matlab
% v = u^2, u = x^3
% dv/du = 2u = 2x^3 (multiplier operator)

% w = v + dv/dx, v = u^2
% dw/du = 2u + d(2u)/dx = 2u + 2u'
% = 2x^3 + 6x^2 (multiplier + derivative)
```

## When to Use

- Computing sensitivity of outputs to inputs
- Linearizing nonlinear operators
- Analyzing how changes propagate through computations
- Performing variational analysis
- Computing derivatives of functionals

## Requirements

- Chebfun Version 5 or higher
- Functions must be compatible with automatic differentiation
