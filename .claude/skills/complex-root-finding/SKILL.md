---
name: complex-root-finding
description: Find genuine complex roots of chebfuns using Chebfun ellipse filtering to avoid spurious roots, and control accuracy with recursion settings. Use when locating complex zeros of functions defined on real intervals.
---

# Complex Root Finding

## The Problem: Spurious Roots

A polynomial approximating a fun on an interval (e.g., [-1, 1]) will have complex roots that are **spurious** - unrelated to the actual function due to approximation properties.

## Finding Genuine Complex Roots

### Basic Command

```matlab
% Find genuine complex roots near the interval
f = chebfun(@(x) exp(x) - 2, [-1, 1]);
roots_complex = roots(f, 'complex');
```

### How Filtering Works

1. First computes all roots: `roots(f, 'all')`
2. Filters results using "Chebfun ellipses"
3. Returns only roots inside the specific ellipse

### Chebfun Ellipse Definition

For interval [-1, 1] and fun length L:

- Map: `(z + 1/z) / 2`
- Circle: `|z| = r` where `r^(-L) = delta`
- Parameter: `delta = sqrt(eps)` (Chebfun tolerance)

The ellipse is the image of this circle under the map.

## Accuracy Control

### Understanding Accuracy Degradation

Complex roots lose accuracy as they move away from the definition interval:
- Near interval: ~11 digits accuracy
- Far from interval: ~5 digits accuracy
- Very far: roots may disappear entirely

### Using the 'norecursion' Flag

For complicated chebfuns with many complex roots:

```matlab
% Standard computation (may miss some roots)
r1 = roots(f, 'complex');

% With recursion control (more accurate, slower)
r2 = roots(f, 'complex', 'norecursion');
```

### Trade-offs

| Setting | Speed | Accuracy |
|---------|-------|----------|
| Default | Fast | May miss roots near interfaces |
| 'norecursion' | Slower | Better accuracy, fewer misses |

### Verification

```matlab
% Check accuracy by evaluating at roots
residuals = abs(f(r2));
max_residual = max(residuals);

% Lower residuals indicate better accuracy
```

## When to Use

- Finding complex roots of functions defined on real intervals
- Avoiding spurious polynomial roots
- Maximizing accuracy for complex root finding
- Analyzing root distribution in complex plane
- Working with functions having many complex roots
