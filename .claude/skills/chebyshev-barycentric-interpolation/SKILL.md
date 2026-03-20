---
name: chebyshev-barycentric-interpolation
description: Evaluate Chebyshev polynomial interpolants using the barycentric formula, which provides stable and efficient computation of polynomial values at arbitrary points. Use when you need to evaluate Chebyshev interpolants, understand the underlying interpolation algorithm, or implement custom polynomial interpolation.
---

# Chebyshev Barycentric Interpolation

## Barycentric Formula

Use the barycentric formula (Theorem 4) to evaluate the polynomial p(x) at any point.

### Formula Structure

```
p(x) = (Sum'' [ (-1)^k * f(x_k) / (x - x_k) ]) / (Sum'' [ (-1)^k / (x - x_k) ])
```

### Notation Rules

- **Sum''**: Summation from k=0 to k=N
- **Double prime**: Terms for k=0 and k=N (endpoints) must be multiplied by 1/2
- **x_k**: Chebyshev points
- **f(x_k)**: Function values at Chebyshev points

### Special Case

If x is exactly equal to a node x_k, return f(x_k) directly to avoid division by zero.

## Implementation

```matlabnfunction p_val = barycentric_eval(x, x_nodes, f_values)
    % Check if x is exactly a node
    [found, idx] = ismembertol(x, x_nodes, 1e-14);
    if found
        p_val = f_values(idx);
        return;
    end
    
    % Compute numerator and denominator
    N = length(x_nodes) - 1;
    numerator = 0;
    denominator = 0;
    
    for k = 0:N
        weight = (-1)^k;
        if k == 0 || k == N
            weight = weight / 2;  % Double prime notation
        end
        
        term = weight / (x - x_nodes(k+1));
        numerator = numerator + term * f_values(k+1);
        denominator = denominator + term;
    end
    
    p_val = numerator / denominator;
end
```

## Variables

- **x**: Evaluation point (scalar)
- **x_k**: Array of Chebyshev points
- **f(x_k)**: Array of function values at Chebyshev points

## When to Use

- Evaluating Chebyshev polynomial interpolants
- Understanding Chebfun's underlying interpolation method
- Implementing custom polynomial interpolation
- Analyzing interpolation error and stability

## Notes

- The barycentric formula is numerically stable
- Avoids explicit polynomial coefficient computation
- Handles the endpoint weights correctly via the double prime notation
- Direct evaluation at nodes prevents division by zero