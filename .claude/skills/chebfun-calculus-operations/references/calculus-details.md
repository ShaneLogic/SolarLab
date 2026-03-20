# Differentiation Stability Analysis

## Error Accumulation

When taking successive derivatives:
- Round-off errors in f are amplified
- Each differentiation can multiply errors by a factor related to the function's frequency content
- After n differentiations, errors may be amplified by O(k^n) where k depends on the function

## Example: Error Growth

```matlabnf = chebfun('exp(x)')
for n = 1:10
    fn = diff(f, n)
    % Compare with exact derivative
    exact = chebfun(['exp(x)'])
    err = norm(fn - exact, inf)
    fprintf('n = %d, error = %.2e\n', n, err)
end
```

## Delta Function Handling

```matlab
% Function with jump
f = chebfun('sign(x)', [-1, 1])

% Derivative includes delta function at x = 0
f_prime = diff(f)

% Integral of derivative recovers jump
integral = sum(f_prime)
% Should equal 2 (jump from -1 to +1)
```

## Non-smooth Operation Breakpoint Detection

```matlab
% Track where breakpoints are introduced
f = chebfun('sin(pi*x)', [-2, 2])
g = abs(f)

% Check number of pieces
fprintf('Number of funs in g: %d\n', length(g.funs))
% Should be 3 pieces: [-2, -1], [-1, 1], [1, 2]
```