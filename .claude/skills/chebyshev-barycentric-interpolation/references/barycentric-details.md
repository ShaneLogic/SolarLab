# Theoretical Background

## Theorem 4 (Barycentric Formula)

The barycentric formula provides an efficient way to evaluate the Chebyshev interpolant without computing the polynomial coefficients explicitly.

## Numerical Stability

The barycentric formula is numerically stable because:
1. It avoids computing large polynomial coefficients
2. Division operations are well-conditioned
3. The alternating signs (-1)^k provide cancellation that improves accuracy

## Chebyshev Points

For Chebyshev points of the first kind:
```
x_k = cos(kπ/N), k = 0, 1, ..., N
```

## Example Usage

```matlab
% Define function
f = @(x) exp(x) .* sin(10*x);

% Get Chebyshev points
N = 20;
x_nodes = chebpts(N+1);

% Evaluate function at nodes
f_values = f(x_nodes);

% Evaluate interpolant at new points
eval_points = linspace(-1, 1, 100);
p_values = zeros(size(eval_points));

for i = 1:length(eval_points)
    p_values(i) = barycentric_eval(eval_points(i), x_nodes, f_values);
end

% Compare with actual function
actual_values = f(eval_points);
plot(eval_points, p_values, 'b-')
hold on
plot(eval_points, actual_values, 'r--')
legend('Interpolant', 'Actual')
```