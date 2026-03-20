# Quasimatrix Examples

## Inner Product Matrix

```matlab
A = [chebfun('1'), chebfun('x')]
G = A' * A
% Returns:
%   2.000000000000000  -0.000000000000000
%  -0.000000000000000   0.666666666666667
```

## Least-Squares Polynomial Fit

```matlab
% Fit function f with degree-5 polynomial
f = chebfun('sin(pi*x)')
A = []
for k = 0:5
    A = [A, chebfun(['x.^', num2str(k)])]
end
c = A \ f

% Reconstruct fit
fit = A * c
plot(f)
hold on
plot(fit, 'r--')
```

## Condition Number Comparison

```matlab
% Monomials (ill-conditioned)
A_mono = []
for k = 0:5
    A_mono = [A_mono, chebfun(['x.^', num2str(k)])]
end
fprintf('Monomial condition number: %.2e\n', cond(A_mono))

% Legendre polynomials (well-conditioned)
A_leg = []
for k = 0:5
    A_leg = [A_leg, legpoly(k)]
end
fprintf('Legendre condition number: %.2e\n', cond(A_leg))

% Chebyshev polynomials (good condition)
A_cheb = []
for k = 0:5
    A_cheb = [A_cheb, chebpoly(k)]
end
fprintf('Chebyshev condition number: %.2e\n', cond(A_cheb))
```