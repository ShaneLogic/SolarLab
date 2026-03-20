# Automatic Differentiation Examples

## Example 1: Simple Chain

```matlab
% Define chain: u -> v -> w
u = chebfun(@(x) x.^2, [0, 1]);
u_ad = adchebfun(u);

v = u_ad.^3;  % v = u^3
w = exp(v);    % w = exp(v)

% Get derivatives
dvdu = v.jacobian;  % 3u^2 = 3x^4
dwdu = w.jacobian;  % exp(v) * 3u^2

% Verify
test = chebfun(@(x) sin(x), [0, 1]);
result = dwdu * test;
manual = exp(u.^3) .* 3*u.^2 .* test;
```

## Example 2: Differential Operator

```matlab
% u = x^2
u = chebfun(@(x) x.^2, [0, 1]);
u_ad = adchebfun(u);

% v = u + du/dx
v = u_ad + diff(u_ad);

% dv/du = I + D (identity + differentiation)
dvdu = v.jacobian;

% Apply to test function
test = chebfun(@(x) x.^3, [0, 1]);
result = dvdu * test;  % x^3 + 3x^2
```

## Example 3: Functional

```matlab
% Define functional: J[u] = ∫ u^2 dx
u = chebfun(@(x) sin(x), [0, pi]);
u_ad = adchebfun(u);

J = sum(u_ad.^2);

% dJ/du = 2u (multiplier operator)
dJdu = J.jacobian;

% Apply to variation
delta_u = chebfun(@(x) 0.1*cos(x), [0, pi]);
delta_J = dJdu * delta_u;
```

## Operator Types

```matlab
% Multiplier: M(f) = m*f
M = chebop(@(f) m .* f);

% Identity: I(f) = f
I = chebop(@(f) f);

% Derivative: D(f) = f'
D = chebop(@(f) diff(f));

% Composition: D∘M
DM = D * M;
```
