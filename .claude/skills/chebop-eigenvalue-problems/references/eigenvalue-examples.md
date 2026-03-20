# Detailed Eigenvalue Examples

## Sturm-Liouville Problem

```matlab
% Solve u'' + λu = 0 with u(-1)=u(1)=0
L = chebop(-1, 1);
L.op = @(x,u) diff(u, 2);
L.bc = 'dirichlet';

[V, D] = eigs(L, 6);

% Eigenvalues should be approximately -π²n²/4
for i = 1:6
    expected = -(i*pi/2)^2;
    actual = D(i,i);
    fprintf('Mode %d: expected %.6f, got %.6f\n', i, expected, actual);
end
```

## Coupled Oscillator System

```matlab
% System: u' = v, v' = -ω²u
omega = 2;
L = chebop(0, 2*pi);
L.op = @(x, u, v) [diff(u) - v; diff(v) + omega^2 * u];
L.lbc = @(u, v) [u - 1; v];  % u(0)=1, v(0)=0

U = L \ 0;
plot(U)
```

## Generalized Problem with Mass Matrix

```matlab
% Solve u'' = λu with different boundary conditions
A = chebop(0, 1);
A.op = @(x,u) diff(u, 2);
A.bc = 'dirichlet';

B = chebop(0, 1);
B.op = @(x,u) u;

[V, D] = eigs(A, B, 5);

% First eigenvalue should be -π²
fprintf('First eigenvalue: %.6f\n', D(1,1));
```