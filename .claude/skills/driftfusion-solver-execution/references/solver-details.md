# Solver Configuration Details

## pdepe Equation Form

The MATLAB pdepe solver expects:

$$c(x,t,u,\frac{\partial u}{\partial x})\frac{\partial u}{\partial t} = x^{-m}\frac{\partial}{\partial x}[x^m f(x,t,u,\frac{\partial u}{\partial x})] + s(x,t,u,\frac{\partial u}{\partial x})$$

For Driftfusion:
- `m = 0` (Cartesian coordinates)
- `u = [V, n, p, c, a]`

## Boundary Condition Coefficients

### Dirichlet (Fixed Value)
```matlab
P(x,t,u) = u - u_boundary
Q(x,t,u) = 0
```

### Neumann (Fixed Flux)
```matlab
P(x,t,u) = 0
Q(x,t,u) = 1
```

## Function Generator Examples

### Linear Voltage Ramp
```matlab
coefficients = [0, V_final, scan_rate];  % V_start, V_end, dV/dt
V_app = fun_gen('linear', t, coefficients);
```

### Square Wave Light
```matlab
coefficients = [frequency, duty_cycle, intensity];
I_light = fun_gen('square', t, coefficients);
```

## Multi-Stage Convergence Example

```matlab
% Stage 1: Fast electronic equilibration
par.K_a = 1e-4;  % Increase ionic mobility
par.K_c = 1e-4;
par.tmax = 1e-3;
sol1 = df(par, V_protocol, light_protocol);

% Stage 2: Full ionic mobility
par.K_a = 1e-8;  % Restore original
par.K_c = 1e-8;
par.tmax = 1e2;
sol = df(sol1, V_protocol, light_protocol);
```