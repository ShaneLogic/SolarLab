---
name: differential-equation-solver
description: Solve linear and nonlinear differential equations using chebops. Use when solving boundary value problems, ODEs, integral equations, or PDEs with spectral methods.
---

# Differential Equation Solver

## Linear Differential/Integral Equations

### Basic Structure
Solve equations of the form **L(u) = f** where:
- L is a linear differential or integral operator
- u is the unknown function
- f is the forcing function

### Step-by-Step Process

1. **Define the operator**:
   ```matlab
   L = chebop(@(x,u) diff(u,2) + x.^3*u, [-3,3]);
   ```

2. **Specify boundary conditions**:
   ```matlab
   L.bc = 0;                                    % Dirichlet (zero at both ends)
   L.bc = 100;                                  % Dirichlet (value at both ends)
   L.bc = @(u) diff(u);                         % Neumann (derivative conditions)
   L.bc = 'periodic';                           % Periodic BCs
   ```

   Alternative single-line syntax:
   ```matlab
   L = chebop(op, domain, bc_left, bc_right);
   ```

3. **Solve**:
   ```matlab
   u = L \ f;
   ```

4. **Verify solution**:
   ```matlab
   norm(L(u) - f)    % Should be near machine precision
   ```

### Example: Second-Order ODE
```matlab
% Define operator: u'' + x^3*u = f
L = chebop(@(x,u) diff(u,2) + x.^3*u, [-1,1]);
L.bc = 0;                    % u(-1) = u(1) = 0
f = chebfun(@(x) exp(x), [-1,1]);
u = L \ f;                   % Solve
```

## Time-Dependent PDEs (Operator Exponential)

For PDEs of the form **du/dt = Lu**:

1. Define spatial operator:
   ```matlab
   L = chebop(@(x,u) diff(u,2), domain);  % Second derivative
   ```

2. Compute operator exponential for time t:
   ```matlab
   E = expm(t * L);
   ```

3. Apply to initial condition:
   ```matlab
   u_t = E * u_0;
   ```

**Example: Heat equation** (u_t = u_xx):
```matlab
L = chebop(@(x,u) diff(u,2), [-1,1]);
L.bc = 0;
u_0 = chebfun(@(x) exp(-10*x.^2), [-1,1]);
u_at_t = expm(0.1 * L) * u_0;  % Solution at t = 0.1
```

## Nonlinear Equations

### Automated Nonlinear Backslash
```matlab
u = L \ f;    % Chebfun uses automatic differentiation
```

Chebfun automatically constructs the Frechet derivative (Jacobian).

### Manual Newton Iteration
```matlab
% Construct Jacobian explicitly
J = chebop(@(x,u) linearized_operator, ...);

% Iterate
u_new = u - J \ (L(u) - f);
```

## Unknown Parameters

Include parameters as unknowns with zero derivatives:

```matlab
% Solve with unknown temperature T
L = chebop(@(x,u,T) diff(u,2) + u.^2 - T, [0,1]);
L.bc = @(u,T) [u(0); diff(u,1) - T];
f = 0;

uT = L \ f;      % Returns chebmatrix

% Extract results
u = uT{1};       % Solution function
T = uT{2};       % Parameter value
```

**Important**: Different initial guesses may converge to different solutions.

## Use When

- Solving boundary value problems (BVPs)
- Working with linear or nonlinear ODEs
- Solving integral equations
- Computing time evolution of PDEs
- Finding eigenvalues/eigenfunctions
- Problems with unknown parameters