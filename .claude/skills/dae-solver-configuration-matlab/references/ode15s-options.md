# ode15s Solver Options Reference

## Common Options for Semiconductor Simulation

### Error Tolerances
```matlab
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
```

### Vector-Specific Absolute Tolerances
```matlabn
AbsTol = [repmat(1e-8, length(P), 1);  % Tolerance for P
           repmat(1e-6, length(phi), 1); % Tolerance for phi
           repmat(1e-8, length(n), 1);   % Tolerance for n
           repmat(1e-8, length(p), 1)];   % Tolerance for p
options = odeset('RelTol', 1e-6, 'AbsTol', AbsTol);
```

### Mass Matrix Specification
If your system has a mass matrix M (singular for DAEs):
```matlab
options = odeset(options, 'Mass', M, 'MassSingular', 'yes');
```

### Event Detection
```matlab
function [value, isterminal, direction] = events(t, u)
    value = u(1) - threshold;  % Event to detect
    isterminal = 1;            % Stop integration
    direction = 1;             % Positive direction only
end

options = odeset(options, 'Events', @events);
```

### Output Function for Progress
```matlab
options = odeset(options, 'OutputFcn', @odeplot);
```

### Jacobian Specification (improves performance)
```matlabno
options = odeset(options, 'Jacobian', @JacobianFunction);
```

## Solver Call Example
```matlab
[t, u] = ode15s(@systemFunction, [t0 tf], u0, options);
```