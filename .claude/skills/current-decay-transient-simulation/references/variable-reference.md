# Variable Reference for Current Decay Transient Simulation

## Key Parameters

### Voltage Profile Parameters

| Variable | Type | Default Value | Description |
|----------|------|---------------|-------------|
| φ_bi | Float | 40 | Built-in potential, the initial applied bias before decay begins |
| β | Float | 10^6 | Profile steepness parameter - controls how rapidly the voltage drops |
| t_end | Float | 1 | End time of the voltage drop, when φ reaches 0 |

### Simulation Parameters

| Variable | Type | Default Value | Description |
|----------|------|---------------|-------------|
| N | Integer | 400 | Number of subintervals for numerical discretization |
| RelTol | Float | 10^-6 | Relative tolerance for numerical solver |
| AbsTol | Float | 10^-8 | Absolute tolerance for numerical solver |

## Voltage Profile Formula

The applied bias φ(t) follows a tanh profile:

```
φ(t) = φ_bi × [1 - tanh(β × t) / tanh(β × t_end)]
```

### Behavior

- **At t = 0:** φ(0) = φ_bi × [1 - 0] = φ_bi (starting bias)
- **At t = t_end:** φ(t_end) = φ_bi × [1 - 1] = 0 (final bias)
- **Transition:** The tanh function provides a smooth, monotonic decrease with steepness controlled by β

### Choosing β

- **β = 10^6** creates an approximately instantaneous voltage drop
- Smaller β values produce a more gradual transition
- Larger β values approach a step function but may cause numerical stiffness

## Expected Results

### Photocurrent Behavior

1. **Short times (t → 0):** 
   - Initial rise in current density
   - Fast transient dynamics not captured by asymptotic solutions
   - Numerical method resolves these timescales

2. **Intermediate times:**
   - Transition from fast dynamics to decay behavior
   - May show characteristic decay patterns

3. **Long times (t → t_end):**
   - Photocurrent approaches asymptotic solution
   - Agreement between numerical and analytical results improves

## Comparison with Asymptotic Solution

The asymptotic solution typically neglects fast timescale transients for mathematical tractability. This protocol's advantage is that the numerical solution:
- Captures the missing fast-time physics
- Still converges to the asymptotic prediction at long times
- Provides a complete picture of the transient response