---
name: photoconductivity-trap-kinetics
description: Model photoconductivity rise and decay behavior in semiconductors with electron traps. Use when analyzing two-stage rise with plateau (trap filling) and bi-exponential decay with slow tail (trap emission).
---

# Photoconductivity Trap Kinetics

Model photoconductivity rise and decay when electron traps dominate the transient behavior in semiconductors.

## When to Use
- Extrinsic semiconductors with empty electron traps (N_t)
- Photoconductors showing two-stage rise with plateau
- Decay curves with fast initial drop followed by slow tail
- Trapping dominates over direct recombination (c_ct * N_t >> c_cv * p)

## Rise Phase Analysis (Trap Filling)

### Phase 1: Trap Dominated Rise

**Condition:** Traps mostly empty (n_t << N_t)

**Behavior:**
- Recombination to valence band is negligible compared to trapping
- Rate equation: dn/dt = g - c_ct * n * N_t
- Density rises linearly then saturates at quasi-steady state

**Quasi-steady state density:**
```
n_1st = g / (c_ct * N_t)
```

**Result:** Creates **plateau** in photoconductivity curve while traps fill

### Phase 2: Steady State Rise

**Condition:** Traps filled (n_t ~ N_t), trapping stops (clogs)

**Behavior:**
- Recombination becomes dominant loss mechanism
- Density rises to final steady state

**Final steady state density:**
```
_st = sqrt(g / c_cv)
```

### Trap Density Estimation

**Plateau duration (Δt):** Time required to fill traps

**Trap density formula:**
```
N_t = g * Δt
```
Valid if recombination is neglected during trap filling

## Decay Phase Analysis (Trap Emission)

### Phase 1: Fast Recombination Decay

**Condition:** High carrier density, traps still full (n_t ~ N_t)

**Behavior:**
- Emission from traps is slow compared to recombination
- Decay follows intrinsic hyperbolic decay initially
- Time constant determined by recombination coefficient and initial density

### Transition Point

**Critical density n_1:** Defined when quasi-Fermi level passes through trap level

```
n_1 = (e_tc * N_t) / c_ct
```

### Phase 2: Slow Decay (Tail)

**Condition:** n < n_1, traps begin emitting faster than capturing

**Behavior:**
- Decay governed by depletion of traps
- Rate equation: dn/dt = e_tc * n_t - c_ct * n * (N_t - n_t)
- 'Emission minus retrapping' determines rate
- **Result:** Long tail in decay curve (bi-exponential-like appearance)

## Key Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Trap density | N_t | Total trap density |
| Trap capture coefficient | c_ct | Capture coefficient of traps |
| Trap emission coefficient | e_tc | Emission coefficient from trap to conduction band |
| Recombination coefficient | c_cv | Band-to-band recombination coefficient |
| Critical density | n_1 | Density where trap emission dominates |

## Output
- Two-stage rise profile with plateau characteristics
- Trap density (N_t) estimation from plateau duration
- Bi-exponential decay profile (fast drop + slow tail)
- Critical density (n_1) for transition between decay phases