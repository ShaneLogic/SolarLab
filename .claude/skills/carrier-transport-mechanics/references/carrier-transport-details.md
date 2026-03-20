# Carrier Transport - Detailed Formulas

## Thermal Velocity Derivation
From equipartition principle for 3 degrees of freedom:
```
1/2 m* v_rms² = 3/2 kT
v_rms = sqrt(3kT/m*)
```

## Drude Model Complete Equations

### Acceleration Between Collisions
```
delta_v = (F * tau) / m*
```

### Drift Velocities
```
vD_n = -(e * F * tau) / m_n
vD_p = (e * F * tau) / m_p
```

### Current Density
```
j_n = n * (-e) * vD_n = (e² * n * tau / m_n) * F
```

### Conductivity
```
sigma_n = e² * n * tau / m_n
```

### Resistance and Resistivity
```
R = d / (A * sigma)
rho = 1/sigma
```

### Joule Heating
```
Q = sigma * F²
```

## Mobility from Conductivity
```
mu_n = e * tau / m_n
sigma_n = e * n * mu_n
```