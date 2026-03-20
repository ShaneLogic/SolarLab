# dfana Class Methods

## Current Calculations

```matlab
Jn = dfana.Jn(sol);  % Electron current density
Jp = dfana.Jp(sol);  % Hole current density
J  = dfana.J(sol);   % Total current density
```

## Quasi-Fermi Levels

```matlab
Efn = dfana.Efn(sol);  % Electron quasi-Fermi level
Efp = dfana.Efp(sol);  % Hole quasi-Fermi level
```

## Recombination Rates

```matlab
Rbtb = dfana.Rbtb(sol);  % Band-to-band recombination
RSRH = dfana.RSRH(sol);  % SRH recombination
Rvsr = dfana.Rvsr(sol);  % Volumetric surface recombination
```

## Carrier Densities

```matlab
n = dfana.n(sol);  % Electron density
p = dfana.p(sol);  % Hole density
c = dfana.c(sol);  % Cation density
a = dfana.a(sol);  % Anion density
```

## dfplot Class Methods

### Current-Voltage Characteristics
```matlab
dfplot.JVapp(sol, position)  % J vs applied voltage
dfplot.JVint(sol, position)  % J vs internal voltage
```

### Spatial Profiles
```matlab
dfplot.n(sol, time_vector)    % Electron density vs position
dfplot.p(sol, time_vector)    % Hole density vs position
dfplot.V(sol, time_vector)    % Potential vs position
dfplot.Efn(sol, time_vector)  % EFn vs position
dfplot.Efp(sol, time_vector)  % EFp vs position
```

### Time Evolution
```matlab
dfplot.J(sol, position)       % Current vs time
dfplot.V(sol, position)       % Voltage vs time
```

### 2D Color Maps
```matlab
dfplot.x2d(sol, 'n')  % 2D map of electron density
dfplot.x2d(sol, 'R')  % 2D map of recombination rate
```