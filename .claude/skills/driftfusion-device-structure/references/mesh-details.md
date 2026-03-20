# Mesh Configuration Details

## Spatial Mesh Types

### Linear Mesh
```matlab
par.xmesh_type = 'linear';
par.layer_points = [50, 100, 50];  % Points per layer
```

### Error Function-Linear Mesh
```matlab
par.xmesh_type = 'erf-linear';
par.layer_points = [50, 100, 50];
par.xmesh_coeff = 5;  % Higher = more boundary density
```

## Time Mesh Examples

### For J-V Scan
```matlab
par.tmesh_type = 'linear';
par.tmax = 100;  % Total simulation time
par.MaxStepFactor = 0.1;  % Max step as fraction of tmax
```

### For Transient Response
```matlab
par.tmesh_type = 'logarithmic';
par.tmax = 1e6;  % Long timescale
par.MaxStepFactor = 0.01;
```

## Interface Grading Examples

```matlab
% In build_device call:
par.grading_type = {'lin_graded', 'exp_graded', 'constant'};

% For constant grading, CSV must include interface values:
% layer_type,epsilon
% layer,3.6
% interface,3.3  % Required for constant grading
% layer,3.6
```