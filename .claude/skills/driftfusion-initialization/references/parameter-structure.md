# Parameter Object Structure

## pc Class Properties

The `pc` class contains default device properties. Key categories include:

### Layer Properties
- `layer_type`: Array defining each layer's type
- `thickness`: Thickness of each layer
- `epsilon`: Dielectric constant

### Electronic Properties
- `Nc`, `Nv`: Conduction/valence band effective density of states
- `mu_n`, `mu_p`: Electron/hole mobilities
- `Eg`: Band gap

### Optical Properties
- `n_ref`: Refractive index
- `alpha_abs`: Absorption coefficient

### Recombination Parameters
- `B`: Band-to-band coefficient
- `tau_n_SRH`, `tau_p_SRH`: SRH time constants

## CSV File Format

The CSV file should contain columns for each property to override. Required column:
- `layer_type`: Must be defined for all layers

Example structure:
```
layer_type,thickness,epsilon,mu_n,mu_p
electrode,0,1,0,0
layer,100e-7,3.6,1e-4,1e-4
active,500e-7,3.6,2e-4,1e-4
layer,50e-7,3.6,1e-4,1e-4
electrode,0,1,0,0
```