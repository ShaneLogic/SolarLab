# Two-dimensional research examples

This directory exercises the new core with a connected lateral and stack grid.
The first verified slice assembles the classical Poisson and carrier continuity
equations and their analytic Jacobian. Its exported arbitrary states are not
accepted steady solutions. Nonlinear steady validation is tracked separately.

## Inspect the equations

From the repository root, using the locked environment described in the
[one-dimensional guide](../research_1d/README.md):

```sh
.venv/bin/python -B -m examples.research_2d.dc_assembly \
  --output outputs/2d_assembly.json
.venv/bin/python -B -m examples.research_2d.dc_assembly \
  --read outputs/2d_assembly.json
```

The example uses five lateral nodes and thirteen stack nodes, with 195 coupled
unknowns, 52 lateral faces and 60 stack faces. It exports coordinates, physical
fields, generation, residuals, the sparse Jacobian, face fluxes and terminal
currents. An explicitly nonuniform lateral state produces lateral carrier and
electric displacement fluxes. A separate uniform-lateral state checks the
reduction to the one-dimensional column equations.

The reader opens existing data without solving. Use a fresh output path for a
new assembly run; existing evidence is preserved.

## Geometry and units

`x` is lateral and `y` follows the layer stack. Flattened fields use C order with
shape `(Ny, Nx)`. Volumes represent control areas per unit out-of-plane depth;
face measures are lengths. Endpoint control widths are physical half widths,
so the geometry must not be equated with the historical one-dimensional
integer-endpoint convention without the documented weight conversion.

| Quantity | Unit |
| --- | --- |
| Control area | m² |
| Face measure | m |
| Particle flux density | m⁻² s⁻¹ |
| Integrated carrier balance per depth | m⁻¹ s⁻¹ |
| Integrated Gauss balance per depth | C/m |
| Cross-section or terminal current per depth | A/m |
| Terminal current divided by lateral width | A/m² |

The initial scope uses fixed carrier reservoirs at the stack ends, zero normal
particle and displacement flux at the lateral boundaries, constant carrier
mobility and the supported Beer–Lambert generation binding. Ions, explicit
defects, grain boundaries, field-dependent transport, periodic boundaries and
spectral/coherent/incoherent optical declarations are outside this entry.
The algebra checks do not establish a steady solution, mesh convergence or
final scientific qualification.
