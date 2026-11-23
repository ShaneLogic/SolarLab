# 2D Grain-Boundary Area Closure

## Scope

The 2D grain-boundary model represents a vertical finite-width band inside a
named electrical layer. The input contract remains:

- lateral centre `x_position` in m;
- physical band `width` in m;
- electron and hole SRH lifetimes `tau_n` and `tau_p` in s;
- target `layer_role`.

This checkpoint closes the lateral area dependence for Neumann-x domains. It
does not validate the lifetime values against an external MAPbI3 data set or
add a two-sided heterointerface defect plane. A separate solver-level
single-mobile-ion transient now exists, but it is not yet a mobile-ion-complete
2D J-V experiment.

## Finite-Volume Form

For lateral control volume `i`, the geometry builder computes the exact overlap

```text
f_i = length(CV_i intersect GB_band) / length(CV_i),  0 <= f_i <= 1.
```

The SRH source at a target-layer node is then

```text
R_i = R_bulk,i + f_i * (R_GB,i - R_bulk,i).
```

Radiative and Auger terms remain bulk material channels. This mixture is
necessary because averaging `tau` first is not equivalent to averaging the
nonlinear SRH rate. The discrete geometry satisfies

```text
sum_i f_i * width(CV_i) = declared GB width
```

to floating-point precision on uniform and nonuniform lateral grids, including
bands much narrower than one cell.

## Fail-Closed Boundaries

The builder rejects nonfinite or nonpositive widths/lifetimes, bands outside
the finite domain, unknown layer roles, and overlapping bands in the same
layer role. Arrays describing the accepted region are read-only.

A non-empty microstructure currently requires `lateral_bc="neumann"`. The
legacy periodic topology stores both `x=0` and `x=L` as independent nodes and
adds a wrap face, so it has no unique physical control-volume partition. It is
therefore not area-certified. Empty microstructures continue through the
historical periodic path unchanged.

## Evidence Boundary

Unit gates cover exact physical-width quadrature across lateral refinements,
nonuniform meshes, target-role isolation, strict schema validation, immutable
geometry, integrated SRH correction, and bit identity of the empty path. Slow
device tests retain only an internal qualitative `V_oc` trend and finiteness
checks. They are not experimental or external-solver validation.

The full "2D perovskite microstructure" claim remains blocked until two-sided
interface recombination, mobile-ion-complete terminal current and experiment
wiring are implemented and a new content-addressed 2D refinement certificate
is run.
