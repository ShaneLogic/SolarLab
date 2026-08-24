# 2D Two-Sided Cross-Node Interface SRH

## Scope

The 2D solver has an explicit research-only interface recombination mode for
the clamp-passive, two-sided Pauwels-Vanhoutte slice already used by the 1D
production transient. It is disabled by default. Opt in with
`interface_srh="two_sided_cross_node"` on a Neumann-x grid.

This checkpoint closes the surface-to-volume source topology. It does not add
interface charge, interface-plane projection, shared occupancy, a local QSS
root, or dynamic interface states. It also does not expose the option through
the existing 2D J-V experiment.

## Discrete Model

For interface row `j` with left and right bulk samples `j-1` and `j+1`, the
two surface rates are

```text
R_A = PV(n[j+1], p[j-1]; n_R,eq * p_L,eq)
R_B = PV(n[j-1], p[j+1]; n_L,eq * p_R,eq).
```

Each channel uses the same calibrated capture velocities and trap-level
densities as the 1D material cache. Each is clamped independently to
`max(R, 0)`. The supported smooth operating slice requires both raw rates to
be strictly positive; tests report the clamp-active boundary separately.

The sheet rate `[m^-2 s^-1]` becomes a carrier sink at the interface row:

```text
S[j, i] = (R_A[i] + R_B[i]) / h_y[j]  [m^-3 s^-1]
dn/dt[j, i] -= S[j, i]
dp/dt[j, i] -= S[j, i].
```

Here `h_y[j]` is the same half-endpoint/interior-centred dual width used by
the 2D control-volume area. Consequently,

```text
sum(j, i) S[j, i] * h_y[j] * h_x[i]
    = sum(i) (R_A[i] + R_B[i]) * h_x[i],
```

so refinement cannot change the integrated sheet strength merely by changing
the interface-row cell width.

## API

```python
material = build_material_arrays_2d(
    grid,
    stack,
    microstructure,
    lateral_bc="neumann",
    interface_srh="two_sided_cross_node",
)
```

The stack must explicitly set `interface_two_sided=True`, declare an
`InterfaceDefect` for each active capture sheet, and resolve exact
`idx-1/idx+1` carrier samples. The mode composes with the research-only
single-positive-mobile-ion state, while remaining independent of the ion
inventory operator.

## Fail-Closed Boundaries

The builder rejects periodic-x sheet topology, missing defect or cross-node
metadata, interface projection, shared occupancy, interface-plane closure or
generation, dynamic interface states, interface charge, heterojunction
de-spiking, and the legacy QSS/generation environment escape paths. Unknown
mode labels also fail before material assembly.

Zero capture velocity is treated as an inactive sheet. An opt-in request with
no active sheet fails rather than silently becoming the off path. Nonfinite
carrier input or nonfinite surface rates fail at evaluation; no carrier state
is clipped.

## Verification Boundary

The unit suite pins exact lateral-uniform parity with the 1D production
source, area conservation on the tensor control volume, independent clamp
behaviour, immutable diagnostics, default-off bit identity, equal electron and
hole loss, and coexistence with the conservative mobile-ion block. A real
small-grid Radau transient retains lateral uniformity and positive carriers
while removing electron-hole pairs only when the interface mode is enabled.

This is an internal solver closure, not an external interface-defect
validation. The independent mobile-ion-complete current evaluator is now
available, but public protocol wiring and a content-addressed combined 2D
refinement certificate remain required before the repository can enlarge its
2D microstructure claim.
