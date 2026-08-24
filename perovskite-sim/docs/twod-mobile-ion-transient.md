# 2D Single-Mobile-Ion Transient

## Scope

The 2D solver has an explicit research-only transient for one positively
charged mobile-ion species. It is disabled by default. The historical frozen
path retains the `(n, p)` state; opting in with
`ion_dynamics="single_mobile"` selects `(n, p, P)`.

This checkpoint closes the solver-level transport topology. It does not expose
an ion-aware 2D J-V experiment and does not upgrade the repository's 2D scope
claim. A separate explicit post-processor now supplies complete instantaneous
mobile-ion current, while the historical carrier-only terminal-current API
continues to reject snapshots carrying an ion state.

## Discrete Model

The mobile-ion Poisson source is

```text
rho = q * (p - n + N_D - N_A + P - P0).
```

Positive-ion particle fluxes use the same Scharfetter-Gummel constitutive law
as the 1D solver in both supported steric modes. The tensor-product operator
computes internal x- and y-face fluxes and applies blocking flux at all four
outer boundaries. There is no post-step clipping.

For Neumann domains, nodal control-volume widths are half intervals at each
outer endpoint and centred dual widths in the interior. The invariant is

```text
I_P = sum(j, i) P[j, i] * h_y[j] * h_x[i]  [m^-1].
```

The same weights appear in the divergence and terminal inventory gate, so
internal face contributions cancel pairwise on uniform and nonuniform grids.
This is intentionally different from the 1D endpoint convention.

## API

```python
material = build_material_arrays_2d(
    grid,
    stack,
    microstructure,
    lateral_bc="neumann",
    ion_dynamics="single_mobile",
)
state0 = np.concatenate([n0.ravel(), p0.ravel(), material.P_ion0_2d.ravel()])
state1, diagnostics = run_transient_2d(
    state0,
    material,
    V_app=0.1,
    t_end=1e-9,
    atol=ComponentwiseAtol(),
    return_ion_diagnostics=True,
)
```

The componentwise tolerance policy appends an ion block scaled from `P0`.
Scalar tolerances and the frozen two-block return type remain unchanged.

## Fail-Closed Boundaries

The opt-in builder rejects:

- periodic-x topology, whose duplicate endpoints do not define a unique
  physical control-volume partition;
- an effective dual-ion stack;
- a stack without positive ionic diffusion or with zero density on an active
  ion node;
- simultaneous `P_ion_static_1d`, because a mobile initial profile belongs in
  the transient state rather than a static Poisson field.

After a successful Radau solve, the terminal gate rejects nonfinite or negative
electron, hole, or ion densities, ion density above the site limit, and
relative inventory drift above `ion_inventory_rtol`. Trial states may cross a
bound inside the implicit solve, but no terminal bound is hidden by clipping.

## Verification Boundary

Unit tests compare every vertical 2D ion face against the 1D flux source of
truth in both steric modes, prove weighted divergence cancellation on a
nonuniform grid, exercise each terminal failure gate, and pin frozen-mode bit
identity. The real 21-node integration probe uses an accelerated but otherwise
unchanged MAPbI3 ion law: it produces a `5.10e-7` maximum active-ion relative
redistribution with zero measured inventory drift and zero lateral variation.

The standard MAPbI3 density-coordinate path also correctly fails closed on a
longer strong-bias probe when Radau returns a negative terminal electron
density. That is a documented numerical boundary, not a reason to relax the
physical gate. Long-time 2D mobile-ion work will need a positivity-preserving
coordinate or a dedicated DAE integrator.

The independent two-sided cross-node interface-SRH sheet closure is documented
in `docs/twod-two-sided-interface-srh.md` and can compose with this state
topology. The complete instantaneous current is documented in
`docs/twod-mobile-ion-current.md`. Remaining work is public experiment/protocol
wiring and a new content-addressed combined grid/tolerance certificate. Until
those are complete, 2D J-V remains frozen-ion and the full microstructure claim
remains blocked.
