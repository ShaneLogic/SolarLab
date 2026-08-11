# Physical interface response for CBO scans

## Scope

SolarLab provides opt-in abrupt-interface boundaries for the cancellation-safe
quasi-Fermi steady-state solver. Both replace the ordinary Scharfetter-Gummel
face at each electrical layer boundary with a locally eliminated four-state
plane:

~~~text
left bulk <-> (n_left, p_left) <-> (n_right, p_right) <-> right bulk
~~~

The `deduplicated_qss` topology uses the historical shared boundary node. The
`two_sided_trace` topology removes that node, uses strict per-material bulk
reservoirs, exact left/right half-cell distances, and exact dielectric series
capacitance. The plane solves reciprocal cross-interface transport and
shared-occupancy interface SRH locally. Only its conservative bulk flux enters
the adjacent finite-volume cells, so the ordinary SG face cannot bypass or
double-count the barrier. Default transient and steady-state paths remain
unchanged.

## Interface transport models

The model must be selected explicitly for a CBO claim:

- fermi_richardson is the existing DOS-bounded Fermi-edge model with
  reciprocal Richardson supply. It is retained for regression continuity, but
  its single-level state <= DOS closure is not a general 3D degenerate-carrier
  density law.
- fermi_dirac_richardson is an opt-in 3D carrier-statistics model. It
  obtains the reduced chemical potential from
  state/DOS = F_1/2(eta), evaluates one-way thermionic supply with
  F_1(eta - barrier/kT), and recovers the SCAPS Boltzmann expression in the
  dilute limit. It removes the artificial state <= DOS cap. With
  `two_sided_trace` it has passed the internal Jsc grid gate described below,
  but it has not passed external SCAPS validation.
- scaps_thermionic implements the common-Richardson Boltzmann thermionic law
  published for SCAPS. It is a compatibility model only while every interface
  state remains dilute. The default certificate requires state/DOS <= 0.1;
  matching a SCAPS curve does not override this condition.
- scaps_thermal_velocity is a reciprocal adjacent-layer thermal-velocity
  alternative. It is also a Boltzmann model and uses the same dilute-state
  certificate.

The SCAPS equations and assumptions are documented in the
[SCAPS manual](https://scaps.elis.ugent.be/SCAPS%20manual%20most%20recent.pdf),
the [original SCAPS model paper](https://scaps.elis.ugent.be/Burgelman%20TSF%202000a.pdf),
and the [heterojunction thermionic/tunnelling paper](https://scaps.elis.ugent.be/Verschraegen%20TSF%202007.pdf).

## Scan protocol

fixed_contacts is the default CBO boundary policy. It varies ETL affinity while
holding the configured electrostatic contact boundary fixed.
recomputed_built_in also changes V_bi and therefore represents a different
physical experiment; results from the two policies cannot share one grid
certificate.

The scanner starts at the declared reference CBO, continues independently in
both directions, inserts bridge points when needed, and refines the 1% Jsc
drop bracket to the requested minimum CBO step. A coarse-grid reference state
may seed the next grid. Exact face drops are conservatively integrated onto
the target grid; a nodal-coordinate predictor may locate the new nonlinear
basin, but the target state is always re-solved and certified in face-drop
coordinates. A failed cross-grid warm start first retries a cold target-grid
solve; a cold target-grid failure may then use a certified 0.8N coarse-grid
basin. The JSON records warm-start failures, cold recoveries, and predictor
recoveries separately.

`het_recomb_despike` belongs to the old shared-node bulk-recombination model.
It is incompatible with `two_sided_trace`. The API rejects a nonzero value;
the CLI can set it to zero only through the explicit
`--disable-legacy-heterojunction-despike` option and records both values.

For the physical interface path, final nonlinear coordinates are the
dimensionless quasi-Fermi drops on transport faces. For each carrier:

~~~text
d_i = z_(i+1) - z_i
d_last = -sum(d_i)
~~~

Both contact increments therefore remain pinned. SG currents consume d_i
directly, while nodal increments reconstructed by cumulative sums feed Poisson
and recombination. This prevents a conductive layer's current-controlling face
drop from disappearing when two O(1) nodal increments differ by only O(1e-14).
Default non-interface QF solves retain the historical nodal coordinates.

For the historical deduplicated topology, the recommended mesh allocates twice
as many intervals to the 800 nm absorber and uses moderate interface
clustering:

~~~text
interval weights = (1, 2, 1)
layer alphas     = (3, 3.25, 3)
~~~

This avoids both an under-resolved absorber interface and the 0.01 nm-scale
thin-layer cells produced by uniformly aggressive clustering. The registered
two-sided gate below deliberately uses the unmodified stack grid protocol and
records its actual post-removal interval counts.

## Certificates

A top-level certified=true requires every enabled gate to pass:

1. Every requested point passes normalized cell residual, integrated electron
   and hole continuity, face-current spread, Poisson, and local-interface QSS
   limits.
2. Interface statistics stay inside the selected model's state/DOS range.
3. A grid ladder has at least three unique actual grids, one physical, mesh,
   and QF-coordinate protocol, a critical-CBO union envelope no wider than
   10 meV, reference Jsc spread below 1%, and consecutive critical-point
   shifts that contract by a ratio no greater than 0.9.
4. Full J-V metrics require at least three strict nested voltage grids. The
   finest certified branch is solved once; coarser metrics use exact subsets
   of that branch. Between the two finest grids, the default limits are 2 mV
   for Voc, 0.001 absolute for FF, and 0.0005 absolute for PCE. Successive
   changes must contract below 0.8 while they remain above 10% of the relevant
   absolute tolerance. Full J-V CLI runs without a voltage ladder fail closed.
5. When a SCAPS reference is supplied, its raw export, source deck, and
   parameter-manifest hashes and exact scan protocol pass the provenance
   audit. The normalized response, critical-interval distance, and independent
   SCAPS bracket width must then pass their declared tolerances.

The schema 1.5 JSON distinguishes numerical_certified for the finest
individual solve from the top-level combined result. A failed finer grid
writes grid_failure and all completed grid_runs, then exits with status 1.
The voltage certificate records requested and retained point counts, metric
changes, contraction ratios, MPP extraction mode, and any voltage-continuation
bridge count.

The model/transmission sensitivity runner is deliberately weaker. Its schema
1.2 can mark a run single_grid_screen_passed, but it always leaves certified
false and directs the candidate to the grid-ladder runner. A curve fit on one
mesh is therefore never promoted to a physical CBO threshold.

## Independent SCAPS reference protocol

Start from direct, non-interpolated SCAPS outputs and the exact SCAPS device
deck used to generate them. Populate
docs/scaps-cbo-parameter-manifest.template.json with the complete layers,
contacts, interfaces, illumination, and numerical settings. The importer
rejects empty sections, reordered CBO values, missing reference CBO, or an
unattested export, then content-addresses all three source artifacts:

~~~bash
python scripts/import_scaps_cbo_reference.py \
  --csv path/to/direct-scaps-cbo-export.csv \
  --source-deck path/to/device.def \
  --parameter-manifest path/to/scaps-cbo-parameters.json \
  --out tests/integration/scaps_cbo_dense_reference.json \
  --solver-version 3.3.11 \
  --extracted-at 2026-08-11 \
  --confirm-independent-scaps-export
~~~

The CSV columns are delta_ec_eV and Jsc_mA_cm2, with optional Voc_V,
FF_percent, and PCE_percent. To resolve the present 1% Jsc onset gate, direct
SCAPS points must bracket the drop within 0.02 eV; interpolation does not count
as independent coverage.

## Reproducible command

Run the internally converged two-sided Fermi-Dirac Jsc ladder:

~~~bash
python scripts/run_interface_cbo_scan.py \
  --config configs/scaps_mirror_v2.yaml \
  --out outputs/interface-cbo/scan-two-sided-fd-grid-40-50-60.json \
  --delta-min 0 --delta-max 0.5 --delta-step 0.25 \
  --grid-ladder 40 50 60 \
  --short-circuit-only \
  --interface-topology two_sided_trace \
  --interface-transport-model fermi_dirac_richardson \
  --disable-legacy-heterojunction-despike \
  --boundary-policy fixed_contacts \
  --minimum-delta-step 0.0005 \
  --maximum-delta-step 0.05 \
  --maximum-grid-envelope-eV 0.01 \
  --maximum-successive-shift-ratio 0.9
~~~

Use scaps_thermionic only for a SCAPS-compatibility comparison; its Boltzmann
validity gate remains active. Omit --short-circuit-only only after the Jsc
onset protocol is accepted, because FF and PCE require complete, certified J-V
curves and a bracketed Voc on every CBO point.

Run the internally converged full-JV development gate after the Jsc ladder:

~~~bash
python scripts/run_interface_cbo_scan.py \
  --config configs/scaps_mirror_v2.yaml \
  --out outputs/interface-cbo/scan-two-sided-fd-grid-40-50-60-voltage-29-57-113-quadratic-mpp.json \
  --delta-min 0 --delta-max 0.4 --delta-step 0.4 \
  --grid-ladder 40 50 60 \
  --voltage-grid-ladder 29 57 113 --V-max 1.4 \
  --mpp-interpolation local_quadratic \
  --interface-topology two_sided_trace \
  --interface-transport-model fermi_dirac_richardson \
  --disable-legacy-heterojunction-despike \
  --boundary-policy fixed_contacts \
  --minimum-delta-step 0.0005 --maximum-delta-step 0.05 \
  --maximum-grid-envelope-eV 0.01 \
  --maximum-successive-shift-ratio 0.9 \
  --maximum-voc-change-mV 2 \
  --maximum-ff-change 0.001 \
  --maximum-pce-change 0.0005 \
  --maximum-voltage-successive-change-ratio 0.8 \
  --voltage-contraction-noise-floor-fraction 0.1
~~~

## Current two-sided evidence (2026-08-11)

For `two_sided_trace`, `fermi_dirac_richardson`, fixed contacts, transmission
1.0, and explicitly disabled legacy de-spike, the nominal N_grid=40/50/60
ladder passes every internal numerical and statistics gate:

| N_grid | Actual intervals | 1% Jsc critical interval (eV) | max state/DOS |
| ---: | ---: | ---: | ---: |
| 40 | 37 | 0.389063-0.389453 | 0.730 |
| 50 | 46 | 0.384766-0.385156 | 0.629 |
| 60 | 58 | 0.382422-0.382813 | 0.573 |

The union envelope is 0.382421875-0.389453125 eV, or 7.03125 meV. Midpoint
shifts contract from 4.296875 meV to 2.34375 meV, ratio 0.545. Reference-Jsc
relative spread is 3.12e-4. Across retained scan states, maximum face-current
spread is 7.79e-5 A/m2 and maximum local QSS residual is 7.17e-8, both below
their unchanged gates.

The existing sparse SCAPS file does not provide external certification. Its
audit lacks protocol metadata and source hashes, and its 0.4-0.5 eV onset
bracket is 0.1 eV wide. At N_grid=60 the matched normalized Jsc maximum error
is 0.38849: SolarLab gives 0.61154 at 0.4 eV while the sparse SCAPS curve gives
1.00003. The internal envelope is therefore a grid-converged development
result for the declared SolarLab model, not an externally validated CBO.

### Stage 4.3 full-JV evidence

The combined N_grid=40/50/60 run above repeats the accepted spatial Jsc gate
and solves nested 29/57/113-point J-V metrics only on the finest N_grid=60
mesh. The actual spatial interval counts remain 37/46/58 and the Jsc envelope
remains 0.382421875-0.389453125 eV (7.03125 meV). The voltage branch stops at
the first point on the common 29-point grid after a certified current crossing,
so every nested subset brackets Voc without entering unnecessary deep forward
injection.

Local-quadratic MPP extraction is explicit and bounded. It uses only the three
certified power samples around the discrete maximum, requires negative local
curvature, and accepts the fitted vertex only inside that voltage bracket.
Otherwise it returns the sampled maximum. The global compute_metrics default
remains sampled MPP.

| delta_Ec (eV) | retained points (29/57/113) | final dVoc (mV) | final dFF | final dPCE | change ratios (Voc/FF/PCE) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 23/45/89 | 1.689 | 4.647e-4 | 2.385e-4 | 0.212/0.012/0.031 |
| 0.4 | 15/29/57 | 0.998 | 3.535e-4 | 6.670e-7 | 0.189/0.179/0.152 |

Both points pass their numerical and voltage-grid certificates without voltage
bisection bridges. The previously failing N60 0-to-12.5 mV step was traced to
a same-grid voltage warm start being forced through cancellation-sensitive
nodal coordinates even though exact edge drops were available. Same-grid
voltage continuation now starts directly from those edge drops. If that direct
attempt leaves Newton's basin, the solver retries the same target with the
nodal predictor before any explicitly enabled voltage bisection. The predictor
also remains necessary for cross-grid or legacy states without exact drops.
The JSON records predictor fallback attempts and failures separately from
voltage bridge points, so this recovery cannot be mistaken for grid refinement.
Both final J-V branches record zero predictor fallbacks as well as zero bridge
points.

This result certifies voltage sampling at delta_Ec=0 and 0.4 eV. It does not
certify an FF or PCE critical CBO: with only those two full-JV CBO samples, both
reported 1% intervals are still [0, 0.4] eV. Adaptive full-JV CBO refinement is
required before quoting an FF/PCE onset. No independent SCAPS file was supplied
to this run, so the external-validation status is also unchanged.

## Historical deduplicated evidence (2026-08-11)

For fermi_richardson, fixed contacts, weights (1,2,1), alphas (3,3.25,3), and
final edge_drop QF coordinates, N30 through N60 all complete numerically:

| Intervals | 1% Jsc critical interval (eV) | max state/DOS |
| ---: | ---: | ---: |
| 30 | 0.390625-0.391016 | 0.887 |
| 40 | 0.386719-0.387109 | 0.759 |
| 50 | 0.382813-0.383203 | 0.664 |
| 60 | 0.380078-0.380469 | 0.600 |

The original N60 failure was deterministic, not a loose local QSS root:
repeated residual evaluations had zero observed noise, the maximum defect was
the electron equation next to the PVK/ETL interface, the equilibrated outer
Jacobian condition number was about 5.7e9, and the stalled nodal Newton step
was about 1.3e-14. Face-drop coordinates remove that representation floor.
The final N60 trace has maximum face-current spread 1.17e-5 A/m2 and local QSS
residual 2.53e-15, both below their unchanged gates.

N30 is pre-asymptotic: including N30 makes the four-grid envelope 10.938 meV
and retains an initial shift ratio of 1.0. The consecutive N40/N50/N60 ladder
passes the declared grid certificate:

- union envelope: 0.380078-0.387109 eV (7.031 meV < 10 meV);
- midpoint shifts: 3.906 meV, then 2.734 meV;
- shift ratio: 0.70 < 0.90;
- reference-Jsc relative spread: 1.75e-5 < 0.01.

This is an internally grid-certified development envelope for the declared
model and mesh protocol. It is not yet an externally validated physical CBO
threshold. The finest run's normalized SCAPS Jsc error is 0.4744, above the
allowed 0.05, so the combined top-level certificate remains false.
Conversely, the SCAPS Boltzmann model can approach the sparse reference trend
on a coarse grid, but its observed state/DOS ratio reaches 1.66-3.05, far
beyond the default 0.1 dilute-state limit.

### Fermi-Dirac/transmission model-form test

The new fermi_dirac_richardson model was first screened at N=20. With a
constant cross-interface transmission of 0.001, its two sparse SCAPS onset
points appeared close: the maximum normalized Jsc error was 0.00805 and the
1% SolarLab interval was 0.42969-0.43008 eV. That was only a single-grid
calibration result, not a certificate.

The formal N=40/50/60 ladder disproved that apparent agreement:

| Intervals | 1% Jsc critical interval (eV) | Jsc(0.4)/Jsc(0) | Jsc(0.5)/Jsc(0) |
| ---: | ---: | ---: | ---: |
| 40 | 0.346875-0.347266 | 0.161959 | 0.003480 |
| 50 | 0.335547-0.335938 | 0.103500 | 0.002225 |
| 60 | 0.328906-0.329297 | 0.079191 | 0.001705 |

All three individual solves and their Fermi-Dirac statistics certificates
passed. The midpoint shifts contracted from 11.328 meV to 6.641 meV, but the
union envelope is 0.328906-0.347266 eV, or 18.359 meV, which exceeds the
10 meV grid limit. At N=60 the maximum normalized SCAPS Jsc error is 0.92084.
The current sparse reference also fails provenance audit and gives only a
0.1 eV onset bracket, wider than the allowed 0.02 eV.

Therefore the transmission=0.001 adjustment is a mesh-dependent compensation
inside the present deduplicated interface-node topology. It must not replace
the default model and does not yield a physical CBO critical value. The useful
development result is that a more complete Fermi-Dirac supply law alone is
insufficient; changing one scalar transmission cannot repair the topology's
grid dependence.

## Remaining physical work

- Generate and audit a dense independent SCAPS export around 0.37-0.43 eV,
  then explain or resolve the remaining normalized-response mismatch without
  fitting a single scalar on one mesh.
- Assemble the tested local implicit Jacobian into a future analytic global
  Newton path. The current finite-difference outer Newton already re-eliminates
  the local state on every perturbation, but an assembled Jacobian would reduce
  cost and make conditioning diagnostics more direct.
- Once the dense external Jsc gate exists, extend the internally converged
  voltage protocol to external FF/PCE comparison and sensitivity studies for
  transmission, DOS, defects, capture velocities, temperature, and contacts.
- Refine the full-JV FF/PCE CBO brackets adaptively on the accepted spatial and
  voltage protocols; the present two-point [0, 0.4] eV brackets are not usable
  physical critical values.
