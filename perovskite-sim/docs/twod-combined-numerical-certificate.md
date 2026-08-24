# Combined 2D Mobile-Ion/Interface-SRH Numerical Certificate

## Certified Slice

`twod-mobile-ion-interface-srh-v1` is an internal numerical certificate for
one explicit research topology:

- a Neumann-x tensor grid with matched 4/6/8 x and per-layer y intervals;
- one finite-width vertical grain boundary mapped by physical overlap;
- ohmic carrier contacts;
- one blocking positive mobile ion on a finite-site lattice;
- one two-sided cross-node `InterfaceDefect` SRH sheet with every clamp
  inactive;
- complete electron, hole, ionic, and instantaneous displacement current;
- an illuminated ascending 0.0/0.05/0.10 V history with a 10 ns dwell at each
  voltage;
- componentwise absolute-tolerance factors 1, 0.1, and 0.01.

Each of the nine cells uses a canonical explicit
`jv-2d-execution-protocol-v1`. The outer numerical protocol fixes the grid and
tolerance ladders before execution.

## Source-Clean Result

The single-threaded matrix completed 9/9 cells with no failed or missing cell
and no unconverged dimension:

| Identity | SHA-256 or value |
|---|---|
| Source commit | `0c9eb2600434020f3a30cbf76318ce40a2bc4221` |
| Source fingerprint | `b9bdc519cd6fcbcb74c8c2353c61355b0becdf9d19657f4d7cd7f77e7a5f0d2b` |
| Run ID | `89d108b8817fb4af5d0749bd5848efada9dda99a1b559ed395a8cb0603eaa55b` |
| Certificate | `b02bc4f8b3b5d470d599f6dacde746b26c263591aafecd14cc6c890a94b677dd` |
| Manifest | `e1c4602ee004955f0b66db1e2ee59a5d29e8109d59208154d105e8da54c6d7b5` |
| Numerical protocol | `2b5371b8f89c2c4a749250fe13844495ca6750181fc4232579ed7c82d1775eee` |
| Lane definition | `9f387d04b4c91d9e271543907eead87625361389da7d2c9279c703a67027eb81` |
| Config | `08b0593cbd4e1c11d4603b521ddfb2e4590ef4daa41fbfa7fe4eac857eb606d1` |

`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and
`VECLIB_MAXIMUM_THREADS` were all 1. The source-change list was empty. A
second identical invocation reused all nine verified cells and reproduced the
same certificate content ID.

## Convergence Evidence

| Observable | Terminal grid difference | Limit | Terminal tolerance difference |
|---|---:|---:|---:|
| Complete terminal current | `6.095485e-3 A m-2` | `7.5e-3 A m-2` | `1.55e-15 A m-2` |
| Interface recombination current | `4.278045e-9 A m-2` | `5.0e-9 A m-2` | `4.76e-22 A m-2` |
| Lateral carrier variation | `4.980758e-5` | `5.0e-4` | `2.73e-15` |
| Maximum ion site fraction | `6.800086e-5` | `1.0e-4` | `3.12e-17` |
| Mobile-ion redistribution | `3.956838e-4` | `5.0e-4` | `2.10e-15` |

All nine cells passed carrier/ion positivity, site occupancy, ion diagnostics,
inactive-clamp, finite positive interface-rate, topology, explicit-protocol,
and three-voltage completion gates. Across the matrix, maximum ion inventory
drift was `7.49e-16`, maximum all-face complete-current spread was
`5.71e-14 A m-2`, maximum current-decomposition relative error was
`2.93e-16`, and maximum grain-boundary-width relative error was `4.97e-16`.
The minimum nontrivial lateral response was `1.95e-3`, and the minimum ionic
redistribution was `1.18e-2`, so the matrix cannot pass by silently collapsing
to the uniform or frozen-ion topology.

## Reproduction

Run from the `perovskite-sim` root:

```bash
python scripts/run_numerical_refinement.py \
  twod-mobile-ion-interface-srh-v1 --dry-run

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python scripts/run_numerical_refinement.py \
  twod-mobile-ion-interface-srh-v1
```

Outputs are immutable, content-addressed artifacts under
`outputs/numerical-refinement/twod-mobile-ion-interface-srh-v1/<run_id>/` and
remain outside the tracked source tree.

## Nonclaims

The positive-ion diffusivity (`1e-10 m2 s-1`), interface parameters, and grain
boundary parameters are synthetic numerical stress inputs. This certificate
does not cover dual ions, mobile-ion Robin/selective contacts, periodic-x,
field mobility, projection, shared or dynamic interface occupancy, interface
charge, photon recycling, long-time hysteresis, or arbitrary cross-node
sampling. It is not SCAPS parity, external-solver validation, a measured-device
fit, an uncertainty bound, or validation of a real material parameter set.
