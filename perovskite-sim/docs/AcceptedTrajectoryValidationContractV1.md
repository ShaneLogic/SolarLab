# Accepted Trajectory Validation Contract V1

This contract supplements the foundation reference checks with the physical
domain of every accepted Radau state. It does not change the governing
equations, material inputs, time history, solver tolerances, or the scope of
the registered heterogeneous two-dimensional example.

`tests/accepted_trajectory.py` subclasses SciPy's public `Radau` solver only
to observe the initial state and the state after each successful `step()`.
Implicit Newton trials are not accepted physical states. The observer does
not evaluate the RHS, change a state, or change the step selection. A
separate control compares returned samples and RHS/Jacobian/LU counts with
ordinary Radau. Forced negative-density and capacity excursions with valid
output endpoints demonstrate why output-only checks are insufficient.

For all observed states, electrons and holes must be finite and positive.
Ions must be finite, non-negative and no larger than their declared local
site capacity. Each blocking ion's weighted number is compared with the
initial number for the entire physical history. A structurally absent ion
must remain exactly zero. The prepared one-dimensional single/dual-ion
reference retains its `1e-10` relative inventory limit. The pre-existing
heterogeneous two-dimensional lane retains its registered `1e-9` limit.
Weights are the same physical control-volume widths or areas as the model.

Coverage comprises all seven space/time/tolerance cases for each prepared
single/dual-ion history, every pair in the independent one-dimensional and
two-dimensional axis comparison, and all nine points of
`twod-mobile-ion-interface-srh-v1`, including its preceding one-dimensional
illuminated preparation. Its subsequent two-dimensional ion number is
referenced to that same initial population, with the lateral width included.
The original convergence comparisons and
quality limits remain active. TPV already validates its accepted Radau states
and integration-quadrature samples in `experiments/open_circuit.py`.

Reports preserve accepted times, density minima, maximum site fractions,
ion-number deviations, first violation and solver outcome. Failed or
discarded solver attempts are identified separately. Numerical production
source and the test observer have separate recorded file identities. Passing
this check is internal trajectory evidence, not material calibration,
external solver agreement, or experimental validation.
