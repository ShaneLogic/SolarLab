# IDA accepted-step observation pilot

This directory holds a source patch for scikit-SUNDAE 1.1.3 and SUNDIALS 7.5.0.
It does not change the installed environment or the SolarLab solver. The pilot
compiles only `_cy_ida`; the original `_cy_common`, `_cy_cvode`, configuration,
and shared libraries are reused with verified RECORD and content identities.
SuperLU_MT/OpenMP and BLAS/LAPACK remain explicit build requirements.

`IDA.last_step_snapshot(expected_step=None)` returns deeply immutable byte
copies plus owner, init/reinit generation, step and source/build identities.
It requires an observed immediate predecessor and output at the accepted native
`tn`. NORMAL calls with hidden intermediate steps and interior root returns
are refused. Eligibility recovers only after consecutive observed endpoints.
The actual predecessor is retained separately from `tn-hused`, with signed
clock-gap terms; no interval is silently shortened or reset.

The packet contains public `tn`, `hused`, `kused`, `nsteps` and separate next
step/order diagnostics `hh`, `kk`, raw y/yp, rounded Dky0 through Dky[kused] at
tn, error weights and local estimates. The source-owned C helper copies the
stored `phi[0:q+1]` and `psi[0:q]` words through exact upstream private header
declarations after public/private scalar checks. It copies native uround and
checks shapes, finite values, capacities and nonaliasing buffers. No internal
pointer or mutable vector is exposed. A second copy and public stamp confirm
the same data remained in place under the owner lock. Actual Dky0/1 calls at
the recorded predecessor retain their raw statuses and return None buffers
on native failure. Event sides and input laws remain caller-owned metadata.

`basis.source_header_sha256` identifies the exact upstream ida_impl.h bytes;
`basis.source_config_sha256` identifies the unedited generated SDK config;
`basis.build_identity == binding.identity` binds recipe, extension, source,
config and original runtime-library hashes. The loaded IDA library path is
obtained through dladdr, without inspecting Python object memory. Unbound
source cannot produce a snapshot. These checks detect ABI mismatches; they
do not establish otherwise missing original private-ABI provenance.

The chosen coefficient route is a version-bound native word copy. Dky remains
rounded and is not an exact coefficient source. The alternative would require
a separate proved rounding envelope for the actual native GetDky recurrence,
including its psi divisions and vector reduction. Neither route alone proves
true DAE/global error, early high derivatives, or nonlinear SG/capture/port
integrals. The prior exact 2^-55 clock gap and early second-derivative failure
remain open evidence for their respective proofs.

## Offline build steps

- `prepare --archive ... --work ...` checks the pinned published sdist and
  applies the exact three-file patch without fuzzy context matching.
- `freeze --recipe ... --recipe-sha256 ... --archive ... --work ...` requires
  explicit source/header/tool/runtime inventories and produces actual argv,
  environment and source hashes. It binds the copy patch to that recipe.
- `build --plan ... --plan-sha256 ... --start-message ...` requires Root review
  and start admission for that exact plan. The supervised commands generate C,
  compile one extension, repair its loader-relative links with the pinned
  system tool, and ad-hoc sign only that new extension. A prototype wheel is
  streamed from the qualified files with regenerated RECORD; a symlink overlay
  reuses originals read-only and never installs into them.

This direct route needs the pinned Cython tool only; it does not execute an
upstream setup script, install dependencies, rebuild a full SDK, or substitute
a newer SUNDIALS library. The pilot is Python 3.13 only, 120 seconds / 2 GiB /
64 MiB combined retained evidence, SDK, tools and outputs. A separately
admitted Python 3.11 build is required to establish the existing minimum.
All failures retain their first error and receipts, without automatic retry.
Producing a wheel does not activate a product dependency or qualify physics.

Source-only checks are `TestSourcePreparation` in
`tests/numerical_contracts/test_ida_observation.py` and require the explicit
`IDA_OBSERVATION_SOURCE_ARCHIVE`. Native checks additionally require
`SOLARLAB_IDA_OBSERVATION_NATIVE_TEST=1`, a reviewed patched artifact and
separate science-slot admission. Unrun/skipped tests are not native evidence.
