# IDA controls, observation and sparse lifetime binding

This directory holds a source patch for scikit-SUNDAE 1.1.3 and SUNDIALS 7.5.0.
It does not change an existing installation or the SolarLab physical model.
The build compiles `_cy_ida` together with the small source-owned
`ida_superlumt_cleanup.c`; the original `_cy_common`, `_cy_cvode`, configuration,
and every shared library retain verified RECORD and content identities.
SuperLU_MT/OpenMP and BLAS/LAPACK remain explicit build requirements.

The same public `IDA` class exposes `statistics()` and `last_step_snapshot()`.
The composed controls patch preserves `nonlin_conv_coef=None` as no setter
call. Explicit values must be finite positive real non-Booleans. Statistics
use checked native integrator, nonlinear-solver, Jacobian and current-cj
getters under the observation owner lock. Requested coefficient metadata is
not a getter for the native coefficient. Reads do not advance or mutate history.

BR01 remains explicit: zero-step method fields can be sentinels or retain a
previous generation's values. They are returned unchanged, accompanied by
`method_fields_valid`, owner and generation. BR02 restricts statistics to the
`init_step`/`step` window; completed batch `solve` closes it. BR03 preserves raw
native counters and cj without inferring Newton iterations, rejected branches
or the caller's requested/returned times and actual Jacobian callback cj.

The native source owns a renamed constructor and its lifetime operations,
derived from the exact SUNDIALS 7.5.0 source in `SourcePinsV1.json`. It installs
the repaired free operation before its first content allocation, so constructor
failures use the same cleanup. Gstat and options start zeroed; the three owned
option arrays and AC/L/U stores are freed independently when allocated. Borrowed
matrix/vector data and permutation aliases are not freed twice. Its initialize
operation retires the same factor-owned data before delegating to the unchanged
native initialize. This matters for same-size `IDAReInit`: the native initialize
resets FIRSTFACTORIZE, so the next refact=NO setup must not overwrite prior owned
allocations. Normal refact=YES setup keeps its existing reuse behavior. Setup,
factorization, solve, ordering and statistic routines remain in the unchanged
original library; this is not a second linear solver.

The Python owner releases IDA before the linear solver, matrices and vectors,
then releases SUNContext last. Failed `init_step` setup frees partial resources
and rethrows the initiating exception. The owner clears freed native pointers;
calling native free again on an already freed nonnull pointer is not supported.
SuperLU's existing process-abort behavior on internal allocation failure remains
unchanged. There is no retained solver, skipped cleanup, early step or `os._exit`
in the production binding.

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
- `build --plan ... --plan-sha256 ... --start-message ... --external-process-group` requires Root review
  and start admission for that exact plan. The supervised commands generate C,
  compile the extension and lifetime source, repair loader-relative links with
  the pinned system tool, then ad-hoc sign and verify that new extension. A wheel
  is streamed from the qualified files with regenerated RECORD. In addition to
  the existing read-only overlay, a new complete installation is extracted and
  verified against that wheel and RECORD. Native checks use the complete
  installation so loader-relative dependencies cannot resolve through an old
  extension symlink. No old installation is overwritten.

This direct route needs the pinned Cython tool only; it does not execute an
upstream setup script, install dependencies, rebuild a full SDK, or substitute
a newer SUNDIALS library. The pilot is Python 3.13 only, 120 seconds / 2 GiB /
64 MiB combined retained evidence, SDK, tools and outputs. A separately
admitted Python 3.11 build is required to establish the existing minimum.
All failures retain their first error and receipts, without automatic retry.
Producing a wheel does not activate a product dependency or qualify physics.
The externally admitted whole-process receipt owns full launch/finalization
accounting; the builder's command-period metrics remain separately labelled.
The external supervisor starts the builder in one new owned process group;
compiler/tool children inherit that group. Internal error cleanup waits only
for its actual child and never kills the builder's own shared group. The outer
supervisor enforces the total deadline/RSS/output limits, reaps its builder
child and records/terminates any remaining owned group before final readback.

Source-only checks are `TestSourcePreparation` in
`tests/numerical_contracts/test_ida_observation.py` and require the explicit
`IDA_OBSERVATION_SOURCE_ARCHIVE`. Native checks additionally require
`SOLARLAB_IDA_OBSERVATION_NATIVE_TEST=1`, a reviewed patched artifact and
separate science-slot admission. Unrun/skipped tests are not native evidence.
`TestNativeCompositionLifetime` adds normal destruction before factorization,
an actual failed-atol setup after sparse allocation, factorization with both
public reads in one process, default/None equivalence, coefficient guards and
generation/reentrancy checks, including a second sparse factorization after
same-size reinitialization followed by normal destruction. The independent process supervisor must also
retain normal exit and process ownership evidence. These small binding markers
do not replace either complete S0/B protocol or independent accuracy checks.
