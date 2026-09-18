# R1 pre-execution study request V1

Formal V4 calculations first write a plan using `--plan-only --plan-file PATH`.
This operation enumerates exact case requests and preparation dependencies,
without solving a physical case or creating a result directory. The caller
retains the printed SHA256 independently of later result bundles.

Production, resume and verification require that plan and its caller-held
`--request-sha256`. Resume and verification additionally require the result
manifest digest. The archived inventory must exactly match the pre-execution
plan, including cases not run, failed cases and unavailable comparisons.
Changing result summaries, failure indices, invocation logs and their hashes
cannot redefine the independently anchored request. Changing the request anchor
is a new study scope and must not be represented as completion of the old one.

Execution may be bounded by case count or a subset invocation. Such limits do
not reduce the planned case set or turn missing cases into successes. Verification
visits all planned, archived cases irrespective of execution filters. Required
finest comparisons are fixed by the explicit requested scope; they do not claim
that an operator-selected highest grid is converged for a larger study.

Every case has one attempt in a formal plan. A retry is a new study with its
own request and provenance; a later successful run cannot erase an earlier
study. This protocol proves the provided case universe and saved content, not
unobserved execution history or the authenticity of arbitrary source code.
An interrupted attempt is retained even without a completion record. It cannot
be continued as AttemptV2 in the same formal study. Present failed states must
be reconstructed; removing their reconstruction context cannot turn them into
an identity-only inspection. A separately validated observer-write failure may
leave one final observed row unpersisted, or omit only the failure annotation
added after the callback. All other prefix differences are rejected. These
checks report saved and persisted extents separately and never certify the
failed experiment or reconstruct its unobserved execution history.

Record availability, comparison coverage, comparison success, verification
completion and scientific qualification remain distinct. Coarse-grid diagnostic
failures remain visible; the frozen scientific criteria decide which finest
comparisons are required. Neither an exit code nor successful file hashing is
itself independent scientific acceptance.

## Post-execution qualification

Calculation and qualification have different immutable requests. Calculations
do not need approval evidence that can only be produced after the run.
Prepare, transient, AC, baseline DC, target DC and conductance results are
enumerated and sealed first. Their original preparation timestamps and full
artifact hashes remain unchanged. A physical preparation content identity is
additional information, never permission to reuse another run's approval.

Use `--qualify --analysis-dir PATH --analysis-plan-file PATH` with the original
calculation's plan/request and manifest anchors, plus `--qualification-file`,
`--qualification-sha256` and `--analysis-approved-standard-sha256`. First use
`--analysis-plan-only` to retain the new `ANALYSIS_REQUEST_SHA256` outside both
result directories. Execute with `--analysis-request-sha256`, then replay with
`--verify-analysis --analysis-manifest-sha256`. Missing collections are rejected
without creating any files. Existing analysis output is never overwritten.

The calculation is fully verified before derivation. The analysis request
binds that collection, the independently reviewed inputs, selected derived
cases, exact WindowSpec, amplitude choice and required frequency intersection.
The default amplitude qualification selects only the final two declared
amplitudes; larger pairs remain diagnostics. `--analysis-frequencies` may name
an externally selected subset of collected points; failed points cannot be
silently trimmed by the result itself. This subset still needs scientific
justification for the intended research claim.

Derived analysis reads existing preparations and DC coordinates. Creating a
new common preparation or solving a new DC is prohibited during derivation;
missing numerical dependencies require a new calculation request. Verification
replays of already sealed inputs occur before this phase and do not replace
their data. `--analysis-linearity-case` binds the selected derived comparison.

The output records calculation failures, missing cases, verification scope,
derived results and device eligibility separately. Exit 2 can mean the analysis
completed correctly but device qualification remains unavailable. Independent
review and full R1-2 completion are never inferred from an analysis exit code.

`R1StudyQualificationInputsV1` contains exactly `schema`, `current_budgets`,
`turnover_evidence`, `double_domain_evidence`, and `trusted_evidence`.
V2 adds `numerical_reconciliations`, keyed by the exact Linearity case, for
separately reviewed explanations of empirical uncertainty conflicts.
Current budgets are keyed by their exact amplitude-step case, turnover evidence
by the AC case, and double-domain inputs by the DoubleDomain case. A double-domain
input has `errors` and `prerequisites`; numeric error values are bound into the
reconstruction application digest. The trusted mapping holds evidence IDs and
canonical content digests selected by the caller after independent review.
This establishes which approval inputs were used, not whether their scientific
review was correct. State, units, coordinates, amplitude, data and application
are checked again by the qualification functions.

Empirical refinement differences remain `estimate_only`. Reviewed estimates
and conditional bounds retain their distinct classifications; neither is
relabeled a rigorous bound. Reconciliation does not waive the frozen numerical
axis acceptance limits. The diagnostic window
amplitude is fixed before execution. It does not become a qualified amplitude
unless its explicitly selected linearity case is replayed successfully against
the same prepared state and approved budget. Analytic fixtures never grant
device qualification. Missing budgets are never replaced by zero.
