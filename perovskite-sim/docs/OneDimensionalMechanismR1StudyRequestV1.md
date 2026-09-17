# R1 pre-execution study request V1

Formal V3 studies first write a plan using `--plan-only --plan-file PATH`.
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

Every case has one attempt in a formal V3 plan. A retry is a new study with its
own request and provenance; a later successful run cannot erase an earlier
study. This protocol proves the provided case universe and saved content, not
unobserved execution history or the authenticity of arbitrary source code.

Record availability, comparison coverage, comparison success, verification
completion and scientific qualification remain distinct. Coarse-grid diagnostic
failures remain visible; the frozen scientific criteria decide which finest
comparisons are required. Neither an exit code nor successful file hashing is
itself independent scientific acceptance.

Response qualification inputs are optional and default to unknown. To use
independently reviewed inputs, the caller supplies `--qualification-file` and
its separately retained `--qualification-sha256`. The file must be outside the
result directory; its digest is fixed in the pre-execution request and checked
again during execution and replay. A result-local approved label is insufficient.

`R1StudyQualificationInputsV1` contains exactly `schema`, `current_budgets`,
`turnover_evidence`, `double_domain_evidence`, and `trusted_evidence`.
Current budgets are keyed by their exact amplitude-step case, turnover evidence
by the AC case, and double-domain inputs by the DoubleDomain case. A double-domain
input has `errors` and `prerequisites`; numeric error values are bound into the
reconstruction application digest. The trusted mapping holds evidence IDs and
canonical content digests selected by the caller after independent review.
This establishes which approval inputs were used, not whether their scientific
review was correct. State, units, coordinates, amplitude, data and application
are checked again by the qualification functions.

Empirical refinement differences remain `estimate_only`. The diagnostic window
amplitude is fixed before execution. It does not become a qualified amplitude
unless its explicitly selected linearity case is replayed successfully against
the same prepared state and approved budget. Analytic fixtures never grant
device qualification. Missing budgets are never replaced by zero.
