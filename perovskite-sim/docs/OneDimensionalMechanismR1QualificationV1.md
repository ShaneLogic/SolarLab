# R1 response qualification V1

This declaration specifies the C-line qualification interface. It does not
approve any real-device numerical error budget or turnover estimate. The
frozen physical thresholds and study geometry remain unchanged. Independent
caller-held approvals are required before real-device qualification.

## Scope and trust boundary

`qualification_scope(record, state_sha256=..., domain=..., window_spec=...)` binds:
`source_sha256`, `prepared_sha256`, `state_sha256`, `reference_sha256`,
`control`, `intervals`, `operating_voltage_V`, and `domain`.
When a window is supplied it also binds `window_spec_sha256`. Device time-domain
qualification requires the caller-frozen `R1ObservationWindowV1`: exact ticks,
0+ regular-current meaning, separate 0- preparation and impulse, and twelve
intervals per decade. Legal decade extensions to 1e5 s and starts down to
1e-12 s retain the same rules. A record cannot define its own completeness.

The caller obtains the state digest from the independently checked initial
prepared state. Both amplitudes use that same initial-state digest, never a
digest of their different final states. Source, preparation, reference,
control, grid and operating bias come from the checked input record.
`domain` is selected by the caller and is either `device` or
`analytic_fixture`. A fixture qualification cannot be borrowed by a device.

`evidence_digest(value)` is SHA-256 of finite, canonical JSON with sorted
keys and compact separators. NumPy arrays/scalars and dataclasses are encoded
as their JSON values; complex scalars are `{real, imag}`. Computing a digest
does not constitute verification or approval.
Runtime integer refinement-level mapping keys are normalized to their JSON
string keys. Boolean/float/other key types and collisions between a string
and an integer key are rejected, so runtime and saved records share one
unambiguous identity.

`trusted_evidence` is a caller-held mapping of reviewed evidence IDs to digests.
It must originate outside the evidence being assessed: independently checked
budget, turnover or prerequisite evidence. Separately replayed device step
records are supplied by role in `verified_record_digests`, not promoted by
membership in the approval mapping. A producer's `review_status: approved`, its own manifest, or the
mere presence of an artifact cannot populate this mapping. The root runner
owns the external request/approval binding. These pure functions neither
authenticate a producer nor approve a scientific method.

## Current error budget

`assess_current_error_budget(budget, response, expected_scope=...,
amplitude_V=..., trusted_evidence=...)` accepts `R1CurrentErrorBudgetV1`:

- `evidence_id`, exact `scope`, `classification` (`bounded`, `conditional_bound`,
  `validated_estimate`, `estimate_only`, or `unknown`), `units: A/m2`, and the exact `amplitude_V`;
- exact `coordinates`, named `components`, and `response_sha256` binding
  every baseline-subtracted current sample;
- nonempty `method` and `source_evidence`, plus
  `propagation_method: sum_nonnegative_absolute_bounds`;
- a nonempty, unique `required_terms` list and `terms_A_m2` mapping with
  exactly those keys and an array matching every response sample/component;
- `review_status: approved` and a nonempty `review_id`, as well as the
  independently held matching digest.

Independent review is responsible for the completeness and applicability of
the declared error decomposition. The checker refuses omitted declared
terms, negative/nonfinite bounds, wrong units/coordinates/components/state,
and reuse for another response or amplitude. It sums the nonnegative absolute
bounds without cancellation. `None` remains unknown and is never replaced
by zero. An empirical refinement estimate remains `estimate_only` merely from
provenance verification. A separately reviewed validated estimate can support
the frozen numerical criteria without claiming a rigorous bound. The assessment
publishes `uncertainty_A_m2`; `bound_A_m2` is reserved for bound classifications.
No real-device budget is approved by this code.

## Endpoint diagnostics and transient linearity

Seven DC endpoint solves remain separate diagnostics. Each adjacent pair
reports whether its difference exceeds the **1% response scale**, and the
budget status remains unknown unless independent bounds are supplied through
the separate budget interface. Exceeding that scale is not identical to
exceeding the complete approved error-plus-response criterion. Endpoint
agreement never certifies the intervening transient.

`assess_transient_linearity(coarse_step, fine_step, ...)` takes independently
checked controlled steps, their zero-minus contact baselines, an externally
fixed `expected_times_s`, a shared initial-state scope and both error budgets.
It requires adjacent amplitudes from the declared ladder; exact step/event
voltages; both contacts and every requested time; complete physical input
certificates; and caller-held step digests. Device qualification additionally
requires the supplied, externally bound complete WindowSpec. Analytic fixtures may use
their explicitly declared shorter grid and remain fixture-only.

The unchanged `compare_amplitude_halving` rule applies at every point:
the current-error/amplitude sum plus 1% of the pointwise response scale; the
error budget must also resolve the signal under the original 1% rule.
Missing budgets yield `budget_unqualified` and no comparison verdict. A
known budget can yield `response_not_linear`, `linearity_undetermined`, or
`within_linearity_budget`. All-time regular-current qualification does not
certify impulse, tail or reconstruction errors, which remain separate.

Device inputs must be the last two levels of the externally declared continuous
amplitude ladder. Measured three-axis uncertainties and verdicts are supplied
separately and recomputed against the actual response. Unexplained disagreements
with reviewed budgets block qualification. A specifically bound independent
reconciliation may explain an empirical uncertainty estimate, but cannot waive
an actual failed numerical axis requirement or the final linearity threshold.

`select_response_amplitude(reports, expected_scope=...,
verified_report_digests=..., diagnostic_amplitude_V=.005)` only selects from
matching, independently replayed full-time qualifications. It returns
`qualified_amplitude_V: null` when none exists and separately retains
`diagnostic_amplitude_V`. The latter never conveys small-signal qualification.
The runner fixes its requested amplitude before execution; this selector
does not authorize changing a frozen study request after observing results.

## Frequency coverage

`frequency_window_report(ac, turnover_evidence=..., expected_scope=...,
trusted_evidence=...)` combines numeric AC checks with
`R1TurnoverEvidenceV1`. That record supplies an evidence ID, matching scope,
`ac_sha256` binding the checked AC record and its actual operating state,
`units: Hz`, positive `turnover_frequency_Hz`,
`coverage: all_applicable_modes`, `unresolved_modes: []`, and independently
bound approved review identity. Approval must cover all applicable ion,
charging, dielectric and trap scales, not merely a visible fitted pole.

The checker uses the frozen protocol: at least one decade beyond both ends
of the needed turnover range, four frequency intervals per decade, protocol
limits 1e-6..1e10 Hz, and no failed numeric point inside the sampled window.
It retains all frequency points and numeric contiguous bands. Endpoint
flatness does not replace mode evidence. The result is `certified`,
`incomplete`, `unknown`, or `invalid`. Without independently bound turnover
evidence it stays unknown/False; approved, state-matched analytic single-pole
evidence exercises a genuine True path scoped only to that fixture.

## Conditional double-domain integration

`assess_double_domain_prerequisites(evidence, expected_scope=...,
frequency_Hz=..., expected_application=..., trusted_evidence=...)` requires:
finite-amplitude linearity, single-axis convergence, window extension,
earlier start, stricter integration, tail/DC agreement, frequency coverage,
the input trajectory and AC content; plus independently bounded current,
early omission, tail omission, interpolation, impulse charge, baseline
current and DC-conductance errors.

Each named item is `R1QualificationEvidenceV1` with its `kind`, evidence ID,
exact scope and `application`, exact frequency array, boolean
`eligible_frequency_points`, qualified/review status and external digest.
Error items retain their approved uncertainty classification. Bare pass flags cannot supply
this evidence. Every required item must qualify at a frequency before that
frequency receives double-domain eligibility.

The application contains `step_sha256`, `ac_sha256`, `conductance_sha256`,
`step_amplitude_V`, `times_s` and `reconstruction_request_sha256`. The last
digest binds the supplied error object and both quadrature tolerances.
The application also includes `window_spec_sha256` for an explicit window.
`reconstruct_study_response` constructs this application from its actual
inputs. Scope alone cannot authorize another amplitude, trajectory, DC
ladder, frequency axis or reconstruction uncertainty. Any additional caller
prerequisite can restrict, but cannot override, the independently bound masks.

The reconstruction still checks numerical component agreement and error
budgets. Missing evidence yields unqualified results even for identical
curves. It returns separately `double_domain_consistent` for the declared
domain and `device_double_domain_consistent`. Analytic tests verify the
software decisions; they do not promise a real device's full-spectrum pass.
`device_response_gate` checks a caller-fixed required frequency intersection,
the exact device scope, complete window, reconstruction application and aligned
numeric/evidence masks. A generic fixture consistency flag cannot satisfy it.
