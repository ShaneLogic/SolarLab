"""Public decision counterexamples from the independent eight-mutation review.

The examples are synthetic library inputs, not device-qualification evidence.
Each keeps the other guards satisfied so one missing check cannot hide behind
an unrelated rejection.
"""
import numpy as np

from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import R1Response, observation_times
from perovskite_sim.experiments.one_dimensional_mechanism_r1_qualification import (
    assess_double_domain_prerequisites, assess_transient_linearity, evidence_digest,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_window import build_window_spec, window_spec_digest
from tests.unit.experiments.test_r1_window_qualification_v4 import linearity_case, reconstruction_case


def test_prerequisites_must_bind_the_external_window_even_when_every_evidence_digest_matches():
    args, kwargs = reconstruction_case(last=100.)
    evidence = kwargs["qualification_evidence"]
    application = evidence["finite_amplitude_linearity"]["application"]
    common = dict(frequency_Hz=args[3]["frequency_Hz"], expected_application=application,
                  window_spec=kwargs["window_spec"])
    control = assess_double_domain_prerequisites(evidence, expected_scope=kwargs["expected_scope"],
        trusted_evidence=kwargs["trusted_evidence"], **common)
    assert control["qualified"] and control["device_qualified"]

    # The caller requires 1000 s, but the supplied complete data/application
    # and its internally consistent window cover only 100 s. Bind every item
    # to the new scope so the missing external-window guard is the only fault.
    requested = build_window_spec(observation_times(last_time_s=1000.))
    scope = {**kwargs["expected_scope"], "window_spec_sha256": window_spec_digest(requested)}
    for item in evidence.values():
        item["scope"] = scope
    trust = {key: evidence_digest(item) for key, item in evidence.items()}
    result = assess_double_domain_prerequisites(evidence, expected_scope=scope,
                                                trusted_evidence=trust, **common)
    assert not result["qualified"] and not result["device_qualified"]
    assert not any(result["eligible_frequency_points"])
    assert result["invalid_prerequisites"]


def test_a_resolvable_refinement_error_still_requires_explaining_a_smaller_reviewed_budget():
    records, responses, kwargs = linearity_case()
    assert assess_transient_linearity(*records, **kwargs)["device_qualified"]
    response = responses[0]
    # This uncertainty is small enough that measured amplitude linearity
    # still passes, but exceeds the reviewed 1e-12 A/m² budget. A separate
    # measured-linearity failure must not mask loss of the conflict check.
    kwargs["measured_evidence"]["coarse_current_error"] = R1Response(
        np.full(response.values.shape, 1e-8), response.coordinates, response.components)
    result = assess_transient_linearity(*records, **kwargs)
    assert result["coarse_budget"]["qualified"] and result["fine_budget"]["qualified"]
    assert result["comparison"]["passed"]
    consistency = result["measured_consistency"]
    assert consistency["comparison"]["passed"]
    assert consistency["concerns"] == ["coarse_declared_uncertainty_smaller_than_refinement_estimate"]
    assert not result["qualified"] and not result["device_qualified"]
    assert consistency["status"] == "unresolved_numerical_evidence_conflict"
