"""Source-proven physical choices and exact execution-only legacy records."""
from dataclasses import replace
import json
import subprocess
import sys

import pytest

from solarlab.config.identity import ExecutionEngine, execution_identity, input_identity, physics_identity
from solarlab.materials.source import SourceDocument
from test_execution_identity_preparation import changed, identity_inputs
from test_execution_identity_preparation import execution as execution, device as device
from test_behavior_context import context, document, ROOT
from test_behavior_context import defaults as defaults, resources as resources, sources as sources, scaps as scaps
from test_behavior_context import r1_material_inputs as r1_material_inputs
from test_legacy_execution_context import r1 as r1


def bound(device, sources, *, driver="scaps_transient", environment=(), arguments=None):
    behavior = context(device, sources, driver=driver, environment=environment, arguments=arguments)
    return identity_inputs(device, effective_behavior=behavior,
                           problem="dc" if driver == "scaps_steady_state" else "transient")


def physical(inputs):
    return physics_identity(inputs).export()["document"]["payload"]


def execute(inputs, description):
    return execution_identity(inputs, description, engine=ExecutionEngine("legacy", inputs.effective_behavior))


def test_legacy_execution_automatically_binds_the_effective_projection(device, sources, execution):
    original = identity_inputs(device)
    behavior = context(device, sources)
    explicit = identity_inputs(device, effective_behavior=behavior)
    result = execution_identity(original, execution, engine=ExecutionEngine("legacy", behavior))
    assert result.export()["document"]["payload"]["H_physics"] == physics_identity(explicit).sha256
    assert physics_identity(explicit).sha256 != physics_identity(original).sha256
    assert input_identity(explicit).sha256 == input_identity(original).sha256
    assert physical(explicit)["domain"] == "prepared-historical-device-1d.v2"
    assert not result.cache_eligible


def test_effective_boolean_replaces_shadowed_declaration_but_execution_keeps_it(scaps, sources, execution):
    disabled = changed(scaps, lambda data: data["settings"].update(dos_band_potentials=False))
    unset = bound(disabled, sources)
    zero = bound(disabled, sources, environment=(("SOLARLAB_DOS_BAND", "0"),))
    one = bound(disabled, sources, environment=(("SOLARLAB_DOS_BAND", "1"),))
    explicit = bound(changed(disabled, lambda data: data["settings"].update(dos_band_potentials=True)), sources)
    assert physics_identity(unset).sha256 == physics_identity(zero).sha256
    assert physics_identity(one).sha256 == physics_identity(explicit).sha256
    assert physics_identity(one).sha256 != physics_identity(zero).sha256
    assert input_identity(one).sha256 != input_identity(explicit).sha256
    assert execute(unset, execution).sha256 != execute(zero, execution).sha256
    assert physical(one)["settings"]["dos_band_potentials"] is True


def test_scaps_declared_incoherence_keeps_the_actual_legacy_value(scaps, sources):
    with pytest.raises(ValueError, match="behavior context"):
        identity_inputs(scaps)
    first = bound(scaps, sources)
    changed_declaration = changed(scaps, lambda data: data["layers"][0]["parameters"].update(incoherent=False))
    second = bound(changed_declaration, sources)
    assert input_identity(first).sha256 != input_identity(second).sha256
    assert physics_identity(first).sha256 == physics_identity(second).sha256
    assert physical(first)["layers"][0]["remaining_effective_parameters"]["incoherent"] is False
    assert first.effective_behavior.export()["source_inspection"]["optics"]["incoherence"][0]["normalized_configured"] is True


def test_legacy_mode_ceiling_is_exact_not_an_interface_blanket(scaps, sources):
    legacy = changed(scaps, lambda data: data["settings"].update(mode="legacy", dos_band_potentials=False,
        band_grading=False, te_physical_norm=False, ion_steric_diffusion_only=False))
    environment = tuple((key, "1") for key in ("SOLARLAB_DOS_BAND", "SOLARLAB_BAND_GRADING", "SOLARLAB_TE_PHYSICAL", "SOLARLAB_ION_STERIC_DIFF"))
    a, b = bound(legacy, sources), bound(legacy, sources, environment=environment)
    assert physics_identity(a).sha256 == physics_identity(b).sha256
    c = bound(legacy, sources, environment=(("SOLARLAB_IFACE_PROJ", "1"),))
    assert physics_identity(c).sha256 != physics_identity(a).sha256
    assert physical(c)["effective_behavior"]["ion_flux_label"] == "whole_flux"


def test_contact_sentinels_and_old_mode_suppression_remain_distinct(scaps, sources):
    null = changed(scaps, lambda data: data["contacts"][0].update(S_n=None))
    zero = changed(scaps, lambda data: data["contacts"][0].update(S_n=0.))
    assert physics_identity(bound(null, sources)).sha256 == physics_identity(bound(zero, sources)).sha256
    full_null = changed(null, lambda data: data["settings"].update(mode="full"))
    full_zero = changed(zero, lambda data: data["settings"].update(mode="full"))
    a, b = bound(full_null, sources), bound(full_zero, sources)
    assert physics_identity(a).sha256 != physics_identity(b).sha256
    assert physical(a)["effective_behavior"]["contacts"][0]["condition"] == "density_dirichlet"
    assert physical(b)["effective_behavior"]["contacts"][0]["condition"] == "blocking_robin"


@pytest.mark.parametrize("key", ("PEROVSKITE_RHS_FINITE_CHECK", "SOLARLAB_SS_JAC_REUSE", "SOLARLAB_SS_GUMMEL"))
def test_numerical_import_policies_affect_execution_only(scaps, sources, execution, key):
    a = bound(scaps, sources, environment=((key, "0"),))
    b = bound(scaps, sources, environment=((key, "1"),))
    assert physics_identity(a).sha256 == physics_identity(b).sha256
    assert execute(a, execution).sha256 != execute(b, execution).sha256


def test_real_driver_settings_and_physical_environment_are_not_protocol(scaps, sources, execution):
    a = bound(scaps, sources)
    b = bound(scaps, sources, arguments={"N_grid": 30, "n_points": 60, "v_rate": 2., "V_max": 1.6, "rtol": 1e-5})
    assert physics_identity(a).sha256 == physics_identity(b).sha256
    assert execute(a, execution).sha256 != execute(b, execution).sha256
    for key, value in (("SOLARLAB_QSS_VTH", "12345"), ("SOLARLAB_IFACE_QSS", "1"), ("SOLARLAB_IFACE_ALLOW_GEN", "1")):
        assert physics_identity(bound(scaps, sources, environment=((key, value),))).sha256 != physics_identity(a).sha256
    steady = bound(scaps, sources, driver="scaps_steady_state")
    assert physical(a)["effective_behavior"]["te"]["softness"] == 0.
    assert physical(steady)["effective_behavior"]["te"]["softness"] == .02
    assert physics_identity(steady).sha256 != physics_identity(a).sha256


def test_units_zero_and_omitted_default_keep_existing_input_policy(scaps, sources):
    a = changed(scaps, lambda data: data["settings"].update(phi_left="-0 V"))
    b = changed(a, lambda data: data["settings"].update(phi_left=0.))
    assert physics_identity(bound(a, sources)).sha256 == physics_identity(bound(b, sources)).sha256
    omitted = changed(scaps, lambda data: data["settings"].pop("te_physical_norm", None))
    explicit = changed(omitted, lambda data: data["settings"].update(te_physical_norm=False))
    assert physics_identity(bound(omitted, sources)).sha256 == physics_identity(bound(explicit, sources)).sha256
    assert input_identity(bound(omitted, sources)).sha256 != input_identity(bound(explicit, sources)).sha256
    equivalent = changed(scaps, lambda data: data["settings"].update(T="300 K"))
    assert physics_identity(bound(scaps, sources)).sha256 == physics_identity(bound(equivalent, sources)).sha256


def test_r1_pair_state_and_numeric_history_stay_in_execution(r1, execution):
    original = r1.export()
    a = identity_inputs(r1.device, effective_behavior=r1)
    pair = replace(r1, prepared=SourceDocument("actual_pair_state", (ROOT / "tests/refactor/fixtures/r1_pair_common_state_preparation.json").read_bytes()),
                   arguments=document("pair_call", {**json.loads(r1.arguments.content), "backend": "pair"}))
    b = identity_inputs(pair.device, effective_behavior=pair)
    assert physics_identity(a).sha256 == physics_identity(b).sha256
    assert execute(a, execution).sha256 != execute(b, execution).sha256
    data = physical(a)["effective_behavior"]
    assert data["projection"] == "historical-r1-effective.v1"
    assert data["material_interface_charge_closure"] == "off"
    assert data["charge_reference"]["f_ref"] == json.loads(r1.reference.content)["f_ref"]
    frozen = replace(r1, arguments=document("frozen_ions", {**json.loads(r1.arguments.content), "control": "C"}))
    c = identity_inputs(frozen.device, effective_behavior=frozen)
    assert physics_identity(c).sha256 != physics_identity(a).sha256
    assert physical(c)["effective_behavior"]["ions"] == data["ions"]
    assert frozen.prepared is r1.prepared and r1.export() == original
    later = replace(r1, arguments=document("longer_short_protocol", {**json.loads(r1.arguments.content), "times_s": [0., 2e-9]}))
    d = identity_inputs(later.device, effective_behavior=later)
    assert physics_identity(d).sha256 == physics_identity(a).sha256
    assert execute(d, execution).sha256 != execute(a, execution).sha256


def test_context_source_provenance_is_retained_and_mismatch_cannot_be_relabelled(scaps, sources, execution):
    inputs = bound(scaps, sources)
    behavior = inputs.effective_behavior
    with pytest.raises(ValueError, match="actual source-bound"):
        replace(inputs, effective_behavior=behavior.export())
    with pytest.raises(ValueError, match="different prepared"):
        identity_inputs(changed(scaps, lambda data: data["settings"].update(Phi=1e21)), effective_behavior=behavior)
    with pytest.raises(ValueError, match="engine differs"):
        execution_identity(inputs, execution, engine=ExecutionEngine("solarlab"))
    altered = replace(behavior, environment=(("SOLARLAB_DOS_BAND", "1"),))
    with pytest.raises(ValueError, match="engine differs"):
        execution_identity(inputs, execution, engine=ExecutionEngine("legacy", altered))
    with pytest.raises(ValueError, match="exact audited"):
        replace(behavior, sources=(replace(behavior.sources[0], content=behavior.sources[0].content + b"\n"), *behavior.sources[1:]))
    result = execute(inputs, execution).export()["document"]["payload"]
    assert result["effective_binding"]["effective_behavior"]["context"] == behavior.export()
    assert result["implementation"]["build_source_commit"] == execution.implementation.build_source_commit
    assert result["documents"]["initial_state"]["sha256"] == execution.initial_state.sha256


def test_effective_identity_import_stays_pure(tmp_path):
    code = "import sys; import solarlab.config.identity; assert not set(sys.modules) & {'perovskite_sim','scipy','sksundae','flint','solarlab_server'}"
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
