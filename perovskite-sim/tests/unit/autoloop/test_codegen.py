# tests/unit/autoloop/test_codegen.py
import pytest
from perovskite_sim.autoloop.types import Gap, Hypothesis, AblationMatrix, AblationProbe
from perovskite_sim.autoloop.codegen import (
    GeneratedLever, CODEGEN_SCHEMA, FakeCodegen, ClaudeCodegen, codegen_prompt,
)


def _gap():
    return Gap(id="trend:Et_PVK ETL:V_oc", metric="V_oc", sweep="Et_PVK ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _hyp():
    return Hypothesis(gap_id="trend:Et_PVK ETL:V_oc", cause="physics",
                      mechanism="missing band-tail Urbach absorption", verdict="confirmed",
                      evidence_for=("ablation: dark J nonzero",))


def _matrix():
    return AblationMatrix(gap_id="trend:Et_PVK ETL:V_oc", baseline_val=40.0,
                          probes=(AblationProbe("SOLARLAB_IFACE_PROJ", "flag", {}, 40.0, 39.9, -0.1, True),))


class _Runtime:
    def __init__(self, obj):
        self.obj = obj
        self.last_prompt = None
    def complete(self, prompt, schema):
        self.last_prompt = prompt
        if isinstance(self.obj, Exception):
            raise self.obj
        return dict(self.obj)


def test_fakecodegen_returns_lever():
    lev = FakeCodegen("return arrays", rationale="noop").generate(_gap(), _hyp(), _matrix())
    assert isinstance(lev, GeneratedLever)
    assert lev.body == "return arrays" and lev.rationale == "noop"


def test_codegen_schema_keys():
    assert CODEGEN_SCHEMA == {"required": ["body", "rationale"]}


def test_claudecodegen_parses_runtime_output():
    rt = _Runtime({"body": "return dataclasses.replace(arrays, chi=arrays.chi)", "rationale": "why"})
    lev = ClaudeCodegen(rt).generate(_gap(), _hyp(), _matrix())
    assert lev.body.startswith("return dataclasses.replace") and lev.rationale == "why"


def test_prompt_contains_contract_and_mechanism():
    p = codegen_prompt(_gap(), _hyp(), _matrix())
    assert "adjust_material_arrays" in p and "Urbach" in p
    assert "chi" in p and "JSON" in p and "return arrays unchanged" in p


def test_claudecodegen_propagates_runtime_error():
    with pytest.raises(RuntimeError):
        ClaudeCodegen(_Runtime(RuntimeError("claude down"))).generate(_gap(), _hyp(), _matrix())


def test_splice_produces_importable_module(tmp_path):
    from perovskite_sim.autoloop.codegen import splice_lever_body, LEVER_TEMPLATE
    body = "import dataclasses\n" if False else "return dataclasses.replace(arrays, chi=arrays.chi + 0.0)"
    src = splice_lever_body(LEVER_TEMPLATE, body)
    f = tmp_path / "lever.py"; f.write_text(src)
    import ast; ast.parse(src)                       # must be syntactically valid
    assert "def adjust_material_arrays(arrays, ctx):" in src
    assert "return dataclasses.replace(arrays, chi=arrays.chi + 0.0)" in src
    assert "_LeverContext" not in src                # _LeverContext lives in _ctx, not spliced file


def test_validate_lever_body_rejects_imports_and_dangerous_calls():
    from perovskite_sim.autoloop.codegen import validate_lever_body
    for bad in ["import os\nreturn arrays",
                "__import__('os').system('x')\nreturn arrays",
                "open('/tmp/x','w').write('h')\nreturn arrays"]:
        with pytest.raises(ValueError):
            validate_lever_body(bad)
    validate_lever_body("return dataclasses.replace(arrays, chi=arrays.chi + 0.0)")  # ok
