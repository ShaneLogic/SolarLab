"""Workflow harness settings must not activate legacy product override gates."""
import ast
import json
from pathlib import Path
import re
import shlex

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
VITEST_COMMAND = 'SOLARLAB_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python" npm run test:run'
PRODUCT_ENV_ALLOWLIST = {
    ("ci.yml", "frontend", "SOLARLAB_PYTHON"): {
        "run": VITEST_COMMAND,
        "reason": (
            "Command-local interpreter selection for Vitest's schema-fixture.mjs DTO exporter; "
            "the exact Python payload below only exports schema metadata and never calls a legacy solver. "
            "The assignment is not inherited by this job's separate Python test steps."
        ),
    },
}
PRODUCT_NAME = re.compile(r"\bSOLARLAB_\w*")


def product_environment_uses(value, location=()):
    """Inspect env mappings and shell scripts; opaque product references require review."""
    uses = []
    if isinstance(value, list):
        for index, item in enumerate(value):
            uses.extend(product_environment_uses(item, (*location, str(index))))
    elif isinstance(value, dict):
        for key, item in value.items():
            here = (*location, key)
            if key == "env":
                assert isinstance(item, dict), (here, "dynamic environment maps require review")
                for name, content in item.items():
                    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name), (here, name)
                    if name.startswith("SOLARLAB_") and content not in (None, ""):
                        uses.append((here, name, content))
            elif key == "run" and isinstance(item, str):
                try:
                    tokens = shlex.split(item, comments=True)
                except ValueError:
                    tokens = [item]  # Unclassified shell syntax cannot silently bypass review.
                for token in tokens:
                    for name in PRODUCT_NAME.findall(token):
                        if token != name + "=":  # A literal empty shell assignment is harmless.
                            uses.append((here, name, item))
            else:
                uses.extend(product_environment_uses(item, here))
    return uses


def test_workflows_do_not_set_unreviewed_product_environment_variables():
    seen = set()
    for path in sorted((ROOT / ".github/workflows").glob("*.y*ml")):
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        for location, name, value in product_environment_uses(workflow):
            job = location[1] if location[:1] == ("jobs",) else None
            key = (path.name, job, name)
            allowed = PRODUCT_ENV_ALLOWLIST.get(key)
            assert allowed and allowed["reason"].strip(), (path.name, location, name, value)
            assert location[-1] == "run" and value == allowed["run"], (location, value)
            seen.add(key)
    assert seen == set(PRODUCT_ENV_ALLOWLIST), "remove stale exceptions instead of retaining unused permissions"


@pytest.mark.parametrize("value", ["1", "0", " ", "${{ matrix.value }}"])
@pytest.mark.parametrize("scope", ["workflow", "job", "step"])
def test_product_guard_covers_inherited_and_step_environment(scope, value):
    body = {"env": {"SOLARLAB_TEST": value}}
    if scope == "job":
        body = {"jobs": {"core": body}}
    elif scope == "step":
        body = {"jobs": {"core": {"steps": [body]}}}
    assert [name for _, name, _ in product_environment_uses(body)] == ["SOLARLAB_TEST"]


@pytest.mark.parametrize("command", [
    "SOLARLAB_TEST=1 python probe.py",
    "export SOLARLAB_TEST=1\npython probe.py",
    'echo "SOLARLAB_TEST=1" >> "$GITHUB_ENV"',
    'echo "SOLARLAB_TEST<<END" >> "$GITHUB_ENV"',
    "python -c 'import os; os.environ[\"SOLARLAB_TEST\"] = \"1\"'",
    '$env:SOLARLAB_TEST = "1"',
])
def test_product_guard_covers_inline_export_and_persistent_assignments(command):
    assert {name for _, name, _ in product_environment_uses({"run": command})} == {"SOLARLAB_TEST"}


def test_product_guard_allows_literal_empty_and_nonproduct_settings():
    assert product_environment_uses({"env": {"SOLARLAB_TEST": "", "SCHEDULED_RUNTIME_EVIDENCE_DIR": "path"},
                                     "run": "SOLARLAB_TEST='' python probe.py"}) == []


def test_product_guard_covers_the_reserved_prefix_without_a_suffix():
    for command in ("SOLARLAB_=1 python probe.py", "prefix=SOLARLAB_; export ${prefix}TEST=1"):
        assert {name for _, name, _ in product_environment_uses({"run": command})} == {"SOLARLAB_"}


def test_vitest_exception_is_command_local_and_its_python_payload_only_exports_metadata():
    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    frontend = workflow["jobs"]["frontend"]
    assert frontend["defaults"]["run"]["working-directory"] == "web"
    assert sum(step.get("run") == VITEST_COMMAND for step in frontend["steps"]) == 1
    assert json.loads((ROOT / "web/package.json").read_text())["scripts"]["test:run"] == "vitest run"
    helper = (ROOT / "web/src/test-helpers/schema-fixture.mjs").read_text()
    payload = helper.split("const code = [", 1)[1].split("].join('\\n');", 1)[0]
    lines = [ast.literal_eval(line.strip().removesuffix(",")) for line in payload.splitlines() if line.strip()]
    assert lines == [
        "from pathlib import Path", "import sys", "import solarlab.config.schema as schema",
        'assert Path(schema.__file__).resolve() == Path("src/solarlab/config/schema.py").resolve()',
        "sys.stdout.buffer.write(schema.configuration_schema_representation()[0])",
    ]
    assert "execFileSync(process.env.SOLARLAB_PYTHON || 'python3', ['-B', '-c', code]" in helper
