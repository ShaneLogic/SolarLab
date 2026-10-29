"""Symbol-level numerical import boundary with an explicit existing-debt ledger.

Import-linter operates on module dependencies; this check inspects the actual
ImportFrom names, including relative imports. The 141 configuration, registry,
serialization and experiment occurrences from AssemblyMigrationInventoryV1
remain explicit exceptions. Public symbols merely aliased privately are not
part of that historical count; promoted numerical APIs may not keep such aliases.

The two temporary steady helper exceptions are now retired. Five additional
BDF-to-experiment imports already present at f0c1027f are listed separately;
their providers are outside the numerical roots. The reviewed 141-entry
configuration/registry/experiment ledger remains unchanged.
"""

from __future__ import annotations

import ast
from collections import Counter
from importlib import import_module
from importlib.util import resolve_name
from pathlib import Path
from typing import TypeAlias

import pytest

ROOT = Path(__file__).resolve().parents[2]
ImportKey: TypeAlias = tuple[str, str, str]
ExceptionGroup: TypeAlias = tuple[str, str, tuple[str, ...]]
CORE_ROOTS = ("contracts", "mesh", "discretization", "physics", "equations", "solvers", "device")

# Every original occurrence is listed; repeated names retain their multiplicity.
# Source: plans/Specs/AssemblyMigrationInventoryV1.json, SHA-256
# b05c96b0bf52f3f7770df7f2b34c79400e7e95e4d4b417dc656af18a57846e78.
KNOWN_LATER_IMPORTS: tuple[ExceptionGroup, ...] = (
    ('solarlab.config.defect_documents', 'solarlab.config.device_import', ('_keys', '_mapping', '_sequence')),
    ('solarlab.config.identity', 'solarlab.config.behavior', ('_material_switches',)),
    ('solarlab.config.identity', 'solarlab.physics.semantic_projection', ('_project_historical_physics', '_project_r1_physics')),
    ('solarlab.config.legacy_execution', 'solarlab.config.behavior', ('_binding', '_bytes', '_environment', '_literal', '_mapping', '_signature')),
    ('solarlab.config.legacy_wire', 'solarlab.config.scaps_input', ('_FIELDS', '_ROLE_ALIAS')),
    ('solarlab.config.migration', 'solarlab.config.behavior', ('_mapping',)),
    ('solarlab.config.scaps_input', 'solarlab.config.device_import', ('_common_device', '_flat_defect', '_keys', '_mapping', '_quantity', '_sequence')),
    ('solarlab.config.tandem_input', 'solarlab.config.device_import', ('_keys', '_mapping', '_quantity', '_sequence')),
    ('solarlab.experiments.execution', 'solarlab.experiments.worker', ('_atomic_json', '_child_main', '_directory_bytes', '_name', '_object', '_process_info', '_regular_bytes', '_sha')),
    ('solarlab.experiments.execution', 'solarlab.io.artifacts', ('_parse_document',)),
    ('solarlab.experiments.impedance', 'solarlab.experiments.jv.operating_points', ('_verify_point',)),
    ('solarlab.experiments.ionic_impedance', 'solarlab.experiments.ionic_transient', ('_admit_evidence', '_same_state')),
    ('solarlab.experiments.ionic_initialization', 'solarlab.experiments.ionic_transient', ('_admit_evidence', '_same_state')),
    ('solarlab.experiments.ionic_initialization', 'solarlab.experiments.transient_results', ('_decode', '_encode')),
    ('solarlab.experiments.ionic_transient_results', 'solarlab.experiments.transient_results', ('_decode', '_digest', '_encode', '_fingerprint', '_same')),
    ('solarlab.experiments.ionic_transient_results', 'solarlab.io.artifacts', ('_json_document',)),
    ('solarlab.experiments.ionic_transient_worker', 'solarlab.experiments.jv.worker', ('_PACKAGES', '_THREADS', '_adapter_sources', '_document', '_verify_sources')),
    ('solarlab.experiments.ionic_transient_worker', 'solarlab.experiments.transient_results', ('_fingerprint', '_decode', '_fingerprint')),
    ('solarlab.experiments.ionic_transient_worker', 'solarlab.io.artifacts', ('_json_document', '_parse_document')),
    ('solarlab.experiments.jv.ionic_results', 'solarlab.experiments.ionic_transient_results', ('_finite', '_plain')),
    ('solarlab.experiments.jv.ionic_results', 'solarlab.experiments.transient_results', ('_decode', '_encode', '_fingerprint', '_same')),
    ('solarlab.experiments.jv.ionic_results', 'solarlab.io.artifacts', ('_json_document',)),
    ('solarlab.experiments.jv.ionic_steady', 'solarlab.experiments.ionic_transient', ('_admit_evidence', '_same_state')),
    ('solarlab.experiments.jv.ionic_steady', 'solarlab.experiments.transient_results', ('_decode', '_encode')),
    ('solarlab.experiments.jv.ionic_worker', 'solarlab.experiments.ionic_transient', ('_admit_evidence',)),
    ('solarlab.experiments.jv.ionic_worker', 'solarlab.experiments.jv.worker', ('_PACKAGES', '_THREADS', '_adapter_sources', '_document', '_verify_sources')),
    ('solarlab.experiments.jv.ionic_worker', 'solarlab.experiments.transient_results', ('_fingerprint',)),
    ('solarlab.experiments.jv.steady_2d', 'solarlab.experiments.transient_results', ('_decode', '_encode')),
    ('solarlab.experiments.jv.steady_2d', 'solarlab.experiments.worker', ('_atomic_json',)),
    ('solarlab.experiments.multiple_ionic_impedance', 'solarlab.experiments.ionic_impedance', ('_decoded', '_encoded', '_finite', '_frequencies')),
    ('solarlab.experiments.multiple_ionic_impedance', 'solarlab.experiments.multiple_ionic_transient', ('_admit', '_same_record')),
    ('solarlab.experiments.multiple_ionic_impedance', 'solarlab.experiments.transient_results', ('_decode', '_encode')),
    ('solarlab.experiments.multiple_ionic_transient', 'solarlab.experiments.ionic_transient', ('_finite_accounting',)),
    ('solarlab.experiments.multiple_ionic_transient', 'solarlab.experiments.transient_results', ('_decode', '_encode')),
    ('solarlab.experiments.protocol', 'solarlab.io.artifacts', ('_json_document', '_parse_document')),
    ('solarlab.experiments.registry', 'solarlab.physics.registry', ('_token', '_verify_evidence_sources')),
    ('solarlab.experiments.results', 'solarlab.experiments.protocol', ('_keys', '_reference', '_ref_schema', '_sha', '_strings', '_text')),
    ('solarlab.experiments.results', 'solarlab.io.artifacts', ('_json_document', '_parse_document')),
    ('solarlab.experiments.transient', 'solarlab.experiments.jv.operating_points', ('_verify_point',)),
    ('solarlab.experiments.transient_results', 'solarlab.io.artifacts', ('_json_document', '_parse_document')),
    ('solarlab.experiments.transient_worker', 'solarlab.experiments.jv.worker', ('_PACKAGES', '_THREADS', '_adapter_sources', '_document', '_verify_sources')),
    ('solarlab.experiments.transient_worker', 'solarlab.experiments.transient_results', ('_payload', '_UNITS', '_decode', '_digest', '_inputs', '_result_digest', '_validate')),
    ('solarlab.experiments.transient_worker', 'solarlab.io.artifacts', ('_json_document', '_parse_document')),
    ('solarlab.experiments.two_dimensional.steady_results', 'solarlab.experiments.transient_results', ('_decode', '_encode', '_fingerprint', '_same')),
    ('solarlab.experiments.two_dimensional.steady_results', 'solarlab.experiments.two_dimensional.steady_worker', ('_request_contract',)),
    ('solarlab.experiments.two_dimensional.steady_results', 'solarlab.io.artifacts', ('_json_document',)),
    ('solarlab.experiments.worker', 'solarlab.io.artifacts', ('_json_document', '_parse_document')),
    ('solarlab.io.run_bundle', 'solarlab.io.artifacts', ('_ArtifactFiles', '_json_document', '_load_npy')),
    ('solarlab.io.run_bundle', 'solarlab.io.run_store', ('_identifier', '_public')),
    ('solarlab.io.run_store', 'solarlab.io.artifacts', ('_ArtifactFiles', '_json_document', '_load_npy', '_parse_document')),
    ('solarlab.physics.presets', 'solarlab.physics.registry', ('_strings', '_token', '_verify_evidence_sources')),
    ('solarlab.physics.presets', 'solarlab.physics.selection', ('_choices',)),
    ('solarlab.physics.selection', 'solarlab.physics.registry', ('_strings', '_token')),
)

BDF_EXPERIMENT_IMPORTS: tuple[ExceptionGroup, ...] = (
    ('solarlab.solvers.time.bdf_1d', 'solarlab.experiments.jv.operating_points', ('_verify_point',)),
    ('solarlab.solvers.time.bdf_1d', 'solarlab.experiments.ionic_transient', ('_admit_evidence', '_finite_accounting', '_newton_settings', '_same_state')),
)


PUBLIC_KERNEL_RENAMES: tuple[tuple[str, str, str], ...] = (
    ('solarlab.device.compile_1d', '_unsupported', 'unsupported'),
    ('solarlab.device.compile_ionic_1d', '_ION_INPUTS', 'ION_INPUTS'),
    ('solarlab.equations.dc_1d', '_quasi_fermi_potential_tangents', 'quasi_fermi_potential_tangents'),
    ('solarlab.equations.ionic_small_signal', '_frequency_voltage', 'frequency_voltage'),
    ('solarlab.equations.multiple_ionic_small_signal', '_scope', 'scope'),
    ('solarlab.solvers.nonlinear.ionic_1d', '_checks', 'checks'),
    ('solarlab.solvers.nonlinear.ionic_1d', '_diagnostics', 'diagnostics'),
    ('solarlab.solvers.nonlinear.steady_2d', '_checks', 'checks'),
    ('solarlab.solvers.nonlinear.steady_2d', '_diagnostics', 'diagnostics'),
    ('solarlab.solvers.nonlinear.steady_2d', '_scope', 'scope'),
    ('solarlab.solvers.nonlinear.steady_2d', '_sources', 'sources'),
    ('solarlab.solvers.time.backward_euler', '_storage_mask', 'storage_mask'),
    ('solarlab.solvers.nonlinear.steady_1d', '_physical_gates', 'physical_gates'),
    ('solarlab.solvers.nonlinear.steady_1d', '_density_basin_seed', 'density_basin_seed'),
)


def exception_counts(groups: tuple[ExceptionGroup, ...]) -> Counter[ImportKey]:
    return Counter((importer, provider, name)
                   for importer, provider, names in groups for name in names)


def private_imports(module: str, source: str, *, is_package: bool = False) -> Counter[ImportKey]:
    result: Counter[ImportKey] = Counter()
    package = module if is_package else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom):
            continue
        provider = (resolve_name("." * node.level + (node.module or ""), package)
                    if node.level else node.module)
        if not provider or not provider.startswith("solarlab.") or provider == module:
            continue
        for alias in node.names:
            if alias.name.startswith("_"):
                result[module, provider, alias.name] += 1
    return result


def source_modules() -> list[tuple[str, Path]]:
    result = []
    for path in sorted((ROOT / "src/solarlab").rglob("*.py")):
        parts = path.relative_to(ROOT / "src").with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        result.append((module, path))
    return result


def test_no_unreviewed_private_cross_module_imports() -> None:
    found: Counter[ImportKey] = Counter()
    for module, path in source_modules():
        found.update(private_imports(module, path.read_text(), is_package=path.name == "__init__.py"))
    allowed = exception_counts(KNOWN_LATER_IMPORTS) + exception_counts(BDF_EXPERIMENT_IMPORTS)
    unexpected = found - allowed
    assert not unexpected, (
        "New private imports require public APIs (numerical roots: "
        + ", ".join(CORE_ROOTS) + "): " + repr(dict(unexpected))
    )
    # The reviewed historical debt stays explicit; numerical promotions may not add debt.
    assert found == allowed, "Update the explicit ledger when a reviewed later migration retires an exception"


@pytest.mark.parametrize(("module", "old", "public"), PUBLIC_KERNEL_RENAMES)
def test_public_helpers_have_exports_and_no_private_alias(module: str, old: str, public: str) -> None:
    implementation = import_module(module)
    assert public in implementation.__all__
    assert public in vars(implementation)
    assert old not in vars(implementation)


def promoted_private_aliases(module: str, source: str, *, is_package: bool = False) -> list[tuple[int, str]]:
    promoted = {(module, public) for module, _, public in PUBLIC_KERNEL_RENAMES}
    package = module if is_package else module.rpartition(".")[0]
    failures: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            provider = (resolve_name("." * node.level + (node.module or ""), package)
                        if node.level else node.module)
            for alias in node.names:
                if ((provider, alias.name) in promoted
                        and alias.asname is not None and alias.asname.startswith("_")):
                    failures.append((node.lineno, alias.asname))
    return failures


def test_promoted_helpers_are_not_imported_under_private_aliases() -> None:
    failures: list[tuple[str, int, str]] = []
    for module, path in source_modules():
        failures.extend((module, line, name) for line, name in promoted_private_aliases(
            module, path.read_text(), is_package=path.name == "__init__.py"))
    assert not failures


def test_detector_rejects_private_aliases_of_absolute_and_relative_public_helpers() -> None:
    assert promoted_private_aliases("solarlab.solvers.time.candidate",
        "from solarlab.solvers.time.backward_euler import storage_mask as _hidden") == [(1, "_hidden")]
    assert promoted_private_aliases("solarlab.solvers.time.candidate",
        "from .backward_euler import storage_mask as _hidden") == [(1, "_hidden")]
    assert not promoted_private_aliases("solarlab.solvers.time.candidate",
        "from .backward_euler import storage_mask as boundary_storage_mask")


def test_only_named_later_debts_are_exempt() -> None:
    assert sum(exception_counts(KNOWN_LATER_IMPORTS).values()) == 141
    assert sum(exception_counts(BDF_EXPERIMENT_IMPORTS).values()) == 5
    assert all(provider.startswith("solarlab.experiments.")
               for _, provider, _ in BDF_EXPERIMENT_IMPORTS)
    assert all(not private_imports("solarlab.equations.candidate", source) <= exception_counts(KNOWN_LATER_IMPORTS)
               for source in ("from solarlab.physics.candidate import _new_secret",))


def test_detector_rejects_absolute_relative_and_duplicate_private_imports() -> None:
    absolute = private_imports("solarlab.equations.candidate",
        "from solarlab.physics.example import _secret\n"
        "from solarlab.physics.example import _secret\n")
    assert absolute == Counter({("solarlab.equations.candidate", "solarlab.physics.example", "_secret"): 2})
    assert private_imports("solarlab.equations.candidate", "from .example import _secret") == Counter({
        ("solarlab.equations.candidate", "solarlab.equations.example", "_secret"): 1})
    assert private_imports("solarlab.equations", "from .example import _secret", is_package=True) == Counter({
        ("solarlab.equations", "solarlab.equations.example", "_secret"): 1})
    assert not private_imports("solarlab.equations.candidate",
        "from solarlab.physics.example import public as _local_alias")
