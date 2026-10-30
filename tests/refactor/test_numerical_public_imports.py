"""Symbol-level numerical import boundary with an explicit existing-debt ledger.

Import-linter operates on module dependencies; this check inspects the actual
ImportFrom names, including relative imports. The 141 configuration, registry,
serialization and experiment occurrences from AssemblyMigrationInventoryV1
remain explicit exceptions. Public symbols merely aliased privately are not
part of that historical count; promoted module-level APIs may not keep aliases.

The two temporary steady helper exceptions are now retired. Five additional
BDF-to-experiment imports already present at f0c1027f are listed separately;
their providers are outside the numerical roots. The reviewed 141-entry
configuration/registry/experiment ledger remains unchanged. Method calls have
a separate exact ledger for owned callers and unrelated same-name methods.
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


# Methods are class APIs, so they do not belong in a module's __all__.
PUBLIC_METHOD_RENAMES = (
    ('solarlab.equations.multiple_ionic_1d', 'CompiledMultipleIonicDevice1D', '_values', 'values'),
    ('solarlab.equations.ionic_small_signal', 'IonicSmallSignal1D', '_moving_voltage_partial', 'moving_voltage_partial'),
    ('solarlab.equations.dc_1d', 'DC1DProblem', '_contact_exchange', 'contact_exchange'),
)
MethodCallKey: TypeAlias = tuple[str, str, str, str]  # module, scope, receiver, method

# These exact callers are owned by the concurrent production-wiring migration.
# Retire each entry together with its delegating alias when that caller moves.
DEFERRED_PRIVATE_METHOD_CALLS: Counter[MethodCallKey] = Counter({
    ('solarlab.solvers.time.multiple_ionic_backward_euler',
     'MultipleIonicBackwardEuler1DStep.inventory_rates_m2_s', 'device', '_values'): 1,
    ('solarlab.solvers.time.backward_euler', 'step_accounting', 'problem', '_contact_exchange'): 1,
})

# This is a conservative syntactic guard, not Python type inference. Each
# unrelated receiver with a colliding name is reviewed at this exact scope;
# a module containing a homonym does not exempt its calls on other receivers.
UNRELATED_PRIVATE_METHOD_CALLS: Counter[MethodCallKey] = Counter({
    # A same-module AssembledIonicSmallSignal1D, not IonicSmallSignal1D.
    ('solarlab.equations.assembly.small_signal',
     'AssembledMultipleIonicSmallSignal1D._moving_voltage_partial', 'first', '_moving_voltage_partial'): 1,
    # DC2DProblem has its own vector-valued contact law.
    ('tests.integration.test_contacts_2d',
     'test_contact_flux_sign_units_and_exact_density_jacobian', 'p', '_contact_exchange'): 1,
    ('tests.integration.test_contacts_2d',
     'test_zero_velocity_is_blocking_and_near_reservoir_excess_is_not_rounded_away',
     'problem', '_contact_exchange'): 2,
    ('tests.integration.test_contacts_2d',
     'test_finite_exchange_has_the_compatible_fixed_reservoir_limit', 'problem', '_contact_exchange'): 1,
    # MultipleIonicSmallSignal1D's distinct domain-combining implementation.
    ('tests.integration.test_multiple_ionic_small_signal_1d',
     'test_full_J_K_and_actual_moving_voltage_drive_directions', 'operator', '_moving_voltage_partial'): 1,
    # Assembled single/multiple operators and the assembled storage adapter.
    ('tests.unit.assembly.test_assembly_small_signal',
     'test_moving_lift_drive_and_rate_match_finite_differences', 'operator', '_moving_voltage_partial'): 1,
    ('tests.unit.assembly.test_assembly_small_signal',
     'test_invalid_inputs_are_refused_as_before', 'operator', '_moving_voltage_partial'): 1,
    ('tests.unit.solvers.time.test_assembled_storage',
     'test_public_exchange_mapping_preserves_bound_channel_law_and_refusals', 'assembled', '_contact_exchange'): 1,
})


def private_method_calls(module: str, source: str) -> Counter[MethodCallKey]:
    """Find calls of promoted names; waive only defining modules/local methods."""
    providers = {old: provider for provider, _, old, _ in PUBLIC_METHOD_RENAMES}
    found: Counter[MethodCallKey] = Counter()

    class Calls(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: list[str] = []
            self.classes: list[ast.ClassDef] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            self.classes.append(node)
            self.generic_visit(node)
            self.classes.pop()
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and node.func.attr in providers:
                receiver, name = node.func.value, node.func.attr
                local_method = (
                    self.classes and isinstance(receiver, ast.Name) and receiver.id in {'self', 'cls'}
                    and any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
                            for item in self.classes[-1].body)
                )
                if module != providers[name] and not local_method:
                    found[module, '.'.join(self.scope) or '<module>', ast.unparse(receiver), name] += 1
            self.generic_visit(node)

    Calls().visit(ast.parse(source))
    return found


def test_no_unreviewed_private_cross_module_method_calls() -> None:
    found: Counter[MethodCallKey] = Counter()
    for directory in ('src', 'tests', 'examples', 'scripts'):
        base = ROOT / 'src' if directory == 'src' else ROOT
        for path in sorted((ROOT / directory).rglob('*.py')):
            parts = path.relative_to(base).with_suffix('').parts
            module = '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)
            found.update(private_method_calls(module, path.read_text()))
    allowed = DEFERRED_PRIVATE_METHOD_CALLS + UNRELATED_PRIVATE_METHOD_CALLS
    assert not found - allowed, (
        'Use the public method, or document an exact unrelated receiver: ' + repr(dict(found - allowed))
    )
    assert found == allowed, 'Retire stale private-method exceptions with their migrated callers'


@pytest.mark.parametrize(('module', 'owner', 'old', 'public'), PUBLIC_METHOD_RENAMES)
def test_public_methods_and_only_required_delegating_aliases(
    module: str, owner: str, old: str, public: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = import_module(module)
    cls = getattr(implementation, owner)
    assert public in vars(cls) and callable(getattr(cls, public))
    assert public not in implementation.__all__
    deferred = {key[-1] for key in DEFERRED_PRIVATE_METHOD_CALLS}
    if old not in deferred:
        assert not hasattr(cls, old)
        return
    sentinel = object()
    calls: list[tuple[object, tuple[object, ...]]] = []

    def public_spy(instance: object, *args: object) -> object:
        calls.append((instance, args))
        return sentinel

    monkeypatch.setattr(cls, public, public_spy)
    instance = object.__new__(cls)
    args = (object(),) if old == '_values' else (object(), object(), object())
    assert getattr(instance, old)(*args) is sentinel
    assert calls == [(instance, args)]


def test_method_detector_rejects_unknown_receivers_aliases_and_repeated_calls() -> None:
    source = (
        'from solarlab.equations.dc_1d import DC1DProblem as Renamed\n'
        'Renamed._contact_exchange(problem, u, channel, node)\n'
        'def evaluate(device):\n'
        '    device._values(y)\n'
        '    device._values(ydot)\n'
        '    operator._moving_voltage_partial()\n'
    )
    assert private_method_calls('solarlab.equations.candidate', source) == Counter({
        ('solarlab.equations.candidate', '<module>', 'Renamed', '_contact_exchange'): 1,
        ('solarlab.equations.candidate', 'evaluate', 'device', '_values'): 2,
        ('solarlab.equations.candidate', 'evaluate', 'operator', '_moving_voltage_partial'): 1,
    })


def test_local_homonym_does_not_hide_a_foreign_receiver() -> None:
    source = (
        'class Local:\n'
        '    def _moving_voltage_partial(self):\n'
        '        return 0\n'
        '    def evaluate(self, first):\n'
        '        self._moving_voltage_partial()\n'
        '        return first._moving_voltage_partial()\n'
    )
    assert private_method_calls('solarlab.equations.candidate', source) == Counter({
        ('solarlab.equations.candidate', 'Local.evaluate', 'first', '_moving_voltage_partial'): 1,
    })
    assert not private_method_calls('solarlab.equations.ionic_small_signal', source)
    assert not private_method_calls('solarlab.equations.candidate',
        'device.values(y)\noperator.moving_voltage_partial()\nproblem.contact_exchange(u, channel, node)')


def test_private_method_deferrals_are_limited_to_the_two_owned_callers() -> None:
    assert sum(DEFERRED_PRIVATE_METHOD_CALLS.values()) == 2
    assert {key[0] for key in DEFERRED_PRIVATE_METHOD_CALLS} == {
        'solarlab.solvers.time.backward_euler', 'solarlab.solvers.time.multiple_ionic_backward_euler',
    }
