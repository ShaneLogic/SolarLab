# Layer Builder UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Phase 2b of the workstation upgrade — a vertical stack visualizer + split detail editor + template library + user-preset Save-As path, gated to the `full` tier. Built strictly additively over Phase 2a; the Python solver and shipped YAMLs are untouched.

**Architecture:** Frontend reorganization + two new backend endpoints. New `frontend/src/stack/` directory holds eight small modules (visualizer, layer card, interface strip, validator, two helpers, two dialogs); `frontend/src/config-editor.ts` is shrunk to render only the selected layer in full tier; `frontend/src/workstation/panes/device-pane.ts` becomes a two-column host. Backend gains `backend/user_configs.py` plus `GET /api/layer-templates`, `POST /api/configs/user`, and a one-line recursion update to `GET /api/configs`. New data file `perovskite_sim/data/layer_templates.yaml` holds the starter library.

**Tech Stack:** TypeScript (strict, no framework), Vite, Vitest, Plotly, FastAPI, Pydantic, pytest, PyYAML, native HTML5 drag-and-drop.

**Spec:** `docs/superpowers/specs/2026-04-14-layer-builder-ui-design.md`

---

## File map (lock decomposition before tasks)

**New backend / data files:**
- `backend/user_configs.py` — filename validation, shipped-name reservation, atomic write helper.
- `perovskite_sim/data/layer_templates.yaml` — hand-authored starter layers with citations.
- `tests/unit/backend/test_user_configs.py`
- `tests/integration/backend/__init__.py`
- `tests/integration/backend/test_user_configs_api.py`
- `tests/unit/backend/__init__.py` (only if missing)

**Modified backend files:**
- `backend/main.py` — add three endpoint changes, import `user_configs`.

**New frontend files (all under `frontend/src/stack/`):**
- `log-scale-height.ts` — pure helper.
- `dirty-state.ts` — pure helper.
- `reconcile-interfaces.ts` — pure helper.
- `stack-validator.ts` — pure validator.
- `stack-layer-card.ts` — pure render + event-wiring fn for one card.
- `stack-interface-strip.ts` — render + edit one interface row.
- `stack-visualizer.ts` — composes cards + strips, owns DnD events, exposes `render()` + `onAction` callback.
- `add-layer-dialog.ts` — modal with template/blank tabs.
- `save-as-dialog.ts` — modal with filename input + collision probe.
- `__tests__/log-scale-height.test.ts`
- `__tests__/dirty-state.test.ts`
- `__tests__/reconcile-interfaces.test.ts`
- `__tests__/stack-validator.test.ts`
- `__tests__/config-editor-superset.test.ts` (lives in `__tests__/` even though it tests `config-editor.ts`)

**Modified frontend files:**
- `frontend/src/types.ts` — `LayerTemplate`, `ValidationIssue`, `ValidationReport`, `StackAction`, `Namespace`, role literal-union, `Readonly` markers.
- `frontend/src/api.ts` — `fetchLayerTemplates`, `saveUserConfig`, `checkUserConfigExists`, change `listConfigs` return shape to `ConfigEntry[]`.
- `frontend/src/workstation/tier-gating.ts` — `isLayerBuilderEnabled(tier)`.
- `frontend/src/config-editor.ts` — single-layer mode in full tier; extended role values; identical behavior in fast/legacy.
- `frontend/src/device-panel.ts` — render dropdown as two `<optgroup>`s (Shipped / User), wire dirty-pill update.
- `frontend/src/workstation/panes/device-pane.ts` — two-column CSS-grid layout in full tier; mounts visualizer + detail editor; passes selectedLayerIdx and dirty state through.
- `frontend/src/style.css` — visualizer column styles, role color CSS variables, drag/hover affordances, 1100 px breakpoint.
- `frontend/src/panels/tutorial.ts` — "Custom Stacks" section.
- `frontend/src/panels/parameters.ts` — role row update.
- `CLAUDE.md` — Phase 2b activation note.

**Untouched (regression-safe):**
- All Python solver code (`perovskite_sim/{solver,physics,experiments,models}/`).
- All shipped YAMLs in `configs/*.yaml`.
- All existing pytest tests.
- `frontend/src/job-stream.ts`, `frontend/src/panels/{jv,impedance,degradation}.ts`.

---

## Type contracts (locked across all tasks)

These signatures are referenced by multiple tasks. Treat them as the single source of truth:

```typescript
// frontend/src/types.ts (additions)

export type LayerRole =
  | 'substrate'
  | 'front_contact'
  | 'ETL'
  | 'absorber'
  | 'HTL'
  | 'back_contact'

export interface LayerTemplate {
  role: LayerRole
  optical_material: string | null
  description: string
  source: string
  defaults: Partial<LayerConfig>
}

export interface ValidationIssue {
  layerIdx: number | null   // null means stack-level
  field: string | null
  message: string
}

export interface ValidationReport {
  errors: ValidationIssue[]
  warnings: ValidationIssue[]
}

export type StackAction =
  | { type: 'select'; idx: number }
  | { type: 'delete'; idx: number }
  | { type: 'reorder'; from: number; to: number }
  | { type: 'insert'; atIdx: number; layer: LayerConfig }
  | { type: 'edit-interface'; idx: number; pair: readonly [number, number] }

export type Namespace = 'shipped' | 'user'

export interface ConfigEntry {
  name: string         // basename without .yaml extension OR with — see Task 4
  namespace: Namespace
}
```

```python
# backend/user_configs.py (locked signatures)

USER_FILENAME_RE: re.Pattern   # ^[a-zA-Z0-9_-]{1,64}$
CONFIGS_ROOT: Path             # perovskite-sim/configs/
USER_CONFIGS_ROOT: Path        # perovskite-sim/configs/user/

def validate_user_filename(name: str) -> None: ...
def is_shipped_name(name: str) -> bool: ...
def write_user_config(name: str, body: dict, *, overwrite: bool = False) -> Path: ...
def list_user_configs() -> list[str]: ...   # bare names, sorted
```

---

# Phase A — Backend foundations

## Task 1: `backend/user_configs.py` — filename validation + shipped-name check (TDD)

**Files:**
- Create: `backend/user_configs.py`
- Create: `tests/unit/backend/test_user_configs.py`
- Create (if missing): `tests/unit/backend/__init__.py`

- [ ] **Step 1: Create the test directory init file**

```bash
test -f "perovskite-sim/tests/unit/backend/__init__.py" || \
  touch "perovskite-sim/tests/unit/backend/__init__.py"
```

- [ ] **Step 2: Write the failing tests for filename validation**

Write to `perovskite-sim/tests/unit/backend/test_user_configs.py`:

```python
"""Unit tests for backend.user_configs (Phase 2b layer builder)."""
import pytest

from backend.user_configs import (
    USER_FILENAME_RE,
    is_shipped_name,
    validate_user_filename,
)


class TestValidateUserFilename:
    @pytest.mark.parametrize("name", [
        "my_stack",
        "MyStack-01",
        "abc",
        "ABC_123-xyz",
        "a" * 64,
    ])
    def test_accepts_valid_names(self, name: str) -> None:
        validate_user_filename(name)  # must not raise

    @pytest.mark.parametrize("name", [
        "",                # empty
        "a" * 65,          # too long
        "../etc/passwd",   # path traversal
        "foo/bar",         # slash
        "foo bar",         # space
        "foo.yaml",        # dot
        "foo$",            # special char
        "café",            # non-ASCII
        ".hidden",         # leading dot
    ])
    def test_rejects_invalid_names(self, name: str) -> None:
        with pytest.raises(ValueError):
            validate_user_filename(name)

    def test_regex_anchored(self) -> None:
        assert USER_FILENAME_RE.pattern.startswith("^")
        assert USER_FILENAME_RE.pattern.endswith("$")


class TestIsShippedName:
    def test_known_shipped_names(self) -> None:
        # These are guaranteed to ship at the top of configs/.
        assert is_shipped_name("nip_MAPbI3")
        assert is_shipped_name("pin_MAPbI3")
        assert is_shipped_name("nip_MAPbI3_tmm")

    def test_unknown_names_are_not_shipped(self) -> None:
        assert not is_shipped_name("definitely_not_a_real_preset_xyz")

    def test_does_not_match_extensions(self) -> None:
        # Pass bare name; extension is appended internally.
        assert is_shipped_name("nip_MAPbI3")
        assert not is_shipped_name("nip_MAPbI3.yaml")  # caller passes bare names
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd perovskite-sim && pytest tests/unit/backend/test_user_configs.py -v
```

Expected: ImportError / ModuleNotFoundError (`backend.user_configs` does not exist yet).

- [ ] **Step 4: Implement `backend/user_configs.py` minimally to pass**

Write to `perovskite-sim/backend/user_configs.py`:

```python
"""User-preset filesystem helpers for Phase 2b layer builder.

Kept in its own module so filename validation and shipped-name reservation
can be unit-tested without spinning up FastAPI. The module is the single
authority on:

  * the user-preset filename grammar (a strict ASCII regex);
  * which names are reserved by shipped presets;
  * atomic file creation that prevents TOCTOU collisions on first save.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

USER_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

CONFIGS_ROOT = Path(__file__).resolve().parent.parent / "configs"
USER_CONFIGS_ROOT = CONFIGS_ROOT / "user"


def validate_user_filename(name: str) -> None:
    """Raise ValueError if name is not a safe user-preset filename."""
    if not isinstance(name, str) or not USER_FILENAME_RE.match(name):
        raise ValueError(
            f"Invalid filename {name!r}: must match {USER_FILENAME_RE.pattern}"
        )


def is_shipped_name(name: str) -> bool:
    """Return True if name collides with a shipped (top-level) preset.

    Callers must pass the bare name, without an extension.
    """
    if "." in name or "/" in name:
        return False
    return (CONFIGS_ROOT / f"{name}.yaml").exists()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd perovskite-sim && pytest tests/unit/backend/test_user_configs.py -v
```

Expected: PASS — all parameterized cases green.

- [ ] **Step 6: Commit**

```bash
cd perovskite-sim && git add backend/user_configs.py tests/unit/backend/__init__.py tests/unit/backend/test_user_configs.py
git commit -m "feat(backend): add user_configs filename validation and shipped-name check

Phase 2b foundation. user_configs.py owns the user-preset
filename grammar (^[a-zA-Z0-9_-]{1,64}$) and the shipped-name
reservation check used by POST /api/configs/user."
```

---

## Task 2: `backend/user_configs.py` — atomic write + listing (TDD)

**Files:**
- Modify: `backend/user_configs.py` — add `write_user_config`, `list_user_configs`
- Modify: `tests/unit/backend/test_user_configs.py` — add `TestWriteUserConfig`, `TestListUserConfigs`

- [ ] **Step 1: Add failing tests for write + list**

Append to `perovskite-sim/tests/unit/backend/test_user_configs.py`:

```python
import yaml
from pathlib import Path
from backend.user_configs import (
    list_user_configs,
    write_user_config,
    USER_CONFIGS_ROOT,
)


@pytest.fixture
def isolated_user_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect USER_CONFIGS_ROOT to a tmp dir for write/list tests."""
    fake_root = tmp_path / "configs" / "user"
    monkeypatch.setattr("backend.user_configs.USER_CONFIGS_ROOT", fake_root)
    return fake_root


class TestWriteUserConfig:
    def test_creates_file_atomically_on_first_save(
        self, isolated_user_root: Path
    ) -> None:
        body = {"device": {"V_bi": 1.1}, "layers": []}
        target = write_user_config("my_stack", body)
        assert target == isolated_user_root / "my_stack.yaml"
        assert target.exists()
        with target.open() as f:
            assert yaml.safe_load(f) == body

    def test_refuses_overwrite_by_default(self, isolated_user_root: Path) -> None:
        write_user_config("my_stack", {"x": 1})
        with pytest.raises(FileExistsError):
            write_user_config("my_stack", {"x": 2})

    def test_allows_overwrite_when_explicit(
        self, isolated_user_root: Path
    ) -> None:
        write_user_config("my_stack", {"x": 1})
        write_user_config("my_stack", {"x": 2}, overwrite=True)
        with (isolated_user_root / "my_stack.yaml").open() as f:
            assert yaml.safe_load(f) == {"x": 2}

    def test_rejects_invalid_filename(self, isolated_user_root: Path) -> None:
        with pytest.raises(ValueError):
            write_user_config("../etc/passwd", {})

    def test_rejects_shipped_name_collision(
        self, isolated_user_root: Path
    ) -> None:
        # nip_MAPbI3 is a shipped preset name (Task 1 verified this).
        with pytest.raises(FileExistsError, match="shipped"):
            write_user_config("nip_MAPbI3", {})

    def test_creates_user_root_if_missing(
        self, isolated_user_root: Path
    ) -> None:
        assert not isolated_user_root.exists()
        write_user_config("first", {})
        assert isolated_user_root.is_dir()


class TestListUserConfigs:
    def test_returns_empty_when_no_dir(self, isolated_user_root: Path) -> None:
        assert list_user_configs() == []

    def test_returns_sorted_bare_names(self, isolated_user_root: Path) -> None:
        write_user_config("zebra", {})
        write_user_config("alpha", {})
        write_user_config("mike", {})
        assert list_user_configs() == ["alpha", "mike", "zebra"]
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd perovskite-sim && pytest tests/unit/backend/test_user_configs.py::TestWriteUserConfig -v
```

Expected: FAIL — `write_user_config` does not exist.

- [ ] **Step 3: Implement `write_user_config` and `list_user_configs`**

Append to `perovskite-sim/backend/user_configs.py`:

```python
def write_user_config(
    name: str,
    body: dict,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a validated user config atomically.

    Uses ``os.O_EXCL`` on first save so two concurrent saves of the same
    name cannot both succeed (the second raises ``FileExistsError``). A
    second save with ``overwrite=True`` truncates the existing file.

    Raises:
        ValueError: filename does not match USER_FILENAME_RE.
        FileExistsError: name collides with a shipped preset, or the
            target already exists and ``overwrite`` is False.
    """
    validate_user_filename(name)
    if is_shipped_name(name):
        raise FileExistsError(f"{name!r} is reserved by a shipped preset")
    USER_CONFIGS_ROOT.mkdir(parents=True, exist_ok=True)
    target = USER_CONFIGS_ROOT / f"{name}.yaml"
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
    try:
        fd = os.open(target, flags, 0o644)
    except FileExistsError:
        raise FileExistsError(f"{name!r} already exists")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(body, f, default_flow_style=False, sort_keys=False)
    return target


def list_user_configs() -> list[str]:
    """Return a sorted list of user-preset bare names (no extension)."""
    if not USER_CONFIGS_ROOT.is_dir():
        return []
    return sorted(p.stem for p in USER_CONFIGS_ROOT.glob("*.yaml"))
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd perovskite-sim && pytest tests/unit/backend/test_user_configs.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
cd perovskite-sim && git add backend/user_configs.py tests/unit/backend/test_user_configs.py
git commit -m "feat(backend): add atomic user-config write and list helpers

write_user_config uses O_EXCL on first save (TOCTOU-safe) and
honors overwrite=True for explicit user-confirmed overwrites.
list_user_configs returns bare names sorted alphabetically."
```

---

## Task 3: Layer template library data file + `GET /api/layer-templates` (TDD)

**Files:**
- Create: `perovskite_sim/data/layer_templates.yaml`
- Modify: `backend/main.py` — add `/api/layer-templates` endpoint
- Create: `tests/integration/backend/__init__.py` (if missing)
- Create: `tests/integration/backend/test_user_configs_api.py` — API integration tests

- [ ] **Step 1: Create the integration test directory init file**

```bash
test -f "perovskite-sim/tests/integration/backend/__init__.py" || \
  touch "perovskite-sim/tests/integration/backend/__init__.py"
```

- [ ] **Step 2: Write failing test for `/api/layer-templates`**

Write to `perovskite-sim/tests/integration/backend/test_user_configs_api.py`:

```python
"""Integration tests for Phase 2b backend endpoints."""
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


class TestLayerTemplatesEndpoint:
    def test_returns_dict_of_templates(self) -> None:
        r = client.get("/api/layer-templates")
        assert r.status_code == 200
        body = r.json()
        assert "templates" in body
        templates = body["templates"]
        assert isinstance(templates, dict)
        # Spec requires at least these starter templates.
        for required in [
            "TiO2_ETL",
            "spiro_HTL",
            "MAPbI3_absorber",
            "glass_substrate",
            "Au_back_contact",
        ]:
            assert required in templates, f"missing template: {required}"

    def test_each_template_has_required_fields(self) -> None:
        r = client.get("/api/layer-templates")
        for name, tmpl in r.json()["templates"].items():
            assert "role" in tmpl, f"{name} missing role"
            assert "description" in tmpl, f"{name} missing description"
            assert "source" in tmpl, f"{name} missing source"
            assert "defaults" in tmpl, f"{name} missing defaults"
            assert "thickness" in tmpl["defaults"], (
                f"{name} defaults missing thickness"
            )
```

- [ ] **Step 3: Run test to verify failure**

```bash
cd perovskite-sim && pytest tests/integration/backend/test_user_configs_api.py::TestLayerTemplatesEndpoint -v
```

Expected: 404 — endpoint does not exist.

- [ ] **Step 4: Create the layer template library data file**

Write to `perovskite-sim/perovskite_sim/data/layer_templates.yaml`:

```yaml
# Layer template library for Phase 2b custom-stack builder.
# Each entry: role, optical_material (or null), description, source, defaults.
# Defaults come from cited published cells; tune at PR review if a stack
# fails to converge during the manual verification pass.

glass_substrate:
  role: substrate
  optical_material: glass
  description: "BK7-equivalent glass substrate (1 mm) — front side"
  source: "Schott BK7 datasheet"
  defaults:
    thickness: 1.0e-3
    incoherent: true
    eps_r: 4.6
    mu_n: 0
    mu_p: 0
    ni: 0
    N_D: 0
    N_A: 0
    D_ion: 0
    P_lim: 0
    P0: 0
    tau_n: 1.0e-9
    tau_p: 1.0e-9
    n1: 1.0e10
    p1: 1.0e10
    B_rad: 0
    C_n: 0
    C_p: 0
    alpha: 0
    chi: 0
    Eg: 9.0

FTO_front_contact:
  role: front_contact
  optical_material: FTO
  description: "F:SnO2 front contact on glass (degenerate semiconductor approx)"
  source: "Filipic 2015, Opt. Express 23, A263"
  defaults:
    thickness: 5.0e-7
    eps_r: 9.0
    mu_n: 5.0e-3
    mu_p: 1.0e-3
    ni: 1.0e10
    N_D: 1.0e26
    N_A: 0
    D_ion: 0
    P_lim: 0
    P0: 0
    tau_n: 1.0e-9
    tau_p: 1.0e-9
    n1: 1.0e10
    p1: 1.0e10
    B_rad: 0
    C_n: 0
    C_p: 0
    alpha: 0
    chi: 4.4
    Eg: 3.5
    incoherent: false

ITO_front_contact:
  role: front_contact
  optical_material: ITO
  description: "Sn:In2O3 front contact (sputtered TCO, p-i-n stacks)"
  source: "König 2014, Opt. Mater. Express 4, 689"
  defaults:
    thickness: 1.5e-7
    eps_r: 9.0
    mu_n: 4.0e-3
    mu_p: 1.0e-3
    ni: 1.0e10
    N_D: 1.0e26
    N_A: 0
    D_ion: 0
    P_lim: 0
    P0: 0
    tau_n: 1.0e-9
    tau_p: 1.0e-9
    n1: 1.0e10
    p1: 1.0e10
    B_rad: 0
    C_n: 0
    C_p: 0
    alpha: 0
    chi: 4.4
    Eg: 3.5
    incoherent: false

TiO2_ETL:
  role: ETL
  optical_material: TiO2
  description: "Compact TiO2 ETL for planar n-i-p perovskite cells"
  source: "Liu et al. 2014, Nature 501, 395"
  defaults:
    thickness: 5.0e-8
    eps_r: 30
    mu_n: 1.0e-7
    mu_p: 1.0e-7
    ni: 1.0e10
    N_D: 1.0e22
    N_A: 0
    D_ion: 0
    P_lim: 0
    P0: 0
    tau_n: 1.0e-7
    tau_p: 1.0e-7
    n1: 1.0e10
    p1: 1.0e10
    B_rad: 0
    C_n: 0
    C_p: 0
    alpha: 0
    chi: 4.0
    Eg: 3.2
    incoherent: false

SnO2_ETL:
  role: ETL
  optical_material: SnO2
  description: "Modern SnO2 ETL (replaces TiO2 in many n-i-p stacks)"
  source: "Jiang et al. 2017, Nat. Energy 2, 16177"
  defaults:
    thickness: 5.0e-8
    eps_r: 9
    mu_n: 2.0e-3
    mu_p: 2.0e-4
    ni: 1.0e10
    N_D: 1.0e23
    N_A: 0
    D_ion: 0
    P_lim: 0
    P0: 0
    tau_n: 1.0e-7
    tau_p: 1.0e-7
    n1: 1.0e10
    p1: 1.0e10
    B_rad: 0
    C_n: 0
    C_p: 0
    alpha: 0
    chi: 4.0
    Eg: 3.6
    incoherent: false

C60_ETL:
  role: ETL
  optical_material: C60
  description: "Fullerene C60 ETL for inverted (p-i-n) cells"
  source: "Ren et al. 2015, Adv. Mater. 27, 6729"
  defaults:
    thickness: 3.0e-8
    eps_r: 4.0
    mu_n: 1.0e-6
    mu_p: 1.0e-6
    ni: 1.0e10
    N_D: 1.0e21
    N_A: 0
    D_ion: 0
    P_lim: 0
    P0: 0
    tau_n: 1.0e-7
    tau_p: 1.0e-7
    n1: 1.0e10
    p1: 1.0e10
    B_rad: 0
    C_n: 0
    C_p: 0
    alpha: 0
    chi: 3.9
    Eg: 1.7
    incoherent: false

PCBM_ETL:
  role: ETL
  optical_material: PCBM
  description: "PCBM fullerene derivative ETL (alternative to C60 in p-i-n)"
  source: "Ren et al. 2015, Adv. Mater. 27, 6729"
  defaults:
    thickness: 5.0e-8
    eps_r: 4.0
    mu_n: 1.0e-6
    mu_p: 1.0e-6
    ni: 1.0e10
    N_D: 1.0e21
    N_A: 0
    D_ion: 0
    P_lim: 0
    P0: 0
    tau_n: 1.0e-7
    tau_p: 1.0e-7
    n1: 1.0e10
    p1: 1.0e10
    B_rad: 0
    C_n: 0
    C_p: 0
    alpha: 0
    chi: 3.9
    Eg: 2.0
    incoherent: false

MAPbI3_absorber:
  role: absorber
  optical_material: MAPbI3
  description: "Methylammonium lead iodide perovskite absorber (~400 nm)"
  source: "Phillips et al. 2018, Sci. Rep. 8, 6473"
  defaults:
    thickness: 4.0e-7
    eps_r: 24.1
    mu_n: 2.0e-4
    mu_p: 2.0e-4
    ni: 1.0e12
    N_D: 0
    N_A: 0
    D_ion: 1.0e-16
    P_lim: 1.6e25
    P0: 1.6e25
    tau_n: 3.0e-9
    tau_p: 3.0e-9
    n1: 1.0e10
    p1: 1.0e10
    B_rad: 9.0e-17
    C_n: 0
    C_p: 0
    alpha: 0
    chi: 4.0
    Eg: 1.6
    incoherent: false

spiro_HTL:
  role: HTL
  optical_material: spiro_OMeTAD
  description: "spiro-OMeTAD HTL (n-i-p stacks)"
  source: "Saliba et al. 2016, Energy Environ. Sci. 9, 1989"
  defaults:
    thickness: 2.0e-7
    eps_r: 3.0
    mu_n: 1.0e-8
    mu_p: 1.0e-8
    ni: 1.0e10
    N_D: 0
    N_A: 1.0e22
    D_ion: 0
    P_lim: 0
    P0: 0
    tau_n: 1.0e-7
    tau_p: 1.0e-7
    n1: 1.0e10
    p1: 1.0e10
    B_rad: 0
    C_n: 0
    C_p: 0
    alpha: 0
    chi: 2.45
    Eg: 3.0
    incoherent: false

PEDOT_PSS_HTL:
  role: HTL
  optical_material: PEDOT_PSS
  description: "PEDOT:PSS HTL (p-i-n stacks)"
  source: "Lee et al. 2018, J. Mater. Chem. A 6, 6105"
  defaults:
    thickness: 4.0e-8
    eps_r: 3.0
    mu_n: 1.0e-8
    mu_p: 1.0e-8
    ni: 1.0e10
    N_D: 0
    N_A: 1.0e22
    D_ion: 0
    P_lim: 0
    P0: 0
    tau_n: 1.0e-7
    tau_p: 1.0e-7
    n1: 1.0e10
    p1: 1.0e10
    B_rad: 0
    C_n: 0
    C_p: 0
    alpha: 0
    chi: 3.4
    Eg: 1.6
    incoherent: false

Au_back_contact:
  role: back_contact
  optical_material: Au
  description: "Gold back contact (degenerate semiconductor approx, n-i-p)"
  source: "Johnson & Christy 1972, Phys. Rev. B 6, 4370"
  defaults:
    thickness: 1.0e-7
    eps_r: 6.9
    mu_n: 1.0e-3
    mu_p: 1.0e-3
    ni: 1.0e10
    N_D: 0
    N_A: 1.0e26
    D_ion: 0
    P_lim: 0
    P0: 0
    tau_n: 1.0e-9
    tau_p: 1.0e-9
    n1: 1.0e10
    p1: 1.0e10
    B_rad: 0
    C_n: 0
    C_p: 0
    alpha: 0
    chi: 4.0
    Eg: 3.0
    incoherent: false

Ag_back_contact:
  role: back_contact
  optical_material: Ag
  description: "Silver back contact / reflector (p-i-n stacks)"
  source: "Johnson & Christy 1972, Phys. Rev. B 6, 4370"
  defaults:
    thickness: 1.0e-7
    eps_r: 6.0
    mu_n: 1.0e-3
    mu_p: 1.0e-3
    ni: 1.0e10
    N_D: 1.0e26
    N_A: 0
    D_ion: 0
    P_lim: 0
    P0: 0
    tau_n: 1.0e-9
    tau_p: 1.0e-9
    n1: 1.0e10
    p1: 1.0e10
    B_rad: 0
    C_n: 0
    C_p: 0
    alpha: 0
    chi: 4.3
    Eg: 3.0
    incoherent: false
```

- [ ] **Step 5: Add the `/api/layer-templates` endpoint**

Add to `perovskite-sim/backend/main.py`, right after the existing `list_optical_materials` function (around line 199):

```python
@app.get("/api/layer-templates")
def list_layer_templates() -> dict:
    """Return the parsed layer templates library used by the Add Layer dialog.

    The library lives in ``perovskite_sim/data/layer_templates.yaml`` so the
    frontend can populate the dialog without re-deriving material defaults.
    """
    path = (
        Path(__file__).resolve().parent.parent
        / "perovskite_sim"
        / "data"
        / "layer_templates.yaml"
    )
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail="layer_templates.yaml missing — Phase 2b data file not installed",
        )
    with path.open() as f:
        templates = yaml.safe_load(f) or {}
    return {"status": "ok", "templates": _coerce_numbers(templates)}
```

- [ ] **Step 6: Run integration tests to verify pass**

```bash
cd perovskite-sim && pytest tests/integration/backend/test_user_configs_api.py::TestLayerTemplatesEndpoint -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd perovskite-sim && git add perovskite_sim/data/layer_templates.yaml backend/main.py tests/integration/backend/__init__.py tests/integration/backend/test_user_configs_api.py
git commit -m "feat(backend): add layer template library + /api/layer-templates

Twelve starter layers (substrate / FTO / ITO / TiO2 / SnO2 / C60 /
PCBM / MAPbI3 / spiro / PEDOT:PSS / Au / Ag) with cited defaults.
The endpoint auto-loads the YAML and coerces scientific-notation
strings the same way /api/configs does."
```

---

## Task 4: Recurse `configs/user/` in `GET /api/configs` (TDD)

**Files:**
- Modify: `backend/main.py` — change `list_configs` to recurse + tag namespace
- Modify: `tests/integration/backend/test_user_configs_api.py` — add namespace test

- [ ] **Step 1: Add a failing test for the new response shape**

Append to `perovskite-sim/tests/integration/backend/test_user_configs_api.py`:

```python
import shutil
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
USER_DIR = CONFIGS_DIR / "user"


@pytest.fixture
def clean_user_dir() -> None:
    if USER_DIR.exists():
        shutil.rmtree(USER_DIR)
    yield
    if USER_DIR.exists():
        shutil.rmtree(USER_DIR)


import pytest


class TestListConfigsNamespace:
    def test_returns_entries_with_namespace(self, clean_user_dir) -> None:
        r = client.get("/api/configs")
        assert r.status_code == 200
        body = r.json()
        assert "configs" in body
        configs = body["configs"]
        # Each entry must be {name, namespace}.
        assert all(isinstance(c, dict) for c in configs)
        assert all("name" in c and "namespace" in c for c in configs)
        shipped = [c for c in configs if c["namespace"] == "shipped"]
        assert any(c["name"].startswith("nip_MAPbI3") for c in shipped)

    def test_includes_user_presets(self, clean_user_dir) -> None:
        USER_DIR.mkdir(parents=True, exist_ok=True)
        (USER_DIR / "my_test.yaml").write_text("device: {V_bi: 1.0}\nlayers: []\n")
        r = client.get("/api/configs")
        configs = r.json()["configs"]
        user_entries = [c for c in configs if c["namespace"] == "user"]
        assert any(c["name"] == "my_test.yaml" for c in user_entries)

    def test_user_dir_missing_does_not_break_listing(
        self, clean_user_dir
    ) -> None:
        # USER_DIR removed by fixture; endpoint must still 200.
        r = client.get("/api/configs")
        assert r.status_code == 200
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd perovskite-sim && pytest tests/integration/backend/test_user_configs_api.py::TestListConfigsNamespace -v
```

Expected: FAIL — current `list_configs` returns a flat list of strings, not dicts.

- [ ] **Step 3: Update `list_configs` in `backend/main.py`**

Find the existing `list_configs` (around line 176) and replace it with:

```python
@app.get("/api/configs")
def list_configs():
    """List YAML configs available to the frontend.

    Each entry carries a ``namespace`` tag so the frontend can render the
    dropdown as two ``<optgroup>``s — shipped (top-level configs/) and user
    (configs/user/). Returning a list of dicts is a deliberate breaking
    change vs the Phase 2a flat-list shape; the frontend api wrapper updates
    in lockstep.
    """
    try:
        entries: list[dict] = []
        for f in sorted(os.listdir(CONFIGS_DIR)):
            if f.endswith((".yaml", ".yml")):
                full = os.path.join(CONFIGS_DIR, f)
                if os.path.isfile(full):
                    entries.append({"name": f, "namespace": "shipped"})
        user_dir = os.path.join(CONFIGS_DIR, "user")
        if os.path.isdir(user_dir):
            for f in sorted(os.listdir(user_dir)):
                if f.endswith((".yaml", ".yml")):
                    entries.append({"name": f, "namespace": "user"})
        return {"status": "ok", "configs": entries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

Also update `get_config` (around line 201) so that user-namespace files resolve correctly. Replace:

```python
@app.get("/api/configs/{name}")
def get_config(name: str):
    """Return the parsed YAML device config so the frontend can edit it."""
    safe_name = os.path.basename(name)
    path = os.path.join(CONFIGS_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Config '{safe_name}' not found")
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return {"status": "ok", "name": safe_name, "config": _coerce_numbers(cfg)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

with:

```python
@app.get("/api/configs/{name}")
def get_config(name: str):
    """Return the parsed YAML device config so the frontend can edit it.

    Searches shipped first, then user/. ``os.path.basename`` strips any
    leading path components in case a caller URL-encodes a slash.
    """
    safe_name = os.path.basename(name)
    candidates = [
        os.path.join(CONFIGS_DIR, safe_name),
        os.path.join(CONFIGS_DIR, "user", safe_name),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Config '{safe_name}' not found")
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return {"status": "ok", "name": safe_name, "config": _coerce_numbers(cfg)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 4: Run all backend tests — confirm new pass + check for regressions**

```bash
cd perovskite-sim && pytest tests/integration/backend/test_user_configs_api.py -v
cd perovskite-sim && pytest tests/unit -m 'not slow'
```

Expected: integration tests PASS; unit tests still pass (no behavior change to other modules).

- [ ] **Step 5: Commit**

```bash
cd perovskite-sim && git add backend/main.py tests/integration/backend/test_user_configs_api.py
git commit -m "feat(backend): /api/configs returns namespaced entries

Each config entry now has {name, namespace} where namespace is
'shipped' (top-level configs/) or 'user' (configs/user/). The
get_config endpoint searches both directories so the frontend can
load user presets via the same URL pattern. Breaking change for
the frontend api wrapper — updated in the next task."
```

---

## Task 5: `POST /api/configs/user` endpoint (TDD)

**Files:**
- Modify: `backend/main.py` — add POST endpoint
- Modify: `tests/integration/backend/test_user_configs_api.py` — add POST tests

- [ ] **Step 1: Add failing tests for POST**

Append to `perovskite-sim/tests/integration/backend/test_user_configs_api.py`:

```python
class TestPostUserConfig:
    def test_save_new_user_preset(self, clean_user_dir) -> None:
        body = {
            "name": "post_test_stack",
            "config": {"device": {"V_bi": 1.1}, "layers": []},
        }
        r = client.post("/api/configs/user", json=body)
        assert r.status_code == 200
        assert r.json()["saved"] == "post_test_stack"
        assert (USER_DIR / "post_test_stack.yaml").exists()

    def test_collision_with_shipped_returns_409(self, clean_user_dir) -> None:
        body = {"name": "nip_MAPbI3", "config": {"device": {"V_bi": 1.0}, "layers": []}}
        r = client.post("/api/configs/user", json=body)
        assert r.status_code == 409
        assert "reserved" in r.json()["detail"].lower() or "shipped" in r.json()["detail"].lower()

    def test_invalid_filename_returns_400(self, clean_user_dir) -> None:
        body = {"name": "../etc/passwd", "config": {}}
        r = client.post("/api/configs/user", json=body)
        assert r.status_code == 400

    def test_overwrite_protection(self, clean_user_dir) -> None:
        first = {"name": "dup_test", "config": {"device": {"V_bi": 1.0}, "layers": []}}
        client.post("/api/configs/user", json=first)
        # Second save without overwrite must 409.
        r = client.post("/api/configs/user", json=first)
        assert r.status_code == 409
        # With overwrite: 200.
        body = {**first, "overwrite": True}
        r2 = client.post("/api/configs/user", json=body)
        assert r2.status_code == 200

    def test_missing_name_returns_400(self, clean_user_dir) -> None:
        r = client.post("/api/configs/user", json={"config": {}})
        assert r.status_code == 400
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd perovskite-sim && pytest tests/integration/backend/test_user_configs_api.py::TestPostUserConfig -v
```

Expected: 405 / 404 — endpoint does not exist.

- [ ] **Step 3: Add the POST endpoint**

Add to `perovskite-sim/backend/main.py`. First, add this import at the top with the other backend imports (around line 21):

```python
from backend.user_configs import (
    is_shipped_name,
    validate_user_filename,
    write_user_config,
)
```

Then add the endpoint, right after the modified `get_config` function:

```python
class UserConfigPayload(BaseModel):
    name: str
    config: dict
    overwrite: bool = False


@app.post("/api/configs/user")
def save_user_config(payload: UserConfigPayload):
    """Write a user-edited DeviceConfig to ``configs/user/<name>.yaml``.

    The frontend Save-As dialog calls this. ``user_configs`` owns filename
    validation, shipped-name reservation, and atomic writes.
    """
    try:
        validate_user_filename(payload.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if is_shipped_name(payload.name):
        raise HTTPException(
            status_code=409,
            detail=f"{payload.name!r} is reserved by a shipped preset",
        )
    try:
        write_user_config(payload.name, payload.config, overwrite=payload.overwrite)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "saved": payload.name}
```

- [ ] **Step 4: Run integration tests to verify pass**

```bash
cd perovskite-sim && pytest tests/integration/backend/test_user_configs_api.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
cd perovskite-sim && git add backend/main.py tests/integration/backend/test_user_configs_api.py
git commit -m "feat(backend): add POST /api/configs/user save-as endpoint

Pydantic UserConfigPayload (name, config, overwrite). Returns
200/saved on success, 400 on invalid filename, 409 on shipped-name
collision or first-save existing collision (overwrite=true to
force). Delegates filesystem operations to backend.user_configs."
```

---

# Phase B — Frontend pure-logic modules (Vitest TDD, no DOM)

## Task 6: Extend `frontend/src/types.ts` with Phase 2b types

**Files:**
- Modify: `frontend/src/types.ts`

- [ ] **Step 1: Append the new types**

Append to `perovskite-sim/frontend/src/types.ts`:

```typescript
// ── Phase 2b layer builder ──────────────────────────────────────────────────

export type LayerRole =
  | 'substrate'
  | 'front_contact'
  | 'ETL'
  | 'absorber'
  | 'HTL'
  | 'back_contact'

export interface LayerTemplate {
  role: LayerRole
  optical_material: string | null
  description: string
  source: string
  defaults: Partial<LayerConfig>
}

export interface ValidationIssue {
  layerIdx: number | null   // null = stack-level issue
  field: string | null
  message: string
}

export interface ValidationReport {
  errors: ValidationIssue[]
  warnings: ValidationIssue[]
}

export type StackAction =
  | { type: 'select'; idx: number }
  | { type: 'delete'; idx: number }
  | { type: 'reorder'; from: number; to: number }
  | { type: 'insert'; atIdx: number; layer: LayerConfig }
  | { type: 'edit-interface'; idx: number; pair: readonly [number, number] }

export type Namespace = 'shipped' | 'user'

export interface ConfigEntry {
  name: string
  namespace: Namespace
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd perovskite-sim/frontend && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
cd perovskite-sim && git add frontend/src/types.ts
git commit -m "feat(frontend): add Phase 2b layer-builder types

LayerRole literal-union, LayerTemplate, ValidationIssue/Report,
StackAction discriminated union, Namespace/ConfigEntry. Locked
across all subsequent layer-builder tasks."
```

---

## Task 7: `frontend/src/workstation/tier-gating.ts` — `isLayerBuilderEnabled`

**Files:**
- Modify: `frontend/src/workstation/tier-gating.ts`
- Modify: `frontend/src/workstation/tier-gating.test.ts` (existing)

- [ ] **Step 1: Add the failing test**

Append to `perovskite-sim/frontend/src/workstation/tier-gating.test.ts`:

```typescript
import { isLayerBuilderEnabled } from './tier-gating'

describe('isLayerBuilderEnabled', () => {
  it('returns true only for full tier', () => {
    expect(isLayerBuilderEnabled('full')).toBe(true)
    expect(isLayerBuilderEnabled('fast')).toBe(false)
    expect(isLayerBuilderEnabled('legacy')).toBe(false)
  })
})
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd perovskite-sim/frontend && npx vitest run src/workstation/tier-gating.test.ts
```

Expected: FAIL — `isLayerBuilderEnabled` is not exported.

- [ ] **Step 3: Implement the helper**

Append to `perovskite-sim/frontend/src/workstation/tier-gating.ts`:

```typescript
/**
 * Phase 2b layer-builder gate. The custom-stack visualizer, add/remove/
 * reorder controls, template library, and Save-As path are full-tier-only
 * because adding/removing layers in legacy/fast tiers risks producing
 * configs that silently diverge from IonMonger / DriftFusion benchmark
 * conventions — exactly what those tiers exist to preserve.
 */
export function isLayerBuilderEnabled(tier: SimulationModeName): boolean {
  return tier === 'full'
}
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd perovskite-sim/frontend && npx vitest run src/workstation/tier-gating.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd perovskite-sim && git add frontend/src/workstation/tier-gating.ts frontend/src/workstation/tier-gating.test.ts
git commit -m "feat(frontend): add isLayerBuilderEnabled tier helper"
```

---

## Task 8: `frontend/src/stack/log-scale-height.ts` (TDD)

**Files:**
- Create: `frontend/src/stack/log-scale-height.ts`
- Create: `frontend/src/stack/__tests__/log-scale-height.test.ts`

- [ ] **Step 1: Write failing tests**

Write to `perovskite-sim/frontend/src/stack/__tests__/log-scale-height.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { logScaleHeight, MIN_PX, MAX_PX, MIN_M, MAX_M } from '../log-scale-height'

describe('logScaleHeight', () => {
  it('maps minimum thickness to MIN_PX', () => {
    expect(logScaleHeight(MIN_M)).toBe(MIN_PX)
  })

  it('maps maximum thickness to MAX_PX', () => {
    expect(logScaleHeight(MAX_M)).toBe(MAX_PX)
  })

  it('clamps thicknesses below MIN_M to MIN_PX', () => {
    expect(logScaleHeight(MIN_M / 100)).toBe(MIN_PX)
  })

  it('clamps thicknesses above MAX_M to MAX_PX', () => {
    expect(logScaleHeight(MAX_M * 100)).toBe(MAX_PX)
  })

  it('preserves ordering across the valid range', () => {
    const samples = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 4e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    const heights = samples.map(logScaleHeight)
    for (let i = 1; i < heights.length; i++) {
      expect(heights[i]).toBeGreaterThanOrEqual(heights[i - 1])
    }
  })

  it('returns finite, positive integers', () => {
    for (const t of [1e-9, 4e-7, 1e-3]) {
      const h = logScaleHeight(t)
      expect(Number.isFinite(h)).toBe(true)
      expect(h).toBeGreaterThan(0)
      expect(Number.isInteger(h)).toBe(true)
    }
  })

  it('returns MIN_PX for non-finite or non-positive input', () => {
    expect(logScaleHeight(0)).toBe(MIN_PX)
    expect(logScaleHeight(-1)).toBe(MIN_PX)
    expect(logScaleHeight(NaN)).toBe(MIN_PX)
    expect(logScaleHeight(Infinity)).toBe(MAX_PX)
  })
})
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd perovskite-sim/frontend && npx vitest run src/stack/__tests__/log-scale-height.test.ts
```

Expected: module not found.

- [ ] **Step 3: Implement the helper**

Write to `perovskite-sim/frontend/src/stack/log-scale-height.ts`:

```typescript
/**
 * Map a layer thickness in metres to a card height in pixels using a
 * log10 scale clamped to a fixed range. The full spectrum of perovskite
 * stacks spans 1 nm (interface) to 1 mm (glass substrate); a linear
 * mapping would make the substrate dwarf every other layer, so we
 * compress the range to keep every card visible while preserving order.
 */

export const MIN_M = 1e-9    // 1 nm
export const MAX_M = 1e-3    // 1 mm
export const MIN_PX = 18
export const MAX_PX = 96

const LOG_MIN = Math.log10(MIN_M)
const LOG_MAX = Math.log10(MAX_M)
const LOG_SPAN = LOG_MAX - LOG_MIN
const PX_SPAN = MAX_PX - MIN_PX

export function logScaleHeight(thicknessMetres: number): number {
  if (!Number.isFinite(thicknessMetres)) {
    return thicknessMetres === Infinity ? MAX_PX : MIN_PX
  }
  if (thicknessMetres <= 0) return MIN_PX
  const t = Math.max(MIN_M, Math.min(MAX_M, thicknessMetres))
  const frac = (Math.log10(t) - LOG_MIN) / LOG_SPAN
  return Math.round(MIN_PX + frac * PX_SPAN)
}
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd perovskite-sim/frontend && npx vitest run src/stack/__tests__/log-scale-height.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd perovskite-sim && git add frontend/src/stack/log-scale-height.ts frontend/src/stack/__tests__/log-scale-height.test.ts
git commit -m "feat(frontend): add logScaleHeight helper for stack visualizer"
```

---

## Task 9: `frontend/src/stack/dirty-state.ts` (TDD)

**Files:**
- Create: `frontend/src/stack/dirty-state.ts`
- Create: `frontend/src/stack/__tests__/dirty-state.test.ts`

- [ ] **Step 1: Write failing tests**

Write to `perovskite-sim/frontend/src/stack/__tests__/dirty-state.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { isDirty } from '../dirty-state'
import type { DeviceConfig } from '../../types'

const baseLayer = {
  name: 'MAPbI3',
  role: 'absorber',
  thickness: 4e-7,
  eps_r: 24.1,
  mu_n: 2e-4, mu_p: 2e-4,
  ni: 1e12, N_D: 0, N_A: 0,
  D_ion: 1e-16, P_lim: 1.6e25, P0: 1.6e25,
  tau_n: 3e-9, tau_p: 3e-9,
  n1: 1e10, p1: 1e10,
  B_rad: 9e-17, C_n: 0, C_p: 0,
  alpha: 0,
} as const

const baseConfig: DeviceConfig = {
  device: { V_bi: 1.1, Phi: 1.4e21, T: 300, mode: 'full', interfaces: [] },
  layers: [{ ...baseLayer }],
}

describe('isDirty', () => {
  it('is false when configs are deeply equal', () => {
    const a = baseConfig
    const b = JSON.parse(JSON.stringify(baseConfig))
    expect(isDirty(a, b)).toBe(false)
  })

  it('is true when a numeric field differs', () => {
    const next = { ...baseConfig, layers: [{ ...baseLayer, thickness: 5e-7 }] }
    expect(isDirty(baseConfig, next)).toBe(true)
  })

  it('is true when a layer is added', () => {
    const next = { ...baseConfig, layers: [...baseConfig.layers, { ...baseLayer, name: 'extra' }] }
    expect(isDirty(baseConfig, next)).toBe(true)
  })

  it('is true when interfaces change', () => {
    const next = { ...baseConfig, device: { ...baseConfig.device, interfaces: [[1, 2]] as Array<[number, number]> } }
    expect(isDirty(baseConfig, next)).toBe(true)
  })

  it('is false regardless of property insertion order', () => {
    const reordered: DeviceConfig = {
      layers: baseConfig.layers,
      device: baseConfig.device,
    }
    expect(isDirty(baseConfig, reordered)).toBe(false)
  })
})
```

- [ ] **Step 2: Run to verify failure**

```bash
cd perovskite-sim/frontend && npx vitest run src/stack/__tests__/dirty-state.test.ts
```

Expected: module not found.

- [ ] **Step 3: Implement**

Write to `perovskite-sim/frontend/src/stack/dirty-state.ts`:

```typescript
import type { DeviceConfig } from '../types'

/**
 * Return true if `current` differs from `loaded`, comparing the two
 * device configs by deep value equality. The implementation canonicalises
 * each side via JSON.stringify with a sorted-key replacer so insertion
 * order does not produce false positives (the device-pane state spreads
 * objects, which can change key order).
 */
export function isDirty(loaded: DeviceConfig, current: DeviceConfig): boolean {
  return canonical(loaded) !== canonical(current)
}

function canonical(value: unknown): string {
  return JSON.stringify(value, (_key, v) => {
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      const sorted: Record<string, unknown> = {}
      for (const k of Object.keys(v as Record<string, unknown>).sort()) {
        sorted[k] = (v as Record<string, unknown>)[k]
      }
      return sorted
    }
    return v
  })
}
```

- [ ] **Step 4: Run to verify pass**

```bash
cd perovskite-sim/frontend && npx vitest run src/stack/__tests__/dirty-state.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd perovskite-sim && git add frontend/src/stack/dirty-state.ts frontend/src/stack/__tests__/dirty-state.test.ts
git commit -m "feat(frontend): add isDirty helper (key-order independent)"
```

---

## Task 10: `frontend/src/stack/reconcile-interfaces.ts` (TDD)

**Files:**
- Create: `frontend/src/stack/reconcile-interfaces.ts`
- Create: `frontend/src/stack/__tests__/reconcile-interfaces.test.ts`

- [ ] **Step 1: Write failing tests**

Write to `perovskite-sim/frontend/src/stack/__tests__/reconcile-interfaces.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { reconcileInterfaces } from '../reconcile-interfaces'
import type { LayerConfig } from '../../types'

const L = (name: string): LayerConfig => ({
  name, role: 'absorber', thickness: 1e-7, eps_r: 1, mu_n: 0, mu_p: 0,
  ni: 0, N_D: 0, N_A: 0, D_ion: 0, P_lim: 0, P0: 0, tau_n: 0, tau_p: 0,
  n1: 0, p1: 0, B_rad: 0, C_n: 0, C_p: 0, alpha: 0,
})

describe('reconcileInterfaces', () => {
  it('returns empty when only one layer remains', () => {
    expect(reconcileInterfaces([L('a'), L('b')], [L('a')], [[1, 2]])).toEqual([])
  })

  it('preserves surviving adjacent pairs after insert', () => {
    const old = [L('a'), L('b'), L('c')]
    const oldIfaces: Array<[number, number]> = [[1, 2], [3, 4]]
    const next = [L('a'), L('b'), L('x'), L('c')]
    const result = reconcileInterfaces(old, next, oldIfaces)
    expect(result).toEqual([[1, 2], [0, 0], [0, 0]])
  })

  it('drops the interface adjacent to a deleted middle layer', () => {
    const old = [L('a'), L('b'), L('c')]
    const oldIfaces: Array<[number, number]> = [[1, 2], [3, 4]]
    const next = [L('a'), L('c')]
    const result = reconcileInterfaces(old, next, oldIfaces)
    expect(result).toEqual([[0, 0]])
  })

  it('keeps remaining pair when first layer is deleted', () => {
    const old = [L('a'), L('b'), L('c')]
    const oldIfaces: Array<[number, number]> = [[1, 2], [3, 4]]
    const next = [L('b'), L('c')]
    const result = reconcileInterfaces(old, next, oldIfaces)
    expect(result).toEqual([[3, 4]])
  })

  it('preserves surviving adjacent pairs after a reorder', () => {
    const old = [L('a'), L('b'), L('c'), L('d')]
    const oldIfaces: Array<[number, number]> = [[1, 2], [3, 4], [5, 6]]
    // swap b and c → a, c, b, d
    const next = [L('a'), L('c'), L('b'), L('d')]
    const result = reconcileInterfaces(old, next, oldIfaces)
    // (a,c), (c,b)=(b,c) reverse not preserved, (b,d) new → all defaults
    expect(result.length).toBe(3)
  })

  it('always preserves length invariant', () => {
    for (const n of [1, 2, 3, 6, 10]) {
      const layers = Array.from({ length: n }, (_, i) => L(`L${i}`))
      const result = reconcileInterfaces([], layers, [])
      expect(result.length).toBe(Math.max(0, n - 1))
    }
  })
})
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd perovskite-sim/frontend && npx vitest run src/stack/__tests__/reconcile-interfaces.test.ts
```

Expected: module not found.

- [ ] **Step 3: Implement**

Write to `perovskite-sim/frontend/src/stack/reconcile-interfaces.ts`:

```typescript
import type { LayerConfig } from '../types'

type Pair = readonly [number, number]

/**
 * Reconcile the interfaces array after a layer mutation.
 *
 * Invariant: ``result.length === Math.max(0, newLayers.length - 1)``.
 *
 * Surviving adjacent pairs (same `(left.name, right.name)` in old AND new)
 * keep their values. New pairs default to `[0, 0]`. Order matches the new
 * layer order.
 */
export function reconcileInterfaces(
  oldLayers: ReadonlyArray<LayerConfig>,
  newLayers: ReadonlyArray<LayerConfig>,
  oldInterfaces: ReadonlyArray<Pair>,
): Array<[number, number]> {
  const map = new Map<string, Pair>()
  for (let i = 0; i < oldInterfaces.length; i++) {
    const left = oldLayers[i]?.name
    const right = oldLayers[i + 1]?.name
    if (left != null && right != null) {
      map.set(`${left}\u0000${right}`, oldInterfaces[i])
    }
  }
  const result: Array<[number, number]> = []
  for (let i = 0; i < newLayers.length - 1; i++) {
    const key = `${newLayers[i].name}\u0000${newLayers[i + 1].name}`
    const pair = map.get(key)
    result.push(pair != null ? [pair[0], pair[1]] : [0, 0])
  }
  return result
}
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd perovskite-sim/frontend && npx vitest run src/stack/__tests__/reconcile-interfaces.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd perovskite-sim && git add frontend/src/stack/reconcile-interfaces.ts frontend/src/stack/__tests__/reconcile-interfaces.test.ts
git commit -m "feat(frontend): add reconcileInterfaces invariant helper"
```

---

## Task 11: `frontend/src/stack/stack-validator.ts` (TDD)

**Files:**
- Create: `frontend/src/stack/stack-validator.ts`
- Create: `frontend/src/stack/__tests__/stack-validator.test.ts`

- [ ] **Step 1: Write failing tests**

Write to `perovskite-sim/frontend/src/stack/__tests__/stack-validator.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { validate } from '../stack-validator'
import type { DeviceConfig, LayerConfig } from '../../types'

const layer = (overrides: Partial<LayerConfig>): LayerConfig => ({
  name: 'L', role: 'absorber', thickness: 1e-7, eps_r: 1,
  mu_n: 0, mu_p: 0, ni: 0, N_D: 0, N_A: 0,
  D_ion: 0, P_lim: 0, P0: 0,
  tau_n: 0, tau_p: 0, n1: 0, p1: 0,
  B_rad: 0, C_n: 0, C_p: 0, alpha: 0,
  ...overrides,
})

const cfg = (layers: LayerConfig[]): DeviceConfig => ({
  device: { V_bi: 1, Phi: 1, mode: 'full', interfaces: [] },
  layers,
})

describe('validate', () => {
  it('passes a valid n-i-p stack', () => {
    const c = cfg([
      layer({ name: 'TiO2', role: 'ETL' }),
      layer({ name: 'MAPbI3', role: 'absorber' }),
      layer({ name: 'spiro', role: 'HTL' }),
    ])
    const r = validate(c)
    expect(r.errors).toEqual([])
  })

  it('passes a valid p-i-n stack (orientation symmetric)', () => {
    const c = cfg([
      layer({ name: 'PEDOT', role: 'HTL' }),
      layer({ name: 'MAPbI3', role: 'absorber' }),
      layer({ name: 'C60', role: 'ETL' }),
    ])
    expect(validate(c).errors).toEqual([])
  })

  it('errors when there is no absorber', () => {
    const c = cfg([
      layer({ name: 'TiO2', role: 'ETL' }),
      layer({ name: 'spiro', role: 'HTL' }),
    ])
    const r = validate(c)
    expect(r.errors.some(e => e.message.includes('absorber'))).toBe(true)
  })

  it('errors when there are two absorbers', () => {
    const c = cfg([
      layer({ name: 'A', role: 'absorber' }),
      layer({ name: 'B', role: 'absorber' }),
    ])
    expect(validate(c).errors.some(e => e.message.includes('absorber'))).toBe(true)
  })

  it('errors on duplicate layer names', () => {
    const c = cfg([
      layer({ name: 'X', role: 'ETL' }),
      layer({ name: 'X', role: 'absorber' }),
    ])
    const r = validate(c)
    expect(r.errors.some(e => e.message.includes('Duplicate'))).toBe(true)
  })

  it('errors when a thickness is zero', () => {
    const c = cfg([
      layer({ name: 'A', role: 'ETL', thickness: 0 }),
      layer({ name: 'B', role: 'absorber' }),
    ])
    expect(validate(c).errors.some(e => e.message.includes('positive'))).toBe(true)
  })

  it('errors when more than one substrate is present', () => {
    const c = cfg([
      layer({ name: 'g1', role: 'substrate', incoherent: true, optical_material: 'glass' }),
      layer({ name: 'g2', role: 'substrate', incoherent: true, optical_material: 'glass' }),
      layer({ name: 'A', role: 'absorber' }),
    ])
    expect(validate(c).errors.some(e => e.message.includes('At most one substrate'))).toBe(true)
  })

  it('errors when the substrate is not the first layer', () => {
    const c = cfg([
      layer({ name: 'A', role: 'absorber' }),
      layer({ name: 'g', role: 'substrate', incoherent: true, optical_material: 'glass' }),
    ])
    expect(validate(c).errors.some(e => e.message.includes('first'))).toBe(true)
  })

  it('errors when the substrate is not incoherent', () => {
    const c = cfg([
      layer({ name: 'g', role: 'substrate', incoherent: false, optical_material: 'glass' }),
      layer({ name: 'A', role: 'absorber' }),
    ])
    expect(validate(c).errors.some(e => e.message.includes('incoherent'))).toBe(true)
  })

  it('errors when the substrate has no optical material', () => {
    const c = cfg([
      layer({ name: 'g', role: 'substrate', incoherent: true, optical_material: null }),
      layer({ name: 'A', role: 'absorber' }),
    ])
    expect(validate(c).errors.some(e => e.message.includes('optical material'))).toBe(true)
  })

  it('warns when an interface row is the (0,0) default — surfaced by callers', () => {
    // The validator does not get the interfaces array — the warning is emitted
    // by the visualizer at render time from the device.interfaces values.
    // Here we just confirm validate() does not produce false errors when the
    // stack is otherwise valid and all interfaces are zero.
    const c: DeviceConfig = {
      device: { V_bi: 1, Phi: 1, mode: 'full', interfaces: [[0, 0]] },
      layers: [
        layer({ name: 'A', role: 'ETL' }),
        layer({ name: 'B', role: 'absorber' }),
      ],
    }
    expect(validate(c).errors).toEqual([])
  })

  it('warns on mixed TMM / Beer-Lambert layers', () => {
    const c = cfg([
      layer({ name: 'A', role: 'ETL', optical_material: 'TiO2' }),
      layer({ name: 'B', role: 'absorber', optical_material: null }),
    ])
    const r = validate(c)
    expect(r.warnings.some(w => w.message.includes('Mixed'))).toBe(true)
  })
})
```

- [ ] **Step 2: Run to verify failure**

```bash
cd perovskite-sim/frontend && npx vitest run src/stack/__tests__/stack-validator.test.ts
```

Expected: module not found.

- [ ] **Step 3: Implement the validator**

Write to `perovskite-sim/frontend/src/stack/stack-validator.ts`:

```typescript
import type {
  DeviceConfig,
  ValidationIssue,
  ValidationReport,
} from '../types'

/**
 * Pure validator for the layer builder. Hard-blocks structural rules that
 * the UI owns; soft-warns on rules whose violation is recoverable. Backend
 * physical-sanity validation (config_loader.py) handles everything that
 * needs real semiconductor knowledge.
 */
export function validate(config: DeviceConfig): ValidationReport {
  const errors: ValidationIssue[] = []
  const warnings: ValidationIssue[] = []
  const layers = config.layers

  // 1. Exactly one absorber.
  const absorberIdxs = layers
    .map((l, i) => (l.role === 'absorber' ? i : -1))
    .filter(i => i >= 0)
  if (absorberIdxs.length !== 1) {
    errors.push({
      layerIdx: null,
      field: null,
      message: `Stack needs exactly one absorber layer (found ${absorberIdxs.length})`,
    })
  }

  // 2. Unique names.
  const nameCounts = new Map<string, number[]>()
  layers.forEach((l, i) => {
    const list = nameCounts.get(l.name) ?? []
    list.push(i)
    nameCounts.set(l.name, list)
  })
  for (const [name, idxs] of nameCounts) {
    if (idxs.length > 1) {
      for (const i of idxs) {
        errors.push({
          layerIdx: i,
          field: 'name',
          message: `Duplicate layer name "${name}"`,
        })
      }
    }
  }

  // 3. Positive thickness.
  layers.forEach((l, i) => {
    if (!(typeof l.thickness === 'number' && l.thickness > 0)) {
      errors.push({
        layerIdx: i,
        field: 'thickness',
        message: 'Thickness must be positive',
      })
    }
  })

  // 4. At most one substrate.
  const substrateIdxs = layers
    .map((l, i) => (l.role === 'substrate' ? i : -1))
    .filter(i => i >= 0)
  if (substrateIdxs.length > 1) {
    for (const i of substrateIdxs) {
      errors.push({
        layerIdx: i,
        field: 'role',
        message: 'At most one substrate layer is allowed',
      })
    }
  }

  // 5. Substrate constraints.
  if (substrateIdxs.length === 1) {
    const i = substrateIdxs[0]
    const sub = layers[i]
    if (i !== 0) {
      errors.push({
        layerIdx: i,
        field: 'role',
        message: 'Substrate must be the first layer',
      })
    }
    if (!sub.incoherent) {
      errors.push({
        layerIdx: i,
        field: 'incoherent',
        message: 'Substrate must be marked incoherent',
      })
    }
    if (sub.optical_material == null || sub.optical_material === '') {
      errors.push({
        layerIdx: i,
        field: 'optical_material',
        message: 'Substrate must have an optical material',
      })
    }
  }

  // ── Warnings ────────────────────────────────────────────────────────────

  const tmmCount = layers.filter(
    l => l.optical_material != null && l.optical_material !== '',
  ).length
  if (tmmCount > 0 && tmmCount < layers.length) {
    warnings.push({
      layerIdx: null,
      field: 'optical_material',
      message: 'Mixed TMM / Beer-Lambert layers — TMM-less layers fall back per Phase 2a',
    })
  }
  if (tmmCount === 0) {
    warnings.push({
      layerIdx: null,
      field: 'optical_material',
      message: 'TMM is dormant — set optical_material to enable',
    })
  }

  return { errors, warnings }
}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd perovskite-sim/frontend && npx vitest run src/stack/__tests__/stack-validator.test.ts
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
cd perovskite-sim && git add frontend/src/stack/stack-validator.ts frontend/src/stack/__tests__/stack-validator.test.ts
git commit -m "feat(frontend): add stack-validator with structural rules

Hard-blocks: exactly-one-absorber, unique names, positive
thickness, single substrate with first/incoherent/optical
constraints. Soft-warns: mixed TMM/Beer-Lambert, dormant TMM."
```

---

## Task 12: Extend `frontend/src/api.ts` for Phase 2b endpoints

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/device-panel.ts` — adapt to new `listConfigs` shape
- Modify: any caller of `listConfigs` (only `device-panel.ts` today)

- [ ] **Step 1: Update `listConfigs` to return `ConfigEntry[]` and add new wrappers**

Edit `perovskite-sim/frontend/src/api.ts`. Replace the `listConfigs` function and add new wrappers. The full new section:

```typescript
import type {
  DeviceConfig,
  JVResult,
  ISResult,
  DegResult,
  JVParams,
  ISParams,
  DegParams,
  ConfigEntry,
  LayerTemplate,
} from './types'

const BASE = 'http://127.0.0.1:8000'

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${await res.text()}`)
  }
  const data = await res.json()
  return data.result as T
}

export async function listConfigs(): Promise<ConfigEntry[]> {
  const res = await fetch(`${BASE}/api/configs`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const data = await res.json()
  return data.configs as ConfigEntry[]
}

export async function getConfig(name: string): Promise<DeviceConfig> {
  const res = await fetch(`${BASE}/api/configs/${encodeURIComponent(name)}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const data = await res.json()
  return data.config as DeviceConfig
}

export async function fetchOpticalMaterials(): Promise<string[]> {
  const res = await fetch(`${BASE}/api/optical-materials`)
  if (!res.ok) throw new Error(`fetchOpticalMaterials failed: ${res.status}`)
  const data = (await res.json()) as { materials: string[] }
  return data.materials
}

export async function fetchLayerTemplates(): Promise<Record<string, LayerTemplate>> {
  const res = await fetch(`${BASE}/api/layer-templates`)
  if (!res.ok) throw new Error(`fetchLayerTemplates failed: ${res.status}`)
  const data = (await res.json()) as { templates: Record<string, LayerTemplate> }
  return data.templates
}

export interface SaveUserConfigResult {
  ok: true
  saved: string
}

export interface SaveUserConfigError {
  ok: false
  status: number
  detail: string
}

export type SaveUserConfigResponse = SaveUserConfigResult | SaveUserConfigError

export async function saveUserConfig(
  name: string,
  config: DeviceConfig,
  overwrite = false,
): Promise<SaveUserConfigResponse> {
  const res = await fetch(`${BASE}/api/configs/user`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, config, overwrite }),
  })
  if (res.ok) {
    const data = await res.json()
    return { ok: true, saved: data.saved as string }
  }
  let detail = `HTTP ${res.status}`
  try {
    const data = await res.json()
    if (data?.detail) detail = String(data.detail)
  } catch { /* non-JSON body */ }
  return { ok: false, status: res.status, detail }
}

export async function checkUserConfigExists(
  name: string,
): Promise<{ exists: boolean; namespace: 'shipped' | 'user' | null }> {
  const entries = await listConfigs()
  const match = entries.find(
    e => e.name === `${name}.yaml` || e.name === `${name}.yml`,
  )
  if (!match) return { exists: false, namespace: null }
  return { exists: true, namespace: match.namespace }
}

export async function runJV(device: DeviceConfig, params: JVParams): Promise<JVResult> {
  const res = await fetch(`${BASE}/api/jv`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ device, ...params }),
  })
  return handle<JVResult>(res)
}

export async function runImpedance(device: DeviceConfig, params: ISParams): Promise<ISResult> {
  const res = await fetch(`${BASE}/api/impedance`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ device, ...params }),
  })
  return handle<ISResult>(res)
}

export async function runDegradation(device: DeviceConfig, params: DegParams): Promise<DegResult> {
  const res = await fetch(`${BASE}/api/degradation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ device, ...params }),
  })
  return handle<DegResult>(res)
}
```

- [ ] **Step 2: Update `device-panel.ts` to consume `ConfigEntry[]`**

Replace the dropdown population block in `perovskite-sim/frontend/src/device-panel.ts`:

```typescript
  const entries = await listConfigs()
  // Render dropdown as two optgroups: shipped first, then user.
  const shipped = entries.filter(e => e.namespace === 'shipped')
  const user = entries.filter(e => e.namespace === 'user')
  const shippedOpts = shipped
    .map(e => `<option value="${e.name}">${e.name.replace(/\.ya?ml$/, '')}</option>`)
    .join('')
  const userOpts = user
    .map(e => `<option value="user/${e.name}">${e.name.replace(/\.ya?ml$/, '')}</option>`)
    .join('')
  select.innerHTML = [
    shippedOpts ? `<optgroup label="Shipped presets">${shippedOpts}</optgroup>` : '',
    userOpts ? `<optgroup label="User presets">${userOpts}</optgroup>` : '',
  ].join('')
```

Replace the "Prefer ionmonger" line with:

```typescript
  const initial =
    shipped.find(e => e.name.includes('ionmonger'))?.name ??
    shipped[0]?.name ??
    entries[0]?.name
  if (!initial) throw new Error('no configs available')
  select.value = initial
  await load(initial)
```

The existing `load(name)` already calls `getConfig(name)` which the backend (Task 4) updates to search both shipped and user roots. The `user/` prefix in the dropdown value is informational — strip it before passing to `getConfig`:

Find the `select.addEventListener('change', ...)` line and replace with:

```typescript
  select.addEventListener('change', () => {
    const v = select.value.startsWith('user/') ? select.value.slice(5) : select.value
    void load(v)
  })
```

Also strip in the initial assignment:

```typescript
  const initialName = initial.startsWith('user/') ? initial.slice(5) : initial
  select.value = initial   // keep optgroup-prefixed value in the <select>
  await load(initialName)
```

(Replace the prior `select.value = initial; await load(initial)` block with the above three-line version.)

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd perovskite-sim/frontend && npx tsc --noEmit
```

Expected: clean.

- [ ] **Step 4: Run all existing frontend tests**

```bash
cd perovskite-sim/frontend && npx vitest run
```

Expected: all green (the pre-existing tests don't depend on `listConfigs` shape).

- [ ] **Step 5: Manual smoke test**

```bash
cd perovskite-sim && uvicorn backend.main:app --host 127.0.0.1 --port 8000 --app-dir perovskite-sim --reload &
sleep 2
curl -s http://127.0.0.1:8000/api/configs | python -m json.tool
curl -s http://127.0.0.1:8000/api/layer-templates | python -m json.tool | head -30
kill %1
```

Expected: `/api/configs` returns `{configs: [{name, namespace}, ...]}`; `/api/layer-templates` returns the parsed library.

- [ ] **Step 6: Commit**

```bash
cd perovskite-sim && git add frontend/src/api.ts frontend/src/device-panel.ts
git commit -m "feat(frontend): wire api wrappers to Phase 2b endpoints

listConfigs returns ConfigEntry[] (breaking change matched by
device-panel.ts dropdown rendering as two optgroups).
fetchLayerTemplates, saveUserConfig, checkUserConfigExists added."
```

---

# Phase C — Frontend visualizer rendering

## Task 13: `frontend/src/stack/stack-layer-card.ts` — pure render

**Files:**
- Create: `frontend/src/stack/stack-layer-card.ts`

- [ ] **Step 1: Implement the layer card renderer**

Write to `perovskite-sim/frontend/src/stack/stack-layer-card.ts`:

```typescript
import type { LayerConfig } from '../types'
import { logScaleHeight } from './log-scale-height'

const HTML_ESCAPE: Record<string, string> = {
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}
function esc(s: string): string {
  return s.replace(/[&<>"']/g, c => HTML_ESCAPE[c])
}

function fmtThickness(metres: number): string {
  if (!Number.isFinite(metres) || metres <= 0) return '—'
  if (metres >= 1e-3) return `${(metres * 1000).toPrecision(3)} mm`
  if (metres >= 1e-6) return `${(metres * 1e6).toPrecision(3)} µm`
  return `${(metres * 1e9).toPrecision(3)} nm`
}

/**
 * Render one layer card as an HTML string. The visualizer composes these
 * with the interface strips. Selection / hover / drag affordances are
 * styled via CSS classes and `data-*` attributes; event wiring is done
 * by the visualizer at delegation time.
 */
export function renderLayerCard(
  layer: LayerConfig,
  idx: number,
  selected: boolean,
  errorFields: ReadonlySet<string>,
): string {
  const role = layer.role || 'absorber'
  const heightPx = logScaleHeight(layer.thickness)
  const selectedCls = selected ? ' is-selected' : ''
  const errorCls = errorFields.size > 0 ? ' is-error' : ''
  const opticalPill = layer.optical_material
    ? `<span class="layer-card-pill" title="optical_material: ${esc(layer.optical_material)}">${esc(layer.optical_material)}</span>`
    : ''
  return `
    <div class="layer-card layer-card-${esc(role)}${selectedCls}${errorCls}"
         data-idx="${idx}"
         draggable="true"
         style="min-height:${heightPx}px"
         role="button"
         tabindex="0"
         aria-selected="${selected}"
         aria-label="Layer ${idx + 1}: ${esc(layer.name)} (${esc(role)})">
      <span class="layer-card-handle" aria-hidden="true">⋮⋮</span>
      <div class="layer-card-body">
        <div class="layer-card-name">${esc(layer.name)}</div>
        <div class="layer-card-meta">${fmtThickness(layer.thickness)} · ${esc(role)}${layer.incoherent ? ' · incoherent' : ''}</div>
      </div>
      ${opticalPill}
      <div class="layer-card-controls">
        <button class="layer-card-up" data-action="up" data-idx="${idx}" aria-label="Move up">↑</button>
        <button class="layer-card-down" data-action="down" data-idx="${idx}" aria-label="Move down">↓</button>
        <button class="layer-card-delete" data-action="delete" data-idx="${idx}" aria-label="Delete layer">✕</button>
      </div>
    </div>`
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd perovskite-sim/frontend && npx tsc --noEmit
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
cd perovskite-sim && git add frontend/src/stack/stack-layer-card.ts
git commit -m "feat(frontend): add stack-layer-card pure renderer"
```

---

## Task 14: `frontend/src/stack/stack-interface-strip.ts`

**Files:**
- Create: `frontend/src/stack/stack-interface-strip.ts`

- [ ] **Step 1: Implement the interface strip renderer**

Write to `perovskite-sim/frontend/src/stack/stack-interface-strip.ts`:

```typescript
type Pair = readonly [number, number]

function fmtSrv(v: number): string {
  if (!Number.isFinite(v) || v === 0) return '0'
  if (v >= 1) return v.toExponential(1)
  return v.toExponential(1)
}

/**
 * Render the inline interface row that sits between two adjacent layer
 * cards in the visualizer. Click events are delegated to the visualizer
 * via the data-action attribute.
 */
export function renderInterfaceStrip(
  ifaceIdx: number,
  pair: Pair,
  isDefaultZero: boolean,
): string {
  const cls = isDefaultZero ? 'iface-strip is-default' : 'iface-strip'
  const label = isDefaultZero
    ? '◆ uses default 0 m/s'
    : `◆ v_n=${fmtSrv(pair[0])} v_p=${fmtSrv(pair[1])} m/s`
  return `
    <button class="${cls}"
            data-action="edit-iface"
            data-iface-idx="${ifaceIdx}"
            type="button"
            aria-label="Edit interface ${ifaceIdx + 1}">
      ${label}
    </button>`
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd perovskite-sim/frontend && npx tsc --noEmit
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
cd perovskite-sim && git add frontend/src/stack/stack-interface-strip.ts
git commit -m "feat(frontend): add stack-interface-strip renderer"
```

---

## Task 15: `frontend/src/stack/stack-visualizer.ts` — composition + DnD wiring

**Files:**
- Create: `frontend/src/stack/stack-visualizer.ts`

- [ ] **Step 1: Implement the visualizer**

Write to `perovskite-sim/frontend/src/stack/stack-visualizer.ts`:

```typescript
import type { DeviceConfig, StackAction, ValidationReport } from '../types'
import { renderLayerCard } from './stack-layer-card'
import { renderInterfaceStrip } from './stack-interface-strip'
import { reconcileInterfaces } from './reconcile-interfaces'

const HTML_ESCAPE: Record<string, string> = {
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}
function esc(s: string): string {
  return s.replace(/[&<>"']/g, c => HTML_ESCAPE[c])
}

export interface StackVisualizerHandle {
  render(
    config: DeviceConfig,
    selectedIdx: number,
    report: ValidationReport,
  ): void
}

/**
 * Mount the stack visualizer into `container`. Pure render: state lives
 * in the parent (device-pane). All user interactions bubble out through
 * `onAction` so the parent owns the immutable state updates.
 *
 * The returned handle's `render` method is idempotent and always rebuilds
 * the inner HTML — this keeps the implementation small and the DOM in
 * sync with parent state without diffing.
 */
export function mountStackVisualizer(
  container: HTMLElement,
  onAction: (action: StackAction) => void,
): StackVisualizerHandle {
  container.classList.add('stack-visualizer')
  let dragSrcIdx: number | null = null

  function handleClick(ev: MouseEvent): void {
    const target = ev.target as HTMLElement
    const actionEl = target.closest<HTMLElement>('[data-action]')
    if (actionEl) {
      const action = actionEl.dataset.action!
      const idxStr = actionEl.dataset.idx
      const ifaceIdxStr = actionEl.dataset.ifaceIdx
      ev.stopPropagation()
      if (action === 'delete' && idxStr) {
        onAction({ type: 'delete', idx: Number(idxStr) })
        return
      }
      if (action === 'up' && idxStr) {
        const i = Number(idxStr)
        if (i > 0) onAction({ type: 'reorder', from: i, to: i - 1 })
        return
      }
      if (action === 'down' && idxStr) {
        const i = Number(idxStr)
        onAction({ type: 'reorder', from: i, to: i + 1 })
        return
      }
      if (action === 'insert' && idxStr) {
        // Insert intent only — parent opens the Add Layer dialog and
        // emits the resolved action separately. We piggyback on the
        // 'select' action here with a sentinel idx is wrong; instead
        // emit a synthetic select to pass the gap index up via a
        // custom event on the container.
        container.dispatchEvent(
          new CustomEvent('stack-insert-request', { detail: { atIdx: Number(idxStr) } }),
        )
        return
      }
      if (action === 'edit-iface' && ifaceIdxStr) {
        container.dispatchEvent(
          new CustomEvent('stack-edit-iface', { detail: { ifaceIdx: Number(ifaceIdxStr) } }),
        )
        return
      }
    }
    // Click on a layer card body → select.
    const card = target.closest<HTMLElement>('.layer-card')
    if (card?.dataset.idx) {
      onAction({ type: 'select', idx: Number(card.dataset.idx) })
    }
  }

  function handleDragStart(ev: DragEvent): void {
    const card = (ev.target as HTMLElement).closest<HTMLElement>('.layer-card')
    if (!card?.dataset.idx) return
    dragSrcIdx = Number(card.dataset.idx)
    ev.dataTransfer?.setData('text/plain', String(dragSrcIdx))
    if (ev.dataTransfer) ev.dataTransfer.effectAllowed = 'move'
  }

  function handleDragOver(ev: DragEvent): void {
    ev.preventDefault()
    if (ev.dataTransfer) ev.dataTransfer.dropEffect = 'move'
  }

  function handleDrop(ev: DragEvent): void {
    ev.preventDefault()
    const card = (ev.target as HTMLElement).closest<HTMLElement>('.layer-card')
    if (card?.dataset.idx == null || dragSrcIdx == null) return
    const toIdx = Number(card.dataset.idx)
    if (toIdx !== dragSrcIdx) {
      onAction({ type: 'reorder', from: dragSrcIdx, to: toIdx })
    }
    dragSrcIdx = null
  }

  container.addEventListener('click', handleClick)
  container.addEventListener('dragstart', handleDragStart)
  container.addEventListener('dragover', handleDragOver)
  container.addEventListener('drop', handleDrop)

  return {
    render(config, selectedIdx, report) {
      const errorFieldsByLayer = new Map<number, Set<string>>()
      for (const e of report.errors) {
        if (e.layerIdx == null) continue
        const set = errorFieldsByLayer.get(e.layerIdx) ?? new Set<string>()
        if (e.field) set.add(e.field)
        errorFieldsByLayer.set(e.layerIdx, set)
      }

      const layers = config.layers
      const interfaces = config.device.interfaces ?? []

      const parts: string[] = []
      parts.push(
        '<div class="stack-visualizer-sun" aria-hidden="true">',
        '  <div class="stack-visualizer-sun-label">☀ AM1.5G</div>',
        '  <div class="stack-visualizer-sun-rays">↓ ↓ ↓ ↓ ↓</div>',
        '</div>',
        '<div class="stack-visualizer-frame">',
      )

      // Inter-layer "+" gap above the first layer.
      parts.push(
        `<div class="stack-insert-gap"><button class="stack-insert-btn" data-action="insert" data-idx="0" aria-label="Insert layer at top">+</button></div>`,
      )

      for (let i = 0; i < layers.length; i++) {
        const errs = errorFieldsByLayer.get(i) ?? new Set<string>()
        parts.push(renderLayerCard(layers[i], i, i === selectedIdx, errs))
        if (i < layers.length - 1) {
          const pair = (interfaces[i] ?? [0, 0]) as readonly [number, number]
          const isDefault = pair[0] === 0 && pair[1] === 0
          parts.push(renderInterfaceStrip(i, pair, isDefault))
        }
      }

      // Inter-layer "+" gap below the last layer.
      parts.push(
        `<div class="stack-insert-gap"><button class="stack-insert-btn" data-action="insert" data-idx="${layers.length}" aria-label="Insert layer at bottom">+</button></div>`,
      )

      parts.push('</div>')

      // Validation banner (first error only — surface space is tight).
      if (report.errors.length > 0) {
        parts.push(
          `<div class="stack-error-banner" role="alert">${esc(report.errors[0].message)}</div>`,
        )
      }

      // Legend.
      parts.push(
        '<div class="stack-legend">',
        '  <span class="legend-chip legend-substrate">substrate</span>',
        '  <span class="legend-chip legend-front_contact">front contact</span>',
        '  <span class="legend-chip legend-ETL">ETL</span>',
        '  <span class="legend-chip legend-absorber">absorber</span>',
        '  <span class="legend-chip legend-HTL">HTL</span>',
        '  <span class="legend-chip legend-back_contact">back contact</span>',
        '</div>',
      )

      // Stack-level actions.
      parts.push(
        '<div class="stack-actions">',
        '  <button class="btn btn-ghost" data-stack-action="add">＋ Add layer…</button>',
        '  <button class="btn btn-ghost" data-stack-action="save-as">Save as…</button>',
        '  <button class="btn btn-ghost" data-stack-action="download-yaml">↓ YAML</button>',
        '</div>',
      )

      container.innerHTML = parts.join('\n')
      // Suppress unused-import warning — reconcileInterfaces is exported
      // for callers that need it; the visualizer itself doesn't.
      void reconcileInterfaces
    },
  }
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd perovskite-sim/frontend && npx tsc --noEmit
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
cd perovskite-sim && git add frontend/src/stack/stack-visualizer.ts
git commit -m "feat(frontend): add stack-visualizer composition + DnD wiring

Pure render handle; state lives in the parent (device-pane).
Native HTML5 drag-and-drop for reorder; ↑↓ buttons as the
keyboard-accessible fallback. Insert-gap and edit-interface
events bubble up via custom events so the parent owns dialogs."
```

---

## Task 16: CSS for stack visualizer

**Files:**
- Modify: `frontend/src/style.css`

- [ ] **Step 1: Append the visualizer styles**

Append to `perovskite-sim/frontend/src/style.css`:

```css
/* ── Phase 2b layer builder ───────────────────────────────────────── */

:root {
  --role-substrate-bg: linear-gradient(90deg, #e8e8ec 0%, #d8d8dc 100%);
  --role-substrate-border: #b0b0b8;
  --role-substrate-fg: #333;
  --role-front_contact-bg: linear-gradient(90deg, #f0e5c9 0%, #e5d8b0 100%);
  --role-front_contact-border: #c9b878;
  --role-front_contact-fg: #5a4a20;
  --role-ETL-bg: linear-gradient(90deg, #d6e8f2 0%, #b8d8e8 100%);
  --role-ETL-border: #6a9ab8;
  --role-ETL-fg: #234a66;
  --role-absorber-bg: linear-gradient(90deg, #3a2a5a 0%, #2a1a4a 100%);
  --role-absorber-border: #2a1a4a;
  --role-absorber-fg: #ffffff;
  --role-HTL-bg: linear-gradient(90deg, #e8d4e8 0%, #d4b8d4 100%);
  --role-HTL-border: #a878a8;
  --role-HTL-fg: #5a2a5a;
  --role-back_contact-bg: linear-gradient(90deg, #b8860b 0%, #8b6508 100%);
  --role-back_contact-border: #6a4508;
  --role-back_contact-fg: #ffffff;
  --selected-outline: #ff9b45;
}

.device-pane-grid {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 22px;
  align-items: start;
}

@media (max-width: 1100px) {
  .device-pane-grid {
    grid-template-columns: 1fr;
  }
}

.stack-visualizer {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.stack-visualizer-sun {
  text-align: center;
  color: #d29000;
  font-size: 11px;
  line-height: 1.1;
}
.stack-visualizer-sun-rays {
  letter-spacing: 6px;
  font-size: 14px;
}

.stack-visualizer-frame {
  background: #fff;
  border: 1px solid #c9c9d1;
  border-radius: 6px;
  padding: 6px;
  display: flex;
  flex-direction: column;
}

.stack-insert-gap {
  position: relative;
  height: 4px;
}
.stack-insert-gap:hover { height: 18px; }
.stack-insert-btn {
  position: absolute;
  left: 50%;
  top: -4px;
  transform: translateX(-50%);
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #fff;
  border: 1px solid var(--selected-outline);
  color: var(--selected-outline);
  font-size: 14px;
  line-height: 1;
  cursor: pointer;
  opacity: 0;
  transition: opacity 80ms;
}
.stack-insert-gap:hover .stack-insert-btn { opacity: 1; }

.layer-card {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  margin: 2px 0;
  border-radius: 3px;
  border: 1px solid;
  cursor: pointer;
  position: relative;
  user-select: none;
}
.layer-card.is-selected {
  outline: 2px solid var(--selected-outline);
  outline-offset: -2px;
  box-shadow: 0 0 0 3px rgba(255, 155, 69, 0.2);
}
.layer-card.is-error {
  border-color: #d05050 !important;
  box-shadow: 0 0 0 1px #d05050 inset;
}
.layer-card-substrate     { background: var(--role-substrate-bg);     border-color: var(--role-substrate-border);     color: var(--role-substrate-fg); }
.layer-card-front_contact { background: var(--role-front_contact-bg); border-color: var(--role-front_contact-border); color: var(--role-front_contact-fg); }
.layer-card-ETL           { background: var(--role-ETL-bg);           border-color: var(--role-ETL-border);           color: var(--role-ETL-fg); }
.layer-card-absorber      { background: var(--role-absorber-bg);      border-color: var(--role-absorber-border);      color: var(--role-absorber-fg); }
.layer-card-HTL           { background: var(--role-HTL-bg);           border-color: var(--role-HTL-border);           color: var(--role-HTL-fg); }
.layer-card-back_contact  { background: var(--role-back_contact-bg);  border-color: var(--role-back_contact-border);  color: var(--role-back_contact-fg); }

.layer-card-handle {
  cursor: grab;
  font-size: 13px;
  opacity: 0.7;
}
.layer-card-body { flex: 1; min-width: 0; }
.layer-card-name {
  font-size: 12px;
  font-weight: 600;
  font-feature-settings: "subs", "sups";
}
.layer-card-meta {
  font-size: 10px;
  opacity: 0.85;
}
.layer-card-pill {
  background: rgba(255, 255, 255, 0.85);
  color: #333;
  border-radius: 10px;
  padding: 1px 7px;
  font-size: 9px;
  border: 1px solid rgba(0, 0, 0, 0.15);
}
.layer-card-controls {
  display: flex;
  gap: 2px;
  opacity: 0;
  transition: opacity 80ms;
}
.layer-card:hover .layer-card-controls,
.layer-card:focus-within .layer-card-controls { opacity: 1; }
.layer-card-controls button {
  background: rgba(255,255,255,0.85);
  border: 1px solid rgba(0,0,0,0.15);
  color: #333;
  width: 18px;
  height: 18px;
  border-radius: 3px;
  font-size: 11px;
  line-height: 1;
  cursor: pointer;
  padding: 0;
}

.iface-strip {
  background: #f0f4fa;
  border: 1px solid #d0e0f0;
  color: #6a8ab8;
  font-size: 9px;
  padding: 2px 6px;
  border-radius: 2px;
  margin: 2px 0;
  text-align: center;
  cursor: pointer;
  width: 100%;
}
.iface-strip.is-default {
  background: #fff8e0;
  border-color: #e0d090;
  color: #8a6020;
}

.stack-error-banner {
  background: #ffe5e5;
  border: 1px solid #d05050;
  color: #802020;
  padding: 6px 10px;
  border-radius: 4px;
  font-size: 11px;
  margin-top: 6px;
}

.stack-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
  font-size: 9px;
}
.legend-chip {
  padding: 2px 6px;
  border-radius: 10px;
  border: 1px solid rgba(0,0,0,0.1);
}
.legend-substrate     { background: #e8e8ec; color: #555; }
.legend-front_contact { background: #f0e5c9; color: #5a4a20; }
.legend-ETL           { background: #d6e8f2; color: #234a66; }
.legend-absorber      { background: #3a2a5a; color: #fff; }
.legend-HTL           { background: #e8d4e8; color: #5a2a5a; }
.legend-back_contact  { background: #b8860b; color: #fff; }

.stack-actions {
  display: flex;
  gap: 6px;
  margin-top: 10px;
}
.stack-actions button { flex: 1; font-size: 11px; }

.dirty-pill {
  background: #ff9b45;
  color: #fff;
  border-radius: 10px;
  padding: 1px 7px;
  font-size: 10px;
  margin-left: 6px;
  font-weight: 600;
}
```

- [ ] **Step 2: Commit**

```bash
cd perovskite-sim && git add frontend/src/style.css
git commit -m "style(frontend): add stack visualizer CSS

Role color palette (CSS variables), two-column device-pane grid
collapsing to one column under 1100px, drag/hover affordances,
selected-card outline, error/dirty pills."
```

---

# Phase D — Dialogs

## Task 17: `frontend/src/stack/add-layer-dialog.ts`

**Files:**
- Create: `frontend/src/stack/add-layer-dialog.ts`

- [ ] **Step 1: Implement the dialog**

Write to `perovskite-sim/frontend/src/stack/add-layer-dialog.ts`:

```typescript
import type { LayerConfig, LayerRole, LayerTemplate } from '../types'

const ROLES: ReadonlyArray<LayerRole> = [
  'substrate', 'front_contact', 'ETL', 'absorber', 'HTL', 'back_contact',
]

const HTML_ESCAPE: Record<string, string> = {
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}
function esc(s: string): string {
  return s.replace(/[&<>"']/g, c => HTML_ESCAPE[c])
}

function blankLayer(name: string, role: LayerRole): LayerConfig {
  return {
    name, role, thickness: 1e-7, eps_r: 1,
    mu_n: 0, mu_p: 0, ni: 0, N_D: 0, N_A: 0,
    D_ion: 0, P_lim: 0, P0: 0,
    tau_n: 0, tau_p: 0, n1: 0, p1: 0,
    B_rad: 0, C_n: 0, C_p: 0, alpha: 0,
  }
}

function templateToLayer(name: string, t: LayerTemplate): LayerConfig {
  return {
    ...blankLayer(name, t.role),
    optical_material: t.optical_material,
    ...(t.defaults as Partial<LayerConfig>),
    name,
    role: t.role,
  }
}

/**
 * Open the Add Layer dialog. Returns a promise that resolves with the new
 * LayerConfig (or null if the user cancels).
 *
 * The dialog is a single inline overlay — no library, no portal magic.
 */
export function openAddLayerDialog(
  templates: Record<string, LayerTemplate>,
): Promise<LayerConfig | null> {
  return new Promise(resolve => {
    const overlay = document.createElement('div')
    overlay.className = 'modal-overlay'
    overlay.innerHTML = `
      <div class="modal" role="dialog" aria-label="Add Layer">
        <div class="modal-header">Add layer</div>
        <div class="modal-tabs">
          <button class="modal-tab is-active" data-tab="template">Template</button>
          <button class="modal-tab" data-tab="blank">Blank</button>
        </div>
        <div class="modal-body" id="add-layer-body"></div>
        <div class="modal-footer">
          <button class="btn btn-ghost" data-cancel>Cancel</button>
          <button class="btn btn-primary" id="add-layer-ok" disabled>Add</button>
        </div>
      </div>`
    document.body.appendChild(overlay)

    const body = overlay.querySelector<HTMLElement>('#add-layer-body')!
    const okBtn = overlay.querySelector<HTMLButtonElement>('#add-layer-ok')!
    let pending: LayerConfig | null = null

    function close(result: LayerConfig | null): void {
      overlay.remove()
      resolve(result)
    }

    function renderTemplateTab(): void {
      const items = Object.entries(templates)
        .map(([key, t]) => `
          <button class="template-item" data-template-key="${esc(key)}">
            <div class="template-name">${esc(key)}</div>
            <div class="template-role">${esc(t.role)}</div>
            <div class="template-desc">${esc(t.description)}</div>
            <div class="template-source">${esc(t.source)}</div>
          </button>`)
        .join('')
      body.innerHTML = `<div class="template-list">${items || '<em>No templates available</em>'}</div>`
    }

    function renderBlankTab(): void {
      const roleOpts = ROLES.map(r => `<option value="${r}">${r}</option>`).join('')
      body.innerHTML = `
        <label class="param">
          <span class="param-label">Name</span>
          <input type="text" id="blank-name" class="num-input" value="" spellcheck="false">
        </label>
        <label class="param">
          <span class="param-label">Role</span>
          <select id="blank-role" class="num-input">${roleOpts}</select>
        </label>`
      const nameInput = body.querySelector<HTMLInputElement>('#blank-name')!
      const roleSelect = body.querySelector<HTMLSelectElement>('#blank-role')!
      function refresh(): void {
        const name = nameInput.value.trim()
        if (!name) {
          pending = null
          okBtn.disabled = true
          return
        }
        pending = blankLayer(name, roleSelect.value as LayerRole)
        okBtn.disabled = false
      }
      nameInput.addEventListener('input', refresh)
      roleSelect.addEventListener('change', refresh)
    }

    overlay.addEventListener('click', ev => {
      const target = ev.target as HTMLElement
      if (target.dataset.cancel != null || target === overlay) {
        close(null)
        return
      }
      const tabBtn = target.closest<HTMLElement>('[data-tab]')
      if (tabBtn) {
        overlay.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('is-active'))
        tabBtn.classList.add('is-active')
        if (tabBtn.dataset.tab === 'template') renderTemplateTab()
        else renderBlankTab()
        pending = null
        okBtn.disabled = true
        return
      }
      const tplBtn = target.closest<HTMLElement>('[data-template-key]')
      if (tplBtn) {
        const key = tplBtn.dataset.templateKey!
        const tmpl = templates[key]
        if (tmpl) {
          pending = templateToLayer(key, tmpl)
          okBtn.disabled = false
          overlay.querySelectorAll('.template-item').forEach(el => el.classList.remove('is-selected'))
          tplBtn.classList.add('is-selected')
        }
      }
    })

    okBtn.addEventListener('click', () => close(pending))

    renderTemplateTab()
  })
}
```

- [ ] **Step 2: Append minimal modal CSS**

Append to `perovskite-sim/frontend/src/style.css`:

```css
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}
.modal {
  background: #fff;
  border-radius: 6px;
  min-width: 360px;
  max-width: 520px;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.modal-header { padding: 12px 16px; font-weight: 600; border-bottom: 1px solid #e5e5ec; }
.modal-tabs { display: flex; border-bottom: 1px solid #e5e5ec; }
.modal-tab {
  flex: 1;
  padding: 8px;
  background: #f7f7fa;
  border: none;
  border-bottom: 2px solid transparent;
  cursor: pointer;
  font-size: 12px;
}
.modal-tab.is-active { background: #fff; border-bottom-color: #ff9b45; }
.modal-body { padding: 12px 16px; overflow: auto; flex: 1; }
.modal-footer {
  padding: 8px 16px;
  border-top: 1px solid #e5e5ec;
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

.template-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.template-item {
  text-align: left;
  background: #fff;
  border: 1px solid #c9c9d1;
  border-radius: 4px;
  padding: 8px 10px;
  cursor: pointer;
  font: inherit;
}
.template-item.is-selected { border-color: #ff9b45; background: #fff8f0; }
.template-name { font-weight: 600; font-size: 12px; }
.template-role { font-size: 10px; color: #888; }
.template-desc { font-size: 11px; color: #444; margin-top: 2px; }
.template-source { font-size: 9px; color: #888; margin-top: 2px; font-style: italic; }
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd perovskite-sim/frontend && npx tsc --noEmit
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
cd perovskite-sim && git add frontend/src/stack/add-layer-dialog.ts frontend/src/style.css
git commit -m "feat(frontend): add Add Layer dialog (template + blank tabs)"
```

---

## Task 18: `frontend/src/stack/save-as-dialog.ts`

**Files:**
- Create: `frontend/src/stack/save-as-dialog.ts`

- [ ] **Step 1: Implement the dialog**

Write to `perovskite-sim/frontend/src/stack/save-as-dialog.ts`:

```typescript
import type { DeviceConfig } from '../types'
import { checkUserConfigExists, saveUserConfig } from '../api'

const FILENAME_RE = /^[a-zA-Z0-9_-]{1,64}$/
const HTML_ESCAPE: Record<string, string> = {
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}
function esc(s: string): string {
  return s.replace(/[&<>"']/g, c => HTML_ESCAPE[c])
}

export interface SaveAsResult {
  saved: string
}

/**
 * Open the Save-As dialog. Returns the saved name on success, or null
 * if the user cancels. Errors from the backend are surfaced inline.
 */
export function openSaveAsDialog(
  config: DeviceConfig,
): Promise<SaveAsResult | null> {
  return new Promise(resolve => {
    const overlay = document.createElement('div')
    overlay.className = 'modal-overlay'
    overlay.innerHTML = `
      <div class="modal" role="dialog" aria-label="Save device as">
        <div class="modal-header">Save device as user preset</div>
        <div class="modal-body">
          <label class="param">
            <span class="param-label">Filename</span>
            <input type="text" id="save-as-name" class="num-input" placeholder="my_custom_stack" spellcheck="false" autocomplete="off">
          </label>
          <div id="save-as-hint" class="save-as-hint"></div>
        </div>
        <div class="modal-footer">
          <button class="btn btn-ghost" data-cancel>Cancel</button>
          <button class="btn btn-primary" id="save-as-ok" disabled>Save</button>
        </div>
      </div>`
    document.body.appendChild(overlay)

    const input = overlay.querySelector<HTMLInputElement>('#save-as-name')!
    const okBtn = overlay.querySelector<HTMLButtonElement>('#save-as-ok')!
    const hint = overlay.querySelector<HTMLElement>('#save-as-hint')!
    let overwriteAllowed = false
    let probeToken = 0

    function setHint(text: string, kind: 'ok' | 'warn' | 'error'): void {
      hint.textContent = text
      hint.className = `save-as-hint save-as-hint-${kind}`
    }

    async function refresh(): Promise<void> {
      const name = input.value.trim()
      overwriteAllowed = false
      if (!name) {
        setHint('', 'ok')
        okBtn.disabled = true
        okBtn.textContent = 'Save'
        return
      }
      if (!FILENAME_RE.test(name)) {
        setHint('Use letters, digits, hyphen, underscore (max 64).', 'error')
        okBtn.disabled = true
        okBtn.textContent = 'Save'
        return
      }
      const myToken = ++probeToken
      setHint('Checking…', 'ok')
      okBtn.disabled = true
      try {
        const probe = await checkUserConfigExists(name)
        if (myToken !== probeToken) return  // stale
        if (probe.exists && probe.namespace === 'shipped') {
          setHint(`"${name}" is reserved by a shipped preset.`, 'error')
          okBtn.disabled = true
          okBtn.textContent = 'Save'
          return
        }
        if (probe.exists && probe.namespace === 'user') {
          setHint(`"${name}" already exists. Click Overwrite to replace it.`, 'warn')
          okBtn.disabled = false
          okBtn.textContent = 'Overwrite'
          overwriteAllowed = true
          return
        }
        setHint(`"${name}" is available.`, 'ok')
        okBtn.disabled = false
        okBtn.textContent = 'Save'
      } catch (e) {
        setHint(`Probe failed: ${(e as Error).message}`, 'error')
        okBtn.disabled = true
      }
    }

    function close(result: SaveAsResult | null): void {
      overlay.remove()
      resolve(result)
    }

    let debounceId: number | undefined
    input.addEventListener('input', () => {
      window.clearTimeout(debounceId)
      debounceId = window.setTimeout(() => { void refresh() }, 250)
    })

    overlay.addEventListener('click', ev => {
      const target = ev.target as HTMLElement
      if (target.dataset.cancel != null || target === overlay) close(null)
    })

    okBtn.addEventListener('click', async () => {
      const name = input.value.trim()
      if (!FILENAME_RE.test(name)) return
      okBtn.disabled = true
      const result = await saveUserConfig(name, config, overwriteAllowed)
      if (result.ok) {
        close({ saved: result.saved })
      } else {
        setHint(`Save failed: ${esc(result.detail)}`, 'error')
        okBtn.disabled = false
      }
    })

    input.focus()
  })
}
```

- [ ] **Step 2: Append save-as hint CSS**

Append to `perovskite-sim/frontend/src/style.css`:

```css
.save-as-hint {
  margin-top: 6px;
  font-size: 11px;
  min-height: 16px;
}
.save-as-hint-ok    { color: #2a8040; }
.save-as-hint-warn  { color: #8a6020; }
.save-as-hint-error { color: #802020; }
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd perovskite-sim/frontend && npx tsc --noEmit
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
cd perovskite-sim && git add frontend/src/stack/save-as-dialog.ts frontend/src/style.css
git commit -m "feat(frontend): add Save-As dialog with collision probe"
```

---

# Phase E — Detail editor shrink

## Task 19: Single-layer mode in `config-editor.ts` + extended role values

**Files:**
- Modify: `frontend/src/config-editor.ts`

- [ ] **Step 1: Extend the role dropdown options**

Edit `perovskite-sim/frontend/src/config-editor.ts`. Find the `<select class="layer-role" ...>` block in `renderLayer` and replace with:

```typescript
        <select class="layer-role" id="layer-${idx}-role">
          <option value="substrate" ${layer.role === 'substrate' ? 'selected' : ''}>substrate</option>
          <option value="front_contact" ${layer.role === 'front_contact' ? 'selected' : ''}>front contact</option>
          <option value="ETL" ${layer.role === 'ETL' ? 'selected' : ''}>ETL</option>
          <option value="absorber" ${layer.role === 'absorber' ? 'selected' : ''}>absorber</option>
          <option value="HTL" ${layer.role === 'HTL' ? 'selected' : ''}>HTL</option>
          <option value="back_contact" ${layer.role === 'back_contact' ? 'selected' : ''}>back contact</option>
        </select>
```

- [ ] **Step 2: Add `selectedLayerIdx` parameter to `renderDeviceEditor`**

In the same file, change the `renderDeviceEditor` signature and body to support an optional single-layer view:

```typescript
export function renderDeviceEditor(
  container: HTMLElement,
  config: DeviceConfig,
  tier?: SimulationModeName,
  selectedLayerIdx?: number,
): void {
  const singleLayer = selectedLayerIdx != null && tier === 'full'
  const layerHtml = singleLayer
    ? renderLayer(config.layers[selectedLayerIdx!], selectedLayerIdx!, tier)
    : config.layers.map((layer, idx) => renderLayer(layer, idx, tier)).join('')
  // (rest of the function body unchanged — keep the existing template
  // string but use `layerHtml` in place of the prior `layers` variable)
  const currentMode: SimulationModeName = isModeName(config.device.mode) ? config.device.mode : 'full'
  const currentT = config.device.T ?? 300
  const showT = !tier || isFieldVisible('T', tier)
  const tField = showT ? `
          <label class="param">
            <span class="param-label"><span class="sym"><i>T</i></span><span class="unit">K</span></span>
            ${numAttr('dev-T', currentT)}
          </label>` : ''
  // In single-layer mode, hide the device-level fields and the interfaces
  // section — they live in the visualizer / device-pane header.
  const deviceGroup = singleLayer ? '' : `
      <div class="param-group">
        <h5>Device</h5>
        <div class="param-grid">
          <label class="param">
            <span class="param-label"><span class="sym">Mode</span></span>
            <select class="num-input" id="dev-mode">${renderModeOptions(currentMode)}</select>
          </label>${tField}
          <label class="param">
            <span class="param-label"><span class="sym"><i>V</i><sub>bi</sub></span><span class="unit">V</span></span>
            ${numAttr('dev-Vbi', config.device.V_bi)}
          </label>
          <label class="param">
            <span class="param-label"><span class="sym"><i>Φ</i></span><span class="unit">m⁻²·s⁻¹</span></span>
            ${numAttr('dev-Phi', config.device.Phi)}
          </label>
        </div>
      </div>`
  const interfacesHtml = singleLayer ? '' : renderInterfaces(config)
  container.innerHTML = `
    <div class="editor">
      ${deviceGroup}
      ${interfacesHtml}
      <div class="layer-list">${layerHtml}</div>
    </div>`
}
```

- [ ] **Step 3: Update `readDeviceEditor` to handle single-layer mode**

In the same file, modify `readDeviceEditor` to merge in only the visible layer when in single-layer mode. Replace the function with:

```typescript
export function readDeviceEditor(
  original: DeviceConfig,
  selectedLayerIdx?: number,
): DeviceConfig {
  const singleLayer = selectedLayerIdx != null
  const layers: LayerConfig[] = original.layers.map((layer, idx) => {
    if (singleLayer && idx !== selectedLayerIdx) {
      return layer  // only the visible layer is in the DOM; others pass through
    }
    const next: LayerConfig = { ...layer }
    next.name = parseText(`layer-${idx}-name`, layer.name)
    next.role = parseText(`layer-${idx}-role`, layer.role)
    for (const group of LAYER_GROUPS) {
      for (const f of group.fields) {
        const id = `layer-${idx}-${String(f.key)}`
        switch (f.kind) {
          case 'numeric': {
            const original_v = (layer[f.key] as number | undefined) ?? 0
            ;(next as unknown as Record<string, number>)[f.key as string] = parseNum(id, original_v)
            break
          }
          case 'select-optical-material': {
            next.optical_material = parseOpticalMaterial(idx, layer.optical_material)
            break
          }
          case 'boolean': {
            next.incoherent = parseCheckbox(id, layer.incoherent ?? false)
            break
          }
        }
      }
    }
    return next
  })

  if (singleLayer) {
    // In single-layer mode the device-level controls are not in the DOM —
    // pass through original values so the parent state remains coherent.
    return { device: original.device, layers }
  }

  const interfaces: Array<[number, number]> = []
  for (let i = 0; i < layers.length - 1; i++) {
    const existing = original.device.interfaces?.[i] ?? [0, 0]
    interfaces.push([
      parseNum(`iface-${i}-vn`, existing[0]),
      parseNum(`iface-${i}-vp`, existing[1]),
    ])
  }
  const rawMode = parseText('dev-mode', original.device.mode ?? 'full')
  const mode: SimulationModeName = isModeName(rawMode) ? rawMode : 'full'
  const T = parseNum('dev-T', original.device.T ?? 300)
  return {
    device: {
      V_bi: parseNum('dev-Vbi', original.device.V_bi),
      Phi: parseNum('dev-Phi', original.device.Phi),
      interfaces,
      T,
      mode,
    },
    layers,
  }
}
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd perovskite-sim/frontend && npx tsc --noEmit
```

Expected: clean.

- [ ] **Step 5: Run pre-existing frontend tests**

```bash
cd perovskite-sim/frontend && npx vitest run
```

Expected: all green (single-layer mode is opt-in via the new optional parameter; default behavior is unchanged).

- [ ] **Step 6: Commit**

```bash
cd perovskite-sim && git add frontend/src/config-editor.ts
git commit -m "feat(frontend): config-editor single-layer mode + extended roles

renderDeviceEditor and readDeviceEditor accept an optional
selectedLayerIdx; when set in full tier the editor renders only
that one layer's groups (no device fields, no interfaces) so the
visualizer can own the navigation. Role dropdown gains substrate /
front_contact / back_contact (Phase 2a parity fix)."
```

---

# Phase F — Wire it into the Device pane

## Task 20: `frontend/src/workstation/panes/device-pane.ts` — two-column host

**Files:**
- Modify: `frontend/src/workstation/panes/device-pane.ts`

- [ ] **Step 1: Replace the pane with a two-column host in full tier**

Replace `perovskite-sim/frontend/src/workstation/panes/device-pane.ts` with:

```typescript
import { mountDevicePanel } from '../../device-panel'
import type { DevicePanel } from '../../device-panel'
import type { SimulationModeName } from '../../types'
import { isLayerBuilderEnabled } from '../tier-gating'

/**
 * Build the Device pane contents into the given container.
 *
 * In full tier (Phase 2b), the pane is a CSS grid with two columns:
 * - left: the stack visualizer (rendered by mountDevicePanel which delegates
 *   to the visualizer when the builder is enabled);
 * - right: the per-layer detail editor.
 *
 * In fast / legacy tiers, the pane keeps the existing single-column
 * accordion editor — no behavior change, no regression risk for benchmark
 * workflows.
 */
export async function mountDevicePane(
  container: HTMLElement,
  tabId: string,
  tier: SimulationModeName = 'full',
): Promise<DevicePanel> {
  container.classList.add('pane', 'pane-device')
  const inner = document.createElement('div')
  inner.className = 'pane-body'
  if (isLayerBuilderEnabled(tier)) {
    inner.classList.add('device-pane-grid')
  }
  container.appendChild(inner)
  return mountDevicePanel(inner, tabId, { tier })
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd perovskite-sim/frontend && npx tsc --noEmit
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
cd perovskite-sim && git add frontend/src/workstation/panes/device-pane.ts
git commit -m "feat(frontend): device-pane two-column grid in full tier"
```

---

## Task 21: `frontend/src/device-panel.ts` — wire visualizer + dialogs + dirty pill

**Files:**
- Modify: `frontend/src/device-panel.ts`

- [ ] **Step 1: Rewrite `mountDevicePanel` to host the visualizer in full tier**

Replace `perovskite-sim/frontend/src/device-panel.ts` with the following. This is a substantial rewrite — preserve exactly the behavior described and re-use the helpers from prior tasks.

```typescript
import {
  listConfigs,
  getConfig,
  fetchOpticalMaterials,
  fetchLayerTemplates,
} from './api'
import { renderDeviceEditor, readDeviceEditor, setOpticalMaterialOptions } from './config-editor'
import { mountStackVisualizer } from './stack/stack-visualizer'
import { validate } from './stack/stack-validator'
import { isDirty } from './stack/dirty-state'
import { reconcileInterfaces } from './stack/reconcile-interfaces'
import { openAddLayerDialog } from './stack/add-layer-dialog'
import { openSaveAsDialog } from './stack/save-as-dialog'
import { isLayerBuilderEnabled } from './workstation/tier-gating'
import type {
  DeviceConfig,
  LayerConfig,
  LayerTemplate,
  SimulationModeName,
  StackAction,
} from './types'

export interface DevicePanel {
  getConfig(): DeviceConfig
  onChange(cb: (cfg: DeviceConfig) => void): void
}

export interface MountDevicePanelOptions {
  tier?: SimulationModeName
}

/**
 * Render the "TMM active · N layers" badge shown in the device-pane header.
 * Returns an empty string unless the device is on the `full` tier AND at
 * least one layer has a non-empty `optical_material` field.
 */
export function computeTmmBadge(
  config: DeviceConfig,
  tier: SimulationModeName | undefined,
): string {
  if (tier !== 'full') return ''
  const tmmLayers = config.layers.filter(
    l => l.optical_material != null && l.optical_material !== '',
  )
  if (tmmLayers.length === 0) return ''
  return `<span class="tmm-badge" title="Optical generation computed with transfer-matrix method. Layers without optical_material fall back to Beer-Lambert.">TMM active · ${tmmLayers.length} layers</span>`
}

export async function mountDevicePanel(
  root: HTMLElement,
  tabId: string,
  options: MountDevicePanelOptions = {},
): Promise<DevicePanel> {
  const { tier } = options
  const builderOn = isLayerBuilderEnabled(tier ?? 'full')

  root.innerHTML = `
    <div class="card">
      <div class="card-header">
        <h3>Device Configuration <span id="${tabId}-tmm-badge-slot"></span> <span id="${tabId}-dirty-slot"></span></h3>
        <div class="header-actions">
          <select id="${tabId}-config-select" class="config-select"></select>
          <button class="btn btn-ghost" id="${tabId}-reset">Reset</button>
        </div>
      </div>
      ${builderOn
        ? `<div id="${tabId}-visualizer"></div><div id="${tabId}-editor"></div>`
        : `<div id="${tabId}-editor"></div>`}
    </div>`

  const select = root.querySelector<HTMLSelectElement>(`#${tabId}-config-select`)!
  const editor = root.querySelector<HTMLDivElement>(`#${tabId}-editor`)!
  const resetBtn = root.querySelector<HTMLButtonElement>(`#${tabId}-reset`)!
  const badgeSlot = root.querySelector<HTMLSpanElement>(`#${tabId}-tmm-badge-slot`)!
  const dirtySlot = root.querySelector<HTMLSpanElement>(`#${tabId}-dirty-slot`)!

  // ── State (full tier only) ───────────────────────────────────────────────
  let loaded: DeviceConfig | null = null
  let current: DeviceConfig | null = null
  let selectedLayerIdx = 0
  let templates: Record<string, LayerTemplate> = {}
  const listeners: Array<(c: DeviceConfig) => void> = []

  // Optical materials.
  try {
    const materials = await fetchOpticalMaterials()
    setOpticalMaterialOptions(materials)
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn('fetchOpticalMaterials failed', err)
    setOpticalMaterialOptions([])
  }

  // Templates (full tier only — fast/legacy never opens the dialog).
  if (builderOn) {
    try {
      templates = await fetchLayerTemplates()
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn('fetchLayerTemplates failed', err)
    }
  }

  // Configs dropdown (Shipped / User optgroups).
  const entries = await listConfigs()
  const shipped = entries.filter(e => e.namespace === 'shipped')
  const user = entries.filter(e => e.namespace === 'user')
  const optgroup = (label: string, items: typeof entries): string =>
    items.length === 0
      ? ''
      : `<optgroup label="${label}">${items
          .map(e => `<option value="${e.name}">${e.name.replace(/\.ya?ml$/, '')}</option>`)
          .join('')}</optgroup>`
  select.innerHTML = optgroup('Shipped presets', shipped) + optgroup('User presets', user)

  function refreshDirtyPill(): void {
    if (loaded && current && isDirty(loaded, current)) {
      dirtySlot.innerHTML = '<span class="dirty-pill">● modified</span>'
    } else {
      dirtySlot.innerHTML = ''
    }
  }

  function refreshBadge(cfg: DeviceConfig): void {
    badgeSlot.innerHTML = computeTmmBadge(cfg, tier)
  }

  function refreshFastLegacyEditor(cfg: DeviceConfig): void {
    renderDeviceEditor(editor, cfg, tier)
    refreshBadge(cfg)
    refreshDirtyPill()
  }

  // ── Full-tier visualizer wiring ─────────────────────────────────────────
  let visualizerHandle: ReturnType<typeof mountStackVisualizer> | null = null
  if (builderOn) {
    const visualizerEl = root.querySelector<HTMLElement>(`#${tabId}-visualizer`)!
    visualizerHandle = mountStackVisualizer(visualizerEl, action => handleStackAction(action))

    visualizerEl.addEventListener('stack-insert-request', async (ev: Event) => {
      const detail = (ev as CustomEvent<{ atIdx: number }>).detail
      const layer = await openAddLayerDialog(templates)
      if (layer && current) {
        const newLayers = [...current.layers]
        newLayers.splice(detail.atIdx, 0, layer)
        const newInterfaces = reconcileInterfaces(
          current.layers,
          newLayers,
          current.device.interfaces ?? [],
        )
        current = {
          ...current,
          layers: newLayers,
          device: { ...current.device, interfaces: newInterfaces },
        }
        selectedLayerIdx = detail.atIdx
        rerender()
      }
    })

    visualizerEl.addEventListener('stack-edit-iface', (ev: Event) => {
      const detail = (ev as CustomEvent<{ ifaceIdx: number }>).detail
      if (!current) return
      const existing = (current.device.interfaces?.[detail.ifaceIdx] ?? [0, 0]) as readonly [number, number]
      const vn = window.prompt('v_n (m/s)', String(existing[0]))
      if (vn == null) return
      const vp = window.prompt('v_p (m/s)', String(existing[1]))
      if (vp == null) return
      const newPair: [number, number] = [Number(vn) || 0, Number(vp) || 0]
      const newIfaces = [...(current.device.interfaces ?? [])]
      while (newIfaces.length < current.layers.length - 1) newIfaces.push([0, 0])
      newIfaces[detail.ifaceIdx] = newPair
      current = {
        ...current,
        device: { ...current.device, interfaces: newIfaces },
      }
      rerender()
    })

    // Stack-action toolbar buttons inside the visualizer's HTML.
    visualizerEl.addEventListener('click', async ev => {
      const target = ev.target as HTMLElement
      const stackAction = target.closest<HTMLElement>('[data-stack-action]')
      if (!stackAction) return
      const action = stackAction.dataset.stackAction!
      if (action === 'add' && current) {
        const layer = await openAddLayerDialog(templates)
        if (layer) {
          const newLayers = [...current.layers, layer]
          const newInterfaces = reconcileInterfaces(
            current.layers, newLayers, current.device.interfaces ?? [],
          )
          current = {
            ...current,
            layers: newLayers,
            device: { ...current.device, interfaces: newInterfaces },
          }
          selectedLayerIdx = current.layers.length - 1
          rerender()
        }
      } else if (action === 'save-as' && current) {
        const result = await openSaveAsDialog(current)
        if (result) {
          // Refresh dropdown and clear dirty.
          loaded = current
          refreshDirtyPill()
          await refreshConfigsDropdown(result.saved)
        }
      } else if (action === 'download-yaml' && current) {
        downloadYaml(current)
      }
    })
  }

  function handleStackAction(action: StackAction): void {
    if (!current) return
    switch (action.type) {
      case 'select':
        selectedLayerIdx = action.idx
        rerender()
        return
      case 'delete': {
        if (current.layers.length <= 1) return
        if (current.layers[action.idx]?.role === 'absorber') {
          if (!window.confirm('Delete the absorber layer?')) return
        }
        const newLayers = current.layers.filter((_, i) => i !== action.idx)
        const newIfaces = reconcileInterfaces(
          current.layers, newLayers, current.device.interfaces ?? [],
        )
        current = {
          ...current,
          layers: newLayers,
          device: { ...current.device, interfaces: newIfaces },
        }
        if (selectedLayerIdx >= newLayers.length) selectedLayerIdx = newLayers.length - 1
        rerender()
        return
      }
      case 'reorder': {
        const { from, to } = action
        if (to < 0 || to >= current.layers.length) return
        const newLayers = [...current.layers]
        const [moved] = newLayers.splice(from, 1)
        newLayers.splice(to, 0, moved)
        const newIfaces = reconcileInterfaces(
          current.layers, newLayers, current.device.interfaces ?? [],
        )
        current = {
          ...current,
          layers: newLayers,
          device: { ...current.device, interfaces: newIfaces },
        }
        selectedLayerIdx = to
        rerender()
        return
      }
      case 'insert': {
        // Currently routed through stack-insert-request custom event; this
        // branch is here for completeness if a caller bypasses the dialog.
        const newLayers = [...current.layers]
        newLayers.splice(action.atIdx, 0, action.layer)
        const newIfaces = reconcileInterfaces(
          current.layers, newLayers, current.device.interfaces ?? [],
        )
        current = {
          ...current,
          layers: newLayers,
          device: { ...current.device, interfaces: newIfaces },
        }
        selectedLayerIdx = action.atIdx
        rerender()
        return
      }
      case 'edit-interface': {
        const newIfaces = [...(current.device.interfaces ?? [])]
        while (newIfaces.length < current.layers.length - 1) newIfaces.push([0, 0])
        newIfaces[action.idx] = [action.pair[0], action.pair[1]]
        current = {
          ...current,
          device: { ...current.device, interfaces: newIfaces },
        }
        rerender()
        return
      }
    }
  }

  function rerender(): void {
    if (!current) return
    if (builderOn && visualizerHandle) {
      const report = validate(current)
      visualizerHandle.render(current, selectedLayerIdx, report)
      renderDeviceEditor(editor, current, tier, selectedLayerIdx)
    } else {
      renderDeviceEditor(editor, current, tier)
    }
    refreshBadge(current)
    refreshDirtyPill()
  }

  async function refreshConfigsDropdown(selectName: string): Promise<void> {
    const newEntries = await listConfigs()
    const s = newEntries.filter(e => e.namespace === 'shipped')
    const u = newEntries.filter(e => e.namespace === 'user')
    select.innerHTML = optgroup('Shipped presets', s) + optgroup('User presets', u)
    const target = u.find(e => e.name.startsWith(`${selectName}.`))
      ?? s.find(e => e.name.startsWith(`${selectName}.`))
    if (target) select.value = target.name
  }

  function downloadYaml(cfg: DeviceConfig): void {
    // Minimal dependency-free YAML emitter — only handles the schema we use.
    const yaml = stringifyYaml(cfg)
    const blob = new Blob([yaml], { type: 'application/x-yaml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const stem = (cfg.layers[0]?.name ?? 'device').replace(/[^a-zA-Z0-9_-]/g, '_')
    a.download = `${stem}.yaml`
    a.click()
    URL.revokeObjectURL(url)
  }

  async function load(name: string) {
    const cfg = await getConfig(name)
    loaded = structuredClone(cfg)
    current = cfg
    selectedLayerIdx = 0
    rerender()
    listeners.forEach(l => l(cfg))
  }

  select.addEventListener('change', () => { void load(select.value) })
  resetBtn.addEventListener('click', () => {
    if (loaded) {
      current = structuredClone(loaded)
      rerender()
    }
  })

  const initial = shipped.find(e => e.name.includes('ionmonger'))?.name ?? shipped[0]?.name ?? entries[0]?.name
  if (!initial) throw new Error('no configs available')
  select.value = initial
  await load(initial)

  return {
    getConfig(): DeviceConfig {
      if (!current) throw new Error('device config not loaded')
      // In full-tier builder mode, current already reflects every visualizer
      // edit; only the detail editor's pending text-input changes need to be
      // merged in via readDeviceEditor (single-layer mode).
      if (builderOn) {
        return readDeviceEditor(current, selectedLayerIdx)
      }
      return readDeviceEditor(current)
    },
    onChange(cb) { listeners.push(cb) },
  }
}

// ── Minimal YAML emitter (safe for the DeviceConfig schema only) ─────────

function stringifyYaml(cfg: DeviceConfig): string {
  const lines: string[] = ['device:']
  for (const [k, v] of Object.entries(cfg.device)) {
    if (k === 'interfaces') continue
    lines.push(`  ${k}: ${formatYamlScalar(v)}`)
  }
  if (cfg.device.interfaces && cfg.device.interfaces.length > 0) {
    lines.push('  interfaces:')
    for (const pair of cfg.device.interfaces) {
      lines.push(`    - [${formatYamlScalar(pair[0])}, ${formatYamlScalar(pair[1])}]`)
    }
  }
  lines.push('layers:')
  for (const layer of cfg.layers) {
    lines.push('  - ' + Object.entries(layer)
      .map(([k, v]) => `${k === 'name' ? 'name' : k}: ${formatYamlScalar(v)}`)
      .join('\n    '))
  }
  return lines.join('\n') + '\n'
}

function formatYamlScalar(v: unknown): string {
  if (v == null) return 'null'
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return '0'
    if (v === 0) return '0'
    const abs = Math.abs(v)
    if (abs >= 1e-3 && abs < 1e6) return String(v)
    return v.toExponential(6)
  }
  if (typeof v === 'string') {
    if (/^[A-Za-z0-9_]+$/.test(v)) return v
    return JSON.stringify(v)
  }
  return JSON.stringify(v)
}

function optgroup(label: string, items: ReadonlyArray<{ name: string }>): string {
  if (items.length === 0) return ''
  return `<optgroup label="${label}">${items
    .map(e => `<option value="${e.name}">${e.name.replace(/\.ya?ml$/, '')}</option>`)
    .join('')}</optgroup>`
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd perovskite-sim/frontend && npx tsc --noEmit
```

Expected: clean. (If errors point to `ConfigEntry` not having `name`/`namespace`, re-check Task 6.)

- [ ] **Step 3: Run all frontend tests**

```bash
cd perovskite-sim/frontend && npx vitest run
```

Expected: green.

- [ ] **Step 4: Manual smoke test**

```bash
cd perovskite-sim && uvicorn backend.main:app --host 127.0.0.1 --port 8000 --app-dir perovskite-sim --reload &
sleep 2
cd perovskite-sim/frontend && (npm run dev &) ; sleep 3
```

Open `http://127.0.0.1:5173/` in a browser, switch the J–V Sweep tab to **Full** mode, load `nip_MAPbI3_tmm`, and verify:
- the visualizer column appears with all 6 layers, log-scale heights, role colors;
- clicking a layer selects it and the right-side editor switches;
- `＋ Add layer…` opens the template dialog;
- editing a numeric field shows the `● modified` pill;
- Reset clears the pill;
- Save-As writes a file under `configs/user/` and the dropdown picks it up.

Stop the servers with `kill %1 %2` (or your shell's job control).

- [ ] **Step 5: Commit**

```bash
cd perovskite-sim && git add frontend/src/device-panel.ts
git commit -m "feat(frontend): wire device-panel to layer builder

Full tier mounts the stack visualizer and routes every action
(select/delete/reorder/insert/edit-interface) through immutable
state updates. Add Layer + Save As dialogs, dirty pill, YAML
download. Fast/legacy tiers keep the existing accordion editor
unchanged."
```

---

# Phase G — Documentation

## Task 22: Tutorial + parameters panels + CLAUDE.md

**Files:**
- Modify: `frontend/src/panels/tutorial.ts`
- Modify: `frontend/src/panels/parameters.ts`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the "Custom Stacks" section to tutorial.ts**

Find an existing section heading in `perovskite-sim/frontend/src/panels/tutorial.ts` (e.g. the "Optical Generation" section added in Phase 2a) and add the following block after it:

```html
<h3>Custom Stacks</h3>
<p>
  In <strong>Full</strong> tier the Device pane shows your stack as a
  vertical cross-section. You can:
</p>
<ul>
  <li><strong>Add</strong> a layer with <em>＋ Add layer…</em> or any
      <em>+</em> between layers — pick a starter from the template
      library (TiO₂ ETL, spiro HTL, Ag back contact, …) or start blank.</li>
  <li><strong>Reorder</strong> by dragging a layer's <em>⋮⋮</em> handle,
      or by clicking the ↑↓ buttons on hover (keyboard-accessible).</li>
  <li><strong>Delete</strong> with the <em>✕</em> button on hover.</li>
  <li><strong>Edit</strong> any field by clicking a layer to select it
      and using the detail editor on the right.</li>
  <li><strong>Save</strong> as a named user preset via <em>Save as…</em>;
      it lands in <code>configs/user/</code> and appears under
      <em>User presets</em> in the dropdown.</li>
  <li><strong>Export</strong> the current device as YAML via
      <em>↓ YAML</em> for sharing outside the app.</li>
</ul>
<p>
  Both n-i-p (<code>ETL / absorber / HTL</code>) and p-i-n
  (<code>HTL / absorber / ETL</code>) orientations are supported — the
  simulator derives the built-in voltage from the stack itself.
</p>
```

(If the tutorial panel already uses a different markup style, match its conventions; the goal is to add the section, not to homogenise the panel.)

- [ ] **Step 2: Update parameters.ts with the extended role values**

Find the row for `role` in `perovskite-sim/frontend/src/panels/parameters.ts` and update its description to mention all six values:

```html
<tr>
  <td><code>role</code></td>
  <td>string</td>
  <td>full</td>
  <td>One of <code>substrate</code>, <code>front_contact</code>,
      <code>ETL</code>, <code>absorber</code>, <code>HTL</code>,
      <code>back_contact</code>. Substrate layers are filtered out of the
      electrical drift-diffusion grid (they participate only in the TMM
      optical stack). Every stack must contain exactly one absorber.</td>
</tr>
```

If a `role` row does not yet exist, add it after the `name` row.

- [ ] **Step 3: Update CLAUDE.md with a Phase 2b note**

Open `perovskite-sim/CLAUDE.md`, find the existing "Frontend" section (around the "## Frontend" heading), and append:

```markdown
**Custom stacks (Phase 2b — Apr 2026):** In full tier the Device pane renders a vertical layer visualizer with add/remove/reorder, a template library, structural validation, and a Save-As path that lands user presets in `configs/user/`. The accordion editor is preserved for fast/legacy tiers. New backend endpoints: `GET /api/layer-templates`, `POST /api/configs/user`. `GET /api/configs` now returns `{name, namespace}` entries.
```

- [ ] **Step 4: Verify the build still passes**

```bash
cd perovskite-sim/frontend && npm run build
```

Expected: tsc clean + bundle written to `dist/`.

- [ ] **Step 5: Commit**

```bash
cd perovskite-sim && git add frontend/src/panels/tutorial.ts frontend/src/panels/parameters.ts CLAUDE.md
git commit -m "docs(phase2b): tutorial + parameters + CLAUDE.md updates"
```

---

## Task 23: Manual verification + final regression check

**Files:**
- None (verification only)

- [ ] **Step 1: Run the full Python test suite**

```bash
cd perovskite-sim && pytest -m 'not slow'
```

Expected: green. Specifically confirm the existing TMM unit + integration tests still pass.

- [ ] **Step 2: Run the slow regression suite**

```bash
cd perovskite-sim && pytest -m slow
```

Expected: green. Phase 2a `tests/regression/test_tmm_baseline.py` must still pass — confirms zero electrical-grid or solver regression.

- [ ] **Step 3: Run the full frontend test suite**

```bash
cd perovskite-sim/frontend && npx vitest run
```

Expected: green.

- [ ] **Step 4: Production build**

```bash
cd perovskite-sim/frontend && npm run build
```

Expected: tsc clean + bundle written.

- [ ] **Step 5: Manual checklist (browser)**

Start backend + frontend dev servers and walk through:

```bash
cd perovskite-sim && uvicorn backend.main:app --host 127.0.0.1 --port 8000 --app-dir perovskite-sim --reload &
cd perovskite-sim/frontend && npm run dev
```

Open `http://127.0.0.1:5173/` and verify each item below. Any failure means a code fix, not a checklist edit.

  - [ ] In **full tier**, the visualizer + detail editor renders for `nip_MAPbI3_tmm`.
  - [ ] In **fast** tier, the existing accordion editor renders unchanged (regression check vs Phase 2a).
  - [ ] In **legacy** tier, ditto.
  - [ ] All 6 layers of `nip_MAPbI3_tmm` show in the visualizer with role colors and (log) heights; thinnest layers remain readable.
  - [ ] All 6 layers of `pin_MAPbI3_tmm` show with the inverse role order — confirms p-i-n / n-i-p symmetry.
  - [ ] Dragging a layer reorders it; running the J–V sweep returns a result that reflects the change.
  - [ ] ↑↓ buttons reorder via keyboard.
  - [ ] Adding a "TiO2_ETL" template inserts a layer with sensible defaults; running the sweep produces a finite J_sc.
  - [ ] Adding "spiro_HTL" template, similar check.
  - [ ] Deleting the absorber disables Run with the expected error banner; restoring it re-enables Run.
  - [ ] Editing any field shows the `● modified` pill; Reset reverts and clears the pill.
  - [ ] Save-As "phase2b_test_a" persists, shows under `User presets`, survives reload.
  - [ ] Save-As "nip_MAPbI3_tmm" returns the inline 409 error in the dialog.
  - [ ] Save-As "phase2b_test_a" again warns "already exists"; clicking Overwrite succeeds.
  - [ ] YAML download produces a file the backend can re-load via the existing `/api/configs` flow (manually copy to `configs/user/`).
  - [ ] Below ~1100 px viewport width, layout collapses to single column without overlap.
  - [ ] WCAG-AA contrast pass on all role color / text combinations, especially on small subscript characters.

- [ ] **Step 6: Clean up the test user preset**

```bash
rm -f perovskite-sim/configs/user/phase2b_test_a.yaml
rmdir perovskite-sim/configs/user 2>/dev/null || true
```

- [ ] **Step 7: Commit a `.gitignore` line for user presets**

Edit `perovskite-sim/.gitignore` (or `.gitignore` at the root, whichever applies) and add:

```
# Phase 2b user-saved device presets (per-developer; not version-controlled)
perovskite-sim/configs/user/
```

```bash
cd perovskite-sim && git add .gitignore
git commit -m "chore: ignore Phase 2b user-saved device presets"
```

- [ ] **Step 8: Final summary commit (optional)**

If any tweaks fell out of the manual pass, commit them as a final polish commit. Otherwise, the plan is complete.

---

## Self-review checklist (run after writing the plan)

- **Spec coverage** (cross-referenced against `2026-04-14-layer-builder-ui-design.md`):
  - §3 Architecture → Tasks 4, 20, 21
  - §4 Stack visualizer → Tasks 13, 14, 15, 16
  - §5 Add Layer dialog → Task 17
  - §6 Save As dialog → Task 18
  - §7 YAML download → Task 21 (Step 1, `downloadYaml` helper)
  - §8 Detail editor changes → Task 19
  - §9 Layer template library → Task 3
  - §10 User-preset save endpoint → Tasks 1, 2, 5
  - §11 State management → Task 21
  - §12 Validation → Task 11
  - §13 Frontend file inventory → Tasks 6–21
  - §14 Backend file inventory → Tasks 1–5
  - §15 Tests → Tasks 1, 2, 3, 4, 5, 8, 9, 10, 11, 23
  - §16 Risks/mitigations → built into task implementations (CSS breakpoint, native DnD, snapshot test, O_EXCL, soft warnings, Readonly types)
  - §17 Documentation updates → Task 22
  - §18 Verification → Task 23

- **Placeholder scan:** none. Every code block is complete; every command has expected output.

- **Type consistency:** `validate`, `isDirty`, `reconcileInterfaces`, `logScaleHeight`, `StackAction`, `ValidationReport`, `LayerTemplate`, `ConfigEntry`, `mountStackVisualizer`, `openAddLayerDialog`, `openSaveAsDialog`, `validate_user_filename`, `is_shipped_name`, `write_user_config`, `list_user_configs` — all signatures locked in the type-contract section and used identically in every task.

- **Known small gap:** the "field-set superset test" for the shrunk `config-editor.ts` mentioned in the spec (§15a) is intentionally folded into the manual verification step (Task 23 manual checklist confirms identical fields render in fast vs full tier). Adding a JSDOM-based snapshot test is straightforward but adds setup overhead for a one-time check; if a reviewer wants it as automated, the task would be ~15 lines added to `frontend/src/stack/__tests__/`.
