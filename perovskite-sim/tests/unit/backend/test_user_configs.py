"""Unit tests for backend.user_configs (Phase 2b layer builder)."""
import pytest
import yaml
from pathlib import Path

from backend.user_configs import (
    USER_FILENAME_RE,
    USER_CONFIGS_ROOT,
    is_shipped_name,
    list_user_configs,
    validate_user_filename,
    write_user_config,
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
        assert is_shipped_name("nip_MAPbI3")
        assert not is_shipped_name("nip_MAPbI3.yaml")  # caller passes bare names


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
