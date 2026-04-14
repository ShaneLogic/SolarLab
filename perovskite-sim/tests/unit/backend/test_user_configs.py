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
        assert is_shipped_name("nip_MAPbI3")
        assert not is_shipped_name("nip_MAPbI3.yaml")  # caller passes bare names
