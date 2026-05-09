from theme.runs import parse_segments


def test_plain_text_one_normal_segment():
    assert parse_segments("Hello") == [("Hello", "normal")]


def test_subscript_emits_sub_segment():
    assert parse_segments("V[sub]oc[/sub]") == [
        ("V", "normal"),
        ("oc", "sub"),
    ]


def test_superscript_emits_sup_segment():
    assert parse_segments("J[sup]2[/sup]") == [
        ("J", "normal"),
        ("2", "sup"),
    ]


def test_mixed_sub_and_sup_in_one_string():
    assert parse_segments("J[sub]sc[/sub] = J[sub]0[/sub] (e[sup]qV/kT[/sup] - 1)") == [
        ("J", "normal"),
        ("sc", "sub"),
        (" = J", "normal"),
        ("0", "sub"),
        (" (e", "normal"),
        ("qV/kT", "sup"),
        (" - 1)", "normal"),
    ]


def test_unmatched_tag_raises():
    import pytest
    with pytest.raises(ValueError, match="unmatched"):
        parse_segments("V[sub]oc")


def test_underscore_literal_raises():
    import pytest
    with pytest.raises(ValueError, match="underscore"):
        parse_segments("V_oc")


def test_caret_literal_raises():
    import pytest
    with pytest.raises(ValueError, match="caret"):
        parse_segments("J^2")
