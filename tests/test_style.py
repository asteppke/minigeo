"""Tests for Style: combining, validation and color names."""

import dataclasses

import pytest

from minigeo.style import DEFAULT_STYLE, Style, color_value


def test_unset_fields_are_none():
    assert Style().fill is None
    assert not Style().is_complete
    assert DEFAULT_STYLE.is_complete


def test_or_lets_the_right_side_win_where_it_is_set():
    glass = Style(fill="lightblue", opacity=0.3)
    red_glass = glass | Style(fill="red")
    assert red_glass.fill == "red"
    assert red_glass.opacity == 0.3


def test_or_does_not_change_the_operands():
    a, b = Style(fill="red"), Style(stroke="blue")
    a | b
    assert a == Style(fill="red") and b == Style(stroke="blue")


def test_or_with_something_else_fails():
    with pytest.raises(TypeError):
        Style() | {"fill": "red"}


def test_with_changes_a_copy():
    style = Style(fill="red")
    assert style.with_(fill="blue").fill == "blue"
    assert style.fill == "red"


def test_style_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Style().fill = "red"


@pytest.mark.parametrize("bad", [{"dash": "wavy"}, {"opacity": 1.5}, {"opacity": -0.1}])
def test_invalid_values_raise(bad):
    with pytest.raises(ValueError):
        Style(**bad)


@pytest.mark.parametrize(
    ("name", "expected"), [("C0", "#1f77b4"), ("C3", "#d62728"), ("peru", "peru"), ("#123456", "#123456")]
)
def test_color_value(name, expected):
    assert color_value(name) == expected
