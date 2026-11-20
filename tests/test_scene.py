"""Tests for Scene and the conversion of geometry into primitives."""

import numpy as np
import pytest

from minigeo import Scene, Style
from minigeo.geometry import BaseAxis, BaseBox, BaseCircle, BaseCylinder, BaseLine, BasePoint
from minigeo.groups import GeometryGroup
from minigeo.render import Face, Label, Marker, Path, to_primitives
from minigeo.style import DEFAULT_STYLE


def unit_box(position=(0, 0, 0)):
    return BaseBox(list(position), dimensions=[1, 1, 1])


def of_type(primitives, kind):
    return [p for p in primitives if isinstance(p, kind)]


# ---------------------------------------------------------------- adding


def test_add_returns_the_geometry():
    scene = Scene()
    box = unit_box()
    assert scene.add(box) is box
    assert len(scene) == 1


def test_scene_keeps_references_so_later_moves_show():
    scene = Scene()
    box = scene.add(unit_box())
    box.place("center", at=[5, 0, 0])
    np.testing.assert_allclose(scene.bounds()[0], [4.5, -0.5, -0.5])


def test_empty_scene():
    scene = Scene()
    assert scene.primitives() == []
    np.testing.assert_allclose(scene.bounds()[0], [0, 0, 0])


# ---------------------------------------------------------------- styles


def test_style_precedence():
    scene = Scene(style=Style(fill="blue", opacity=0.5))
    scene.add(unit_box(), Style(fill="green", stroke="black"), stroke="red")
    style = scene.resolved_style(scene.items[0])
    assert style.fill == "green"  # item style beats scene style
    assert style.stroke == "red"  # keyword beats the item's Style
    assert style.opacity == 0.5  # from the scene
    assert style.line_width == DEFAULT_STYLE.line_width  # from the defaults
    assert style.is_complete


def test_axes_are_arrows_by_default_but_can_be_changed():
    scene = Scene()
    scene.add(BaseAxis([1, 0, 0]))
    scene.add(BaseAxis([0, 1, 0]), arrow=False)
    first, second = scene.primitives()
    assert first.style.arrow and not second.style.arrow


def test_unknown_style_keyword_fails():
    with pytest.raises(TypeError):
        Scene().add(unit_box(), colour="red")


# ---------------------------------------------------------------- primitives


def test_box_becomes_six_fully_outlined_closed_faces():
    faces = to_primitives(unit_box(), DEFAULT_STYLE)
    assert len(faces) == 6
    assert all(isinstance(f, Face) and f.fully_outlined and f.closed for f in faces)


def test_cylinder_outlines_only_its_rings():
    cylinder = BaseCylinder([0, 0, 0], dimensions=[1, 2])
    faces = to_primitives(cylinder, DEFAULT_STYLE)
    side, bottom, top = faces[0], faces[-2], faces[-1]
    assert side.outline == ((0, 1), (2, 3))  # bottom and top side of the quad, no vertical seams
    assert bottom.fully_outlined and top.fully_outlined


def test_flat_shapes_are_not_closed():
    (face,) = to_primitives(BaseCircle([0, 0, 0], dimensions=[1]), DEFAULT_STYLE)
    assert not face.closed


def test_lines_points_and_groups():
    scene = Scene()
    scene.add(BaseLine([0, 0, 0], [1, 0, 0]))
    scene.add(BasePoint([0, 0, 1]))
    scene.add(GeometryGroup([unit_box(), unit_box((2, 0, 0))]), fill="red")
    primitives = scene.primitives()
    assert len(of_type(primitives, Path)) == 1
    assert len(of_type(primitives, Marker)) == 1
    faces = of_type(primitives, Face)
    assert len(faces) == 12 and all(f.style.fill == "red" for f in faces)


def test_label_sits_at_its_anchor():
    scene = Scene()
    scene.add(unit_box(), label="box", label_at="top", text_color="red")
    (label,) = of_type(scene.primitives(), Label)
    assert label.text == "box"
    np.testing.assert_allclose(label.position, [0, 0, 0.5])
    assert label.style.text_color == "red"


def test_line_label_at_line_anchor():
    scene = Scene()
    scene.add(BaseLine([0, 0, 0], [4, 0, 0]), label="x", label_at="near end")
    (label,) = of_type(scene.primitives(), Label)
    np.testing.assert_allclose(label.position, [3, 0, 0])


def test_unknown_backend():
    with pytest.raises(ValueError):
        Scene().render("povray")
