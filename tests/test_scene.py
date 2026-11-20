"""Tests for Scene and the conversion of geometry into primitives."""

import numpy as np
import pytest

from minigeo import Scene, Style
from minigeo.geometry import Axis, Box, Circle, Cylinder, Line, Point
from minigeo.groups import Group
from minigeo.render import Face, Label, Marker, Path, to_primitives
from minigeo.style import DEFAULT_STYLE


def unit_box(position=(0, 0, 0)):
    return Box(list(position), size=[1, 1, 1])


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
    scene.add(Axis([1, 0, 0]))
    scene.add(Axis([0, 1, 0]), arrow=False)
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
    cylinder = Cylinder([0, 0, 0], radius=1, height=2)
    faces = to_primitives(cylinder, DEFAULT_STYLE)
    side, bottom, top = faces[0], faces[-2], faces[-1]
    assert side.outline == ((0, 1), (2, 3))  # bottom and top side of the quad, no vertical seams
    assert bottom.fully_outlined and top.fully_outlined


def test_flat_shapes_are_not_closed():
    (face,) = to_primitives(Circle([0, 0, 0], radius=1), DEFAULT_STYLE)
    assert not face.closed


def test_lines_points_and_groups():
    scene = Scene()
    scene.add(Line([0, 0, 0], [1, 0, 0]))
    scene.add(Point([0, 0, 1]))
    scene.add(Group([unit_box(), unit_box((2, 0, 0))]), fill="red")
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
    scene.add(Line([0, 0, 0], [4, 0, 0]), label="x", label_at="near end")
    (label,) = of_type(scene.primitives(), Label)
    np.testing.assert_allclose(label.position, [3, 0, 0])


def test_unknown_backend():
    with pytest.raises(ValueError):
        Scene().render("povray")


# ---------------------------------------------------------------- face styles


def test_face_styles_by_name_and_index():
    scene = Scene()
    scene.add(
        unit_box(), fill="white", face_styles={"top": Style(fill="blue"), 2: Style(fill="red", opacity=0.5)}
    )
    faces = of_type(scene.primitives(), Face)
    fills = [f.style.fill for f in faces]
    assert fills[Box.FACE_NAMES.index("top")] == "blue"
    assert fills[2] == "red" and faces[2].style.opacity == 0.5
    assert fills.count("white") == 4


def test_face_styles_only_override_what_they_set():
    scene = Scene()
    scene.add(unit_box(), fill="white", stroke="black", face_styles={"top": Style(fill="blue")})
    top = of_type(scene.primitives(), Face)[Box.FACE_NAMES.index("top")]
    assert top.style.stroke == "black"


@pytest.mark.parametrize("bad", [{"lid": Style()}, {6: Style()}])
def test_unknown_faces_raise(bad):
    with pytest.raises(ValueError):
        Scene().add(unit_box(), face_styles=bad)


def test_face_styles_need_a_shape():
    with pytest.raises(TypeError):
        Scene().add(Line([0, 0, 0], [1, 0, 0]), face_styles={0: Style()})


# ---------------------------------------------------------------- labels


def test_several_labels_on_one_geometry_follow_it():
    scene = Scene()
    box = scene.add(unit_box())
    scene.label(box, "top", at="top")
    scene.label(box, "right", at="right", text_color="red")
    box.place("center", at=[10, 0, 0])
    labels = of_type(scene.primitives(), Label)
    np.testing.assert_allclose([label.position for label in labels], [[10, 0, 0.5], [10.5, 0, 0]])
    assert labels[1].style.text_color == "red"


def test_label_at_a_fixed_point():
    scene = Scene()
    scene.label([1, 2, 3], "here")
    (label,) = of_type(scene.primitives(), Label)
    np.testing.assert_allclose(label.position, [1, 2, 3])
    assert label.style.is_complete


# ---------------------------------------------------------------- display in Jupyter


def test_jupyter_shows_svg():
    scene = Scene()
    scene.add(unit_box())
    bundle = scene._repr_mimebundle_()
    assert bundle["image/svg+xml"].startswith("<svg")


# ---------------------------------------------------------------- coordinate frames


def frame_paths(scene):
    return [p for p in scene.primitives() if isinstance(p, Path) and p.style.arrow]


def test_frame_of_a_shape_follows_its_axes():
    scene = Scene()
    box = scene.add(Box([1, 2, 3], size=[2, 1, 0.5]), frame=True)
    box.rotate("z", 90, about="center")  # after adding: the frame must follow
    paths = frame_paths(scene)
    assert len(paths) == 3
    for path, axis in zip(paths, box.axes, strict=True):
        np.testing.assert_allclose(path.points[0], box.center)
        direction = path.points[1] - path.points[0]
        np.testing.assert_allclose(direction / np.linalg.norm(direction), axis, atol=1e-12)


def test_frame_sticks_out_of_the_shape():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=[2, 1, 0.5]), frame=True)
    lengths = [np.linalg.norm(p.points[1] - p.points[0]) for p in frame_paths(scene)]
    np.testing.assert_allclose(lengths, 1.5)  # 0.75 of the longest side; longer than every half side


def test_frame_colors_labels_and_style():
    scene = Scene()
    box = scene.add(Box([0, 0, 0], size=[1, 1, 1]))
    scene.add_frame(box, length=2, line_width=3)
    primitives = scene.primitives()
    paths = frame_paths(scene)
    assert [p.style.stroke for p in paths] == ["red", "green", "blue"]
    assert all(p.style.line_width == 3 and p.on_top for p in paths)
    labels = [p for p in primitives if isinstance(p, Label)]
    assert [label.text for label in labels] == ["x′", "y′", "z′"]
    assert [label.style.text_color for label in labels] == ["red", "green", "blue"]
    np.testing.assert_allclose(labels[2].position, [0, 0, 2])


def test_world_frame_at_origin_and_at_a_point():
    scene = Scene()
    scene.add_frame()
    scene.add_frame([1, 1, 1], length=0.5, labels=("a", "b", "c"), colors=("k", "k", "k"), on_top=False)
    first, second = frame_paths(scene)[:3], frame_paths(scene)[3:]
    np.testing.assert_allclose([p.points[1] for p in first], np.eye(3))
    np.testing.assert_allclose([p.points[1] for p in second], 1 + 0.5 * np.eye(3))
    assert not any(p.on_top for p in second)
    texts = [p.text for p in scene.primitives() if isinstance(p, Label)]
    assert texts == ["x", "y", "z", "a", "b", "c"]


def test_frame_needs_a_shape_or_a_point():
    with pytest.raises(TypeError):
        Scene().add(Line([0, 0, 0], [1, 0, 0]), frame=True)
    with pytest.raises(TypeError):
        Scene().add_frame([1, 2])
