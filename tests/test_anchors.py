"""Tests for TikZ-style anchors on shapes."""

import itertools

import numpy as np
import pytest

from minigeo.geometry import (
    BaseAxis,
    BaseBox,
    BaseCircle,
    BaseCone,
    BaseCylinder,
    BaseLine,
    BasePoint,
    BasePolygon,
    BaseRectangle,
    Transform,
    anchor_direction,
    line_position,
)
from minigeo.groups import GeometryGroup


def make_box(position=(0, 0, 0), **kwargs):
    # width (x) 2, length (y) 4, height (z) 6, so every side is easy to tell apart
    return BaseBox(list(position), dimensions=[2, 4, 6], **kwargs)


# ---------------------------------------------------------------- anchor names


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("center", [0, 0, 0]),
        ("right", [1, 0, 0]),
        ("left", [-1, 0, 0]),
        ("back", [0, 1, 0]),
        ("front", [0, -1, 0]),
        ("top", [0, 0, 1]),
        ("bottom", [0, 0, -1]),
        ("top front left", [-1, -1, 1]),
        ("bottom-back-right", [1, 1, -1]),
        ("Top_Front", [0, -1, 1]),
        ((0, 0, 0.5), [0, 0, 0.5]),
    ],
)
def test_anchor_direction(name, expected):
    np.testing.assert_array_equal(anchor_direction(name), expected)


def test_anchor_word_order_does_not_matter():
    for words in itertools.permutations(["top", "front", "left"]):
        np.testing.assert_array_equal(anchor_direction(" ".join(words)), [-1, -1, 1])


@pytest.mark.parametrize("bad", ["north", "top bottom", "left left", "", "center top", (1, 2)])
def test_invalid_anchor_names_raise(bad):
    with pytest.raises(ValueError):
        anchor_direction(bad)


# ---------------------------------------------------------------- anchors on a box


def test_box_face_centers():
    box = make_box()
    np.testing.assert_allclose(box.anchor("center"), [0, 0, 0])
    np.testing.assert_allclose(box.anchor("top"), [0, 0, 3])
    np.testing.assert_allclose(box.anchor("bottom"), [0, 0, -3])
    np.testing.assert_allclose(box.anchor("right"), [1, 0, 0])
    np.testing.assert_allclose(box.anchor("front"), [0, -2, 0])


def test_box_edges_and_corners():
    box = make_box()
    np.testing.assert_allclose(box.anchor("top front"), [0, -2, 3])
    np.testing.assert_allclose(box.anchor("top front left"), [-1, -2, 3])


def test_all_corner_anchors_are_vertices():
    box = make_box(position=(1, 2, 3))
    box.rotate([1, 1, 1], 25)
    corners = [
        box.anchor(f"{z} {y} {x}")
        for z in ("top", "bottom")
        for y in ("front", "back")
        for x in ("left", "right")
    ]
    for corner in corners:
        assert np.min(np.linalg.norm(box.vertices - corner, axis=1)) < 1e-12


def test_default_anchor_is_center():
    assert make_box().anchor() == pytest.approx([0, 0, 0])


def test_numeric_anchor_interpolates():
    np.testing.assert_allclose(make_box().anchor((0, 0, 0.5)), [0, 0, 1.5])


# ---------------------------------------------------------------- anchors move and turn with the shape


def test_anchor_follows_translation():
    box = make_box()
    Transform().translate([10, 0, 0]) @ box
    np.testing.assert_allclose(box.anchor("top"), [10, 0, 3])


def test_anchor_follows_rotation():
    box = make_box()
    box.rotate("x", 90)  # local top (z') now points along world -y
    np.testing.assert_allclose(box.anchor("top"), [0, -3, 0], atol=1e-12)


def test_anchor_follows_scaling():
    box = make_box()
    Transform().scale([1, 1, 2]) @ box
    np.testing.assert_allclose(box.anchor("top"), [0, 0, 6])


# ---------------------------------------------------------------- placing shapes by an anchor


def test_create_with_anchor():
    box = make_box(position=(0, 0, 0), anchor="bottom front left")
    np.testing.assert_allclose(box.anchor("bottom front left"), [0, 0, 0])
    np.testing.assert_allclose(box.center, [1, 2, 3])


def test_place_moves_without_turning():
    box = make_box()
    box.rotate("z", 30)
    axes = box.axes
    returned = box.place("bottom", at=[0, 0, 10])
    assert returned is box
    np.testing.assert_allclose(box.anchor("bottom"), [0, 0, 10], atol=1e-12)
    np.testing.assert_allclose(box.axes, axes)


def test_stack_shapes_with_anchors():
    table = BaseBox([0, 0, 0], dimensions=[2, 1, 0.1])
    cup = BaseCylinder(table.anchor("top"), dimensions=[0.1, 0.2])
    lamp = BaseBox([0, 0, 0], dimensions=[0.2, 0.2, 0.5]).place("bottom", at=table.anchor("top right"))
    np.testing.assert_allclose(cup.anchor("bottom"), [0, 0, 0.05])
    np.testing.assert_allclose(lamp.anchor("bottom"), [1, 0, 0.05])


def test_stack_on_a_tilted_shape():
    table = BaseBox([0, 0, 0], dimensions=[2, 1, 0.1])
    table.rotate("x", 30)
    book = BaseBox([0, 0, 0], dimensions=[0.3, 0.2, 0.05])
    book.rotate("x", 30)
    book.place("bottom", at=table.anchor("top"))
    np.testing.assert_allclose(book.anchor("bottom"), table.anchor("top"), atol=1e-12)
    np.testing.assert_allclose(book.axes, table.axes, atol=1e-12)


# ---------------------------------------------------------------- other shapes


def test_cylinder_stands_on_its_position_by_default():
    cylinder = BaseCylinder([0, 0, 1], dimensions=(0.5, 2))
    np.testing.assert_allclose(cylinder.anchor("bottom"), [0, 0, 1])
    np.testing.assert_allclose(cylinder.center, [0, 0, 2])  # center is the middle
    np.testing.assert_allclose(cylinder.anchor("top"), [0, 0, 3])


def test_cylinder_with_center_anchor():
    cylinder = BaseCylinder([0, 0, 0], dimensions=(0.5, 2), anchor="center")
    np.testing.assert_allclose(cylinder.center, [0, 0, 0])


def test_cylinder_side_anchor_lies_on_the_surface():
    cylinder = BaseCylinder([0, 0, 0], dimensions=(0.5, 2))
    np.testing.assert_allclose(cylinder.anchor("right"), [0.5, 0, 1])


def test_cone_stands_on_its_position_by_default():
    cone = BaseCone([0, 0, 0], dimensions=(0.5, 0.2, 1))
    np.testing.assert_allclose(cone.anchor("bottom"), [0, 0, 0])
    np.testing.assert_allclose(cone.anchor("top"), [0, 0, 1])


def test_flat_shapes():
    rectangle = BaseRectangle([0, 0, 0], dimensions=[2, 4])
    circle = BaseCircle([0, 0, 0], dimensions=[1])
    np.testing.assert_allclose(rectangle.anchor("back right"), [1, 2, 0])
    np.testing.assert_allclose(circle.anchor("top"), [0, 0, 0])  # flat: top == center
    np.testing.assert_allclose(circle.anchor("right"), [1, 0, 0])


def test_polygon_vertices_ignore_anchor_and_have_anchors():
    poly = BasePolygon([0, 0, 0], [[0, 0, 0], [2, 0, 0], [0, 2, 0]])
    np.testing.assert_allclose(poly.anchor("center"), [1, 1, 0])
    np.testing.assert_allclose(poly.anchor("back left"), [0, 2, 0])


# ---------------------------------------------------------------- lines


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("start", 0.0),
        ("very near start", 0.125),
        ("near start", 0.25),
        ("midway", 0.5),
        ("center", 0.5),
        ("near end", 0.75),
        ("very near end", 0.875),
        ("end", 1.0),
        ("Near-End", 0.75),
        (0.3, 0.3),
    ],
)
def test_line_position(name, expected):
    assert line_position(name) == expected


@pytest.mark.parametrize("bad", ["top", "nearly end", ""])
def test_invalid_line_anchor_raises(bad):
    with pytest.raises(ValueError):
        line_position(bad)


def test_line_anchors():
    line = BaseLine([0, 0, 0], [4, 0, 0])
    np.testing.assert_allclose(line.anchor("start"), [0, 0, 0])
    np.testing.assert_allclose(line.anchor("end"), [4, 0, 0])
    np.testing.assert_allclose(line.anchor(), [2, 0, 0])  # midway by default
    np.testing.assert_allclose(line.anchor("near end"), [3, 0, 0])
    np.testing.assert_allclose(line.anchor(0.1), [0.4, 0, 0])


def test_line_anchor_beyond_the_ends():
    line = BaseLine([0, 0, 0], [1, 0, 0])
    np.testing.assert_allclose(line.anchor(1.5), [1.5, 0, 0])  # like TikZ, pos may leave the segment


def test_line_anchors_follow_transforms():
    line = BaseLine([0, 0, 0], [2, 0, 0])
    line.rotate("z", 90)
    np.testing.assert_allclose(line.anchor("end"), [0, 2, 0], atol=1e-12)


def test_line_place():
    line = BaseLine([0, 0, 0], [1, 1, 0])
    returned = line.place("start", at=[5, 0, 0])
    assert returned is line
    np.testing.assert_allclose(line.vertices, [[5, 0, 0], [6, 1, 0]])


def test_line_from_a_shape_anchor_to_another():
    a = BaseBox([0, 0, 0], dimensions=[1, 1, 1])
    b = BaseBox([3, 0, 0], dimensions=[1, 1, 1])
    connector = BaseLine(a.anchor("right"), b.anchor("left"))
    np.testing.assert_allclose(connector.anchor("midway"), [1.5, 0, 0])


def test_axis_anchors():
    axis = BaseAxis([0, 0, 1], [1, 0, 0])
    np.testing.assert_allclose(axis.anchor("start"), [1, 0, 0])
    np.testing.assert_allclose(axis.anchor("end"), [1, 0, 1])
    np.testing.assert_allclose(axis.anchor("midway"), [1, 0, 0.5])


# ---------------------------------------------------------------- points and the generic fallback


def test_point_has_only_a_center_anchor():
    point = BasePoint([1, 2, 3])
    np.testing.assert_allclose(point.anchor(), [1, 2, 3])
    with pytest.raises(ValueError):
        point.anchor("top")


def test_point_place():
    point = BasePoint([1, 2, 3]).place("center", at=[0, 0, 0])
    np.testing.assert_allclose(point.position, [0, 0, 0])


# ---------------------------------------------------------------- groups


def make_group():
    # a 2 x 2 x 1 block from (0, 0, 0) to (2, 2, 1), plus a line sticking out to x = 3
    block = BaseBox([0, 0, 0], dimensions=[2, 2, 1], anchor="bottom front left")
    line = BaseLine([2, 1, 0.5], [3, 1, 0.5])
    return GeometryGroup([block, line]), block, line


def test_group_anchors_use_the_bounding_box():
    group, _, _ = make_group()
    np.testing.assert_allclose(group.anchor("bottom front left"), [0, 0, 0])
    np.testing.assert_allclose(group.anchor("top back right"), [3, 2, 1])
    np.testing.assert_allclose(group.anchor("top"), [1.5, 1, 1])
    np.testing.assert_allclose(group.anchor(), [1.5, 1, 0.5])


def test_group_center_is_the_center_anchor():
    group, _, _ = make_group()
    np.testing.assert_allclose(group.center, group.anchor("center"))
    np.testing.assert_allclose(group.center, [1.5, 1, 0.5])


def test_group_rotate_about_center_spins_in_place():
    group, _, _ = make_group()
    low, high = group.anchor("bottom front left"), group.anchor("top back right")
    group.rotate("z", 180, about="center")
    np.testing.assert_allclose(group.anchor("bottom front left"), low, atol=1e-12)
    np.testing.assert_allclose(group.anchor("top back right"), high, atol=1e-12)


def test_group_place_moves_all_members_together():
    group, block, line = make_group()
    returned = group.place("bottom front left", at=[10, 0, 0])
    assert returned is group
    np.testing.assert_allclose(block.anchor("bottom front left"), [10, 0, 0])
    np.testing.assert_allclose(line.anchor("end"), [13, 1, 0.5])


def test_group_anchor_refers_to_world_axes_after_rotation():
    group, _, _ = make_group()
    group.rotate("z", 90, about=[0, 0, 0])
    # the block now spans x in [-2, 0] and y in [0, 2]; the line reaches y = 3
    np.testing.assert_allclose(group.anchor("bottom front left"), [-2, 0, 0], atol=1e-12)
    np.testing.assert_allclose(group.anchor("top back right"), [0, 3, 1], atol=1e-12)


def test_nested_groups():
    inner, block, _ = make_group()
    outer = GeometryGroup([inner, BasePoint([-1, 0, 0])])
    np.testing.assert_allclose(outer.anchor("left"), [-1, 1, 0.5])
    outer.place("left", at=[0, 1, 0.5])
    np.testing.assert_allclose(block.anchor("bottom front left"), [1, 0, 0])


def test_empty_group_has_no_anchors():
    with pytest.raises(ValueError):
        GeometryGroup().anchor()


def test_stack_a_group_on_a_shape():
    table = BaseBox([0, 0, 0], dimensions=[2, 1, 0.1], anchor="bottom")
    lamp = GeometryGroup(
        [
            BaseCylinder([0, 0, 0], dimensions=[0.1, 0.4]),
            BaseCone([0, 0, 0.4], dimensions=[0.2, 0.1, 0.2]),
        ]
    )
    lamp.place("bottom", at=table.anchor("top"))
    np.testing.assert_allclose(lamp.anchor("bottom"), [0, 0, 0.1])
    np.testing.assert_allclose(lamp.anchor("top"), [0, 0, 0.7])
