"""Regression tests for transforms and how shapes keep their state in sync."""

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
    Transform,
)
from minigeo.groups import GeometryGroup


def unit_box(center=(0, 0, 0)):
    return BaseBox(list(center), dimensions=[1, 1, 1])


# ---------------------------------------------------------------- Transform API


def test_matmul_returns_the_geometry():
    box = unit_box()
    result = Transform().translate([1, 0, 0]) @ box
    assert result is box


def test_apply_returns_the_geometry():
    box = unit_box()
    assert Transform().translate([1, 0, 0]).apply(box) is box


def test_matmul_composes_transforms_right_to_left():
    move = Transform().translate([1, 0, 0])
    turn = Transform().rotate_z(90)
    # (turn @ move) means: first move, then turn
    point = (turn @ move) @ BasePoint([0, 0, 0])
    np.testing.assert_allclose(point.position, [0, 1, 0], atol=1e-12)


def test_matmul_composition_does_not_modify_operands():
    a, b = Transform().translate([1, 0, 0]), Transform().scale([2, 2, 2])
    a_before, b_before = a.matrix.copy(), b.matrix.copy()
    a @ b
    np.testing.assert_array_equal(a.matrix, a_before)
    np.testing.assert_array_equal(b.matrix, b_before)


def test_apply_to_unsupported_type_raises():
    with pytest.raises(TypeError):
        Transform().apply(42)


@pytest.mark.parametrize("axis", [[0, 0, 1], (0, 0, 1), np.array([0, 0, 1])])
def test_rotate_accepts_any_array_like_axis(axis):
    expected = Transform().rotate_z(90).matrix
    np.testing.assert_allclose(Transform().rotate(axis, 90).matrix, expected, atol=1e-12)


def test_rotate_rejects_unknown_axis_name():
    with pytest.raises(ValueError):
        Transform().rotate("w", 90)


def test_shear_requires_four_factors():
    with pytest.raises(ValueError):
        Transform().shear([1, 0, 0])


def test_shear_xy_shifts_x_by_y():
    point = Transform().shear([0.5, 0, 0, 0]) @ BasePoint([0, 2, 0])
    np.testing.assert_allclose(point.position, [1, 2, 0])


# ---------------------------------------------------------------- centers stay in sync


def test_rotate_moves_shape_center():
    box = unit_box(center=(5, 0, 0))
    box.rotate("z", 90)
    np.testing.assert_allclose(box.center, [0, 5, 0], atol=1e-12)
    np.testing.assert_allclose(box.vertices.mean(axis=0), box.center, atol=1e-12)


def test_transform_moves_shape_center():
    box = unit_box()
    Transform().translate([1, 2, 3]) @ box
    np.testing.assert_allclose(box.center, [1, 2, 3])


def test_setting_center_moves_vertices():
    box = unit_box()
    box.center = [3, 0, 0]
    np.testing.assert_allclose(box.vertices.mean(axis=0), [3, 0, 0])


def test_center_of_point_and_line():
    np.testing.assert_allclose(BasePoint([1, 2, 3]).center, [1, 2, 3])
    np.testing.assert_allclose(BaseLine([0, 0, 0], [2, 0, 0]).center, [1, 0, 0])


def test_axis_origin_follows_translation():
    axis = BaseAxis([1, 0, 0])
    Transform().translate([0, 1, 0]) @ axis
    np.testing.assert_allclose(axis.origin, [0, 1, 0])
    np.testing.assert_allclose(axis.direction, [1, 0, 0])


def test_axis_direction_follows_rotation():
    axis = BaseAxis([1, 0, 0])
    Transform().rotate_z(90) @ axis
    np.testing.assert_allclose(axis.direction, [0, 1, 0], atol=1e-12)


# ---------------------------------------------------------------- groups


def test_group_accepts_tuple_and_can_add():
    group = GeometryGroup((unit_box(),))
    group.add(BasePoint([0, 0, 0]))
    assert len(group.geometries) == 2


def test_group_center_with_lines_and_points():
    group = GeometryGroup([BasePoint([0, 0, 0]), BaseLine([2, 0, 0], [4, 0, 0])])
    np.testing.assert_allclose(group.center, [1.5, 0, 0])


def test_group_transform_moves_member_centers():
    box = unit_box()
    group = GeometryGroup([box, BaseLine([0, 0, 0], [1, 0, 0])])
    Transform().translate([1, 0, 0]) @ group
    np.testing.assert_allclose(box.center, [1, 0, 0])


def test_group_rotate():
    point = BasePoint([1, 0, 0])
    GeometryGroup([point]).rotate("z", 90)
    np.testing.assert_allclose(point.position, [0, 1, 0], atol=1e-12)


# ---------------------------------------------------------------- round shapes


def test_circle_has_no_duplicate_vertex():
    circle = BaseCircle([0, 0, 0], dimensions=[1])
    assert len(np.unique(circle.vertices.round(9), axis=0)) == len(circle.vertices)


def test_cylinder_has_one_side_face_per_segment():
    cylinder = BaseCylinder([0, 0, 0], dimensions=(1, 2))
    n = len(cylinder.vertices) // 2
    assert len(cylinder.side_faces) == n
    for face in cylinder.side_faces:
        # every side quad has two bottom (z=0) and two top (z=2) corners
        assert sorted(face[:, 2]) == [0, 0, 2, 2]


def test_cylinder_top_and_bottom_faces():
    cylinder = BaseCylinder([0, 0, 0], dimensions=(1, 2))
    np.testing.assert_allclose(cylinder.bottom_faces[0][:, 2], 0)
    np.testing.assert_allclose(cylinder.top_faces[0][:, 2], 2)


def test_cylinder_edges_stay_within_a_ring():
    cylinder = BaseCylinder([0, 0, 0], dimensions=(1, 2))
    for start, end in cylinder.edges:
        assert start[2] == end[2]


def test_cone_repr_names_dimensions_correctly():
    text = repr(BaseCone([0, 0, 0], dimensions=(0.5, 0.2, 1)))
    assert "0.5 (bottom)" in text and "0.2 (top)" in text and "height 1" in text


# ---------------------------------------------------------------- rotating about a pivot


def test_transform_rotate_about_point():
    # spinning around the vertical line through (1, 0, 0)
    point = Transform().rotate("z", 90, about=[1, 0, 0]) @ BasePoint([2, 0, 0])
    np.testing.assert_allclose(point.position, [1, 1, 0], atol=1e-12)


def test_rotate_about_equals_explicit_sandwich():
    pivot = np.array([1.0, 2.0, 3.0])
    direction = [1, 1, 0]
    explicit = Transform().translate(-pivot).rotate(direction, 30).translate(pivot)
    shorthand = Transform().rotate(direction, 30, about=pivot)
    np.testing.assert_allclose(shorthand.matrix, explicit.matrix, atol=1e-12)


def test_rotate_about_chains_with_other_transforms():
    t = Transform().translate([0, 0, 5]).rotate("z", 90, about=[0, 0, 0]).scale([2, 2, 2])
    point = t @ BasePoint([1, 0, 0])
    np.testing.assert_allclose(point.position, [0, 2, 10], atol=1e-12)


def test_shape_rotate_about_center_keeps_center():
    box = BaseBox([5, 0, 0], dimensions=[2, 1, 1])
    box.rotate("z", 90, about="center")
    np.testing.assert_allclose(box.center, [5, 0, 0], atol=1e-12)
    # the long side now points along y
    extent = box.vertices.max(axis=0) - box.vertices.min(axis=0)
    np.testing.assert_allclose(extent, [1, 2, 1], atol=1e-12)


def test_shape_rotate_about_point():
    box = unit_box(center=(2, 0, 0))
    box.rotate("z", 180, about=[1, 0, 0])
    np.testing.assert_allclose(box.center, [0, 0, 0], atol=1e-12)


def test_shape_rotate_without_about_uses_origin():
    box = unit_box(center=(2, 0, 0))
    box.rotate("z", 180)
    np.testing.assert_allclose(box.center, [-2, 0, 0], atol=1e-12)


def test_line_rotate_about_center():
    line = BaseLine([0, 0, 0], [2, 0, 0])
    line.rotate("z", 90, about="center")
    np.testing.assert_allclose(line.vertices, [[1, -1, 0], [1, 1, 0]], atol=1e-12)


def test_rotate_about_unknown_name_raises():
    with pytest.raises(ValueError):
        unit_box().rotate("z", 90, about="corner")


def test_group_rotate_about_center_keeps_arrangement():
    a, b = BasePoint([1, 0, 0]), BasePoint([3, 0, 0])
    group = GeometryGroup([a, b])
    group.rotate("z", 90, about="center")
    np.testing.assert_allclose(a.position, [2, -1, 0], atol=1e-12)
    np.testing.assert_allclose(b.position, [2, 1, 0], atol=1e-12)
    np.testing.assert_allclose(group.center, [2, 0, 0], atol=1e-12)
