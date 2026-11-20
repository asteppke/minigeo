"""Tests for shapes with a pose: their own perpendicular axes x', y', z'."""

import numpy as np
import pytest

from minigeo.geometry import (
    Box,
    Cylinder,
    Polygon,
    Transform,
    split_rotation,
)

X, Y, Z = np.eye(3)


def make_box(center=(0, 0, 0), size=(2, 1, 1)):
    return Box(list(center), size=list(size))


def assert_perpendicular_unit_axes(shape):
    np.testing.assert_allclose(shape.axes @ shape.axes.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(shape.axes) == pytest.approx(1.0)  # right-handed


def extent(shape):
    return shape.vertices.max(axis=0) - shape.vertices.min(axis=0)


# ---------------------------------------------------------------- creation


def test_axes_line_up_with_world_at_creation():
    box = make_box(center=(1, 2, 3))
    np.testing.assert_allclose(box.axes, np.eye(3))
    np.testing.assert_allclose(box.center, [1, 2, 3])


def test_local_vertices_are_around_the_local_origin():
    box = make_box(center=(5, 0, 0))
    np.testing.assert_allclose(box.local_vertices.mean(axis=0), [0, 0, 0])
    np.testing.assert_allclose(box.vertices.mean(axis=0), [5, 0, 0])


def test_polygon_vertices_are_relative_to_position():
    poly = Polygon([0, 0, 2], [[1, 1, 1], [1, 1, 0], [1, 0, 1]])
    np.testing.assert_allclose(poly.vertices, [[1, 1, 3], [1, 1, 2], [1, 0, 3]])


def test_polygon_without_vertices_raises():
    with pytest.raises(TypeError):
        Polygon([0, 0, 0])


# ---------------------------------------------------------------- rigid transforms move the pose


def test_axes_rotate_with_the_shape():
    box = make_box(center=(5, 0, 0))
    box.rotate("z", 90, about="center")
    x_axis, y_axis, z_axis = box.axes
    np.testing.assert_allclose(x_axis, Y, atol=1e-12)
    np.testing.assert_allclose(y_axis, -X, atol=1e-12)
    np.testing.assert_allclose(z_axis, Z, atol=1e-12)


def test_translation_moves_center_not_axes():
    box = make_box()
    Transform().translate([1, 2, 3]) @ box
    np.testing.assert_allclose(box.center, [1, 2, 3])
    np.testing.assert_allclose(box.axes, np.eye(3))


def test_rigid_transform_keeps_local_vertices_exactly():
    box = make_box()
    before = box.local_vertices
    for _ in range(100):
        box.rotate([1, 2, 3], 7, about=[1, 0, 0])
    np.testing.assert_array_equal(box.local_vertices, before)
    assert_perpendicular_unit_axes(box)


def test_rotate_about_own_axis_by_name():
    box = make_box()
    box.rotate("z", 90, about="center")
    # x' now points along world y; spinning around it keeps the long side on y
    box.rotate("x'", 90, about="center")
    np.testing.assert_allclose(extent(box), [1, 2, 1], atol=1e-12)
    np.testing.assert_allclose(box.axes[0], Y, atol=1e-12)
    np.testing.assert_allclose(box.axes[1], Z, atol=1e-12)


def test_rotate_about_unknown_local_axis_raises():
    with pytest.raises(ValueError):
        make_box().rotate("w'", 90)


def test_set_center_keeps_orientation():
    box = make_box()
    box.rotate("z", 30)
    axes = box.axes
    box.center = [3, 0, 0]
    np.testing.assert_allclose(box.center, [3, 0, 0])
    np.testing.assert_allclose(box.axes, axes)
    np.testing.assert_allclose(box.vertices.mean(axis=0), [3, 0, 0], atol=1e-12)


# ---------------------------------------------------------------- scale and shear deform the vertices


def test_scaling_changes_vertices_but_not_axes():
    box = make_box(center=(1, 0, 0))
    Transform().scale([2, 3, 4]) @ box
    np.testing.assert_allclose(extent(box), [4, 3, 4])
    np.testing.assert_allclose(box.center, [2, 0, 0])
    np.testing.assert_allclose(box.axes, np.eye(3), atol=1e-12)


def test_scaling_along_own_axes_keeps_rotated_axes():
    box = make_box()
    box.rotate("z", 30)
    axes = box.axes
    # stretch by 2 along the box's own x' axis, expressed as a world transform
    along_x_prime = Transform().rotate("z", -30).scale([2, 1, 1]).rotate("z", 30)
    along_x_prime @ box
    np.testing.assert_allclose(box.axes, axes, atol=1e-12)
    np.testing.assert_allclose(np.ptp(box.local_vertices, axis=0), [4, 1, 1], atol=1e-12)


def test_shear_keeps_axes_perpendicular():
    box = make_box()
    Transform().shear([0.5, 0, 0, 0]) @ box
    assert_perpendicular_unit_axes(box)


def test_world_vertices_are_exactly_the_transformed_vertices():
    """Whatever the split into pose and stretch, the world vertices must be right."""
    box = make_box(center=(1, 2, 3))
    box.rotate("x", 20)
    t = Transform().shear([0.3, 0.1, -0.2, 0.4]).scale([1, 2, -1]).rotate([1, 1, 0], 40).translate([0, 1, 0])
    expected = t.apply_to_vertices(box.vertices)
    t @ box
    np.testing.assert_allclose(box.vertices, expected, atol=1e-12)
    assert_perpendicular_unit_axes(box)


def test_mirroring_keeps_frame_right_handed():
    box = make_box()
    Transform().scale([-1, 1, 1]) @ box
    assert_perpendicular_unit_axes(box)
    np.testing.assert_allclose(extent(box), [2, 1, 1])


def test_set_world_vertices():
    box = make_box()
    box.rotate("z", 45)
    target = box.vertices + [1, 0, 0]
    box.vertices = target
    np.testing.assert_allclose(box.vertices, target, atol=1e-12)


# ---------------------------------------------------------------- split_rotation


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(3),
        np.diag([2.0, 3.0, 4.0]),
        np.diag([-1.0, 1.0, 1.0]),
        np.array([[1.0, 0.5, 0], [0, 1, 0], [0, 0, 1]]),
        Transform().rotate([1, 2, 3], 50).matrix[:3, :3] @ np.diag([1.0, 2.0, 0.5]),
    ],
)
def test_split_rotation_recombines(matrix):
    rotation, stretch = split_rotation(matrix)
    np.testing.assert_allclose(rotation @ stretch, matrix, atol=1e-12)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_split_rotation_of_pure_rotation():
    r = Transform().rotate([1, 2, 3], 50).matrix[:3, :3]
    rotation, stretch = split_rotation(r)
    np.testing.assert_allclose(rotation, r, atol=1e-12)
    np.testing.assert_allclose(stretch, np.eye(3), atol=1e-12)


# ---------------------------------------------------------------- cylinder keeps its placement


def test_cylinder_base_sits_at_position():
    cylinder = Cylinder([0, 0, 1], radius=1, height=2)
    np.testing.assert_allclose(cylinder.vertices[:, 2].min(), 1)
    np.testing.assert_allclose(cylinder.vertices[:, 2].max(), 3)
