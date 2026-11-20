"""Tests for the mesh of shapes: faces and edges as vertex indices, with outward normals."""

import numpy as np
import pytest

from minigeo.geometry import (
    BaseBox,
    BaseCircle,
    BaseCone,
    BaseCylinder,
    BasePolygon,
    BaseRectangle,
    Transform,
    face_normal,
)


def solids():
    return [
        BaseBox([1, 2, 3], dimensions=[2, 3, 4]),
        BaseCylinder([0, 0, 0], dimensions=[1, 2]),
        BaseCone([0, 0, 0], dimensions=[1, 0.5, 2]),
    ]


def assert_normals_point_outward(shape):
    for face in shape.faces:
        outward = face.mean(axis=0) - shape.center
        assert np.dot(face_normal(face), outward) > 0


# ---------------------------------------------------------------- face_normal


def test_face_normal_counter_clockwise_points_up():
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    np.testing.assert_allclose(face_normal(square), [0, 0, 1])
    np.testing.assert_allclose(face_normal(square[::-1]), [0, 0, -1])


def test_face_normal_of_degenerate_face_is_zero():
    np.testing.assert_allclose(face_normal([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), [0, 0, 0])


# ---------------------------------------------------------------- outward winding


@pytest.mark.parametrize("shape", solids(), ids=lambda s: type(s).__name__)
def test_solid_normals_point_outward(shape):
    assert_normals_point_outward(shape)


@pytest.mark.parametrize("shape", solids(), ids=lambda s: type(s).__name__)
def test_normals_stay_outward_after_rotation_and_scaling(shape):
    Transform().rotate([1, 2, 3], 40).scale([1, 2, 0.5]).translate([3, 0, 0]) @ shape
    assert_normals_point_outward(shape)


@pytest.mark.parametrize("shape", solids(), ids=lambda s: type(s).__name__)
def test_normals_stay_outward_after_mirroring(shape):
    Transform().scale([-1, 1, 1]) @ shape
    assert_normals_point_outward(shape)
    Transform().scale([1, 1, -1]) @ shape  # mirroring twice restores the original winding
    assert_normals_point_outward(shape)


def test_flat_shapes_face_up():
    for shape in [
        BaseRectangle([0, 0, 0], dimensions=[2, 1]),
        BaseCircle([0, 0, 0], dimensions=[1]),
        BasePolygon([0, 0, 0], [[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
    ]:
        (face,) = shape.faces
        np.testing.assert_allclose(face_normal(face), [0, 0, 1], atol=1e-12)


# ---------------------------------------------------------------- box


def test_box_face_names_match_face_positions():
    box = BaseBox([0, 0, 0], dimensions=[2, 2, 2])
    directions = {"bottom": [0, 0, -1], "top": [0, 0, 1], "front": [0, -1, 0],
                  "back": [0, 1, 0], "left": [-1, 0, 0], "right": [1, 0, 0]}  # fmt: skip
    for name, face in zip(BaseBox.FACE_NAMES, box.faces, strict=True):
        np.testing.assert_allclose(face.mean(axis=0), directions[name])
        np.testing.assert_allclose(face_normal(face), directions[name])


def test_box_has_twelve_edges_each_once():
    box = BaseBox([0, 0, 0], dimensions=[1, 1, 1])
    assert box.edge_indices.shape == (12, 2)
    assert len({frozenset(edge) for edge in box.edge_indices.tolist()}) == 12
    lengths = np.linalg.norm(box.edges[:, 1] - box.edges[:, 0], axis=1)
    np.testing.assert_allclose(lengths, 1)


def test_faces_are_world_coordinates():
    box = BaseBox([5, 0, 0], dimensions=[1, 1, 1])
    top = box.faces[BaseBox.FACE_NAMES.index("top")]
    np.testing.assert_allclose(top.mean(axis=0), [5, 0, 0.5])


# ---------------------------------------------------------------- round shapes


def test_cylinder_faces():
    cylinder = BaseCylinder([0, 0, 0], dimensions=[1, 2])
    n = len(cylinder.vertices) // 2
    assert len(cylinder.face_indices) == n + 2
    assert len(cylinder.side_face_indices) == n
    assert len(cylinder.bottom_face_indices[0]) == n


def test_circle_is_one_polygon_without_extra_center_point():
    circle = BaseCircle([0, 0, 0], dimensions=[1])
    assert circle.face_indices == [list(range(len(circle.vertices)))]


def test_polygon_edges_go_around():
    poly = BasePolygon([0, 0, 0], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert poly.edge_indices.tolist() == [[0, 1], [1, 2], [2, 0]]
