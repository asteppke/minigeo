"""Where lines meet shapes."""

import numpy as np
import pytest

from minigeo import (
    Axis,
    Box,
    Circle,
    Cone,
    Curve,
    Cylinder,
    Line,
    Mesh,
    Polyline,
    Rectangle,
    Sphere,
    Transform,
    Vector,
)
from minigeo.intersect import intersections


def test_a_line_through_a_box_enters_and_leaves():
    box = Box([0, 0, 0], size=1)
    assert np.allclose(
        box.intersect(Line([-1, 0.1, 0.2], [1, 0.1, 0.2])), [[-0.5, 0.1, 0.2], [0.5, 0.1, 0.2]]
    )
    # sorted along the line, also when it runs the other way
    assert np.allclose(box.intersect(Line([1, 0, 0], [-1, 0, 0])), [[0.5, 0, 0], [-0.5, 0, 0]])


def test_lines_are_infinite_unless_segment():
    box = Box([0, 0, 0], size=1)
    short = Line([-2, 0, 0], [0, 0, 0])
    assert len(box.intersect(short)) == 2
    assert np.allclose(box.intersect(short, segment=True), [[-0.5, 0, 0]])
    assert box.intersect(Line([-2, 0, 0], [-1, 0, 0]), segment=True).shape == (0, 3)


def test_misses_and_corners():
    box = Box([0, 0, 0], size=1)
    assert box.intersect(Line([-1, 0, 2], [1, 0, 2])).shape == (0, 3)
    # through two corners: each corner is shared by three faces, but counts once
    assert np.allclose(box.intersect(Line([-1, -1, -1], [1, 1, 1])), [[-0.5] * 3, [0.5] * 3])


def test_turned_and_stretched_boxes():
    box = Box([0, 0, 0], size=1)
    box.rotate("z", 45)
    reach = np.sqrt(0.5)
    assert np.allclose(box.intersect(([-2, 0, 0], [2, 0, 0])), [[-reach, 0, 0], [reach, 0, 0]])
    Transform().scale([1, 1, 4]) @ box
    assert np.allclose(box.intersect(Axis([0, 0, 1]))[:, 2], [-2, 2])


def test_spheres_are_met_exactly():
    ball = Sphere([1, 2, 3], radius=2, segments=6)  # few facets, still round
    hits = ball.intersect(Line([1, 2, 0], [1, 2, 1]))
    assert np.allclose(hits, [[1, 2, 1], [1, 2, 5]])
    touching = ball.intersect(Line([3, 0, 3], [3, 4, 3]))
    assert np.allclose(touching, [[3, 2, 3]])
    direction = np.array([1, 1, 0]) / np.sqrt(2)
    for point in ball.intersect(Line([1, 2, 3], [1 + direction[0], 2 + direction[1], 3])):
        assert np.linalg.norm(point - [1, 2, 3]) == pytest.approx(2)


def test_stretched_spheres_are_ellipsoids():
    ball = Sphere([0, 0, 0], radius=1)
    Transform().scale([3, 1, 1]) @ ball
    assert np.allclose(ball.intersect(Line([-5, 0, 0], [5, 0, 0])), [[-3, 0, 0], [3, 0, 0]])


def test_cylinders_are_met_on_their_side_and_caps():
    can = Cylinder([0, 0, 0], radius=1, height=2)  # stands on the origin
    assert np.allclose(can.intersect(Line([-3, 0, 1], [3, 0, 1])), [[-1, 0, 1], [1, 0, 1]])
    assert np.allclose(can.intersect(Line([0.5, 0, -1], [0.5, 0, 3])), [[0.5, 0, 0], [0.5, 0, 2]])
    slanted = can.intersect(Line([-2, 0, -1], [2, 0, 3]))  # in through the bottom, out through the side
    assert np.allclose(slanted, [[-1, 0, 0], [1, 0, 2]])
    assert can.intersect(Line([-3, 0, 2.5], [3, 0, 2.5])).shape == (0, 3)
    can.rotate("y", 90, about=[0, 0, 0])
    assert np.allclose(can.intersect(Line([1, 0, -3], [1, 0, 3])), [[1, 0, -1], [1, 0, 1]])


def test_flat_shapes_cones_and_meshes_use_their_faces():
    assert np.allclose(Circle([0, 0, 0], radius=1).intersect(Axis([0, 0, 1])), [[0, 0, 0]])
    assert Rectangle([0, 0, 0], size=1).intersect(Line([0, 0, 0], [1, 0, 0])).shape == (0, 3)  # in its plane
    cone = Cone([0, 0, 0], radius=1, height=1)
    assert np.allclose(cone.intersect(Axis([0, 0, 1]))[:, 2], [0, 1])
    tetrahedron = Mesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    )
    hits = tetrahedron.intersect(Line([0.2, 0.2, -1], [0.2, 0.2, 1]))
    assert np.allclose(hits, [[0.2, 0.2, 0], [0.2, 0.2, 0.6]])


def test_vectors_polylines_and_curves():
    box = Box([0, 0, 0], size=1)
    assert len(box.intersect(Vector([1, 0, 0]))) == 2  # extended like a line
    assert np.allclose(box.intersect(Vector([1, 0, 0]), segment=True), [[0.5, 0, 0]])
    zigzag = Polyline([[-1, 0, 0], [0, 0, 0], [0, 0, 2], [2, 0, 2]])
    assert np.allclose(box.intersect(zigzag), [[-0.5, 0, 0], [0, 0, 0.5]])
    helix = Curve(lambda t: (np.cos(t), np.sin(t), 0.2 * t), t=(-np.pi, np.pi), samples=400)
    hits = Cylinder([1, 0, -1], radius=0.3, height=2).intersect(helix)
    assert len(hits) == 2 and np.allclose(np.linalg.norm(hits[:, :2] - [1, 0], axis=1), 0.3, atol=1e-3)


def test_line_checks():
    with pytest.raises(ValueError, match="two different points"):
        intersections(([0, 0, 0], [0, 0, 0]), Box([0, 0, 0], size=1))
