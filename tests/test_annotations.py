"""Tests for vectors, planes, angle arcs and dimension lines."""

import numpy as np
import pytest

from minigeo import Angle, Axis, Box, Dimension, Line, Plane, Polyline, Scene, Transform, Vector
from minigeo.render import Face, Label, Path

X, Y, Z = np.eye(3)


def primitives_of(geometry, **style):
    scene = Scene()
    scene.add(geometry, **style)
    return scene.primitives()


# ---------------------------------------------------------------- vectors


def test_vector_basics():
    v = Vector([3, 4, 0], origin=[1, 1, 1])
    np.testing.assert_allclose(v.components, [3, 4, 0])
    np.testing.assert_allclose(v.end, [4, 5, 1])
    assert v.length == 5
    np.testing.assert_allclose(v.direction, [0.6, 0.8, 0])
    np.testing.assert_allclose(Vector.between([1, 0, 0], [1, 2, 0]).components, [0, 2, 0])


def test_vector_algebra():
    a, b = Vector([1, 0, 0], origin=[0, 0, 5]), Vector([0, 2, 0])
    assert a.dot(b) == 0 and a.dot([2, 0, 0]) == 2
    cross = a.cross(b)
    np.testing.assert_allclose(cross.components, [0, 0, 2])
    np.testing.assert_allclose(cross.origin, [0, 0, 5])  # results start at the first vector's origin
    np.testing.assert_allclose((a + b).components, [1, 2, 0])
    np.testing.assert_allclose((a - b).components, [1, -2, 0])
    np.testing.assert_allclose((2 * a).components, [2, 0, 0])
    np.testing.assert_allclose((a * 2).components, [2, 0, 0])
    np.testing.assert_allclose((a / 2).components, [0.5, 0, 0])
    np.testing.assert_allclose((-a).components, [-1, 0, 0])


def test_angles_and_projections():
    a = Vector([1, 0, 0])
    assert a.angle_to([0, 1, 0]) == pytest.approx(90)
    assert a.angle_to([1, 1, 0], in_degrees=False) == pytest.approx(np.pi / 4)
    np.testing.assert_allclose(Vector([2, 3, 0]).projected_onto([5, 0, 0]).components, [2, 0, 0])
    with pytest.raises(ValueError):
        a.angle_to([0, 0, 0])
    with pytest.raises(ValueError):
        a.projected_onto([0, 0, 0])


def test_vector_is_an_arrow_with_label_at_the_tip():
    scene = Scene()
    scene.add(Vector([1, 0, 0]), label="v")
    path, label = scene.primitives()
    assert path.style.arrow
    np.testing.assert_allclose(label.position, [1, 0, 0])


def test_vector_moves_with_transforms():
    v = Vector([1, 0, 0])
    Transform().rotate("z", 90) @ v
    np.testing.assert_allclose(v.components, [0, 1, 0], atol=1e-12)


def test_vector_needs_three_components():
    with pytest.raises(ValueError):
        Vector([1, 2])


# ---------------------------------------------------------------- planes


def test_plane_orientation_and_distances():
    plane = Plane([0, 0, 1], normal=[0, 0, 5])
    np.testing.assert_allclose(plane.normal, Z)
    np.testing.assert_allclose(plane.point, [0, 0, 1])
    assert plane.distance_to([3, 4, 3]) == pytest.approx(2)
    assert plane.distance_to([0, 0, 0]) == pytest.approx(-1)
    np.testing.assert_allclose(plane.project([3, 4, 3]), [3, 4, 1])
    assert plane.contains([7, -2, 1]) and not plane.contains([0, 0, 0])


@pytest.mark.parametrize("normal", [[1, 0, 0], [0, 0, -1], [1, 2, 3], [0, 0, 1]])
def test_plane_axes_are_perpendicular_with_z_along_the_normal(normal):
    plane = Plane([0, 0, 0], normal=normal)
    np.testing.assert_allclose(plane.axes @ plane.axes.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(plane.normal, np.asarray(normal) / np.linalg.norm(normal), atol=1e-12)
    assert np.allclose(plane.vertices @ plane.normal, 0)  # the patch lies in the plane


def test_plane_through_three_points():
    plane = Plane.through([0, 0, 0], [1, 0, 0], [0, 1, 0])
    np.testing.assert_allclose(plane.normal, Z, atol=1e-12)
    for point in ([0, 0, 0], [1, 0, 0], [0, 1, 0]):
        assert plane.contains(point)
    with pytest.raises(ValueError):
        Plane.through([0, 0, 0], [1, 0, 0], [2, 0, 0])


def test_plane_line_intersection():
    plane = Plane([0, 0, 2], normal=Z)
    np.testing.assert_allclose(plane.intersect(Line([1, 1, 0], [1, 1, 1])), [[1, 1, 2]])  # beyond the segment
    assert plane.intersect(Line([1, 1, 0], [1, 1, 1]), segment=True).shape == (0, 3)
    np.testing.assert_allclose(plane.intersect(Vector([0, 0, 1], origin=[3, 0, 0])), [[3, 0, 2]])
    np.testing.assert_allclose(plane.intersect(([0, 0, 0], [1, 0, 4])), [[0.5, 0, 2]])
    np.testing.assert_allclose(plane.intersect(Axis([0, 0, 1], origin=[5, 5, 0])), [[5, 5, 2]])
    zigzag = Polyline([[0, 0, 0], [0, 0, 3], [1, 0, 3], [1, 0, 0]])
    np.testing.assert_allclose(plane.intersect(zigzag), [[0, 0, 2], [1, 0, 2]])
    np.testing.assert_allclose(plane.intersect(Line([9, 9, 0], [9, 9, 1])), [[9, 9, 2]])  # beyond the patch
    assert plane.intersect(Line([0, 0, 0], [1, 0, 0])).shape == (0, 3)  # parallel


def test_mirrored_plane_turns_its_normal():
    plane = Plane([0, 0, 1], normal=Z)
    Transform().scale([1, 1, -1]) @ plane
    np.testing.assert_allclose(plane.normal, -Z, atol=1e-12)
    np.testing.assert_allclose(plane.center, [0, 0, -1])
    assert plane.distance_to([0, 0, -3]) == pytest.approx(2)
    from minigeo.geometry import face_normal

    np.testing.assert_allclose(face_normal(plane.faces[0]), plane.normal, atol=1e-12)


def test_plane_normal_vector_and_default_look():
    plane = Plane([1, 2, 3], normal=[0, 1, 0], size=(2, 1))
    normal = plane.normal_vector(length=2)
    np.testing.assert_allclose(normal.origin, [1, 2, 3])
    np.testing.assert_allclose(normal.components, [0, 2, 0], atol=1e-12)
    (face,) = primitives_of(plane)
    assert face.style.opacity == 0.4 and not face.closed


def test_plane_needs_a_normal():
    with pytest.raises(ValueError):
        Plane([0, 0, 0], normal=[0, 0, 0])


# ---------------------------------------------------------------- angles


def test_angle_size_and_arc():
    angle = Angle([0, 0, 0], [2, 0, 0], [1, 1, 0], radius=0.5)
    assert angle.degrees == pytest.approx(45) and angle.radians == pytest.approx(np.pi / 4)
    arc = angle.arc_points
    np.testing.assert_allclose(np.linalg.norm(arc, axis=1), 0.5)
    np.testing.assert_allclose(arc[0], [0.5, 0, 0])
    np.testing.assert_allclose(arc[-1], [0.5 / np.sqrt(2), 0.5 / np.sqrt(2), 0])
    np.testing.assert_allclose(arc[:, 2], 0)  # in the plane of the two rays


def test_angle_anchors():
    angle = Angle([1, 0, 0], [2, 0, 0], [1, 1, 0], radius=0.4)
    np.testing.assert_allclose(angle.anchor("vertex"), [1, 0, 0])
    middle = angle.anchor("midway") - [1, 0, 0]
    np.testing.assert_allclose(middle, 0.4 * np.array([1, 1, 0]) / np.sqrt(2), atol=1e-12)
    assert np.linalg.norm(angle.anchor("label") - [1, 0, 0]) > 0.4  # outside the arc
    with pytest.raises(ValueError):
        angle.anchor("top")


def test_right_angle_gets_a_square_mark():
    angle = Angle([0, 0, 0], [1, 0, 0], [0, 3, 0], radius=0.5)
    assert angle.is_right and angle.shows_square
    (path,) = primitives_of(angle)
    assert len(path.points) == 3 and path.on_top
    plain = Angle([0, 0, 0], [1, 0, 0], [0, 3, 0], right_angle_mark=False)
    (arc,) = primitives_of(plain)
    assert len(arc.points) == plain.segments + 1


def test_obtuse_and_straight_angles():
    obtuse = Angle([0, 0, 0], [1, 0, 0], [-1, 1, 0])
    assert obtuse.degrees == pytest.approx(135)
    straight = Angle([0, 0, 0], [1, 0, 0], [-1, 0, 0], radius=1)
    assert straight.degrees == pytest.approx(180)
    np.testing.assert_allclose(np.linalg.norm(straight.arc_points, axis=1), 1)


def test_angle_between_vectors_and_fill():
    a, b = Vector([2, 0, 0], origin=[1, 1, 0]), Vector([0, 1, 1], origin=[5, 5, 5])
    angle = Angle.between(a, b, radius=0.5)
    np.testing.assert_allclose(angle.vertex, [1, 1, 0])
    assert angle.degrees == pytest.approx(90)
    face, path = primitives_of(angle, fill="khaki")
    assert isinstance(face, Face) and isinstance(path, Path)
    assert len(primitives_of(angle)) == 1  # no fill by default


def test_angle_label_is_centered_outside_the_arc():
    scene = Scene()
    angle = scene.add(Angle([0, 0, 0], [1, 0, 0], [1, 1, 0]), label="θ")
    (label,) = [p for p in scene.primitives() if isinstance(p, Label)]
    assert label.align == "center"
    np.testing.assert_allclose(label.position, angle.anchor("label"))


def test_angle_needs_rays():
    with pytest.raises(ValueError):
        Angle([0, 0, 0], [0, 0, 0], [1, 0, 0])


# ---------------------------------------------------------------- dimensions


def test_dimension_length_and_text():
    dim = Dimension([0, 0, 0], [2, 0, 0], offset=[0, -0.5, 0], unit=" m")
    assert dim.length == 2 and dim.text == "2 m"
    assert Dimension([0, 0, 0], [1 / 3, 0, 0]).text == "0.33"
    assert Dimension([0, 0, 0], [1 / 3, 0, 0], digits=4).text == "0.3333"
    assert Dimension([0, 0, 0], [1, 0, 0], text="a").text == "a"


def test_dimension_parts():
    dim = Dimension([0, 0, 0], [2, 0, 0], offset=[0, -0.5, 0])
    a, b = dim.line_points
    np.testing.assert_allclose(a, [0, -0.5, 0])
    np.testing.assert_allclose(b, [2, -0.5, 0])
    first, second = dim.extension_lines
    np.testing.assert_allclose(first[0], [0, 0, 0])
    assert first[1][1] < -0.5  # a little beyond the dimension line
    paths = [p for p in primitives_of(dim) if isinstance(p, Path)]
    assert len(paths) == 4 and all(p.on_top for p in paths)
    assert sum(p.style.arrow for p in paths) == 2  # arrows at both ends
    (label,) = [p for p in primitives_of(dim) if isinstance(p, Label)]
    assert label.text == "2" and label.align == "center"
    np.testing.assert_allclose(label.position, [1, -0.75, 0])  # beyond the line, away from the object


def test_dimension_without_offset_and_text_follows_scaling():
    dim = Dimension([0, 0, 0], [1, 0, 0])
    assert dim.extension_lines == []
    Transform().scale([3, 1, 1]) @ dim
    assert dim.text == "3"


def test_dimension_of_a_box_edge():
    box = Box([0, 0, 0], size=(2, 1, 0.5), anchor="bottom front left")
    dim = Dimension(box.anchor("bottom front left"), box.anchor("bottom front right"), offset=[0, -0.3, 0])
    assert dim.length == pytest.approx(2)


def test_dimension_needs_two_points():
    with pytest.raises(ValueError):
        Dimension([1, 1, 1], [1, 1, 1])


# ---------------------------------------------------------------- all backends


@pytest.mark.parametrize("backend", ["svg", "plotly", "matplotlib"])
def test_everything_renders(backend):
    import matplotlib

    matplotlib.use("Agg")
    scene = Scene()
    a = scene.add(Vector([2, 0, 0]), label="a")
    b = scene.add(Vector([0, 1, 0]), label="b")
    scene.add(Angle.between(a, b), label="90°")
    scene.add(Angle([0, 0, 0], [1, 0, 0], [1, 1, 1]), fill="khaki")
    scene.add(Plane([1, 0.5, 0], normal=[0, 0, 1], size=3))
    scene.add(Dimension([0, 0, 0], [2, 0, 0], offset=[0, -0.5, 0], unit=" m"))
    scene.render(backend)
