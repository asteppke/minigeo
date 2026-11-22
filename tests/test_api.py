"""Tests for the user-facing API: names, named sizes, copies, spheres, saving and old names."""

import numpy as np
import pytest

import minigeo
from minigeo import Box, Circle, Cone, Cylinder, Group, Line, Point, Rectangle, Scene, Sphere, Transform
from minigeo.geometry import face_normal

# ---------------------------------------------------------------- names and named sizes


def test_everything_needed_is_importable_from_minigeo():
    for name in ["Axis", "Box", "Circle", "Cone", "Cylinder", "Group", "Line", "Point", "Polygon",
                 "Rectangle", "Scene", "Sphere", "Style", "Transform"]:  # fmt: skip
        assert hasattr(minigeo, name), name


def test_named_sizes_are_kept_as_attributes():
    box = Box([0, 0, 0], size=(2, 1, 0.5))
    np.testing.assert_allclose(box.size, [2, 1, 0.5])
    cylinder = Cylinder([0, 0, 0], radius=0.5, height=2)
    assert (cylinder.radius, cylinder.height) == (0.5, 2)
    cone = Cone([0, 0, 0], radius=0.5, height=1, top_radius=0.2)
    assert (cone.radius, cone.top_radius, cone.height) == (0.5, 0.2, 1)
    assert Circle([0, 0, 0], radius=2).radius == 2
    np.testing.assert_allclose(Rectangle([0, 0, 0], size=(2, 1)).size, [2, 1])


def test_one_number_for_a_cube_or_a_square():
    np.testing.assert_allclose(Box([0, 0, 0], size=2).size, [2, 2, 2])
    np.testing.assert_allclose(Rectangle([0, 0, 0], size=3).size, [3, 3])


def test_sizes_are_positional_too():
    assert Cylinder([0, 0, 0], 0.5, 2).height == 2
    np.testing.assert_allclose(Box([0, 0, 0], (1, 2, 3)).size, [1, 2, 3])


@pytest.mark.parametrize(
    "make",
    [
        lambda: Box([0, 0, 0], size=(1, -1, 1)),
        lambda: Box([0, 0, 0], size=(1, 2)),
        lambda: Cylinder([0, 0, 0], radius=-1, height=1),
        lambda: Circle([0, 0, 0], radius=1, segments=2),
    ],
)
def test_invalid_sizes_raise(make):
    with pytest.raises(ValueError):
        make()


def test_reprs_use_the_new_names():
    assert repr(Box([0, 0, 0], size=1)).startswith("Box at")
    assert "radius 0.5 and height 2" in repr(Cylinder([0, 0, 0], radius=0.5, height=2))


# ---------------------------------------------------------------- segments


def test_segments_set_the_resolution_of_round_shapes():
    assert len(Circle([0, 0, 0], radius=1, segments=32).vertices) == 32
    assert len(Cylinder([0, 0, 0], radius=1, height=1, segments=8).side_face_indices) == 8
    sphere = Sphere([0, 0, 0], radius=1, segments=12)
    assert len(sphere.vertices) == 2 + (12 // 2 - 1) * 12  # poles plus the parallels


# ---------------------------------------------------------------- cones and spheres


def test_pointed_cone_has_no_top_face_and_no_top_outline():
    cone = Cone([0, 0, 0], radius=1, height=2)
    assert len(cone.face_indices) == cone.segments + 1  # sides and bottom
    assert len(cone.edge_indices) == cone.segments  # only the bottom ring
    np.testing.assert_allclose(cone.anchor("top"), [0, 0, 2])


def test_sphere_is_round_and_centered():
    sphere = Sphere([1, 2, 3], radius=2, segments=24)
    distances = np.linalg.norm(sphere.vertices - [1, 2, 3], axis=1)
    np.testing.assert_allclose(distances, 2)
    np.testing.assert_allclose(sphere.anchor("bottom"), [1, 2, 1])
    assert len(sphere.edge_indices) == 0  # no outline, only shading


def test_sphere_faces_point_outward_and_cover_it():
    sphere = Sphere([0, 0, 0], radius=1, segments=10)
    for face in sphere.faces:
        assert np.dot(face_normal(face), face.mean(axis=0)) > 0
    used = {i for face in sphere.face_indices for i in face}
    assert used == set(range(len(sphere.vertices)))


@pytest.mark.parametrize("backend", ["svg", "plotly", "matplotlib"])
def test_new_shapes_render_everywhere(backend):
    import matplotlib

    matplotlib.use("Agg")
    scene = Scene()
    scene.add(Sphere([0, 0, 0], radius=1), fill="C0")
    scene.add(Cone([3, 0, 0], radius=1, height=2), fill="C1")
    scene.render(backend)


# ---------------------------------------------------------------- copies


def test_copy_is_independent():
    original = Box([0, 0, 0], size=1)
    duplicate = original.copy()
    duplicate.rotate("z", 45, about="center")
    Transform().translate([5, 0, 0]) @ duplicate
    np.testing.assert_allclose(original.center, [0, 0, 0])
    np.testing.assert_allclose(original.axes, np.eye(3))
    assert type(duplicate) is Box and duplicate.size is not original.size


def test_copy_of_points_lines_and_groups():
    point, line = Point([1, 1, 1]), Line([0, 0, 0], [1, 0, 0])
    group = Group([point, line])
    duplicate = group.copy()
    duplicate.place("center", at=[10, 0, 0])
    np.testing.assert_allclose(point.position, [1, 1, 1])
    assert duplicate.geometries[0] is not point


# ---------------------------------------------------------------- saving


@pytest.mark.parametrize("suffix", [".svg", ".html", ".png", ".pdf"])
def test_save_chooses_the_backend_from_the_extension(tmp_path, suffix):
    import matplotlib

    matplotlib.use("Agg")
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1), fill="C1")
    target = tmp_path / f"box{suffix}"
    scene.save(target, elev=20)
    head = target.read_bytes()[:16].lower()
    expected = {".svg": b"<svg", ".html": b"<!doctype html", ".png": b"\x89png", ".pdf": b"%pdf"}[suffix]
    assert head.startswith(expected)


def test_save_unknown_extension():
    with pytest.raises(ValueError):
        Scene().save("scene.jpeg")


def test_old_names_are_gone():
    import minigeo.geometry as geo

    with pytest.raises(AttributeError):
        geo.BaseBox  # noqa: B018
