"""Details settled before 1.0: constructor arguments, transforms, styles, saving and edge cases."""

import numpy as np
import pytest

from minigeo import (
    Angle,
    Axis,
    Box,
    Group,
    Line,
    Mesh,
    Point,
    Polygon,
    Scene,
    Sphere,
    Style,
    Transform,
    Vector,
    read_meshes,
    write_mesh,
)
from minigeo.geometry.mesh import polygon_triangles, triangulate

ARROW = np.array([[0, 0, 0], [2, 0, 0], [1, 0.5, 0], [2, 1, 0], [0, 1, 0]], dtype=float)  # concave


# ---------------------------------------------------------------- points, lines, axes, vectors


def test_points_and_lines_store_floats():
    assert Point([1, 2, 3]).position.dtype == float
    assert Line([0, 0, 0], [1, 2, 3]).vertices.dtype == float


def test_line_ends_are_views_of_its_vertices():
    line = Line([0, 0, 0], [1, 0, 0])
    line.start[0] = -1
    line.end = [2, 0, 0]
    assert np.allclose(line.vertices, [[-1, 0, 0], [2, 0, 0]])
    line.vertices = [[0, 1, 0], [0, 2, 0]]
    assert np.allclose(line.start, [0, 1, 0]) and np.allclose(line.end, [0, 2, 0])


def test_label_is_given_to_the_scene_not_the_geometry():
    with pytest.raises(TypeError):
        Point([0, 0, 0], label="O")
    with pytest.raises(TypeError):
        Line([0, 0, 0], [1, 0, 0], label="a")


def test_axis_takes_origin_like_a_vector():
    axis = Axis([0, 0, 2], origin=[1, 0, 0], length=3)
    assert np.allclose(axis.origin, [1, 0, 0]) and np.allclose(axis.direction, [0, 0, 1])
    Transform().translate([0, 1, 0]) @ axis
    assert np.allclose(axis.origin, [1, 1, 0])
    with pytest.raises(TypeError):
        Axis([0, 0, 1], position=[1, 0, 0])


def test_vectors_add_from_either_side():
    v = Vector([1, 0, 0], origin=[5, 5, 5])
    for total in ([0, 1, 0] + v, np.array([0, 1, 0]) + v, v + [0, 1, 0]):
        assert isinstance(total, Vector)
        assert np.allclose(total.components, [1, 1, 0]) and np.allclose(total.origin, [5, 5, 5])
    difference = np.array([0, 1, 0]) - v
    assert isinstance(difference, Vector) and np.allclose(difference.components, [-1, 1, 0])


# ---------------------------------------------------------------- rotating about anchors


def test_rotate_about_an_anchor():
    box = Box([0, 0, 0], size=(2, 1, 1), anchor="bottom")
    box.rotate("y", 90, about="bottom right")
    assert np.allclose(box.anchor("bottom right"), [1, 0, 0])  # the pivot stays
    group = Group([Box([0, 0, 0], size=1)])
    group.rotate("z", 90, about="left")
    assert np.allclose(group.anchor("center"), [-0.5, 0.5, 0])
    with pytest.raises(ValueError):
        box.rotate("z", 10, about="nowhere")


# ---------------------------------------------------------------- transforms


def test_scale_takes_one_number():
    assert np.allclose(Transform().scale(2).matrix[:3, :3], 2 * np.eye(3))


def test_shear_in_all_six_ways():
    sheared = Transform().shear(yz=0.5, zy=0.25).apply_to_vertex(np.array([0, 1, 2]))
    assert np.allclose(sheared, [0, 2, 2.25])
    assert np.allclose(Transform().shear([0.6, 0, 0, 0]).matrix, Transform().shear(xy=0.6).matrix)
    assert np.allclose(Transform().shear([0, 0, 0, 0, 1, 0]).matrix, Transform().shear(yz=1).matrix)
    with pytest.raises(ValueError, match="4 factors"):
        Transform().shear([1, 2, 3])


def test_inverse_copy_and_matrix():
    t = Transform().rotate("z", 30).scale([1, 2, 3]).translate([1, 2, 3])
    assert np.allclose((t.inverse() @ t).matrix, np.eye(4))
    rigid = Transform().rotate("x", 40).translate([0, 1, 0])
    assert np.allclose((rigid @ rigid.inverse()).matrix, np.eye(4))
    copy = t.copy()
    copy.translate([1, 0, 0])
    assert not np.allclose(copy.matrix, t.matrix)
    assert np.allclose(Transform(t.matrix).matrix, t.matrix)
    with pytest.raises(ValueError, match="cannot be undone"):
        Transform().scale([1, 0, 1]).inverse()
    with pytest.raises(ValueError, match="4x4"):
        Transform(np.eye(3))


# ---------------------------------------------------------------- styles


def test_styles_check_colors_and_sizes_right_away():
    with pytest.raises(ValueError, match="fill: Unknown color 'tomatoe'"):
        Style(fill="tomatoe")
    with pytest.raises(ValueError, match="line_width"):
        Style(line_width=-1)
    with pytest.raises(ValueError, match="font_size"):
        Style(font_size=0)
    with pytest.raises(TypeError, match="stroke"):
        Style(stroke=(1, 0, 0))
    Style(fill="none", stroke="k", text_color="C3")  # all fine
    with pytest.raises(ValueError):
        Scene().add(Box([0, 0, 0], size=1), fill="nope")


def test_one_letter_colors_work_everywhere():
    from minigeo.colors import to_rgb

    assert to_rgb("k") == (0, 0, 0) and to_rgb("r") == (1, 0, 0)
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1), fill="r")
    assert "#ff0000" in str(scene.render("svg"))


# ---------------------------------------------------------------- saving


def test_save_with_an_explicit_backend(tmp_path):
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1))
    scene.save(tmp_path / "a.svg", backend="svg")
    scene.save(tmp_path / "b.svg", backend="matplotlib")
    assert (tmp_path / "b.svg").read_text().lstrip().startswith("<?xml")
    pytest.importorskip("PIL")
    scene.save(tmp_path / "c.jpg", backend="raytrace", width=40)
    assert (tmp_path / "c.jpg").read_bytes()[:2] == b"\xff\xd8"
    with pytest.raises(ValueError, match="svg backend cannot write .png"):
        scene.save(tmp_path / "d.png", backend="svg")
    with pytest.raises(ValueError, match="Unknown backend"):
        scene.save(tmp_path / "e.png", backend="povray")


# ---------------------------------------------------------------- edge cases


def test_angle_radius_follows_scaling():
    angle = Angle([0, 0, 0], [1, 0, 0], [0, 1, 0], radius=0.5)
    Transform().scale(2) @ angle
    assert angle.radius == pytest.approx(1)
    assert np.allclose(np.linalg.norm(angle.arc_points, axis=1), 1)
    Transform().rotate("z", 30) @ angle
    assert angle.radius == pytest.approx(1)


def test_tangent_hits_at_the_start_of_the_line():
    rng = np.random.default_rng(1)
    missed = 0
    for _ in range(200):  # the line starts where it touches: b = 0, and c is 0 up to rounding
        center, radius = rng.normal(size=3), rng.uniform(0.1, 3)
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        along = np.cross(normal, rng.normal(size=3))
        touch = center + radius * normal
        ball = Sphere(center, radius=radius)
        ball.rotate(rng.normal(size=3), rng.uniform(0, 360), about="center")
        missed += len(ball.intersect(Line(touch, touch + along))) == 0
    assert missed == 0


def test_concave_faces_are_cut_correctly():
    triangles = polygon_triangles(ARROW)
    area = sum(np.linalg.norm(np.cross(*(ARROW[t[1:]] - ARROW[t[0]]))) / 2 for t in triangles)
    assert area == pytest.approx(1.5)  # 2 x 1 minus the notch of 0.5
    _, from_faces = triangulate([[0, 1, 2, 3, 4]], ARROW)
    assert np.array_equal(from_faces, triangles)
    arrow = Polygon([0, 0, 0], ARROW)
    assert arrow.intersect(Line([1.8, 0.5, -1], [1.8, 0.5, 1])).shape == (0, 3)  # in the notch
    assert len(arrow.intersect(Line([0.5, 0.5, -1], [0.5, 0.5, 1]))) == 1


def test_concave_faces_in_stl_and_the_ray_tracer(tmp_path):
    write_mesh(tmp_path / "arrow.stl", Polygon([0, 0, 0], ARROW))
    data = (tmp_path / "arrow.stl").read_bytes()
    assert int.from_bytes(data[80:84], "little") == 3
    pytest.importorskip("PIL")
    scene = Scene()
    scene.add(Polygon([0, 0, 0], ARROW), fill="black")
    pixels = scene.render("raytrace", width=200, elev=90, azim=-90).pixels
    notch = pixels[pixels.shape[0] // 2, int(0.85 * pixels.shape[1])]
    assert notch[3] == 0  # the notch stays empty


def test_empty_group_has_no_vertices():
    assert Group().vertices.shape == (0, 3)


def test_frame_needs_a_shape():
    with pytest.raises(TypeError, match="frame=True"):
        Scene().add(Line([0, 0, 0], [1, 0, 0]), frame=True)


def test_obj_files_are_utf8(tmp_path):
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1), label="θ-Würfel")
    scene.save(tmp_path / "cube.obj")
    assert "o θ-Würfel" in (tmp_path / "cube.obj").read_text(encoding="utf-8")
    assert list(read_meshes(tmp_path / "cube.obj")) == ["θ-Würfel"]


def test_repr_says_when_sizes_are_as_created():
    box = Box([0, 0, 0], size=1)
    assert "scaled" not in repr(box)
    box.rotate("z", 30)
    assert "scaled" not in repr(box)
    Transform().scale([2, 1, 1]) @ box
    assert "as created; scaled since" in repr(box)


def test_mesh_triangles_cover_concave_faces():
    mesh = Mesh(ARROW, [[0, 1, 2, 3, 4]])
    assert len(mesh.triangles) == 3
    assert mesh.volume == 0  # open
