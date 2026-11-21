"""Meshes, and reading and writing them as OBJ and STL files."""

import numpy as np
import pytest

from minigeo import (
    Box,
    Cylinder,
    Line,
    Mesh,
    Polyline,
    Scene,
    Sphere,
    Transform,
    read_mesh,
    read_meshes,
    write_mesh,
)
from minigeo.meshfiles import merge_corners

CUBE = Box([0, 0, 0], size=1)


def cube_triangles():
    quads = CUBE.face_indices
    return [[q[0], q[1], q[2]] for q in quads] + [[q[0], q[2], q[3]] for q in quads]


def tetrahedron():
    return Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])


# ---------------------------------------------------------------- the mesh


def test_a_mesh_keeps_its_world_points_and_is_centered_on_its_box():
    mesh = Mesh(CUBE.vertices + [3, 0, 1], cube_triangles())
    assert np.allclose(mesh.vertices, CUBE.vertices + [3, 0, 1])
    assert np.allclose(mesh.center, [3, 0, 1])
    assert np.allclose(mesh.anchor("top"), [3, 0, 1.5])
    assert "8 vertices and 12 faces" in repr(mesh)


def test_closed_meshes_and_their_volume():
    assert tetrahedron().is_closed and tetrahedron().volume == pytest.approx(1 / 6)
    cube = Mesh(CUBE.vertices, cube_triangles())
    assert cube.is_closed and cube.volume == pytest.approx(1)
    Transform().scale([2, 1, 1]) @ cube
    assert cube.volume == pytest.approx(2)


def test_faces_pointing_inward_are_turned_outward():
    inward = Mesh(CUBE.vertices, [face[::-1] for face in cube_triangles()])
    assert inward.face_indices == cube_triangles()
    assert inward.volume == pytest.approx(1)


def test_open_meshes():
    open_box = Mesh(CUBE.vertices, CUBE.face_indices[1:])  # no bottom
    assert not open_box.is_closed and open_box.volume == 0
    # two faces that run along their shared edge the same way do not close a mesh
    assert not Mesh(CUBE.vertices, CUBE.face_indices[:5] + [[5, 6, 2, 1]]).is_closed


def test_only_feature_edges_are_outlined():
    cube = Mesh(CUBE.vertices, cube_triangles())
    outline = {frozenset(edge) for edge in cube.edge_indices.tolist()}
    assert outline == {frozenset(edge) for edge in CUBE.edge_indices.tolist()}  # no diagonals
    assert len(Mesh(CUBE.vertices, cube_triangles(), crease=None).edge_indices) == 18

    flat = Mesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2], [0, 3, 2]])  # second one flipped
    assert len(flat.edge_indices) == 4  # the border, not the shared diagonal


def test_a_smooth_mesh_has_no_outline_across_it():
    sphere = Sphere([0, 0, 0], radius=1, segments=32)
    mesh = Mesh(sphere.vertices, sphere.face_indices)
    assert mesh.is_closed and len(mesh.edge_indices) == 0
    assert mesh.volume == pytest.approx(4 / 3 * np.pi, rel=0.02)


def test_mirrored_meshes_still_point_outward():
    mesh = tetrahedron()
    Transform().scale([-1, 1, 1]) @ mesh
    assert mesh.volume == pytest.approx(1 / 6)


def test_mesh_checks():
    with pytest.raises(ValueError, match="three coordinates"):
        Mesh([[0, 0]], [])
    with pytest.raises(ValueError, match="three corners"):
        Mesh([[0, 0, 0], [1, 0, 0]], [[0, 1]])
    with pytest.raises(ValueError, match="does not exist"):
        Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 3]])


def test_meshes_draw_and_can_be_picked():
    from minigeo.measure import pickable_points

    scene = Scene()
    scene.add(tetrahedron(), label="tet")
    assert "<polygon" in str(scene.render("svg"))
    assert "\\fill" in scene.render("tikz").code or "\\filldraw" in scene.render("tikz").code
    names = [spot.name for spot in pickable_points(scene)]
    assert "tet: center" in names and "tet: top front left" in names
    pytest.importorskip("PIL")
    assert scene.render("raytrace", width=40).width == 40


# ---------------------------------------------------------------- files


def test_merging_corners_of_separate_triangles():
    corners = CUBE.vertices[np.array(cube_triangles())]
    vertices, faces = merge_corners(corners + 1e-13)
    assert len(vertices) == 8 and len(faces) == 12
    flat = CUBE.vertices[np.array([[0, 1, 1]])]
    assert merge_corners(np.concatenate([corners, flat]))[1] == faces  # collapsed triangle dropped


def test_stl_round_trip(tmp_path):
    box, cylinder = Box([0, 0, 0], size=(2, 1, 1)), Cylinder([3, 0, 0], radius=0.5, height=1)
    write_mesh(tmp_path / "parts.stl", [box, cylinder])
    data = (tmp_path / "parts.stl").read_bytes()
    count = int.from_bytes(data[80:84], "little")
    assert len(data) == 84 + 50 * count and count == 12 + 4 * 16 - 4
    mesh = read_mesh(tmp_path / "parts.stl")
    assert mesh.is_closed  # two closed solids, merged corners
    assert len(mesh.vertices) == len(box.vertices) + len(cylinder.vertices)
    assert mesh.volume == pytest.approx(2 + 16 * 0.5 * 0.25 * np.sin(2 * np.pi / 16))
    assert np.allclose(mesh.anchor("bottom left"), [-1, 0, -0.5])


def test_text_stl(tmp_path):
    path = tmp_path / "tet.stl"
    triangles = tetrahedron().vertices[np.array(tetrahedron().face_indices)]
    lines = ["solid tet"]
    for a, b, c in triangles:
        lines += ["facet normal 0 0 0", "outer loop", *(f"vertex {x} {y} {z}" for x, y, z in (a, b, c))]
        lines += ["endloop", "endfacet"]
    path.write_text("\n".join([*lines, "endsolid tet"]))
    mesh = read_mesh(path)
    assert mesh.is_closed and mesh.volume == pytest.approx(1 / 6)


def test_obj_round_trip_with_names_colors_and_lines(tmp_path):
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1), fill="red", label="crate")
    scene.add(Box([2, 0, 0], size=1), fill="blue", opacity=0.5)
    scene.add(Polyline([[0, 0, 0], [1, 1, 1], [2, 0, 0]]))
    scene.add(Line([0, 0, 0], [0, 0, 2]))
    scene.save(tmp_path / "scene.obj")

    text = (tmp_path / "scene.obj").read_text()
    assert "mtllib scene.mtl" in text and "o crate" in text and "o Box_2" in text
    assert "l 17 18 19" in text and "l 20 21" in text
    material = (tmp_path / "scene.mtl").read_text()
    assert "Kd 1 0 0" in material and "Kd 0 0 1" in material and "d 0.5" in material

    parts = read_meshes(tmp_path / "scene.obj")
    assert list(parts) == ["crate", "Box_2"]
    assert np.allclose(parts["Box_2"].center, [2, 0, 0]) and len(parts["Box_2"].vertices) == 8
    assert parts["crate"].is_closed
    assert len(read_mesh(tmp_path / "scene.obj").face_indices) == 12


def test_obj_corner_forms_groups_and_negative_numbers(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text(
        "# a square and a triangle\n"
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\n"
        "g square\nf 1/1/1 2/1/1 3//1 4\n"
        "g triangle\nv 5 0 0\nv 6 0 0\nv 5 1 0\nf -3 -2 -1  # the last three\n"
    )
    parts = read_meshes(path)
    assert list(parts) == ["square", "triangle"]
    assert parts["square"].face_indices == [[0, 1, 2, 3]]
    assert np.allclose(parts["triangle"].vertices, [[5, 0, 0], [6, 0, 0], [5, 1, 0]])
    assert Mesh.read(path).face_indices == [[0, 1, 2, 3], [4, 5, 6]]


def test_file_errors(tmp_path):
    (tmp_path / "bad.obj").write_text("v 0 0 0\nf 1 2 3\n")
    with pytest.raises(ValueError, match="line 2"):
        read_mesh(tmp_path / "bad.obj")
    (tmp_path / "empty.obj").write_text("v 0 0 0\n")
    with pytest.raises(ValueError, match="no faces"):
        read_mesh(tmp_path / "empty.obj")
    (tmp_path / "junk.stl").write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="neither"):
        read_mesh(tmp_path / "junk.stl")
    with pytest.raises(ValueError, match="obj or .stl"):
        read_mesh(tmp_path / "model.ply")
    with pytest.raises(ValueError, match="no shapes"):
        write_mesh(tmp_path / "line.obj", Line([0, 0, 0], [1, 0, 0]))
    with pytest.raises(TypeError, match="no options"):
        Scene().save(tmp_path / "x.stl", width=3)
