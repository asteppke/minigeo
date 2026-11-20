"""Tests for the plotly backend."""

import numpy as np
import pytest

from minigeo import Scene
from minigeo.camera import Camera
from minigeo.geometry import BaseAxis, BaseBox, BaseCylinder, BaseLine, BasePoint


def traces(figure, kind):
    return [t for t in figure.data if type(t).__name__ == kind]


def test_box_is_one_mesh_of_twelve_triangles():
    scene = Scene()
    scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]), fill="red")
    (mesh,) = traces(scene.render("plotly"), "Mesh3d")
    assert len(mesh.i) == 12  # 6 square faces, 2 triangles each
    assert set(mesh.facecolor) == {"red"}


def test_meshes_are_grouped_by_opacity():
    scene = Scene()
    scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]), fill="red")
    scene.add(BaseBox([2, 0, 0], dimensions=[1, 1, 1]), fill="C1")
    scene.add(BaseBox([4, 0, 0], dimensions=[1, 1, 1]), opacity=0.4)
    meshes = traces(scene.render("plotly"), "Mesh3d")
    assert sorted(m.opacity for m in meshes) == [0.4, 1.0]
    opaque = next(m for m in meshes if m.opacity == 1)
    assert set(opaque.facecolor) == {"red", "#ff7f0e"}


def test_shared_edges_are_drawn_once():
    scene = Scene()
    scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]), stroke="black")
    (outline,) = traces(scene.render("plotly"), "Scatter3d")
    assert list(outline.x).count(None) == 12  # 12 edges, each followed by a gap


def test_cylinder_outline_has_only_the_rings():
    scene = Scene()
    cylinder = scene.add(BaseCylinder([0, 0, 0], dimensions=[1, 2]), stroke="black")
    (outline,) = traces(scene.render("plotly"), "Scatter3d")
    assert list(outline.x).count(None) == len(cylinder.vertices)  # n edges per ring, 2 rings


def test_lines_arrows_points_and_labels():
    scene = Scene()
    scene.add(BaseLine([0, 0, 0], [1, 0, 0]), stroke="blue", dash="dashed", label="a", label_at="end")
    scene.add(BaseAxis([0, 0, 1]), stroke="green")
    scene.add(BasePoint([1, 1, 1]), fill="red")
    figure = scene.render("plotly")
    lines = [t for t in traces(figure, "Scatter3d") if t.mode == "lines"]
    assert {t.line.dash for t in lines} == {"dash", "solid"}
    (cone,) = traces(figure, "Cone")
    np.testing.assert_allclose([cone.u[0], cone.v[0], cone.w[0]], [0, 0, 1])
    assert cone.anchor == "tip"
    (marker,) = [t for t in traces(figure, "Scatter3d") if t.mode == "markers"]
    assert marker.marker.color == "red"
    (text,) = [t for t in traces(figure, "Scatter3d") if t.mode == "text"]
    assert text.text == ("a",)


@pytest.mark.parametrize(("elev", "azim", "vertical"), [(30, -60, "z"), (20, 45, "x")])
def test_camera_matches_the_other_backends(elev, azim, vertical):
    scene = Scene()
    scene.add(BaseBox([0, 0, 0], dimensions=[1, 2, 3]))
    layout = scene.render("plotly", elev=elev, azim=azim, vertical_axis=vertical).layout.scene
    eye = np.array([layout.camera.eye.x, layout.camera.eye.y, layout.camera.eye.z])
    np.testing.assert_allclose(eye / np.linalg.norm(eye), Camera(elev, azim, vertical).toward_viewer)
    up = [layout.camera.up.x, layout.camera.up.y, layout.camera.up.z]
    assert up[{"x": 0, "y": 1, "z": 2}[vertical]] == 1
    assert layout.camera.projection.type == "orthographic"
    assert layout.aspectmode == "data"


def test_hide_axes():
    scene = Scene()
    scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]))
    layout = scene.render("plotly", show_axes=False).layout.scene
    assert not layout.xaxis.visible and not layout.yaxis.visible and not layout.zaxis.visible


def test_empty_scene_and_html(tmp_path):
    figure = Scene().render("plotly")
    assert len(figure.data) == 0
    target = tmp_path / "scene.html"
    figure.write_html(target, include_plotlyjs="cdn")
    assert "plotly" in target.read_text()
