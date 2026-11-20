"""Tests for the SVG backend."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from minigeo import Scene
from minigeo.backends.svg import SvgDrawing
from minigeo.geometry import Axis, Box, Cylinder, Line, Point, Rectangle

NS = {"svg": "http://www.w3.org/2000/svg"}


def parse(drawing: SvgDrawing) -> ET.Element:
    return ET.fromstring(str(drawing))


def find(root, tag):
    return root.findall(f".//svg:{tag}", NS)


def one_box(**style):
    scene = Scene()
    scene.add(Box([0, 0, 0], size=[1, 1, 1]), **style)
    return scene


def test_render_returns_a_drawing_with_size():
    root = parse(one_box().render("svg", width=300))
    assert root.get("width") == "300"
    assert int(root.get("height")) > 0
    assert root.get("viewBox") == f"0 0 300 {root.get('height')}"


def test_opaque_box_shows_three_faces():
    root = parse(one_box(fill="red").render("svg", shading=False))
    faces = [p for p in find(root, "polygon") if p.get("fill") == "red"]
    assert len(faces) == 3


def test_transparent_box_shows_all_six_faces():
    root = parse(one_box(fill="red", opacity=0.5).render("svg", shading=False))
    faces = [p for p in find(root, "polygon") if p.get("fill") == "red"]
    assert len(faces) == 6
    assert all(p.get("fill-opacity") == "0.5" for p in faces)


def test_shading_adds_dark_overlays():
    root = parse(one_box(fill="red").render("svg"))
    overlays = [p for p in find(root, "polygon") if p.get("fill") == "black"]
    assert 1 <= len(overlays) <= 3


def test_colors_are_translated():
    root = parse(one_box(fill="C1").render("svg"))
    assert any(p.get("fill") == "#ff7f0e" for p in find(root, "polygon"))


def test_cylinder_outline_has_no_seams():
    scene = Scene()
    scene.add(Cylinder([0, 0, 0], radius=1, height=2), stroke="black")
    root = parse(scene.render("svg"))
    seam_free = [p for p in find(root, "polygon") if p.get("stroke") not in ("black",)]
    outlines = [p for p in find(root, "path") if p.get("stroke") == "black"]
    assert seam_free and outlines  # side faces are not outlined; their rings are drawn as paths


def test_lines_dashes_and_arrows():
    scene = Scene()
    scene.add(Line([0, 0, 0], [1, 0, 0]), stroke="blue", dash="dashed", line_width=2)
    scene.add(Axis([0, 0, 1]), stroke="blue")
    scene.add(Axis([0, 1, 0]), stroke="green")
    root = parse(scene.render("svg"))
    lines = find(root, "polyline")
    (dashed,) = [line for line in lines if line.get("stroke-dasharray")]
    arrows = [line for line in lines if line is not dashed]
    assert dashed.get("stroke-dasharray") == "12.0 8.0"  # scaled with the line width
    assert all(a.get("marker-end", "").startswith("url(#arrow-") for a in arrows)
    assert len(find(root, "marker")) == 2  # one arrow tip per color


def test_points_and_labels():
    scene = Scene()
    scene.add(Point([0, 0, 0]), fill="red")
    scene.add(Line([0, 0, 0], [1, 0, 0]), label="a < b", label_at="end", text_color="green")
    root = parse(scene.render("svg"))
    assert find(root, "circle")[0].get("fill") == "red"
    (text,) = find(root, "text")
    assert text.text == "a < b" and text.get("fill") == "green"


def test_labels_are_painted_last():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=[1, 1, 1]), label="box")
    root = parse(scene.render("svg"))
    assert list(root)[-1].tag.endswith("text")


def test_small_object_on_tabletop_is_painted_after_it():
    scene = Scene()
    tabletop = scene.add(Rectangle([0, 0, 0], size=[4, 2]), fill="brown")
    # further back than the tabletop's center, but well inside it on screen
    scene.add(Cylinder(tabletop.anchor("back left") + [1.0, -0.6, 0], radius=0.1, height=0.3), fill="blue")
    fills = [p.get("fill") for p in find(parse(scene.render("svg", shading=False)), "polygon")]
    assert fills.index("brown") < min(i for i, f in enumerate(fills) if f == "blue")


def test_view_options_change_the_picture():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=[1, 2, 3]))
    assert str(scene.render("svg")) != str(scene.render("svg", vertical_axis="x"))
    assert str(scene.render("svg")) != str(scene.render("svg", elev=10, azim=20))


def test_empty_scene():
    root = parse(Scene().render("svg"))
    assert root.tag.endswith("svg")


def test_jupyter_display_and_save(tmp_path):
    drawing = one_box().render("svg")
    assert drawing._repr_svg_().startswith("<svg")
    target = tmp_path / "box.svg"
    drawing.save(target)
    assert ET.parse(target).getroot().tag.endswith("svg")


def test_wide_scene_keeps_proportions():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=[10, 1, 1]))
    root = parse(scene.render("svg", width=500, elev=0, azim=-90))  # seen from the front
    width, height = int(root.get("width")), int(root.get("height"))
    assert width == 500 and height == pytest.approx(40 + 460 / 10, abs=2)
    np.testing.assert_allclose(width, 500)


def test_frame_arrows_are_painted_over_the_shape():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=[1, 1, 1]), frame=True)
    root = parse(scene.render("svg"))
    tags = [element.tag.split("}")[1] for element in root if not element.tag.endswith("defs")]
    last_polygon = max(i for i, tag in enumerate(tags) if tag == "polygon")
    arrows = [i for i, tag in enumerate(tags) if tag == "polyline"]
    assert len(arrows) == 3 and min(arrows) > last_polygon
