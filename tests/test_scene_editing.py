"""Removing geometries from a scene, and layers."""

import pytest

from minigeo import Box, Dimension, Scene, Sphere
from minigeo.render import Face, Label, Path


def test_remove_takes_labels_and_frames_along():
    scene = Scene()
    box = scene.add(Box([0, 0, 0], size=1), label="box", frame=True)
    ball = scene.add(Sphere([2, 0, 0], radius=0.5))
    scene.label(box, "top", at="top")
    scene.remove(box)
    assert [item.geometry for item in scene.items] == [ball]
    assert scene.labels == [] and scene.frames == []
    with pytest.raises(ValueError, match="not in this scene"):
        scene.remove(box)


def test_layers_hide_and_show():
    scene = Scene()
    box = scene.add(Box([0, 0, 0], size=1), label="box")
    scene.add(Dimension([0, 0, 0], [1, 0, 0]), layer="dimensions")
    scene.label(box, "note", layer="notes")
    assert scene.layers == ["main", "dimensions", "notes"]

    scene.hide("dimensions", "notes")
    assert not any(isinstance(p, Path) for p in scene.primitives())
    assert [p.text for p in scene.primitives() if isinstance(p, Label)] == ["box"]
    assert "hidden: dimensions, notes" in repr(scene)
    scene.show("notes")
    assert {p.text for p in scene.primitives() if isinstance(p, Label)} == {"box", "note"}
    scene.show()
    assert scene.hidden == set()


def test_label_and_frame_go_on_the_item_layer():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1), label="box", frame=True, layer="parts")
    assert scene.layers == ["parts"]
    scene.hide("parts")
    assert scene.primitives() == []


def test_hidden_layers_stay_out_of_every_backend():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1))
    scene.add(Sphere([3, 0, 0], radius=0.5), fill="red", layer="extra")
    scene.hide("extra")
    assert "#ff0000" not in str(scene.render("svg")).lower() and "red" not in str(scene.render("svg"))
    assert all(isinstance(p, Face) for p in scene.primitives())
    pytest.importorskip("PIL")
    pixels = scene.render("raytrace", width=80).pixels
    assert not ((pixels[..., 0] > 200) & (pixels[..., 1] < 60)).any()


def test_clear_one_layer_or_everything():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1))
    scene.add(Dimension([0, 0, 0], [1, 0, 0]), layer="dimensions", label="d")
    scene.hide("dimensions")
    scene.clear("dimensions")
    assert scene.layers == ["main"] and scene.hidden == set() and scene.labels == []
    scene.clear()
    assert len(scene) == 0 and scene.layers == []


def test_unknown_layers_are_errors():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1))
    with pytest.raises(ValueError, match="No layer named 'dimension'; layers: main"):
        scene.hide("dimension")
    with pytest.raises(ValueError, match="No layer"):
        scene.clear("nothing")
