"""Picking anchors and measuring between them: the shared logic and the matplotlib tool."""

import warnings

import numpy as np
import pytest

from minigeo import (
    Angle,
    Box,
    Circle,
    Cone,
    Cylinder,
    Dimension,
    Group,
    Line,
    Point,
    Rectangle,
    Scene,
    Sphere,
)
from minigeo.measure import Measurement, Spot, nearest, pickable_points


def names(spots, prefix):
    return {spot.name.split(": ", 1)[1] for spot in spots if spot.name.startswith(prefix + ":")}


# ---------------------------------------------------------------- which points can be picked


def test_a_box_offers_all_27_anchors_at_the_right_places():
    scene = Scene()
    box = scene.add(Box([1, 2, 3], size=(2, 1, 0.5)), label="crate")
    box.rotate("z", 30, about="center")
    spots = pickable_points(scene)
    assert len(spots) == 27
    for spot in spots:
        anchor = spot.name.split(": ", 1)[1]
        assert np.allclose(spot.position, box.anchor(anchor))
    assert spots[0].name == "crate: center"  # the simplest names come first


def test_round_shapes_offer_only_anchors_on_them():
    scene = Scene()
    scene.add(Cylinder([0, 0, 0], radius=1, height=1))
    scene.add(Sphere([3, 0, 0], radius=1))
    scene.add(Cone([6, 0, 0], radius=1, height=1))
    spots = pickable_points(scene)
    cylinder = names(spots, "Cylinder 1")
    assert {"center", "top", "bottom", "front", "top front", "bottom right"} <= cylinder
    assert "top front left" not in cylinder and len(cylinder) == 15
    assert names(spots, "Sphere 1") == {"center", "top", "bottom", "left", "right", "front", "back"}
    assert names(spots, "Cone 1") == {
        "center",
        "top",
        "bottom",
        "bottom left",
        "bottom right",
        "bottom front",
        "bottom back",
    }


def test_flat_shapes_and_coinciding_anchors():
    scene = Scene()
    scene.add(Rectangle([0, 0, 0], size=1))
    scene.add(Circle([3, 0, 0], radius=1))
    spots = pickable_points(scene)
    assert len(names(spots, "Rectangle 1")) == 9  # top and bottom coincide with the middle
    assert names(spots, "Circle 1") == {"center", "left", "right", "front", "back"}


def test_lines_points_groups_and_annotations():
    scene = Scene()
    scene.add(Line([0, 0, 0], [2, 0, 0]))
    scene.add(Point([5, 5, 5]), label="tip")
    scene.add(Group([Box([0, 3, 0], size=1), Box([2, 3, 0], size=1)]), label="pair")
    scene.add(Angle([0, 0, 0], [1, 0, 0], [0, 1, 0]))
    scene.add(Dimension([0, 0, 0], [1, 0, 0]))
    spots = pickable_points(scene)
    assert names(spots, "Line 1") == {"start", "midway", "end"}
    assert [s.name for s in spots if s.name.startswith("tip")] == ["tip"]
    assert sum(s.name.startswith("pair/Box") for s in spots) == 54
    assert not any(s.name.startswith(("Angle", "Dimension")) for s in spots)


def test_nearest():
    spots = [Spot(np.array([0.0, 0, 0]), "a"), Spot(np.array([1.0, 0, 0]), "b")]
    assert nearest(spots, [0.9, 0.2, 0]).name == "b"
    with pytest.raises(ValueError):
        nearest([], [0, 0, 0])


# ---------------------------------------------------------------- measuring


def test_measurement_with_two_picks_and_a_restart():
    a, b, c = (
        Spot(np.array(p, dtype=float), n) for p, n in [([0, 0, 0], "a"), ([3, 4, 0], "b"), ([1, 1, 1], "c")]
    )
    measurement = Measurement(digits=2)
    assert measurement.distance is None and measurement.text() == "click an anchor"
    measurement.pick(a)
    assert measurement.distance is None and "second anchor" in measurement.text()
    measurement.pick(b)
    assert measurement.distance == pytest.approx(5)
    assert np.allclose(measurement.difference, [3, 4, 0])
    assert "|AB| = 5.00" in measurement.text() and "Δ = (3.00, 4.00, 0.00)" in measurement.text()
    measurement.pick(c)  # starts again
    assert [spot.name for spot in measurement.picks] == ["c"]
    measurement.clear()
    assert measurement.picks == []


# ---------------------------------------------------------------- the matplotlib tool

pytest.importorskip("matplotlib")


@pytest.fixture
def tool():
    import matplotlib

    matplotlib.use("Agg")
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1), fill="C0", label="cube")
    scene.add(Cylinder([2, 0, -0.5], radius=0.4, height=1), fill="C1")
    tool = scene.measure()
    tool.figure.canvas.draw()
    yield tool
    import matplotlib.pyplot as plt

    plt.close(tool.figure)


def mouse(tool, kind, spot_name=None, offset=(0, 0), button=1, at=None):
    from matplotlib.backend_bases import MouseEvent

    if at is None:
        spot = next(s for s in tool.spots if s.name == spot_name)
        at = tool._on_screen(spot.position[None])[0] + offset
    MouseEvent(kind, tool.figure.canvas, *at, button=button)._process()
    return at


def click(tool, name, offset=(0, 0)):
    at = mouse(tool, "button_press_event", name, offset)
    mouse(tool, "button_release_event", at=at)


def test_hover_shows_the_nearest_anchor(tool):
    mouse(tool, "motion_notify_event", "cube: top front right", offset=(3, -2))
    assert tool.hover_info.get_visible()
    assert tool.hover_info.get_text() == "cube: top front right  (0.500, -0.500, 0.500)"
    mouse(tool, "motion_notify_event", at=(2, 2))  # far from every anchor
    assert not tool.hover_info.get_visible()


def test_two_clicks_measure(tool):
    click(tool, "cube: top front right", offset=(2, 2))
    assert [s.name for s in tool.measurement.picks] == ["cube: top front right"]
    click(tool, "Cylinder 1: top")
    assert tool.measurement.distance == pytest.approx(np.sqrt(1.5**2 + 0.5**2))
    assert "|AB| = 1.581" in tool.info.get_text()
    assert len(tool._pick_artists) == 4  # the marks, the letters A and B, and the line
    click(tool, "cube: center")
    assert len(tool.measurement.picks) == 1


def test_dragging_turns_the_view_without_picking(tool):
    start = mouse(tool, "button_press_event", "cube: top")
    mouse(tool, "motion_notify_event", at=start + (40, 0))
    mouse(tool, "button_release_event", at=start + (40, 0))
    assert tool.measurement.picks == []


def test_releasing_outside_the_canvas_does_not_crash(tool):
    from matplotlib.backend_bases import MouseEvent

    mouse(tool, "button_press_event", "cube: top")
    release = MouseEvent("button_release_event", tool.figure.canvas, None, None, button=1)
    release._process()  # matplotlib sets x/y to None when the release is outside the canvas
    assert tool.measurement.picks == []


def test_measuring_the_same_axes_again_detaches_the_old_measurer(tool):
    old_dots, old_info, old_hover_info = tool.dots, tool.info, tool.hover_info

    scene = Scene()
    scene.add(Box([0, 0, 0], size=1), fill="C0", label="cube")
    new_tool = scene.measure(ax=tool.ax)

    assert new_tool is not tool
    assert tool._cids == []  # the old handlers were disconnected
    assert old_dots not in tool.ax.collections  # the old artists were removed, not left piling up
    assert old_info not in tool.ax.texts
    assert old_hover_info not in tool.ax.texts

    click(new_tool, "cube: top")
    assert tool.measurement.picks == []  # the detached old tool no longer reacts to clicks
    assert [s.name for s in new_tool.measurement.picks] == ["cube: top"]


def test_picking_from_code(tool):
    tool.pick("cube: top")
    tool.pick("cube: bottom")
    assert tool.measurement.distance == pytest.approx(1)
    with pytest.raises(KeyError, match="cube: top"):
        tool.pick("cube: up")
    tool.clear()
    assert tool.measurement.picks == [] and tool.info.get_text() == "click an anchor"
    assert "Measurer with" in repr(tool)


def test_scene_measure_checks_the_backend():
    with pytest.raises(ValueError, match="matplotlib"):
        Scene().measure(backend="plotly")


def test_inline_backend_warns(monkeypatch):
    from minigeo.backends import matplotlib as mpl_backend

    monkeypatch.setattr(mpl_backend, "_inline_backend", lambda: True)
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1))
    with pytest.warns(UserWarning, match="interactive"):
        tool = scene.measure()
    import matplotlib.pyplot as plt

    plt.close(tool.figure)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        monkeypatch.setattr(mpl_backend, "_inline_backend", lambda: False)
        plt.close(scene.measure().figure)
