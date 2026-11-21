"""Polylines and curves given by a function."""

import numpy as np
import pytest

from minigeo import Curve, Frame, Group, Polyline, Scene, Transform
from minigeo.render import Path, to_primitives
from minigeo.style import DEFAULT_STYLE


def helix(samples=400):
    return Curve(lambda t: (np.cos(t), np.sin(t), 0.1 * t), t=(0, 2 * np.pi), samples=samples)


# ---------------------------------------------------------------- polylines


def test_polyline_length_and_anchors_by_length():
    path = Polyline([[0, 0, 0], [2, 0, 0], [2, 1, 0]])
    assert path.length == pytest.approx(3)
    assert np.allclose(path.anchor("start"), [0, 0, 0])
    assert np.allclose(path.anchor("end"), [2, 1, 0])
    assert np.allclose(path.anchor("midway"), [1.5, 0, 0])
    assert np.allclose(path.anchor(2 / 3), [2, 0, 0])
    assert np.allclose(path.anchor("near end"), [2, 0.25, 0])


def test_closed_polyline_returns_to_its_start():
    square = Polyline([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], closed=True)
    assert square.length == pytest.approx(4)
    assert np.allclose(square.path[-1], square.path[0])
    assert np.allclose(square.anchor("end"), [0, 0, 0])
    assert np.allclose(square.anchor(0.875), [0, 0.5, 0])
    assert "closed" in repr(square).lower()


def test_polyline_checks_its_points():
    with pytest.raises(ValueError, match="two points"):
        Polyline([[0, 0, 0]])
    with pytest.raises(ValueError, match="three coordinates"):
        Polyline([[0, 0], [1, 1]])


def test_polyline_with_repeated_points():
    path = Polyline([[0, 0, 0], [0, 0, 0]])
    assert path.length == 0 and np.allclose(path.anchor("midway"), [0, 0, 0])
    stuttering = Polyline([[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert np.allclose(stuttering.anchor("midway"), [1, 0, 0])


def test_polylines_move_and_turn():
    path = Polyline([[1, 0, 0], [2, 0, 0]])
    Transform().rotate("z", 90) @ path
    assert np.allclose(path.vertices, [[0, 1, 0], [0, 2, 0]])
    path.place("start", at=[5, 5, 5])
    assert np.allclose(path.anchor("start"), [5, 5, 5])


def test_polyline_draws_as_one_path():
    square = Polyline([[0, 0, 0], [1, 0, 0], [1, 1, 0]], closed=True)
    (path,) = to_primitives(square, DEFAULT_STYLE)
    assert isinstance(path, Path) and len(path.points) == 4


# ---------------------------------------------------------------- curves


def test_curve_samples_and_exact_points():
    curve = helix(samples=50)
    assert len(curve.vertices) == 50
    assert np.allclose(curve.vertices[0], [1, 0, 0]) and np.allclose(curve.vertices[-1], [1, 0, 0.2 * np.pi])
    assert np.allclose(curve.at(np.pi / 2), [0, 1, 0.05 * np.pi])
    assert curve.length == pytest.approx(2 * np.pi * np.sqrt(1 + 0.01), rel=1e-3)


def test_tangent_is_the_unit_direction_of_growing_t():
    curve = helix()
    expected = np.array([0, 1, 0.1]) / np.sqrt(1.01)
    assert np.allclose(curve.tangent_at(0), expected, atol=1e-6)
    standing = Curve(lambda t: (0, 0, 0), t=(0, 1))
    assert np.allclose(standing.tangent_at(0.5), 0)


def test_curves_remember_their_transforms():
    curve = helix()
    Transform().scale([2, 2, 2]).rotate("z", 90).translate([0, 0, 1]) @ curve
    assert np.allclose(curve.at(0), [0, 2, 1])
    assert np.allclose(curve.at(0), curve.vertices[0])
    assert np.allclose(curve.tangent_at(0), np.array([-1, 0, 0.1]) / np.sqrt(1.01), atol=1e-6)


def test_curves_follow_groups_and_frames():
    curve = helix()
    Group([curve]).rotate("x", 90)
    assert np.allclose(
        curve.at(np.pi / 2),
        curve.vertices[np.argmin(np.linalg.norm(curve.vertices - curve.at(np.pi / 2), axis=1))],
        atol=1e-2,
    )
    slide = Frame().prismatic("x")
    attached = slide.attach(helix())
    slide.value = 3
    assert np.allclose(attached.at(0), [4, 0, 0])


def test_curve_checks():
    with pytest.raises(ValueError, match="2 samples"):
        Curve(lambda t: (t, 0, 0), samples=1)
    with pytest.raises(ValueError, match="three coordinates"):
        Curve(lambda t: (t, 0))


def test_curves_in_every_backend():
    scene = Scene()
    scene.add(helix(), stroke="C3", arrow=True, label="helix")
    assert "<polyline" in str(scene.render("svg"))
    assert "Stealth" in scene.render("tikz").code
    pytest.importorskip("PIL")
    assert scene.render("raytrace", width=60).width == 60


def test_picking_on_curves_and_polylines():
    from minigeo.measure import pickable_points

    scene = Scene()
    scene.add(helix(), label="spring")
    scene.add(Polyline([[0, 0, 0], [1, 0, 0], [1, 1, 0]]))
    names = [spot.name for spot in pickable_points(scene)]
    assert names[:3] == ["spring: start", "spring: midway", "spring: end"]
    assert "Polyline 1: point 2" in names and "Polyline 1: midway" in names
