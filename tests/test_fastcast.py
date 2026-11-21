"""The optional numba engine of the ray tracer: the same results as the numpy engine, faster."""

import numpy as np
import pytest

from minigeo import Box, Cone, Cylinder, Rectangle, Scene, Sphere, Transform
from minigeo.backends import _fastcast, raytrace

needs_numba = pytest.mark.skipif(not _fastcast.available(), reason="numba is not installed")


def mixed_scene():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=(4, 4, 0.1), anchor="top"), fill="whitesmoke")
    scene.add(Box([-1, -1, 0.4], size=(0.6, 0.8, 0.8)), fill="C0", opacity=0.6)
    scene.add(Sphere([1, 0, 0.5], radius=0.5, segments=8), fill="C1")
    tilted = scene.add(Cylinder([0, 1, 0], radius=0.4, height=1, segments=7), fill="C2")
    Transform().scale([1, 0.5, 1]).rotate("x", 30, about=[0, 1, 0]) @ tilted
    scene.add(Cone([-1, 1, 0], radius=0.4, height=0.9, segments=20), fill="C3")
    scene.add(Rectangle([0, 0, 1.5], size=1), fill="gold", opacity=0.5)
    return scene


def both_engines(scene):
    contents = raytrace._Contents.collect(scene)
    return (
        raytrace._Surfaces(contents.surfaces, 5, "numpy"),
        raytrace._Surfaces(contents.surfaces, 5, "numba"),
    )


@needs_numba
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parallel_rays_hit_the_same_surfaces(seed):
    rng = np.random.default_rng(seed)
    slow, fast = both_engines(mixed_scene())
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    origins = rng.uniform(-3, 3, size=(4000, 3)) - 6 * direction
    start = rng.uniform(0, 2, size=4000)
    a, b = slow.cast(origins, direction, start), fast.cast(origins, direction, start)
    assert np.allclose(a.distance, b.distance)
    hit = np.isfinite(a.distance)
    assert hit.mean() > 0.1
    agree = a.surface == b.surface
    assert agree.mean() > 0.995  # the rest hit two faces at their shared edge, at the same distance
    exact = np.isin(a.surface, slow.exact) & agree
    assert np.array_equal(a.part[exact], b.part[exact])  # the same part of the cylinder


@needs_numba
def test_rays_in_different_directions():
    rng = np.random.default_rng(5)
    slow, fast = both_engines(mixed_scene())
    origins = rng.uniform(-0.5, 0.5, size=(300, 3)) + [0, 0, 0.6]
    directions = rng.normal(size=(300, 3))
    distance, primitive, part = (
        fast._accelerator.cast(origins, directions, np.zeros(300))
        if fast._accelerator
        else (None, None, None)
    )
    if distance is None:  # build the tree on first use
        fast.cast(origins[:1], directions[0], np.zeros(1))
        distance, primitive, part = fast._accelerator.cast(origins, directions, np.zeros(300))
    for k in range(0, 300, 7):  # each ray on its own through the numpy engine
        single = slow.cast(origins[k : k + 1], directions[k] / np.linalg.norm(directions[k]), np.zeros(1))
        expected = single.distance[0] / np.linalg.norm(directions[k])  # numpy uses a unit direction
        assert np.isclose(distance[k], expected) or (np.isinf(distance[k]) and np.isinf(expected))


@needs_numba
@pytest.mark.parametrize("look", ["natural", "diagram"])
def test_pictures_are_the_same(look):
    scene = mixed_scene()
    slow = raytrace.render(scene, width=160, look=look, engine="numpy").pixels.astype(int)
    fast = raytrace.render(scene, width=160, look=look, engine="numba").pixels.astype(int)
    difference = np.abs(slow - fast).max(axis=2)
    # Rays exactly on an edge shared by two faces may pick either face (at the same distance),
    # which changes a few antialiased edge pixels a little; everything else is identical.
    assert (difference > 2).mean() < 0.01
    assert difference.mean() < 1  # out of 255


@needs_numba
def test_the_tree_holds_every_surface_in_nested_boxes():
    rng = np.random.default_rng(3)
    triangles = rng.uniform(-5, 5, size=(500, 1, 3)) + rng.uniform(-0.3, 0.3, size=(500, 3, 3))
    accelerator = _fastcast.Accelerator(triangles, [], [], [])
    low, high, left, right, first, size, axes, order = accelerator.tree
    assert sorted(order.tolist()) == list(range(500))
    leaves = size > 0
    assert size[leaves].sum() == 500
    for node in np.nonzero(~leaves)[0]:
        for child in (left[node], right[node]):
            assert np.all(low[child] >= low[node] - 1e-12) and np.all(high[child] <= high[node] + 1e-12)
    for node in np.nonzero(leaves)[0]:
        members = triangles[order[first[node] : first[node] + size[node]]]
        assert np.all(members.min(axis=1) >= low[node] - 1e-12) and np.all(
            members.max(axis=1) <= high[node] + 1e-12
        )


@needs_numba
def test_empty_and_edge_cases():
    empty = _fastcast.Accelerator(np.zeros((0, 3, 3)), [], [], [])
    distance, primitive, _ = empty.cast(np.zeros((3, 3)), np.array([0, 0, 1.0]), np.zeros(3))
    assert np.isinf(distance).all() and (primitive == -1).all()
    one = _fastcast.Accelerator(np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float), [], [], [])
    distance, primitive, _ = one.cast(np.array([[0.2, 0.2, 1.0], [0.2, 0.2, 1.0]]), np.array([0, 0, -1.0]),
                                       np.array([0.0, 2.0]))  # fmt: skip
    assert distance[0] == pytest.approx(1) and primitive[0] == 0
    assert np.isinf(distance[1])  # the hit lies before the start
    edge_on = one.cast(np.array([[0.2, -1.0, 0.0]]), np.array([0, 1.0, 0]), np.zeros(1))
    assert np.isinf(edge_on[0][0])


def test_engine_choice(monkeypatch):
    with pytest.raises(ValueError, match="Unknown engine"):
        raytrace.render(Scene(), engine="cuda")
    monkeypatch.setattr(_fastcast, "available", lambda: False)
    assert raytrace._resolve_engine("auto") == "numpy"
    with pytest.raises(ImportError, match="minigeo\\[fast\\]"):
        raytrace.render(Scene(), engine="numba")
    monkeypatch.setattr(_fastcast, "numba", None)
    with pytest.raises(ImportError, match="numba"):
        _fastcast.Accelerator(np.zeros((0, 3, 3)), [], [], [])


@needs_numba
def test_engine_through_scene_and_animation():
    scene = Scene()
    box = scene.add(Box([0, 0, 0], size=1))
    assert scene.render("raytrace", width=40, engine="numba").width == 40
    movie = scene.animate(
        lambda i: box.rotate("z", 30), frames=2, backend="raytrace", width=40, engine="numba"
    )
    assert len(movie) == 2
