"""Tests for the back-to-front drawing order."""

import numpy as np

from minigeo import Scene
from minigeo.depth import is_back_face, painter_order, view_direction
from minigeo.geometry import Box, Cylinder, Line
from minigeo.render import Face, Label

FROM_FRONT_ABOVE = view_direction(elev=30, azim=-60)


def order_of(scene, toward_viewer=FROM_FRONT_ABOVE):
    primitives = scene.primitives()
    return primitives, painter_order(primitives, toward_viewer)


def position_in_order(primitives, order, predicate):
    return [rank for rank, index in enumerate(order) if predicate(primitives[index])]


def test_view_direction():
    np.testing.assert_allclose(view_direction(90, 0), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(view_direction(0, 0), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(view_direction(0, 90), [0, 1, 0], atol=1e-12)


def test_order_is_a_permutation():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=[1, 1, 1]))
    scene.add(Cylinder([2, 0, 0], radius=0.5, height=1))
    primitives, order = order_of(scene)
    assert sorted(order) == list(range(len(primitives)))


def test_small_object_on_a_large_table_is_drawn_after_the_tabletop():
    """Sorting by centers alone gets this wrong: the cup's center is behind the table's."""
    scene = Scene()
    table = scene.add(Box([0, 0, 0], size=[4, 2, 0.1], anchor="top"), fill="brown")
    scene.add(Cylinder(table.anchor("top back left") + [0.3, -0.3, 0], radius=0.1, height=0.3), fill="blue")
    primitives, order = order_of(scene)
    tabletop = position_in_order(
        primitives, order, lambda p: p.style.fill == "brown" and p.points[:, 2].min() > -1e-9
    )
    # the cup's bottom lies in the tabletop's plane but faces away, so it is never shown
    cup = position_in_order(
        primitives, order, lambda p: p.style.fill == "blue" and not is_back_face(p, FROM_FRONT_ABOVE)
    )
    assert max(tabletop) < min(cup)


def test_box_behind_is_drawn_first():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=[1, 1, 1]), fill="green")
    scene.add(Box([-5, 5, 0], size=[1, 1, 1]), fill="red")  # further from a viewer at azim -60
    primitives, order = order_of(scene)
    far = position_in_order(primitives, order, lambda p: p.style.fill == "red")
    near = position_in_order(primitives, order, lambda p: p.style.fill == "green")
    assert max(far) < min(near)


def test_labels_come_last():
    scene = Scene()
    scene.add(Line([0, 0, 0], [1, 0, 0]), label="a")
    scene.add(Box([0, 0, 0], size=[1, 1, 1]), label="b")
    primitives, order = order_of(scene)
    assert all(isinstance(primitives[i], Label) for i in order[-2:])


def test_back_faces():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=[1, 1, 1]))
    faces = [p for p in scene.primitives() if isinstance(p, Face)]
    from_above = np.array([0, 0, 1.0])
    hidden = [f for f in faces if is_back_face(f, from_above)]
    assert len(hidden) == 1  # only the bottom turns away from a viewer straight above
    np.testing.assert_allclose(hidden[0].points[:, 2], -0.5)


def test_on_top_paths_come_after_faces_but_before_labels():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=[1, 1, 1]), label="box", frame=True)
    primitives, order = order_of(scene)
    kinds = [
        "label"
        if isinstance(primitives[i], Label)
        else "top"
        if getattr(primitives[i], "on_top", False)
        else "solid"
        for i in order
    ]
    assert kinds == sorted(kinds, key=["solid", "top", "label"].index)
    assert kinds.count("top") == 3
