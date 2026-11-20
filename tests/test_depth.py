"""Tests for the back-to-front drawing order."""

import numpy as np

from minigeo import Scene
from minigeo.depth import is_back_face, painter_order, view_direction
from minigeo.geometry import BaseBox, BaseCylinder, BaseLine
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
    scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]))
    scene.add(BaseCylinder([2, 0, 0], dimensions=[0.5, 1]))
    primitives, order = order_of(scene)
    assert sorted(order) == list(range(len(primitives)))


def test_small_object_on_a_large_table_is_drawn_after_the_tabletop():
    """Sorting by centers alone gets this wrong: the cup's center is behind the table's."""
    scene = Scene()
    table = scene.add(BaseBox([0, 0, 0], dimensions=[4, 2, 0.1], anchor="top"), fill="brown")
    scene.add(
        BaseCylinder(table.anchor("top back left") + [0.3, -0.3, 0], dimensions=[0.1, 0.3]), fill="blue"
    )
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
    scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]), fill="near")
    scene.add(BaseBox([-5, 5, 0], dimensions=[1, 1, 1]), fill="far")  # further from a viewer at azim -60
    primitives, order = order_of(scene)
    far = position_in_order(primitives, order, lambda p: p.style.fill == "far")
    near = position_in_order(primitives, order, lambda p: p.style.fill == "near")
    assert max(far) < min(near)


def test_labels_come_last():
    scene = Scene()
    scene.add(BaseLine([0, 0, 0], [1, 0, 0]), label="a")
    scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]), label="b")
    primitives, order = order_of(scene)
    assert all(isinstance(primitives[i], Label) for i in order[-2:])


def test_back_faces():
    scene = Scene()
    scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]))
    faces = [p for p in scene.primitives() if isinstance(p, Face)]
    from_above = np.array([0, 0, 1.0])
    hidden = [f for f in faces if is_back_face(f, from_above)]
    assert len(hidden) == 1  # only the bottom turns away from a viewer straight above
    np.testing.assert_allclose(hidden[0].points[:, 2], -0.5)
