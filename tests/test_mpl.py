"""Tests for the matplotlib backend."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from minigeo import mpl_geometry as mgeo  # noqa: E402
from minigeo.drawing_properties import DrawingProperty  # noqa: E402
from minigeo.geometry import Transform  # noqa: E402


@pytest.fixture
def ax():
    fig = plt.figure()
    yield fig.add_subplot(projection="3d")
    plt.close(fig)


def test_cylinder_update_keeps_side_faces_only(ax, monkeypatch):
    cylinder = mgeo.MplCylinder(ax, [0, 0, 0], (1, 1))
    received = []
    monkeypatch.setattr(cylinder.poly, "set_verts", received.append)
    Transform().translate([1, 0, 0]) @ cylinder
    # the side collection must not also receive the top and bottom faces
    assert len(received[0]) == len(cylinder.side_faces)


def test_group_does_not_draw_members_twice(ax):
    line = mgeo.MplLine(ax, [0, 0, 0], [1, 0, 0])
    n_lines = len(ax.lines)
    mgeo.MplGroup(ax, [line])
    assert len(ax.lines) == n_lines


def test_cube_with_default_colors(ax):
    cube = mgeo.MplCube(ax, [0, 0, 0], 1)
    assert cube.face_colors == ["red", "red", "blue", "blue", "orange", "orange"]


def test_drawing_property_repr():
    assert "alpha=0.5" in repr(DrawingProperty(alpha=0.5))


def test_cone_draws_and_transforms(ax):
    cone = mgeo.MplCone(ax, [0, 0, 0], lower_radius=0.5, upper_radius=0.2, height=1)
    Transform().translate([0, 0, 1]) @ cone
    np.testing.assert_allclose(cone.center, [0, 0, 1])
