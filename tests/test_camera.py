"""Tests for the camera shared by all backends."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402

from minigeo.camera import Camera  # noqa: E402

VIEWS = [(30, -60, "z"), (30, 45, "x"), (-20, 130, "y"), (89, 10, "z")]


@pytest.mark.parametrize(("elev", "azim", "vertical"), VIEWS)
def test_basis_is_orthonormal_and_right_handed(elev, azim, vertical):
    camera = Camera(elev, azim, vertical)
    basis = np.array([camera.right, camera.up, camera.toward_viewer])
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(basis) == pytest.approx(1.0)


def test_default_is_matplotlibs_default_view():
    assert (Camera().elev, Camera().azim, Camera().vertical_axis) == (30, -60, "z")


def test_up_follows_the_vertical_axis():
    np.testing.assert_allclose(Camera(0, 0, "z").up, [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(Camera(0, 0, "x").up, [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(Camera(0, 0, "y").up, [0, 1, 0], atol=1e-12)


def test_invalid_vertical_axis():
    with pytest.raises(ValueError):
        Camera(vertical_axis="w")


def test_project():
    camera = Camera(0, 0)  # looking from +x: y is right, z is up
    xy, depth = camera.project([[0, 1, 2], [5, 0, 0]])
    np.testing.assert_allclose(xy, [[1, 2], [0, 0]], atol=1e-12)
    assert depth[1] < depth[0]  # the point at x=5 is closer to the viewer


@pytest.mark.parametrize(("elev", "azim", "vertical"), VIEWS[:3])
def test_matches_matplotlib(elev, azim, vertical):
    """Right, up and toward the viewer must mean the same as in matplotlib's 3D axes."""
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim, vertical_axis=vertical)
    camera = Camera(elev, azim, vertical)

    def screen(point):
        x, y, z = proj3d.proj_transform(*point, ax.get_proj())
        return np.array([x, y, z])

    origin = screen([0, 0, 0])
    right, up, toward = (screen(0.3 * v) - origin for v in (camera.right, camera.up, camera.toward_viewer))
    assert right[0] > 0 and abs(right[1]) < 1e-9
    assert up[1] > 0 and abs(up[0]) < 1e-9
    assert np.abs(toward[:2]).max() < 1e-9 and toward[2] < 0  # matplotlib: smaller depth is closer
    plt.close(fig)
