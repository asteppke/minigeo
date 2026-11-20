"""Tests for the matplotlib backend."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backend_bases import MouseEvent  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from minigeo import Scene  # noqa: E402
from minigeo.backends.matplotlib import to_png  # noqa: E402
from minigeo.geometry import BaseAxis, BaseBox, BaseCylinder, BaseLine, BasePoint  # noqa: E402


@pytest.fixture
def ax():
    fig = plt.figure()
    yield fig.add_subplot(projection="3d")
    plt.close(fig)


def small_scene():
    scene = Scene()
    box = scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]), fill="red", label="box", label_at="top")
    scene.add(BaseCylinder(box.anchor("top"), dimensions=[0.2, 0.5]), fill="C2")
    scene.add(BaseLine([0, 0, 0], [2, 0, 0]), stroke="blue", dash="dashed")
    scene.add(BaseAxis([0, 0, 1]))
    scene.add(BasePoint([1, 1, 1]))
    return scene


def test_render_returns_the_axes(ax):
    assert small_scene().render(ax=ax) is ax


def test_render_without_axes_makes_a_figure():
    ax = small_scene().render()
    assert ax.name == "3d"
    plt.close(ax.figure)


def test_render_needs_3d_axes():
    fig, flat = plt.subplots()
    with pytest.raises(ValueError):
        small_scene().render(ax=flat)
    plt.close(fig)


def test_one_collection_per_face(ax):
    scene = Scene()
    scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]), fill="red")
    scene.render(ax=ax)
    assert len(ax.collections) == 6
    red, green, blue, _ = to_rgba(ax.collections[0].get_facecolor()[0])
    assert red > 0 and green == blue == 0  # red, darkened by the shading


def test_opaque_back_faces_are_hidden_transparent_ones_not(ax):
    scene = Scene()
    scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]))
    scene.add(BaseBox([3, 0, 0], dimensions=[1, 1, 1]), opacity=0.5)
    scene.render(ax=ax)
    opaque, transparent = ax.collections[:6], ax.collections[6:]
    assert sum(c.get_visible() for c in opaque) == 3  # a box shows at most three faces
    assert all(c.get_visible() for c in transparent)


def test_labels_dashes_and_arrows(ax):
    small_scene().render(ax=ax)
    assert [t.get_text() for t in ax.texts] == ["box"]
    assert any(line.get_linestyle() == "--" for line in ax.lines)
    assert any(type(artist).__name__ == "_Arrow3D" for artist in ax.patches)


def test_releasing_the_mouse_after_rotating_resorts_the_faces(ax):
    small_scene().render(ax=ax)
    ax.figure.canvas.draw()
    before = [c.get_zorder() for c in ax.collections]
    visible_before = [c.get_visible() for c in ax.collections]

    ax.view_init(elev=-60, azim=120)  # look from below, from the other side
    x, y = ax.bbox.x0 + ax.bbox.width / 2, ax.bbox.y0 + ax.bbox.height / 2
    release = MouseEvent("button_release_event", ax.figure.canvas, x, y)
    ax.figure.canvas.callbacks.process("button_release_event", release)

    assert [c.get_zorder() for c in ax.collections] != before
    assert [c.get_visible() for c in ax.collections] != visible_before  # other faces turn away


def test_clean_view_without_axes(ax):
    small_scene().render(ax=ax, show_axes=False, elev=10, azim=20)
    assert not ax.axison
    assert ax.elev == 10 and ax.azim == 20


def test_png():
    assert to_png(small_scene(), size=2, dpi=50).startswith(b"\x89PNG")


@pytest.mark.parametrize("vertical", ["x", "y", "z"])
def test_equal_scale_for_every_vertical_axis(ax, vertical):
    from mpl_toolkits.mplot3d import proj3d

    scene = Scene()
    scene.add(BaseBox([0, 0, 0], dimensions=[1.7, 6.6, 4.8]))
    scene.render(ax=ax, vertical_axis=vertical)
    box_aspect = ax._roll_to_vertical(ax._box_aspect)  # what matplotlib uses internally
    world = proj3d.world_transformation(
        *ax.get_xlim3d(), *ax.get_ylim3d(), *ax.get_zlim3d(), pb_aspect=box_aspect
    )
    scales = np.diag(world)[:3]
    np.testing.assert_allclose(scales / scales[0], [1, 1, 1], rtol=1e-6)
