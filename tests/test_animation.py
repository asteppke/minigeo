"""Tests for animations: snapshots, the plotly backend and the matplotlib backend."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from minigeo import Scene  # noqa: E402
from minigeo.backends.matplotlib import MatplotlibAnimation  # noqa: E402
from minigeo.geometry import BaseBox  # noqa: E402
from minigeo.render import Face  # noqa: E402


def turning_box():
    scene = Scene()
    box = scene.add(BaseBox([0, 0, 0], dimensions=[2, 1, 0.5]), fill="C0", frame=True)
    return scene, box


def turn(box, degrees=30):
    return lambda frame: box.rotate("z", degrees, about="center")


# ---------------------------------------------------------------- snapshots


def test_snapshots_capture_each_frame():
    scene, box = turning_box()
    calls = []

    def update(frame):
        calls.append(frame)
        box.rotate("z", 90, about="center")

    snapshots = scene.snapshots(update, frames=3)
    assert calls == [1, 2]  # frame 0 shows the scene as it is
    first, _, last = ([p for p in snapshot if isinstance(p, Face)] for snapshot in snapshots)

    def corners(face):
        return np.array(sorted(map(tuple, np.round(face.points, 9))))

    np.testing.assert_allclose(corners(last[0]), corners(first[0]))  # after 180° the same corners
    assert not np.allclose(snapshots[1][0].points, first[0].points)
    np.testing.assert_allclose(box.axes[0], [-1, 0, 0], atol=1e-12)  # geometry stays in the last state


def test_at_least_one_frame():
    scene, box = turning_box()
    with pytest.raises(ValueError):
        scene.snapshots(turn(box), frames=0)


def test_unsupported_animation_backend():
    scene, box = turning_box()
    with pytest.raises(ValueError):
        scene.animate(turn(box), frames=3, backend="svg")


# ---------------------------------------------------------------- plotly


def test_plotly_animation_has_frames_controls_and_a_fixed_view():
    scene, box = turning_box()
    figure = scene.animate(turn(box), frames=4, interval=80)
    assert len(figure.frames) == 4
    assert [f.name for f in figure.frames] == ["0", "1", "2", "3"]
    assert len(figure.frames[0].data) == len(figure.data)
    play = figure.layout.updatemenus[0].buttons[0]
    assert play.method == "animate" and play.args[1]["frame"]["duration"] == 80
    assert len(figure.layout.sliders[0].steps) == 4
    assert not figure.layout.scene.xaxis.autorange  # one view for all frames
    assert figure.layout.uirevision  # keeps the user's camera while playing


def test_plotly_view_covers_all_frames():
    scene = Scene()
    box = scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]))
    figure = scene.animate(lambda frame: box.place("center", at=[3 * frame, 0, 0]), frames=3)
    low, high = figure.layout.scene.xaxis.range
    assert low < -0.5 and high > 6.5


def test_plotly_animation_rejects_changing_structure():
    scene = Scene()
    box = scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]))
    with pytest.raises(ValueError):
        scene.animate(
            lambda frame: scene.add(BaseBox([2, 0, 0], dimensions=[1, 1, 1]), opacity=0.5), frames=2
        )
    assert box  # the scene itself is unharmed


# ---------------------------------------------------------------- matplotlib


def test_matplotlib_animation_redraws_each_frame():
    scene, box = turning_box()
    animation = scene.animate(turn(box), frames=3, backend="matplotlib", interval=40)
    assert isinstance(animation, MatplotlibAnimation)
    assert animation.frames == 3 and "3 frames" in repr(animation)
    html = animation.to_jshtml()  # draws every frame
    assert html.count("data:image/png") == 3
    ax = animation.figure.axes[0]
    assert len(ax.collections) == 6  # the old frame's faces are removed, not piled up
    plt.close(animation.figure)


def test_matplotlib_animation_in_jupyter_without_ipympl_is_an_html_player():
    scene, box = turning_box()
    animation = scene.animate(turn(box), frames=2, backend="matplotlib")
    assert "<script" in animation._repr_html_()
    plt.close(animation.figure)


def test_matplotlib_animation_into_given_axes_and_saved_as_gif(tmp_path):
    scene, box = turning_box()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    animation = scene.animate(turn(box), frames=2, backend="matplotlib", ax=ax, show_axes=False)
    assert animation.figure is fig and not ax.axison
    target = tmp_path / "turn.gif"
    animation.save(target)
    assert target.read_bytes()[:3] == b"GIF"
    plt.close(fig)


@pytest.mark.filterwarnings("ignore:Animation was deleted")  # this test never plays it
def test_live_backend_lets_the_figure_show_itself(monkeypatch):
    import minigeo.backends.matplotlib as backend

    scene, box = turning_box()
    animation = scene.animate(turn(box), frames=2, backend="matplotlib")
    monkeypatch.setattr(backend, "_backend_name", lambda: "module://ipympl.backend_nbagg")
    assert animation._repr_html_() is None  # ipympl shows the live figure; no second copy
    plt.close(animation.figure)
