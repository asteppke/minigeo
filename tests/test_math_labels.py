"""Math in labels: typeset by matplotlib's mathtext for SVG and the ray tracer."""

import re

import numpy as np
import pytest

from minigeo import Dimension, Point, Scene
from minigeo.math_labels import has_math, outline, plain, raster

pytest.importorskip("matplotlib")


def test_which_labels_have_math():
    assert has_math(r"$\theta$") and has_math(r"angle $\theta$ and $\phi$")
    assert not has_math("costs $5") and not has_math("plain") and not has_math(r"\$5 or \$6")


def test_plain_text_for_backends_without_math():
    assert plain(r"angle $\theta_1^2$") == "angle θ<sub>1</sub><sup>2</sup>"
    assert plain(r"$x^{10}$ mm") == "x<sup>10</sup> mm"
    assert plain("costs $5") == "costs $5"


def test_outline_starts_at_the_baseline():
    box = outline("x", 10).get_extents()
    assert box.y0 == pytest.approx(0, abs=0.2) and 3 < box.y1 < 7
    tall = outline(r"$\frac{a}{b}$", 10).get_extents()
    assert tall.y0 < -1 and tall.y1 > 6  # the fraction reaches below and above the baseline
    with pytest.raises(ValueError, match="Cannot typeset"):
        outline(r"$\frac{a$", 10)


def test_raster_is_smooth_and_colored():
    pixels, x, y = raster(r"$\alpha$", 20, (255, 0, 0))
    assert pixels.dtype == np.uint8 and pixels.shape[2] == 4
    alpha = pixels[..., 3]
    assert alpha.max() == 255 and np.any((alpha > 0) & (alpha < 255))  # smooth edges
    assert np.all(pixels[alpha == 255][:, :3] == [255, 0, 0])
    assert 0 < x < 5 and 0 < y <= pixels.shape[0]  # the baseline's start lies in the image


def scene_with(text, **style):
    scene = Scene()
    scene.add(Point([0, 0, 0]))
    scene.add(Point([1, 1, 0]))
    scene.label([0.5, 0.5, 0], text, **style)
    return scene


def test_svg_draws_math_as_outlines():
    code = str(scene_with(r"$\theta_1$", text_color="red").render("svg"))
    assert "<text" not in code
    assert re.search(r'<path stroke="none" d="M [^"]+ Z[^"]*" fill="red"', code)
    assert "<text" in str(scene_with("costs $5").render("svg"))


def test_svg_centers_math_on_dimensions():
    scene = Scene()
    scene.add(Dimension([0, 0, 0], [2, 0, 0], text=r"$\ell$"))
    drawing = scene.render("svg", elev=90, azim=-90)
    (path,) = re.findall(r'<path stroke="none" d="(M[^"]+)"', str(drawing))
    numbers = np.array(re.findall(r"-?\d+\.?\d*", path), dtype=float).reshape(-1, 2)
    width = float(re.search(r'width="([\d.]+)"', str(drawing)).group(1))
    assert np.mean(numbers[:, 0]) == pytest.approx(width / 2, abs=4)


def test_ray_tracer_draws_math():
    pytest.importorskip("PIL")
    with_math = np.asarray(scene_with(r"$\sum_i x_i$", text_color="red").render("raytrace", width=200).pixels)
    without = np.asarray(scene_with("", text_color="red").render("raytrace", width=200).pixels)
    red = (with_math[..., 0] > 200) & (with_math[..., 1] < 80)
    assert red.sum() > 20 and not ((without[..., 0] > 200) & (without[..., 1] < 80)).any()


def test_plotly_shows_plain_text():
    pytest.importorskip("plotly")
    figure = scene_with(r"$x^2$").render("plotly")
    assert [trace.text for trace in figure.data if getattr(trace, "mode", None) == "text"] == [
        ("x<sup>2</sup>",)
    ]


def test_tikz_hands_math_to_latex():
    assert r"{$\theta_1$}" in scene_with(r"$\theta_1$").render("tikz").code


def test_without_matplotlib_math_is_shown_as_text(monkeypatch):
    import minigeo.backends.svg as backend

    monkeypatch.setattr(backend, "matplotlib_available", lambda: False)
    with pytest.warns(UserWarning, match="needs matplotlib"):
        code = str(scene_with(r"$\theta$").render("svg"))
    assert "<text" in code and "θ" in code
