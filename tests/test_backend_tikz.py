"""The TikZ backend, the shared 2D pipeline and the color table."""

import re
import shutil

import numpy as np
import pytest

from minigeo import Angle, Box, Cylinder, Dimension, Line, Point, Rectangle, Scene, Style
from minigeo.backends.tikz import TikzPicture, latex_text, render
from minigeo.camera import Camera
from minigeo.colors import CSS_COLORS, to_rgb
from minigeo.flat import SHADING_STRENGTH, flatten
from minigeo.render import Face, Label

COORDINATE = re.compile(r"\((-?[\d.]+),(-?[\d.]+)\)")


def coordinates(code: str) -> np.ndarray:
    return np.array([[float(x), float(y)] for x, y in COORDINATE.findall(code)])


def box_scene(**style) -> Scene:
    scene = Scene()
    scene.add(Box([0, 0, 0], size=(2, 1, 1)), **style)
    return scene


# ---------------------------------------------------------------- colors


def test_colors():
    assert to_rgb("C0") == pytest.approx((31 / 255, 119 / 255, 180 / 255))
    assert to_rgb("peru") == pytest.approx((205 / 255, 133 / 255, 63 / 255))
    assert to_rgb("PeRu") == to_rgb("peru")
    assert to_rgb("#ff8800") == pytest.approx((1, 136 / 255, 0))
    assert to_rgb("#f80") == to_rgb("#ff8800")
    assert len(CSS_COLORS) == 148 and CSS_COLORS["rebeccapurple"] == (102, 51, 153)
    for wrong in ("#12345", "#gggggg", "notacolor"):
        with pytest.raises(ValueError):
            to_rgb(wrong)


# ---------------------------------------------------------------- the shared 2D pipeline


def test_flatten_fits_the_width_and_hides_back_faces():
    scene = box_scene()
    camera = Camera()
    picture = flatten(scene.primitives(), camera, width=10, margin=1, y_down=False)
    assert picture.width == pytest.approx(10)
    points = np.vstack([item.points for item in picture.items])
    assert points[:, 0].min() == pytest.approx(1) and points[:, 0].max() == pytest.approx(9)
    assert points[:, 1].min() == pytest.approx(1)
    assert len(picture.items) == 3  # a box shows three faces


def test_flatten_y_down_mirrors_y():
    scene = box_scene()
    up = flatten(scene.primitives(), Camera(), width=10, margin=0, y_down=False)
    down = flatten(scene.primitives(), Camera(), width=10, margin=0, y_down=True)
    for a, b in zip(up.items, down.items, strict=True):
        assert np.allclose(a.points[:, 1], up.height - b.points[:, 1])


def test_flatten_shading():
    scene = box_scene()
    shaded = flatten(scene.primitives(), Camera(), width=10, margin=0, y_down=False)
    darkness = sorted(item.darkness for item in shaded.items)
    assert 0 <= darkness[0] < darkness[-1] <= SHADING_STRENGTH
    flat = flatten(scene.primitives(), Camera(), width=10, margin=0, y_down=False, shading=False)
    assert all(item.darkness == 0 for item in flat.items)


def test_labels_come_last():
    scene = box_scene(label="box")
    picture = flatten(scene.primitives(), Camera(), width=10, margin=0, y_down=False)
    assert isinstance(picture.items[-1].primitive, Label)


# ---------------------------------------------------------------- TikZ code


def test_picture_structure():
    picture = render(box_scene(fill="peru"), width=6)
    lines = picture.code.splitlines()
    assert lines[0].startswith("%")
    assert lines[1].startswith(r"\begin{tikzpicture}") and lines[-1] == r"\end{tikzpicture}"
    assert r"\definecolor{mgperu}{RGB}{205,133,63}" in picture.code
    assert picture.code.count(r"\filldraw") == 3
    xy = coordinates(picture.code)
    assert xy[:, 0].min() == pytest.approx(0) and xy[:, 0].max() == pytest.approx(6)
    assert str(picture) == picture.code


def test_faces_are_shaded_by_mixing_with_black():
    code = render(box_scene(fill="C1")).code
    mixes = [int(m) for m in re.findall(r"mgC1!(\d+)!black", code)]
    assert mixes and all(100 * (1 - SHADING_STRENGTH) <= m <= 100 for m in mixes)
    assert "!black" not in render(box_scene(fill="C1"), shading=False).code


def test_transparent_faces_and_open_outlines():
    scene = Scene()
    scene.add(Rectangle([0, 0, 0], size=1), opacity=0.5)
    scene.add(Cylinder([2, 0, 0], radius=0.5, height=1))
    code = render(scene).code
    assert "fill opacity=0.5" in code
    assert "line width=0.2pt" in code  # seam cover of the cylinder's side faces
    assert re.search(r"\\draw\[draw=mgdimgray[^\]]*\] \([^)]*\) -- \([^)]*\) \(", code)  # ring segments


def test_lines_dashes_arrows_points():
    scene = Scene()
    scene.add(Line([0, 0, 0], [1, 0, 0]), dash="dashed", line_width=2)
    scene.add(Line([0, 1, 0], [1, 1, 0]), arrow=True, stroke="C3")
    scene.add(Point([0.5, 0.5, 0]), fill="red", opacity=0.5)
    scene.add(Line([0, 2, 0], [1, 2, 0]), stroke="none")
    code = render(scene, width=5).code
    width = 2 * 5 * 72.27 / 2.54 / 500
    assert f"line width={width:.3f}pt" in code
    assert f"dash pattern=on {6 * width:.3f}pt off {4 * width:.3f}pt" in code
    assert "-{Stealth[" in code
    assert "circle[radius=" in code and "opacity=0.5" in code
    assert code.count(r"\draw[") == 2  # the invisible line is left out


def test_labels():
    scene = box_scene(label="top & 50%", label_at="top")
    scene.add(Dimension([0, 0, 0], [1, 0, 0], unit=" m"))
    code = render(scene).code
    assert r"{top \& 50\%}" in code
    assert "anchor=south west" in code and "anchor=center" in code
    assert r"\fontsize{10}{12}\selectfont" in code


def test_latex_text():
    assert latex_text("a_b # {c}") == r"a\_b \# \{c\}"
    assert latex_text(r"$2\theta_1$ and θ′") == r"$2\theta_1$ and $\theta$$'$"
    assert latex_text("10°, 5 µm, 2×3") == r"10$^\circ$, 5 $\mu$m, 2$\times$3"
    assert latex_text("~^\\") == r"\textasciitilde{}\textasciicircum{}\textbackslash{}"
    assert latex_text("Größe") == "Größe"  # other letters pass through (utf-8)


def test_on_top_paths_after_faces():
    scene = box_scene(frame=True)
    code = render(scene).code
    last_face = max(i for i, line in enumerate(code.splitlines()) if line.startswith(r"\filldraw"))
    first_arrow = min(i for i, line in enumerate(code.splitlines()) if "Stealth" in line)
    assert first_arrow > last_face


def test_document_and_saving(tmp_path):
    picture = render(box_scene())
    document = picture.document()
    assert document.startswith(r"\documentclass[tikz")
    assert r"\usetikzlibrary{arrows.meta}" in document and picture.code in document
    picture.save(tmp_path / "picture.tex")
    assert (tmp_path / "picture.tex").read_text().startswith("% made with minigeo")
    picture.save(tmp_path / "document.tex", standalone=True)
    assert (tmp_path / "document.tex").read_text() == document
    assert picture._repr_markdown_().startswith("```latex")


def test_scene_render_and_save(tmp_path):
    scene = box_scene()
    assert isinstance(scene.render("tikz"), TikzPicture)
    scene.save(tmp_path / "scene.tex", width=4)
    assert r"\begin{tikzpicture}" in (tmp_path / "scene.tex").read_text()
    scene.save(tmp_path / "scene.txt", backend="tikz")
    assert (tmp_path / "scene.txt").exists()


def test_missing_engine():
    with pytest.raises(RuntimeError, match="not installed"):
        render(box_scene()).to_pdf(engine="no-such-latex")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="needs pdflatex")
def test_compiles_with_pdflatex(tmp_path):
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1), fill="C0", frame=True, label="box_1 & $x^2$ 50%")
    scene.add(Cylinder([1.5, 0, 0], radius=0.4, height=1), fill="peru", opacity=0.6)
    scene.add(Angle([0, 0, 0], [1, 0, 0], [0, 1, 0]))
    scene.add(Dimension([0, 0, 0], [1, 0, 0], offset=[0, -0.3, 0], unit=" µm"))
    scene.add(Line([0, 1, 0], [1, 1, 1]), dash="dotted", arrow=True, label="θ = 30°")
    scene.save(tmp_path / "scene.pdf", backend="tikz")
    assert (tmp_path / "scene.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="needs pdflatex")
def test_latex_errors_are_reported():
    broken = TikzPicture(r"\begin{tikzpicture}\undefinedcommand\end{tikzpicture}")
    with pytest.raises(RuntimeError, match="Undefined control sequence"):
        broken.to_pdf()


def test_face_without_fill_is_outlined_only():
    scene = Scene()
    scene.add(Box([0, 0, 0], size=1), fill="none", stroke="C3", style=Style(dash="dashed"))
    code = render(scene).code
    assert r"\filldraw" not in code and r"\fill[" not in code
    assert code.count(r"\draw[draw=mgC3") >= 3


def test_faces_are_kept_as_primitives():
    assert all(isinstance(p, Face) for p in box_scene().primitives())
