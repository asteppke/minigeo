"""
Every example of the manual (``docs/manual``) runs and draws its picture.

The examples are found with the manual's own Sphinx extension (``docs/_ext``), so the tests
run exactly the code the documentation shows.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("docutils")
pytest.importorskip("sphinx")
pytest.importorskip("svg")

MANUAL = Path(__file__).parents[1] / "docs" / "manual"
sys.path.insert(0, str(MANUAL.parent / "_ext"))

import minigeo_manual  # noqa: E402

EXAMPLES = [example for page in sorted(MANUAL.glob("*.rst")) for example in minigeo_manual.examples_in(page)]


def test_the_manual_has_examples():
    assert len(EXAMPLES) > 30


@pytest.mark.parametrize("example", EXAMPLES, ids=[f"{Path(e.source).stem}:{e.line}" for e in EXAMPLES])
def test_example_runs_and_draws(example):
    if example.options.get("backend") == "raytrace":
        pytest.importorskip("PIL")
    if example.options.get("backend") == "matplotlib":
        pytest.importorskip("matplotlib")
    result = minigeo_manual.run_example(example.code)
    html = minigeo_manual.output_html(result, example.options)
    assert html.startswith('<div class="minigeo-output"')
    if result.scene is not None or result.animation is not None:
        assert "<img" in html


def test_failing_examples_are_reported():
    with pytest.raises(ZeroDivisionError):
        minigeo_manual.run_example("1 / 0")
    with pytest.raises(ValueError, match="animation"):
        minigeo_manual.run_example("x = 1")


def test_options_are_parsed(tmp_path):
    page = tmp_path / "page.rst"
    page.write_text(
        "Text\n\n.. minigeo-example::\n   :width: 300\n   :elev: 10\n   :vertical-axis: x\n\n"
        "   scene = Scene()\n   scene.add(Box([0, 0, 0], size=1))\n\nMore text\n"
    )
    (example,) = minigeo_manual.examples_in(page)
    assert example.options == {"width": 300, "elev": 10.0, "vertical_axis": "x"}
    assert example.code == "scene = Scene()\nscene.add(Box([0, 0, 0], size=1))"
    assert (
        minigeo_manual.render_note(example.options)
        == '# drawn with scene.render("svg", elev=10, vertical_axis="x")'
    )
    with pytest.raises(ValueError, match="option"):
        minigeo_manual.parse_options({"colour": "red"})


def test_render_option_passes_settings_and_is_shown():
    options = minigeo_manual.parse_options(
        {"backend": "raytrace", "render": 'look="cel", palette=["whitesmoke", "slategray"], bands=4'}
    )
    assert options["render"] == {"look": "cel", "palette": ["whitesmoke", "slategray"], "bands": 4}
    note = minigeo_manual.render_note(options)
    assert note.startswith('# drawn with scene.render("raytrace", look="cel"')
    assert all(len(line) <= 64 for line in note.splitlines())
    with pytest.raises(ValueError, match="keyword"):
        minigeo_manual.render_arguments('"cel"')


def test_picture_column_has_the_picture_width():
    """Without an explicit width, pictures in a shrink-to-fit column can collapse to nothing."""
    result = minigeo_manual.run_example("scene = Scene()\nscene.add(Box([0, 0, 0], size=1))")
    assert '<div class="minigeo-output" style="width: 180px">' in minigeo_manual.output_html(
        result, {"width": 180}
    )


def test_animations_can_be_stored_as_files():
    code = (
        "scene = Scene()\nbox = scene.add(Box([0, 0, 0], size=1))\n"
        "animation = scene.animate(lambda i: box.rotate('z', 30), frames=2, backend='raytrace', width=40)"
    )
    pytest.importorskip("PIL")
    result = minigeo_manual.run_example(code)
    stored = []
    html = minigeo_manual.output_html(result, {}, store=lambda data: stored.append(data) or "_images/x.png")
    assert '<img src="_images/x.png"' in html
    assert stored and stored[0].startswith(b"\x89PNG") and b"acTL" in stored[0]
    assert "data:image/png" in minigeo_manual.output_html(result, {})  # without a store: in the page
