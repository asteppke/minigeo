"""
Sphinx extension for the minigeo manual: examples with their picture next to the code.

In a page of the manual::

    .. minigeo-example::
       :width: 220

       scene = Scene()
       scene.add(Box([0, 0, 0], size=(1, 1, 1)), fill="C1")

The code runs while the documentation is built, after ``import numpy as np`` and
``from minigeo import *``. Each example runs on its own. If it defines ``scene``, the scene is
drawn next to the code; whatever it prints is shown below the picture. An example that fails
stops the build (``sphinx-build -W``), and ``tests/test_manual.py`` runs every example with
pytest, using :func:`examples_in` and :func:`run_example` from this module.

Options:
    width: Width of the picture in pixels (default 220).
    backend: ``svg`` (default), ``raytrace`` or ``matplotlib``.
    elev, azim, vertical-axis: The view, as in ``scene.render``. When these or ``backend``
        are given, the shown code ends with a comment saying how the picture was drawn.
"""

from __future__ import annotations

import base64
import contextlib
import html
import io
import re
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective

PREAMBLE = "import numpy as np\nfrom minigeo import *\n"
DEFAULT_WIDTH = 220
BACKENDS = ("svg", "raytrace", "matplotlib")


# ---------------------------------------------------------------- running examples


@dataclass
class Example:
    """One example of the manual: its code, options and where it is."""

    code: str
    options: dict = field(default_factory=dict)
    source: str = ""
    line: int = 0


@dataclass
class Result:
    scene: object | None
    printed: str


def run_example(code: str) -> Result:
    """Run the code of an example in a fresh namespace; return its ``scene`` and what it printed."""
    namespace: dict = {"__name__": "__minigeo_example__"}
    exec(PREAMBLE, namespace)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(code, "<minigeo example>", "exec"), namespace)
    result = Result(namespace.get("scene"), printed.getvalue())
    if result.scene is None and not result.printed:
        raise ValueError("A manual example must define `scene` or print something.")
    return result


def picture(scene, options: dict) -> tuple[str, bytes]:
    """Draw the scene with the example's options: returns the MIME type and the image data."""
    width = int(options.get("width", DEFAULT_WIDTH))
    backend = options.get("backend", "svg")
    view = {key: options[key] for key in ("elev", "azim", "vertical_axis") if key in options}
    if backend == "svg":
        return "image/svg+xml", str(scene.render("svg", width=width, **view)).encode()
    if backend == "raytrace":  # twice the pixels, for sharp pictures on fine screens
        return "image/png", scene.render("raytrace", width=2 * width, **view).to_png()
    if backend == "matplotlib":
        from minigeo.backends.matplotlib import to_png

        return "image/png", to_png(scene, size=width / 50, dpi=100, show_axes=False, **view)
    raise ValueError(f"Unknown backend {backend!r}; use one of {', '.join(BACKENDS)}.")


def output_html(result: Result, options: dict) -> str:
    """The picture and the printed text, as HTML."""
    parts = []
    if result.scene is not None:
        mime, data = picture(result.scene, options)
        width = int(options.get("width", DEFAULT_WIDTH))
        source = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        parts.append(f'<img src="{source}" width="{width}" alt="the picture drawn by the code">')
    if result.printed:
        parts.append(f'<pre class="minigeo-printed">{html.escape(result.printed.rstrip())}</pre>')
    return '<div class="minigeo-output">' + "".join(parts) + "</div>"


def render_note(options: dict) -> str:
    """A comment for the shown code, saying how the picture was drawn if not with the defaults."""
    view = [key for key in ("elev", "azim", "vertical_axis") if key in options]
    if not view and options.get("backend", "svg") == "svg":
        return ""
    arguments = [f'"{options.get("backend", "svg")}"']
    for key in view:
        value = options[key]
        arguments.append(f"{key}={value:g}" if isinstance(value, float) else f'{key}="{value}"')
    return f"# drawn with scene.render({', '.join(arguments)})"


# ---------------------------------------------------------------- finding examples in pages

_DIRECTIVE = re.compile(r"^(\s*)\.\. minigeo-example::\s*$")
_OPTION = re.compile(r"^:([\w-]+):\s*(.*)$")


def parse_options(raw: dict[str, str]) -> dict:
    """Convert option text to values, with the names ``scene.render`` uses."""
    options = {}
    for name, value in raw.items():
        if name == "width":
            options["width"] = int(value)
        elif name in ("elev", "azim"):
            options[name] = float(value)
        elif name == "vertical-axis":
            options["vertical_axis"] = value
        elif name == "backend":
            if value not in BACKENDS:
                raise ValueError(f"Unknown backend {value!r}; use one of {', '.join(BACKENDS)}.")
            options["backend"] = value
        else:
            raise ValueError(f"Unknown option {name!r} for minigeo-example.")
    return options


def examples_in(path: Path) -> Iterator[Example]:
    """All ``minigeo-example`` blocks of a reStructuredText file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        match = _DIRECTIVE.match(lines[index])
        index += 1
        if not match:
            continue
        start, outer = index, len(match.group(1))
        body = []
        while index < len(lines) and (not lines[index].strip() or _indent(lines[index]) > outer):
            body.append(lines[index])
            index += 1
        indent = min((_indent(line) for line in body if line.strip()), default=0)
        body = [line[indent:] for line in body]
        raw_options = {}
        while body and _OPTION.match(body[0]):
            name, value = _OPTION.match(body.pop(0)).groups()
            raw_options[name] = value
        code = "\n".join(body).strip("\n")
        yield Example(code, parse_options(raw_options), str(path), start)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


# ---------------------------------------------------------------- the directive


class MinigeoExample(SphinxDirective):
    """Show an example's picture next to its code."""

    has_content = True
    option_spec = {
        "width": directives.positive_int,
        "backend": lambda value: directives.choice(value, BACKENDS),
        "elev": float,
        "azim": float,
        "vertical-axis": lambda value: directives.choice(value, ("x", "y", "z")),
    }

    def run(self) -> list[nodes.Node]:
        code = "\n".join(self.content)
        options = parse_options({name: str(value) for name, value in self.options.items()})
        try:
            output = output_html(run_example(code), options)
        except Exception:
            raise self.error(f"This minigeo example failed:\n{traceback.format_exc()}") from None

        note = render_note(options)
        shown = f"{code}\n\n{note}" if note else code
        code_block = nodes.literal_block(shown, shown)
        code_block["language"] = "python"
        code_column = nodes.container(classes=["minigeo-code"])
        code_column += code_block
        example = nodes.container(classes=["minigeo-example"])
        example += nodes.raw("", output, format="html")
        example += code_column
        return [example]


def setup(app):
    app.add_directive("minigeo-example", MinigeoExample)
    app.add_css_file("minigeo-manual.css")
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
