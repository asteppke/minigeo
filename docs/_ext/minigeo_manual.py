"""
Sphinx extension for the minigeo manual: examples with their picture next to the code.

In a page of the manual::

    .. minigeo-example::
       :width: 220

       scene = Scene()
       scene.add(Box([0, 0, 0], size=(1, 1, 1)), fill="C1")

The code runs while the documentation is built, after ``import numpy as np`` and
``from minigeo import *``. Each example runs on its own. If it defines ``scene``, the scene is
drawn next to the code; if it defines ``animation`` (from ``scene.animate(..., backend=
"raytrace")``), the animation is shown instead, as an animated PNG. Whatever the example
prints is shown below the picture. An example that fails
stops the build (``sphinx-build -W``), and ``tests/test_manual.py`` runs every example with
pytest, using :func:`examples_in` and :func:`run_example` from this module.

Options:
    width: Width of the picture in pixels (default 220).
    backend: ``svg`` (default), ``raytrace`` or ``matplotlib``.
    elev, azim, vertical-axis: The view, as in ``scene.render``.
    render: Further keyword arguments for ``scene.render``, e.g. ``look="cel", bands=4``.

When any option but ``width`` is given, the shown code ends with a comment saying how the
picture was drawn.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import hashlib
import html
import io
import re
import traceback
from collections.abc import Callable, Iterator
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
    animation: object | None = None


def run_example(code: str) -> Result:
    """Run the code of an example in a fresh namespace; return its ``scene`` and what it printed."""
    namespace: dict = {"__name__": "__minigeo_example__"}
    exec(PREAMBLE, namespace)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(code, "<minigeo example>", "exec"), namespace)
    result = Result(namespace.get("scene"), printed.getvalue(), namespace.get("animation"))
    if result.scene is None and result.animation is None and not result.printed:
        raise ValueError("A manual example must define `scene` or `animation`, or print something.")
    return result


def picture(scene, options: dict) -> tuple[str, bytes]:
    """Draw the scene with the example's options: returns the MIME type and the image data."""
    width = int(options.get("width", DEFAULT_WIDTH))
    backend = options.get("backend", "svg")
    view = {key: options[key] for key in ("elev", "azim", "vertical_axis") if key in options}
    view.update(options.get("render", {}))
    if backend == "svg":
        return "image/svg+xml", str(scene.render("svg", width=width, **view)).encode()
    if backend == "raytrace":  # twice the pixels, for sharp pictures on fine screens
        return "image/png", scene.render("raytrace", width=2 * width, **view).to_png()
    if backend == "matplotlib":
        from minigeo.backends.matplotlib import to_png

        return "image/png", to_png(scene, size=width / 50, dpi=100, show_axes=False, **view)
    raise ValueError(f"Unknown backend {backend!r}; use one of {', '.join(BACKENDS)}.")


def output_html(result: Result, options: dict, store: Callable[[bytes], str] | None = None) -> str:
    """
    The picture and the printed text, as HTML.

    Pictures are written into the page itself, as ``data:`` URLs. Animations are much larger,
    and a page with such a URL may not show it, so they are handed to ``store``, which saves
    them as a file and returns its address (without ``store`` they are written into the page
    too, e.g. in the tests).
    """
    parts = []
    column = ""
    if result.scene is not None or result.animation is not None:
        if result.animation is not None:
            mime, data = "image/png", result.animation.to_png()  # browsers play animated PNGs
        else:
            mime, data = picture(result.scene, options)
        width = int(options.get("width", DEFAULT_WIDTH))
        column = f' style="width: {width}px"'  # printed text wraps at the picture's width
        if result.animation is not None and store is not None:
            source = store(data)
        else:
            source = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        parts.append(f'<img src="{source}" width="{width}" alt="the picture drawn by the code">')
    if result.printed:
        parts.append(f'<pre class="minigeo-printed">{html.escape(result.printed.rstrip())}</pre>')
    return f'<div class="minigeo-output"{column}>' + "".join(parts) + "</div>"


def render_note(options: dict) -> str:
    """A comment for the shown code, saying how the picture was drawn if not with the defaults."""
    view = [key for key in ("elev", "azim", "vertical_axis") if key in options]
    if not view and not options.get("render") and options.get("backend", "svg") == "svg":
        return ""
    arguments = [f'"{options.get("backend", "svg")}"']
    for key in view:
        value = options[key]
        arguments.append(f"{key}={value:g}" if isinstance(value, float) else f'{key}="{value}"')
    for key, value in options.get("render", {}).items():
        arguments.append(f"{key}={_python(value)}")
    return _wrapped_comment("drawn with scene.render(", arguments, ")")


def _python(value) -> str:
    """A literal as Python code, with double quotes as in the examples."""
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, list | tuple):
        inner = ", ".join(_python(item) for item in value)
        return f"[{inner}]" if isinstance(value, list) else f"({inner})"
    return repr(value)


def _wrapped_comment(start: str, arguments: list[str], end: str, limit: int = 64) -> str:
    """A comment ``# start a, b, c end``, continued on further comment lines when too long."""
    lines, current = [], f"# {start}"
    for k, argument in enumerate(arguments):
        piece = argument + (", " if k < len(arguments) - 1 else end)
        if len(current) + len(piece.rstrip()) > limit and not current.endswith("("):
            lines.append(current.rstrip())
            current = "#     "
        current += piece
    lines.append(current)
    return "\n".join(lines)


def render_arguments(text: str) -> dict:
    """The keyword arguments written in a ``:render:`` option, e.g. ``look="cel", bands=4``."""
    call = ast.parse(f"f({text})", mode="eval").body
    if call.args:
        raise ValueError(f"The render option takes only keyword arguments, not {text!r}.")
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}


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
        elif name == "render":
            options["render"] = render_arguments(value)
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
        "render": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        code = "\n".join(self.content)
        options = parse_options({name: str(value) for name, value in self.options.items()})
        try:
            output = output_html(run_example(code), options, store=self._store)
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

    def _store(self, data: bytes) -> str:
        """Save an animation as ``_images/minigeo-<hash>.png`` in the output; return its address."""
        name = f"minigeo-{hashlib.sha1(data).hexdigest()[:16]}.png"
        folder = Path(self.env.app.outdir) / "_images"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
        return "../" * self.env.docname.count("/") + f"_images/{name}"


def setup(app):
    app.add_directive("minigeo-example", MinigeoExample)
    app.add_css_file("minigeo-manual.css")
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
