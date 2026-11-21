r"""
Math in labels, typeset with matplotlib's mathtext.

Text between ``$`` signs is math, written as in LaTeX: ``r"$2\theta$"``,
``r"$\alpha_1^2$"`` or ``r"$\frac{a}{b}$"``. Labels can mix text and math:
``r"angle $\theta$"``. A single ``$`` (or ``\$``) stays an ordinary dollar sign.

Each backend shows math as well as it can:

* matplotlib typesets it itself, and TikZ hands it to LaTeX;
* SVG and the ray tracer draw the outlines that mathtext produces (:func:`outline`), so
  the picture needs no fonts and no LaTeX;
* plotly cannot show math in 3D scenes, so it gets Unicode letters with raised and lowered
  parts (:func:`plain`).

The outlines need matplotlib (``pip install "minigeo[mpl]"``); without it, math labels are
shown as plain text and a warning says so.
"""

from __future__ import annotations

import re
import warnings

import numpy as np

# a ``$`` that is not escaped as ``\$``
_DOLLAR = re.compile(r"(?<!\\)\$")


def has_math(text: str) -> bool:
    """
    Whether a label contains math: an even number of ``$`` signs, as in matplotlib.

    Example:
        >>> has_math(r"angle $\theta$"), has_math("costs $5")
        (True, False)
    """
    count = len(_DOLLAR.findall(text))
    return count > 0 and count % 2 == 0


def outline(text: str, size: float):
    """
    The outline of a label as a matplotlib path, ``size`` units high per em.

    The text starts at x = 0 with its baseline at y = 0, and y points up (as in TeX).

    Raises:
        ValueError: If the math is not valid mathtext.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    try:
        return TextPath((0, 0), text, size=size, prop=FontProperties(family=["DejaVu Sans"]))
    except ValueError as error:
        raise ValueError(f"Cannot typeset the label {text!r}:\n{error}") from None


def raster(text: str, size: float, color: tuple[int, int, int]) -> tuple[np.ndarray, float, float]:
    """
    A label drawn into an RGBA image with matplotlib's Agg renderer (smooth edges).

    Args:
        text: The label, with math between ``$`` signs.
        size: Font size in pixels.
        color: Red, green and blue from 0 to 255.

    Returns:
        The image as an ``(height, width, 4)`` array of bytes, and where the start of the
        baseline lies in it, as ``x`` and ``y`` in pixels from the top left.
    """
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.transforms import Affine2D

    path = outline(text, size)
    box = path.get_extents()
    pad = 2
    width = int(np.ceil(box.width)) + 2 * pad
    height = int(np.ceil(box.height)) + 2 * pad
    renderer = RendererAgg(width, height, 72)
    gc = renderer.new_gc()
    gc.set_linewidth(0)
    # Agg's y axis points up from the bottom of the image, like the outline's
    shift = Affine2D().translate(pad - box.x0, pad - box.y0)
    renderer.draw_path(gc, path, shift, tuple(channel / 255 for channel in color))
    gc.restore()
    image = np.asarray(renderer.buffer_rgba()).copy()
    return image, pad - box.x0, height - (pad - box.y0)


def plain(text: str) -> str:
    r"""
    A label without math, for backends that cannot typeset it: math letters become their
    Unicode characters, superscripts and subscripts are wrapped in ``<sup>`` and ``<sub>``
    (which plotly understands). Fractions and roots lose their shape.

    Example:
        >>> plain(r"angle $\theta_1^2$")
        'angle θ<sub>1</sub><sup>2</sup>'
    """
    if not has_math(text):
        return text
    parts = re.split(r"((?<!\\)\$.*?(?<!\\)\$)", text)
    try:
        return "".join(_plain_math(part[1:-1]) if part.startswith("$") else part for part in parts)
    except ImportError:
        warn_without_matplotlib()
        return _DOLLAR.sub("", text)


def _plain_math(math: str) -> str:
    """One piece of math as Unicode letters, placed by matplotlib's layout."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser

    try:
        layout = MathTextParser("path").parse(f"${math}$", dpi=72, prop=FontProperties(size=10))
    except ValueError as error:
        raise ValueError(f"Cannot typeset the math {math!r}:\n{error}") from None
    glyphs = sorted(layout.glyphs, key=lambda glyph: glyph[3])  # left to right
    if not glyphs:
        return ""
    biggest = max(glyph[1] for glyph in glyphs)
    baseline = np.median([glyph[4] for glyph in glyphs if glyph[1] == biggest])

    result, current = [], ""
    for _, size, code, _, y in glyphs:
        if size < biggest and y > baseline + 1:
            place = "sup"
        elif size < biggest and y < baseline - 1:
            place = "sub"
        else:
            place = ""
        if place != current:
            result.append(f"</{current}>" if current else "")
            result.append(f"<{place}>" if place else "")
            current = place
        result.append(chr(code))
    if current:
        result.append(f"</{current}>")
    return "".join(result)


def warn_without_matplotlib() -> None:
    """Say once that math labels need matplotlib."""
    warnings.warn(
        'Math in labels needs matplotlib: pip install "minigeo[mpl]". Showing it as plain text.',
        stacklevel=3,
    )


def matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True
