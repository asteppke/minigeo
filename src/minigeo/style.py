"""
How geometries look: colors, opacity, line widths, dashes, arrows and text.

A :class:`Style` only holds the fields that were set; everything else is ``None`` and is
filled in later from more general styles. Styles combine with ``|`` like dictionaries do:
in ``base | override`` every field that ``override`` sets wins.

Example:
    >>> glass = Style(fill="lightblue", opacity=0.3)
    >>> red_glass = glass | Style(fill="red")      # still transparent
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from .colors import COLOR_CYCLE, color_value, to_rgb

DASH_STYLES = ("solid", "dashed", "dotted")


@dataclass(frozen=True)
class Style:
    """
    The look of a geometry. Unset fields (``None``) are taken from more general styles.

    Colors are CSS color names (``"peru"``), hex codes (``"#ff8800"``), ``"C0"`` to ``"C9"``
    for matplotlib's default colors, its one-letter colors (``"k"``), or ``"none"`` for
    nothing. Colors and sizes are checked when the style is made, so a typo fails right away.

    Attributes:
        fill: Color of faces.
        stroke: Color of lines and of the outline of faces.
        opacity: From 0 (invisible) to 1 (opaque), for faces and lines.
        line_width: Width of lines and outlines, in points.
        dash: ``"solid"``, ``"dashed"`` or ``"dotted"``.
        arrow: Whether lines end in an arrow tip.
        marker_size: Size of points, in points.
        text_color: Color of labels.
        font_size: Size of labels, in points.
    """

    fill: str | None = None
    stroke: str | None = None
    opacity: float | None = None
    line_width: float | None = None
    dash: str | None = None
    arrow: bool | None = None
    marker_size: float | None = None
    text_color: str | None = None
    font_size: float | None = None

    def __post_init__(self):
        """Check the fields right away, so that a typo fails here and not in a backend."""
        if self.dash is not None and self.dash not in DASH_STYLES:
            raise ValueError(f"Unknown dash style {self.dash!r}; use one of {', '.join(DASH_STYLES)}.")
        if self.opacity is not None and not 0 <= self.opacity <= 1:
            raise ValueError(f"opacity must be between 0 and 1, got {self.opacity}.")
        for name in ("fill", "stroke", "text_color"):
            color = getattr(self, name)
            if color is None or color == "none":
                continue
            if not isinstance(color, str):
                raise TypeError(f"{name} must be a color name or code, got {color!r}.")
            try:
                to_rgb(color)
            except ValueError as error:
                raise ValueError(f"{name}: {error}") from None
        for name, smallest in (("line_width", 0), ("marker_size", 0), ("font_size", 1e-9)):
            value = getattr(self, name)
            if value is not None and not value >= smallest:
                limit = "positive" if smallest else "zero or more"
                raise ValueError(f"{name} must be {limit}, got {value!r}.")

    def __or__(self, other: Style) -> Style:
        """Combine two styles: every field set in ``other`` wins."""
        if not isinstance(other, Style):
            return NotImplemented
        changes = {
            f.name: getattr(other, f.name) for f in fields(other) if getattr(other, f.name) is not None
        }
        return replace(self, **changes)

    def with_(self, **changes) -> Style:
        """Return a copy with some fields changed, e.g. ``style.with_(fill="red")``."""
        return replace(self, **changes)

    @property
    def is_complete(self) -> bool:
        """Whether every field is set, as needed for rendering."""
        return all(getattr(self, f.name) is not None for f in fields(self))


#: The style everything starts from; scenes and items override parts of it.
DEFAULT_STYLE = Style(
    fill=COLOR_CYCLE[0],
    stroke="dimgray",
    opacity=1.0,
    line_width=1.0,
    dash="solid",
    arrow=False,
    marker_size=20.0,
    text_color="black",
    font_size=10.0,
)


__all__ = ["COLOR_CYCLE", "DASH_STYLES", "DEFAULT_STYLE", "Style", "color_value"]
