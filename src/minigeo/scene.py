"""
A scene: geometries together with how they look, ready to be drawn by a backend.

The geometries stay independent of drawing. A scene only keeps references to them, so
moving a shape after adding it moves it in the next rendering, too.

Example:
    >>> scene = Scene()
    >>> table = scene.add(BaseBox([0, 0, 0], dimensions=[2, 1, 0.1]), fill="peru")
    >>> cup = scene.add(BaseCylinder(table.anchor("top"), dimensions=[0.1, 0.2]), label="cup")
    >>> scene.render()  # with matplotlib; later also "svg" and "plotly"
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from .geometry import BaseGeometry
from .groups import GeometryGroup
from .render import Label, Primitive, bounds, to_primitives, type_style
from .style import DEFAULT_STYLE, Style

Drawable = TypeVar("Drawable", BaseGeometry, GeometryGroup)

#: Backend names and the modules that implement them; each module has ``render(scene, **options)``.
BACKENDS = {
    "matplotlib": "minigeo.backends.matplotlib",
}


@dataclass
class SceneItem:
    """One geometry in a scene, with its own style and an optional label."""

    geometry: BaseGeometry | GeometryGroup
    style: Style
    label: str | None = None
    label_at: str | float | np.ndarray = "center"


class Scene:
    """
    A collection of geometries with styles, to be drawn by a backend.

    Styles are combined from general to specific: the library defaults, then defaults for
    the kind of geometry (axes are arrows), then the scene's ``style``, then each item's own.
    """

    def __init__(self, style: Style | None = None):
        """
        Args:
            style: Defaults for everything in this scene, e.g. ``Style(opacity=0.8)``.
        """
        self.style = style if style is not None else Style()
        self.items: list[SceneItem] = []

    def add(
        self,
        geometry: Drawable,
        style: Style | None = None,
        *,
        label: str | None = None,
        label_at: str | float | np.ndarray = "center",
        **style_fields,
    ) -> Drawable:
        """
        Add a geometry and return it, so it can be created and added in one line.

        Args:
            geometry: A shape, line, point, axis or group.
            style: A :class:`Style` for this geometry.
            label: Text to show next to the geometry.
            label_at: The anchor where the label goes, e.g. ``"top"`` or ``"near end"``.
            **style_fields: Shortcuts for single style fields, e.g. ``fill="red"``;
                they win over ``style``.

        Example:
            >>> box = scene.add(BaseBox([0, 0, 0], dimensions=[1, 1, 1]), fill="red", opacity=0.5)
        """
        item_style = (style if style is not None else Style()) | Style(**style_fields)
        self.items.append(SceneItem(geometry, item_style, label, label_at))
        return geometry

    def resolved_style(self, item: SceneItem) -> Style:
        """The complete style an item is drawn with."""
        return DEFAULT_STYLE | type_style(item.geometry) | self.style | item.style

    def primitives(self) -> list[Primitive]:
        """Everything in the scene as neutral primitives (faces, paths, markers, labels)."""
        result: list[Primitive] = []
        for item in self.items:
            style = self.resolved_style(item)
            result.extend(to_primitives(item.geometry, style))
            if item.label is not None:
                result.append(Label(item.geometry.anchor(item.label_at), item.label, style))
        return result

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The lowest and highest corner of the box around everything in the scene."""
        return bounds(self.primitives())

    def render(self, backend: str = "matplotlib", **options):
        """
        Draw the scene with a backend and return what the backend produces.

        Args:
            backend: ``"matplotlib"`` (returns the 3D axes).
            **options: Passed on to the backend, e.g. ``ax=...`` or ``show_axes=False``.
        """
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}; available: {', '.join(BACKENDS)}.")
        module = importlib.import_module(BACKENDS[backend])
        return module.render(self, **options)

    def _repr_png_(self) -> bytes:
        """Show the scene as an image when it is the last line of a Jupyter cell."""
        from .backends.matplotlib import to_png

        return to_png(self)

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        return f"Scene with {len(self.items)} items"
