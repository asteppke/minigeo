"""
Named points on geometries, in the spirit of TikZ.

Shapes and groups use words like ``"top front left"``; lines use positions like ``"near end"``.
"""

from __future__ import annotations

import numpy as np

# Each anchor word picks one side of the shape along one of its local axes.
ANCHOR_WORDS = {
    "left": (0, -1),
    "right": (0, +1),
    "front": (1, -1),
    "back": (1, +1),
    "bottom": (2, -1),
    "top": (2, +1),
}


def anchor_direction(anchor: str | np.ndarray) -> np.ndarray:
    """
    Translate an anchor into a position within the shape's local bounding box.

    The result has one entry per local axis: -1 is the low side, +1 the high side, 0 the middle.

    Args:
        anchor: Either ``"center"`` or a combination of the words left/right (x'),
            front/back (y'), bottom/top (z') in any order, separated by spaces, ``-`` or ``_``,
            e.g. ``"top"``, ``"top front"``, ``"bottom-back-left"``. Or three numbers, where
            ``(0, 0, 0.5)`` is halfway between the center and the top.

    Example:
        >>> anchor_direction("top front left")
        array([-1., -1.,  1.])
    """
    if not isinstance(anchor, str):
        direction = np.asarray(anchor, dtype=float)
        if direction.shape != (3,):
            raise ValueError(f"A numeric anchor needs three values, got {anchor!r}.")
        return direction

    words = anchor.lower().replace("-", " ").replace("_", " ").split()
    direction = np.zeros(3)
    if words == ["center"]:
        return direction

    used_axes = set()
    for word in words:
        if word not in ANCHOR_WORDS:
            raise ValueError(
                f"Unknown anchor word {word!r} in {anchor!r}; use {', '.join(ANCHOR_WORDS)} or 'center'."
            )
        axis, side = ANCHOR_WORDS[word]
        if axis in used_axes:
            raise ValueError(f"Anchor {anchor!r} names the same axis twice.")
        used_axes.add(axis)
        direction[axis] = side
    if not words:
        raise ValueError("Empty anchor name.")
    return direction


def point_in_bounding_box(vertices: np.ndarray, anchor: str | np.ndarray) -> np.ndarray:
    """
    Return the anchor point on the axis-aligned bounding box of ``vertices``.

    Example:
        >>> point_in_bounding_box(box.local_vertices, "top")  # middle of the top face
    """
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    middle, half_size = (low + high) / 2, (high - low) / 2
    return middle + anchor_direction(anchor) * half_size


# TikZ's names for positions along a path, from 0 (start) to 1 (end).
LINE_POSITIONS = {
    "start": 0.0,
    "very near start": 0.125,
    "near start": 0.25,
    "midway": 0.5,
    "center": 0.5,
    "near end": 0.75,
    "very near end": 0.875,
    "end": 1.0,
}


def line_position(anchor: str | float) -> float:
    """
    Translate a TikZ position name (e.g. ``"near end"``) or a number into a fraction of the way.

    Example:
        >>> line_position("near start")
        0.25
    """
    if not isinstance(anchor, str):
        return float(anchor)
    name = " ".join(anchor.lower().replace("-", " ").replace("_", " ").split())
    if name not in LINE_POSITIONS:
        raise ValueError(f"Unknown line anchor {anchor!r}; use {', '.join(LINE_POSITIONS)} or a number.")
    return LINE_POSITIONS[name]


def point_on_segment(start: np.ndarray, end: np.ndarray, anchor: str | float) -> np.ndarray:
    """Return the point ``anchor`` along the straight segment from ``start`` to ``end``."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    return start + line_position(anchor) * (end - start)


def resolve_pivot(about: np.ndarray | str | None, owner) -> np.ndarray | None:
    """
    Turn the ``about`` argument of ``rotate`` into a point.

    ``None`` stays ``None`` (rotate about the origin), ``"center"`` becomes ``owner.center``,
    and anything else is treated as a point.
    """
    if isinstance(about, str):
        if about != "center":
            raise ValueError(f"Unknown pivot {about!r}; use 'center' or a point.")
        return owner.center
    return about
