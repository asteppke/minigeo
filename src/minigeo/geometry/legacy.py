"""
The old class names, kept so that existing code keeps working.

Using them gives a ``DeprecationWarning`` that names the replacement. The old shapes took
their sizes as a ``dimensions`` list; the new ones have named arguments, e.g.
``Box(position, size=...)`` or ``Cylinder(position, radius=..., height=...)``.
"""

from __future__ import annotations

import numpy as np

from .shapes import Box, Circle, Cone, Cylinder, Rectangle


class BaseBox(Box):
    """Old name of :class:`Box`, with ``dimensions=(width, length, height)``."""

    def __init__(self, position, dimensions, anchor=None):
        super().__init__(position, size=dimensions, anchor=anchor)

    @property
    def dimensions(self) -> np.ndarray:
        return self.size


class BaseRectangle(Rectangle):
    """Old name of :class:`Rectangle`, with ``dimensions=(width, length)``."""

    def __init__(self, position, dimensions, anchor=None):
        super().__init__(position, size=dimensions, anchor=anchor)

    @property
    def dimensions(self) -> np.ndarray:
        return self.size


class BaseCircle(Circle):
    """Old name of :class:`Circle`, with ``dimensions=(radius,)``."""

    def __init__(self, position, dimensions, anchor=None):
        (radius,) = np.atleast_1d(dimensions)
        super().__init__(position, radius=radius, anchor=anchor)

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([self.radius])


class BaseCylinder(Cylinder):
    """Old name of :class:`Cylinder`, with ``dimensions=(radius, height)``."""

    def __init__(self, position, dimensions, anchor=None):
        radius, height = dimensions
        super().__init__(position, radius=radius, height=height, anchor=anchor)

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([self.radius, self.height])


class BaseCone(Cone):
    """Old name of :class:`Cone`, with ``dimensions=(bottom radius, top radius, height)``."""

    def __init__(self, position, dimensions, anchor=None):
        radius, top_radius, height = dimensions
        super().__init__(position, radius=radius, height=height, top_radius=top_radius, anchor=anchor)

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([self.radius, self.top_radius, self.height])
