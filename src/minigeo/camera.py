"""
Where the viewer looks from, shared by all backends.

The angles follow matplotlib: ``elev`` is the angle above the horizontal plane, ``azim``
the angle around the vertical axis, and ``vertical_axis`` says which axis points up on the
screen. So ``Camera(30, -60)`` is matplotlib's default view, and the same camera gives the
same picture in every backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class Camera:
    """
    An orthographic camera looking at the scene from the direction given by two angles.

    Attributes:
        elev: Elevation in degrees, above the plane perpendicular to the vertical axis.
        azim: Azimuth in degrees, around the vertical axis.
        vertical_axis: ``"x"``, ``"y"`` or ``"z"``: the axis that points up on the screen.
    """

    elev: float = 30.0
    azim: float = -60.0
    vertical_axis: str = "z"

    def __post_init__(self):
        if self.vertical_axis not in AXIS_INDEX:
            raise ValueError(f"vertical_axis must be 'x', 'y' or 'z', got {self.vertical_axis!r}.")

    def _to_world(self, vector: np.ndarray) -> np.ndarray:
        """Turn a vector given as if z were vertical into one for the chosen vertical axis."""
        return np.roll(vector, AXIS_INDEX[self.vertical_axis] - 2)

    @property
    def toward_viewer(self) -> np.ndarray:
        """The unit vector from the scene toward the viewer."""
        elev, azim = np.radians(self.elev), np.radians(self.azim)
        return self._to_world(
            np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        )

    @property
    def right(self) -> np.ndarray:
        """The unit vector that points to the right on the screen."""
        azim = np.radians(self.azim)
        return self._to_world(np.array([-np.sin(azim), np.cos(azim), 0.0]))

    @property
    def up(self) -> np.ndarray:
        """The unit vector that points up on the screen."""
        return np.cross(self.toward_viewer, self.right)

    @property
    def light(self) -> np.ndarray:
        """The direction of the light used for shading: from the upper left, behind the viewer."""
        light = self.toward_viewer + 0.6 * self.up - 0.4 * self.right
        return light / np.linalg.norm(light)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points onto the screen.

        Returns:
            (xy, depth): ``(n, 2)`` screen coordinates (x right, y up) in world units, and
            the depth of each point (larger is further away).
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        xy = np.column_stack([points @ self.right, points @ self.up])
        return xy, -(points @ self.toward_viewer)
