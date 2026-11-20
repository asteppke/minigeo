"""
The concrete shapes: flat ones (polygon, rectangle, circle) and solids (box, cylinder, cone).

Each shape lists its faces as vertex indices in ``_face_layout`` (see :attr:`BaseShape.face_indices`). The
vertices of every face go counter-clockwise when seen from outside the shape, so the
face normals point outward. Flat shapes have a single face whose front side is +z'.
"""

from __future__ import annotations

import numpy as np

from .shape import BaseShape, ring


class BasePolygon(BaseShape):
    """A flat polygon in 3D space, defined by its corner points."""

    def __init__(self, position: np.ndarray, vertices: np.ndarray):
        """
        Args:
            position (array-like): Where the polygon's local origin is placed.
            vertices (array-like): The corner points, relative to ``position``, in order
                around the polygon.
        """
        super().__init__(position, vertices=vertices)

    def calc_initial_vertices(self) -> np.ndarray:
        raise ValueError("A polygon needs explicit vertices.")

    def _face_layout(self) -> list[list[int]]:
        return [list(range(len(self._local_vertices)))]


class BaseRectangle(BaseShape):
    """A flat rectangle in the x'y' plane; ``dimensions`` are ``(width, length)``."""

    def calc_initial_vertices(self) -> np.ndarray:
        width, length = self.dimensions
        return np.array(
            [
                [-width / 2, -length / 2, 0],
                [width / 2, -length / 2, 0],
                [width / 2, length / 2, 0],
                [-width / 2, length / 2, 0],
            ]
        )

    def _face_layout(self) -> list[list[int]]:
        return [[0, 1, 2, 3]]


class BaseCircle(BaseShape):
    """A flat disc in the x'y' plane, drawn as a regular polygon; ``dimensions`` are ``(radius,)``."""

    def calc_initial_vertices(self) -> np.ndarray:
        radius = self.dimensions[0]
        return ring(radius, z=0)

    def _face_layout(self) -> list[list[int]]:
        return [list(range(len(self._local_vertices)))]

    def __repr__(self) -> str:
        return f"BaseCircle at {self.center} with radius {self.dimensions[0]}"


class BaseBox(BaseShape):
    """
    A box; ``dimensions`` are ``(width, length, height)`` along x', y', z'.

    Vertices 0-3 are the bottom corners and 4-7 the top corners, both going
    counter-clockwise seen from above, starting at the front left.
    """

    is_closed = True

    #: The order of :attr:`face_indices`, e.g. for giving each face its own color.
    FACE_NAMES = ("bottom", "top", "front", "back", "left", "right")

    def calc_initial_vertices(self) -> np.ndarray:
        width, length, height = self.dimensions
        return np.array(
            [
                [-width / 2, -length / 2, -height / 2],
                [width / 2, -length / 2, -height / 2],
                [width / 2, length / 2, -height / 2],
                [-width / 2, length / 2, -height / 2],
                [-width / 2, -length / 2, height / 2],
                [width / 2, -length / 2, height / 2],
                [width / 2, length / 2, height / 2],
                [-width / 2, length / 2, height / 2],
            ]
        )

    def _face_layout(self) -> list[list[int]]:
        return [
            [0, 3, 2, 1],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [3, 7, 6, 2],  # back
            [0, 4, 7, 3],  # left
            [1, 2, 6, 5],  # right
        ]


class BaseCylinder(BaseShape):
    """
    A cylinder along z'; ``dimensions`` are ``(radius, height)``.

    The vertices are two rings of ``n`` points: the bottom ring first, then the top ring.
    """

    #: ``position`` is the middle of the bottom face, so a cylinder stands on the given point.
    default_anchor = "bottom"
    is_closed = True

    def calc_initial_vertices(self) -> np.ndarray:
        radius, height = self.dimensions
        return np.vstack([ring(radius, z=-height / 2), ring(radius, z=height / 2)])

    @property
    def _ring_size(self) -> int:
        return len(self._local_vertices) // 2

    def _side_layout(self) -> list[list[int]]:
        n = self._ring_size
        return [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]

    def _bottom_layout(self) -> list[list[int]]:
        return [list(reversed(range(self._ring_size)))]  # clockwise from above: outward is down

    def _top_layout(self) -> list[list[int]]:
        n = self._ring_size
        return [list(range(n, 2 * n))]

    def _face_layout(self) -> list[list[int]]:
        return self._side_layout() + self._bottom_layout() + self._top_layout()

    @property
    def side_face_indices(self) -> list[list[int]]:
        return self._oriented(self._side_layout())

    @property
    def bottom_face_indices(self) -> list[list[int]]:
        return self._oriented(self._bottom_layout())

    @property
    def top_face_indices(self) -> list[list[int]]:
        return self._oriented(self._top_layout())

    @property
    def edge_indices(self) -> np.ndarray:
        """Only the two rings: the vertical seams between side faces are not outlined."""
        n = self._ring_size
        bottom = [[i, (i + 1) % n] for i in range(n)]
        top = [[n + i, n + (i + 1) % n] for i in range(n)]
        return np.array(bottom + top)

    @property
    def side_faces(self) -> list[np.ndarray]:
        return self._faces_from(self.side_face_indices)

    @property
    def bottom_faces(self) -> list[np.ndarray]:
        return self._faces_from(self.bottom_face_indices)

    @property
    def top_faces(self) -> list[np.ndarray]:
        return self._faces_from(self.top_face_indices)

    def __repr__(self) -> str:
        return (
            f"BaseCylinder at {self.center} with radius {self.dimensions[0]} and height {self.dimensions[1]}"
        )


class BaseCone(BaseCylinder):
    """A cone (or frustum) along z'; ``dimensions`` are ``(bottom radius, top radius, height)``."""

    def calc_initial_vertices(self) -> np.ndarray:
        lower_radius, upper_radius, height = self.dimensions
        return np.vstack([ring(lower_radius, z=-height / 2), ring(upper_radius, z=height / 2)])

    def __repr__(self) -> str:
        lower_radius, upper_radius, height = self.dimensions
        return (
            f"BaseCone at {self.center} with radii {lower_radius} (bottom), "
            f"{upper_radius} (top) and height {height}"
        )
