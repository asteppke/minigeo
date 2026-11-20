"""
The concrete shapes: flat ones (polygon, rectangle, circle) and solids (box, cylinder, cone,
sphere).

Sizes are named arguments, e.g. ``Box(position, size=(2, 1, 0.5))`` or
``Cylinder(position, radius=0.5, height=2)``. They describe the shape as created; scaling
transforms change the vertices but not these numbers.

Each shape lists its faces as vertex indices in ``_face_layout`` (see :attr:`Shape.face_indices`).
The vertices of every face go counter-clockwise when seen from outside the shape, so the
face normals point outward. Flat shapes have a single face whose front side is +z'.
"""

from __future__ import annotations

import numpy as np

from .shape import Shape, ring

#: Default number of corners for the round shapes (circle, cylinder, cone, sphere).
SEGMENTS = 16


def _sizes(value, count: int, name: str) -> np.ndarray:
    """``count`` positive sizes from a sequence, or a single number used for all of them."""
    sizes = np.broadcast_to(np.asarray(value, dtype=float), (count,)).copy()
    if np.any(sizes < 0):
        raise ValueError(f"{name} must not be negative, got {value!r}.")
    return sizes


def _segments(segments: int) -> int:
    if segments < 3:
        raise ValueError(f"A round shape needs at least 3 segments, got {segments}.")
    return int(segments)


class Polygon(Shape):
    """A flat polygon in 3D space, defined by its corner points."""

    def __init__(self, position: np.ndarray, vertices: np.ndarray):
        """
        Args:
            position (array-like): Where the polygon's local origin is placed.
            vertices (array-like): The corner points, relative to ``position``, in order
                around the polygon.

        Example:
            >>> Polygon([0, 0, 0], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # a triangle
        """
        super().__init__(position, vertices=vertices)

    def _face_layout(self) -> list[list[int]]:
        return [list(range(len(self._local_vertices)))]


class Rectangle(Shape):
    """A flat rectangle in the x'y' plane."""

    def __init__(self, position: np.ndarray, size, *, anchor: str | None = None):
        """
        Args:
            position (array-like): Where the ``anchor`` (default: the center) is placed.
            size: ``(width, length)`` along x' and y', or one number for a square.
            anchor: Which point ends up at ``position``, e.g. ``"front left"``.

        Example:
            >>> Rectangle([0, 0, 0], size=(2, 1))
        """
        self.size = _sizes(size, 2, "size")
        super().__init__(position, anchor=anchor)

    def calc_initial_vertices(self) -> np.ndarray:
        width, length = self.size
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

    def __repr__(self) -> str:
        return f"{type(self).__name__} at {self.center} of size {self.size}"


class Circle(Shape):
    """A flat disc in the x'y' plane, drawn as a regular polygon."""

    def __init__(
        self, position: np.ndarray, radius: float, *, segments: int = SEGMENTS, anchor: str | None = None
    ):
        """
        Args:
            position (array-like): Where the ``anchor`` (default: the center) is placed.
            radius: The radius.
            segments: Number of corners of the polygon that draws the circle.
            anchor: Which point ends up at ``position``.

        Example:
            >>> Circle([0, 0, 0], radius=1, segments=32)
        """
        self.radius = float(_sizes(radius, 1, "radius")[0])
        self.segments = _segments(segments)
        super().__init__(position, anchor=anchor)

    def calc_initial_vertices(self) -> np.ndarray:
        return ring(self.radius, z=0, n=self.segments)

    def _face_layout(self) -> list[list[int]]:
        return [list(range(len(self._local_vertices)))]

    def __repr__(self) -> str:
        return f"{type(self).__name__} at {self.center} with radius {self.radius}"


class Box(Shape):
    """
    A box with edges along its own axes x', y', z'.

    Vertices 0-3 are the bottom corners and 4-7 the top corners, both going
    counter-clockwise seen from above, starting at the front left.
    """

    is_closed = True

    #: The order of :attr:`face_indices`, e.g. for giving each face its own color.
    FACE_NAMES = ("bottom", "top", "front", "back", "left", "right")

    def __init__(self, position: np.ndarray, size, *, anchor: str | None = None):
        """
        Args:
            position (array-like): Where the ``anchor`` (default: the center) is placed.
            size: ``(width, length, height)`` along x', y' and z', or one number for a cube.
            anchor: Which point ends up at ``position``, e.g. ``"bottom"``.

        Example:
            >>> Box([0, 0, 0], size=(2, 1, 0.5))
            >>> Box([0, 0, 0], size=1, anchor="bottom")  # a unit cube standing on the origin
        """
        self.size = _sizes(size, 3, "size")
        super().__init__(position, anchor=anchor)

    def calc_initial_vertices(self) -> np.ndarray:
        width, length, height = self.size
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

    def __repr__(self) -> str:
        return f"{type(self).__name__} at {self.center} of size {self.size}"


class Cylinder(Shape):
    """
    A cylinder along z'. It stands on its position: ``position`` is the middle of the bottom.

    The vertices are two rings of ``segments`` points: the bottom ring first, then the top ring.
    """

    #: ``position`` is the middle of the bottom face, so a cylinder stands on the given point.
    default_anchor = "bottom"
    is_closed = True

    def __init__(
        self,
        position: np.ndarray,
        radius: float,
        height: float,
        *,
        segments: int = SEGMENTS,
        anchor: str | None = None,
    ):
        """
        Args:
            position (array-like): Where the ``anchor`` (default: the middle of the bottom) is placed.
            radius: The radius.
            height: The height along z'.
            segments: Number of side faces.
            anchor: Which point ends up at ``position``, e.g. ``"center"``.

        Example:
            >>> Cylinder([0, 0, 0], radius=0.5, height=2)
        """
        self.radius = float(_sizes(radius, 1, "radius")[0])
        self.height = float(_sizes(height, 1, "height")[0])
        self.segments = _segments(segments)
        super().__init__(position, anchor=anchor)

    def calc_initial_vertices(self) -> np.ndarray:
        h, n = self.height, self.segments
        return np.vstack([ring(self.radius, z=-h / 2, n=n), ring(self.radius, z=h / 2, n=n)])

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
        return f"{type(self).__name__} at {self.center} with radius {self.radius} and height {self.height}"


class Cone(Cylinder):
    """
    A cone along z', standing on its position; with ``top_radius`` > 0 a truncated cone.
    """

    def __init__(
        self,
        position: np.ndarray,
        radius: float,
        height: float,
        *,
        top_radius: float = 0.0,
        segments: int = SEGMENTS,
        anchor: str | None = None,
    ):
        """
        Args:
            position (array-like): Where the ``anchor`` (default: the middle of the bottom) is placed.
            radius: The radius at the bottom.
            height: The height along z'.
            top_radius: The radius at the top; 0 for a pointed cone.
            segments: Number of side faces.
            anchor: Which point ends up at ``position``.

        Example:
            >>> Cone([0, 0, 0], radius=0.5, height=1)                  # pointed
            >>> Cone([0, 0, 0], radius=0.5, height=1, top_radius=0.2)  # a lampshade
        """
        self.top_radius = float(_sizes(top_radius, 1, "top_radius")[0])
        super().__init__(position, radius, height, segments=segments, anchor=anchor)

    def calc_initial_vertices(self) -> np.ndarray:
        h, n = self.height, self.segments
        return np.vstack([ring(self.radius, z=-h / 2, n=n), ring(self.top_radius, z=h / 2, n=n)])

    @property
    def _pointed(self) -> bool:
        return self.top_radius == 0

    def _top_layout(self) -> list[list[int]]:
        return [] if self._pointed else super()._top_layout()  # a tip has no top face

    @property
    def edge_indices(self) -> np.ndarray:
        """The bottom ring, and the top ring unless the cone ends in a tip."""
        edges = super().edge_indices
        return edges[: self._ring_size] if self._pointed else edges

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__} at {self.center} with radius {self.radius} (bottom), "
            f"{self.top_radius} (top) and height {self.height}"
        )


class Sphere(Shape):
    """
    A sphere, drawn as a globe of ``segments`` meridians and ``segments // 2`` parallels.

    Vertex 0 is the north pole (+z'), the last vertex the south pole; between them come the
    parallels from north to south. It has no outline, only shaded faces.
    """

    is_closed = True

    def __init__(
        self, position: np.ndarray, radius: float, *, segments: int = SEGMENTS, anchor: str | None = None
    ):
        """
        Args:
            position (array-like): Where the ``anchor`` (default: the center) is placed.
            radius: The radius.
            segments: Number of meridians; more segments give a rounder sphere.
            anchor: Which point ends up at ``position``, e.g. ``"bottom"`` to rest on it.

        Example:
            >>> Sphere([0, 0, 0], radius=1, segments=24)
        """
        self.radius = float(_sizes(radius, 1, "radius")[0])
        self.segments = _segments(segments)
        super().__init__(position, anchor=anchor)

    @property
    def _parallels(self) -> int:
        return max(self.segments // 2, 2)

    def calc_initial_vertices(self) -> np.ndarray:
        polar_angles = np.linspace(0, np.pi, self._parallels + 1)[1:-1]
        parallels = [
            ring(self.radius * np.sin(angle), z=self.radius * np.cos(angle), n=self.segments)
            for angle in polar_angles
        ]
        return np.vstack([[0, 0, self.radius], *parallels, [0, 0, -self.radius]])

    def _face_layout(self) -> list[list[int]]:
        n, rings = self.segments, self._parallels - 1
        north, south = 0, 1 + rings * n

        def at(ring_index: int, j: int) -> int:
            return 1 + ring_index * n + j % n

        faces = [[north, at(0, j), at(0, j + 1)] for j in range(n)]
        for r in range(rings - 1):
            faces += [[at(r, j), at(r + 1, j), at(r + 1, j + 1), at(r, j + 1)] for j in range(n)]
        faces += [[south, at(rings - 1, j + 1), at(rings - 1, j)] for j in range(n)]
        return faces

    @property
    def edge_indices(self) -> np.ndarray:
        """No outline: a sphere is drawn with shading only."""
        return np.zeros((0, 2), dtype=int)

    def __repr__(self) -> str:
        return f"{type(self).__name__} at {self.center} with radius {self.radius}"
