"""Shapes with a pose: their own coordinate system x', y', z'."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from ..anchors import point_in_bounding_box
from ..transform import Transform, split_rotation
from .base import Geometry


def face_normal(points: np.ndarray) -> np.ndarray:
    """
    The unit normal of a (roughly planar) face whose corners go counter-clockwise.

    Uses Newell's method, which also works for faces with more than three corners.
    """
    points = np.asarray(points, dtype=float)
    following = np.roll(points, -1, axis=0)
    normal = np.cross(points, following).sum(axis=0)
    length = np.linalg.norm(normal)
    return normal / length if length else normal


def ring(radius: float, z: float, n: int = 16) -> np.ndarray:
    """Return ``n`` points on a horizontal circle of the given radius at height ``z``."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n, z)])


class Shape(Geometry):
    """
    A shape with its own coordinate system x', y', z' (its *pose*).

    The vertices are stored in the shape's local coordinates, around the local origin.
    ``pose`` is a rigid transform (rotation + translation) that places them in the world::

        world_vertices = pose @ local_vertices

    From the pose follow:

    * ``center``: the local origin in world coordinates
    * ``axes``: the local x', y', z' directions in world coordinates (always perpendicular unit vectors)

    Applying a transform updates the pose with its rotation and translation. Any scaling or
    shear is applied to the local vertices instead, so the shape deforms while its axes stay
    perpendicular.

    Anchors (as in TikZ) name points on the shape's local bounding box: ``"top"`` is the middle
    of the top face, ``"top front left"`` a corner, and so on (see :func:`anchor_direction`).
    Seen from the front, x' points right, y' points away from you (to the back) and z' up.
    """

    #: Which anchor ``position`` refers to when none is given. ``"center"`` means the middle.
    default_anchor: str = "center"

    #: Whether the faces enclose a volume (a solid) rather than forming a flat sheet.
    is_closed: bool = False

    def __init__(
        self,
        position: np.ndarray | list,
        *,
        vertices: np.ndarray | None = None,
        anchor: str | np.ndarray | None = None,
    ):
        """
        Place the shape at ``position``. Subclasses set their sizes before calling this.

        The local axes x', y', z' initially line up with the world axes x, y, z.

        Args:
            position (array-like): Where the shape's ``anchor`` is placed.
            vertices (array-like): Explicit vertices, relative to ``position``. If None, they are
                calculated by :meth:`calc_initial_vertices` from the shape's sizes.
            anchor (str or array-like): Which point of the shape ends up at ``position``,
                e.g. ``"bottom"`` or ``"bottom front left"``. Defaults to ``default_anchor``.
                Ignored for explicit ``vertices``, which are always relative to ``position``.

        Example:
            >>> table = Box([0, 0, 0], size=[2, 1, 0.1])
            >>> cup = Cylinder(table.anchor("top"), radius=0.1, height=0.2)  # stands on the table
        """
        position = np.asarray(position, dtype=float)
        self.pose = Transform().translate(position)
        local = self.calc_initial_vertices() if vertices is None else vertices
        self._local_vertices = np.asarray(local, dtype=float)
        self._mirrored = False  # set by mirroring transforms, see face_indices

        if vertices is None:
            anchor = self.default_anchor if anchor is None else anchor
            self.pose.matrix[:3, 3] += position - self.anchor(anchor)

    def calc_initial_vertices(self) -> np.ndarray:
        """Return the vertices in local coordinates, i.e. around the local origin."""
        raise NotImplementedError(f"{type(self).__name__} needs explicit vertices.")

    # ------------------------------------------------------------ derived from the pose

    @property
    def center(self) -> np.ndarray:
        """The local origin in world coordinates."""
        return self.pose.matrix[:3, 3].copy()

    @center.setter
    def center(self, value):
        """Move the shape so that its center ends up at ``value`` (orientation is kept)."""
        self.pose.matrix[:3, 3] = value

    @property
    def axes(self) -> np.ndarray:
        """The local axes as rows ``[x', y', z']``: perpendicular unit vectors in world coordinates."""
        return self.pose.matrix[:3, :3].T.copy()

    def anchor(self, name: str | np.ndarray = "center") -> np.ndarray:
        """
        Return a named point of the shape in world coordinates.

        Anchors lie on the shape's local bounding box and move and turn with the shape.

        Args:
            name: ``"center"``, a combination of left/right, front/back, bottom/top
                (e.g. ``"top front"``), or three numbers between -1 and 1 (see :func:`anchor_direction`).

        Example:
            >>> box.anchor("top")              # middle of the top face
            >>> box.anchor("bottom front left")  # a corner
        """
        return self.pose.apply_to_vertex(point_in_bounding_box(self._local_vertices, name))

    @property
    def local_vertices(self) -> np.ndarray:
        """The vertices in the shape's own coordinate system."""
        return self._local_vertices.copy()

    @property
    def vertices(self) -> np.ndarray:
        """The vertices in world coordinates."""
        return self.pose.apply_to_vertices(self._local_vertices)

    @vertices.setter
    def vertices(self, value):
        """Set the world vertices; they are stored relative to the current pose."""
        rotation, translation = self.pose.matrix[:3, :3], self.pose.matrix[:3, 3]
        # inverse of a rigid transform: R^T (v - t), written for row vectors
        self._local_vertices = (np.asarray(value, dtype=float) - translation) @ rotation

    # ------------------------------------------------------------ mesh: faces and edges

    @abstractmethod
    def _face_layout(self) -> list[list[int]]:
        """The faces as vertex indices, counter-clockwise from outside (before any mirroring)."""

    @property
    def face_indices(self) -> list[list[int]]:
        """
        The faces as lists of vertex indices.

        The vertices of each face go counter-clockwise when seen from outside the shape, so
        the face normal (right-hand rule) points outward. Faces may have different sizes.
        This stays true after mirroring transforms.
        """
        return self._oriented(self._face_layout())

    def _oriented(self, face_indices: list[list[int]]) -> list[list[int]]:
        """Reverse the faces of a mirrored shape, so that their normals point outward again."""
        if self._mirrored:
            return [face[::-1] for face in face_indices]
        return face_indices

    @property
    def edge_indices(self) -> np.ndarray:
        """
        The outline as an ``(m, 2)`` array of vertex index pairs.

        By default, every side of every face, each counted once. Shapes can override this,
        e.g. a cylinder does not outline the seams between its side faces.
        """
        edges = {}
        for face in self.face_indices:
            for a, b in zip(face, face[1:] + face[:1], strict=True):
                edges.setdefault(frozenset((a, b)), (a, b))
        return np.array(list(edges.values()), dtype=int)

    @property
    def faces(self) -> list[np.ndarray]:
        """The faces in world coordinates: one ``(k, 3)`` array of corner points per face."""
        return self._faces_from(self.face_indices)

    @property
    def edges(self) -> np.ndarray:
        """The outline in world coordinates: an ``(m, 2, 3)`` array of start and end points."""
        return self.vertices[self.edge_indices]

    def _faces_from(self, face_indices: list[list[int]]) -> list[np.ndarray]:
        vertices = self.vertices
        return [vertices[face] for face in face_indices]

    # ------------------------------------------------------------ meeting lines

    def intersect(self, line, *, segment: bool = False) -> np.ndarray:
        """
        The points where a line meets the shape, sorted along the line.

        Spheres and cylinders are met exactly, other shapes at their flat faces
        (see :func:`minigeo.intersect.intersections`).

        Args:
            line: A Line, Vector, Axis, Polyline, Curve, or a pair of points ``(start, end)``.
            segment: Only points between the line's start and end; otherwise the line
                is extended in both directions, as in :meth:`Plane.intersect`.

        Returns:
            An ``(n, 3)`` array of points; empty if the line misses the shape.

        Example:
            >>> box = Box([0, 0, 0], size=1)
            >>> box.intersect(Line([-1, 0, 0], [1, 0, 0]))  # in at x = -0.5, out at 0.5
        """
        from ..intersect import intersections

        return intersections(line, self, segment=segment)

    # ------------------------------------------------------------ transforming

    def apply_transform(self, transform: Transform) -> None:
        """
        Apply a world transform: rotation and translation move the pose, while scaling and
        shear deform the local vertices.
        """
        linear, offset = transform.matrix[:3, :3], transform.matrix[:3, 3]
        rotation, translation = self.pose.matrix[:3, :3], self.pose.matrix[:3, 3]

        new_rotation, stretch = split_rotation(linear @ rotation)
        self.pose.matrix[:3, :3] = new_rotation
        self.pose.matrix[:3, 3] = linear @ translation + offset

        if not np.allclose(stretch, np.eye(3)):  # skip for rigid transforms, avoids rounding drift
            self._local_vertices = self._local_vertices @ stretch.T
            if np.linalg.det(stretch) < 0:  # mirrored: faces would now wind the wrong way
                self._mirrored = not self._mirrored

    def rotate(
        self,
        axis: str | np.ndarray,
        angle: float,
        in_degrees: bool = True,
        *,
        about: np.ndarray | str | None = None,
    ) -> None:
        """
        Rotate the shape in place, like :meth:`Geometry.rotate`.

        In addition, ``axis`` can be one of the shape's own axes ``"x'"``, ``"y'"`` or ``"z'"``.

        Example:
            >>> box.rotate("x'", 30, about="center")  # tilt the box around its own x' axis
        """
        if isinstance(axis, str) and axis.endswith("'"):
            local_axes = {"x'": 0, "y'": 1, "z'": 2}
            if axis not in local_axes:
                raise ValueError(f'Unknown local axis {axis!r}; use "x\'", "y\'" or "z\'".')
            axis = self.axes[local_axes[axis]]
        super().rotate(axis, angle, in_degrees, about=about)

    def __repr__(self):
        return f"{self.__class__.__name__} at {self.center} with vertices {self.vertices}"
