"""
Minimal 3D geometry library for visualizing 3D shapes and transformations.

The library provides a set of base classes for 3D geometry objects such as points, lines, shapes, and axes
and is independent of any specific visualization library.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod

import numpy as np


def rotate_arbitrary_vector_matrix(
    rotation_axis_vector, angle: float | None = None, in_degrees=True
) -> np.ndarray:
    """Calculate the rotation matrix for an arbitrary rotation axis vector.
    The rotation matrix is calculated using the Rodrigues' rotation formula.
    Args:
        rotation_axis_vector (np.array): The rotation axis vector.
        angle (float): The rotation angle in degrees. If None, the angle is calculated from the rotation_axis_vector.
        in_degrees (bool): If True, the angle is in degrees. If False, the angle is in radians.
    """
    theta = np.linalg.norm(rotation_axis_vector)
    if theta < sys.float_info.epsilon:
        return np.eye(3)

    r = rotation_axis_vector / theta
    if angle is not None:
        theta = np.radians(angle) if in_degrees else angle

    identity = np.eye(3)
    r_rT = np.array(
        [
            [r[0] * r[0], r[0] * r[1], r[0] * r[2]],
            [r[1] * r[0], r[1] * r[1], r[1] * r[2]],
            [r[2] * r[0], r[2] * r[1], r[2] * r[2]],
        ]
    )
    r_cross = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    rotation_mat = np.cos(theta) * identity + (1 - np.cos(theta)) * r_rT + np.sin(theta) * r_cross
    return rotation_mat


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


def split_rotation(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a 3x3 linear map into ``rotation @ stretch`` (polar decomposition).

    ``rotation`` is a proper rotation (orthonormal, determinant +1) and ``stretch`` holds
    everything else: scaling, shear and mirroring. For a pure rotation, ``stretch`` is the
    identity; for a scaling along the rotated axes, ``rotation`` is unchanged.

    Returns:
        (rotation, stretch): two 3x3 arrays with ``rotation @ stretch == matrix``.
    """
    is_rotation = np.allclose(matrix.T @ matrix, np.eye(3)) and np.linalg.det(matrix) > 0
    if is_rotation:
        return matrix.copy(), np.eye(3)

    u, _, vt = np.linalg.svd(matrix)
    if np.linalg.det(u @ vt) < 0:
        # A mirroring map: keep the frame right-handed and leave the flip in the stretch.
        u[:, -1] *= -1
    rotation = u @ vt
    stretch = rotation.T @ matrix
    return rotation, stretch


class Transform:
    """
    Represents an affine transformation in 3D. The transformation is represented by a 4x4 matrix (homogenous).
    """

    def __init__(self):
        """
        Initialize the transform with an identity 4x4 matrix.
        """
        self.matrix = np.eye(4)

    def apply_to_vertex(self, vertex: np.ndarray) -> np.ndarray:
        """
        Applies the current affine transformation to a 3D vertex.
        """
        hom_vertex = np.hstack([vertex, 1])
        transformed = self.matrix @ hom_vertex
        return transformed[:3]

    def apply_to_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """
        Applies the current affine transformation to an array of 3D vertices.

        Args:
            vertices (np.ndarray): An (n, 3) array representing the vertices.

        Returns:
            np.ndarray: The transformed vertices as an (n, 3) array.
        """
        n = vertices.shape[0]
        # Convert to homogeneous coordinates
        hom_vertices = np.hstack([vertices, np.ones((n, 1))])
        transformed = (self.matrix @ hom_vertices.T).T
        return transformed[:, :3]

    def translate(self, translation: np.ndarray) -> Transform:
        """
        Applies a translation.

        Args:
            translation (np.ndarray): A 3-element array.

        Returns:
            Transform: Self to allow method chaining.
        """
        T = np.eye(4)
        T[:3, 3] = np.array(translation)
        self.matrix = T @ self.matrix
        return self

    def scale(self, factors: np.ndarray) -> Transform:
        """
        Applies scaling along each axis.

        Args:
            factors (np.ndarray): A 3-element array for scaling factors.

        Returns:
            Transform: Self to allow method chaining.
        """
        S = np.eye(4)
        S[0, 0], S[1, 1], S[2, 2] = factors
        self.matrix = S @ self.matrix
        return self

    def shear(self, factors: np.ndarray) -> Transform:
        """
        Applies shearing in the xy and xz planes.

        Args:
            factors (np.ndarray): A 4-element array ``(xy, yx, xz, zx)``, where e.g. ``xy``
                is how much x is shifted per unit of y.

        Returns:
            Transform: Self to allow method chaining.
        """
        if len(factors) != 4:
            raise ValueError(f"shear expects 4 factors (xy, yx, xz, zx), got {len(factors)}")
        S = np.eye(4)
        S[0, 1], S[1, 0] = factors[0], factors[1]
        S[0, 2], S[2, 0] = factors[2], factors[3]
        self.matrix = S @ self.matrix
        return self

    def rotate_x(self, angle: float, in_degrees: bool = True) -> Transform:
        """
        Rotates around the x-axis.

        Args:
            angle (float): The rotation angle.
            in_degrees (bool): If True, converts degrees to radians.

        Returns:
            Transform: Self to allow method chaining.
        """
        if in_degrees:
            angle = np.radians(angle)

        R = np.eye(4)
        R[1, 1] = np.cos(angle)
        R[1, 2] = -np.sin(angle)
        R[2, 1] = np.sin(angle)
        R[2, 2] = np.cos(angle)
        self.matrix = R @ self.matrix
        return self

    def rotate_y(self, angle: float, in_degrees: bool = True) -> Transform:
        """
        Rotates around the y-axis.

        Args:
            angle (float): The rotation angle.
            in_degrees (bool): If True, converts degrees to radians.

        Returns:
            Transform: Self to allow method chaining.
        """
        if in_degrees:
            angle = np.radians(angle)
        R = np.eye(4)
        R[0, 0] = np.cos(angle)
        R[0, 2] = np.sin(angle)
        R[2, 0] = -np.sin(angle)
        R[2, 2] = np.cos(angle)
        self.matrix = R @ self.matrix
        return self

    def rotate_z(self, angle: float, in_degrees: bool = True) -> Transform:
        """
        Rotates around the z-axis.

        Args:
            angle (float): The rotation angle.
            in_degrees (bool): If True, converts degrees to radians.

        Returns:
            Transform: Self to allow method chaining.
        """
        if in_degrees:
            angle = np.radians(angle)
        R = np.eye(4)
        R[0, 0] = np.cos(angle)
        R[0, 1] = -np.sin(angle)
        R[1, 0] = np.sin(angle)
        R[1, 1] = np.cos(angle)
        self.matrix = R @ self.matrix
        return self

    def rotate_arbitrary(self, axis_vector: np.ndarray, angle: float, in_degrees: bool = True) -> Transform:
        """
        Rotates around an arbitrary axis. This can be used to rotate around the local axis of a shape.

        Args:
            axis_vector (np.ndarray):  The arbitrary axis vector to rotate around.
            angle (float): The rotation angle.
            in_degrees (bool): If True, the angle is in degrees, else in radians.

        Returns:
            Transform: Self to allow method chaining.
        """
        R = np.eye(4)
        R[:3, :3] = rotate_arbitrary_vector_matrix(axis_vector, angle, in_degrees)
        self.matrix = R @ self.matrix
        return self

    def rotate(
        self,
        axis: str | np.ndarray,
        angle: float,
        in_degrees: bool = True,
        *,
        about: np.ndarray | None = None,
    ) -> Transform:
        """
        Rotates around a specified axis.

        Without ``about`` the rotation axis passes through the origin. With ``about``, it
        passes through that point instead: the point is moved to the origin, the rotation
        is applied, and the point is moved back.

        Args:
            axis (str or array-like): The axis direction ('x', 'y', 'z') or a direction vector.
            angle (float): The rotation angle.
            in_degrees (bool): If True, converts degrees to radians.
            about (array-like, optional): A point the rotation axis passes through.

        Returns:
            Transform: Self to allow method chaining.

        Example:
            >>> Transform().rotate("z", 90, about=[1, 0, 0])  # spin around the line x=1, y=0
        """
        if about is not None:
            pivot = np.asarray(about, dtype=float)
            return self.translate(-pivot).rotate(axis, angle, in_degrees).translate(pivot)

        if not isinstance(axis, str):
            self.rotate_arbitrary(np.asarray(axis, dtype=float), angle, in_degrees)
        elif axis.lower() == "x":
            self.rotate_x(angle, in_degrees)
        elif axis.lower() == "y":
            self.rotate_y(angle, in_degrees)
        elif axis.lower() == "z":
            self.rotate_z(angle, in_degrees)
        else:
            raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z' or provide a rotation axis vector.")
        return self

    def apply(self, other: BaseGeometry) -> BaseGeometry:
        """
        Applies the transform to a geometry object (in place).

        Args:
            other (BaseGeometry): The geometry (or group) to transform.

        Returns:
            The same object, so calls can be chained.
        """
        if not hasattr(other, "apply_transform"):
            raise TypeError(f"Cannot apply a Transform to {type(other).__name__}.")
        other.apply_transform(self)
        return other

    def __matmul__(self, other: Transform | BaseGeometry) -> Transform | BaseGeometry:
        """
        ``t1 @ t2`` composes two transforms into a new one (``t2`` is applied first).
        ``t @ shape`` applies the transform to the shape in place and returns it.
        """
        # TODO: Think about whether `t @ shape` should return a transformed copy instead.
        if isinstance(other, Transform):
            composed = Transform()
            composed.matrix = self.matrix @ other.matrix
            return composed
        return self.apply(other)


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


class BaseGeometry(ABC):
    """
    Base class for all geometry classes.
    Every geometry contains vertices (np.ndarrray) as underlying properties."""

    @property
    @abstractmethod
    def vertices(self) -> np.ndarray:
        """Return the vertices of the geometry."""

    @vertices.setter
    @abstractmethod
    def vertices(self, value: np.ndarray):
        """Set the vertices of the geometry."""

    @property
    def center(self) -> np.ndarray:
        """The mean of all vertices. Shapes override this with their defined center."""
        return self.vertices.mean(axis=0)

    def apply_transform(self, transform: Transform) -> None:
        """Transform all vertices in place. Subclasses extend this to update derived state."""
        self.vertices = transform.apply_to_vertices(self.vertices)
        self.update_geometry()

    def rotate(
        self,
        axis: str | np.ndarray,
        angle: float,
        in_degrees: bool = True,
        *,
        about: np.ndarray | str | None = None,
    ) -> None:
        """
        Rotate the geometry in place.

        Args:
            axis (str or array-like): The axis direction ('x', 'y', 'z') or a direction vector.
            angle (float): The rotation angle.
            in_degrees (bool): If True, the angle is in degrees, else in radians.
            about (array-like or "center", optional): A point the rotation axis passes through.
                ``"center"`` uses the geometry's own center. Default: the origin.

        Example:
            >>> box.rotate("z", 45, about="center")  # spin the box in place
        """
        pivot = resolve_pivot(about, self)
        self.apply_transform(Transform().rotate(axis, angle, in_degrees, about=pivot))

    def anchor(self, name: str = "center") -> np.ndarray:
        """
        Return a named point of the geometry in world coordinates.

        Every geometry has the ``"center"`` anchor; lines and shapes add their own names.
        """
        if isinstance(name, str) and name == "center":
            return self.center
        raise ValueError(f"{type(self).__name__} has no anchor {name!r}; only 'center'.")

    def place(self, anchor: str | np.ndarray | float, at: np.ndarray) -> BaseGeometry:
        """
        Move the geometry (without turning it) so that its ``anchor`` ends up at ``at``.

        Returns the geometry itself, so it can be used right after creating it.

        Example:
            >>> lamp.place("bottom", at=table.anchor("top"))  # put the lamp on the table
        """
        offset = np.asarray(at, dtype=float) - self.anchor(anchor)
        self.apply_transform(Transform().translate(offset))
        return self

    def update_geometry(self) -> None:  # noqa: B027 - optional hook, empty by design
        """
        Hook for updating any geometry representation (e.g. redrawing).
        """


class BasePoint(BaseGeometry):
    """
    Represents a single 3D point or vertex.
    """

    def __init__(self, position: np.ndarray, label: str | None = None):
        """
        Initialize a single point.

        Args:
            position (np.ndarray): The position of the point.
            label (Optional[str]): An optional label.
        """
        self.position = np.array(position)
        self.label = label if label is not None else ""

    @property
    def vertex(self) -> np.ndarray:
        """Return the position vertex."""
        return self.position

    @vertex.setter
    def vertex(self, value: np.ndarray):
        self.position = value

    @property
    def vertices(self) -> np.array:
        return np.reshape(self.position, (1, 3))

    @vertices.setter
    def vertices(self, value):
        self.position = value.flatten()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} at {self.position}"


class BaseLine(BaseGeometry):
    def __init__(
        self,
        start: np.ndarray,
        end: np.ndarray,
        label: str | None = None,
    ):

        self.start = np.array(start)
        self.end = np.array(end)
        self.label = label if label is not None else ""
        self._vertices = np.array([self.start, self.end])

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @vertices.setter
    def vertices(self, value: np.ndarray):
        self._vertices = value
        self.start = value[0]
        self.end = value[1]

    @property
    def edges(self) -> np.ndarray:
        return self.vertices

    def anchor(self, name: str | float = "midway") -> np.ndarray:
        """
        Return a point along the line, named as in TikZ.

        Args:
            name: ``"start"``, ``"end"``, ``"midway"`` (or ``"center"``), ``"near start"``,
                ``"near end"``, ``"very near start"``, ``"very near end"``, or a number
                like TikZ's ``pos``: 0 is the start, 1 the end.

        Example:
            >>> line.anchor("near end")  # three quarters of the way along
        """
        return point_on_segment(self.start, self.end, name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} from {self.start} to {self.end}"


class BaseAxis(BaseGeometry):
    def __init__(
        self,
        direction: np.ndarray,
        position: np.ndarray | None = None,
        anchor: str = "center",
        label: str | None = None,
    ):
        """
        Initialize an axis with an origin and direction.

        Args:
            direction (np.ndarray): The direction vector (will be normalized).
            position (np.ndarray | None): The origin of the axis (default is [0,0,0]).
            anchor (str): (Reserved for future use – similar to BaseShape).
            label (Optional[str]): An optional label.
        """
        self.origin = np.array(position) if position is not None else np.zeros(3)
        # Normalize direction to keep it as a unit vector.
        self.direction = np.array(direction)
        norm = np.linalg.norm(self.direction)
        if norm:
            self.direction = self.direction / norm
        self.label = label if label is not None else ""
        self._vertices = self.calc_initial_vertices()

    def calc_initial_vertices(self) -> np.ndarray:
        """
        Calculates two endpoints of the axis.
        By default, it creates a line segment starting at the origin and
        extending one unit along the direction.
        """
        length = 1.0
        return np.array([self.origin, self.origin + self.direction * length])

    @property
    def vertices(self) -> np.ndarray:
        """Return the current endpoints (vertices) of the axis."""
        return self._vertices

    @vertices.setter
    def vertices(self, value: np.ndarray):
        """Set the endpoints; origin and direction are updated to match."""
        self._vertices = np.asarray(value)
        self.origin = self._vertices[0]
        vec = self._vertices[1] - self._vertices[0]
        norm = np.linalg.norm(vec)
        self.direction = vec / norm if norm else vec

    @property
    def edges(self) -> np.ndarray:
        """
        For an axis, the edge is defined by the line connecting its two endpoints.
        """
        return self.vertices

    def anchor(self, name: str | float = "midway") -> np.ndarray:
        """Return a point along the axis segment; the same names as :meth:`BaseLine.anchor`."""
        return point_on_segment(self.vertices[0], self.vertices[1], name)

    def update_geometry(self):
        """
        Hook for updating any drawn geometry of the axis.
        Subclasses may, for example, re-draw a line.
        """

    def __repr__(self):
        return f"{self.__class__.__name__} at {self.origin} with direction {self.direction}"


def ring(radius: float, z: float, n: int = 16) -> np.ndarray:
    """Return ``n`` points on a horizontal circle of the given radius at height ``z``."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n, z)])


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


class BaseShape(BaseGeometry):
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

    def __init__(
        self,
        position: np.ndarray | list,
        dimensions: np.ndarray | None = None,
        vertices: np.ndarray | None = None,
        anchor: str | np.ndarray | None = None,
    ):
        """
        Initialize a shape at ``position`` from its dimensions or explicit vertices.

        The local axes x', y', z' initially line up with the world axes x, y, z.

        Args:
            position (array-like): Where the shape's ``anchor`` is placed.
            dimensions (array-like): The dimensions of the shape, e.g. [width, length, height]
                for a box. Their meaning depends on the shape.
            vertices (array-like): Explicit vertices, relative to ``position``. If None, they are
                calculated from the dimensions.
            anchor (str or array-like): Which point of the shape ends up at ``position``,
                e.g. ``"bottom"`` or ``"bottom front left"``. Defaults to ``default_anchor``.
                Ignored for explicit ``vertices``, which are always relative to ``position``.

        Example:
            >>> table = BaseBox([0, 0, 0], dimensions=[2, 1, 0.1])
            >>> cup = BaseCylinder(table.anchor("top"), dimensions=[0.1, 0.2])  # stands on the table
        """
        # TODO: think about a good way for the dimensions parameter. Either different keywords for
        # different shapes or a general approach where the dimensions are interpreted differently.
        self.dimensions = np.array(dimensions)
        self.pose = Transform().translate(position)
        local = self.calc_initial_vertices() if vertices is None else vertices
        self._local_vertices = np.asarray(local, dtype=float)

        if vertices is None:
            anchor = self.default_anchor if anchor is None else anchor
            # No update_geometry() here: subclasses have not finished their setup yet.
            self.pose.matrix[:3, 3] += position - self.anchor(anchor)

    @abstractmethod
    def calc_initial_vertices(self) -> np.ndarray:
        """Return the vertices in local coordinates, i.e. around the local origin."""

    # ------------------------------------------------------------ derived from the pose

    @property
    def center(self) -> np.ndarray:
        """The local origin in world coordinates."""
        return self.pose.matrix[:3, 3].copy()

    @center.setter
    def center(self, value):
        """Move the shape so that its center ends up at ``value`` (orientation is kept)."""
        self.pose.matrix[:3, 3] = value
        self.update_geometry()

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

    @property
    def faces(self) -> np.ndarray:
        """
        Returns an array of faces, where each face is defined by vertices.
        """

    @property
    def edges(self) -> np.ndarray:
        """
        Returns an array of edges, where each edge is defined by two vertices.
        """
        raise NotImplementedError("Subclasses must implement the edges property.")

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
        self.update_geometry()

    def rotate(
        self,
        axis: str | np.ndarray,
        angle: float,
        in_degrees: bool = True,
        *,
        about: np.ndarray | str | None = None,
    ) -> None:
        """
        Rotate the shape in place, like :meth:`BaseGeometry.rotate`.

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

    def update_geometry(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__} at {self.center} with vertices {self.vertices}"


class BasePolygon(BaseShape):
    """
    Represents a 2D (planar) polygon in 3D space. The polygon is defined by its vertices.
    """

    def __init__(self, position: np.ndarray, vertices: np.ndarray):
        """
        Args:
            position (array-like): Where the polygon's local origin is placed.
            vertices (array-like): The corner points, relative to ``position``.
        """
        super().__init__(position, vertices=vertices)

    def calc_initial_vertices(self) -> np.ndarray:
        raise ValueError("A polygon needs explicit vertices.")

    @property
    def faces(self) -> np.array:
        """
        Returns an array of faces, where each face is defined by vertices.
        Here we assume the polygon is planar and has only one face.
        """
        return np.array([list(self.vertices)])

    @property
    def edges(self) -> np.array:
        """
        Returns an array of edges, where each edge is defined by two vertices.
        """
        return np.array(
            [
                [self.vertices[i], self.vertices[(i + 1) % len(self.vertices)]]
                for i in range(len(self.vertices))
            ]
        )

    def update_geometry(self) -> None:
        """
        Hook for updating any drawn geometry of the polygon.
        """


class BaseRectangle(BaseShape):
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

    @property
    def faces(self) -> np.ndarray:
        return np.array(
            [
                [self.vertices[0], self.vertices[1], self.vertices[2], self.vertices[3]],
            ]
        )

    @property
    def edges(self) -> np.ndarray:
        return np.array(
            [
                [self.vertices[0], self.vertices[1]],
                [self.vertices[1], self.vertices[2]],
                [self.vertices[2], self.vertices[3]],
                [self.vertices[3], self.vertices[0]],
            ]
        )

    def update_geometry(self) -> None:
        pass


class BaseCircle(BaseShape):
    def calc_initial_vertices(self) -> np.ndarray:
        radius = self.dimensions[0]
        return ring(radius, z=0)

    @property
    def faces(self) -> np.ndarray:
        return np.array(
            [
                [self.vertices[i], self.vertices[(i + 1) % len(self.vertices)], self.center]
                for i in range(len(self.vertices))
            ]
        )

    @property
    def edges(self) -> np.ndarray:
        return np.array(
            [
                [self.vertices[i], self.vertices[(i + 1) % len(self.vertices)]]
                for i in range(len(self.vertices))
            ]
        )

    def __repr__(self) -> str:
        return f"BaseCircle at {self.center} with radius {self.dimensions[0]}"


class BaseBox(BaseShape):
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

    @property
    def faces(self) -> np.ndarray:
        return np.array(
            [
                [self.vertices[0], self.vertices[1], self.vertices[2], self.vertices[3]],  # Front face
                [self.vertices[4], self.vertices[5], self.vertices[6], self.vertices[7]],  # Back face
                [self.vertices[0], self.vertices[1], self.vertices[5], self.vertices[4]],  # Bottom face
                [self.vertices[3], self.vertices[2], self.vertices[6], self.vertices[7]],  # Top face
                [self.vertices[0], self.vertices[3], self.vertices[7], self.vertices[4]],  # Left face
                [self.vertices[1], self.vertices[2], self.vertices[6], self.vertices[5]],  # Right face
            ]
        )

    @property
    def edges(self) -> np.ndarray:
        return np.array(
            [
                [self.vertices[0], self.vertices[1]],
                [self.vertices[1], self.vertices[2]],  # Front face
                [self.vertices[2], self.vertices[3]],
                [self.vertices[3], self.vertices[0]],  # Front face
                [self.vertices[4], self.vertices[5]],
                [self.vertices[5], self.vertices[6]],  # Back face
                [self.vertices[6], self.vertices[7]],
                [self.vertices[7], self.vertices[4]],  # Back face
                [self.vertices[0], self.vertices[4]],
                [self.vertices[1], self.vertices[5]],  # Side faces
                [self.vertices[2], self.vertices[6]],
                [self.vertices[3], self.vertices[7]],  # Side faces
            ]
        )

    def update_geometry(self) -> None:
        pass


class BaseCylinder(BaseShape):
    #: ``position`` is the middle of the bottom face, so a cylinder stands on the given point.
    default_anchor = "bottom"

    def calc_initial_vertices(self) -> np.ndarray:
        radius, height = self.dimensions
        return np.vstack([ring(radius, z=-height / 2), ring(radius, z=height / 2)])

    @property
    def faces(self) -> list:
        # using list here because of the uneven number of faces for the different parts
        return list(self.side_faces) + list(self.top_faces) + list(self.bottom_faces)

    @property
    def side_faces(self):
        n = len(self.vertices) // 2
        v = self.vertices
        return np.array([[v[i], v[(i + 1) % n], v[n + (i + 1) % n], v[n + i]] for i in range(n)])

    @property
    def bottom_faces(self) -> np.ndarray:
        n = len(self.vertices) // 2
        return self.vertices[np.newaxis, :n]

    @property
    def top_faces(self) -> np.ndarray:
        n = len(self.vertices) // 2
        return self.vertices[np.newaxis, n:]

    @property
    def edges(self) -> np.ndarray:
        """The edges of the bottom ring followed by the edges of the top ring."""
        n = len(self.vertices) // 2
        v = self.vertices
        bottom = [[v[i], v[(i + 1) % n]] for i in range(n)]
        top = [[v[n + i], v[n + (i + 1) % n]] for i in range(n)]
        return np.array(bottom + top)

    def update_geometry(self) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"BaseCylinder at {self.center} with radius {self.dimensions[0]} and height {self.dimensions[1]}"
        )


class BaseCone(BaseCylinder):
    def calc_initial_vertices(self) -> np.ndarray:
        lower_radius, upper_radius, height = self.dimensions
        return np.vstack([ring(lower_radius, z=-height / 2), ring(upper_radius, z=height / 2)])

    def __repr__(self) -> str:
        lower_radius, upper_radius, height = self.dimensions
        return (
            f"BaseCone at {self.center} with radii {lower_radius} (bottom), "
            f"{upper_radius} (top) and height {height}"
        )
