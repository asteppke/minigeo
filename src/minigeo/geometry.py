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

    def update_geometry(self):
        """
        Hook for updating any drawn geometry of the axis.
        Subclasses may, for example, re-draw a line.
        """

    def __repr__(self):
        return f"{self.__class__.__name__} at {self.origin} with direction {self.direction}"


class BaseShape(BaseGeometry):
    def __init__(
        self,
        position: np.ndarray | list,
        dimensions: np.ndarray | None = None,
        vertices: np.ndarray | None = None,
        anchor: str = "center",
        enable_axes: bool = False,
        axes_length: float = 1.0,
    ):
        """
        Initialize a BaseShape which has a position and dimensions or vertices.

        Args:
            position (np.array): The center (anchor) position of the shape.
            dimensions (np.array): The dimensions of the shape, e.g. [width, length, height, ...] for basic shapes.
                                   This is interpreted differently for different shapes.
            vertices (np.array): The vertices of the shape. If None, the vertices are calculated from the dimensions.
                                 The vertices are defined in a counter-clockwise order, relative to the center position.
            anchor (str): The anchor point of the shape. Default is 'center', other options depend on the shape, e.g.
                          'bottom_corner', 'top_corner', 'front_face_center'.
        """
        # TODO: make anchor and shift work correctly.
        # TODO: think about a good way for the dimensions parameter. Either different keywords for different shapes
        # or a general approach where the dimensions are interpreted differently for different shapes.
        self.anchor = anchor
        self._center = np.array(position)
        self.dimensions = np.array(dimensions)
        self.rotation_matrix = np.eye(4)
        self._vertices = self.calc_initial_vertices(vertices) if vertices is None else vertices
        # self._vertices = self._interpret_position(self._vertices, self.dimensions, self.anchor)
        self.axes = self.generate_axes() if enable_axes else None

    def _interpret_position(self, vertices: np.ndarray, dims: np.ndarray, anchor: str) -> np.ndarray:
        """
        Adjusts the given position according to the provided anchor type.
        """
        if anchor == "center":
            return vertices + self._center
        elif anchor == "bottom_corner":
            # pos is the bottom (min) corner; center is half the extents away.
            return vertices + dims / 2
        elif anchor == "top_corner":
            # pos is the top (max) corner.
            return vertices - dims / 2
        elif anchor == "front_face_center":
            # Assume "front" along the negative y-axis.
            return vertices + np.array([0, dims[1] / 2, 0])

        raise ValueError(f"Unrecognized anchor type: {anchor}")

    @property
    def center(self) -> np.array:
        return self._center

    @center.setter
    def center(self, value):
        """Move the shape so that its center ends up at ``value``."""
        value = np.asarray(value, dtype=float)
        self._vertices = self.vertices + (value - self._center)
        self._center = value
        self.update_geometry()

    @abstractmethod
    def calc_initial_vertices(self, vertices) -> np.array:
        pass

    @property
    def vertices(self) -> np.array:
        return np.array(self._vertices)

    @vertices.setter
    def vertices(self, value):
        self._vertices = value

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

    def apply_transform(self, transform: Transform) -> None:
        """Transform the vertices and keep the center and the shape's own axes in sync."""
        self._center = transform.apply_to_vertex(self._center)
        for axis in self.axes or []:
            axis.apply_transform(transform)
        super().apply_transform(transform)

    def update_geometry(self):
        pass

    def generate_axes(self) -> list[BaseAxis]:
        """
        Generate a set of axes for the shape centered at its origin.
        """
        axes = []
        for name, axis in zip("xyz", np.eye(3), strict=True):
            axes.append(BaseAxis(axis, self.center, label=f"{name}'"))
        return axes

    def __repr__(self):
        return f"{self.__class__.__name__} at {self.center} with vertices {self.vertices}"


class BasePolygon(BaseShape):
    """
    Represents a 2D (planar) polygon in 3D space. The polygon is defined by its vertices.
    """

    def __init__(
        self,
        position: np.ndarray,
        vertices: np.ndarray | None = None,
        anchor: str = "center",
    ):
        super().__init__(position, dimensions=None, vertices=vertices, anchor=anchor)

        self._vertices = self.calc_initial_vertices(vertices) if vertices is not None else []
        self._vertices = self._interpret_position(self._vertices, self.dimensions, self.anchor)

    def calc_initial_vertices(self, vertices) -> np.array:
        """
        Calculates the vertices of the polygon.
        """
        return vertices

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
    def calc_initial_vertices(self, vertices=None) -> np.ndarray:
        x, y, z = self.center
        width, length = self.dimensions
        return np.array(
            [
                [x - width / 2, y - length / 2, z],
                [x + width / 2, y - length / 2, z],
                [x + width / 2, y + length / 2, z],
                [x - width / 2, y + length / 2, z],
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
    def calc_initial_vertices(self, vertices=None) -> np.ndarray:
        x, y, z = self.center
        radius = self.dimensions[0]
        n = 16  # Number of vertices
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        vertices = np.zeros((n, 3))
        vertices[:, 0] = x + radius * np.cos(theta)
        vertices[:, 1] = y + radius * np.sin(theta)
        vertices[:, 2] = z
        return vertices

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
    def calc_initial_vertices(self, vertices=None) -> np.ndarray:
        x, y, z = self.center
        width, length, height = self.dimensions
        return np.array(
            [
                [x - width / 2, y - length / 2, z - height / 2],
                [x + width / 2, y - length / 2, z - height / 2],
                [x + width / 2, y + length / 2, z - height / 2],
                [x - width / 2, y + length / 2, z - height / 2],
                [x - width / 2, y - length / 2, z + height / 2],
                [x + width / 2, y - length / 2, z + height / 2],
                [x + width / 2, y + length / 2, z + height / 2],
                [x - width / 2, y + length / 2, z + height / 2],
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
    def calc_initial_vertices(self, vertices=None) -> np.ndarray:
        x, y, z = self.center
        radius, height = self.dimensions
        n = 16  # Number of vertices
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        vertices = np.zeros((n, 3))
        vertices[:, 0] = x + radius * np.cos(theta)
        vertices[:, 1] = y + radius * np.sin(theta)
        vertices[:, 2] = z
        vertices = np.vstack([vertices, vertices + [0, 0, height]])

        return vertices

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
    def calc_initial_vertices(self, vertices=None) -> np.ndarray:
        x, y, z = self.center
        lower_radius, upper_radius, height = self.dimensions
        n = 16  # Number of vertices
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        vertices_bottom, vertices_top = np.zeros((n, 3)), np.zeros((n, 3))
        vertices_bottom[:, 0] = x + lower_radius * np.cos(theta)
        vertices_bottom[:, 1] = y + lower_radius * np.sin(theta)
        vertices_bottom[:, 2] = z
        vertices_top[:, 0] = x + upper_radius * np.cos(theta)
        vertices_top[:, 1] = y + upper_radius * np.sin(theta)
        vertices_top[:, 2] = z + height

        vertices = np.vstack([vertices_bottom, vertices_top])

        return vertices

    def __repr__(self) -> str:
        lower_radius, upper_radius, height = self.dimensions
        return (
            f"BaseCone at {self.center} with radii {lower_radius} (bottom), "
            f"{upper_radius} (top) and height {height}"
        )
