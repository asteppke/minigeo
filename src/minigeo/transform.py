"""Affine transforms in 3D: translation, rotation, scaling and shear as 4x4 matrices."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # only for type hints; avoids a circular import
    from .frames import Frame
    from .geometry import Geometry, Shape


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
    An affine transformation in 3D, kept as a 4x4 matrix in homogeneous coordinates.

    Transforms are built step by step (``Transform().rotate("z", 30).translate([1, 0, 0])``),
    combined with ``@``, and applied to geometry with ``t @ shape``.
    """

    def __init__(self, matrix: np.ndarray | None = None):
        """
        Args:
            matrix (array-like, optional): A 4x4 matrix to start from (copied), whose last
                row is ``[0, 0, 0, 1]``. Default: the identity, which changes nothing.

        Example:
            >>> Transform()                      # the identity
            >>> Transform(other.matrix @ m)      # from a matrix
        """
        if matrix is None:
            self.matrix = np.eye(4)
            return
        matrix = np.array(matrix, dtype=float)
        if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("A transform needs a 4x4 matrix whose last row is [0, 0, 0, 1].")
        self.matrix = matrix

    def copy(self) -> Transform:
        """An independent copy: changing one does not change the other."""
        return Transform(self.matrix)

    def inverse(self) -> Transform:
        """
        The transform that undoes this one, as a new transform.

        Raises:
            ValueError: If the transform flattens space (e.g. a scaling by 0) and cannot be undone.

        Example:
            >>> t = Transform().rotate("z", 30).translate([1, 0, 0])
            >>> np.allclose((t.inverse() @ t).matrix, np.eye(4))
            True
        """
        linear, offset = self.matrix[:3, :3], self.matrix[:3, 3]
        inverse = np.eye(4)
        if np.allclose(linear.T @ linear, np.eye(3)):  # a rotation (or mirroring): exact and cheap
            inverse[:3, :3] = linear.T
        elif abs(np.linalg.det(linear)) < 1e-15:
            raise ValueError("This transform flattens space and cannot be undone.")
        else:
            inverse[:3, :3] = np.linalg.inv(linear)
        inverse[:3, 3] = -inverse[:3, :3] @ offset
        return Transform(inverse)

    def __repr__(self) -> str:
        return f"Transform(\n{np.array2string(self.matrix, precision=4, suppress_small=True)})"

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

    def scale(self, factors: float | np.ndarray) -> Transform:
        """
        Applies scaling along each axis.

        Args:
            factors: One factor per axis, or one number for all three.

        Returns:
            Transform: Self to allow method chaining.

        Example:
            >>> Transform().scale(2)            # twice as large
            >>> Transform().scale([1, 1, -1])   # mirrored in the xy plane
        """
        S = np.eye(4)
        S[0, 0], S[1, 1], S[2, 2] = np.broadcast_to(np.asarray(factors, dtype=float), (3,))
        self.matrix = S @ self.matrix
        return self

    def shear(
        self,
        factors=None,
        *,
        xy: float = 0.0,
        yx: float = 0.0,
        xz: float = 0.0,
        zx: float = 0.0,
        yz: float = 0.0,
        zy: float = 0.0,
    ) -> Transform:
        """
        Applies shearing: each coordinate is shifted in proportion to another one.

        ``xy`` is how much x is shifted per unit of y, ``zy`` how much z is shifted per unit
        of y, and so on.

        Args:
            factors: Optionally all factors in a row: ``(xy, yx, xz, zx)`` or
                ``(xy, yx, xz, zx, yz, zy)``.
            xy, yx, xz, zx, yz, zy: Single factors by name (when ``factors`` is not given).

        Returns:
            Transform: Self to allow method chaining.

        Example:
            >>> Transform().shear(xz=0.5)      # x leans over by half the height
            >>> Transform().shear(yz=0.2, zy=0.1)
        """
        if factors is not None:
            factors = [float(f) for f in factors]
            if len(factors) not in (4, 6):
                raise ValueError(
                    f"shear expects 4 factors (xy, yx, xz, zx) or 6 (... yz, zy), got {len(factors)}."
                )
            xy, yx, xz, zx, yz, zy = [*factors, 0.0, 0.0][:6]
        S = np.eye(4)
        S[0, 1], S[1, 0] = xy, yx
        S[0, 2], S[2, 0] = xz, zx
        S[1, 2], S[2, 1] = yz, zy
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

    def in_frame(self, frame: Frame | Shape) -> Transform:
        """
        Read this transform in the coordinates of a frame or shape instead of the world's.

        Translations then go along the frame's axes x', y', z', and rotations turn about its
        axes through its origin. The frame's current position is used, so apply the result
        right away (the frame may move later).

        Args:
            frame (Frame or Shape): Anything with a rigid ``pose``.

        Returns:
            Transform: Self to allow method chaining.

        Example:
            >>> Transform().translate([0.1, 0, 0]).in_frame(x) @ carriage  # along x's own axis
            >>> Transform().rotate("z", 30).in_frame(box) @ box            # about the box's z'
        """
        pose = frame.pose.matrix
        self.matrix = pose @ self.matrix @ np.linalg.inv(pose)
        return self

    def apply(self, other: Geometry) -> Geometry:
        """
        Applies the transform to a geometry object (in place).

        Args:
            other (Geometry): The geometry (or group) to transform.

        Returns:
            The same object, so calls can be chained.
        """
        if not hasattr(other, "apply_transform"):
            raise TypeError(f"Cannot apply a Transform to {type(other).__name__}.")
        other.apply_transform(self)
        return other

    def __matmul__(self, other: Transform | Geometry) -> Transform | Geometry:
        """
        ``t1 @ t2`` composes two transforms into a new one (``t2`` is applied first).
        ``t @ shape`` applies the transform to the shape in place and returns it.
        """
        # TODO: Think about whether `t @ shape` should return a transformed copy instead.
        if isinstance(other, Transform):
            return Transform(self.matrix @ other.matrix)
        return self.apply(other)
