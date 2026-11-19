"""
Minimal 3D geometry library for visualizing 3D shapes and transformations.

The library provides a set of base classes for 3D geometry objects such as points, lines, shapes, and axes
and is independent of any specific visualization library.
"""

from __future__ import annotations
from typing import Optional, List
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
    if theta < np.finfo(float).eps:
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
        Applies shearing along each axis.

        Args:
            factors (np.ndarray): A 3-element array for shearing factors.

        Returns:
            Transform: Self to allow method chaining.
        """
        S = np.eye(4)
        S[0, 1], S[1, 0] = factors[0], factors[0]
        S[0, 2], S[2, 0] = factors[1], factors[1]
        S[1, 2], S[2, 1] = factors[2], factors[2]
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

    def rotate(self, axis: str | np.ndarray, angle: float, in_degrees: bool = True) -> Transform:
        """
        Rotates around a specified axis.

        Args:
            axis (str or np.ndarray): The axis to rotate around ('x', 'y', 'z') or a 1D numpy array.
            angle (float): The rotation angle.
            in_degrees (bool): If True, converts degrees to radians.

        Returns:
            Transform: Self to allow method chaining.
        """
        if isinstance(axis, np.ndarray):
            return self.rotate_arbitrary(axis, angle, in_degrees)
        elif axis.lower() == "x":
            return self.rotate_x(angle, in_degrees)
        elif axis.lower() == "y":
            return self.rotate_y(angle, in_degrees)
        elif axis.lower() == "z":
            return self.rotate_z(angle, in_degrees)
        else:
            raise ValueError("Invalid axis specified. Use 'x', 'y', 'z' or a numpy array.")

    def apply(self, other: BaseGeometry, inplace=True) -> BaseGeometry:
        """
        Applies the transform to a geometry object.

        Args:
            other (BaseGeometry): The geometry object to apply the transform to.

        """
        if not hasattr(other, "vertices"):
            raise ValueError("The object must have vertices.")

        new_vertices = self.apply_to_vertices(other.vertices)
        other.vertices = new_vertices
        
        if hasattr(other, "_center"):
            other._center = self.apply_to_vertex(other._center)

        other.update_geometry()
        return other


class BaseGeometry(ABC):
    """
    Base class for all geometry classes.
    Every geometry contains vertices (np.ndarray) as underlying properties.
    """

    @property
    @abstractmethod
    def vertices(self) -> np.ndarray:
        pass

    @vertices.setter
    @abstractmethod
    def vertices(self, value: np.ndarray):
        pass

    def rotate(self, axis: str | np.ndarray, angle: float, in_degrees: bool = True):
        """
        Rotate the geometry around the given axis.
        """
        transform = Transform().rotate(axis, angle, in_degrees)
        self.vertices = transform.apply_to_vertices(self.vertices)
        self.update_geometry()

    def update_geometry(self) -> None:
        """
        Hook for updating any geometry representation (e.g. redrawing).
        """
        pass
"""