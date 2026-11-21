"""
Meshes: shapes made of any faces, e.g. models read from OBJ or STL files.

A mesh is given by its corner points and its faces, each face a list of vertex indices.
Like every shape it has a pose, its anchors lie on its bounding box, and it moves, turns
and scales like the other shapes.
"""

from __future__ import annotations

import numpy as np

from .shape import Shape

#: Neighbouring faces that meet at more than this angle (in degrees) get an outline.
CREASE = 30.0


class Mesh(Shape):
    """
    A shape given by vertices and faces, e.g. a model read from a file.

    The mesh is *closed* (a solid) when every edge belongs to exactly two faces that run
    along it in opposite directions. Then the faces are turned outward if they all point
    inward. Only the feature edges are outlined: the border of an open mesh, and the edges
    where two faces meet at more than ``crease`` degrees. So a cube made of triangles is
    outlined like a box, and a smooth surface has no outline across it.

    The shape's center is the middle of the bounding box, and its axes are the world axes.

    Example:
        >>> tetrahedron = Mesh(
        ...     [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ...     [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        ... )
        >>> tetrahedron.is_closed
        True
    """

    def __init__(self, vertices, faces, *, crease: float | None = CREASE):
        """
        Args:
            vertices (array-like): The corner points in world coordinates, shape ``(n, 3)``.
            faces: For each face, the indices of its corners (counting from 0), in order
                around the face. Faces can have different numbers of corners.
            crease: Outline the edges where faces meet at more than this many degrees;
                ``None`` outlines every edge.
        """
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError(
                f"A mesh needs vertices with three coordinates each, got shape {vertices.shape}."
            )
        faces = [[int(index) for index in face] for face in faces]
        for face in faces:
            if len(face) < 3:
                raise ValueError(f"Every face needs at least three corners, got {face}.")
            if min(face) < 0 or max(face) >= len(vertices):
                raise ValueError(
                    f"Face {face} refers to a vertex that does not exist ({len(vertices)} vertices)."
                )

        middle = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        super().__init__(middle, vertices=vertices - middle)
        self.crease = crease
        self.is_closed = bool(faces) and encloses(faces)
        if self.is_closed and signed_volume(self._local_vertices, faces) < 0:
            faces = [face[::-1] for face in faces]  # all faces pointed inward
        self._faces = faces

    @classmethod
    def read(cls, path, *, crease: float | None = CREASE) -> Mesh:
        """Read a mesh from an ``.obj`` or ``.stl`` file (see :func:`minigeo.read_mesh`)."""
        from ..meshfiles import read_mesh

        return read_mesh(path, crease=crease)

    def _face_layout(self) -> list[list[int]]:
        return [list(face) for face in self._faces]

    @property
    def triangles(self) -> np.ndarray:
        """The faces cut into triangles, as an ``(m, 3)`` array of vertex indices."""
        return triangulate(self.face_indices)[1]

    @property
    def volume(self) -> float:
        """The enclosed volume of a closed mesh (0 for an open one)."""
        return signed_volume(self._local_vertices, self.face_indices) if self.is_closed else 0.0

    @property
    def edge_indices(self) -> np.ndarray:
        """
        The feature edges as an ``(m, 2)`` array of vertex index pairs: edges with only
        one face (or more than two), and edges where two faces meet at more than ``crease``
        degrees.
        """
        if self.crease is None:
            return super().edge_indices
        faces = self.face_indices
        starts, ends, owners = sides(faces)
        edges, inverse, counts = np.unique(
            np.sort(np.column_stack([starts, ends]), axis=1), axis=0, return_inverse=True, return_counts=True
        )
        inverse = inverse.ravel()

        # where each edge's sides are listed one after the other
        order = np.argsort(inverse, kind="stable")
        first = np.searchsorted(inverse[order], np.arange(len(edges)))
        shared = counts == 2
        one, other = order[first[shared]], order[first[shared] + 1]

        normals = area_vectors(self._local_vertices, faces)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-300)
        cosine = np.einsum("ij,ij->i", normals[owners[one]], normals[owners[other]])
        cosine[starts[one] == starts[other]] *= -1  # both faces run the same way: one is flipped

        feature = ~shared
        feature[shared] = cosine < np.cos(np.radians(self.crease))
        return edges[feature]

    def __repr__(self) -> str:
        kind = "closed" if self.is_closed else "open"
        return f"{kind.capitalize()} mesh at {self.center} with {len(self._local_vertices)} vertices and {len(self._faces)} faces"


def sides(faces: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every side of every face, as start vertices, end vertices and the face it belongs to."""
    starts = np.array([index for face in faces for index in face], dtype=int)
    ends = np.array([index for face in faces for index in face[1:] + face[:1]], dtype=int)
    owners = np.repeat(np.arange(len(faces)), [len(face) for face in faces])
    return starts, ends, owners


def encloses(faces: list[list[int]]) -> bool:
    """
    Whether the faces close up: every side ``a -> b`` occurs once, and so does ``b -> a``
    (in the neighbouring face).
    """
    starts, ends, _ = sides(faces)
    forward = np.unique(np.column_stack([starts, ends]), axis=0)
    backward = np.unique(np.column_stack([ends, starts]), axis=0)
    return len(forward) == len(starts) and np.array_equal(forward, backward)


def triangulate(faces: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Cut faces into triangles that fan out from each face's first corner.

    This is exact for convex faces, which most mesh faces are.

    Returns:
        For each triangle the face it came from, and the ``(m, 3)`` vertex indices.
    """
    owners, triangles = [], []
    for k, face in enumerate(faces):
        for i in range(1, len(face) - 1):
            owners.append(k)
            triangles.append((face[0], face[i], face[i + 1]))
    return np.array(owners, dtype=int), np.array(triangles, dtype=int).reshape(-1, 3)


def area_vectors(vertices: np.ndarray, faces: list[list[int]]) -> np.ndarray:
    """
    For each face, its normal times twice its area.

    The cross products of a triangle fan add up to this for any face, even a non-convex one.
    """
    owners, triangles = triangulate(faces)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    result = np.zeros((len(faces), 3))
    np.add.at(result, owners, np.cross(b - a, c - a))
    return result


def signed_volume(vertices: np.ndarray, faces: list[list[int]]) -> float:
    """The volume enclosed by closed faces: positive when they point outward."""
    _, triangles = triangulate(faces)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)
