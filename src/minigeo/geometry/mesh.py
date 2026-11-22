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
        return triangulate(self.face_indices, self._local_vertices)[1]

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


def fan(faces: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Cut faces into triangles that fan out from each face's first corner.

    The triangles cover a face exactly only if it is convex, but their (signed) areas and
    normals always add up to the face's, so a fan is enough for :func:`area_vectors` and
    :func:`signed_volume`. To draw or hit faces, use :func:`triangulate`.

    Returns:
        For each triangle the face it came from, and the ``(m, 3)`` vertex indices.
    """
    owners, triangles = [], []
    for k, face in enumerate(faces):
        for i in range(1, len(face) - 1):
            owners.append(k)
            triangles.append((face[0], face[i], face[i + 1]))
    return np.array(owners, dtype=int), np.array(triangles, dtype=int).reshape(-1, 3)


def triangulate(faces: list[list[int]], vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Cut flat faces into triangles that cover them exactly, also concave ones.

    Convex faces (most of them) fan out from their first corner; concave faces are cut by
    ear clipping (:func:`polygon_triangles`).

    Args:
        faces: For each face, the indices of its corners in order around it.
        vertices: The ``(n, 3)`` points the indices refer to.

    Returns:
        For each triangle the face it came from, and the ``(m, 3)`` vertex indices.
    """
    vertices = np.asarray(vertices, dtype=float)
    owners, triangles = [np.empty(0, dtype=int)], [np.empty((0, 3), dtype=int)]
    by_size: dict[int, list[int]] = {}
    for k, face in enumerate(faces):
        by_size.setdefault(len(face), []).append(k)
    for size, numbers in by_size.items():
        numbers = np.array(numbers)
        corners = np.array([faces[k] for k in numbers], dtype=int)  # (m, size)
        convex = is_convex(vertices[corners])
        local = np.array([[0, i, i + 1] for i in range(1, size - 1)])  # the fan
        owners.append(np.repeat(numbers[convex], size - 2))
        triangles.append(corners[convex][:, local].reshape(-1, 3))
        for k, face in zip(numbers[~convex], corners[~convex], strict=True):
            pieces = face[polygon_triangles(vertices[face])]
            owners.append(np.full(len(pieces), k))
            triangles.append(pieces)
    return np.concatenate(owners), np.concatenate(triangles)


def is_convex(polygons: np.ndarray) -> np.ndarray:
    """
    For ``(m, k, 3)`` corner points of flat polygons: whether each one is convex, i.e. turns
    the same way at every corner (straight corners count as convex).
    """
    following = np.roll(polygons, -1, axis=1)
    normal = np.cross(polygons, following).sum(axis=1)  # Newell's method, (m, 3)
    edges = following - polygons
    turns = np.einsum("mkj,mj->mk", np.cross(edges, np.roll(edges, -1, axis=1)), normal)
    scale = np.linalg.norm(edges, axis=2) * np.linalg.norm(np.roll(edges, -1, axis=1), axis=2)
    return np.all(turns >= -1e-9 * scale * np.linalg.norm(normal, axis=1)[:, None], axis=1)


def polygon_triangles(points: np.ndarray) -> np.ndarray:
    """
    Triangles that cover a flat polygon, concave or not, as ``(k - 2, 3)`` corner indices.

    Ear clipping: a corner whose triangle with its two neighbours turns the right way and
    contains no other corner is an *ear*; it is cut off, and so on until a triangle is left.

    Example:
        >>> polygon_triangles(np.array([[0, 0, 0], [2, 0, 0], [1, 0.5, 0], [2, 1, 0], [0, 1, 0]]))
        array([[0, 1, 2],
               [4, 0, 2],
               [2, 3, 4]])
    """
    points = np.asarray(points, dtype=float)
    count = len(points)
    fan_triangles = np.array([[0, i, i + 1] for i in range(1, count - 1)], dtype=int).reshape(-1, 3)
    normal = np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)
    if count <= 3 or np.linalg.norm(normal) < 1e-300 or is_convex(points[np.newaxis])[0]:
        return fan_triangles

    # flat coordinates in the polygon's plane, turning counter-clockwise
    normal /= np.linalg.norm(normal)
    first = points[1] - points[0]
    first -= (first @ normal) * normal
    first = first / np.linalg.norm(first) if np.linalg.norm(first) > 0 else _perpendicular(normal)
    flat = points @ np.column_stack([first, np.cross(normal, first)])
    tolerance = 1e-12 * float(np.ptp(flat, axis=0).max()) ** 2

    def turn(a, b, c):  # twice the signed area of the triangle a, b, c
        return (flat[b, 0] - flat[a, 0]) * (flat[c, 1] - flat[a, 1]) - (flat[b, 1] - flat[a, 1]) * (
            flat[c, 0] - flat[a, 0]
        )

    remaining = list(range(count))
    triangles = []
    while len(remaining) > 3:
        for position in range(len(remaining)):
            a, b, c = remaining[position - 1], remaining[position], remaining[(position + 1) % len(remaining)]
            if turn(a, b, c) <= tolerance:
                continue  # a corner that turns the wrong way (or not at all)
            others = [i for i in remaining if i not in (a, b, c)]
            inside = [
                turn(a, b, i) >= -tolerance and turn(b, c, i) >= -tolerance and turn(c, a, i) >= -tolerance
                for i in others
            ]
            if any(inside):
                continue
            triangles.append([a, b, c])
            remaining.pop(position)
            break
        else:  # no ear: the outline crosses itself; fan out the rest
            triangles += [
                [remaining[0], remaining[i], remaining[i + 1]] for i in range(1, len(remaining) - 1)
            ]
            remaining = []
            break
    if remaining:
        triangles.append(remaining)
    return np.array(triangles, dtype=int)


def _perpendicular(vector: np.ndarray) -> np.ndarray:
    """Some unit vector perpendicular to ``vector``."""
    helper = np.eye(3)[np.argmin(np.abs(vector))]
    across = np.cross(vector, helper)
    return across / np.linalg.norm(across)


def area_vectors(vertices: np.ndarray, faces: list[list[int]]) -> np.ndarray:
    """
    For each face, its normal times twice its area.

    The cross products of a triangle fan add up to this for any face, even a non-convex one.
    """
    owners, triangles = fan(faces)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    result = np.zeros((len(faces), 3))
    np.add.at(result, owners, np.cross(b - a, c - a))
    return result


def signed_volume(vertices: np.ndarray, faces: list[list[int]]) -> float:
    """The volume enclosed by closed faces: positive when they point outward."""
    _, triangles = fan(faces)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)
