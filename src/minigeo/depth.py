"""
Drawing order for backends that paint from back to front (matplotlib, SVG).

Sorting faces by the depth of their centers fails in common scenes: a cup standing on a
large tabletop has its center further back than the tabletop's center, so the tabletop
would be painted over it. Instead, overlapping primitives are ordered with plane tests
(the core of Newell's painter's algorithm): if every point of B lies on the viewer's side
of face A's plane, A is drawn first. The depth of the centers only breaks ties.
"""

from __future__ import annotations

import heapq

import numpy as np

from .camera import Camera
from .geometry import face_normal
from .render import Face, Label, Marker, Path, Primitive

TOLERANCE = 1e-9


def view_direction(elev: float, azim: float, vertical_axis: str = "z") -> np.ndarray:
    """The unit vector pointing from the scene toward the viewer, for angles in degrees."""
    return Camera(elev, azim, vertical_axis).toward_viewer


def is_back_face(face: Face, toward_viewer: np.ndarray) -> bool:
    """Whether the face turns away from the viewer (hidden if it belongs to a closed, opaque solid)."""
    return float(np.dot(face_normal(face.points), toward_viewer)) < 0


def painter_order(primitives: list[Primitive], toward_viewer: np.ndarray) -> list[int]:
    """
    Return the indices of ``primitives`` in the order to draw them, from back to front.

    Paths marked ``on_top`` come after everything else, and labels always come last, so
    text is never hidden.
    """
    toward_viewer = np.asarray(toward_viewer, dtype=float)
    toward_viewer = toward_viewer / np.linalg.norm(toward_viewer)
    labels = [i for i, p in enumerate(primitives) if isinstance(p, Label)]
    on_top = [i for i, p in enumerate(primitives) if isinstance(p, Path) and p.on_top]
    solid = [i for i in range(len(primitives)) if i not in set(labels) | set(on_top)]

    points = [_points(primitives[i]) for i in solid]
    depth = [-float(np.dot(p.mean(axis=0), toward_viewer)) for p in points]  # larger = further away
    screen = _on_screen(points, toward_viewer)
    convex = [_is_convex(primitives[i]) for i in solid]

    # draw_before[a] holds the items that must be drawn after a
    draw_before: list[set[int]] = [set() for _ in solid]
    waiting_for = [0] * len(solid)
    for a, b in _overlapping_box_pairs(screen):
        if convex[a] and convex[b] and not _convex_overlap(screen[a], screen[b]):
            continue
        order = _compare(primitives[solid[a]], points[a], primitives[solid[b]], points[b], toward_viewer)
        if order:
            first, second = (a, b) if order < 0 else (b, a)
            draw_before[first].add(second)
            waiting_for[second] += 1

    return [solid[k] for k in _topological_order(draw_before, waiting_for, depth)] + on_top + labels


# ---------------------------------------------------------------- helpers


def _points(primitive: Primitive) -> np.ndarray:
    if isinstance(primitive, Face | Path):
        return np.asarray(primitive.points, dtype=float)
    if isinstance(primitive, Marker):
        return np.asarray(primitive.position, dtype=float)[np.newaxis]
    raise TypeError(f"No points for {type(primitive).__name__}.")


def _on_screen(points: list[np.ndarray], toward_viewer: np.ndarray) -> list[np.ndarray]:
    """The points projected onto a plane facing the viewer, as 2D coordinates."""
    helper = np.array([0.0, 0.0, 1.0]) if abs(toward_viewer[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(helper, toward_viewer)
    right /= np.linalg.norm(right)
    up = np.cross(toward_viewer, right)
    return [np.column_stack([p @ right, p @ up]) for p in points]


def _overlapping_box_pairs(screen: list[np.ndarray]) -> list[tuple[int, int]]:
    """All pairs ``(a, b)`` with ``a < b`` whose 2D bounding boxes overlap (a quick first test)."""
    if not screen:
        return []
    low = np.array([flat.min(axis=0) for flat in screen])
    high = np.array([flat.max(axis=0) for flat in screen])
    overlap = np.all(
        (low[:, np.newaxis, :] < high[np.newaxis, :, :] - TOLERANCE)
        & (low[np.newaxis, :, :] < high[:, np.newaxis, :] - TOLERANCE),
        axis=2,
    )
    a, b = np.nonzero(np.triu(overlap, k=1))
    return list(zip(a.tolist(), b.tolist(), strict=True))


def _is_convex(primitive: Primitive) -> bool:
    """Faces, single segments and points are convex; longer paths are not."""
    return isinstance(primitive, Face | Marker) or (
        isinstance(primitive, Path) and len(primitive.points) == 2
    )


def _convex_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Whether two convex 2D shapes (polygons, segments or points) overlap by more than touching.

    Separating axis theorem: two convex shapes are apart exactly if, along the normal of
    one of their sides, their shadows do not overlap.
    """
    for shape in (a, b):
        sides = np.roll(shape, -1, axis=0) - shape
        for dx, dy in sides:
            axis = np.array([-dy, dx])
            length = np.linalg.norm(axis)
            if length < TOLERANCE:
                continue
            axis /= length
            a_shadow, b_shadow = a @ axis, b @ axis
            if a_shadow.max() <= b_shadow.min() + TOLERANCE or b_shadow.max() <= a_shadow.min() + TOLERANCE:
                return False
    return True


def _side_of_plane(face_points: np.ndarray, other_points: np.ndarray, toward_viewer: np.ndarray) -> int:
    """
    +1 if ``other_points`` all lie on the viewer's side of the face's plane, -1 if all lie
    behind it, 0 if they straddle it, lie in it, or the plane is seen edge-on.
    """
    normal = face_normal(face_points)
    facing = float(np.dot(normal, toward_viewer))
    if abs(facing) < TOLERANCE:
        return 0
    distances = (other_points - face_points[0]) @ normal * np.sign(facing)
    scale = max(1.0, float(np.abs(face_points).max()))
    if np.all(distances >= -TOLERANCE * scale) and np.any(distances > TOLERANCE * scale):
        return 1
    if np.all(distances <= TOLERANCE * scale) and np.any(distances < -TOLERANCE * scale):
        return -1
    return 0


def _compare(a: Primitive, a_points, b: Primitive, b_points, toward_viewer) -> int:
    """-1 if ``a`` must be drawn before ``b``, +1 if after, 0 if the plane tests cannot tell."""
    if isinstance(a, Face):
        side = _side_of_plane(a_points, b_points, toward_viewer)
        if side:
            return -side  # b in front of a's plane: a first
    if isinstance(b, Face):
        side = _side_of_plane(b_points, a_points, toward_viewer)
        if side:
            return side  # a in front of b's plane: b first
    return 0


def _topological_order(draw_before: list[set[int]], waiting_for: list[int], depth: list[float]) -> list[int]:
    """
    Order the items so that every constraint is kept, the furthest available item first.

    If the constraints contain a cycle, the furthest remaining item is drawn anyway.
    """
    waiting_for = list(waiting_for)
    ready = [(-depth[k], k) for k in range(len(depth)) if waiting_for[k] == 0]
    heapq.heapify(ready)
    done = [False] * len(depth)
    order: list[int] = []
    while len(order) < len(depth):
        if not ready:  # a cycle: break it at the furthest remaining item
            k = min((k for k in range(len(depth)) if not done[k]), key=lambda k: -depth[k])
            waiting_for[k] = 0
            heapq.heappush(ready, (-depth[k], k))
        _, k = heapq.heappop(ready)
        if done[k]:
            continue
        done[k] = True
        order.append(k)
        for later in draw_before[k]:
            waiting_for[later] -= 1
            if waiting_for[later] == 0 and not done[later]:
                heapq.heappush(ready, (-depth[later], later))
    return order
