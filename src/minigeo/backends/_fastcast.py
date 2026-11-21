"""
Fast ray casting with numba, for the ray tracer (optional: ``pip install "minigeo[fast]"``).

The numpy ray tracer casts many parallel rays at once and tests every surface against the
rays inside its outline; its cost grows with the number of triangles. Here each ray walks a
*bounding volume hierarchy* instead: a tree of boxes around groups of surfaces, so it only
tests the few surfaces near its path. The walk runs as compiled code, for many rays in
parallel on all processor cores.

The results are the same as those of the numpy path (up to rounding), for triangles and for
exact spheres and cylinders (see :mod:`minigeo.exact`). Rays need not be parallel, so this is
also the way to perspective cameras and random sampling later.

The compiled code is cached on disk, so only the first use on a machine takes a few seconds.
"""

from __future__ import annotations

import numpy as np

try:
    import numba
except ImportError:  # the extra is not installed
    numba = None

#: Surfaces per leaf of the tree.
LEAF_SIZE = 4

#: Rays handled one after another by each parallel task.
CHUNK = 256

TRIANGLE, SPHERE, CYLINDER = 0, 1, 2


def available() -> bool:
    """Whether numba is installed."""
    return numba is not None


class Accelerator:
    """
    A bounding volume hierarchy over triangles and exact shapes, and the kernel that walks it.

    Args:
        triangles: ``(m, 3, 3)`` corners of the triangles.
        kinds: For each exact shape, ``"sphere"`` or ``"cylinder"``.
        from_world: For each exact shape, its ``(4, 4)`` map from world to unit shape.
        boxes: For each exact shape, the ``(8, 3)`` corners of a box around it.
    """

    def __init__(
        self, triangles: np.ndarray, kinds: list[str], from_world: list[np.ndarray], boxes: list[np.ndarray]
    ):
        if numba is None:
            raise ImportError('Fast ray casting needs numba: pip install "minigeo[fast]"')
        triangles = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
        self.v0 = np.ascontiguousarray(triangles[:, 0])
        self.e1 = np.ascontiguousarray(triangles[:, 1] - triangles[:, 0])
        self.e2 = np.ascontiguousarray(triangles[:, 2] - triangles[:, 0])
        self.area = np.linalg.norm(np.cross(self.e1, self.e2), axis=1)  # twice the area
        self.inverse = np.ascontiguousarray(np.array(from_world, dtype=np.float64).reshape(-1, 4, 4))

        m, q = len(triangles), len(kinds)
        self.kind = np.array(
            [TRIANGLE] * m + [SPHERE if k == "sphere" else CYLINDER for k in kinds], dtype=np.int64
        )
        self.ref = np.concatenate([np.arange(m), np.arange(q)]).astype(np.int64)
        low = [triangles.min(axis=1)] if m else []
        high = [triangles.max(axis=1)] if m else []
        if q:
            corners = np.array(boxes, dtype=np.float64)
            low.append(corners.min(axis=1))
            high.append(corners.max(axis=1))
        low = np.concatenate(low) if low else np.zeros((0, 3))
        high = np.concatenate(high) if high else np.zeros((0, 3))
        self.tree = _build_tree(low, high)

    def cast(
        self, origins: np.ndarray, directions: np.ndarray, start: np.ndarray, stop: np.ndarray | None = None
    ):
        """
        The nearest hit of each ray ``origin + t * direction`` with ``start < t < stop``.

        ``directions`` is ``(n, 3)``, or ``(3,)`` for parallel rays. ``stop`` (default: no
        limit) lets rays that only look nearby, e.g. for ambient occlusion, skip the rest.

        Returns:
            (distance, primitive, part): ``distance`` is inf where nothing was hit;
            ``primitive`` indexes triangles first, then exact shapes (-1: nothing);
            ``part`` is 0 for the side of a cylinder or a sphere, 1 for the bottom, 2 for the top.
        """
        origins = np.ascontiguousarray(origins, dtype=np.float64).reshape(-1, 3)
        directions = np.ascontiguousarray(np.atleast_2d(directions), dtype=np.float64)
        start = np.ascontiguousarray(start, dtype=np.float64)
        count = len(origins)
        stop = np.full(count, np.inf) if stop is None else np.ascontiguousarray(stop, dtype=np.float64)
        distance = np.full(count, np.inf)
        primitive = np.full(count, -1, dtype=np.int64)
        part = np.zeros(count, dtype=np.int64)
        if count and len(self.kind):
            low, high, left, right, first, size, axes, order = self.tree
            _walk(origins, directions, start, stop, low, high, left, right, first, size, axes, order,
                  self.kind, self.ref, self.v0, self.e1, self.e2, self.area, self.inverse,
                  distance, primitive, part)  # fmt: skip
        return distance, primitive, part


def _build_tree(low: np.ndarray, high: np.ndarray):
    """
    Build the tree. Each node is split where the *surface area heuristic* expects the fewest
    tests: a ray hits a box with a chance proportional to its surface, so a split is cheap if
    small boxes hold many surfaces. Candidate splits are the borders of 16 bins of the
    surfaces' centers along each axis.

    Returns arrays per node: box ``low``/``high``, children ``left``/``right`` (-1 for leaves),
    the ``axis`` it is split along, and the leaves' range ``first``/``size`` into ``order``,
    the reordered primitive indices.
    """
    count = len(low)
    centers = (low + high) / 2
    order = np.arange(count, dtype=np.int64)
    nodes_low, nodes_high, left, right, first, size, axes = [], [], [], [], [], [], []

    def node(begin: int, end: int) -> int:
        index = len(nodes_low)
        members = order[begin:end]
        box_low, box_high = low[members].min(axis=0), high[members].max(axis=0)
        nodes_low.append(box_low)
        nodes_high.append(box_high)
        left.append(-1)
        right.append(-1)
        first.append(begin)
        size.append(end - begin)
        axes.append(0)
        if end - begin > LEAF_SIZE:
            split = _best_split(low[members], high[members], centers[members], _area(box_low, box_high))
            if split is not None:
                axis, below = split
                order[begin:end] = np.concatenate([members[below], members[~below]])
                middle = begin + int(below.sum())
                size[index] = 0  # an inner node
                axes[index] = axis
                left[index] = node(begin, middle)
                right[index] = node(middle, end)
        return index

    if count:
        node(0, count)
    as_array = lambda values, kind: np.ascontiguousarray(np.array(values, dtype=kind))  # noqa: E731
    return (
        as_array(nodes_low, np.float64).reshape(-1, 3),
        as_array(nodes_high, np.float64).reshape(-1, 3),
        as_array(left, np.int64),
        as_array(right, np.int64),
        as_array(first, np.int64),
        as_array(size, np.int64),
        as_array(axes, np.int64),
        order,
    )


BINS = 16


def _area(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Half the surface area of boxes (enough for comparing them)."""
    extent = np.maximum(high - low, 0)
    return extent[..., 0] * extent[..., 1] + extent[..., 1] * extent[..., 2] + extent[..., 2] * extent[..., 0]


def _best_split(low: np.ndarray, high: np.ndarray, centers: np.ndarray, parent_area: float):
    """
    The cheapest split of a node's surfaces as ``(axis, below)``, where ``below`` marks the
    surfaces going to the first child; None if keeping them in one leaf is cheaper.
    """
    count = len(low)
    best_cost, best = float(count), None  # cost of a leaf: test every surface
    for axis in range(3):
        lo, hi = centers[:, axis].min(), centers[:, axis].max()
        if hi - lo < 1e-12:
            continue
        bins = np.minimum(((centers[:, axis] - lo) / (hi - lo) * BINS).astype(int), BINS - 1)
        counts = np.bincount(bins, minlength=BINS)
        bin_low = np.full((BINS, 3), np.inf)
        bin_high = np.full((BINS, 3), -np.inf)
        np.minimum.at(bin_low, bins, low)
        np.maximum.at(bin_high, bins, high)
        # boxes and counts of everything below and above each of the BINS - 1 borders
        below_low = np.minimum.accumulate(bin_low, axis=0)[:-1]
        below_high = np.maximum.accumulate(bin_high, axis=0)[:-1]
        above_low = np.minimum.accumulate(bin_low[::-1], axis=0)[::-1][1:]
        above_high = np.maximum.accumulate(bin_high[::-1], axis=0)[::-1][1:]
        below_count = np.cumsum(counts)[:-1]
        above_count = count - below_count
        cost = 0.5 + (
            _area(below_low, below_high) * below_count + _area(above_low, above_high) * above_count
        ) / max(parent_area, 1e-300)
        cost[(below_count == 0) | (above_count == 0)] = np.inf
        border = int(np.argmin(cost))
        if cost[border] < best_cost:
            best_cost, best = float(cost[border]), (axis, bins <= border)
    if best is None and count > 4 * LEAF_SIZE:  # e.g. all centers in one point: split in halves anyway
        half = np.zeros(count, dtype=bool)
        half[np.argsort(centers[:, 0], kind="stable")[: count // 2]] = True
        return 0, half
    return best


if numba is not None:

    @numba.njit(cache=True, inline="always")
    def _hit_box(ox, oy, oz, ix, iy, iz, low, high, near, far):
        """Whether the ray enters the box between ``near`` and ``far`` (slab test)."""
        t0 = (low[0] - ox) * ix
        t1 = (high[0] - ox) * ix
        if t0 > t1:
            t0, t1 = t1, t0
        near = max(near, t0)
        far = min(far, t1)
        t0 = (low[1] - oy) * iy
        t1 = (high[1] - oy) * iy
        if t0 > t1:
            t0, t1 = t1, t0
        near = max(near, t0)
        far = min(far, t1)
        t0 = (low[2] - oz) * iz
        t1 = (high[2] - oz) * iz
        if t0 > t1:
            t0, t1 = t1, t0
        near = max(near, t0)
        far = min(far, t1)
        return near <= far

    @numba.njit(cache=True, inline="always")
    def _hit_triangle(ox, oy, oz, dx, dy, dz, v0, e1, e2, area, start, best):
        """Distance to a triangle (Möller-Trumbore), or inf; edges count as inside."""
        px = dy * e2[2] - dz * e2[1]
        py = dz * e2[0] - dx * e2[2]
        pz = dx * e2[1] - dy * e2[0]
        det = e1[0] * px + e1[1] * py + e1[2] * pz
        length = np.sqrt(dx * dx + dy * dy + dz * dz)
        if abs(det) <= 1e-12 * area * length:
            return np.inf  # seen edge-on
        inverse = 1.0 / det
        sx, sy, sz = ox - v0[0], oy - v0[1], oz - v0[2]
        u = (sx * px + sy * py + sz * pz) * inverse
        if u < -1e-9 or u > 1 + 1e-9:
            return np.inf
        qx = sy * e1[2] - sz * e1[1]
        qy = sz * e1[0] - sx * e1[2]
        qz = sx * e1[1] - sy * e1[0]
        v = (dx * qx + dy * qy + dz * qz) * inverse
        if v < -1e-9 or u + v > 1 + 1e-9:
            return np.inf
        t = (e2[0] * qx + e2[1] * qy + e2[2] * qz) * inverse
        return t if start < t < best else np.inf

    @numba.njit(cache=True, inline="always")
    def _hit_exact(kind, matrix, ox, oy, oz, dx, dy, dz, start, best):
        """Distance and part of the nearest hit with the unit sphere or cylinder, or (inf, 0)."""
        ux = matrix[0, 0] * ox + matrix[0, 1] * oy + matrix[0, 2] * oz + matrix[0, 3]
        uy = matrix[1, 0] * ox + matrix[1, 1] * oy + matrix[1, 2] * oz + matrix[1, 3]
        uz = matrix[2, 0] * ox + matrix[2, 1] * oy + matrix[2, 2] * oz + matrix[2, 3]
        vx = matrix[0, 0] * dx + matrix[0, 1] * dy + matrix[0, 2] * dz
        vy = matrix[1, 0] * dx + matrix[1, 1] * dy + matrix[1, 2] * dz
        vz = matrix[2, 0] * dx + matrix[2, 1] * dy + matrix[2, 2] * dz
        found, part = np.inf, 0
        if kind == SPHERE:
            a = vx * vx + vy * vy + vz * vz
            b = 2 * (ux * vx + uy * vy + uz * vz)
            c = ux * ux + uy * uy + uz * uz - 1
            discriminant = b * b - 4 * a * c
            if discriminant >= 0:
                root = np.sqrt(discriminant)
                for t in ((-b - root) / (2 * a), (-b + root) / (2 * a)):
                    if start < t < best and t < found:
                        found = t
            return found, 0
        a = vx * vx + vy * vy
        if a > 1e-15:  # the side
            b = 2 * (ux * vx + uy * vy)
            c = ux * ux + uy * uy - 1
            discriminant = b * b - 4 * a * c
            if discriminant >= 0:
                root = np.sqrt(discriminant)
                for t in ((-b - root) / (2 * a), (-b + root) / (2 * a)):
                    if start < t < best and t < found and abs(uz + t * vz) <= 0.5:
                        found, part = t, 0
        if abs(vz) > 1e-15:  # the bottom and the top
            for z, which in ((-0.5, 1), (0.5, 2)):
                t = (z - uz) / vz
                x, y = ux + t * vx, uy + t * vy
                if start < t < best and t < found and x * x + y * y <= 1:
                    found, part = t, which
        return found, part

    @numba.njit(parallel=True, cache=True)
    def _walk(origins, directions, start, stop, low, high, left, right, first, size, axes, order,
              kind, ref, v0, e1, e2, area, inverse, distance, primitive, part):  # fmt: skip
        count = origins.shape[0]
        shared = directions.shape[0] == 1
        chunks = (count + CHUNK - 1) // CHUNK
        for chunk in numba.prange(chunks):
            stack = np.empty(128, dtype=np.int64)
            for r in range(chunk * CHUNK, min(count, (chunk + 1) * CHUNK)):
                d = 0 if shared else r
                ox, oy, oz = origins[r, 0], origins[r, 1], origins[r, 2]
                dx, dy, dz = directions[d, 0], directions[d, 1], directions[d, 2]
                ix = 1.0 / dx if dx != 0 else np.inf
                iy = 1.0 / dy if dy != 0 else np.inf
                iz = 1.0 / dz if dz != 0 else np.inf
                tmin = start[r]
                best, hit, which = stop[r], -1, 0
                stack[0] = 0
                depth = 1
                while depth > 0:
                    depth -= 1
                    node = stack[depth]
                    if not _hit_box(ox, oy, oz, ix, iy, iz, low[node], high[node], tmin, best):
                        continue
                    if size[node] == 0:  # visit the nearer child first: push it last
                        if directions[d, axes[node]] > 0:
                            stack[depth], stack[depth + 1] = right[node], left[node]
                        else:
                            stack[depth], stack[depth + 1] = left[node], right[node]
                        depth += 2
                        continue
                    for k in range(first[node], first[node] + size[node]):
                        p = order[k]
                        index = ref[p]
                        if kind[p] == TRIANGLE:
                            t = _hit_triangle(ox, oy, oz, dx, dy, dz, v0[index], e1[index], e2[index],
                                              area[index], tmin, best)  # fmt: skip
                            if t < best:
                                best, hit, which = t, p, 0
                        else:
                            t, piece = _hit_exact(kind[p], inverse[index], ox, oy, oz, dx, dy, dz, tmin, best)
                            if t < best:
                                best, hit, which = t, p, piece
                distance[r] = best if hit >= 0 else np.inf
                primitive[r] = hit
                part[r] = which
