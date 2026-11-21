"""
Reading and writing meshes as OBJ and STL files, with numpy only.

* **OBJ** (Wavefront) is text: vertices ``v x y z``, faces ``f i j k ...`` with any number
  of corners, and named objects ``o name``. Colors go into a material file (``.mtl``)
  next to it. Most 3D programs read and write OBJ.
* **STL** has triangles only, without shared corners, names or colors. It is what 3D
  printers and CAD programs exchange. Binary STL is written; binary and text STL are read.

Example:
    >>> scene.save("table.obj")          # every shape, with colors in table.mtl
    >>> bunny = read_mesh("bunny.stl")   # a Mesh, ready to place in a scene
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path as FilePath

import numpy as np

from .colors import to_rgb
from .geometry import Geometry, Line, Polyline, Shape
from .geometry.mesh import CREASE, Mesh, triangulate

# one STL triangle: normal, three corners, and two bytes nobody uses
_STL_TRIANGLE = np.dtype([("normal", "<f4", (3,)), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])


# ---------------------------------------------------------------- reading


def read_mesh(path, *, crease: float | None = CREASE) -> Mesh:
    """
    Read an ``.obj`` or ``.stl`` file as one mesh (all objects of an OBJ file together).

    Args:
        path: The file to read.
        crease: Outline the edges where faces meet at more than this many degrees
            (see :class:`~minigeo.Mesh`).

    Example:
        >>> part = scene.add(read_mesh("bracket.stl"), fill="silver")
    """
    vertices, objects = _read(FilePath(path))
    faces = [face for object_faces in objects.values() for face in object_faces]
    return _mesh(vertices, faces, crease)


def read_meshes(path, *, crease: float | None = CREASE) -> dict[str, Mesh]:
    """
    Read an ``.obj`` file as one mesh per object, by name (an STL file gives one mesh).

    Example:
        >>> parts = read_meshes("robot.obj")
        >>> parts["arm"].rotate("z", 30, about="bottom")
    """
    vertices, objects = _read(FilePath(path))
    return {name: _mesh(vertices, faces, crease) for name, faces in objects.items() if faces}


def _read(path: FilePath) -> tuple[np.ndarray, dict[str, list[list[int]]]]:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return _read_obj(path)
    if suffix == ".stl":
        vertices, faces = _read_stl(path)
        return vertices, {path.stem: faces}
    raise ValueError(f"Cannot read {path.name}: use an .obj or .stl file.")


def _mesh(vertices: np.ndarray, faces: list[list[int]], crease: float | None) -> Mesh:
    """A mesh of only the vertices these faces use."""
    used = np.unique(np.concatenate([np.asarray(face) for face in faces]))
    renumber = np.full(len(vertices), -1)
    renumber[used] = np.arange(len(used))
    return Mesh(vertices[used], [renumber[face].tolist() for face in faces], crease=crease)


def _read_obj(path: FilePath) -> tuple[np.ndarray, dict[str, list[list[int]]]]:
    """Vertices and faces by object (``o``, or groups ``g`` in files without objects)."""
    vertices: list[list[float]] = []
    faces: list[tuple[str | None, str | None, list[int]]] = []  # object, group, corners
    current = {"o": None, "g": None}
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        words = line.split("#", 1)[0].split()
        if not words:
            continue
        keyword, values = words[0], words[1:]
        if keyword == "v":
            vertices.append([float(value) for value in values[:3]])
        elif keyword == "f":
            # corners are "v", "v/vt", "v//vn" or "v/vt/vn"; negative numbers count from the end
            corners = [int(value.split("/")[0]) for value in values]
            corners = [c - 1 if c > 0 else len(vertices) + c for c in corners]
            if len(corners) < 3 or min(corners) < 0 or max(corners) >= len(vertices):
                raise ValueError(f"{path.name}, line {number}: face {line.strip()!r} is not valid.")
            faces.append((current["o"], current["g"], corners))
        elif keyword in current:
            current[keyword] = " ".join(values) or None
    if not faces:
        raise ValueError(f"{path.name} has no faces.")

    by = 0 if any(object_name for object_name, _, _ in faces) else 1  # objects, else groups
    objects: dict[str, list[list[int]]] = {}
    for face in faces:
        objects.setdefault(face[by] or path.stem, []).append(face[2])
    return np.array(vertices, dtype=float), objects


def _read_stl(path: FilePath) -> tuple[np.ndarray, list[list[int]]]:
    data = path.read_bytes()
    count = int.from_bytes(data[80:84], "little") if len(data) >= 84 else -1
    if len(data) == 84 + count * _STL_TRIANGLE.itemsize:
        corners = np.frombuffer(data, dtype=_STL_TRIANGLE, count=count, offset=84)["corners"]
    elif data.lstrip().startswith(b"solid"):
        numbers = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)
        corners = np.array(numbers, dtype=float).reshape(-1, 3, 3)
    else:
        raise ValueError(f"{path.name} is neither a binary nor a text STL file.")
    if len(corners) == 0:
        raise ValueError(f"{path.name} has no triangles.")
    return merge_corners(corners.astype(float))


def merge_corners(corners: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """
    Vertices and faces from separate triangles, with equal corners merged into one vertex.

    Triangles that collapse to a line or a point are left out.

    Args:
        corners: An ``(m, 3, 3)`` array: three corner points per triangle.
    """
    points = corners.reshape(-1, 3)
    size = float(np.ptp(points, axis=0).max()) or 1.0
    keys = np.round(points / (size * 1e-9))  # equal up to rounding
    _, first, index = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    triangles = index.ravel().reshape(-1, 3)
    distinct = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 2] != triangles[:, 0])
    )
    return points[first], triangles[distinct].tolist()


# ---------------------------------------------------------------- writing


@dataclass
class _Part:
    """A shape or line to write, with its name and (for scenes) its color."""

    name: str
    geometry: Geometry
    color: tuple[float, float, float] | None = None
    opacity: float = 1.0


def write_mesh(path, what) -> None:
    """
    Write shapes as an ``.obj`` or ``.stl`` file.

    Every shape is written as its faces, so spheres and cylinders become the facets they
    are drawn with. An OBJ file gets one object per shape, named after its label in a scene,
    and lines and polylines as OBJ lines. For a scene, the colors and opacities go into a
    material file with the same name and the extension ``.mtl``. STL keeps only triangles.

    Args:
        path: The file to write; its extension chooses the format.
        what: A :class:`~minigeo.Scene`, a geometry, a group, or a list of these.

    Example:
        >>> write_mesh("parts.stl", [box, cylinder])
        >>> scene.save("parts.obj")   # the same as write_mesh("parts.obj", scene)
    """
    path = FilePath(path)
    parts = _parts(what)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        _write_obj(path, parts)
    elif suffix == ".stl":
        _write_stl(path, [part for part in parts if isinstance(part.geometry, Shape)])
    else:
        raise ValueError(f"Cannot write {path.name}: use .obj or .stl.")


def _parts(what) -> list[_Part]:
    from .measure import _geometry_names, _members
    from .scene import Scene

    if isinstance(what, Scene):
        names = _geometry_names(what)
        parts = []
        for item in what.items:
            style = what.resolved_style(item)
            color = to_rgb(style.fill) if style.fill not in (None, "none") else None
            opacity = 1.0 if style.opacity is None else float(style.opacity)
            parts += [_Part(names[id(g)], g, color, opacity) for g in _members(item.geometry)]
    else:
        items = what if isinstance(what, list | tuple) else [what]
        members = [member for item in items for member in _members(item)]
        counts: dict[str, int] = {}
        parts = []
        for geometry in members:
            kind = type(geometry).__name__
            counts[kind] = counts.get(kind, 0) + 1
            parts.append(_Part(f"{kind} {counts[kind]}", geometry))
    parts = [part for part in parts if isinstance(part.geometry, Shape | Line | Polyline)]
    if not any(isinstance(part.geometry, Shape) for part in parts):
        raise ValueError("Nothing to write: there are no shapes.")
    return parts


def _write_obj(path: FilePath, parts: list[_Part]) -> None:
    lines = ["# written by minigeo"]
    materials: dict[tuple, str] = {}
    if any(part.color is not None and isinstance(part.geometry, Shape) for part in parts):
        lines.append(f"mtllib {path.with_suffix('.mtl').name}")
    offset = 1  # OBJ counts vertices from 1, through the whole file
    for part in parts:
        geometry = part.geometry
        lines.append(f"o {_obj_name(part.name)}")
        if isinstance(geometry, Shape):
            points, faces = geometry.vertices, geometry.face_indices
        elif isinstance(geometry, Polyline):
            points, faces = geometry.path, None
        else:  # a line
            points, faces = np.array([geometry.start, geometry.end], dtype=float), None
        lines += [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in points]
        if faces is None:
            lines.append("l " + " ".join(str(offset + k) for k in range(len(points))))
        else:
            if part.color is not None:
                key = (part.color, part.opacity)
                materials.setdefault(key, f"material_{len(materials) + 1}")
                lines.append(f"usemtl {materials[key]}")
            lines += ["f " + " ".join(str(offset + k) for k in face) for face in faces]
        offset += len(points)
    path.write_text("\n".join(lines) + "\n")

    if materials:
        mtl = ["# written by minigeo"]
        for ((r, g, b), opacity), name in materials.items():
            mtl += [f"newmtl {name}", f"Kd {r:.4g} {g:.4g} {b:.4g}", f"d {opacity:.4g}", ""]
        path.with_suffix(".mtl").write_text("\n".join(mtl))


def _obj_name(name: str) -> str:
    """OBJ names end at a space or #, so those become underscores."""
    return re.sub(r"[\s#]+", "_", name.strip()) or "object"


def _write_stl(path: FilePath, parts: list[_Part]) -> None:
    corners = []
    for part in parts:
        shape = part.geometry
        _, triangles = triangulate(shape.face_indices)
        corners.append(shape.vertices[triangles])
    corners = np.concatenate(corners)

    records = np.zeros(len(corners), dtype=_STL_TRIANGLE)
    records["corners"] = corners
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    records["normal"] = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    header = b"binary STL written by minigeo".ljust(80, b" ")
    path.write_bytes(header + len(records).to_bytes(4, "little") + records.tobytes())
