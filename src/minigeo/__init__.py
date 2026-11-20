"""
minigeo: describe simple 3D geometry and draw it as SVG, with matplotlib or with plotly.

Example:
    >>> from minigeo import Box, Cylinder, Scene
    >>> scene = Scene()
    >>> table = scene.add(Box([0, 0, 0], size=(2, 1, 0.1)), fill="peru")
    >>> scene.add(Cylinder(table.anchor("top"), radius=0.1, height=0.2))
"""

from importlib.metadata import PackageNotFoundError, version

from .geometry import Axis, Box, Circle, Cone, Cylinder, Line, Point, Polygon, Rectangle, Sphere, Transform
from .groups import Group
from .scene import Scene
from .style import Style

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "unknown"

__all__ = [
    "Axis",
    "Box",
    "Circle",
    "Cone",
    "Cylinder",
    "Group",
    "Line",
    "Point",
    "Polygon",
    "Rectangle",
    "Scene",
    "Sphere",
    "Style",
    "Transform",
    "__version__",
]
