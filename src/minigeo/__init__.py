"""minigeo: a minimal 3D geometry library for interactive visualization."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "unknown"

from .scene import Scene
from .style import Style

__all__ = ["Scene", "Style", "__version__"]
