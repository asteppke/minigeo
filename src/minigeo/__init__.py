"""minigeo: a minimal 3D geometry library for interactive visualization."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "unknown"
