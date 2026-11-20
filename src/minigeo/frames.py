"""
Frames and joints: parts that carry other parts, like the stages of an actuator.

A :class:`Frame` is a coordinate system that can move relative to the frame carrying it
(its *parent*). The frames form a tree: moving a frame moves all frames below it, and every
geometry attached to them. A *joint* is a frame whose position relative to its parent is set
by a single number, its ``value``:

* a **prismatic** joint slides along an axis (``value`` is a distance),
* a **revolute** joint turns about an axis (``value`` is an angle in degrees),
* a **fixed** frame does not move relative to its parent.

Example:
    >>> base = Frame()
    >>> x = base.prismatic("x", limits=(0, 0.5))   # X carries Y, Y carries Z
    >>> y = x.prismatic("y")
    >>> z = y.prismatic("z")
    >>> tool = z.attach(Cone([0, 0, 0], radius=0.05, height=0.2))
    >>> x.value = 0.3                              # absolute: the tool is 0.3 along x
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from .anchors import resolve_pivot
from .transform import Transform, split_rotation

if TYPE_CHECKING:  # only for type hints; avoids a circular import
    from .geometry import Geometry
    from .groups import Group

AXIS_NAMES = {"x": 0, "y": 1, "z": 2}

#: How far (relative) a value may overshoot a limit through rounding; it is then set to the limit.
LIMIT_TOLERANCE = 1e-9


def _transform(matrix: np.ndarray) -> Transform:
    transform = Transform()
    transform.matrix = np.array(matrix, dtype=float)
    return transform


def _rigid_inverse(matrix: np.ndarray) -> np.ndarray:
    """The inverse of a rotation plus translation, without a general matrix inversion."""
    inverse = np.eye(4)
    rotation = matrix[:3, :3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ matrix[:3, 3]
    return inverse


class Frame:
    """
    A coordinate system x', y', z' that carries attached geometry and further frames.

    A frame without a parent stands in the world. Further frames are made from it with
    :meth:`fixed`, :meth:`prismatic` and :meth:`revolute`.

    Everything you pass in is in **world coordinates, as things stand now**: positions such as
    ``at=box.anchor("top")``, axes such as ``"x"``, and transforms such as
    ``Transform().translate(...) @ part``. Internally, positions are kept relative to the
    carrying frame, so when a joint moves later, everything it carries goes along. To give a
    transform in a frame's own coordinates instead, use :meth:`Transform.in_frame`.

    Attributes:
        parent (Frame or None): The frame carrying this one.
        children (list[Frame]): The frames carried by this one.
        kind (str): ``"fixed"``, ``"prismatic"`` or ``"revolute"``.
        limits (tuple or None): The smallest and largest allowed ``value``, or None.
        name (str): An optional name, used in error messages and ``repr``.
    """

    def __init__(self, position=(0, 0, 0), *, name: str = ""):
        """
        Make a frame standing in the world, with its axes along x, y, z.

        Args:
            position (array-like): Where the frame's origin is.
            name (str): An optional name.

        Example:
            >>> base = Frame()
        """
        self.parent: Frame | None = None
        self.children: list[Frame] = []
        self.kind = "fixed"
        self.limits: tuple[float, float] | None = None
        self.name = name
        self._mount = Transform().translate(position).matrix  # relative to the parent at value 0
        self._axis = np.zeros(3)  # the joint axis, in mount coordinates
        self._value = 0.0
        self._attached: list[Geometry | Group] = []
        self._synced = self.pose.matrix  # where the attached geometry currently is

    # ------------------------------------------------------------ making frames

    def fixed(self, *, at=None, name: str = "") -> Frame:
        """
        Make a frame carried by this one that does not move relative to it.

        Args:
            at (array-like, optional): Where its origin is (world coordinates); default this
                frame's origin.
            name (str): An optional name.
        """
        return self._child("fixed", None, at, None, 0.0, name)

    def prismatic(self, axis, *, at=None, limits=None, value: float = 0.0, name: str = "") -> Frame:
        """
        Make a sliding joint carried by this frame; its ``value`` is the distance along ``axis``.

        Args:
            axis (str or array-like): The direction to slide in: ``"x"``, ``"y"``, ``"z"`` or a
                vector (world coordinates), or ``"x'"``, ``"y'"``, ``"z'"`` for this frame's axes.
            at (array-like, optional): Where the joint's origin is at value 0 (world
                coordinates); default this frame's origin.
            limits (tuple, optional): The smallest and largest allowed value.
            value (float): The starting value.
            name (str): An optional name.

        Example:
            >>> x = base.prismatic("x", limits=(0, 0.5))
        """
        return self._child("prismatic", axis, at, limits, value, name)

    def revolute(self, axis, *, at=None, limits=None, value: float = 0.0, name: str = "") -> Frame:
        """
        Make a turning joint carried by this frame; its ``value`` is the angle in degrees.

        The joint turns about ``axis`` through its origin ``at``, using the right-hand rule.

        Args:
            axis (str or array-like): The axis to turn about, as for :meth:`prismatic`.
            at (array-like, optional): A point on the axis (world coordinates); default this
                frame's origin.
            limits (tuple, optional): The smallest and largest allowed angle in degrees.
            value (float): The starting angle in degrees.
            name (str): An optional name.

        Example:
            >>> theta = base.revolute("z", at=[0, 0, 0.1])
            >>> theta.value = 30
        """
        return self._child("revolute", axis, at, limits, value, name)

    def _child(self, kind, axis, at, limits, value, name) -> Frame:
        child = Frame(name=name)
        child.parent = self
        child.kind = kind
        world = self.pose.matrix
        origin = world[:3, 3] if at is None else np.asarray(at, dtype=float)
        # the child's axes start out along this frame's axes; its origin is at `at`
        child._mount = _rigid_inverse(world) @ Transform().translate(origin).matrix @ _rotation_only(world)
        if axis is not None:
            child._axis = world[:3, :3].T @ self._direction(axis)  # world -> mount coordinates
        if limits is not None:
            low, high = (float(limit) for limit in limits)
            if low > high:
                raise ValueError(f"The limits {limits} need the smaller value first.")
            child.limits = (low, high)
        self.children.append(child)
        child._synced = child.pose.matrix
        child.value = value
        return child

    def _direction(self, axis) -> np.ndarray:
        """A unit direction in world coordinates, from a name or a vector."""
        if isinstance(axis, str):
            if axis.endswith("'") and axis[:-1] in AXIS_NAMES:
                return self.axes[AXIS_NAMES[axis[:-1]]]
            if axis in AXIS_NAMES:
                return np.eye(3)[AXIS_NAMES[axis]]
            raise ValueError(f'Unknown axis {axis!r}; use "x", "y", "z", "x\'", "y\'", "z\'" or a vector.')
        direction = np.asarray(axis, dtype=float)
        length = np.linalg.norm(direction)
        if direction.shape != (3,) or length == 0:
            raise ValueError(f"An axis needs three numbers, not all zero; got {axis!r}.")
        return direction / length

    # ------------------------------------------------------------ the joint value

    @property
    def value(self) -> float:
        """
        The joint position: a distance for a prismatic joint, an angle in degrees for a
        revolute one. Setting it moves everything this frame carries.
        """
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        value = float(value)
        if value != 0 and self.kind == "fixed":
            raise ValueError(f"{self._title()} is fixed; only joints have a value.")
        if self.limits is not None:
            low, high = self.limits
            tolerance = LIMIT_TOLERANCE * max(1.0, abs(low), abs(high))
            if not low - tolerance <= value <= high + tolerance:
                raise ValueError(f"{self._title()}: value {value:g} is outside the limits {self.limits}.")
            value = min(max(value, low), high)  # rounding, e.g. after many small moves
        self._value = value
        self._sync()

    def move(self, distance: float) -> Frame:
        """
        Move the joint by ``distance`` along its axis (an angle in degrees for a revolute joint).

        Returns the frame, so calls can be chained.

        Example:
            >>> scene.animate(lambda i: x.move(0.01), frames=50)
        """
        self.value = self._value + distance
        return self

    # ------------------------------------------------------------ where the frame is

    @property
    def local_pose(self) -> Transform:
        """The frame's pose relative to its parent (including the joint's current value)."""
        if self.kind == "prismatic":
            motion = Transform().translate(self._axis * self._value)
        elif self.kind == "revolute":
            motion = Transform().rotate(self._axis, self._value)
        else:
            motion = Transform()
        return _transform(self._mount @ motion.matrix)

    @property
    def pose(self) -> Transform:
        """The frame's pose in world coordinates: frame coordinates -> world coordinates."""
        local = self.local_pose
        if self.parent is None:
            return local
        return _transform(self.parent.pose.matrix @ local.matrix)

    @property
    def origin(self) -> np.ndarray:
        """The frame's origin in world coordinates."""
        return self.pose.matrix[:3, 3].copy()

    center = origin  # so frames work wherever a center is expected, e.g. ``about="center"``

    @property
    def axes(self) -> np.ndarray:
        """The frame's axes as rows ``[x', y', z']``, in world coordinates."""
        return self.pose.matrix[:3, :3].T.copy()

    @property
    def axis(self) -> np.ndarray | None:
        """The joint axis in world coordinates as it is now, or None for a fixed frame."""
        if self.kind == "fixed":
            return None
        parent = self.parent.pose.matrix if self.parent is not None else np.eye(4)
        return parent[:3, :3] @ self._mount[:3, :3] @ self._axis

    def to_world(self, points) -> np.ndarray:
        """
        Convert points from this frame's coordinates to world coordinates.

        Example:
            >>> z.to_world([0, 0, -0.2])  # where the tool tip, 0.2 below the Z origin, is now
        """
        return self.pose.apply_to_vertices(np.atleast_2d(points)).reshape(np.shape(points))

    def to_local(self, points) -> np.ndarray:
        """Convert points from world coordinates to this frame's coordinates."""
        inverse = _transform(_rigid_inverse(self.pose.matrix))
        return inverse.apply_to_vertices(np.atleast_2d(points)).reshape(np.shape(points))

    @property
    def root(self) -> Frame:
        """The frame at the top of the tree."""
        frame = self
        while frame.parent is not None:
            frame = frame.parent
        return frame

    def walk(self) -> Iterator[Frame]:
        """This frame and all frames below it, parents before children."""
        yield self
        for child in self.children:
            yield from child.walk()

    # ------------------------------------------------------------ attached geometry

    def attach(self, *geometries):
        """
        Let this frame carry geometries: they move along whenever the frame moves.

        Geometries keep their current world position when attached, so build them where they
        should be (e.g. with anchors) and attach them. Points, lines, shapes, annotations and
        groups can all be attached; each geometry to only one frame.

        Returns:
            The geometry (or a tuple of them), so it can be made, attached and added in one line.

        Example:
            >>> carriage = scene.add(x.attach(Box(rail.anchor("top"), size=(0.3, 1, 0.05), anchor="bottom")))
        """
        if not geometries:
            raise TypeError("attach() needs at least one geometry.")
        for geometry in geometries:
            if not hasattr(geometry, "apply_transform"):
                raise TypeError(f"Cannot attach {type(geometry).__name__}; it cannot be moved.")
            for frame in self.root.walk():
                if any(item is geometry for item in frame._attached):
                    raise ValueError(
                        f"This {type(geometry).__name__} is already attached to {frame._title()}."
                    )
            self._attached.append(geometry)
        return geometries[0] if len(geometries) == 1 else geometries

    def detach(self, geometry) -> None:
        """Stop carrying a geometry; it stays where it is now."""
        for index, item in enumerate(self._attached):
            if item is geometry:
                del self._attached[index]
                return
        raise ValueError(f"This {type(geometry).__name__} is not attached to {self._title()}.")

    @property
    def attached(self) -> list:
        """The geometries carried directly by this frame (not by the frames below it)."""
        return list(self._attached)

    def _sync(self) -> None:
        """Move the attached geometry of this frame and all frames below it to where they belong."""
        for frame in self.walk():
            now = frame.pose.matrix
            if not np.array_equal(now, frame._synced):
                step = _transform(now @ _rigid_inverse(frame._synced))
                for geometry in frame._attached:
                    geometry.apply_transform(step)
                frame._synced = now

    # ------------------------------------------------------------ moving the frame itself

    def apply_transform(self, transform: Transform) -> None:
        """
        Move the frame (and everything it carries) by a rigid world transform.

        This changes where the frame sits on its parent, not the joint value. Scaling and
        shear are not allowed, since a frame's axes stay perpendicular unit vectors.
        """
        _, stretch = split_rotation(transform.matrix[:3, :3])
        if not np.allclose(stretch, np.eye(3)):
            raise ValueError("A frame can only be moved and turned, not scaled, sheared or mirrored.")
        parent = self.parent.pose.matrix if self.parent is not None else np.eye(4)
        self._mount = _rigid_inverse(parent) @ transform.matrix @ parent @ self._mount
        self._sync()

    def rotate(self, axis, angle: float, in_degrees: bool = True, *, about=None) -> None:
        """
        Turn the frame (and everything it carries), like :meth:`Geometry.rotate`.

        ``axis`` can also be one of the frame's own axes ``"x'"``, ``"y'"`` or ``"z'"``, and
        ``about="center"`` turns it about its own origin.
        """
        if isinstance(axis, str) and axis.endswith("'"):
            axis = self._direction(axis)
        pivot = resolve_pivot(about, self)
        self.apply_transform(Transform().rotate(axis, angle, in_degrees, about=pivot))

    # ------------------------------------------------------------ printing

    def _title(self) -> str:
        return f"frame {self.name!r}" if self.name else f"this {self.kind} frame"

    def __repr__(self) -> str:
        name = f" {self.name!r}" if self.name else ""
        text = f"Frame{name} ({self.kind}) at {np.round(self.origin, 6)}"
        if self.kind != "fixed":
            text += f", value {self._value:g}"
        return text


def _rotation_only(matrix: np.ndarray) -> np.ndarray:
    rotation = np.eye(4)
    rotation[:3, :3] = matrix[:3, :3]
    return rotation
