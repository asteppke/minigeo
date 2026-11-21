==================
A shape's own axes
==================

Every shape carries its own coordinate system, with axes x′, y′ and z′. When the shape is
made, they point along x, y and z; when the shape turns, they turn with it. They are what
"left", "front" and "top" refer to, and they can be used to turn a shape about itself.

Showing the axes
================

``frame=True`` draws a shape's own axes at its center: x′ red, y′ green, z′ blue.

.. minigeo-example::

   box = Box([0, 0, 0], size=(2, 1, 0.5))
   box.rotate("z", -20, about="center")
   box.rotate("x'", 40, about="center")

   scene = Scene()
   scene.add(box, fill="lightsteelblue", frame=True)
   scene.add_frame([-1.5, -1.5, 0], length=0.6)

The box was first turned about the world's z axis, then about its *own* x′ axis, its long
side. Rotations accept ``"x'"``, ``"y'"`` and ``"z'"`` next to ``"x"``, ``"y"`` and ``"z"``.
The small frame in the corner shows the world axes for comparison.

The axes are a property of the shape:

.. minigeo-example::

   box = Box([0, 0, 0], size=(2, 1, 0.5))
   box.rotate("z", 90)
   print("x' y' z' as rows:")
   print(box.axes.round(3))
   print("center:", box.center.round(3))

Moving along the shape's axes
=============================

``Transform().in_frame(shape)`` reads a transform in the shape's own coordinates: a
translation along ``[1, 0, 0]`` then moves along x′, wherever x′ points.

.. minigeo-example::

   box = Box([0, 0, 0], size=(1, 0.5, 0.25))
   box.rotate("z", 45, about="center")
   moved = box.copy()
   Transform().translate([1.2, 0, 0]).in_frame(moved) @ moved

   scene = Scene()
   scene.add(box, opacity=0.3, frame=True)
   scene.add(moved, fill="C1", frame=True)

Stretching keeps the axes upright
=================================

Scaling and shearing change a shape's form, but its axes stay perpendicular to each other.
A turned box stretched along the world's x axis becomes slanted, while its frame stays a
right-angled frame that follows the box.

.. minigeo-example::

   box = Box([0, 0, 0], size=(1, 1, 1))
   box.rotate("z", 30, about="center")
   Transform().scale([2, 1, 1]) @ box

   scene = Scene()
   scene.add(box, fill="lightsteelblue", frame=True)

This is a deliberate choice: anchors and rotations about ``"x'"`` keep a clear meaning,
however the shape was deformed.
