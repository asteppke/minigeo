==========
Transforms
==========

A :class:`~minigeo.Transform` moves, turns, scales or shears geometry. It is built step by
step and applied with ``@``: ``transform @ shape`` changes the shape in place and returns it.

Moving and turning
==================

To compare a shape before and after a transform, keep a copy of it. ``copy()`` makes an
independent copy, so transforming one does not change the other.

.. minigeo-example::

   before = Box([1, 0, 0], size=(1, 0.5, 0.25))
   after = before.copy()
   Transform().translate([0, 1, 0.5]) @ after

   scene = Scene()
   scene.add(before, opacity=0.3)
   scene.add(after, fill="C1")
   scene.add_frame()

A rotation turns about an axis through the origin, by the right-hand rule: with the thumb
along the axis, the fingers curl in the direction of a positive angle. Angles are in
degrees.

.. minigeo-example::

   before = Box([1.5, 0, 0], size=(1, 0.5, 0.25))
   after = before.copy()
   Transform().rotate("z", 60) @ after

   scene = Scene()
   scene.add(before, opacity=0.3)
   scene.add(after, fill="C1")
   scene.add_frame()

The box went around the origin. To turn a shape in place, rotate it ``about`` its own
center, about one of its anchors (``about="bottom left"``), or about any other point.
``shape.rotate`` is a shorthand for a rotation transform.

.. minigeo-example::

   before = Box([1.5, 0, 0], size=(1, 0.5, 0.25))
   after = before.copy()
   after.rotate("z", 60, about="center")

   scene = Scene()
   scene.add(before, opacity=0.3)
   scene.add(after, fill="C1")
   scene.add_frame()

The axis can also be any direction, given as a vector:

.. minigeo-example::

   axis = np.array([1, 1, 1])
   cube = Box([0, 0, 0], size=(1, 1, 1))
   cube.rotate(axis, 45, about="center")

   scene = Scene()
   scene.add(cube, fill="C2")
   scene.add(Line(-axis, axis), stroke="black", dash="dashed")

Order matters
=============

The steps of a transform are applied from left to right. Moving and then turning gives a
different result than turning and then moving:

.. minigeo-example::

   move, turn = [1.5, 0, 0], 90
   move_then_turn = Transform().translate(move).rotate("z", turn)
   turn_then_move = Transform().rotate("z", turn).translate(move)

   bar = Box([0, 0, 0], size=(1, 0.4, 0.4))

   scene = Scene()
   scene.add(move_then_turn @ bar.copy(), fill="C0")
   scene.add(turn_then_move @ bar.copy(), fill="C1")
   scene.add_frame()

The blue bar was moved along x and then turned about the origin; the orange bar was turned
first and then moved. Two transforms also combine with ``@``, like matrices: the one on the
right is applied first, so ``rotation @ translation`` moves first and then turns.

Scaling and shearing
====================

Scaling stretches along the world axes; one number scales evenly. Shearing shifts one
coordinate in proportion to another: ``xy`` is how far x moves per unit of y, and likewise
``yx``, ``xz``, ``zx``, ``yz`` and ``zy``.

.. minigeo-example::

   stretch = Transform().scale([2, 1, 0.5])
   shear = Transform().shear(xy=0.6)

   scene = Scene()
   scene.add(stretch @ Box([0, 0, 0], size=1), fill="C0")
   scene.add(shear @ Box([0, 2, 0], size=1), fill="C1")

Transforms are 4 × 4 matrices in homogeneous coordinates, which you can read in
``transform.matrix`` (and pass to ``Transform(matrix)``). ``transform.inverse()`` undoes a
transform, and ``transform.copy()`` makes an independent copy:

.. minigeo-example::

   turn = Transform().rotate("z", 90).translate([1, 0, 0])
   print(turn.matrix.round(3))
   print(turn.apply_to_vertex([1, 0, 0]))
