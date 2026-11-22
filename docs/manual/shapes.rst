=======================
Shapes and other things
=======================

minigeo has a few kinds of geometry. *Shapes* have faces and their own coordinate system;
points, lines and axes are drawn as dots and strokes.

Solids
======

A box is given by its size along x, y and z. Cylinders and cones take a radius and a
height, a sphere a radius. Sizes are always named.

.. minigeo-example::

   scene = Scene()
   scene.add(Box([0, 0, 0], size=(1, 1, 1)), fill="C0")
   can = Cylinder([1.5, 0, -0.5], radius=0.5, height=1)
   scene.add(can, fill="C1")
   scene.add(Cone([3, 0, -0.5], radius=0.5, height=1), fill="C2")
   scene.add(Sphere([4.5, 0, 0], radius=0.5), fill="C3")

A box and a sphere are placed by their middle; cylinders and cones stand on their position,
which is the middle of their bottom. That is why the cylinder and the cone start at
``z = -0.5`` here. The :doc:`anchors` chapter shows how to choose another point.

Round shapes are made of flat faces. ``segments`` sets how many go around; more segments
look rounder. A cone with a ``top_radius`` is cut off at the top.

.. minigeo-example::

   scene = Scene()
   six = Cylinder([0, 0, 0], radius=0.5, height=1, segments=6)
   scene.add(six, fill="C1")
   cut = Cone([1.5, 0, 0], radius=0.5, height=1, top_radius=0.25)
   scene.add(cut, fill="C2")
   ball = Sphere([3, 0, 0.5], radius=0.5, segments=32)
   scene.add(ball, fill="C3")

Flat shapes
===========

Rectangles, circles and polygons lie flat, in the plane of their own x′ and y′ axes. A
polygon's corners are given relative to its position.

.. minigeo-example::

   scene = Scene()
   scene.add(Rectangle([0, 0, 0], size=(1.5, 1)), fill="C0")
   scene.add(Circle([2, 0, 0], radius=0.5), fill="C1")
   triangle = [[0, 0, 0], [1, 0, 0], [0.3, 0.8, 0]]
   scene.add(Polygon([3, -0.4, 0], triangle), fill="C2")

Points, lines and axes
======================

A :class:`~minigeo.Point` is drawn as a dot, a :class:`~minigeo.Line` as a stroke from its
start to its end. An :class:`~minigeo.Axis` is a direction from a point, drawn as an arrow
of the given ``length``.

.. minigeo-example::

   scene = Scene()
   scene.add(Point([0, 0, 0]), fill="black")
   scene.add(Line([0, 0, 0], [1, 1, 0]))
   up = Axis([0, 0, 1], origin=[1, 1, 0], length=0.8)
   scene.add(up, stroke="C3")

Groups
======

A :class:`~minigeo.Group` moves, turns and places several geometries together. Its members
keep their own styles when they are added to the scene one by one.

.. minigeo-example::
   :width: 160

   stand = Cylinder([0, 0, 0], radius=0.04, height=0.5)
   shade = Cone(stand.anchor("top"), radius=0.25, height=0.25,
                top_radius=0.1)
   lamp = Group([stand, shade])
   lamp.rotate("y", 30, about="center")

   scene = Scene()
   scene.add(stand, fill="dimgray")
   scene.add(shade, fill="gold")

Adding the whole group at once draws all members with the same style:
``scene.add(lamp, fill="gold")``.
