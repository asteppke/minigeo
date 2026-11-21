=========
Measuring
=========

Besides drawing, minigeo can calculate with geometry: vectors know dot and cross products,
planes know distances and intersections, and angles and dimension lines show the results
in the picture.

Vectors
=======

A :class:`~minigeo.Vector` is an arrow with vector algebra. The results of ``+``, ``-``,
scaling, ``cross`` and ``projected_onto`` start where the first vector starts, so they can
be drawn right away.

.. minigeo-example::

   a = Vector([2, 0.3, 0])
   b = Vector([0.6, 1.6, 0])

   scene = Scene()
   scene.add(a, stroke="C0", label="a")
   scene.add(b, stroke="C2", label="b")
   scene.add(a.cross(b) / 2, stroke="C3", label="a × b / 2")
   scene.add(b.projected_onto(a), stroke="black", dash="dashed")
   print(f"a · b = {a.dot(b):.2f}, angle = {a.angle_to(b):.1f}°")

``Vector.between(p, q)`` is the vector from point ``p`` to point ``q``; ``components``,
``length`` and ``direction`` give its numbers.

Angles
======

An :class:`~minigeo.Angle` draws the arc between two rays from a vertex. Right angles get a
small square instead. ``Angle.between`` takes two lines or vectors.

.. minigeo-example::

   a = Vector([2, 0.3, 0])
   b = Vector([0.6, 1.6, 0])
   c = Vector([0, 0, 1.2])

   scene = Scene()
   for v, color in [(a, "C0"), (b, "C2"), (c, "C3")]:
       scene.add(v, stroke=color)
   angle = Angle.between(a, b, radius=0.5)
   scene.add(angle, label=f"{angle.degrees:.0f}°")
   scene.add(Angle.between(a, c, radius=0.4))

Planes
======

A :class:`~minigeo.Plane` is given by a point and a normal, or by three points. It is drawn
as a translucent square, but calculates as an infinite plane: ``distance_to``, ``project``
and ``intersect``.

.. minigeo-example::

   ground = Plane([0, 0, 0], normal=[0, 0, 1], size=3)
   p = np.array([0.6, 0.3, 1.2])
   foot = ground.project(p)

   scene = Scene()
   scene.add(ground, fill="lightgray")
   scene.add(Point(p), fill="C3", label="p")
   scene.add(Line(p, foot), dash="dashed", stroke="C3")
   scene.add(Point(foot), fill="black")
   print("distance:", ground.distance_to(p))

.. minigeo-example::

   tilted = Plane.through([0, 0, 0], [1, 0, 0.5], [0, 1, 0])
   ray = Line([-0.5, 0.2, 1.5], [0.8, 0.5, -0.5])
   hit = tilted.intersect(ray)

   scene = Scene()
   scene.add(tilted, fill="gold")
   scene.add(tilted.normal_vector(0.8), stroke="black")
   scene.add(ray, stroke="C3")
   scene.add(Point(hit), fill="C3")
   print("meets the plane at", hit.round(3))

Dimension lines
===============

A :class:`~minigeo.Dimension` shows a distance as in a technical drawing, with extension
lines, arrows and the length as text. ``offset`` moves the dimension line away from the
measured points; ``unit`` and ``digits`` format the text.

.. minigeo-example::
   :width: 280

   box = Box([0, 0, 0], size=(2, 1, 0.6), anchor="bottom")

   scene = Scene()
   scene.add(box, fill="peru")
   width = Dimension(box.anchor("bottom front left"),
                     box.anchor("bottom front right"),
                     offset=[0, -0.4, 0], unit=" m")
   height = Dimension(box.anchor("bottom front right"),
                      box.anchor("top front right"),
                      offset=[0.4, 0, 0], unit=" m")
   scene.add(width)
   scene.add(height)

Picking with the mouse
======================

``scene.measure()`` draws the scene with matplotlib for measuring with the mouse. It needs an
interactive backend: ``%matplotlib widget`` in Jupyter (from the ``ipympl`` package), or a
desktop window. Every anchor that lies on a shape is shown as a small dot, as are the ends
and middles of lines:

* moving the mouse highlights the nearest anchor and shows its name and coordinates,
* the first click picks it as point A,
* the second click picks point B and shows the distance and the difference along x, y, z,
* a third click starts a new measurement. Dragging still turns the view.

Picks can also be made from code, by name, which is how the picture here was made:

.. minigeo-example::
   :width: 380

   scene = Scene()
   cube = Box([0, 0, 0], size=1)
   scene.add(cube, fill="C0", label="cube")
   can = Cylinder([2, 0, -0.5], radius=0.4, height=1)
   scene.add(can, fill="C1")

   tool = scene.measure(show_axes=False)
   tool.pick("cube: top front right")
   tool.pick("Cylinder 1: top")
   print(f"{tool.measurement.distance:.3f}")
   figure = tool.figure

Geometries are named by their label, or by their kind and number (``"Cylinder 1"``). After
measuring, ``tool.measurement`` holds the picks, ``distance`` and ``difference``, for use in
further calculations.

.. code-block:: python

   %matplotlib widget
   tool = scene.measure()

