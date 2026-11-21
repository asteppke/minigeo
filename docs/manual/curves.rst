======
Curves
======

Not everything is made of straight lines between two points. A
:class:`~minigeo.Polyline` connects any number of points, and a :class:`~minigeo.Curve`
draws a function of one parameter ``t``, such as a helix or the path of a particle.

Polylines
=========

A polyline goes through its points in order. ``closed=True`` returns to the first point.
Its anchors are measured along its length: ``"midway"`` is halfway along the whole path,
not halfway through the list of points, and numbers from 0 to 1 work as for lines.

.. minigeo-example::

   steps = Polyline(
       [[0, 0, 0], [1, 0, 0], [1, 0, 0.5], [2, 0, 0.5]]
   )
   frame = Polyline(
       [[0, 1, 0], [2, 1, 0], [2, 2, 0], [0, 2, 0]],
       closed=True,
   )

   scene = Scene()
   scene.add(steps, stroke="C0", line_width=2)
   scene.add(frame, stroke="C2", dash="dashed")
   scene.add(Point(steps.anchor("midway")), fill="C3")
   scene.label(frame, "closed", at=0.375)
   print(f"length {steps.length}, {frame.length}")

Curves from a function
======================

A curve takes a function that turns a number ``t`` into a point, and the range of ``t``.
It is drawn through ``samples`` points (100 by default). ``at(t)`` gives the exact point
for any ``t`` and ``tangent_at(t)`` the direction in which the curve goes on.

.. minigeo-example::

   def spiral(t):
       return np.cos(t), np.sin(t), 0.25 * t

   helix = Curve(spiral, t=(0, 4 * np.pi), samples=200)
   t = 3 * np.pi

   scene = Scene()
   scene.add(helix, stroke="C0", arrow=True, label="helix")
   scene.add(Point(helix.at(t)), fill="C3")
   scene.add(
       Vector(helix.tangent_at(t), origin=helix.at(t)),
       stroke="C3",
   )
   print("length:", round(helix.length, 3))

A curve's label goes at its end, where the arrow is.

Moving curves
=============

Curves and polylines move, turn and scale like everything else. A curve remembers what
was done to it, so ``at`` and ``tangent_at`` keep giving points on the moved curve.

.. minigeo-example::

   wave = Curve(
       lambda t: (t, 0, 0.3 * np.sin(3 * t)), t=(0, 2 * np.pi)
   )
   lift = Transform().rotate("x", 90).translate([0, 1, 0])
   tilted = lift @ wave.copy()

   scene = Scene()
   scene.add(wave, stroke="C0")
   scene.add(tilted, stroke="C1")
   scene.add(Point(tilted.at(np.pi / 6)), fill="C1")
   print(tilted.at(np.pi / 6).round(3))
