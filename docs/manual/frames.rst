=================
Frames and joints
=================

Machines are chains of parts that carry other parts: a turntable carries an arm, an X
carriage carries the Y carriage. A :class:`~minigeo.Frame` is a coordinate system that
carries geometry and further frames. A *joint* is a frame whose position on its parent is
set by one number, its ``value``.

A hinge
=======

``revolute(axis, at=...)`` makes a turning joint; its value is an angle in degrees. Parts are
built as usual, where they are at value 0, and then attached to the joint that carries them.

.. minigeo-example::

   post = Cylinder([0, 0, 0], radius=0.08, height=0.6)
   hinge = Frame().revolute("z", at=post.anchor("top"))
   arm = Box(post.anchor("top"), size=(1, 0.1, 0.1),
             anchor="bottom left")
   hinge.attach(arm)
   hinge.value = 60

   scene = Scene()
   scene.add(post, fill="gray")
   scene.add(arm, fill="C1")
   scene.add_frame(hinge, length=0.4)

Values are absolute: ``hinge.value = 60`` always gives the same picture, however often it
is set. ``hinge.move(10)`` turns by another 10°.

A chain
=======

Joints made from a joint are carried by it. Here a turntable carries a slide, which carries
a block: the slide moves along the turntable's own x′ axis, wherever the turntable points.

.. minigeo-example::

   base = Frame()
   turn = base.revolute("z")
   slide = turn.prismatic("x'", limits=(0, 1))

   table = Cylinder([0, 0, 0], radius=1, height=0.1, anchor="top")
   rail = Box(table.anchor("top"), size=(1.8, 0.15, 0.05),
              anchor="bottom")
   turn.attach(rail)
   block = Box(rail.anchor("top"), size=0.3, anchor="bottom")
   slide.attach(block)

   turn.value = 40
   slide.value = 0.5

   scene = Scene()
   scene.add(table, fill="whitesmoke")
   scene.add(rail, fill="slategray")
   scene.add(block, fill="C0")

``prismatic`` makes a sliding joint; its value is a distance. ``limits`` keeps a value in
range: setting ``slide.value = 2`` raises an error. Axis names without a prime are world
axes, names with a prime the parent frame's own axes.

Where is the tool?
==================

A fixed frame marks a point of interest, such as a tool tip, and follows its parent.
``origin`` and ``to_world`` give positions in world coordinates as things stand now.

.. minigeo-example::

   base = Frame()
   x = base.prismatic("x")
   z = x.prismatic("z")
   tip = z.fixed(at=[0, 0, 0.5])

   x.value, z.value = 0.3, -0.1
   print("tip at", tip.origin.round(3))
   print("10 cm above it:", tip.to_world([0, 0, 0.1]).round(3))

Moving parts by hand
====================

Points and transforms are given in world coordinates as things stand now. Parts stay
attached when they are moved, so the joint carries them along from their new place. To move
along a frame's own axes, read the transform in the frame with ``in_frame``.

.. minigeo-example::

   hinge = Frame().revolute("z")
   arm = hinge.attach(Box([0.5, 0, 0], size=(1, 0.1, 0.1)))
   hinge.value = 90  # the arm now points along y

   weight = hinge.attach(Box([0, 0, 0], size=(0.2, 0.2, 0.2)))
   along_arm = Transform().translate([0.8, 0, 0]).in_frame(hinge)
   along_arm @ weight

   hinge.value = 30
   scene = Scene()
   scene.add(arm, fill="C1")
   scene.add(weight, fill="C0")
   scene.add_frame(length=0.5)

The weight was moved along the arm while the arm pointed along y; after turning the hinge
back to 30°, it is still at the same place on the arm.
