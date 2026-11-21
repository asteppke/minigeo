=======
Anchors
=======

As in TikZ, an anchor is a named point of a geometry. Anchors make it easy to put things
next to, on top of or below each other without computing coordinates by hand.

Anchors of shapes
=================

The names combine *left*/*right* (along x′), *front*/*back* (along y′) and
*bottom*/*top* (along z′). Seen from the front, x′ points to the right, y′ away from you and
z′ up. ``"center"`` is the middle.

.. minigeo-example::
   :width: 300

   box = Box([0, 0, 0], size=(2, 1, 1))

   scene = Scene()
   scene.add(box, fill="whitesmoke")
   names = ["top", "front", "right",
            "top front left", "bottom front right"]
   for name in names:
       scene.add(Point(box.anchor(name)), fill="C3")
       scene.label(box, name, at=name)

One word names the middle of a face, two words the middle of an edge and three words a
corner. Anchors lie on the shape's box-shaped hull, so a cylinder has them too:
``"top"`` is the middle of its top and ``"top right"`` is on the rim.

Anchors move and turn with the shape, because they are measured along its own axes:

.. minigeo-example::

   box = Box([0, 0, 0], size=(2, 1, 1))
   box.rotate("y", -30, about="center")

   scene = Scene()
   scene.add(box, fill="whitesmoke")
   for name in ["top", "top front right"]:
       scene.add(Point(box.anchor(name)), fill="C3")
       scene.label(box, name, at=name)

For points between the named ones, give three numbers from -1 to 1, one per axis:
``box.anchor([0.5, -1, 1])`` lies halfway to the right on the top front edge.

Creating at an anchor
=====================

When a shape is made, ``anchor`` says which of its points ends up at the position. A box
standing on the floor with its left side at x = 0:

.. minigeo-example::

   scene = Scene()
   low = Box([0, 0, 0], size=1, anchor="bottom left")
   high = Box([0, 0, 0], size=1, anchor="top right")
   scene.add(low, fill="C0")
   scene.add(high, fill="C1")
   scene.add(Point([0, 0, 0]), fill="black")

Placing
=======

``place(anchor, at=point)`` moves a geometry, without turning it, so that one of its anchors
ends up at a point, usually another geometry's anchor. It returns the geometry, so it can
be used right after creating it.

.. minigeo-example::

   base = Box([0, 0, 0], size=(1.5, 1.5, 0.3))
   pillar = Cylinder([0, 0, 0], radius=0.2, height=1)
   pillar.place("bottom", at=base.anchor("top"))
   ball = Sphere([0, 0, 0], radius=0.3)
   ball.place("bottom", at=pillar.anchor("top"))
   side = Box([0, 0, 0], size=0.4)
   side.place("left", at=base.anchor("right"))

   scene = Scene()
   colors = ["C0", "C1", "C2", "C3"]
   for shape, color in zip([base, pillar, ball, side], colors):
       scene.add(shape, fill=color)

Anchors of lines
================

Lines use TikZ's positions along a path: ``"start"``, ``"very near start"``,
``"near start"``, ``"midway"``, ``"near end"``, ``"very near end"`` and ``"end"``, or a
number from 0 (start) to 1 (end).

.. minigeo-example::
   :width: 300

   line = Line([0, 0, 0], [3, 0, 0])

   scene = Scene()
   scene.add(line, stroke="black")
   positions = ["start", "near start", "midway", "near end", "end"]
   for position in positions:
       scene.add(Point(line.anchor(position)), fill="C3")
       scene.label(line, position, at=position)

Anchors of groups
=================

A group has no axes of its own, so its anchors refer to the world axes: they lie on the
box around all members, with left/right along x, front/back along y and bottom/top along z.

.. minigeo-example::

   stand = Cylinder([0, 0, 0], radius=0.04, height=0.5)
   shade = Cone(stand.anchor("top"), radius=0.25, height=0.25,
                top_radius=0.1)
   lamp = Group([stand, shade])
   table = Box([0, 0, 0], size=(1.5, 1, 0.08))
   spot = table.anchor("top back right") + [-0.3, -0.3, 0]
   lamp.place("bottom", at=spot)

   scene = Scene()
   scene.add(table, fill="peru")
   scene.add(stand, fill="dimgray")
   scene.add(shade, fill="gold")
