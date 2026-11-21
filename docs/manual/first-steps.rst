===========
First steps
===========

This chapter draws a small table with a cup on it, one step at a time. On the way it shows
the ideas the following chapters explain in detail: scenes, shapes, anchors and styles.

A scene and a box
=================

Everything that is drawn goes into a :class:`~minigeo.Scene`. ``scene.add`` puts a geometry
into the scene and returns it, so it can be created, added and named in one line. A
:class:`~minigeo.Box` is given by a position and its size along x, y and z.

.. minigeo-example::

   scene = Scene()
   scene.add(Box([0, 0, 0], size=(2, 1, 0.1)))

In a notebook, a scene on the last line of a cell shows itself as a picture like this one.

Legs at the corners
===================

Instead of computing where the legs go, we ask the tabletop for its corners. An *anchor* is
a named point of a shape: ``table.anchor("bottom front left")`` is the corner at the bottom,
in front, on the left. Each leg is placed so that its own top corner sits at that point.

.. minigeo-example::

   scene = Scene()
   table = scene.add(Box([0, 0, 1], size=(2, 1, 0.1)))
   for corner in ["front left", "front right",
                  "back left", "back right"]:
       leg = Box(table.anchor(f"bottom {corner}"),
                 size=(0.08, 0.08, 1),
                 anchor=f"top {corner}")
       scene.add(leg)

If the tabletop is made larger or moved, the legs follow, because nothing in the code
depends on numbers we computed by hand.

Colors
======

How things look is not part of the geometry. It is given when a geometry is added to the
scene: either with keywords such as ``fill`` and ``stroke``, or with a :class:`~minigeo.Style`
that can be reused. Keywords given next to a style override it.

.. minigeo-example::

   wood = Style(fill="peru", stroke="saddlebrown")

   scene = Scene()
   table = scene.add(Box([0, 0, 1], size=(2, 1, 0.1)), wood)
   for corner in ["front left", "front right",
                  "back left", "back right"]:
       leg = Box(table.anchor(f"bottom {corner}"),
                 size=(0.08, 0.08, 1),
                 anchor=f"top {corner}")
       scene.add(leg, wood, fill="sienna")

A cup with a label
==================

A :class:`~minigeo.Cylinder` stands on its position: the position is the middle of its
bottom. Anchors are plain points, so we can add an offset to them. ``label`` writes a text
at one of the cup's anchors.

.. minigeo-example::

   wood = Style(fill="peru", stroke="saddlebrown")

   scene = Scene()
   table = scene.add(Box([0, 0, 1], size=(2, 1, 0.1)), wood)
   for corner in ["front left", "front right",
                  "back left", "back right"]:
       leg = Box(table.anchor(f"bottom {corner}"),
                 size=(0.08, 0.08, 1),
                 anchor=f"top {corner}")
       scene.add(leg, wood, fill="sienna")

   spot = table.anchor("top") + [-0.5, 0.2, 0]
   scene.add(Cylinder(spot, radius=0.1, height=0.25), fill="C0",
             label="cup", label_at="top")

From another side
=================

The same scene can be looked at from any side: ``elev`` is the angle above the floor and
``azim`` the angle around the vertical axis. Every backend takes these two angles.

.. minigeo-example::
   :elev: 10
   :azim: 30

   wood = Style(fill="peru", stroke="saddlebrown")

   scene = Scene()
   table = scene.add(Box([0, 0, 1], size=(2, 1, 0.1)), wood)
   for corner in ["front left", "front right",
                  "back left", "back right"]:
       leg = Box(table.anchor(f"bottom {corner}"),
                 size=(0.08, 0.08, 1),
                 anchor=f"top {corner}")
       scene.add(leg, wood, fill="sienna")

   spot = table.anchor("top") + [-0.5, 0.2, 0]
   scene.add(Cylinder(spot, radius=0.1, height=0.25), fill="C0",
             label="cup", label_at="top")

The next chapters look at each of these steps more closely.
