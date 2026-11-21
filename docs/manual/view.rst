========================
Coordinates and the view
========================

minigeo uses right-handed coordinates x, y, z. All positions are in the same units; which
unit that is (metres, millimetres, ångström) is up to you.

The world axes
==============

``scene.add_frame()`` draws the world axes at the origin: x red, y green, z blue. It helps
to keep one in a scene while it is being built.

.. minigeo-example::

   scene = Scene()
   box = Box([1, 0.5, 0.25], size=(2, 1, 0.5))
   scene.add(box, fill="lightsteelblue")
   scene.add_frame(length=1.5)

By default we look at the scene from the front left and a bit from above, with z pointing
up. The x axis goes to the right and toward us, y to the right and away from us.

Looking from another side
=========================

``elev`` is the angle (in degrees) above the horizontal plane, and ``azim`` the angle
around the vertical axis. The defaults are ``elev=30`` and ``azim=-60``, as in matplotlib.

.. minigeo-example::
   :elev: 90
   :azim: -90

   scene = Scene()
   box = Box([1, 0.5, 0.25], size=(2, 1, 0.5))
   scene.add(box, fill="lightsteelblue")
   scene.add_frame(length=1.5)

Seen straight from above (``elev=90``) with ``azim=-90``, x points right and y up, as on
paper.

.. minigeo-example::
   :elev: 0
   :azim: -90

   scene = Scene()
   box = Box([1, 0.5, 0.25], size=(2, 1, 0.5))
   scene.add(box, fill="lightsteelblue")
   scene.add_frame(length=1.5)

``elev=0`` looks horizontally: here from the front, along y.

Another axis pointing up
========================

In some fields another axis points up, for example x in many diffraction experiments.
``vertical_axis`` chooses which world axis is drawn upward. The geometry stays the same;
only the view changes.

.. minigeo-example::
   :vertical-axis: x

   scene = Scene()
   box = Box([1, 0.5, 0.25], size=(2, 1, 0.5))
   scene.add(box, fill="lightsteelblue")
   scene.add_frame(length=1.5)

The camera is orthographic: parallel lines stay parallel and things far away are not drawn
smaller. This keeps lengths and angles easy to compare, as in a technical drawing.
