==============
Axes and grids
==============

Coordinate axes with numbered ticks, and grids like graph paper, help to read positions
off a picture. Both are geometries: they are added, styled and moved like any other.

Axes with ticks
===============

:class:`~minigeo.Axes` draws three axes from ``low`` to ``high``, with a tick at every
multiple of ``step`` and the numbers beside them. The axes cross at the origin (or at the
nearest point of the range); that tick gets no number. ``axes.grid("xy")`` makes a grid
with a line at every tick of two axes.

.. minigeo-example::

   axes = Axes(low=[-1, -1, 0], high=[2, 2, 1.5], step=0.5)

   scene = Scene()
   scene.add(axes)
   scene.add(axes.grid("xy"))
   scene.add(Box([1, 1, 0], size=0.5, anchor="bottom"),
             fill="C1")

The ticks of the x axis point along y, those of the y and z axes along x. ``step`` can
also be one number per axis, ``names`` changes the names at the tips, and
``tick_length`` the length of the ticks.

Flat plots
==========

An axis whose ends are equal is left out, so ``high=[4, 3, 0]`` gives a flat x-y
diagram. Seen from above, with a curve, it becomes a plot.

.. minigeo-example::
   :elev: 90
   :azim: -90

   axes = Axes(high=[4, 3, 0], step=1)
   wave = Curve(lambda t: (t, 1.5 + np.sin(2 * t), 0),
                t=(0, 4))

   scene = Scene()
   scene.add(axes.grid("xy"))
   scene.add(axes)
   scene.add(wave, stroke="C3", line_width=1.5)

Grids on their own
==================

``Grid.flat`` makes a grid in the plane of two world axes: ``low`` and ``high`` are its
corners in the two coordinates of the plane, and ``at`` sets the third coordinate. The
border is always drawn, and inside it a line at every multiple of ``step``.

.. minigeo-example::

   scene = Scene()
   scene.add(Grid.flat(low=(-2, -1), high=(2, 1), step=0.5))
   scene.add(Grid.flat(low=(-2, 0), high=(2, 1), step=0.5,
                       plane="xz", at=1),
             stroke="C0")
   scene.add(Sphere([0, 0, 0.5], radius=0.5), fill="C2")
