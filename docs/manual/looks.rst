==================
Looks for diagrams
==================

The ray tracer draws scenes with light and shadows, like a photo. For diagrams in papers,
slides and lecture notes, a calmer picture often reads better: flat bands of color, dark
outlines, soft shading in corners. These *looks* are chosen with ``look=``, and every single
setting can be changed on top of a look.

All examples on this page draw the same small scene:

.. code-block:: python

   scene = Scene()
   floor = Box([0, 0, 0], size=(3, 2, 0.1), anchor="top")
   scene.add(floor, fill="whitesmoke", stroke="none")
   scene.add(Sphere([-0.8, 0, 0.4], radius=0.4), fill="C0")
   can = Cylinder([0.2, 0, 0], radius=0.3, height=0.8)
   scene.add(can, fill="C1")
   scene.add(Cone([1, 0.2, 0], radius=0.3, height=0.6), fill="C2")

Natural and soft
================

``"natural"`` is the default: smooth light from the upper left and cast shadows.

.. minigeo-example::
   :backend: raytrace
   :render: look="natural"

   scene = Scene()
   floor = Box([0, 0, 0], size=(3, 2, 0.1), anchor="top")
   scene.add(floor, fill="whitesmoke", stroke="none")
   scene.add(Sphere([-0.8, 0, 0.4], radius=0.4), fill="C0")
   can = Cylinder([0.2, 0, 0], radius=0.3, height=0.8)
   scene.add(can, fill="C1")
   scene.add(Cone([1, 0.2, 0], radius=0.3, height=0.6), fill="C2")

``"soft"`` leaves out the hard shadows and uses *ambient occlusion* instead: places that
light from the surroundings reaches less easily (corners, creases, the floor right next to
an object) become darker. It shows how things touch and stand on each other.

.. minigeo-example::
   :backend: raytrace
   :render: look="soft"

   scene = Scene()
   floor = Box([0, 0, 0], size=(3, 2, 0.1), anchor="top")
   scene.add(floor, fill="whitesmoke", stroke="none")
   scene.add(Sphere([-0.8, 0, 0.4], radius=0.4), fill="C0")
   can = Cylinder([0.2, 0, 0], radius=0.3, height=0.8)
   scene.add(can, fill="C1")
   scene.add(Cone([1, 0.2, 0], radius=0.3, height=0.6), fill="C2")

Cel shading and outlines
========================

``"cel"`` shades in a few flat bands, as in comics and technical illustrations, and draws
outlines around every object and along sharp edges. The outlines come from the rendered
picture, so they follow the exact round shapes.

.. minigeo-example::
   :backend: raytrace
   :render: look="cel"

   scene = Scene()
   floor = Box([0, 0, 0], size=(3, 2, 0.1), anchor="top")
   scene.add(floor, fill="whitesmoke", stroke="none")
   scene.add(Sphere([-0.8, 0, 0.4], radius=0.4), fill="C0")
   can = Cylinder([0.2, 0, 0], radius=0.3, height=0.8)
   scene.add(can, fill="C1")
   scene.add(Cone([1, 0.2, 0], radius=0.3, height=0.6), fill="C2")

``"diagram"`` combines cel shading and outlines with ambient occlusion and no cast shadows:
clear shapes, and still a sense of what touches what.

.. minigeo-example::
   :backend: raytrace
   :render: look="diagram"

   scene = Scene()
   floor = Box([0, 0, 0], size=(3, 2, 0.1), anchor="top")
   scene.add(floor, fill="whitesmoke", stroke="none")
   scene.add(Sphere([-0.8, 0, 0.4], radius=0.4), fill="C0")
   can = Cylinder([0.2, 0, 0], radius=0.3, height=0.8)
   scene.add(can, fill="C1")
   scene.add(Cone([1, 0.2, 0], radius=0.3, height=0.6), fill="C2")

For print
=========

``"print"`` is the diagram look in gray tones, for black-and-white printing. Lines and labels
turn gray as well.

.. minigeo-example::
   :backend: raytrace
   :render: look="print"

   scene = Scene()
   floor = Box([0, 0, 0], size=(3, 2, 0.1), anchor="top")
   scene.add(floor, fill="whitesmoke", stroke="none")
   scene.add(Sphere([-0.8, 0, 0.4], radius=0.4), fill="C0")
   can = Cylinder([0.2, 0, 0], radius=0.3, height=0.8)
   scene.add(can, fill="C1")
   scene.add(Cone([1, 0.2, 0], radius=0.3, height=0.6), fill="C2")

Changing single settings
========================

Settings given next to ``look`` change only that part of it: here two bands and thick
outlines in dark blue.

.. minigeo-example::
   :backend: raytrace
   :render: look="diagram", bands=2, outline_width=3, outline_color="midnightblue"

   scene = Scene()
   floor = Box([0, 0, 0], size=(3, 2, 0.1), anchor="top")
   scene.add(floor, fill="whitesmoke", stroke="none")
   scene.add(Sphere([-0.8, 0, 0.4], radius=0.4), fill="C0")
   can = Cylinder([0.2, 0, 0], radius=0.3, height=0.8)
   scene.add(can, fill="C1")
   scene.add(Cone([1, 0.2, 0], radius=0.3, height=0.6), fill="C2")

A ``palette`` replaces every surface color with the nearest of a few colors, so that a set of
figures shares the same restrained colors:

.. minigeo-example::
   :backend: raytrace
   :render: look="cel", palette=["whitesmoke", "slategray", "goldenrod"]

   scene = Scene()
   floor = Box([0, 0, 0], size=(3, 2, 0.1), anchor="top")
   scene.add(floor, fill="whitesmoke", stroke="none")
   scene.add(Sphere([-0.8, 0, 0.4], radius=0.4), fill="C0")
   can = Cylinder([0.2, 0, 0], radius=0.3, height=0.8)
   scene.add(can, fill="C1")
   scene.add(Cone([1, 0.2, 0], radius=0.3, height=0.6), fill="C2")

The settings
============

=====================  ============================================================
``shading``            ``"smooth"``, ``"cel"`` or ``"flat"`` (no light at all)
``bands``              brightness steps of cel shading
``shadows``            whether objects cast shadows
``ambient``            brightness without direct light, 0 to 1
``occlusion``          strength of ambient occlusion, 0 (off) to 1
``occlusion_radius``   how far surfaces darken each other, as a part of the scene size
``outlines``           outlines around objects and along sharp edges
``outline_color``      their color
``outline_width``      their width, in pixels of a 500-pixel-wide image
``crease_angle``       edges that bend more than this (in degrees) are outlined
``palette``            colors that surface colors are snapped to
``grayscale``          gray tones for print
=====================  ============================================================

A look can also be kept and reused as an object:

.. code-block:: python

   from minigeo.backends.raytrace import LOOKS

   house_style = LOOKS["diagram"].with_(bands=2, palette=["whitesmoke", "slategray"])
   scene.render("raytrace", look=house_style)
