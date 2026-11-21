=================
Styles and labels
=================

Geometry says *where* things are; a :class:`~minigeo.Style` says how they look. Styles are
given when geometry is added to a scene, so the same shape can be drawn differently in two
scenes.

Fill, stroke and opacity
========================

``fill`` colors faces and points, ``stroke`` lines and the outlines of faces. Colors are CSS
names (``"peru"``), hex codes (``"#3388ff"``), ``"C0"`` to ``"C9"`` for matplotlib's
default colors, or ``"none"``. ``opacity`` goes from 0 (invisible) to 1.

.. minigeo-example::

   scene = Scene()
   scene.add(Box([0, 0, 0], size=(1, 1, 1)), fill="C0")
   yellow = Box([1.5, 0, 0], size=1)
   scene.add(yellow, fill="#ffcc00", stroke="black")
   scene.add(Box([3, 0, 0], size=1), fill="C2", opacity=0.4)
   scene.add(Box([4.5, 0, 0], size=1), fill="none", stroke="C3")

Lines
=====

Lines have a ``line_width`` in points, a ``dash`` style (``"solid"``, ``"dashed"`` or
``"dotted"``) and an optional ``arrow`` tip at their end.

.. minigeo-example::

   scene = Scene()
   for i, dash in enumerate(["solid", "dashed", "dotted"]):
       line = Line([0, i, 0], [2, i, 0])
       scene.add(line, dash=dash, line_width=1.5)
   arrow = Line([0, 3, 0], [2, 3, 0])
   scene.add(arrow, arrow=True, stroke="C3", line_width=1.5)

Reusing and combining styles
============================

A :class:`~minigeo.Style` only holds the fields that were set. Styles combine with ``|``:
in ``base | change``, every field set in ``change`` wins. Keywords next to a style work
the same way.

.. minigeo-example::

   glass = Style(fill="lightblue", stroke="steelblue", opacity=0.5)
   red_glass = glass | Style(fill="salmon", stroke="firebrick")

   scene = Scene()
   scene.add(Cylinder([0, 0, 0], radius=0.4, height=1), glass)
   scene.add(Cylinder([1, 0, 0], radius=0.4, height=1), red_glass)
   scene.add(Cylinder([2, 0, 0], radius=0.4, height=1), glass,
             opacity=1)

A scene can have a style of its own, which applies to everything in it. Styles are combined
from general to specific: minigeo's defaults, defaults for the kind of geometry (axes are
arrows, planes translucent), the scene's style, and the item's own style.

.. minigeo-example::

   drawing = Style(fill="whitesmoke", stroke="black",
                   line_width=0.6)
   scene = Scene(style=drawing)
   scene.add(Box([0, 0, 0], size=(1, 1, 1)))
   scene.add(Sphere([1.4, 0, 0], radius=0.5))
   scene.add(Cone([2.8, 0, -0.5], radius=0.5, height=1), fill="C1")

Single faces
============

``face_styles`` styles single faces of a shape. Box faces have names: ``"bottom"``,
``"top"``, ``"front"``, ``"back"``, ``"left"`` and ``"right"``; other shapes use face
numbers.

.. minigeo-example::
   :width: 160

   scene = Scene()
   scene.add(
       Box([0, 0, 0], size=(1, 1, 1)),
       fill="whitesmoke",
       face_styles={"top": Style(fill="C3"),
                    "front": Style(fill="C0")},
   )

Labels
======

``label`` writes a text at a geometry's anchor when it is added; ``scene.label`` adds more
texts later, at any anchor or at a fixed point. Labels follow the geometry when it moves.
``text_color`` and ``font_size`` set how they look.

.. minigeo-example::

   box = Box([0, 0, 0], size=(2, 1, 0.5))

   scene = Scene()
   scene.add(box, fill="whitesmoke", label="box", label_at="top")
   scene.label(box, "front", at="front", text_color="C0")
   scene.label(box, "right end", at="right", text_color="C3",
               font_size=8)
   scene.label([0, 0, 1], "a point above", text_color="gray")

Math in labels
==============

Text between ``$`` signs is math, written as in LaTeX. A raw string (``r"..."``) keeps the
backslashes as they are. A single ``$``, as in ``"costs $5"``, stays a dollar sign.

.. minigeo-example::

   a = Vector([2, 0.3, 0])
   b = Vector([0.6, 1.6, 0])
   angle = Angle.between(a, b, radius=0.5)

   scene = Scene()
   scene.add(a, label=r"$\vec{a}$")
   scene.add(b, label=r"$\vec{b}$")
   scene.add(angle, label=r"$\theta = \frac{\pi}{3}$")
   scene.add(
       Dimension([0, -0.4, 0], [2, -0.4, 0],
                 text=r"$\ell_0 = 2\,\mathrm{m}$")
   )

The same labels work in every backend. matplotlib typesets them itself, and TikZ hands
them to LaTeX. SVG and the ray tracer draw the letters that matplotlib's mathtext lays
out, so they need matplotlib (``pip install "minigeo[mpl]"``) but no LaTeX and no fonts
in the finished picture. Plotly cannot show math in 3D scenes; it gets Greek letters
and raised and lowered numbers instead: ``x²``, ``θ₁``.
