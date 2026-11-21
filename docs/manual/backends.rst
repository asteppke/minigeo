========
Backends
========

A scene can be drawn in several ways. All backends take the same view options
(``elev``, ``azim``, ``vertical_axis``), so a scene looks the same in each, apart from what
the backend itself adds or cannot do.

SVG
===

``scene.render("svg")`` draws a vector picture, like a TikZ picture: sharp at any size and
small enough to put into documents and web pages. It is what a scene shows in a notebook and
what this manual uses. ``scene.save("scene.svg")`` writes it to a file.

Faces are painted from back to front and shaded by how they turn toward the light. This
cannot always be right: where solids cut through each other, or a line pierces a solid,
one of them is drawn in front as a whole.

.. minigeo-example::

   scene = Scene()
   scene.add(Box([0, 0, 0], size=(1, 1, 1)), fill="C0")
   scene.add(Plane([0, 0, 0], [1, 0, 0.4], size=1.8), fill="gold")

TikZ
====

``scene.render("tikz")`` writes the picture as TikZ code for LaTeX documents. It is drawn
like the SVG picture (same projection, order and shading), but as plain 2D TikZ paths that
compile quickly and can be edited by hand. ``width`` is in centimetres. Labels are LaTeX,
so ``$2\theta$`` is math; symbols such as ′, ° and Greek letters also work with pdflatex.

.. minigeo-example::

   scene = Scene()
   scene.add(Box([0, 0, 0], size=(1, 1, 0.5)), fill="C0",
             label="box", label_at="top")
   picture = scene.render("tikz", width=4)
   lines = picture.code.splitlines()
   print(len(lines), "lines of TikZ, starting with")
   print("\n".join(lines[1:3]))

The picture is the SVG preview of the same scene. The code needs ``\usepackage{tikz}`` and
``\usetikzlibrary{arrows.meta}``.

.. code-block:: python

   picture.save("figure.tex")                    # then \input{figure.tex}
   picture.save("figure.tex", standalone=True)   # a complete document
   picture.save("figure.pdf")                    # compiled with pdflatex
   scene.save("figure.tex", width=6)             # the same in one step

Ray tracing
===========

``scene.render("raytrace")`` follows rays of light through every pixel. Surfaces cast
shadows, transparent faces show what lies behind them, spheres and cylinders are perfectly
round, and solids that cut through each other are drawn correctly. Edges and lines are
hidden where something lies in front of them. The result is an image;
``image.save("scene.png")`` writes it.

.. minigeo-example::
   :backend: raytrace

   scene = Scene()
   scene.add(Box([0, 0, 0], size=(1, 1, 1)), fill="C0")
   scene.add(Plane([0, 0, 0], [1, 0, 0.4], size=1.8), fill="gold")

.. minigeo-example::
   :backend: raytrace

   scene = Scene()
   floor = Box([0, 0, 0], size=(3, 2, 0.1), anchor="top")
   scene.add(floor, fill="whitesmoke")
   scene.add(Sphere([-0.8, 0, 0.4], radius=0.4), fill="C0")
   can = Cylinder([0.2, 0, 0], radius=0.3, height=0.8)
   scene.add(can, fill="C1")
   coarse = Sphere([1, 0.2, 0.3], radius=0.3, segments=6)
   scene.add(coarse, fill="C2", opacity=0.6)

The sphere on the right has only six segments, which the other backends draw as a coarse
polyhedron; the ray tracer still makes it round.

The ray tracer can also draw in other *looks*, for diagrams rather than photos; see
:doc:`looks`.

matplotlib
==========

``scene.render("matplotlib")`` draws into a matplotlib 3D axes, which can be combined with
other plots, given a title and saved in matplotlib's formats. In interactive matplotlib
backends (``%matplotlib widget``) the view can be turned with the mouse.

.. minigeo-example::
   :backend: matplotlib

   scene = Scene()
   scene.add(Box([0, 0, 0], size=(1, 1, 1)), fill="C0")
   can = Cylinder([1.2, 0, -0.5], radius=0.4, height=1)
   scene.add(can, fill="C1")

plotly
======

``scene.render("plotly")`` returns an interactive plotly figure, to turn and zoom in the
browser; ``figure.write_html("scene.html")`` saves it as a web page. It shows true depth,
also where solids intersect. (Plotly figures need a browser, so this manual shows no
picture of one.)

.. code-block:: python

   figure = scene.render("plotly", height=500)
   figure.write_html("scene.html")

Animations
==========

``scene.animate(update, frames)`` draws the scene, then calls ``update(i)`` before each
further frame, which changes the geometry. With plotly the result has a play button and a
slider; with matplotlib it plays live or as an HTML player in Jupyter.

.. code-block:: python

   scene.animate(lambda i: box.rotate("z", 10, about="center"), frames=36)
   scene.animate(lambda i: hinge.move(5), frames=18, backend="matplotlib")

The ray tracer traces every frame and makes an animated PNG, which plays wherever images are
shown: in Jupyter, on web pages and in a README. All frames share one view, so the picture
does not jump while things move, and every look works.

.. minigeo-example::

   scene = Scene()
   plate = Cylinder([0, 0, 0], radius=1, height=0.1,
                    anchor="top")
   scene.add(plate, fill="whitesmoke", stroke="none")
   box = Box([0, 0, 0.3], size=(1.2, 0.5, 0.6))
   scene.add(box, fill="C1")

   def turn(i):
       box.rotate("z", 20, about="center")

   animation = scene.animate(turn, frames=18, backend="raytrace",
                             look="diagram")

``animation.save("turning.png")`` writes the animated PNG; ``.gif`` and ``.webp`` work too.
Tracing takes about half a second per frame.

Saving
======

``scene.save`` chooses the backend from the file extension: ``.svg`` (SVG), ``.html``
(plotly), ``.tex`` (TikZ), ``.png`` and ``.pdf`` (matplotlib). ``backend="raytrace"`` saves
a ray traced image instead, and ``backend="tikz"`` a PDF compiled from TikZ. Options such as ``elev`` or ``width`` are passed on to the backend.

.. code-block:: python

   scene.save("scene.svg", elev=20)
   scene.save("scene.tex", width=8)
   scene.save("scene.png", backend="raytrace", width=1000)
