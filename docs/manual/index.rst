======
Manual
======

This manual introduces minigeo's ideas one at a time, each with small examples. As in the
TikZ & PGF manual, every picture stands next to the code that draws it. The pictures are
drawn from that very code whenever the documentation is built, so they always show what the
current version of minigeo does.

To follow along, install minigeo from PyPI with all backends and the fast ray tracer:
``pip install "minigeo[all]"``. A plain ``pip install minigeo`` installs only the library and
numpy (see the :doc:`overview <../readme>` for single backends).

Every example starts as if with::

   import numpy as np
   from minigeo import *

and draws the scene called ``scene`` with the SVG backend, unless a comment at the end of
the code says otherwise. Anything an example prints appears below its picture.

The chapters build on each other, but each one can also be read on its own. For the
complete list of classes and arguments, see the :doc:`API reference <../api>`.

.. toctree::
   :maxdepth: 2

   first-steps
   shapes
   view
   transforms
   own-axes
   anchors
   styles
   editing
   curves
   meshes
   measuring
   axes
   frames
   backends
   looks
