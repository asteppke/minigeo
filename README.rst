.. image:: https://github.com/asteppke/minigeo/actions/workflows/python-app.yml/badge.svg
   :alt: pytest
   :target: https://github.com/asteppke/minigeo/actions/workflows/python-app.yml


=======
minigeo
=======


A minimalistic 3D geometry library designed for visualizing 3D shapes and transformations using Python. The focus lies on simple interactive usage and compatibility with different visualization libraries or backends. 
It provides a set of base classes for 3D geometry objects, such as points, lines, shapes, and axes, and is independent of any specific visualization library.

Features
--------
- **3D Geometry Classes**: Base classes for various geometric shapes.
- **Transformations**: Support for affine transformations in 3D space.
- **Grouping**: Ability to group multiple geometries and manipulate them collectively.
- **Matplotlib Integration**: Classes for rendering geometries using Matplotlib.

Examples
--------
.. code-block::
   :caption: Example of creating a box and applying a translation
   :name: example-1

   import minigeo.geometry as geo
   
   # Create a box at the origin with dimensions 1x2x3
   box = geo.BaseBox([0, 0, 0], dimensions=[1, 2, 3])
   
   # Define a translation along the x-axis by 1
   move_x = geo.Transform().translate([1, 0, 0])
   
   # and apply it to the box
   move_x @ box
   
   print(box)
   
   BaseBox at [1. 0. 0.] with vertices [[ 0.5 -1.  -1.5]
    [ 1.5 -1.  -1.5]
    [ 1.5  1.  -1.5]
    [ 0.5  1.  -1.5]
    [ 0.5 -1.   1.5]
    [ 1.5 -1.   1.5]
    [ 1.5  1.   1.5]
    [ 0.5  1.   1.5]]


.. code-block::
   :caption: Placing shapes with TikZ-style anchors

   import minigeo.geometry as geo

   table = geo.BaseBox([0, 0, 0], dimensions=[2, 1, 0.1], anchor="bottom")

   # anchors are points on a shape: "top", "top front", "bottom back left", ...
   cup = geo.BaseCylinder(table.anchor("top front left"), dimensions=[0.1, 0.2])

   # or move an existing shape so that one of its anchors sits on a point
   book = geo.BaseBox([0, 0, 0], dimensions=[0.5, 0.35, 0.08])
   book.rotate("z", 20, about="center")
   book.place("bottom", at=table.anchor("top"))

Every shape has its own axes x', y', z' (``shape.axes``) that turn with it. Anchor words
refer to these axes: left/right (x'), front/back (y'), bottom/top (z'). So after rotating
a shape, ``shape.anchor("top")`` is still the middle of its own top face, and
``shape.rotate("x'", 30, about="center")`` tilts it around its own x' axis.

Lines use TikZ's position names (``"start"``, ``"near start"``, ``"midway"``, ``"near end"``,
``"end"``, or a number between 0 and 1), and groups have the same anchors as shapes, taken
from their bounding box along the world axes:

.. code-block::

   from minigeo.groups import GeometryGroup

   connector = geo.BaseLine(table.anchor("right"), table.anchor("right") + [1, 0, 0])
   label_at = connector.anchor("near end")

   lamp = GeometryGroup([geo.BaseCylinder([0, 0, 0], dimensions=[0.05, 0.4]),
                         geo.BaseCone([0, 0, 0.4], dimensions=[0.2, 0.1, 0.2])])
   lamp.place("bottom", at=table.anchor("top back right"))


Installation
------------
Clone the repository and install it with the matplotlib backend:

.. code-block:: bash

    pip install ".[mpl]"

For development (tests and linting):

.. code-block:: bash

    pip install -e ".[dev]"
    pytest
    ruff check .



How does it look?
------------
The obligatory screenshot:

.. image:: examples/minigeo.png
   :alt: Example Image
   :width: 600px


Contributing
-------------
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.
