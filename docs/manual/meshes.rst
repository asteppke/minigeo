======
Meshes
======

The built-in shapes cover boxes, cylinders, cones and spheres. For anything else there is
:class:`~minigeo.Mesh`: a shape made of any faces, written out in code or read from a
file that another program made.

Vertices and faces
==================

A mesh is given by its corner points and its faces. Each face lists the numbers of its
corners (counting from 0) in order around the face. A mesh behaves like the other shapes:
it has anchors on its bounding box, and it moves and turns.

.. minigeo-example::

   pyramid = Mesh(
       [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
        [0, 0, 1.2]],
       [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4],
        [3, 0, 4]],
   )

   scene = Scene()
   scene.add(pyramid, fill="gold", label="pyramid",
             label_at="top")
   print(pyramid.is_closed, round(pyramid.volume, 3))

When every edge belongs to exactly two faces, the mesh is *closed*: a solid with a
``volume``. Closed meshes are turned outward automatically, so the order of the corners
only has to agree from face to face.

Surfaces and outlines
=====================

A grid of points makes a surface, here the graph of a function ``z = f(x, y)``. Not every
edge gets an outline: only the border, and edges where faces meet at more than
``crease`` degrees (30 by default). So a smooth surface shows its shape and not its
grid, while a cube made of triangles is outlined like a box. ``crease=None`` outlines
every edge.

.. minigeo-example::

   n = 20
   x, y = np.meshgrid(np.linspace(-2, 2, n),
                      np.linspace(-2, 2, n))
   z = np.exp(-(x**2 + y**2))
   points = np.column_stack(
       [x.ravel(), y.ravel(), z.ravel()]
   )
   squares = [
       [i * n + j, i * n + j + 1,
        (i + 1) * n + j + 1, (i + 1) * n + j]
       for i in range(n - 1) for j in range(n - 1)
   ]

   scene = Scene()
   scene.add(Mesh(points, squares), fill="C0")

Files: OBJ and STL
==================

minigeo reads and writes the two most common mesh formats, without any extra package:

* **OBJ** keeps faces with any number of corners and named objects. Colors and opacities
  go into a material file (``.mtl``) next to it. Most 3D programs, such as Blender, read it.
* **STL** keeps only triangles, without names or colors. 3D printers and CAD programs
  exchange it.

``Scene.save`` writes every shape of a scene, named after its label. Spheres and
cylinders become the facets they are drawn with, and lines and polylines become OBJ lines.
:func:`~minigeo.read_mesh` reads a whole file as one mesh, and
:func:`~minigeo.read_meshes` gives one mesh per object of an OBJ file::

   scene.save("table.obj")    # with colors in table.mtl
   scene.save("table.stl")    # for a 3D printer

   bracket = read_mesh("bracket.stl")
   bracket.place("bottom", at=[0, 0, 0])
   parts = read_meshes("robot.obj")
   parts["arm"].rotate("z", 30, about="bottom")

``write_mesh(path, shapes)`` writes a list of shapes without a scene. CAD formats such as
STEP describe exact surfaces rather than faces and need a CAD kernel to read; export
them as STL from the CAD program instead.
