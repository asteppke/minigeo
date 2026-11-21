===============
Editing a scene
===============

A scene keeps references to its geometries, so moving a geometry after adding it moves it
in the next picture. Beyond that, geometries can be taken out again, and layers group
parts of a scene so they can be hidden, shown and cleared together.

Removing
========

``scene.remove(geometry)`` takes a geometry out, together with its labels and its frame.

.. minigeo-example::

   scene = Scene()
   box = scene.add(Box([0, 0, 0], size=1), label="box")
   ball = scene.add(Sphere([1.5, 0, 0], radius=0.5),
                    fill="C1", label="ball")
   scene.remove(box)
   print(scene)

Layers
======

Everything sits on a layer: ``"main"``, unless ``add``, ``label`` or ``add_frame`` get a
``layer``. ``scene.hide`` leaves layers out of every backend until ``scene.show`` brings
them back (``scene.show()`` shows all). ``scene.clear("name")`` removes a layer's content
for good, and ``scene.clear()`` everything.

.. minigeo-example::

   scene = Scene()
   box = scene.add(Box([0, 0, 0], size=(2, 1, 1)), fill="C0")
   for corner in ["top front left", "top back right"]:
       scene.add(Point(box.anchor(corner)), fill="C3",
                 layer="construction")
   scene.add(Dimension(box.anchor("bottom front left"),
                       box.anchor("bottom front right"),
                       offset=[0, -0.4, 0]),
             layer="dimensions")
   scene.hide("construction")
   print(scene.layers, scene.hidden)

A scene's label goes on the layer of its geometry when it is given with ``add``. Hidden
layers are also left out when picking points (:doc:`measuring`) and when saving meshes.
