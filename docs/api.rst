=============
API reference
=============

Everything listed here can be imported from ``minigeo`` directly, e.g.
``from minigeo import Box, Scene, Style``, unless a module is named.

.. currentmodule:: minigeo

Scenes and styles
=================

A :class:`Scene` collects geometries with their :class:`Style` and draws them with a backend.

.. autoclass:: Scene
   :members: add, label, add_frame, render, save, animate, snapshots, primitives, bounds

.. autoclass:: Style
   :members: with_, is_complete

Shapes
======

All shapes share the methods of :class:`~minigeo.geometry.Shape`: a pose with their own
axes x', y', z', anchors, transforms, faces and edges.

.. autoclass:: Box
.. autoclass:: Cylinder
.. autoclass:: Cone
.. autoclass:: Sphere
.. autoclass:: Rectangle
.. autoclass:: Circle
.. autoclass:: Polygon

.. autoclass:: minigeo.geometry.Shape
   :members: center, axes, anchor, place, rotate, copy, apply_transform, vertices, local_vertices,
             face_indices, edge_indices, faces, edges, default_anchor, is_closed
   :no-show-inheritance:

Points, lines and vectors
=========================

.. autoclass:: Point
.. autoclass:: Line
   :members: anchor
.. autoclass:: Vector
   :members: between, origin, components, length, direction, dot, cross, angle_to, projected_onto
.. autoclass:: Axis
   :members: length

Planes, angles and dimensions
=============================

.. autoclass:: Plane
   :members: through, normal, point, distance_to, project, contains, intersect, normal_vector
.. autoclass:: Angle
   :members: between, degrees, radians, is_right, anchor
.. autoclass:: Dimension
   :members: length, text, anchor

Groups
======

.. autoclass:: Group
   :members: add, remove, center, anchor, place, rotate, copy

Frames and joints
=================

.. automodule:: minigeo.frames
   :no-members:

.. autoclass:: Frame
   :members: fixed, prismatic, revolute, value, move, attach, detach, attached, pose, local_pose,
             origin, axes, axis, to_world, to_local, root, walk, apply_transform, rotate

Transforms
==========

.. autoclass:: Transform
   :members: translate, rotate, rotate_x, rotate_y, rotate_z, scale, shear, in_frame, apply, apply_to_vertices

.. autofunction:: minigeo.transform.split_rotation

Anchors
=======

.. automodule:: minigeo.anchors
   :members: anchor_direction, line_position

Camera
======

.. autoclass:: minigeo.camera.Camera
   :members: toward_viewer, right, up, light, project

Backends
========

Usually used through :meth:`Scene.render`, :meth:`Scene.save` and :meth:`Scene.animate`.

.. automodule:: minigeo.backends.svg
   :members: render, SvgDrawing

.. automodule:: minigeo.backends.matplotlib
   :members: render, animate, to_figure, to_png, MatplotlibAnimation

.. automodule:: minigeo.backends.plotly
   :members: render, animate

.. automodule:: minigeo.backends.raytrace
   :members: render, RaytracedImage

For backend authors
===================

Backends draw four neutral primitives; :func:`~minigeo.depth.painter_order` sorts them for
backends that paint from back to front.

.. automodule:: minigeo.render
   :members: Face, Path, Marker, Label

.. automodule:: minigeo.depth
   :members: painter_order, is_back_face
