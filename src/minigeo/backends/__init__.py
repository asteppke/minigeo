"""
Backends draw a :class:`~minigeo.scene.Scene`.

Each backend is a module with a ``render(scene, **options)`` function. It only handles the
neutral primitives from :mod:`minigeo.render`, and imports its plotting library itself, so
that the library is only needed when the backend is used.
"""
