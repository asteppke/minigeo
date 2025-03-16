.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/minigeo.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/minigeo
    .. image:: https://readthedocs.org/projects/minigeo/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://minigeo.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/minigeo/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/minigeo
    .. image:: https://img.shields.io/pypi/v/minigeo.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/minigeo/
    .. image:: https://img.shields.io/conda/vn/conda-forge/minigeo.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/minigeo
    .. image:: https://pepy.tech/badge/minigeo/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/minigeo
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/minigeo

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=======
minigeo
=======


A minimal 3D geometry library for Python. The focus lies on simple interactive usage and compatibility with different visualization libraries or backends.


Examples
--------

````python
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
````









