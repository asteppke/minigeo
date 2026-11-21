"""
Draw the picture at the top of the README: the ray tracing example scene in the "diagram" look.

Run it from the repository's root with ``python examples/readme_image.py`` (needs
``pip install ".[ray]"``); it writes ``examples/diagram.png``.
"""

from pathlib import Path

from minigeo import Box, Cone, Cylinder, Scene, Sphere


def build_scene() -> Scene:
    """The scene of the manual's chapter on looks."""
    scene = Scene()
    floor = Box([0, 0, 0], size=(3, 2, 0.1), anchor="top")
    scene.add(floor, fill="whitesmoke", stroke="none")
    scene.add(Sphere([-0.8, 0, 0.4], radius=0.4), fill="C0")
    scene.add(Cylinder([0.2, 0, 0], radius=0.3, height=0.8), fill="C1")
    scene.add(Cone([1, 0.2, 0], radius=0.3, height=0.6), fill="C2")
    return scene


if __name__ == "__main__":
    image = build_scene().render("raytrace", look="diagram", width=1000, background="white")
    image.save(Path(__file__).parent / "diagram.png")
