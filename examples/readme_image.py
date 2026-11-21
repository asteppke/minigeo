"""
Draw the pictures of the README: the ray tracing example scene in the "diagram" look, and an
animation of a box turning on a plate (an animated PNG).

Run it from the repository's root with ``python examples/readme_image.py`` (needs
``pip install ".[ray]"``); it writes ``examples/diagram.png`` and ``examples/turning.png``.
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


def turning_box():
    """The animation of the manual's chapter on backends: a box turning on a plate."""
    scene = Scene()
    plate = Cylinder([0, 0, 0], radius=1, height=0.1, anchor="top")
    scene.add(plate, fill="whitesmoke", stroke="none")
    box = Box([0, 0, 0.3], size=(1.2, 0.5, 0.6))
    scene.add(box, fill="C1")

    def turn(i):
        box.rotate("z", 20, about="center")

    return scene.animate(turn, frames=18, backend="raytrace", look="diagram", background="white")


if __name__ == "__main__":
    here = Path(__file__).parent
    picture = build_scene().render("raytrace", look="diagram", width=1000, background="white")
    picture.save(here / "diagram.png")
    turning_box().save(here / "turning.png")
