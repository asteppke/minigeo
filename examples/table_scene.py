"""
A small room scene built entirely with anchors, in the spirit of TikZ.

Nothing here uses hand-computed coordinates: every object is placed relative to another
one through named points such as "top", "bottom front left" or "near end".

Run it with ``python examples/table_scene.py`` (needs ``pip install ".[mpl]"``),
or copy ``build_scene`` into a notebook.
"""

import matplotlib.pyplot as plt

import minigeo.mpl_geometry as mgeo

TABLE_SIZE = (2.0, 1.0, 0.1)  # width (x), depth (y), thickness (z)
LEG_HEIGHT = 1.0
LEG_SIZE = (0.08, 0.08, LEG_HEIGHT)


def build_scene(ax):
    """Draw the scene into a 3D matplotlib axes and return the objects by name."""
    # matplotlib draws objects in the order they are added (computed_zorder=False below),
    # so things further back are added first. The table is created without drawing it,
    # so that its anchors can already be used for the legs.
    table = mgeo.MplBox(ax, [0, 0, LEG_HEIGHT], TABLE_SIZE, color="peru", anchor="bottom", draw=False)

    def add_leg(corner):
        # Each leg hangs from a bottom corner of the table, flush with the table's edges.
        return mgeo.MplBox(
            ax, table.anchor(f"bottom {corner}"), LEG_SIZE, color="sienna", anchor=f"top {corner}"
        )

    back_legs = [add_leg("back left"), add_leg("back right")]
    table.draw()
    front_legs = [add_leg("front left"), add_leg("front right")]

    # A cup stands near the front left corner of the tabletop (cylinders stand on their position).
    cup_spot = table.anchor("top front left") + [0.3, 0.3, 0]
    cup = mgeo.MplCylinder(ax, cup_spot, (0.08, 0.2), color="C0")

    # A book lies in the middle, turned a little around its own vertical axis.
    book = mgeo.MplBox(ax, [0, 0, 0], (0.5, 0.35, 0.06), color="C3")
    book.rotate("z'", 20, about="center")
    book.place("bottom", at=table.anchor("top"))

    # A lamp is a group: a stand with a shade on top. The group is placed as a whole.
    stand = mgeo.MplCylinder(ax, [0, 0, 0], (0.03, 0.45), color="dimgray")
    shade = mgeo.MplCone(ax, stand.anchor("top"), 0.2, 0.08, 0.25, color="gold")
    lamp = mgeo.MplGroup(ax, [stand, shade])
    lamp.place("bottom", at=table.anchor("top back right") - [0.3, 0.25, 0])

    # A dashed line from the lamp shade to the book, labelled near its end.
    beam = mgeo.MplLine(ax, shade.anchor("bottom"), book.anchor("top"), color="orange")
    beam.line.set_linestyle("--")
    ax.text(*beam.anchor("near end"), "  light", color="darkorange")

    legs = back_legs + front_legs
    return {"table": table, "legs": legs, "cup": cup, "book": book, "lamp": lamp, "beam": beam}


def main():
    fig = plt.figure(figsize=(7, 6))
    # Draw in the order objects were added; matplotlib's automatic depth sorting works per object
    # and would hide the small items behind the large tabletop.
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(0, 2), xlabel="x", ylabel="y", zlabel="z")
    ax.set_box_aspect((3, 3, 2))  # equal scale on all axes, so the boxes are not distorted
    build_scene(ax)
    plt.show()


if __name__ == "__main__":
    main()
