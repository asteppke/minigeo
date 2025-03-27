from typing import Protocol, List
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .geometry import BaseRectangle, BaseBox, BasePolygon, BaseLine, BaseCylinder, BaseCone
from .groups import GeometryGroup


class MplDrawable(Protocol):
    """
    Protocol for drawable objects using Matplotlib.
    """
    def _draw(self) -> None:
        ...



class MplPoly(BasePolygon):
    def __init__(self, ax, center, vertices, color="C0", alpha=1, draw=True):
        super().__init__(center, vertices=vertices)

        self.ax = ax
        self.color = color
        self.alpha = alpha
        self.rotation_matrix = np.eye(4)

        if draw:
            self._draw()

    def _draw(self) -> None:
        self.poly = Poly3DCollection(self.faces, facecolors=self.color, edgecolors="grey", alpha=self.alpha)
        self.ax.add_collection3d(self.poly)

    def update_geometry(self):
        self.poly.set_verts(self.faces)


class MplSurface(BaseRectangle):
    def __init__(self, ax, center, vertices, color="C0", alpha=1, draw=True):
        super().__init__(center, vertices=vertices)

        self.ax = ax
        self.color = color
        self.alpha = alpha
        self.rotation_matrix = np.eye(4)

        if draw:
            self._draw()

    def _draw(self) -> None:
        self.poly = Poly3DCollection(self.faces, facecolors=self.color, edgecolors="grey", alpha=self.alpha)
        self.ax.add_collection3d(self.poly)

    def update_geometry(self):
        self.poly.set_verts(self.faces)


class MplBox(BaseBox):
    def __init__(self, ax, center, dimensions, color="C0", alpha=1, draw=True):
        super().__init__(center, dimensions)

        self.ax = ax

        self.color = color
        self.alpha = alpha
        self.rotation_matrix = np.eye(4)

        if draw:
            self._draw()

    def _draw(self) -> None:
        self.poly = Poly3DCollection(self.faces, facecolors=self.color, edgecolors="grey", alpha=self.alpha)
        self.ax.add_collection3d(self.poly)

    def update_geometry(self):
        self.poly.set_verts(self.faces)



class MplCube(MplBox):
    def __init__(
        self,
        ax,
        center,
        size,
        face_colors=["red", "red", "blue", "blue", "orange", "orange"],
        alpha=0.75,
    ):
        super().__init__(ax, center, (size, size, size), color="white", alpha=alpha, draw=False)
        self.face_colors = face_colors
        self.edge_colors = ["gray"] * len(face_colors)
        self._draw()

    def _draw(self) -> None:
        self.poly = Poly3DCollection(
            self.faces,
            facecolors=self.face_colors,
            edgecolors=self.edge_colors,
            shade=False,
            alpha=self.alpha,
            zorder=2,
        )
        self.ax.add_collection3d(self.poly)

    def add_face_labels(self):
        # Add text labels to cube faces
        original_face_labels = ["Front", "Back", "Bottom", "Top", "Left", "Right"]
        face_labels = ["Front", "ab", "Bottom", "ac", "Left", "bc"]
        for i, face in enumerate(self.faces):
            if face_labels[i] in ["Front", "Bottom", "Left"]:
                continue
            center = np.mean(face, axis=0)
            self.ax.text(
                center[0], center[1], center[2], face_labels[i], color="black", ha="center", va="center"
            )

class MplLine(BaseLine):
    def __init__(self, ax, start, end, color="C0", alpha=1, draw=True, arrow_tip=False):
        super().__init__(start, end)

        self.ax = ax
        self.color = color
        self.alpha = alpha
        self.arrow_tip = arrow_tip

        if draw:
            self._draw()

    def _draw(self) -> None:
        self.line, = self.ax.plot([self.start[0], self.end[0]], [self.start[1], self.end[1]], [self.start[2], self.end[2]], color=self.color, alpha=self.alpha)

        # Draw arrow tip if specified
        if self.arrow_tip:
            self.ax.quiver(self.start[0], self.start[1], self.start[2], self.end[0], self.end[1], self.end[2], color=self.color, arrow_length_ratio=0.2, normalize=False)
    
    def update_geometry(self):
        self.line.set_data_3d([self.start[0], self.end[0]], [self.start[1], self.end[1]], [self.start[2], self.end[2]])

class MplCylinder(BaseCylinder):
    def __init__(self, ax, center, dimensions : tuple, color="C0", alpha=1, draw=True):
        radius, height = dimensions

        super().__init__(center, (radius, height))

        self.ax = ax
        self.color = color
        self.face_colors = color
        self.edge_colors = "grey"
        self.alpha = alpha
        self.rotation_matrix = np.eye(4)

        if draw:
            self._draw()

    def _draw(self) -> None:
        self.poly = Poly3DCollection(
            self.side_faces,
            facecolors=self.face_colors,
            edgecolors=self.edge_colors,
            shade=True,
            alpha=self.alpha,
            zorder=2,
        )
        self.ax.add_collection3d(self.poly)

        self.poly2 = Poly3DCollection(
            self.top_faces,
            facecolors=self.face_colors,
            edgecolors=self.edge_colors,
            shade=True,
            alpha=self.alpha,
            zorder=2,
        )
        self.ax.add_collection3d(self.poly2)

        self.poly3 = Poly3DCollection(
            self.bottom_faces,
            facecolors=self.face_colors,
            edgecolors=self.edge_colors,
            shade=True,
            alpha=self.alpha,
            zorder=2,
        )
        self.ax.add_collection3d(self.poly3)

    def update_geometry(self):
        self.poly.set_verts(self.faces)
        self.poly2.set_verts(self.top_faces)
        self.poly3.set_verts(self.bottom_faces)

class MplCone(BaseCone, MplCylinder):
    def __init__(self, ax, center, lower_radius, upper_radius, height, color="C0", alpha=1, draw=True):
        BaseCone.__init__(self, center, (lower_radius, upper_radius, height))

        self.ax = ax
        self.color = color
        self.face_colors = color
        self.edge_colors = "grey"
        self.alpha = alpha
        self.rotation_matrix = np.eye(4)

        if draw:
            self._draw()


class MplGroup(GeometryGroup):
    def __init__(self, ax, geometries : List[MplDrawable]=None, draw=True):
        """
        Groups Mpl geometries together and keeps their relative positions.

        """
    
        super().__init__(geometries)
        # TODO: think if we need to keep a reference to the ax
        # this is at the moment only to have a similar interface
        self.ax = ax
        if draw:
            self._draw()

    def _draw(self):
        # Call each geometry’s draw method
        for geom in self.geometries:
            if hasattr(geom, '_draw'):
                geom._draw()


class Detector:
    def __init__(self, ax, center, dimensions, color="C0", alpha=1, draw=True, pixel_pitch=0.01):
        self.ax = ax
        self.center = np.array(center)
        self.dimensions = np.array(dimensions)
        self.pixel_pitch = pixel_pitch
        self.color = color
        self.alpha = alpha
        self.rotation_matrix = np.eye(3)
        self.vertices = self._calculate_vertices()
        self.faces = self._calculate_faces()
        if draw:
            self._draw()

        # calc pixel centers
        self.pixel_centers = self._calculate_pixel_centers()

    def _calc_basis_vectors(self):
        x_basis = np.array([1, 0, 0])
        y_basis = np.array([0, 1, 0])
        z_basis = np.array([0, 0, 1])
        return x_basis, y_basis, z_basis

    def _calculate_pixel_centers(self):
        num_pixels_x = int(self.dimensions[0] / self.pixel_pitch)
        num_pixels_y = int(self.dimensions[1] / self.pixel_pitch)
        pixel_centers = []
        for i in range(num_pixels_x):
            for j in range(num_pixels_y):
                x = i * self.pixel_pitch - self.dimensions[0] / 2
                y = j * self.pixel_pitch - self.dimensions[1] / 2
                z = self.center[2]
                pixel_centers.append([x, y, z])
        return np.array(pixel_centers)

    def rotate(self, axis: str, angle: float, in_degrees: bool = True, around_origin: bool = False):
        """
        Rotate the detector around a specified axis.

        Args:
            axis (str): The axis to rotate around ('x', 'y', 'z').
            angle (float): The rotation angle.
            in_degrees (bool): If True, the angle is in degrees. If False, the angle is in radians.
            around_origin (bool): If True, rotate around the origin. If False, rotate around the center of the detector.
        """
        self.transform.rotate(axis, angle, in_degrees)
        self.apply_rotation(self.transform.matrix, around_origin)
