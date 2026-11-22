# minigeo plan

minigeo is TikZ for 3D in a Jupyter notebook: describe simple geometry in a few readable
lines, place things relative to each other, and look at the result from any side. It is for
sketches, teaching and understanding geometry. **Readability beats speed** in every trade-off.

This file guides decisions. When a change conflicts with a rule below, either follow the
rule or change the rule here first, on purpose.

## Design rules

1. **Geometry knows nothing about drawing.** Shapes, frames and transforms never import a
   backend. A `Scene` adds styles; backends turn the scene into output.
2. **Backends only draw four primitives**: `Face`, `Path`, `Marker`, `Label`
   (`render.py`). A new backend is a new module in `backends/` plus an entry in `BACKENDS`,
   and needs no change to the geometry. Backends that need more than primitives (exact
   spheres and cylinders for ray tracing) use one shared, explicit hook, not `isinstance`
   checks in each backend.
3. **World coordinates in, relative storage inside.** Everything a user passes (points,
   anchors, axes, transforms) is in world coordinates as things stand now. Shapes store
   vertices relative to their pose; attached parts store themselves relative to their frame.
   Frame-relative input is explicit: `"x'"`, `Transform().in_frame(...)`.
4. **A pose is rigid.** Shapes have perpendicular, right-handed axes x′ y′ z′. Scaling,
   shear and mirroring go into the local vertices; faces keep winding outward.
5. **TikZ words.** Anchors (`"top front left"`, `"near end"`), `place(anchor, at=...)`,
   named sizes (`radius=`, `size=`) and keyword-only options.
6. **The same camera everywhere.** `elev`, `azim`, `vertical_axis` mean the same in every
   backend and match matplotlib.
7. **Core depends on numpy only.** Every backend is an optional extra (`mpl`, `svg`, `plotly`,
   `ray`, ...) and fails with a message naming the extra.
8. **Every feature arrives with** a docstring example, tests, a manual section with
   `minigeo-example` blocks (code lines of at most 64 characters), and an entry in
   `docs/api.rst`. Geometric facts go into `tests/test_geometry_foundations.py`.
9. **Compatibility.** Renamed API keeps the old name for one minor version with a
   `DeprecationWarning` (module `__getattr__`, warn once).
10. **Tooling.** Python ≥ 3.12, ruff, pytest, `sphinx-build -W` in CI. Examples must run.

## Architecture

```
geometry/   points, lines, vectors, curves, shapes (pose + local vertices), meshes, planes,
            annotations
meshfiles.py  OBJ (+ .mtl colors) and STL read/write, numpy only
math_labels.py  math in labels via matplotlib's mathtext (outlines, raster, plain text)
intersect.py  where lines meet shapes (shape.intersect)
colors.py   CSS colors and hex codes as RGB, without dependencies
frames.py   kinematic tree: fixed, prismatic and revolute frames that carry geometry
groups.py   geometries moved together
transform.py  4x4 affine transforms, split_rotation
anchors.py  TikZ-style named points
scene.py    items + styles + labels + frames -> primitives; render/save/animate
render.py   the four primitives, geometry -> primitives
camera.py, depth.py   shared camera; painter's order for 2D backends
flat.py     the shared 2D steps: hide, project, sort, shade (SVG, TikZ)
exact.py    exact spheres and cylinders as images of unit shapes (ray tracer, Mitsuba)
measure.py  pickable anchors and measurements, for interactive backends
backends/   svg, tikz, matplotlib, plotly, raytrace (+ _fastcast: optional numba engine)
```

## Roadmap

Done: 1. foundations test suite (from galgebra's identities); 2. this plan; 6. the concept
manual (`docs/manual`, first version); 3. TikZ backend; 5. looks for the ray tracer;
7. picking and measuring (matplotlib); curves (`Polyline`, `Curve`) and `Mesh` with OBJ/STL
files; 8. math in labels (`math_labels.py`: mathtext outlines for SVG and the ray tracer,
LaTeX for TikZ, Unicode with `<sup>`/`<sub>` for plotly); lines meeting shapes
(`intersect.py`: exact for spheres and cylinders via `exact.unit_map`, faces otherwise);
`Axes` with ticks and `Grid` (`geometry/axes.py`); `Scene.remove` and layers (`hide`, `show`,
`clear`; backends draw `scene.visible(...)` only).

Suggested order for the next steps, with what each one needs first:

| step | what | needs first | notes |
|---|---|---|---|
| 6 | concept manual, code next to output (like the PGF manual) | done | `minigeo-example` blocks in `docs/manual/*.rst` run at build time (`docs/_ext/minigeo_manual.py`) and in `tests/test_manual.py`. Next: a chapter per new feature, and golden-image checks if pictures should be guarded too. |
| 3 | TikZ backend | done | `flat.py` holds the shared 2D steps; `backends/tikz.py` writes plain 2D paths, xcolor mixes for shading, `arrows.meta` tips, LaTeX labels. Next: cel-style shading for SVG/TikZ too. |
| 5 | stylised rendering: cel shading, few colors, ambient occlusion, outlines | done | `Look` and `LOOKS` in the ray tracer. Occlusion uses fixed direction sets (parallel rays), interleaved per pixel and smoothed per surface. |
| 7 | pick points and anchors, measure distances | done (matplotlib) | `measure.py` holds the backend-free part (pickable anchors, `Measurement`); `scene.measure()` uses matplotlib mouse events (ipympl or desktop). Next: plotly, via `FigureWidget` click events or a small JavaScript handler. |
| 4 | Mitsuba 3 backend | done (orthographic) | `backends/mitsuba.py`, extra `mitsuba` (not in `all`). Exact spheres/cylinders (+ disc caps) when not stretched, else fine meshes; faces as meshes per item and style; twosided diffuse, `mask` for opacity; sRGB <-> linear; lights `"sun"` (directional + constant, as the ray tracer) and `"sky"`. Lines/points/labels by the ray tracer's `_Canvas` on Mitsuba's depth AOV (owners from an albedo AOV with one color per shape; `shape_index` was not usable). Default variant `scalar_rgb` (`llvm_ad_rgb` crashed with a hard LLVM error on one machine). Next: perspective camera, studio lights, glass for opacity, animations. |

Shared groundwork: the exact-shape hook is done (`exact.py`, used by the ray tracer and meant
for Mitsuba); the optional numba engine (`backends/_fastcast.py`, extra `fast`) casts rays in
any direction. Still open: a perspective option for `Camera` (for 4 and ray tracing phase 2),
which the numba engine now makes easy. With numba, ambient occlusion already uses random rays
per point (no rings or blotches).

## Backlog

- Painter's backends (SVG, matplotlib) cannot draw intersecting solids or lines piercing
  solids; the ray tracer and plotly can. Document, don't fight.
- Ray tracer phase 2: perspective camera; lines and labels tinted behind translucent faces.
- `Angle` and `Dimension` keep fixed points; to follow moving geometry, attach them to the
  same frame.
- Sizes like `radius` and `size` describe a shape as created, not after scaling.
- Anchor words always mean the local axes (`"top"` is +z′), also with `vertical_axis="x"`.
- Run the example notebooks in CI.
- Add `py.typed`.
- Meshes: no STEP (needs a CAD kernel; users export STL). OBJ colors are written, not read;
  faces are triangulated as fans (fine for convex faces). Curves are sampled once; a
  `samples` change means a new `Curve`.

## Open questions

- Should `t @ shape` return a transformed copy instead of changing the shape in place?
- Should labels get an optional depth test in the SVG backend, as in the ray tracer?
