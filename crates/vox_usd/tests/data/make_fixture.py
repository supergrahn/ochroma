#!/usr/bin/env python3
"""Regenerate the vox_usd test fixtures with real Pixar USD.

This is the source of truth for every committed fixture in this directory.
The committed `.usdc`/`.usda` files MUST match what this script produces, so
that the pure-Rust openusd-rs importer is tested against genuine Pixar-authored
crate (binary) and text scenes — never hand-faked bytes.

Regenerate the python env + run:

    uv venv /tmp/usdenv && uv pip install --python /tmp/usdenv/bin/python usd-core
    /tmp/usdenv/bin/python crates/vox_usd/tests/data/make_fixture.py

(usd-core 26.5 was used to author the committed fixtures.)

Fixtures produced (all in this directory):
  cube_lit.usdc       — 2 m cube Mesh (triangulated) + SphereLight + Camera. The Done-When scene.
  instancer.usdc      — PointInstancer with 3 instances at known positions (3DGS path).
  red_cube.usdc       — unit cube Mesh bound to a red (1,0,0) UsdPreviewSurface material (color->spectrum).
  points_text.usda    — point3f[] geometry in TEXT form (negative case -> UnsupportedTextArray).
"""

import os
from pxr import Usd, UsdGeom, UsdLux, UsdShade, Sdf, Gf

HERE = os.path.dirname(os.path.abspath(__file__))


def cube_points(half):
    """8 corners of an axis-aligned cube of half-extent `half`, centered at origin."""
    return [
        Gf.Vec3f(-half, -half, -half),
        Gf.Vec3f( half, -half, -half),
        Gf.Vec3f( half,  half, -half),
        Gf.Vec3f(-half,  half, -half),
        Gf.Vec3f(-half, -half,  half),
        Gf.Vec3f( half, -half,  half),
        Gf.Vec3f( half,  half,  half),
        Gf.Vec3f(-half,  half,  half),
    ]


# Quad faces (CCW), shared by every cube. 6 faces, 4 verts each.
CUBE_FACE_COUNTS = [4, 4, 4, 4, 4, 4]
CUBE_FACE_INDICES = [
    0, 1, 2, 3,   # -Z
    4, 5, 6, 7,   # +Z
    0, 1, 5, 4,   # -Y
    2, 3, 7, 6,   # +Y
    0, 3, 7, 4,   # -X
    1, 2, 6, 5,   # +X
]


def author_mesh(stage, path, half):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(cube_points(half))
    mesh.CreateFaceVertexCountsAttr(CUBE_FACE_COUNTS)
    mesh.CreateFaceVertexIndicesAttr(CUBE_FACE_INDICES)
    return mesh


def make_cube_lit():
    path = os.path.join(HERE, "cube_lit.usdc")
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # 2 m cube => half-extent 1.0, corners at +/-1 on every axis.
    author_mesh(stage, "/Cube", 1.0)

    # SphereLight: intensity 1000, white.
    light = UsdLux.SphereLight.Define(stage, "/Light")
    light.CreateIntensityAttr(1000.0)
    light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    # Camera: focalLength 50, horizontalAperture 36, verticalAperture chosen so
    # fovY is computable. world pos (0,1,6). 2*atan(verticalAperture/(2*focal)).
    cam = UsdGeom.Camera.Define(stage, "/Camera")
    cam.CreateFocalLengthAttr(50.0)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(20.25)  # 16:9-ish; fovY = 2*atan(20.25/100)
    UsdGeom.XformCommonAPI(cam).SetTranslate(Gf.Vec3d(0.0, 1.0, 6.0))

    stage.GetRootLayer().Save()
    print("wrote", path)


def make_instancer():
    path = os.path.join(HERE, "instancer.usdc")
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    inst = UsdGeom.PointInstancer.Define(stage, "/Instancer")
    positions = [
        Gf.Vec3f(1.0, 2.0, 3.0),
        Gf.Vec3f(-4.0, 5.0, -6.0),
        Gf.Vec3f(7.0, -8.0, 9.0),
    ]
    inst.CreatePositionsAttr(positions)
    inst.CreateProtoIndicesAttr([0, 0, 0])
    inst.CreateScalesAttr([Gf.Vec3f(0.5, 0.5, 0.5)] * 3)
    inst.CreateOrientationsAttr([Gf.Quath(1, 0, 0, 0)] * 3)

    # A prototype prim (required for a valid PointInstancer; not sampled by us).
    proto = UsdGeom.Scope.Define(stage, "/Instancer/Prototypes")
    author_mesh(stage, "/Instancer/Prototypes/Cube", 0.5)
    inst.CreatePrototypesRel().SetTargets([Sdf.Path("/Instancer/Prototypes/Cube")])

    stage.GetRootLayer().Save()
    print("wrote", path)


def make_red_cube():
    path = os.path.join(HERE, "red_cube.usdc")
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    mesh = author_mesh(stage, "/RedCube", 0.5)  # unit cube

    mat = UsdShade.Material.Define(stage, "/RedCube/Mat")
    shader = UsdShade.Shader.Define(stage, "/RedCube/Mat/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 0.0, 0.0)
    )
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)

    stage.GetRootLayer().Save()
    print("wrote", path)


def make_points_text():
    """TEXT (.usda) scene whose only geometry is a point3f[] array.

    openusd-rs's USDA parser cannot read arrays/tuples, so this scene's geometry
    is invisible to it — the importer must surface UnsupportedTextArray rather
    than silently importing an empty scene.
    """
    path = os.path.join(HERE, "points_text.usda")
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    pts = UsdGeom.Points.Define(stage, "/Points")
    pts.CreatePointsAttr([
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(1.0, 0.0, 0.0),
        Gf.Vec3f(0.0, 1.0, 0.0),
    ])
    pts.CreateWidthsAttr([0.1, 0.1, 0.1])

    stage.GetRootLayer().Save()
    print("wrote", path)


if __name__ == "__main__":
    make_cube_lit()
    make_instancer()
    make_red_cube()
    make_points_text()
