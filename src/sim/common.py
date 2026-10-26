"""Shared Isaac Sim helpers for the learned-manipulation demos.

Import only after ``SimulationApp`` has started. Quaternions are scalar-first
(w, x, y, z), matching Isaac Sim.
"""

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float32)


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = [float(v) for v in a]
    bw, bx, by, bz = [float(v) for v in b]
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float32)


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return (q / np.linalg.norm(q)).astype(np.float32)


def yaw_of_quat(q) -> float:
    """Heading of the body x-axis projected onto the ground plane."""
    m = quat_to_matrix(q)
    return math.atan2(m[1, 0], m[0, 0])


def down_hand_quat(yaw: float) -> np.ndarray:
    """Panda hand pointing straight down (+z along world -z), rotated by yaw.

    With R = Rz(yaw) * Rx(pi) the finger closing axis (hand +y) lies along
    world angle ``yaw - pi/2``.
    """
    return quat_mul(yaw_quat(yaw), np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))


def look_at_quat(position, target, up_hint=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Isaac Camera 'world' convention: +X forward, +Y left, +Z up."""
    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1e-9)
    up_hint = np.asarray(up_hint, dtype=np.float64)
    if abs(float(np.dot(forward, up_hint))) > 0.97:
        up_hint = np.array([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    left = np.cross(up_hint, forward)
    left /= max(np.linalg.norm(left), 1e-9)
    up = np.cross(forward, left)
    return matrix_to_quat(np.column_stack([forward, left, up]))


def make_material(stage, path: str, color, roughness=0.5, metallic=0.0, emissive=None):
    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(path + "/Surface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in color]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in emissive]))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def set_material_color(stage, material_path: str, color):
    shader = UsdShade.Shader.Get(stage, Sdf.Path(material_path + "/Surface"))
    shader.GetInput("diffuseColor").Set(Gf.Vec3f(*[float(c) for c in color]))


def bind(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def physics_material(stage, path: str, static=1.0, dynamic=0.9, restitution=0.0):
    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(static))
    api.CreateDynamicFrictionAttr(float(dynamic))
    api.CreateRestitutionAttr(float(restitution))
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.CreateFrictionCombineModeAttr("max")
    return material


def bind_physics(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def static_box(stage, path, center, size, material=None, collide=True, yaw=0.0):
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw:
        xf.AddRotateZOp().Set(math.degrees(float(yaw)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if material is not None:
        bind(cube.GetPrim(), material)
    return cube.GetPrim()


def static_cylinder(stage, path, center, radius, height, material=None, collide=True):
    cyl = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    cyl.CreateAxisAttr(UsdGeom.Tokens.z)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if collide:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    if material is not None:
        bind(cyl.GetPrim(), material)
    return cyl.GetPrim()


def studio_lights(stage, dome=300.0, key=1400.0, rim=450.0, key_rot=(-48.0, -14.0, -34.0)):
    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Lights/Dome"))
    dome_light.CreateIntensityAttr(float(dome))
    dome_light.CreateColorAttr(Gf.Vec3f(0.70, 0.78, 0.92))
    key_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lights/Key"))
    key_light.CreateIntensityAttr(float(key))
    key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.86, 0.70))
    key_light.CreateAngleAttr(2.0)
    UsdGeom.Xformable(key_light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(*key_rot))
    rim_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lights/Rim"))
    rim_light.CreateIntensityAttr(float(rim))
    rim_light.CreateColorAttr(Gf.Vec3f(0.50, 0.70, 1.0))
    UsdGeom.Xformable(rim_light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(20.0, -30.0, 140.0))
    return dome_light, key_light, rim_light


def configure_physics_scene(world, gpu=True):
    stage = world.stage
    context = world.get_physics_context()
    prim = stage.GetPrimAtPath(context.prim_path)
    if prim and prim.IsValid():
        scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
        scene.CreateEnableCCDAttr(True)
        scene.CreateEnableStabilizationAttr(True)
        scene.CreateSolverTypeAttr("TGS")
    if hasattr(context, "enable_gpu_dynamics"):
        context.enable_gpu_dynamics(False)
