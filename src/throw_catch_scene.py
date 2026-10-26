"""Isaac Sim scene helpers for a six-arm visual throw/catch relay."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera


DT = 1.0 / 60.0
ARM_COUNT = 6
RING_RADIUS = 0.92
BALL_Z = 1.48
BALL_RADIUS = 0.11
HANDOFF_DWELL_S = 0.24
ACTION_EVERY_STEPS = 6
VIDEO_EVERY_STEPS = 2
ARM_PALETTE = [
    (0.20, 0.47, 0.88),
    (0.16, 0.73, 0.65),
    (0.74, 0.43, 0.86),
    (0.92, 0.53, 0.17),
    (0.83, 0.29, 0.37),
    (0.43, 0.66, 0.24),
]


class Cube:
    def __init__(self, stage, path, color, dimensions, position=(0, 0, 0), yaw=0.0):
        prim = UsdGeom.Cube.Define(stage, Sdf.Path(path))
        prim.CreateSizeAttr(1.0)
        prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        xf = UsdGeom.Xformable(prim.GetPrim())
        self.translate = xf.AddTranslateOp()
        self.rotate = xf.AddRotateZOp()
        self.scale = xf.AddScaleOp()
        self.set(position, yaw, dimensions)

    def set(self, position, yaw=0.0, dimensions=None):
        self.translate.Set(Gf.Vec3d(*[float(x) for x in position]))
        self.rotate.Set(float(math.degrees(yaw)))
        if dimensions is not None:
            self.scale.Set(Gf.Vec3f(*[float(x) for x in dimensions]))


class Sphere:
    def __init__(self, stage, path, color, radius, position=(0, 0, 0), emissive=False):
        prim = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
        prim.CreateRadiusAttr(float(radius))
        prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        self.prim = prim.GetPrim()
        self.translate = UsdGeom.Xformable(self.prim).AddTranslateOp()
        self.material_shader = None
        if emissive:
            material_path = Sdf.Path(path + "Material")
            material = UsdShade.Material.Define(stage, material_path)
            shader = UsdShade.Shader.Define(stage, material_path.AppendChild("PreviewSurface"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(2.0 * float(x) for x in color)))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI(self.prim).Bind(material)
            self.material_shader = shader
        self.set(position)

    def set(self, position):
        self.translate.Set(Gf.Vec3d(*[float(x) for x in position]))

    def set_color(self, color):
        UsdGeom.Gprim(self.prim).GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        if self.material_shader is not None:
            self.material_shader.GetInput("diffuseColor").Set(Gf.Vec3f(*color))
            self.material_shader.GetInput("emissiveColor").Set(Gf.Vec3f(*(2.0 * float(x) for x in color)))


class Cylinder:
    def __init__(self, stage, path, color, radius, height, position):
        prim = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
        prim.CreateRadiusAttr(float(radius))
        prim.CreateHeightAttr(float(height))
        prim.CreateAxisAttr(UsdGeom.Tokens.z)
        prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        self.translate = UsdGeom.Xformable(prim.GetPrim()).AddTranslateOp()
        self.set(position)

    def set(self, position):
        self.translate.Set(Gf.Vec3d(*[float(x) for x in position]))


def station_xy(index: int, radius: float = RING_RADIUS) -> np.ndarray:
    angle = 2.0 * math.pi * (index % ARM_COUNT) / ARM_COUNT
    return np.array([radius * math.cos(angle), radius * math.sin(angle)], dtype=np.float32)


def _link(stage, path, color, start, end, width=0.16, height=0.14):
    a = np.asarray(start, dtype=np.float32)
    b = np.asarray(end, dtype=np.float32)
    d = b - a
    length = float(np.linalg.norm(d[:2]))
    yaw = math.atan2(float(d[1]), float(d[0]))
    center = (a + b) * 0.5
    return Cube(stage, path, color, (length, width, height), center, yaw)


def build_relay_scene():
    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    stage = world.stage
    world.scene.add_default_ground_plane()

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(1500.0)
    dome.CreateColorAttr(Gf.Vec3f(0.88, 0.91, 1.0))

    # Low circular platform makes the ring arrangement easy to read on camera.
    platform = Cylinder(stage, "/World/RelayPlatform", (0.15, 0.19, 0.25), 2.55, 0.18, (0, 0, 0.09))
    del platform

    arms = []
    for i in range(ARM_COUNT):
        theta = 2.0 * math.pi * i / ARM_COUNT
        radial = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        tangent = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float32)
        color = ARM_PALETTE[i]
        base_xy = 2.08 * radial
        shoulder_xy = 1.88 * radial
        elbow_xy = 1.38 * radial
        wrist_xy = station_xy(i)
        base = Cylinder(stage, f"/World/Arm{i}/Base", (0.20, 0.23, 0.29), 0.20, 0.48, (base_xy[0], base_xy[1], 0.24))
        shoulder = Sphere(stage, f"/World/Arm{i}/Shoulder", (0.78, 0.81, 0.86), 0.15, (shoulder_xy[0], shoulder_xy[1], 1.34))
        elbow = Sphere(stage, f"/World/Arm{i}/Elbow", (0.78, 0.81, 0.86), 0.12, (elbow_xy[0], elbow_xy[1], 1.34))
        upper = _link(stage, f"/World/Arm{i}/UpperLink", color, (shoulder_xy[0], shoulder_xy[1], 1.34), (elbow_xy[0], elbow_xy[1], 1.34), 0.18, 0.18)
        forearm = _link(stage, f"/World/Arm{i}/Forearm", (0.78, 0.81, 0.86), (elbow_xy[0], elbow_xy[1], 1.34), (wrist_xy[0], wrist_xy[1], BALL_Z), 0.13, 0.12)
        hand = Sphere(stage, f"/World/Arm{i}/Hand", (0.12, 0.14, 0.18), 0.15, (wrist_xy[0], wrist_xy[1], BALL_Z))
        # Two gripper fingers sit tangent to the ring and frame the ball.
        fingers = []
        for sign in (-1.0, 1.0):
            p = np.array([wrist_xy[0], wrist_xy[1], BALL_Z], dtype=np.float32)
            p[:2] += tangent * sign * 0.105
            finger_name = "FingerL" if sign < 0 else "FingerR"
            fingers.append(Cube(stage, f"/World/Arm{i}/{finger_name}", color, (0.23, 0.045, 0.055), p, theta + math.pi / 2))
        # Oversized, saturated status beacons remain legible after the camera image
        # is compressed to the policy's compact 224x224 input.
        led_xy = elbow_xy + tangent * 0.24
        led = Sphere(stage, f"/World/Arm{i}/StatusLED", (0.05, 0.06, 0.08), 0.14, (led_xy[0], led_xy[1], 1.80), emissive=True)
        arms.append({"base": base, "shoulder": shoulder, "elbow": elbow, "upper": upper, "forearm": forearm, "hand": hand, "fingers": fingers, "led": led, "theta": theta, "tangent": tangent, "radial": radial, "color": color, "index": i, "shoulder_xy": shoulder_xy, "elbow_xy": elbow_xy, "wrist_xy": wrist_xy})

    ball = Sphere(stage, "/World/RelayBall", (0.97, 0.025, 0.035), BALL_RADIUS)

    camera = Camera(
        prim_path="/World/RelayCamera",
        name="relay_camera",
        position=np.array([0.0, 0.0, 7.1], dtype=np.float32),
        orientation=np.array([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], dtype=np.float32),
        frequency=30,
        resolution=(720, 720),
    )
    camera.set_focal_length(1.4)

    world.reset()
    camera.initialize()
    for _ in range(30):
        world.step(render=True)
    return world, camera, arms, ball


def set_arm_leds(arms, receiving=None, holding=False, ready=False):
    inactive = (0.025, 0.035, 0.055)
    for arm in arms:
        i = arm["index"]
        if i != receiving:
            color = inactive
        elif ready:
            color = (0.0, 0.82, 1.0)
        elif holding:
            color = (0.04, 1.0, 0.10)
        else:
            color = (1.0, 0.28, 0.015)
        arm["led"].set_color(color)


def pose_arm(arm, phase="idle", amount=0.0):
    """Animate one stylized arm through a short wind-up, extension, or catch reach."""
    amount = float(np.clip(amount, 0.0, 1.0))
    index = arm["index"]
    radial = arm["radial"]
    tangent = arm["tangent"]
    shoulder = np.array([*(1.88 * radial), 1.34], dtype=np.float32)
    elbow = np.array([*(1.38 * radial), 1.34], dtype=np.float32)
    wrist = np.array([*station_xy(index), BALL_Z], dtype=np.float32)
    direction = station_xy(index + 1) - station_xy(index)
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    if phase == "throw":
        wrist[:2] += direction * ((amount - 0.5) * 0.34)
        wrist[2] += 0.10 * math.sin(math.pi * amount)
        elbow[:2] -= tangent * (0.06 * (1.0 - amount))
    elif phase == "catch":
        wrist[:2] += direction * (0.16 * amount)
        wrist[2] += 0.08 * math.sin(math.pi * amount)
        elbow[:2] += direction * (0.04 * amount)

    elbow[2] += 0.04 * math.sin(math.pi * amount) if phase in ("throw", "catch") else 0.0
    def update_link(link, start, end, width, height):
        delta = end - start
        yaw = math.atan2(float(delta[1]), float(delta[0]))
        center = (start + end) * 0.5
        link.set(center, yaw, (max(float(np.linalg.norm(delta[:2])), 0.02), width, height))

    update_link(arm["upper"], shoulder, elbow, 0.18, 0.18)
    update_link(arm["forearm"], elbow, wrist, 0.13, 0.12)
    arm["elbow"].set(elbow)
    arm["hand"].set(wrist)
    for sign, finger in zip((-1.0, 1.0), arm["fingers"]):
        finger_xy = wrist.copy()
        finger_xy[:2] += tangent * sign * 0.105
        finger.set(finger_xy, arm["theta"] + math.pi / 2, (0.23, 0.045, 0.055))
    arm["led"].set([float(elbow[0] + tangent[0] * 0.24), float(elbow[1] + tangent[1] * 0.24), float(elbow[2] + 0.46)])


def flight_position(source: int, target: int, t: float, peak_m: float):
    t = float(np.clip(t, 0.0, 1.0))
    a = station_xy(source)
    b = station_xy(target)
    xy = (1.0 - t) * a + t * b
    z = BALL_Z + 4.0 * peak_m * t * (1.0 - t)
    return np.array([xy[0], xy[1], z], dtype=np.float32)


def camera_rgb(camera):
    rgb = camera.get_rgb(device="cpu")
    if rgb is None:
        return None
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim == 3 and rgb.shape[2] > 3:
        rgb = rgb[:, :, :3]
    return rgb


def write_frame_png(rgb, path: Path):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(path)
