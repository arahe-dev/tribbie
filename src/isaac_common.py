"""Shared Isaac Sim scene helpers. Import this only after SimulationApp starts."""

import math

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path


DT = 1.0 / 60.0
ROBOT_START_Z = 0.034
TARGET_HEIGHT = 0.90
FRONT_CAMERA_RESOLUTION = (320, 240)
CAMERA_MOUNT = np.array([0.22, 0.0, 0.42], dtype=np.float32)
CAMERA_PITCH_DOWN_RAD = 0.0
STOP_DISTANCE_M = 1.50
WHEEL_RADIUS = 0.03
WHEEL_BASE = 0.1125


def yaw_quaternion(yaw):
    """Isaac Sim uses scalar-first (w, x, y, z) quaternions."""
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float32)


def multiply_quaternions(left, right):
    """Multiply scalar-first quaternions (right-hand local camera pitch)."""
    aw, ax, ay, az = [float(item) for item in left]
    bw, bx, by, bz = [float(item) for item in right]
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def yaw_from_quaternion(q):
    return 2.0 * math.atan2(float(q[3]), float(q[0]))


def rotate_vector(q, v):
    w, x, y, z = [float(item) for item in q]
    vx, vy, vz = [float(item) for item in v]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array(
        [vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)],
        dtype=np.float32,
    )


class _MutableVisualMaterial:
    def __init__(self, display_color_attr):
        self.display_color_attr = display_color_attr

    def set_color(self, color):
        rgb = np.asarray(color, dtype=np.float32).reshape(3)
        self.display_color_attr.Set([Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))])


class _MutableEmissionMaterial:
    def __init__(self, color_input):
        self.color_input = color_input

    def set_color(self, color):
        rgb = np.asarray(color, dtype=np.float32).reshape(3)
        self.color_input.Set(Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2])))


class StaticStopPole:
    """A fixed-shape, kinematic red post used by the reactive-stop experiment."""

    def __init__(self, stage, prim_path="/World/StopPole", color=(1.0, 0.08, 0.04)):
        cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(prim_path))
        cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
        cylinder.CreateRadiusAttr(0.09)
        cylinder.CreateHeightAttr(TARGET_HEIGHT)
        self.prim = cylinder.GetPrim()
        self.display_color_attr = cylinder.CreateDisplayColorAttr(
            [Gf.Vec3f(*[float(value) for value in color])]
        )
        xform = UsdGeom.Xformable(self.prim)
        self.translate_op = xform.AddTranslateOp()
        self.rotate_op = xform.AddRotateZOp()
        UsdPhysics.CollisionAPI.Apply(self.prim)
        rigid = UsdPhysics.RigidBodyAPI.Apply(self.prim)
        rigid.CreateRigidBodyEnabledAttr(True)
        rigid.CreateKinematicEnabledAttr(True)

        self.material = _MutableVisualMaterial(self.display_color_attr)
        self.position = np.array([0.0, 0.0, TARGET_HEIGHT / 2.0], dtype=np.float32)
        self.yaw = 0.0
        self.set_world_pose(self.position)

    def set_world_pose(self, position, orientation=None):
        self.position = np.asarray(position, dtype=np.float32).reshape(3).copy()
        self.translate_op.Set(Gf.Vec3d(*[float(value) for value in self.position]))
        if orientation is not None:
            self.yaw = yaw_from_quaternion(orientation)
            self.rotate_op.Set(float(math.degrees(self.yaw)))

    def get_world_pose(self):
        return self.position.copy(), yaw_quaternion(self.yaw)


class _StaticBox:
    """A self-lit cuboid with explicit USD transforms and no gravity body."""

    def __init__(self, stage, prim_path, color, position):
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(prim_path))
        cube.CreateSizeAttr(1.0)
        self.prim = cube.GetPrim()
        material = UsdShade.Material.Define(stage, Sdf.Path(f"{prim_path}Material"))
        shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{prim_path}Material/Surface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        self.emissive_input = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
        self.emissive_input.Set(Gf.Vec3f(*[float(value) for value in color]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(self.prim).Bind(material)
        xform = UsdGeom.Xformable(self.prim)
        self.translate_op = xform.AddTranslateOp()
        self.scale_op = xform.AddScaleOp()
        self.scale_op.Set(Gf.Vec3f(0.18, 0.18, float(TARGET_HEIGHT)))
        self.material = _MutableEmissionMaterial(self.emissive_input)
        self.position = np.zeros(3, dtype=np.float32)
        self.yaw = 0.0
        self.set_world_pose(position)

    def set_world_pose(self, position, orientation=None):
        self.position = np.asarray(position, dtype=np.float32).reshape(3).copy()
        self.translate_op.Set(Gf.Vec3d(*[float(value) for value in self.position]))
        if orientation is not None:
            self.yaw = yaw_from_quaternion(orientation)

    def get_world_pose(self):
        return self.position.copy(), yaw_quaternion(self.yaw)

    def get_applied_visual_material(self):
        return self.material


def build_scene(with_overhead=False, static_stop_pole=False):
    """Create a simple lit world, NVIDIA Jetbot, red target, and RGB camera(s)."""
    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Isaac Sim did not return an asset root")

    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    world.scene.add_default_ground_plane()
    dome = UsdLux.DomeLight.Define(world.stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(1200.0)

    robot = world.scene.add(
        WheeledRobot(
            prim_path="/World/Jetbot",
            name="jetbot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            usd_path=assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
            position=np.array([0.0, 0.0, ROBOT_START_Z], dtype=np.float32),
        )
    )
    target = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Target",
            name="target",
            position=np.array([1.8, 0.0, TARGET_HEIGHT / 2.0], dtype=np.float32),
            scale=np.array([0.18, 0.18, TARGET_HEIGHT], dtype=np.float32),
            size=1.0,
            color=np.array([1.0, 0.08, 0.04], dtype=np.float32),
        )
    )
    if static_stop_pole:
        hidden_target = world.stage.GetPrimAtPath("/World/Target")
        UsdGeom.Imageable(hidden_target).MakeInvisible()
        collision = UsdPhysics.CollisionAPI.Get(world.stage, hidden_target.GetPath())
        if collision:
            collision.CreateCollisionEnabledAttr(False)
        # Keep the marker gravity-free while preserving the fixed cuboid shape.
        target = _StaticBox(
            world.stage,
            "/World/StopPole",
            color=(0.08, 0.80, 0.08),
            position=np.array([1.8, 0.0, TARGET_HEIGHT / 2.0], dtype=np.float32),
        )
    front_camera = Camera(
        prim_path="/World/RobotCamera",
        name="robot_camera",
        position=np.array([CAMERA_MOUNT[0], CAMERA_MOUNT[1], CAMERA_MOUNT[2]], dtype=np.float32),
        orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        frequency=30,
        resolution=FRONT_CAMERA_RESOLUTION,
    )
    # USD stores lens values in tenths of the stage unit; with a 1 m stage,
    # 0.8 here corresponds to an 8 mm wide-angle lens.
    front_camera.set_focal_length(0.8)
    overhead_camera = None
    if with_overhead:
        overhead_camera = Camera(
            prim_path="/World/OverheadCamera",
            name="overhead_camera",
            position=np.array([0.85, 0.0, 3.0], dtype=np.float32),
            orientation=np.array([math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0], dtype=np.float32),
            frequency=30,
            resolution=FRONT_CAMERA_RESOLUTION,
        )
        overhead_camera.set_focal_length(0.8)

    world.reset()
    front_camera.initialize()
    if overhead_camera is not None:
        overhead_camera.initialize()
    controller = DifferentialController(name="jetbot_differential", wheel_radius=WHEEL_RADIUS, wheel_base=WHEEL_BASE)
    update_front_camera(front_camera, robot)
    return world, robot, target, front_camera, overhead_camera, controller


def update_front_camera(camera, robot):
    position, orientation = robot.get_world_pose()
    offset_world = rotate_vector(orientation, CAMERA_MOUNT)
    pitch = np.array(
        [math.cos(CAMERA_PITCH_DOWN_RAD / 2.0), 0.0, math.sin(CAMERA_PITCH_DOWN_RAD / 2.0), 0.0],
        dtype=np.float32,
    )
    camera.set_world_pose(
        position=np.asarray(position, dtype=np.float32) + offset_world,
        orientation=multiply_quaternions(orientation, pitch),
        camera_axes="world",
    )


def place_target_relative(target, robot_position, robot_yaw, bearing, distance):
    global_bearing = robot_yaw + bearing
    target_xy = np.array(
        [robot_position[0] + distance * math.cos(global_bearing), robot_position[1] + distance * math.sin(global_bearing)],
        dtype=np.float32,
    )
    target.set_world_pose(position=np.array([target_xy[0], target_xy[1], TARGET_HEIGHT / 2.0], dtype=np.float32))
    return target_xy


def target_relative_state(robot, target):
    robot_position, robot_orientation = robot.get_world_pose()
    target_position, _ = target.get_world_pose()
    dx = float(target_position[0] - robot_position[0])
    dy = float(target_position[1] - robot_position[1])
    yaw = yaw_from_quaternion(robot_orientation)
    rel_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    rel_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return math.hypot(dx, dy), math.atan2(rel_y, rel_x), yaw


def teacher_action(distance, bearing, turn_threshold=0.16, stop_distance=STOP_DISTANCE_M):
    if distance <= stop_distance and abs(bearing) <= turn_threshold:
        return "STOP"
    if bearing > turn_threshold:
        return "LEFT"
    if bearing < -turn_threshold:
        return "RIGHT"
    return "FORWARD"


def action_command(action):
    return {
        "LEFT": (0.0, 0.72),
        "FORWARD": (0.18, 0.0),
        "RIGHT": (0.0, -0.72),
        "STOP": (0.0, 0.0),
    }[action]


def camera_rgb(camera):
    frame = camera.get_rgb(device="cpu")
    if frame is None:
        return None
    return np.asarray(frame, dtype=np.uint8)
