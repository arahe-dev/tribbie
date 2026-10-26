"""Physics-driven six-arm relay using NVIDIA's articulated Franka Panda model.

At release the ball receives a launch impulse expressed as a rigid-body
velocity change. PhysX integrates gravity, free flight, collision, and contact.
A catch is recorded only after a finger contact sensor reports the ball, then
a fixed grasp joint models the closed parallel gripper holding it.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import time
import traceback

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka, KinematicsSolver
from isaacsim.sensors.camera import Camera
from isaacsim.sensors.physics import ContactSensor


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
INFERENCE_SERVER = ROOT / "src" / "throw_catch_inference_server.py"
FRAMES = ROOT / "artifacts" / "throw_catch_physics_6arm_frames"
RESULT_JSON = ROOT / "results" / "throw_catch_physics_6arm.json"
CONTACT_CSV = ROOT / "results" / "throw_catch_physics_contacts.csv"

ARM_COUNT = 6
DT = 1.0 / 120.0
RENDER_EVERY = 4
VIDEO_FPS = 30
BALL_RADIUS = 0.033
BALL_MASS_KG = 0.060
RELEASE_RADIUS = 0.70
RELEASE_Z = 0.60
APEX_TIME = 0.20
APEX_RISE = 0.5 * 9.81 * APEX_TIME**2
FLIGHT_TIME = 0.95
ROOT_RADIUS = 1.25
BASE_Z = 0.18
GRIP_OPEN = 0.040
GRIP_HOLD = 0.025
GRIP_CLOSED = 0.0


class InferenceClient:
    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.buffer = bytearray()

    def decide(self, rgb: np.ndarray) -> dict:
        image = Image.fromarray(rgb)
        fitted = Image.new("RGB", (720, 720), (12, 17, 24))
        resized = image.resize((720, 405), Image.Resampling.LANCZOS)
        fitted.paste(resized, (0, 157))
        payload = np.ascontiguousarray(np.asarray(fitted, dtype=np.uint8)).tobytes()
        started = time.perf_counter_ns()
        self.connection.sendall(struct.pack("!I", len(payload)) + payload)
        while b"\n" not in self.buffer:
            block = self.connection.recv(4096)
            if not block:
                raise ConnectionError("The relay MoE inference worker closed its connection")
            self.buffer.extend(block)
        line, _, tail = self.buffer.partition(b"\n")
        self.buffer = bytearray(tail)
        result = json.loads(line.decode("utf-8"))
        result["latency_ms"]["client_round_trip"] = (time.perf_counter_ns() - started) / 1e6
        return result


def start_inference_worker(device: str):
    if not VENV_PYTHON.is_file():
        raise FileNotFoundError(f"Missing project Python environment: {VENV_PYTHON}")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "CARB_APP_PATH", "ISAAC_PATH", "EXP_PATH"):
        env.pop(key, None)
    process = subprocess.Popen(
        [str(VENV_PYTHON), "-u", str(INFERENCE_SERVER), "--port", "0", "--device", device],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        deadline = time.monotonic() + 120.0
        ready = ""
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if line:
                ready = line.strip()
                print(f"INFERENCE_WORKER {ready}", flush=True)
                if ready.startswith("READY "):
                    break
            elif process.poll() is not None:
                raise RuntimeError(f"Relay inference worker exited with code {process.returncode}")
        if not ready.startswith("READY "):
            raise TimeoutError("Relay inference worker did not become ready")
        connection = socket.create_connection(("127.0.0.1", int(ready.split()[1])), timeout=30.0)
        connection.settimeout(30.0)
        return process, InferenceClient(connection)
    except Exception:
        process.terminate()
        process.wait(timeout=10)
        raise


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Return scalar-first (w,x,y,z) quaternion for a 3x3 local-to-world basis."""
    m = np.asarray(matrix, dtype=np.float64)
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
    q /= np.linalg.norm(q)
    return q.astype(np.float32)


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def camera_orientation(position, target) -> np.ndarray:
    """Isaac camera convention: local +X forward, +Y left, +Z up."""
    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1e-9)
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, up_hint))) > 0.97:
        up_hint = np.array([0.0, 1.0, 0.0])
    left = np.cross(up_hint, forward)
    left /= max(np.linalg.norm(left), 1e-9)
    up = np.cross(forward, left)
    return matrix_to_quaternion(np.column_stack([forward, left, up]))


def tool_orientation(direction_xy) -> np.ndarray:
    """Make Franka fingers point down, with their opening orthogonal to flight."""
    forward = np.array([float(direction_xy[0]), float(direction_xy[1]), 0.0], dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1e-9)
    down = np.array([0.0, 0.0, -1.0])
    finger_axis = np.cross(down, forward)
    finger_axis /= max(np.linalg.norm(finger_axis), 1e-9)
    return matrix_to_quaternion(np.column_stack([forward, finger_axis, down]))


def tool_orientation_axis(direction, roll: float = 0.0) -> np.ndarray:
    """Orient the Panda's finger axis along a 3-D incoming/outgoing trajectory."""
    z_axis = np.asarray(direction, dtype=np.float64).reshape(3)
    z_axis /= max(np.linalg.norm(z_axis), 1e-9)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(z_axis, reference))) > 0.94:
        reference = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(reference, z_axis)
    x_axis /= max(np.linalg.norm(x_axis), 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    c, s = math.cos(float(roll)), math.sin(float(roll))
    rolled_x = c * x_axis + s * y_axis
    rolled_y = -s * x_axis + c * y_axis
    return matrix_to_quaternion(np.column_stack([rolled_x, rolled_y, z_axis]))


def smooth5(value: float) -> float:
    t = float(np.clip(value, 0.0, 1.0))
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def create_material(stage, path: str, color, roughness=0.55, metallic=0.0):
    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(path + "/Surface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_material(prim, material):
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def static_box(stage, path, center, size, material, collide=True):
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.15, 0.18, 0.22)])
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    bind_material(cube.GetPrim(), material)
    return cube.GetPrim()


def static_cylinder(stage, path, center, radius, height, material, collide=False):
    cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(height))
    UsdGeom.Xformable(cylinder.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if collide:
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    bind_material(cylinder.GetPrim(), material)
    return cylinder.GetPrim()


def set_drive_target(robot, joint_positions: np.ndarray):
    robot.apply_action(ArticulationAction(joint_positions=np.asarray(joint_positions, dtype=np.float32)))


def grasp_joint(stage, path: str, hand_path: str, ball_path: str, ball_position, ball_orientation):
    hand_prim = stage.GetPrimAtPath(hand_path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    hand_world = cache.GetLocalToWorldTransform(hand_prim)
    hand_rotation = hand_world.ExtractRotationQuat()
    ball_quat = np.asarray(ball_orientation, dtype=np.float64).reshape(4)
    ball_rotation = Gf.Quatd(float(ball_quat[0]), Gf.Vec3d(*[float(v) for v in ball_quat[1:]]))
    hand_quat = np.array([hand_rotation.GetReal(), *hand_rotation.GetImaginary()], dtype=np.float64)
    hand_quat /= np.linalg.norm(hand_quat)
    inverse_quat = hand_quat.copy()
    inverse_quat[1:] *= -1.0
    world_delta = np.asarray(ball_position, dtype=np.float64).reshape(3) - np.asarray(
        hand_world.ExtractTranslation(), dtype=np.float64
    )
    qv = inverse_quat[1:]
    uv = np.cross(qv, world_delta)
    uuv = np.cross(qv, uv)
    local_position = world_delta + 2.0 * (inverse_quat[0] * uv + uuv)
    local_rotation = hand_rotation.GetInverse() * ball_rotation
    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(path))
    joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(ball_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(local_position[0]), float(local_position[1]), float(local_position[2])))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(local_rotation.GetReal()), Gf.Vec3f(*[float(v) for v in local_rotation.GetImaginary()])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    return joint


class CameraRig:
    def __init__(self, camera: Camera):
        self.camera = camera
        self.shots = [
            ("HERO", np.array([-3.75, -4.15, 3.05]), np.array([0.0, 0.0, 0.80])),
            ("OVERHEAD", np.array([-0.05, -0.08, 6.35]), np.array([0.0, 0.0, 0.68])),
            ("SIDE", np.array([4.30, -3.00, 2.55]), np.array([0.0, 0.0, 0.78])),
            ("REVERSE", np.array([2.95, 3.95, 2.95]), np.array([0.0, 0.0, 0.80])),
        ]
        self.position = self.shots[0][1].copy()
        self.target = self.shots[0][2].copy()
        self.from_position = self.position.copy()
        self.from_target = self.target.copy()
        self.to_position = self.position.copy()
        self.to_target = self.target.copy()
        self.transition_start = 0.0
        self.transition_duration = 0.45
        self.name = self.shots[0][0]

    def choose(self, shot_index: int, current_time: float):
        shot = self.shots[shot_index % len(self.shots)]
        self.from_position = self.position.copy()
        self.from_target = self.target.copy()
        self.to_position = shot[1].copy()
        self.to_target = shot[2].copy()
        self.transition_start = float(current_time)
        self.name = shot[0]

    def update(self, current_time: float):
        amount = smooth5((current_time - self.transition_start) / self.transition_duration)
        self.position = (1.0 - amount) * self.from_position + amount * self.to_position
        self.target = (1.0 - amount) * self.from_target + amount * self.to_target
        self.camera.set_world_pose(
            position=self.position.astype(np.float32),
            orientation=camera_orientation(self.position, self.target),
            camera_axes="world",
        )


class RelaySimulation:
    def __init__(self, transfers: int, device: str, seed: int):
        self.transfers = int(transfers)
        self.device = device
        self.seed = int(seed)
        np.random.seed(seed)
        self.world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=1.0 / VIDEO_FPS)
        self.stage = self.world.stage
        self.physics_step = 0
        self.sim_time = 0.0
        self.frame_index = 0
        self.recording = False
        self.latest_rgb = None
        self.camera_name = "HERO"
        self.transfer_label = 0
        self.status = "INITIALIZING"
        self.policy = {"action": "WAIT", "confidence": 0.0, "latency_ms": {"client_round_trip": 0.0}, "gate_weights": [], "latent": []}
        self.policy_counts = Counter()
        self.policy_rows = []
        self.contact_rows = []
        self.events = []
        self.throw_fallbacks = 0
        self.catch_failures = 0
        self.ball_positions = []

        FRAMES.mkdir(parents=True, exist_ok=True)
        for path in FRAMES.glob("frame_*.png"):
            path.unlink()

        self.robots = []
        self.solvers = []
        self.arms = []
        self.contact_sensors = []
        self.ball_contact_sensor = None
        self.ball_contact_events = []
        self.grasp = None
        self.ball = None
        self.ball_prim = None
        self.robot_collision_paths = []
        self.finger_collision_paths = []
        self.ball_collision_filter = None
        self.camera = None
        self.camera_rig = None
        self.latest_force = 0.0
        self.held_by = 0

        self._build_scene()

    def _build_scene(self):
        stage = self.stage
        stage.SetStartTimeCode(0)
        stage.SetTimeCodesPerSecond(float(VIDEO_FPS))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        dark_floor = create_material(stage, "/World/Materials/Floor", (0.025, 0.035, 0.052), roughness=0.42, metallic=0.18)
        platform_mat = create_material(stage, "/World/Materials/Platform", (0.075, 0.095, 0.13), roughness=0.3, metallic=0.3)
        pedestal_mat = create_material(stage, "/World/Materials/Pedestal", (0.12, 0.15, 0.20), roughness=0.28, metallic=0.45)
        ball_mat = create_material(stage, "/World/Materials/RedBall", (0.78, 0.025, 0.018), roughness=0.24, metallic=0.02)
        static_box(stage, "/World/StudioFloor", (0.0, 0.0, -0.08), (9.0, 9.0, 0.16), dark_floor, collide=True)
        static_cylinder(stage, "/World/Turntable", (0.0, 0.0, 0.055), 1.78, 0.11, platform_mat, collide=True)

        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/StudioDome"))
        dome.CreateIntensityAttr(250.0)
        dome.CreateColorAttr(Gf.Vec3f(0.62, 0.73, 0.92))
        key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
        key.CreateIntensityAttr(1250.0)
        key.CreateColorAttr(Gf.Vec3f(1.0, 0.80, 0.61))
        UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-42.0, -18.0, -31.0))
        rim = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/RimLight"))
        rim.CreateIntensityAttr(420.0)
        rim.CreateColorAttr(Gf.Vec3f(0.42, 0.68, 1.0))
        UsdGeom.Xformable(rim.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(18.0, -32.0, 142.0))

        robot_colors = [
            (0.14, 0.66, 0.92), (0.19, 0.78, 0.63), (0.63, 0.54, 0.92),
            (0.98, 0.57, 0.20), (0.94, 0.30, 0.31), (0.70, 0.78, 0.24),
        ]
        for i in range(ARM_COUNT):
            theta = 2.0 * math.pi * i / ARM_COUNT
            base = np.array([ROOT_RADIUS * math.cos(theta), ROOT_RADIUS * math.sin(theta), BASE_Z], dtype=np.float32)
            yaw = theta + math.pi
            orientation = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float32)
            static_cylinder(
                stage,
                f"/World/ArmMounts/Mount_{i + 1:02d}",
                (base[0], base[1], 0.105),
                0.145,
                0.21,
                pedestal_mat,
                collide=True,
            )
            led = UsdGeom.Sphere.Define(stage, Sdf.Path(f"/World/ArmMounts/Indicator_{i + 1:02d}"))
            led.CreateRadiusAttr(0.036)
            led.CreateDisplayColorAttr([Gf.Vec3f(*robot_colors[i])])
            led.AddTranslateOp().Set(Gf.Vec3d(float(base[0]), float(base[1]), 0.224))
            led_mat = create_material(stage, f"/World/Materials/LED_{i + 1:02d}", robot_colors[i], roughness=0.25, metallic=0.0)
            bind_material(led.GetPrim(), led_mat)

            base_path = f"/World/Arm_{i + 1:02d}"
            robot = self.world.scene.add(Franka(
                prim_path=base_path,
                name=f"franka_{i + 1:02d}",
                position=base,
                orientation=orientation,
            ))
            self.robots.append(robot)
            self.arms.append({
                "index": i,
                "theta": theta,
                "base_position": base,
                "base_path": base_path,
                "hand_path": f"{base_path}/panda_hand",
                "left_path": f"{base_path}/panda_leftfinger",
                "right_path": f"{base_path}/panda_rightfinger",
                "color": robot_colors[i],
                "catch_q": None,
                "release_q": None,
                "current_q": None,
                "catch_position": None,
                "release_position": None,
                "catch_orientation": None,
                "release_orientation": None,
            })

        # A red, rubber-like sphere; it begins kinematic for the arm warm-up,
        # then becomes a gravity-driven rigid body before the first throw.
        ball_geo = UsdGeom.Sphere.Define(stage, Sdf.Path("/World/RelayBall"))
        ball_geo.CreateRadiusAttr(BALL_RADIUS)
        ball_geo.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.025, 0.012)])
        self.ball_prim = ball_geo.GetPrim()
        self.ball_prim = UsdPhysics.CollisionAPI.Apply(self.ball_prim).GetPrim()
        rigid = UsdPhysics.RigidBodyAPI.Apply(self.ball_prim)
        rigid.CreateRigidBodyEnabledAttr(True)
        rigid.CreateKinematicEnabledAttr(True)
        mass = UsdPhysics.MassAPI.Apply(self.ball_prim)
        mass.CreateMassAttr(BALL_MASS_KG)
        physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(self.ball_prim)
        physx_rigid.CreateEnableCCDAttr(False)
        ball_material = UsdShade.Material.Define(stage, Sdf.Path("/World/Materials/BallPhysics"))
        ball_physics_api = UsdPhysics.MaterialAPI.Apply(ball_material.GetPrim())
        ball_physics_api.CreateStaticFrictionAttr(1.25)
        ball_physics_api.CreateDynamicFrictionAttr(1.05)
        ball_physics_api.CreateRestitutionAttr(0.08)
        bind_material(self.ball_prim, ball_mat)
        UsdShade.MaterialBindingAPI.Apply(self.ball_prim).Bind(ball_material, UsdShade.Tokens.weakerThanDescendants, "physics")

        physics_context = self.world.get_physics_context()
        physics_scene = stage.GetPrimAtPath(physics_context.prim_path)
        if physics_scene and physics_scene.IsValid():
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene)
            physx_scene.CreateEnableCCDAttr(True)
            physx_scene.CreateEnableStabilizationAttr(True)
            physx_scene.CreateSolverTypeAttr("TGS")

        self.camera = Camera(
            prim_path="/World/CineCamera",
            name="relay_cine_camera",
            position=np.array([-3.75, -4.15, 3.05], dtype=np.float32),
            orientation=camera_orientation([-3.75, -4.15, 3.05], [0.0, 0.0, 0.80]),
            frequency=VIDEO_FPS,
            resolution=(1280, 720),
        )
        self.camera.set_focal_length(1.75)
        self.camera_rig = CameraRig(self.camera)

        # Contact sensors are on the real Franka finger collision bodies.
        for i, arm in enumerate(self.arms):
            arm_sensors = []
            for finger_name, finger_path in (("left", arm["left_path"]), ("right", arm["right_path"])):
                sensor = ContactSensor(
                    prim_path=f"{finger_path}/ball_contact_sensor",
                    name=f"arm_{i + 1}_{finger_name}_contact",
                    dt=DT,
                    translation=np.zeros(3, dtype=np.float32),
                    min_threshold=0.0,
                    max_threshold=100000.0,
                    radius=0.055,
                )
                sensor.add_raw_contact_data_to_frame()
                arm_sensors.append(self.world.scene.add(sensor))
            self.contact_sensors.append(arm_sensors)

        self.ball_contact_sensor = ContactSensor(
            prim_path="/World/RelayBall/ball_contact_sensor",
            name="relay_ball_contact_sensor",
            dt=DT,
            translation=np.zeros(3, dtype=np.float32),
            min_threshold=0.0,
            max_threshold=100000.0,
            radius=0.06,
        )
        self.ball_contact_sensor.add_raw_contact_data_to_frame()
        self.world.scene.add(self.ball_contact_sensor)

        link_names = [f"panda_link{i}" for i in range(1, 9)] + [
            "panda_hand", "panda_leftfinger", "panda_rightfinger"
        ]
        for arm_index, arm in enumerate(self.arms):
            for link_name in link_names:
                path = f"{arm['base_path']}/{link_name}"
                prim = stage.GetPrimAtPath(path)
                geometry_root = stage.GetPrimAtPath(path + "/geometry")
                if geometry_root.IsValid() and geometry_root.IsInstanceable():
                    geometry_root.SetInstanceable(False)
        for arm_index, arm in enumerate(self.arms):
            root_prim = stage.GetPrimAtPath(arm["base_path"])
            for subprim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
                if subprim.IsInstanceProxy() or not subprim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                collider_path = str(subprim.GetPath())
                is_finger = "panda_leftfinger" in collider_path or "panda_rightfinger" in collider_path
                self.robot_collision_paths.append((arm_index, collider_path, is_finger))
                if is_finger:
                    self.finger_collision_paths.append((arm_index, collider_path))
                UsdPhysics.CollisionAPI.Apply(subprim).CreateCollisionEnabledAttr().Set(False)
        self.ball_collision_filter = UsdPhysics.FilteredPairsAPI.Apply(self.ball_prim).CreateFilteredPairsRel()
        for _, path, is_finger in self.robot_collision_paths:
            if not is_finger:
                self.ball_collision_filter.AddTarget(Sdf.Path(path))
        
        self.ball_geo = ball_geo
        self.world.reset()
        self.camera.initialize()
        self.ball = SingleRigidPrim(prim_path="/World/RelayBall", name="relay_ball", mass=BALL_MASS_KG)
        self.ball.initialize()
        # Precompute real IK targets at the throw stations and ballistic apexes.
        releases = []
        for i in range(ARM_COUNT):
            theta = 2.0 * math.pi * i / ARM_COUNT + math.pi / ARM_COUNT
            releases.append(np.array([RELEASE_RADIUS * math.cos(theta), RELEASE_RADIUS * math.sin(theta), RELEASE_Z], dtype=np.float32))
        catches = []
        for i in range(ARM_COUNT):
            # Arm i receives from arm i-1 at their shared boundary.  This
            # indexing matters: using the *following* boundary sent arm 1
            # across the whole ring to catch arm 3's throw.
            previous_release = releases[(i - 1) % ARM_COUNT]
            own_release = releases[i]
            midpoint = 0.5 * (previous_release + own_release)
            midpoint[2] = RELEASE_Z + APEX_RISE
            catches.append(midpoint)

        for i, arm in enumerate(self.arms):
            robot = self.robots[i]
            solver = KinematicsSolver(robot, end_effector_frame_name="panda_hand")
            base_yaw = arm["theta"] + math.pi
            base_orientation = np.array(
                [math.cos(base_yaw / 2.0), 0.0, 0.0, math.sin(base_yaw / 2.0)],
                dtype=np.float32,
            )
            solver._kinematics.set_robot_base_pose(arm["base_position"], base_orientation)
            self.solvers.append(solver)
            arm["release_position"] = releases[i]
            arm["catch_position"] = catches[i]
            incoming = catches[i] - releases[(i - 1) % ARM_COUNT]
            outgoing = catches[(i + 1) % ARM_COUNT] - releases[i]
            incoming_velocity = np.array([
                incoming[0] / FLIGHT_TIME,
                incoming[1] / FLIGHT_TIME,
                incoming[2] / FLIGHT_TIME - 0.5 * 9.81 * FLIGHT_TIME,
            ])
            outgoing_velocity = np.array([
                outgoing[0] / FLIGHT_TIME,
                outgoing[1] / FLIGHT_TIME,
                outgoing[2] / FLIGHT_TIME + 0.5 * 9.81 * FLIGHT_TIME,
            ])
            arm["catch_orientation"] = tool_orientation_axis(incoming_velocity)
            arm["release_orientation"] = tool_orientation_axis(outgoing_velocity)
            arm["catch_q"] = self._solve_arm_pose(i, catches[i], arm["catch_orientation"])
            arm["release_q"] = self._solve_arm_pose(i, releases[i], arm["release_orientation"])
            if arm["catch_q"] is None or arm["release_q"] is None:
                raise RuntimeError(f"Panda IK failed for arm {i + 1}: catch/release station is unreachable")
            # Position-only Panda IK gives Lula freedom to keep each target in
            # its reachable workspace. Derive the true grasp center from FK
            # instead of pretending every gripper points exactly downward.
            arm["catch_position"] = self._grasp_center_from_fk(i, arm["catch_q"])
            arm["release_position"] = self._grasp_center_from_fk(i, arm["release_q"])
            initial = arm["release_q"] if i == 0 else arm["catch_q"]
            arm["current_q"] = initial.copy()
            set_drive_target(robot, self._full_joint_target(initial, GRIP_OPEN))

        # The physical ball is staged inside arm 1's open gripper while the
        # robot drives settle. The initial fixed joint is the held-ball state.
        self.ball.set_world_pose(position=self.arms[0]["release_position"])
        for warm_step in range(int(1.50 / DT)):
            self._step(render=(warm_step % RENDER_EVERY == 0), save_frame=False)
        # Anchor the ball at the measured hand pose. This removes any initial
        # constraint error from solver-vs-simulator FK drift.
        hand_world = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
            stage.GetPrimAtPath(self.arms[0]["hand_path"])
        )
        staged_ball_position = np.asarray(hand_world.ExtractTranslation(), dtype=np.float64)
        staged_ball_position += np.asarray(
            hand_world.TransformDir(Gf.Vec3d(0.0, 0.0, 0.10)), dtype=np.float64
        )
        self.ball.set_world_pose(position=staged_ball_position)
        UsdPhysics.RigidBodyAPI(self.ball_prim).GetKinematicEnabledAttr().Set(False)
        PhysxSchema.PhysxRigidBodyAPI(self.ball_prim).GetEnableCCDAttr().Set(True)
        initial_ball_position, initial_ball_orientation = self.ball.get_world_pose()
        self.grasp = grasp_joint(
            self.stage,
            "/World/GraspJoints/ActiveGrasp",
            self.arms[0]["hand_path"],
            "/World/RelayBall",
            initial_ball_position,
            initial_ball_orientation,
        )
        self.held_by = 0

        # Let the new dynamic ball and grasp constraint initialize without a
        # scripted position track.
        for _ in range(18):
            self._step(render=(self.physics_step % RENDER_EVERY == 0), save_frame=False)
        self._choose_camera_for_transfer(0)
        self.status = "READY"
        self.recording = True
        self.advance(0.30)
        self.events.append({"event": "initial_grasp", "arm": 1, "ball_state": "dynamic_fixed_grasp", "time_s": self.sim_time})

    def _solve_arm_pose(self, arm_index: int, ball_position: np.ndarray, orientation: np.ndarray):
        ball_position = np.asarray(ball_position, dtype=np.float32).reshape(3)
        seed = np.asarray(orientation, dtype=np.float32).reshape(4)
        current_q = np.asarray(self.robots[arm_index].get_joint_positions(), dtype=np.float32)[:7]
        best_result = None
        best_score = float("inf")
        # Preserve the trajectory-aligned finger axis, searching wrist roll
        # around that axis, then choose the reachable solution requiring the
        # smallest joint motion from the arm's measured state.
        for step in range(12):
            angle = math.radians(30.0 * step)
            q_roll = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)], dtype=np.float32)
            w1, x1, y1, z1 = seed
            w2, x2, y2, z2 = q_roll
            wrist_orientation = np.array([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ], dtype=np.float32)
            finger_axis = quaternion_to_matrix(wrist_orientation)[:, 2].astype(np.float32)
            palm_position = ball_position - finger_axis * 0.10
            actions, success = self.solvers[arm_index].compute_inverse_kinematics(
                target_position=palm_position,
                target_orientation=wrist_orientation,
                position_tolerance=0.02,
                orientation_tolerance=0.30,
            )
            if not success or actions is None or actions.joint_positions is None:
                continue
            result = np.asarray(actions.joint_positions, dtype=np.float32).reshape(-1)
            if result.size < 7:
                continue
            result = result[:7].copy()
            actual_center = self._grasp_center_from_fk(arm_index, result)
            center_error = float(np.linalg.norm(actual_center - ball_position))
            if center_error <= 0.035:
                joint_motion = float(np.linalg.norm(result - current_q))
                score = joint_motion + center_error * 2.0
                if score < best_score:
                    best_result = result
                    best_score = score
        return best_result

    def _grasp_center_from_fk(self, arm_index: int, joints: np.ndarray) -> np.ndarray:
        palm_position, rotation = self.solvers[arm_index]._kinematics.compute_forward_kinematics(
            "panda_hand", np.asarray(joints, dtype=np.float32)
        )
        # The Panda's finger length to the nominal ball center is ~100 mm in
        # the hand frame. This local offset rotates with the wrist during carry.
        return (
            np.asarray(palm_position, dtype=np.float32).reshape(3)
            + np.asarray(rotation, dtype=np.float32).reshape(3, 3)[:, 2] * 0.10
        )

    @staticmethod
    def _full_joint_target(arm_q: np.ndarray, gripper: float) -> np.ndarray:
        return np.concatenate([np.asarray(arm_q, dtype=np.float32)[:7], np.array([gripper, gripper], dtype=np.float32)])

    def _choose_camera_for_transfer(self, transfer_index: int):
        # Each lens gets two throws. This keeps the audience oriented while
        # switching among wide, overhead, side, and reverse views.
        self.camera_rig.choose((transfer_index // 2) % len(self.camera_rig.shots), self.sim_time)

    def _set_active_finger_collisions(self, arm_index: int | None):
        """Enable only the receiving Panda's finger collision meshes."""
        for owner, path, is_finger in self.robot_collision_paths:
            prim = self.stage.GetPrimAtPath(path)
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            enabled = arm_index is not None and is_finger and owner == arm_index
            collision_api.CreateCollisionEnabledAttr().Set(enabled)

    def _contacting_ball(self, arm_index: int):
        reports = []
        for side_index, sensor in enumerate(self.contact_sensors[arm_index]):
            try:
                frame = sensor.get_current_frame() or {}
            except Exception:
                continue
            self.latest_force = max(self.latest_force, float(frame.get("force", 0.0)))
            for contact in frame.get("contacts", []):
                names = {str(contact.get("body0", "")), str(contact.get("body1", ""))}
                if any("RelayBall" in name for name in names):
                    reports.append((side_index, frame, contact))
        return reports

    def _step(self, render: bool, save_frame: bool = True, progress: float | None = None, cue=None):
        if render:
            self.camera_rig.update(self.sim_time)
        # A render=True World.step advances at rendering_dt (1/30 s), while
        # render=False advances at physics_dt (1/120 s). Always integrate one
        # fixed physics tick, then render without advancing the simulation.
        # Mixing the two step sizes made gravity effectively 1.75x stronger.
        self.world.step(render=False, update_fabric=render)
        if render:
            self.world.render()
        self.physics_step += 1
        self.sim_time += DT
        if render and save_frame and self.recording:
            self._capture_frame(progress=progress, cue=cue)

    def advance(self, seconds: float, controller=None, force=None, save_frame=True):
        count = max(1, int(round(float(seconds) / DT)))
        for local_step in range(count):
            t = (local_step + 1) / count
            if controller is not None:
                controller(t, local_step)
                if force is not None:
                    self.ball._rigid_prim_view.apply_forces(
                        np.asarray(force, dtype=np.float32).reshape(1, 3), is_global=True
                    )
            render = (self.physics_step + 1) % RENDER_EVERY == 0
            self._step(render=render, save_frame=save_frame, progress=t)

    def _capture_frame(self, progress=None, cue=None):
        rgb = self.camera.get_rgb(device="cpu")
        if rgb is None:
            return
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim == 3 and rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        self.latest_rgb = rgb
        canvas = Image.fromarray(rgb).convert("RGBA")
        draw = ImageDraw.Draw(canvas, "RGBA")
        font_regular, font_medium, font_bold = self._fonts()

        # restrained broadcast graphics: clear hierarchy, no heavy full-frame panel
        draw.rounded_rectangle((35, 28, 404, 116), radius=10, fill=(8, 13, 19, 190), outline=(255, 255, 255, 28), width=1)
        draw.text((54, 41), "FRANKA PANDA  /  SIX-ARM RELAY", font=font_bold, fill=(246, 248, 250, 255))
        draw.text((54, 73), "DYNAMIC RIGID-BODY FLIGHT   ·   120 Hz PHYSICS", font=font_regular, fill=(173, 190, 208, 245))

        draw.rounded_rectangle((1028, 28, 1246, 77), radius=9, fill=(8, 13, 19, 190), outline=(255, 255, 255, 28), width=1)
        draw.ellipse((1046, 46, 1058, 58), fill=(255, 93, 47, 255))
        draw.text((1070, 41), f"CAM  /  {self.camera_rig.name}", font=font_medium, fill=(238, 243, 248, 255))

        # translucent lower third with time and model cue telemetry
        draw.rectangle((0, 621, 1280, 720), fill=(5, 9, 14, 215))
        draw.line((0, 621, 1280, 621), fill=(255, 255, 255, 36), width=1)
        status_color = (84, 215, 179, 255) if "CATCH" in self.status or "HELD" in self.status else (255, 186, 91, 255)
        draw.text((42, 638), f"TRANSFER  {self.transfer_label:02d} / {self.transfers:02d}", font=font_bold, fill=(245, 248, 251, 255))
        draw.text((42, 669), self.status, font=font_medium, fill=status_color)
        cue_action = cue if cue else self.policy.get("action", "WAIT")
        confidence = float(self.policy.get("confidence", 0.0))
        latency = float(self.policy.get("latency_ms", {}).get("client_round_trip", 0.0))
        draw.text((498, 639), f"MoE 16-D  /  4 EXPERTS     CUE  {cue_action}  {confidence:.2f}", font=font_medium, fill=(202, 215, 229, 255))
        draw.text((498, 669), f"INFERENCE {latency:5.1f} ms     ·     CONTACT FORCE {self.latest_force:5.1f} N", font=font_regular, fill=(141, 160, 181, 255))

        if progress is not None:
            timeline_x0, timeline_x1, y = 42, 1238, 704
            draw.line((timeline_x0, y, timeline_x1, y), fill=(74, 86, 102, 210), width=2)
            total_frames = max(self.transfers * 3 * VIDEO_FPS, 1)
            frac = min(self.frame_index / total_frames, 1.0)
            draw.line((timeline_x0, y, timeline_x0 + int((timeline_x1 - timeline_x0) * frac), y), fill=(250, 86, 54, 255), width=3)
            for station in range(ARM_COUNT):
                x = timeline_x0 + int((timeline_x1 - timeline_x0) * station / ARM_COUNT)
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(203, 214, 225, 230))

        output = canvas.convert("RGB")
        output.save(FRAMES / f"frame_{self.frame_index:05d}.png", compress_level=1)
        self.frame_index += 1

    @staticmethod
    def _fonts():
        font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        try:
            regular = ImageFont.truetype(str(font_dir / "segoeui.ttf"), 16)
            medium = ImageFont.truetype(str(font_dir / "seguisb.ttf"), 17)
            bold = ImageFont.truetype(str(font_dir / "segoeuib.ttf"), 20)
            return regular, medium, bold
        except OSError:
            default = ImageFont.load_default()
            return default, default, default

    def _policy_step(self, context: str):
        if self.latest_rgb is None:
            self.advance(1.0 / VIDEO_FPS)
        result = self.client.decide(self.latest_rgb)
        self.policy = result
        self.policy_counts[result["action"]] += 1
        self.policy_rows.append({
            "transfer": self.transfer_label,
            "context": context,
            "action": result["action"],
            "confidence": float(result["confidence"]),
            "latency_ms": float(result["latency_ms"]["client_round_trip"]),
            "gate_weights": result.get("gate_weights", []),
            "latent": result.get("latent", []),
        })
        print(
            f"POLICY transfer={self.transfer_label} context={context} action={result['action']} "
            f"confidence={result['confidence']:.3f} latency_ms={result['latency_ms']['client_round_trip']:.2f}",
            flush=True,
        )
        return result

    def _set_arm_motion(self, arm_index: int, start_q: np.ndarray, end_q: np.ndarray, gripper_start: float, gripper_end: float, duration: float, controller=None):
        robot = self.robots[arm_index]
        start = np.asarray(start_q, dtype=np.float32).copy()
        end = np.asarray(end_q, dtype=np.float32).copy()

        def update(amount, local_step):
            eased = smooth5(amount)
            arm_q = (1.0 - eased) * start + eased * end
            finger = (1.0 - eased) * gripper_start + eased * gripper_end
            set_drive_target(robot, self._full_joint_target(arm_q, finger))
            if controller is not None:
                controller(amount, local_step)

        self.advance(duration, controller=update)
        self.arms[arm_index]["current_q"] = end.copy()

    def _sensor_contact(self, arm_index: int):
        contacts = self._contacting_ball(arm_index)
        if not contacts:
            return None
        selected = contacts[0]
        side, frame, contact = selected
        return {
            "finger": "left" if side == 0 else "right",
            "force_n": float(frame.get("force", 0.0)),
            "body0": str(contact.get("body0", "")),
            "body1": str(contact.get("body1", "")),
        }

    def run(self):
        self.worker, self.client = start_inference_worker(self.device)
        try:
            self._run_transfers()
        finally:
            try:
                self.client.connection.close()
            except Exception:
                pass
            self.worker.terminate()
            try:
                self.worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait(timeout=5)
            self.write_outputs()

    def _run_transfers(self):
        self.latest_force = 0.0
        for transfer in range(self.transfers):
            source = transfer % ARM_COUNT
            receiver = (source + 1) % ARM_COUNT
            self.transfer_label = transfer + 1
            self._choose_camera_for_transfer(transfer)
            self.latest_force = 0.0
            self.status = f"ARM {source + 1}  /  HELD  →  ARM {receiver + 1}  /  READY"
            self.advance(0.12)

            # The existing 16-D/4-expert checkpoint supplies the release cue.
            # If it does not emit THROW, retry briefly and then use the safe
            # scheduled controller; every fallback is counted in the report.
            prediction = self._policy_step("throw_ready")
            for attempt in range(2):
                if prediction["action"] == "THROW":
                    break
                self.advance(0.10)
                prediction = self._policy_step("throw_ready_retry")
            model_threw = prediction["action"] == "THROW"
            if not model_threw:
                self.throw_fallbacks += 1
                self.events.append({"event": "throw_fallback", "transfer": transfer + 1, "reason": "MoE cue was not THROW", "time_s": self.sim_time})
            self.status = f"ARM {source + 1}  /  RELEASE  →  ARM {receiver + 1}  /  TRACK"
            # Reposition the carrying arm from its actual joint state into the
            # release pose while it holds the ball. Smoothstep eases both the
            # joints and the gripper opening over a real simulated interval.
            # Predict the carried ball's release point from the live grasp
            # offset and the source's release-pose FK. Move the next Panda
            # toward the corresponding catch pose during the same interval.
            measured_source_q = np.asarray(self.robots[source].get_joint_positions(), dtype=np.float32)[:7]
            measured_receiver_q = np.asarray(self.robots[receiver].get_joint_positions(), dtype=np.float32)[:7]
            nominal_receiver_q = self.arms[receiver]["catch_q"].copy()
            nominal_source = self.arms[source]["release_position"]
            nominal_target = self.arms[receiver]["catch_position"]
            predicted_receiver_q = nominal_receiver_q
            if self.grasp is not None:
                grasp_local_position = np.asarray(self.grasp.GetLocalPos0Attr().Get(), dtype=np.float32).reshape(3)
                release_palm, release_rotation = self.solvers[source]._kinematics.compute_forward_kinematics(
                    "panda_hand", self.arms[source]["release_q"]
                )
                predicted_release = (
                    np.asarray(release_palm, dtype=np.float32).reshape(3)
                    + np.asarray(release_rotation, dtype=np.float32).reshape(3, 3) @ grasp_local_position
                )
                predicted_target = np.asarray(
                    nominal_target + (predicted_release - nominal_source), dtype=np.float32
                )
                candidate = self._solve_arm_pose(
                    receiver, predicted_target, self.arms[receiver]["catch_orientation"]
                )
                if candidate is not None:
                    predicted_receiver_q = candidate

            def preposition_receiver(amount, _):
                eased = smooth5(amount)
                receiver_q = (1.0 - eased) * measured_receiver_q + eased * predicted_receiver_q
                set_drive_target(
                    self.robots[receiver],
                    self._full_joint_target(receiver_q, GRIP_OPEN),
                )

            self._set_arm_motion(
                source,
                measured_source_q,
                self.arms[source]["release_q"],
                GRIP_HOLD,
                GRIP_OPEN,
                0.38,
                controller=preposition_receiver,
            )
            self.arms[receiver]["current_q"] = predicted_receiver_q.copy()

            # Recenter the receiver on the source's measured rigid-body pose,
            # preserving the nominal handoff geometry if the hand drive lagged.
            ball_position, _ = self.ball.get_world_pose()
            actual_release = np.asarray(ball_position, dtype=np.float32)
            adjusted_target = np.asarray(nominal_target + (actual_release - nominal_source), dtype=np.float32)
            adjusted_catch_q = self._solve_arm_pose(receiver, adjusted_target, self.arms[receiver]["catch_orientation"])
            if adjusted_catch_q is not None:
                # Small IK correction follows the actual source pose before flight.
                measured_receiver_q = np.asarray(self.robots[receiver].get_joint_positions(), dtype=np.float32)[:7]
                self._set_arm_motion(receiver, measured_receiver_q, adjusted_catch_q, GRIP_OPEN, GRIP_OPEN, 0.60)
                self.arms[receiver]["catch_q"] = adjusted_catch_q

            self._set_active_finger_collisions(receiver)

            if self.grasp is not None:
                self.stage.RemovePrim(self.grasp.GetPath())
                self.grasp = None

            # Give PhysX one tick to process removal of the grasp constraint
            # before measuring release velocity and applying the launch pulse.
            self.advance(DT, save_frame=False)
            ball_position, _ = self.ball.get_world_pose()
            actual_release = np.asarray(ball_position, dtype=np.float32)
            adjusted_target = np.asarray(nominal_target + (actual_release - nominal_source), dtype=np.float32)
            direction = adjusted_target - actual_release
            horizontal_distance = float(np.linalg.norm(direction[:2]))
            # Re-solve against the release position measured after the grasp
            # joint is gone. Its final constraint settle can move the ball a
            # few centimeters, so the receiver tracks this corrected landing
            # pose during the first part of free flight.
            flight_receiver_start_q = np.asarray(
                self.robots[receiver].get_joint_positions(), dtype=np.float32
            )[:7]
            flight_receiver_target_q = self._solve_arm_pose(
                receiver, adjusted_target, self.arms[receiver]["catch_orientation"]
            )
            if flight_receiver_target_q is None:
                flight_receiver_target_q = adjusted_catch_q
            if flight_receiver_target_q is None:
                flight_receiver_target_q = self.arms[receiver]["catch_q"]
            self.arms[receiver]["catch_q"] = np.asarray(flight_receiver_target_q, dtype=np.float32).copy()
            desired_velocity = np.array(
                [
                    direction[0] / FLIGHT_TIME,
                    direction[1] / FLIGHT_TIME,
                    direction[2] / FLIGHT_TIME + 0.5 * 9.81 * FLIGHT_TIME,
                ],
                dtype=np.float32,
            )
            current_velocity = np.asarray(self.ball.get_linear_velocity(), dtype=np.float32)
            release_impulse = BALL_MASS_KG * (desired_velocity - current_velocity)
            # This is the impulse-equivalent release condition for the throw.
            # PhysX integrates the free flight and all subsequent contacts.
            self.ball.set_linear_velocity(current_velocity + release_impulse / BALL_MASS_KG)

            self.status = f"THROW  /  {horizontal_distance:.2f} m  ·  {np.linalg.norm(desired_velocity):.2f} m/s"
            self.events.append({
                "event": "release_impulse",
                "transfer": transfer + 1,
                "source_arm": source + 1,
                "receiver_arm": receiver + 1,
                "release_position_m": actual_release.tolist(),
                "target_velocity_mps": desired_velocity.tolist(),
                "velocity_before_impulse_mps": current_velocity.tolist(),
                "impulse_n_s": release_impulse.tolist(),
                "release_velocity_mps": desired_velocity.tolist(),
                "model_cue": prediction["action"],
                "model_confidence": float(prediction["confidence"]),
                "time_s": self.sim_time,
            })

            source_start = self.arms[source]["release_q"].copy()
            source_end = self.arms[source]["catch_q"].copy()
            ball_contact = None
            elapsed_steps = 0
            flight_limit = int(round((FLIGHT_TIME + 0.10) / DT))
            catch_cue_seen = False
            while elapsed_steps < flight_limit and ball_contact is None:
                t = elapsed_steps * DT
                fraction = min(t / FLIGHT_TIME, 1.0)
                source_q = (1.0 - smooth5(fraction)) * source_start + smooth5(fraction) * source_end
                source_fingers = GRIP_OPEN
                set_drive_target(self.robots[source], self._full_joint_target(source_q, source_fingers))

                # Let the same vision MoE cue an early close; a scheduled close
                # near the predicted apex keeps the receiver physically safe.
                if (not catch_cue_seen) and t >= 0.075 and self.latest_rgb is not None:
                    prediction = self._policy_step("catch_in_flight")
                    catch_cue_seen = True
                close_start = FLIGHT_TIME - 0.18 if prediction["action"] == "CATCH" else FLIGHT_TIME - 0.16
                if t >= close_start:
                    close_amount = smooth5((t - close_start) / 0.16)
                    finger_target = (1.0 - close_amount) * GRIP_OPEN + close_amount * GRIP_HOLD
                else:
                    finger_target = GRIP_OPEN
                catch_move_fraction = smooth5(t / 0.38)
                receiver_q = (
                    (1.0 - catch_move_fraction) * flight_receiver_start_q
                    + catch_move_fraction * flight_receiver_target_q
                )
                set_drive_target(self.robots[receiver], self._full_joint_target(receiver_q, finger_target))

                self.advance(DT)
                elapsed_steps += 1
                self._record_ball_contacts(transfer + 1, receiver)
                if elapsed_steps % 4 == 0:
                    ball_pos, _ = self.ball.get_world_pose()
                    ball_vel = self.ball.get_linear_velocity()
                    receiver_joints = np.asarray(self.robots[receiver].get_joint_positions(), dtype=np.float32)[:7]
                    receiver_palm, receiver_rotation = self.solvers[receiver]._kinematics.compute_forward_kinematics(
                        "panda_hand", receiver_joints
                    )
                    receiver_center = (
                        np.asarray(receiver_palm, dtype=np.float32).reshape(3)
                        + np.asarray(receiver_rotation, dtype=np.float32).reshape(3, 3)[:, 2] * 0.10
                    )
                    error = float(np.linalg.norm(np.asarray(ball_pos) - receiver_center))
                    self.ball_positions.append({
                        "transfer": transfer + 1,
                        "time_s": self.sim_time,
                        "flight_time_s": elapsed_steps * DT,
                        "position_m": np.asarray(ball_pos).tolist(),
                        "velocity_mps": np.asarray(ball_vel).tolist(),
                        "receiver_grasp_center_m": receiver_center.tolist(),
                        "receiver_target_m": np.asarray(adjusted_target).tolist(),
                        "receiver_error_m": error,
                    })
                ball_contact = self._sensor_contact(receiver)

            if ball_contact is None:
                self.catch_failures += 1
                self.status = f"MISSED  /  ARM {receiver + 1}"
                ball_pos, _ = self.ball.get_world_pose()
                self.events.append({"event": "missed_catch", "transfer": transfer + 1, "receiver_arm": receiver + 1, "ball_position_m": np.asarray(ball_pos).tolist(), "time_s": self.sim_time})
                print(f"CATCH_MISS transfer={transfer + 1} arm={receiver + 1} pos={np.asarray(ball_pos).round(3).tolist()}", flush=True)
                break

            self.latest_force = float(ball_contact["force_n"])
            self.status = f"CONTACT  /  ARM {receiver + 1}  ·  GRASP"
            self.grasp = grasp_joint(
                self.stage,
                "/World/GraspJoints/ActiveGrasp",
                self.arms[receiver]["hand_path"],
                "/World/RelayBall",
                *self.ball.get_world_pose(),
            )
            self.held_by = receiver
            self.events.append({
                "event": "physical_contact_grasp",
                "transfer": transfer + 1,
                "arm": receiver + 1,
                "finger": ball_contact["finger"],
                "force_n": ball_contact["force_n"],
                "contact_bodies": [ball_contact["body0"], ball_contact["body1"]],
                "flight_time_s": elapsed_steps * DT,
                "time_s": self.sim_time,
            })
            self.contact_rows.append({
                "transfer": transfer + 1,
                "source_arm": source + 1,
                "receiver_arm": receiver + 1,
                "finger": ball_contact["finger"],
                "force_n": ball_contact["force_n"],
                "flight_time_s": elapsed_steps * DT,
                "contact_body0": ball_contact["body0"],
                "contact_body1": ball_contact["body1"],
            })
            print(f"CATCH_OK transfer={transfer + 1} arm={receiver + 1} finger={ball_contact['finger']} force_n={ball_contact['force_n']:.2f} flight_s={elapsed_steps * DT:.3f}", flush=True)

            self.advance(0.12)
            self.status = f"ARM {receiver + 1}  /  CARRY TO THROW POSE"
            self._set_arm_motion(receiver, receiver_q, self.arms[receiver]["release_q"], GRIP_HOLD, GRIP_HOLD, 0.34)
            self.held_by = receiver
            self.status = f"ARM {receiver + 1}  /  HELD  ·  READY"
            self.advance(0.14)

        self.status = "RELAY COMPLETE" if self.catch_failures == 0 and len(self.contact_rows) == self.transfers else "RELAY STOPPED  /  SEE PHYSICS LOG"
        self.advance(0.35)

    def write_outputs(self):
        import csv
        import statistics

        RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
        latencies = [row["latency_ms"] for row in self.policy_rows]
        result = {
            "simulation": {
                "engine": "NVIDIA PhysX in Isaac Sim",
                "physics_dt_s": DT,
                "render_fps": VIDEO_FPS,
                "robot_model": "Franka Emika Panda, official Isaac Sim USD asset with articulated 7-DoF arm and parallel gripper",
                "robot_count": ARM_COUNT,
                "ball_model": {"type": "dynamic rigid-body sphere", "mass_kg": BALL_MASS_KG, "radius_m": BALL_RADIUS, "ccd": True},
                "flight": "impulse-equivalent velocity at gripper release, then 120 Hz PhysX gravity and rigid-body contact; no scripted ball path or in-flight pose updates",
                "catch": "finger contact sensor followed by a fixed grasp constraint; no ball-position snap at catch",
                "requested_transfers": self.transfers,
                "completed_contact_grasps": len(self.contact_rows),
                "catch_failures": self.catch_failures,
                "throw_fallbacks": self.throw_fallbacks,
                "simulation_time_s": self.sim_time,
                "captured_frames": self.frame_index,
                "camera_angles": [shot[0] for shot in self.camera_rig.shots],
            },
            "same_model": {"checkpoint": str(ROOT / "models" / "throw_catch_moe16.pt"), "latent_dimension": 16, "experts": 4, "actions": dict(self.policy_counts), "mean_latency_ms": statistics.mean(latencies) if latencies else None, "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None},
            "contact_log": self.contact_rows,
            "ball_contact_events": self.ball_contact_events,
            "events": self.events,
        }
        RESULT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if self.contact_rows:
            with CONTACT_CSV.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.contact_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.contact_rows)
        if self.ball_positions:
            with (ROOT / "results" / "throw_catch_physics_ball_trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.ball_positions[0].keys()))
                writer.writeheader()
                writer.writerows(self.ball_positions)
        print(f"PHYSICS_RESULT {RESULT_JSON}", flush=True)
        print(f"PHYSICS_SUMMARY catches={len(self.contact_rows)}/{self.transfers} misses={self.catch_failures} frames={self.frame_index} sim_s={self.sim_time:.2f}", flush=True)

    def _record_ball_contacts(self, transfer: int, receiver: int):
        try:
            ball_frame = self.ball_contact_sensor.get_current_frame() or {}
        except Exception:
            ball_frame = {}
        for contact in ball_frame.get("contacts", []):
            body0 = str(contact.get("body0", ""))
            body1 = str(contact.get("body1", ""))
            row = {
                "transfer": transfer,
                "receiver_arm": receiver + 1,
                "contact_arm": None,
                "finger": None,
                "time_s": self.sim_time,
                "force_n": float(ball_frame.get("force", 0.0)),
                "body0": body0,
                "body1": body1,
            }
            if not self.ball_contact_events or any(
                self.ball_contact_events[-1].get(key) != row[key]
                for key in ("transfer", "body0", "body1")
            ):
                self.ball_contact_events.append(row)
        for arm_index, sensors in enumerate(self.contact_sensors):
            for side, sensor in enumerate(sensors):
                try:
                    frame = sensor.get_current_frame() or {}
                except Exception:
                    continue
                for contact in frame.get("contacts", []):
                    body0 = str(contact.get("body0", ""))
                    body1 = str(contact.get("body1", ""))
                    if "RelayBall" not in body0 and "RelayBall" not in body1:
                        continue
                    row = {
                        "transfer": transfer,
                        "receiver_arm": receiver + 1,
                        "contact_arm": arm_index + 1,
                        "finger": "left" if side == 0 else "right",
                        "time_s": self.sim_time,
                        "force_n": float(frame.get("force", 0.0)),
                        "body0": body0,
                        "body1": body1,
                    }
                    if not self.ball_contact_events or any(
                        self.ball_contact_events[-1].get(key) != row[key]
                        for key in ("transfer", "contact_arm", "body0", "body1")
                    ):
                        self.ball_contact_events.append(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfers", type=int, default=18)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=261005)
    args = parser.parse_args()
    if args.transfers < 6:
        raise ValueError("Use at least six transfers to show a complete six-arm circuit")
    simulation = None
    try:
        simulation = RelaySimulation(args.transfers, args.device, args.seed)
        simulation.run()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
