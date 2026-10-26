"""Six-arm relay scene: Franka Pandas around a ring of shared hand-off trays.

Arm i picks from tray i-1 (on its right as seen from the ring centre) and
places on tray i, so objects circulate counter-clockwise. Everything that
moves is simulated by PhysX: articulated arms on joint position drives,
friction grasps with the real Panda fingers, dynamic rigid objects. Nothing is
welded, teleported or velocity-set after ``reset``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCapsule, DynamicCuboid, DynamicCylinder, DynamicSphere
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka, KinematicsSolver
from isaacsim.sensors.camera import Camera

from sim.ee import cap, ee_goal
from sim.common import (
    bind,
    bind_physics,
    configure_physics_scene,
    look_at_quat,
    make_material,
    physics_material,
    quat_mul,
    set_material_color,
    static_box,
    static_cylinder,
    studio_lights,
    yaw_quat,
)

ARM_COUNT = 6
R_ARM = 1.05
R_PAD = 0.70
PAD_HALF = 0.09
PAD_TOP = 0.012
PHYSICS_DT = 1.0 / 120.0
CONTROL_HZ = 15
SUBSTEPS = int(round(1.0 / (CONTROL_HZ * PHYSICS_DT)))
STATION_RES = int(os.environ.get("RELAY_STATION_RES", "96"))
WRIST_RES = int(os.environ.get("RELAY_WRIST_RES", "64"))
HOME_Q = np.array([0.0, -0.55, 0.0, -2.30, 0.0, 1.75, 0.785], dtype=np.float32)
GRIP_OPEN = 0.04
TCP_OFFSET = 0.1034

PALETTE = {
    "red": (0.85, 0.08, 0.06), "orange": (0.95, 0.42, 0.05), "yellow": (0.92, 0.80, 0.08),
    "green": (0.10, 0.70, 0.22), "cyan": (0.05, 0.70, 0.80), "blue": (0.10, 0.25, 0.90),
    "purple": (0.55, 0.18, 0.80), "white": (0.88, 0.88, 0.86),
}


@dataclass(frozen=True)
class Shape:
    name: str
    kind: str
    dims: tuple          # cuboid: (x, y, z); cylinder: (radius, height); sphere: (radius,); capsule: (radius, length)
    mass: float
    symmetry: float      # yaw symmetry of the grasp in radians (0 = rotationally symmetric)
    rest_half_height: float
    grasp_dz: float      # grasp point above the object centre
    grasp_width: float   # finger separation when closed on the object


SHAPES = [
    Shape("cube", "cuboid", (0.05, 0.05, 0.05), 0.12, math.pi / 2, 0.025, 0.0, 0.050),
    Shape("tall cylinder", "cylinder", (0.022, 0.10), 0.10, 0.0, 0.050, 0.02, 0.044),
    Shape("ball", "sphere", (0.028,), 0.08, 0.0, 0.028, 0.0, 0.056),
    Shape("bar", "cuboid", (0.13, 0.03, 0.03), 0.10, math.pi, 0.015, 0.0, 0.030),
    Shape("puck", "cylinder", (0.031, 0.035), 0.09, 0.0, 0.0175, 0.0, 0.062),
    Shape("capsule", "capsule", (0.02, 0.07), 0.08, math.pi, 0.020, 0.0, 0.040),
]


def arm_theta(i: int) -> float:
    return 2.0 * math.pi * i / ARM_COUNT


def pad_center(j: int) -> np.ndarray:
    a = arm_theta(j) + math.pi / ARM_COUNT
    return np.array([R_PAD * math.cos(a), R_PAD * math.sin(a), PAD_TOP], dtype=np.float32)


def source_pad(i: int) -> int:
    return (i - 1) % ARM_COUNT


def dest_pad(i: int) -> int:
    return i % ARM_COUNT


class RelayScene:
    def __init__(self, object_count: int = 4, video: bool = False, cameras: bool = True, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.object_count = int(object_count)
        self.world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=1.0 / CONTROL_HZ)
        self.stage = self.world.stage
        self.robots = []
        self.solvers = []
        self.objects = []          # one rigid body per shape; inactive ones parked out of view
        self.station_cams = []
        self.wrist_cams = []
        self.video_cam = None
        self.active = []           # indices into SHAPES currently in play
        self.object_colors = {}
        self.sim_time = 0.0
        self.use_cameras = cameras
        self._build(video)

    # ------------------------------------------------------------------ build
    def _build(self, video: bool):
        stage = self.stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        floor_mat = make_material(stage, "/World/Looks/Floor", (0.03, 0.04, 0.055), roughness=0.5, metallic=0.1)
        table_mat = make_material(stage, "/World/Looks/Table", (0.16, 0.18, 0.21), roughness=0.35, metallic=0.25)
        tray_mat = make_material(stage, "/World/Looks/Tray", (0.30, 0.33, 0.37), roughness=0.4, metallic=0.2)
        rim_mat = make_material(stage, "/World/Looks/TrayRim", (0.85, 0.87, 0.90), roughness=0.3, metallic=0.4)
        friction = physics_material(stage, "/World/Physics/HighFriction", 1.2, 1.0, 0.0)
        static_box(stage, "/World/Floor", (0, 0, -0.75), (12.0, 12.0, 0.02), floor_mat)
        table = static_cylinder(stage, "/World/Table", (0, 0, -0.02), 1.55, 0.04, table_mat)
        bind_physics(table, friction)
        static_cylinder(stage, "/World/TableLeg", (0, 0, -0.39), 0.35, 0.70, table_mat, collide=False)
        studio_lights(stage)

        for j in range(ARM_COUNT):
            c = pad_center(j)
            a = arm_theta(j) + math.pi / ARM_COUNT
            plate = static_box(stage, f"/World/Trays/Tray_{j}/Plate", (c[0], c[1], PAD_TOP / 2), (2 * PAD_HALF, 2 * PAD_HALF, PAD_TOP), tray_mat, yaw=a)
            bind_physics(plate, friction)
            for k, (dx, dy, sx, sy) in enumerate([(PAD_HALF, 0, 0.008, 2 * PAD_HALF), (-PAD_HALF, 0, 0.008, 2 * PAD_HALF),
                                                   (0, PAD_HALF, 2 * PAD_HALF, 0.008), (0, -PAD_HALF, 2 * PAD_HALF, 0.008)]):
                wx = c[0] + dx * math.cos(a) - dy * math.sin(a)
                wy = c[1] + dx * math.sin(a) + dy * math.cos(a)
                static_box(stage, f"/World/Trays/Tray_{j}/Rim_{k}", (wx, wy, PAD_TOP + 0.006), (sx, sy, 0.012), rim_mat, yaw=a)

        for i in range(ARM_COUNT):
            theta = arm_theta(i)
            base = np.array([R_ARM * math.cos(theta), R_ARM * math.sin(theta), 0.0], dtype=np.float32)
            yaw = theta + math.pi
            path = f"/World/Arm_{i}"
            robot = self.world.scene.add(Franka(prim_path=path, name=f"arm_{i}", position=base, orientation=yaw_quat(yaw)))
            self.robots.append(robot)

        self.object_mats = {}
        for k, shape in enumerate(SHAPES):
            path = f"/World/Objects/Obj_{k}"
            park = np.array([3.0 + 0.3 * k, 3.0, 0.1], dtype=np.float32)
            color = np.array(PALETTE["white"], dtype=np.float32)
            if shape.kind == "cuboid":
                obj = DynamicCuboid(prim_path=path, name=f"obj_{k}", position=park, scale=np.array(shape.dims), size=1.0, mass=shape.mass, color=color)
            elif shape.kind == "cylinder":
                obj = DynamicCylinder(prim_path=path, name=f"obj_{k}", position=park, radius=shape.dims[0], height=shape.dims[1], mass=shape.mass, color=color)
            elif shape.kind == "sphere":
                obj = DynamicSphere(prim_path=path, name=f"obj_{k}", position=park, radius=shape.dims[0], mass=shape.mass, color=color)
            else:
                obj = DynamicCapsule(prim_path=path, name=f"obj_{k}", position=park, radius=shape.dims[0], height=shape.dims[1], mass=shape.mass, color=color)
                UsdGeom.Capsule(stage.GetPrimAtPath(path)).GetAxisAttr().Set(UsdGeom.Tokens.x)
            self.world.scene.add(obj)
            prim = stage.GetPrimAtPath(path)
            look = make_material(stage, f"/World/Looks/Obj_{k}", PALETTE["white"], roughness=0.45, metallic=0.05)
            bind(prim, look)
            bind_physics(prim, friction)
            self.object_mats[k] = f"/World/Looks/Obj_{k}"
            self.objects.append(obj)

        configure_physics_scene(self.world)

        if self.use_cameras:
            for i in range(ARM_COUNT):
                theta = arm_theta(i)
                yaw = theta + math.pi
                base = np.array([R_ARM * math.cos(theta), R_ARM * math.sin(theta), 0.0])
                fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
                # Overhead between the arm's two trays, tilted slightly back toward the arm,
                # so both trays, the gripper and the neighbours' hands are in view.
                cam_pos = base + 0.62 * fwd + np.array([0, 0, 1.30])
                look = base + 0.40 * fwd
                cam = Camera(prim_path=f"/World/StationCams/Cam_{i}", name=f"station_{i}", position=cam_pos.astype(np.float32),
                             orientation=look_at_quat(cam_pos, look, up_hint=tuple(fwd)), resolution=(STATION_RES, STATION_RES))
                self.station_cams.append(cam)
                hand = f"/World/Arm_{i}/panda_hand"
                local_pos = np.array([0.075, 0.0, 0.025])
                wcam = Camera(prim_path=f"{hand}/WristCam", name=f"wrist_{i}", translation=local_pos.astype(np.float32),
                              orientation=look_at_quat(local_pos, np.array([0.0, 0.0, 0.13]), up_hint=(1.0, 0.0, 0.0)),
                              resolution=(WRIST_RES, WRIST_RES))
                self.wrist_cams.append(wcam)
        if video:
            self.video_cam = Camera(prim_path="/World/VideoCam", name="video", position=np.array([2.6, -2.4, 2.0], dtype=np.float32),
                                    orientation=look_at_quat([2.6, -2.4, 2.0], [0, 0, 0.1]), resolution=(1280, 720))

        self.world.reset()
        for cam in self.station_cams + self.wrist_cams:
            cam.initialize()
            cam.set_focal_length(1.6)
        for cam in self.station_cams + self.wrist_cams:
            cam.set_clipping_range(0.01, 8.0)
        if self.video_cam is not None:
            self.video_cam.initialize()
            self.video_cam.set_focal_length(1.9)
            self.video_cam.set_clipping_range(0.05, 40.0)

        for i, robot in enumerate(self.robots):
            solver = KinematicsSolver(robot, end_effector_frame_name="panda_hand")
            theta = arm_theta(i)
            base = np.array([R_ARM * math.cos(theta), R_ARM * math.sin(theta), 0.0], dtype=np.float32)
            solver._kinematics.set_robot_base_pose(base, yaw_quat(theta + math.pi))
            self.solvers.append(solver)

    # ------------------------------------------------------------------ episode
    def reset(self, seed: int | None = None, shapes=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng = self.rng
        self.active = list(shapes) if shapes is not None else sorted(rng.choice(len(SHAPES), self.object_count, replace=False).tolist())
        colors = rng.choice(list(PALETTE.keys()), len(self.active), replace=False)
        pads = rng.choice(ARM_COUNT, len(self.active), replace=False)
        self.object_colors = {}
        for k, obj in enumerate(self.objects):
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
            if k not in self.active:
                obj.set_world_pose(position=np.array([3.0 + 0.3 * k, 3.0, 0.1]), orientation=np.array([1, 0, 0, 0.0]))
        for slot, k in enumerate(self.active):
            shape = SHAPES[k]
            c = pad_center(int(pads[slot]))
            jitter = rng.uniform(-0.025, 0.025, size=2)
            pos = np.array([c[0] + jitter[0], c[1] + jitter[1], PAD_TOP + shape.rest_half_height + 0.002])
            self.objects[k].set_world_pose(position=pos, orientation=yaw_quat(rng.uniform(-math.pi, math.pi)))
            set_material_color(self.stage, self.object_mats[k], PALETTE[str(colors[slot])])
            self.object_colors[k] = str(colors[slot])
        for robot in self.robots:
            q = np.concatenate([HOME_Q + rng.normal(0, 0.03, 7).astype(np.float32), [GRIP_OPEN, GRIP_OPEN]]).astype(np.float32)
            robot.set_joint_positions(q)
            robot.set_joint_velocities(np.zeros(9, dtype=np.float32))
            robot.apply_action(ArticulationAction(joint_positions=q))
        for _ in range(30):
            self.world.step(render=False)
        from pxr import Gf, UsdGeom
        for cam in self.station_cams + self.wrist_cams:
            UsdGeom.Camera(self.stage.GetPrimAtPath(cam.prim_path)).GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 8.0))
        for i in range(60):  # render until every policy camera delivers a real (non-black) image
            self.world.render()
            if not self.station_cams:
                if i >= 5:
                    break
                continue
            st, wr = self.images()
            if i >= 5 and min(x.mean() for x in st + wr) > 3:
                break
        self.sim_time = 0.0

    def control_step(self, arm_q_targets, grip_close, render=True):
        """Apply one 15 Hz control command to every arm, then integrate physics."""
        for i, robot in enumerate(self.robots):
            finger = 0.0 if grip_close[i] else GRIP_OPEN
            target = np.concatenate([arm_q_targets[i], [finger, finger]]).astype(np.float32)
            robot.apply_action(ArticulationAction(joint_positions=target))
        for _ in range(SUBSTEPS):
            self.world.step(render=False)
        self.sim_time += SUBSTEPS * PHYSICS_DT
        if render:
            self.world.render()

    # ------------------------------------------------------------------ state
    def arm_state(self, i: int):
        q = np.asarray(self.robots[i].get_joint_positions(), dtype=np.float32)
        return q[:7].copy(), float(q[7] + q[8])

    def hand_pose(self, i: int, q7=None):
        if q7 is None:
            q7, _ = self.arm_state(i)
        pos, rot = self.solvers[i]._kinematics.compute_forward_kinematics("panda_hand", np.asarray(q7, dtype=np.float64))
        return np.asarray(pos, dtype=np.float64).reshape(3), np.asarray(rot, dtype=np.float64).reshape(3, 3)

    def tcp(self, i: int, q7=None):
        pos, rot = self.hand_pose(i, q7)
        return pos + rot[:, 2] * TCP_OFFSET, rot

    def object_state(self, k: int):
        pos, quat = self.objects[k].get_world_pose()
        vel = self.objects[k].get_linear_velocity()
        return np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64), np.asarray(vel, dtype=np.float64)

    @staticmethod
    def _rgb(cam, res):
        img = np.asarray(cam.get_rgba())
        if img.ndim != 3 or img.shape[0] != res or img.shape[1] != res:
            return np.zeros((res, res, 3), dtype=np.uint8)
        img = img[..., :3]
        if img.dtype != np.uint8:
            img = np.clip(img * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img)

    def images(self):
        """Policy camera frames; re-render (up to 9 times) if the RTX readback returns an unfinished flat frame."""
        for attempt in range(10):
            st = [self._rgb(c, STATION_RES) for c in self.station_cams]
            wr = [self._rgb(c, WRIST_RES) for c in self.wrist_cams]
            if min(x.astype(np.float32).std() for x in st + wr) > 10:
                break
            self.flat_retries = getattr(self, "flat_retries", 0) + 1
            self.world.render()
        return st, wr

    def solve_ik(self, i: int, hand_position, hand_quat):
        action, ok = self.solvers[i].compute_inverse_kinematics(
            target_position=np.asarray(hand_position, dtype=np.float32),
            target_orientation=np.asarray(hand_quat, dtype=np.float32),
            position_tolerance=0.004,
            orientation_tolerance=0.05,
        )
        if not ok or action is None or action.joint_positions is None:
            return None
        return np.asarray(action.joint_positions, dtype=np.float32).reshape(-1)[:7]

    def ee_to_q(self, i: int, q7, tcp, rot, ee):
        """Fixed actuation layer: arm-frame gripper velocity -> next joint targets (IK)."""
        hand, quat = ee_goal(tcp, rot, ee, arm_theta(i) + math.pi, SUBSTEPS * PHYSICS_DT)
        q_ik = self.solve_ik(i, hand, quat)
        if q_ik is None:
            return np.asarray(q7, dtype=np.float32)
        return np.asarray(q7, dtype=np.float32) + cap(q_ik - q7)
