"""Mobile-manipulation course: pick an object from a table, drive through a
randomized obstacle field, drop it into a commanded bin.

Robot: Clearpath Ridgeback + Franka Panda (Isaac Sim asset). The base moves on
Isaac's holonomic base model (prismatic x/y and revolute z "dummy" joints with
position drives), not on simulated wheel friction. The arm, gripper, objects,
obstacles and bins are all PhysX bodies; grasps are friction only.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCapsule, DynamicCuboid, DynamicCylinder, DynamicSphere
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import interface_config_loader
from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path

from relay.scene import PALETTE, SHAPES
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

PHYSICS_DT = 1.0 / 120.0
CONTROL_HZ = 15
SUBSTEPS = int(round(1.0 / (CONTROL_HZ * PHYSICS_DT)))
HEAD_RES = int(os.environ.get("COURSE_HEAD_RES", "96"))
WRIST_RES = int(os.environ.get("COURSE_WRIST_RES", "64"))
TCP_OFFSET = 0.1034
GRIP_OPEN = 0.04
FINGER_KP = float(os.environ.get("COURSE_FINGER_KP", "6000"))
FINGER_KD = float(os.environ.get("COURSE_FINGER_KD", "300"))
ARM_BASE_OFFSET = np.array([0.30, 0.0, 0.280])     # panda_link0 relative to the base frame

TABLE_CENTER = np.array([0.0, 0.88, 0.0])
TABLE_SIZE = np.array([0.80, 0.56, 0.40])           # x, y, height
TABLE_LIP = 0.025
COURSE_SHAPES = [0, 2, 3, 4, 5]                     # cube, ball, bar, puck, capsule (tall cylinder excluded: tips over)
TABLE_TOP = float(TABLE_SIZE[2])
START_POSE = np.array([0.0, -0.04, math.pi / 2])     # x, y, yaw: facing the table
BIN_X = 5.6
BIN_YS = (-1.15, 0.0, 1.15)
BIN_INNER = 0.42
BIN_HEIGHT = 0.30
BIN_COLORS = {"teal": (0.05, 0.55, 0.55), "magenta": (0.80, 0.10, 0.55), "gold": (0.85, 0.65, 0.10)}
BIN_NAMES = list(BIN_COLORS.keys())
ARENA_X = (-1.0, 6.4)
ARENA_Y = (-2.2, 2.2)
OBSTACLE_X = (1.6, 3.5)
OBJECT_COLORS = ["red", "orange", "green", "blue", "purple", "white"]

STOW_Q = np.array([0.0, -1.25, 0.0, -2.75, 0.0, 1.55, 0.785], dtype=np.float32)


@dataclass
class Obstacle:
    kind: str            # "box" | "cyl"
    center: np.ndarray   # x, y
    size: np.ndarray     # box: (sx, sy); cyl: (r, r)
    yaw: float
    height: float


def bin_goal_pose(b: int) -> np.ndarray:
    """Base pose (x, y, yaw) from which the arm drops into bin b."""
    return np.array([BIN_X - 0.92, BIN_YS[b], 0.0])


class CourseScene:
    def __init__(self, video: bool = False, cameras: bool = True, seed: int = 0, max_objects: int = 3, max_obstacles: int = 3,
                 planner_cams: bool = False):
        self.planner_cams = planner_cams
        self.rng = np.random.default_rng(seed)
        self.world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=1.0 / CONTROL_HZ)
        self.stage = self.world.stage
        self.use_cameras = cameras
        self.max_objects = max_objects
        self.max_obstacles = max_obstacles
        self.objects = []
        self.object_mats = []
        self.object_shape = []
        self.active = []            # list of slots in play
        self.colors = {}            # slot -> color name
        self.obstacles = []
        self.base_target = START_POSE.copy()
        self.sim_time = 0.0
        self._build(video)

    # ------------------------------------------------------------------ build
    def _build(self, video):
        stage = self.stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        floor_mat = make_material(stage, "/World/Looks/Floor", (0.33, 0.35, 0.38), roughness=0.6)
        wall_mat = make_material(stage, "/World/Looks/Wall", (0.18, 0.20, 0.24), roughness=0.5)
        table_mat = make_material(stage, "/World/Looks/Table", (0.42, 0.30, 0.20), roughness=0.5)
        self.obstacle_mat = make_material(stage, "/World/Looks/Obstacle", (0.90, 0.45, 0.10), roughness=0.45)
        friction = physics_material(stage, "/World/Physics/HighFriction", 1.2, 1.0, 0.0)
        floor = static_box(stage, "/World/Floor", (2.7, 0.0, -0.01), (9.0, 6.0, 0.02), floor_mat)
        bind_physics(floor, friction)
        # Low perimeter walls (visual + physical boundary).
        cx, cy = 0.5 * (ARENA_X[0] + ARENA_X[1]), 0.5 * (ARENA_Y[0] + ARENA_Y[1])
        lx, ly = ARENA_X[1] - ARENA_X[0], ARENA_Y[1] - ARENA_Y[0]
        for k, (c, s) in enumerate([((cx, ARENA_Y[0], 0.15), (lx, 0.05, 0.30)), ((cx, ARENA_Y[1], 0.15), (lx, 0.05, 0.30)),
                                    ((ARENA_X[0], cy, 0.15), (0.05, ly, 0.30)), ((ARENA_X[1], cy, 0.15), (0.05, ly, 0.30))]):
            static_box(stage, f"/World/Walls/W{k}", c, s, wall_mat)
        table = static_box(stage, "/World/Table", (TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP / 2), TABLE_SIZE, table_mat)
        bind_physics(table, friction)
        # A low lip keeps bumped objects on the table so a failed grasp can be retried.
        lx, ly, t = TABLE_SIZE[0], TABLE_SIZE[1], 0.015
        for k, (dx, dy, sx, sy) in enumerate([(lx / 2 - t / 2, 0, t, ly), (-lx / 2 + t / 2, 0, t, ly), (0, ly / 2 - t / 2, lx, t), (0, -ly / 2 + t / 2, lx, t)]):
            lip = static_box(stage, f"/World/TableLip/L{k}", (TABLE_CENTER[0] + dx, TABLE_CENTER[1] + dy, TABLE_TOP + TABLE_LIP / 2), (sx, sy, TABLE_LIP), wall_mat)
            bind_physics(lip, friction)
        for b, name in enumerate(BIN_NAMES):
            mat = make_material(stage, f"/World/Looks/Bin_{name}", BIN_COLORS[name], roughness=0.4)
            x, y = BIN_X, BIN_YS[b]
            t = 0.025
            h = BIN_INNER / 2 + t / 2
            static_box(stage, f"/World/Bins/{name}/Floor", (x, y, 0.01), (BIN_INNER + 2 * t, BIN_INNER + 2 * t, 0.02), mat)
            for k, (dx, dy, sx, sy) in enumerate([(h, 0, t, BIN_INNER + 2 * t), (-h, 0, t, BIN_INNER + 2 * t),
                                                  (0, h, BIN_INNER, t), (0, -h, BIN_INNER, t)]):
                static_box(stage, f"/World/Bins/{name}/Wall{k}", (x + dx, y + dy, BIN_HEIGHT / 2), (sx, sy, BIN_HEIGHT), mat)
        studio_lights(stage, dome=700.0, key=2200.0, rim=500.0)

        # Obstacle prims are created once and moved per episode.
        self.obstacle_prims = []
        for k in range(self.max_obstacles):
            box = static_box(stage, f"/World/Obstacles/Box{k}", (0, -8 - k, 0.3), (0.4, 0.4, 0.6), self.obstacle_mat)
            cyl = static_cylinder(stage, f"/World/Obstacles/Cyl{k}", (0, -12 - k, 0.3), 0.22, 0.6, self.obstacle_mat)
            self.obstacle_prims.append((box, cyl))

        root = get_assets_root_path()
        add_reference_to_stage(root + "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd", "/World/Robot")
        # The asset's root "world" link is free-floating; anchor it so the base
        # moves only through its x/y/yaw joints (as in Isaac Lab's Ridgeback config).
        anchor = UsdPhysics.FixedJoint.Define(stage, Sdf.Path("/World/Robot/world/RootAnchor"))
        anchor.CreateBody1Rel().SetTargets([Sdf.Path("/World/Robot/world")])
        self.robot = self.world.scene.add(SingleArticulation("/World/Robot", name="robot"))

        for k, shape in enumerate(SHAPES):
            path = f"/World/Objects/Obj_{k}"
            park = np.array([-0.6 - 0.3 * k, -1.9, 0.1])
            if shape.kind == "cuboid":
                obj = DynamicCuboid(prim_path=path, name=f"obj_{k}", position=park, scale=np.array(shape.dims), size=1.0, mass=shape.mass)
            elif shape.kind == "cylinder":
                obj = DynamicCylinder(prim_path=path, name=f"obj_{k}", position=park, radius=shape.dims[0], height=shape.dims[1], mass=shape.mass)
            elif shape.kind == "sphere":
                obj = DynamicSphere(prim_path=path, name=f"obj_{k}", position=park, radius=shape.dims[0], mass=shape.mass)
            else:
                obj = DynamicCapsule(prim_path=path, name=f"obj_{k}", position=park, radius=shape.dims[0], height=shape.dims[1], mass=shape.mass)
                UsdGeom.Capsule(stage.GetPrimAtPath(path)).GetAxisAttr().Set(UsdGeom.Tokens.x)
            self.world.scene.add(obj)
            look = make_material(stage, f"/World/Looks/Obj_{k}", PALETTE["white"], roughness=0.45)
            bind(stage.GetPrimAtPath(path), look)
            bind_physics(stage.GetPrimAtPath(path), friction)
            self.objects.append(obj)
            self.object_mats.append(f"/World/Looks/Obj_{k}")
            self.object_shape.append(k)
        self._friction = friction
        configure_physics_scene(self.world)

        self.head_cam = self.wrist_cam = None
        if self.use_cameras:
            base_path = "/World/Robot/base_link"
            head_local = np.array([0.62, 0.0, 0.46])
            self.head_cam = Camera(prim_path=f"{base_path}/HeadCam", name="head", translation=head_local.astype(np.float32),
                                   orientation=look_at_quat(head_local, np.array([2.2, 0.0, 0.0])), resolution=(HEAD_RES, HEAD_RES))
            wrist_local = np.array([0.075, 0.0, 0.025])
            self.wrist_cam = Camera(prim_path="/World/Robot/panda_hand/WristCam", name="wrist", translation=wrist_local.astype(np.float32),
                                    orientation=look_at_quat(wrist_local, np.array([0.0, 0.0, 0.13]), up_hint=(1.0, 0.0, 0.0)),
                                    resolution=(WRIST_RES, WRIST_RES))
        self.video_cam = None
        if video:
            self.video_cam = Camera(prim_path="/World/VideoCam", name="video", position=np.array([2.6, -5.2, 4.2], dtype=np.float32),
                                    orientation=look_at_quat([2.6, -5.2, 4.2], [2.6, 0.0, 0.0]), resolution=(1280, 720))

        self.table_cam = self.bins_cam = None
        if self.planner_cams:
            tpos = np.array([0.0, 0.62, 1.55])
            self.table_cam = Camera(prim_path="/World/PlannerCams/Table", name="table_cam", position=tpos.astype(np.float32),
                                    orientation=look_at_quat(tpos, [0.0, 0.85, TABLE_TOP], up_hint=(0.0, 1.0, 0.0)), resolution=(480, 360))
            bpos = np.array([BIN_X - 1.6, 0.0, 2.4])
            self.bins_cam = Camera(prim_path="/World/PlannerCams/Bins", name="bins_cam", position=bpos.astype(np.float32),
                                   orientation=look_at_quat(bpos, [BIN_X, 0.0, 0.1]), resolution=(480, 360))
        self.world.reset()
        names = list(self.robot.dof_names)
        self.idx_arm = np.array([names.index(f"panda_joint{j}") for j in range(1, 8)])
        self.idx_base = np.array([names.index(n) for n in ("dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint")])
        self.idx_fingers = np.array([names.index("panda_finger_joint1"), names.index("panda_finger_joint2")])
        self.n_dof = len(names)
        kp, kd = self.robot.get_articulation_controller().get_gains()
        kp, kd = np.asarray(kp, dtype=np.float32).copy(), np.asarray(kd, dtype=np.float32).copy()
        kp[self.idx_fingers], kd[self.idx_fingers] = FINGER_KP, FINGER_KD
        kp[self.idx_base], kd[self.idx_base] = 2.0e6, 2.0e5
        self.robot.get_articulation_controller().set_gains(kp, kd)
        if self.head_cam is not None:
            for cam in (self.head_cam, self.wrist_cam):
                cam.initialize()
            self.head_cam.set_focal_length(1.4)
            self.head_cam.set_clipping_range(0.02, 30.0)
            self.wrist_cam.set_focal_length(1.6)
            self.wrist_cam.set_clipping_range(0.01, 6.0)
        if self.video_cam is not None:
            self.video_cam.initialize()
            self.video_cam.set_focal_length(1.8)
            self.video_cam.set_clipping_range(0.05, 60.0)
        for cam in (self.table_cam, self.bins_cam):
            if cam is not None:
                cam.initialize()
                cam.set_focal_length(1.8)
                cam.set_clipping_range(0.05, 30.0)
        cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self.kin = LulaKinematicsSolver(**cfg)

    # ------------------------------------------------------------------ episode
    def _place_obstacles(self, rng, count):
        self.obstacles = []
        for k in range(self.max_obstacles):
            box, cyl = self.obstacle_prims[k]
            for prim in (box, cyl):
                _set_translate(prim, (0.0, -8.0 - 2 * k - (0 if prim is box else 1), 0.3))
        placed = 0
        tries = 0
        while placed < count and tries < 400:
            tries += 1
            x = rng.uniform(*OBSTACLE_X)
            y = rng.uniform(-1.3, 1.3)
            if any(np.linalg.norm(np.array([x, y]) - o.center) < 2.1 for o in self.obstacles):
                continue
            if rng.random() < 0.5:
                size = rng.uniform(0.30, 0.55, size=2)
                yaw = rng.uniform(0, math.pi)
                height = rng.uniform(0.35, 0.8)
                self.obstacles.append(Obstacle("box", np.array([x, y]), size, yaw, height))
                prim = self.obstacle_prims[placed][0]
                _set_box(prim, (x, y, height / 2), (size[0], size[1], height), yaw)
            else:
                r = rng.uniform(0.15, 0.28)
                height = rng.uniform(0.35, 0.8)
                self.obstacles.append(Obstacle("cyl", np.array([x, y]), np.array([r, r]), 0.0, height))
                prim = self.obstacle_prims[placed][1]
                _set_cyl(prim, (x, y, height / 2), r, height)
            placed += 1

    def reset(self, seed=None, n_objects=None, n_obstacles=None, target_slot=None, target_bin=None, shapes=None,
              colors=None, start_bin=None, prefill=0, fixed=False):
        """start_bin: start parked at that bin's drop pose (as after a delivery).
        prefill: how many non-target objects already sit in random bins."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng = self.rng
        n_objects = int(n_objects if n_objects is not None else rng.integers(1, self.max_objects + 1))
        n_obstacles = int(n_obstacles if n_obstacles is not None else rng.integers(2, self.max_obstacles + 1))
        if fixed:
            # Narrow "demo world": one obstacle layout, three fixed object slots.
            self._place_obstacles(np.random.default_rng(424242), 2)
            n_objects, shapes, colors = 3, [0, 2, 4], ["red", "blue", "green"]
        else:
            self._place_obstacles(rng, n_obstacles)
        shape_ids = list(shapes) if shapes is not None else rng.choice(COURSE_SHAPES, n_objects, replace=False).tolist()
        colors = list(colors) if colors is not None else rng.choice(OBJECT_COLORS, n_objects, replace=False).tolist()
        self.active = [int(k) for k in shape_ids]
        self.colors = {}
        for k, color in zip(self.active, colors):
            set_material_color(self.stage, self.object_mats[k], PALETTE[str(color)])
            self.colors[k] = str(color)
        self.target_slot = int(target_slot) if target_slot is not None else int(rng.choice(self.active))
        self.target_bin = int(target_bin) if target_bin is not None else int(rng.integers(0, len(BIN_NAMES)))
        others = [k for k in self.active if k != self.target_slot]
        prefilled = list(rng.choice(others, min(int(prefill), len(others)), replace=False)) if prefill else []
        on_table = [k for k in self.active if k not in prefilled]
        xs = np.linspace(-0.20, 0.20, len(on_table)) if len(on_table) > 1 else np.array([0.0])
        xs = xs + rng.uniform(-0.03, 0.03, len(on_table))
        rng.shuffle(xs)
        if fixed:
            slot_x = {0: -0.18, 2: 0.0, 4: 0.18}
            xs = np.array([slot_x[k] + rng.uniform(-0.015, 0.015) for k in on_table])
        self.world.reset()
        self._apply_gains()
        for k, obj in enumerate(self.objects):
            shape = SHAPES[k]
            if k in on_table:
                x = xs[on_table.index(k)]
                pos = np.array([TABLE_CENTER[0] + x, TABLE_CENTER[1] - 0.10 + rng.uniform(-0.015 if fixed else -0.04, 0.015 if fixed else 0.04), TABLE_TOP + shape.rest_half_height + 0.003])
                obj.set_world_pose(position=pos, orientation=yaw_quat(rng.uniform(-0.3, 0.3) if fixed else rng.uniform(-math.pi, math.pi)))
            elif k in prefilled:
                b = int(rng.integers(0, len(BIN_NAMES)))
                pos = np.array([BIN_X + rng.uniform(-0.08, 0.08), BIN_YS[b] + rng.uniform(-0.08, 0.08), 0.02 + shape.rest_half_height + 0.01])
                obj.set_world_pose(position=pos, orientation=yaw_quat(rng.uniform(-math.pi, math.pi)))
            else:
                obj.set_world_pose(position=np.array([-0.6 - 0.3 * k, -1.9, 0.1]), orientation=np.array([1.0, 0, 0, 0]))
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        if start_bin is None:
            start = START_POSE + np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), rng.uniform(-0.05, 0.05)])
        else:
            start = bin_goal_pose(int(start_bin)) + np.array([rng.uniform(-0.12, 0.05), rng.uniform(-0.1, 0.1), rng.uniform(-0.25, 0.25)])
        q = np.zeros(self.n_dof, dtype=np.float32)
        q[self.idx_base] = start
        q[self.idx_arm] = STOW_Q + rng.normal(0, 0.03, 7).astype(np.float32)
        q[self.idx_fingers] = GRIP_OPEN
        self.robot.set_joint_positions(q)
        self.robot.set_joint_velocities(np.zeros(self.n_dof, dtype=np.float32))
        self.robot.apply_action(ArticulationAction(joint_positions=q))
        self.base_target = start.copy()
        for _ in range(40):
            self.world.step(render=False)
        self._warm_cameras()
        self.sim_time = 0.0

    def _force_clipping(self):
        """Write the near/far clip straight onto the USD camera prims (the default near
        plane of ~1 m otherwise hides the floor, table and objects in front of the robot)."""
        from pxr import Gf, UsdGeom
        for path, near, far in (("/World/Robot/base_link/HeadCam", 0.02, 30.0), ("/World/Robot/panda_hand/WristCam", 0.01, 6.0),
                                ("/World/PlannerCams/Table", 0.05, 30.0), ("/World/PlannerCams/Bins", 0.05, 30.0), ("/World/VideoCam", 0.05, 60.0)):
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid():
                UsdGeom.Camera(prim).GetClippingRangeAttr().Set(Gf.Vec2f(near, far))

    def _warm_cameras(self, max_frames=60):
        """Render until both policy cameras deliver real (non-black) images."""
        self._force_clipping()
        for i in range(max_frames):
            self.world.render()
            if self.head_cam is None:
                if i >= 5:
                    return
                continue
            head, wrist = self.images()
            if i >= 5 and head.mean() > 3 and wrist.mean() > 3:
                return

    def _apply_gains(self):
        kp, kd = self.robot.get_articulation_controller().get_gains()
        kp, kd = np.asarray(kp, dtype=np.float32).copy(), np.asarray(kd, dtype=np.float32).copy()
        kp[self.idx_fingers], kd[self.idx_fingers] = FINGER_KP, FINGER_KD
        kp[self.idx_base], kd[self.idx_base] = 2.0e6, 2.0e5
        self.robot.get_articulation_controller().set_gains(kp, kd)

    def set_command(self, slot: int, bin_index: int):
        self.target_slot, self.target_bin = int(slot), int(bin_index)

    # ------------------------------------------------------------------ control
    def control_step(self, base_vel, arm_q_target, close, render=True):
        """base_vel: (vx, vy, wz) in the robot frame (m/s, rad/s)."""
        dt = 1.0 / CONTROL_HZ
        x, y, yaw = self.base_target
        vx, vy, wz = [float(v) for v in base_vel]
        c, s = math.cos(yaw), math.sin(yaw)
        self.base_target = np.array([x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, yaw + wz * dt])
        # Keep the commanded pose from running away from the measured pose (e.g. when blocked).
        measured = self.base_pose()
        lead = self.base_target[:2] - measured[:2]
        if np.linalg.norm(lead) > 0.15:
            self.base_target[:2] = measured[:2] + lead / np.linalg.norm(lead) * 0.15
        q = np.zeros(self.n_dof, dtype=np.float32)
        q[self.idx_base] = self.base_target
        q[self.idx_arm] = arm_q_target
        q[self.idx_fingers] = 0.0 if close else GRIP_OPEN
        self.robot.apply_action(ArticulationAction(joint_positions=q))
        for _ in range(SUBSTEPS):
            self.world.step(render=False)
        self.sim_time += SUBSTEPS * PHYSICS_DT
        if render:
            self.world.render()

    # ------------------------------------------------------------------ state
    def joint_state(self):
        q = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        v = np.asarray(self.robot.get_joint_velocities(), dtype=np.float64)
        return q, v

    def base_pose(self):
        q, _ = self.joint_state()
        return q[self.idx_base].copy()

    def base_velocity_local(self):
        q, v = self.joint_state()
        yaw = q[self.idx_base][2]
        vx, vy, wz = v[self.idx_base]
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * vx + s * vy, -s * vx + c * vy, wz])

    def arm_q(self):
        q, _ = self.joint_state()
        return q[self.idx_arm].astype(np.float32), float(q[self.idx_fingers].sum())

    def arm_base_world(self, pose=None):
        x, y, yaw = self.base_pose() if pose is None else pose
        c, s = math.cos(yaw), math.sin(yaw)
        p = np.array([x + c * ARM_BASE_OFFSET[0] - s * ARM_BASE_OFFSET[1], y + s * ARM_BASE_OFFSET[0] + c * ARM_BASE_OFFSET[1], ARM_BASE_OFFSET[2]])
        return p, yaw_quat(yaw)

    def tcp(self, q7=None, pose=None):
        if q7 is None:
            q7, _ = self.arm_q()
        p, quat = self.arm_base_world(pose)
        self.kin.set_robot_base_pose(p, quat)
        pos, rot = self.kin.compute_forward_kinematics("panda_hand", np.asarray(q7, dtype=np.float64))
        pos, rot = np.asarray(pos, dtype=np.float64).reshape(3), np.asarray(rot, dtype=np.float64).reshape(3, 3)
        return pos + rot[:, 2] * TCP_OFFSET, rot

    def solve_ik(self, hand_position, hand_quat, warm_start, pose=None):
        p, quat = self.arm_base_world(pose)
        self.kin.set_robot_base_pose(p, quat)
        q, ok = self.kin.compute_inverse_kinematics("panda_hand", np.asarray(hand_position, dtype=np.float64),
                                                    np.asarray(hand_quat, dtype=np.float64),
                                                    warm_start=np.asarray(warm_start, dtype=np.float64),
                                                    position_tolerance=0.004, orientation_tolerance=0.05)
        if not ok:
            return None
        return np.asarray(q, dtype=np.float32)[:7]

    def ee_to_q(self, q7, tcp, rot, ee, pose):
        """Fixed actuation layer: arm-frame gripper velocity -> next arm joint targets (IK)."""
        hand, quat = ee_goal(tcp, rot, ee, pose[2], SUBSTEPS * PHYSICS_DT)
        q_ik = self.solve_ik(hand, quat, q7, pose=pose)
        if q_ik is None:
            return np.asarray(q7, dtype=np.float32)
        return np.asarray(q7, dtype=np.float32) + cap(q_ik - q7)

    def object_state(self, slot):
        pos, quat = self.objects[slot].get_world_pose()
        vel = self.objects[slot].get_linear_velocity()
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
        """Policy camera frames. The RTX readback occasionally returns an unfinished
        (flat, sky-only) frame; re-render (up to 9 times) until the frame has content."""
        for attempt in range(10):
            head, wrist = self._rgb(self.head_cam, HEAD_RES), self._rgb(self.wrist_cam, WRIST_RES)
            if head.astype(np.float32).std() > 10:   # only the head camera suffers unfinished frames; a stowed wrist legitimately sees flat surfaces
                break
            self.flat_retries = getattr(self, "flat_retries", 0) + 1
            self.world.render()
        return head, wrist

    @staticmethod
    def frame(cam):
        img = np.asarray(cam.get_rgba())
        if img.ndim != 3 or img.size == 0:
            return None
        img = img[..., :3]
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img)


def UsdShadeMaterial(stage, path):
    from pxr import UsdShade
    return UsdShade.Material(stage.GetPrimAtPath(path))


def _xform_ops(prim):
    xf = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    return xf, ops


def _set_translate(prim, t):
    xf, ops = _xform_ops(prim)
    ops["xformOp:translate"].Set(Gf.Vec3d(*[float(v) for v in t]))


def _set_box(prim, center, size, yaw):
    xf, ops = _xform_ops(prim)
    ops["xformOp:translate"].Set(Gf.Vec3d(*[float(v) for v in center]))
    if "xformOp:rotateZ" not in ops:
        xf.ClearXformOpOrder()
        t = xf.AddTranslateOp()
        r = xf.AddRotateZOp()
        s = xf.AddScaleOp()
        t.Set(Gf.Vec3d(*[float(v) for v in center]))
        r.Set(math.degrees(yaw))
        s.Set(Gf.Vec3f(*[float(v) for v in size]))
        return
    ops["xformOp:rotateZ"].Set(math.degrees(yaw))
    ops["xformOp:scale"].Set(Gf.Vec3f(*[float(v) for v in size]))


def _set_cyl(prim, center, radius, height):
    _set_translate(prim, center)
    cyl = UsdGeom.Cylinder(prim)
    cyl.GetRadiusAttr().Set(float(radius))
    cyl.GetHeightAttr().Set(float(height))
