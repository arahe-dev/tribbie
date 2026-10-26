"""Skill layer for the mobile manipulator (hierarchical control).

Planner primitives (what Claude Opus calls):
  face(heading)   heading in {north, east, west}: fixed heading controller, NOT learned
  goto(target)    target in {table, teal, magenta, gold}: learned base translation (vx, vy)
  pick(color)     learned arm motion (gripper x/y/z velocity) + gripper close/open
  drop()          learned arm motion + release over the bin in front of the robot

The learned policy never outputs rotation: base heading and wrist angle are held
by the actuation layer. This module also holds the privileged teacher for the
learned skills (used only to label training data / DAgger).
"""

from __future__ import annotations

import math

import numpy as np

from course.expert import NavField, ROBOT_HALF
from course.scene import (
    BIN_HEIGHT,
    BIN_INNER,
    BIN_NAMES,
    BIN_X,
    BIN_YS,
    OBJECT_COLORS,
    STOW_Q,
    TABLE_CENTER,
    TABLE_SIZE,
    TABLE_TOP,
    TCP_OFFSET,
)
from relay.scene import SHAPES
from sim.common import down_hand_quat, wrap_angle
from sim.ee import cap, ee_goal, hand_yaw

SKILLS = ["goto", "pick", "drop"]
TARGETS = ["table", "teal", "magenta", "gold"]
HEADINGS = {"north": math.pi / 2, "east": 0.0, "west": math.pi, "south": -math.pi / 2}
PHASES = ["move", "arrived", "approach", "descend", "close", "lift", "stow_held", "picked",
          "extend", "release", "stow_empty", "dropped", "fail"]
PHASE_ID = {p: i for i, p in enumerate(PHASES)}
DONE_PHASES = {"arrived", "picked", "dropped"}
CMD_DIM = len(SKILLS) + len(TARGETS) + len(OBJECT_COLORS)
V_BASE = 0.55
V_ARM = 0.5
K_ARM = 6.0
PRE_GRASP = 0.11
RELEASE_Z = BIN_HEIGHT + 0.32
TABLE_DOCK = np.array([0.0, -0.04])


def dock_xy(target: str) -> np.ndarray:
    if target == "table":
        return TABLE_DOCK.copy()
    b = BIN_NAMES.index(target)
    return np.array([BIN_X - 0.92, BIN_YS[b]])


def command_vector(skill: str, target: str | None = None, color: str | None = None) -> np.ndarray:
    v = np.zeros(CMD_DIM, dtype=np.float32)
    v[SKILLS.index(skill)] = 1.0
    if target is not None:
        v[len(SKILLS) + TARGETS.index(target)] = 1.0
    if color is not None:
        v[len(SKILLS) + len(TARGETS) + OBJECT_COLORS.index(color)] = 1.0
    return v


def arm_label(tcp, target, base_yaw) -> np.ndarray:
    """Gripper velocity label in the arm frame (no wrist rotation term)."""
    v = K_ARM * (np.asarray(target, dtype=np.float64) - np.asarray(tcp, dtype=np.float64))
    n = float(np.linalg.norm(v))
    if n > V_ARM:
        v *= V_ARM / n
    c, s = math.cos(base_yaw), math.sin(base_yaw)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]], dtype=np.float32)


class SkillWorld:
    """Teacher + actuation layer around a CourseScene."""

    def __init__(self, scene):
        self.scene = scene
        self.fields = {}
        self.yaw_table = None
        self.yaw_bin = None

    def new_episode(self):
        s = self.scene
        self.fields = {t: NavField(s.obstacles, dock_xy(t)) for t in TARGETS}
        pose_t = np.array([TABLE_DOCK[0], TABLE_DOCK[1], HEADINGS["north"]])
        self.yaw_table = self._comfort_yaw(TABLE_CENTER + np.array([0.0, -0.10, TABLE_TOP + 0.15]), pose_t)
        pose_b = np.array([BIN_X - 0.92, 0.0, HEADINGS["east"]])
        self.yaw_bin = self._comfort_yaw(np.array([BIN_X, 0.0, RELEASE_Z]), pose_b)
        return all(f.reachable(TABLE_DOCK + np.array([0.0, -0.2])) for f in self.fields.values())

    def _comfort_yaw(self, tcp_point, pose):
        best, best_cost = 0.0, float("inf")
        for deg in range(-180, 180, 10):
            yaw = math.radians(deg)
            q = self.scene.solve_ik(np.asarray(tcp_point) + np.array([0, 0, TCP_OFFSET]), down_hand_quat(yaw), STOW_Q, pose=pose)
            if q is None:
                continue
            cost = float(np.linalg.norm(q - STOW_Q)) + 2.0 * abs(q[6] - STOW_Q[6])
            if cost < best_cost:
                best, best_cost = yaw, cost
        return best

    # ---------------------------------------------------------------- state
    def state(self):
        s = self.scene
        q, width = s.arm_q()
        tcp, rot = s.tcp(q)
        pose = s.base_pose()
        objs = {k: s.object_state(k) for k in s.active}
        return {"q": q, "width": width, "tcp": tcp, "rot": rot, "pose": pose, "objs": objs, "base_vel": s.base_velocity_local()}

    def held(self, st):
        for k, (pos, _, _) in st["objs"].items():
            shape = SHAPES[k]
            d = st["rot"].T @ (pos + np.array([0, 0, shape.grasp_dz]) - st["tcp"])
            half_len = 0.5 * max(shape.dims) if shape.kind == "cuboid" else 0.04
            if abs(d[2]) < 0.05 and abs(d[1]) < 0.5 * shape.grasp_width + 0.015 and abs(d[0]) < half_len + 0.01:
                if 0.004 < st["width"] < shape.grasp_width + 0.004:
                    return k
        return None

    @staticmethod
    def on_table(pos):
        return (abs(pos[0] - TABLE_CENTER[0]) < TABLE_SIZE[0] / 2 and abs(pos[1] - TABLE_CENTER[1]) < TABLE_SIZE[1] / 2
                and TABLE_TOP < pos[2] < TABLE_TOP + 0.08)

    @staticmethod
    def in_bin(pos):
        for b, by in enumerate(BIN_YS):
            if abs(pos[0] - BIN_X) < BIN_INNER / 2 and abs(pos[1] - by) < BIN_INNER / 2 and pos[2] < BIN_HEIGHT:
                return BIN_NAMES[b]
        return None

    def stow_tcp(self, pose):
        tcp, _ = self.scene.tcp(STOW_Q, pose=pose)
        return tcp

    def slot_of(self, color):
        for k, c in self.scene.colors.items():
            if c == color:
                return k
        return None

    # ---------------------------------------------------------------- teacher
    def teacher(self, skill, target, color, st):
        """Labels for a learned skill: (base_v (2,), arm_v (3,), close, phase)."""
        pose, tcp = st["pose"], st["tcp"]
        zero2 = np.zeros(2, np.float32)
        stow = self.stow_tcp(pose)
        held = self.held(st)
        if skill == "goto":
            goal = dock_xy(target)
            d = goal - pose[:2]
            dist = float(np.linalg.norm(d))
            arm = arm_label(tcp, stow, pose[2])
            close = held is not None
            if dist < 0.05:
                return zero2, arm, close, "arrived"
            if dist < 0.18:
                vw = np.clip(1.6 * d, -0.3, 0.3)
            else:
                wp, _ = self.fields[target].waypoint(pose[:2])
                dd = wp - pose[:2]
                vw = dd / max(float(np.linalg.norm(dd)), 1e-6) * min(V_BASE, dist)
            c, s = math.cos(pose[2]), math.sin(pose[2])
            return np.array([c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1]], np.float32), arm, close, "move"

        if skill == "pick":
            k = self.slot_of(color)
            if k is None:
                return zero2, arm_label(tcp, stow, pose[2]), False, "fail"
            pos = st["objs"][k][0]
            if held == k:
                if tcp[2] < TABLE_TOP + 0.22 and np.linalg.norm(tcp[:2] - stow[:2]) > 0.05:
                    return zero2, arm_label(tcp, np.array([tcp[0], tcp[1], TABLE_TOP + 0.28]), pose[2]), True, "lift"
                if np.linalg.norm(tcp - stow) > 0.04:
                    return zero2, arm_label(tcp, stow, pose[2]), True, "stow_held"
                return zero2, np.zeros(3, np.float32), True, "picked"
            if not self.on_table(pos):
                return zero2, arm_label(tcp, stow, pose[2]), False, "fail"
            grasp = pos + np.array([0.0, 0.0, SHAPES[k].grasp_dz])
            dxy = float(np.linalg.norm(tcp[:2] - grasp[:2]))
            if tcp[2] < grasp[2] + 0.07 and dxy > 0.035:
                return zero2, arm_label(tcp, np.array([tcp[0], tcp[1], grasp[2] + PRE_GRASP]), pose[2]), False, "approach"
            allow = min(PRE_GRASP, max(0.0, 5.0 * (dxy - 0.008)))
            if dxy > 0.015 or tcp[2] > grasp[2] + 0.012:
                return zero2, arm_label(tcp, grasp + np.array([0, 0, allow]), pose[2]), False, ("approach" if allow > 0.05 else "descend")
            if st["width"] < SHAPES[k].grasp_width - 0.004:
                return zero2, arm_label(tcp, grasp, pose[2]), False, "descend"
            return zero2, arm_label(tcp, grasp, pose[2]), True, "close"

        # drop
        b = None
        for name, by in zip(BIN_NAMES, BIN_YS):
            if abs(pose[1] - by) < 0.3 and pose[0] > BIN_X - 1.3:
                b = by
        if held is not None:
            if b is None:
                return zero2, arm_label(tcp, stow, pose[2]), True, "fail"
            place = np.array([BIN_X, b, RELEASE_Z])
            if np.linalg.norm(tcp - place) > 0.04:
                return zero2, arm_label(tcp, place, pose[2]), True, "extend"
            return zero2, np.zeros(3, np.float32), False, "release"
        if np.linalg.norm(tcp - stow) > 0.04:
            return zero2, arm_label(tcp, stow, pose[2]), False, "stow_empty"
        return zero2, np.zeros(3, np.float32), False, "dropped"

    # ---------------------------------------------------------------- actuation
    def actuate(self, skill, heading, base_v, arm_v, close, st):
        """Fixed actuation layer: heading hold + IK. Learned outputs only translate."""
        s = self.scene
        pose = st["pose"]
        wz = float(np.clip(2.2 * wrap_angle(heading - pose[2]), -0.9, 0.9))
        base = np.array([base_v[0], base_v[1], wz]) if skill == "goto" else np.array([0.0, 0.0, wz])
        yaw_t = {"pick": self.yaw_table, "drop": self.yaw_bin}.get(skill)
        if yaw_t is None:
            _, stow_rot = s.tcp(STOW_Q, pose=pose)
            yaw_t = hand_yaw(stow_rot)
        w_wrist = float(np.clip(6.0 * wrap_angle(yaw_t - hand_yaw(st["rot"])), -2.0, 2.0))
        ee = np.array([arm_v[0], arm_v[1], arm_v[2], w_wrist], dtype=np.float32)
        q_target = s.ee_to_q(st["q"], st["tcp"], st["rot"], ee, pose)
        s.control_step(base, q_target, bool(close), render=True)

    def face(self, heading, st, close=None):
        """Planner primitive executed by the fixed heading controller (one control step)."""
        pose = st["pose"]
        err = wrap_angle(heading - pose[2])
        wz = float(np.clip(2.2 * err, -0.9, 0.9))
        ee = np.zeros(4, np.float32)
        q_target = self.scene.ee_to_q(st["q"], st["tcp"], st["rot"], ee, pose)
        held = (self.held(st) is not None) if close is None else bool(close)
        self.scene.control_step(np.array([0.0, 0.0, wz]), q_target, held, render=True)
        return abs(err) < 0.04
