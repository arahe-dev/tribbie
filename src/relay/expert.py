"""Privileged teacher for the relay: it reads true object/arm state from PhysX.

The teacher only labels data. It never drives the robots in a demo run; there
the camera-only student is in control. Its decisions are (almost) stateless
functions of the current simulator state, so it can relabel any state the
student visits (DAgger).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sim.common import down_hand_quat, wrap_angle, yaw_of_quat
from sim.ee import ee_label
from relay.scene import (
    ARM_COUNT,
    HOME_Q,
    PAD_TOP,
    SHAPES,
    TCP_OFFSET,
    arm_theta,
    dest_pad,
    pad_center,
    source_pad,
)

PHASES = ["home", "approach", "descend", "close", "lift", "carry", "wait", "place", "release", "retreat"]
PHASE_ID = {name: i for i, name in enumerate(PHASES)}
DQ_MAX = 0.09            # rad per 15 Hz control step (1.35 rad/s)
CARRY_Z = 0.24
PRE_GRASP = 0.11
NEAR_PAD = 0.20


@dataclass
class StepContext:
    q: list = field(default_factory=list)
    width: list = field(default_factory=list)
    tcp: list = field(default_factory=list)
    rot: list = field(default_factory=list)
    objects: dict = field(default_factory=dict)     # k -> (pos, quat, vel)
    held: list = field(default_factory=list)        # arm -> object index or None


def gather(scene) -> StepContext:
    ctx = StepContext()
    for i in range(ARM_COUNT):
        q, w = scene.arm_state(i)
        tcp, rot = scene.tcp(i, q)
        ctx.q.append(q)
        ctx.width.append(w)
        ctx.tcp.append(tcp)
        ctx.rot.append(rot)
    for k in scene.active:
        ctx.objects[k] = scene.object_state(k)
    for i in range(ARM_COUNT):
        ctx.held.append(held_object(ctx, i))
    return ctx


def held_object(ctx: StepContext, i: int):
    tcp, rot, width = ctx.tcp[i], ctx.rot[i], ctx.width[i]
    for k, (pos, _, _) in ctx.objects.items():
        shape = SHAPES[k]
        d = rot.T @ (pos + np.array([0, 0, shape.grasp_dz]) - tcp)
        if abs(d[2]) < 0.035 and abs(d[1]) < 0.5 * shape.grasp_width + 0.012 and abs(d[0]) < 0.04:
            if 0.004 < width < shape.grasp_width + 0.003:
                return k
    return None


def object_on_pad(ctx: StepContext, k: int, j: int) -> bool:
    pos, _, vel = ctx.objects[k]
    c = pad_center(j)
    shape = SHAPES[k]
    return (np.linalg.norm(pos[:2] - c[:2]) < 0.085
            and pos[2] < PAD_TOP + shape.rest_half_height + 0.015
            and np.linalg.norm(vel) < 0.08)


def arm_near_pad(ctx: StepContext, arm: int, j: int) -> bool:
    return float(np.linalg.norm(ctx.tcp[arm][:2] - pad_center(j)[:2])) < NEAR_PAD


def hand_yaw(rot) -> float:
    return math.atan2(rot[1, 0], rot[0, 0])


class RelayExpert:
    def __init__(self, scene):
        self.scene = scene
        self.home_tcp = []
        self.neutral_yaw = []
        for i in range(ARM_COUNT):
            tcp, rot = scene.tcp(i, HOME_Q)
            self.home_tcp.append(tcp)
            self.neutral_yaw.append(hand_yaw(rot))
        self.ik_failures = 0
        self.debug = [None] * ARM_COUNT
        # Comfortable straight-down hand yaw for each arm over each of its two
        # trays: the yaw whose IK solution stays closest to the home posture.
        self.comfort = [dict() for _ in range(ARM_COUNT)]
        for i in range(ARM_COUNT):
            for j in (source_pad(i), dest_pad(i)):
                self.comfort[i][j] = self._comfort_yaw(i, j)

    def _comfort_yaw(self, i: int, j: int) -> float:
        kin = self.scene.solvers[i]._kinematics
        pos = pad_center(j).astype(np.float64) + np.array([0.0, 0.0, 0.15 + TCP_OFFSET])
        best, best_cost = self.neutral_yaw[i], float("inf")
        for deg in range(-180, 180, 10):
            yaw = math.radians(deg)
            q, ok = kin.compute_inverse_kinematics("panda_hand", pos, down_hand_quat(yaw), warm_start=HOME_Q.astype(np.float64),
                                                   position_tolerance=0.004, orientation_tolerance=0.05)
            if not ok:
                continue
            q = np.asarray(q, dtype=np.float64)[:7]
            cost = float(np.linalg.norm(q - HOME_Q)) + 2.0 * abs(q[6] - HOME_Q[6])
            if cost < best_cost:
                best, best_cost = yaw, cost
        return best

    def grasp_yaw(self, i: int, k: int, quat, pad: int) -> float:
        shape = SHAPES[k]
        ref = self.comfort[i].get(pad, self.neutral_yaw[i])
        if shape.symmetry <= 0.0:
            return ref
        phi = yaw_of_quat(quat)
        best, best_err = phi, 1e9
        for n in range(-4, 5):
            cand = phi + n * shape.symmetry
            err = abs(wrap_angle(cand - ref))
            if err < best_err:
                best, best_err = cand, err
        return ref + wrap_angle(best - ref)

    def plan(self, ctx: StepContext, i: int):
        """Return (tcp_target, yaw, close_gripper, phase)."""
        tcp, rot, width = ctx.tcp[i], ctx.rot[i], ctx.width[i]
        src, dst = source_pad(i), dest_pad(i)
        prev_arm, next_arm = (i - 1) % ARM_COUNT, (i + 1) % ARM_COUNT
        cur_yaw = hand_yaw(rot)
        held = ctx.held[i]
        others_on_dst = any(object_on_pad(ctx, k, dst) for k in ctx.objects if k != held)
        dst_free = not others_on_dst and not arm_near_pad(ctx, next_arm, dst)

        if held is not None:
            shape = SHAPES[held]
            place = pad_center(dst).astype(np.float64) + np.array([0, 0, shape.rest_half_height + shape.grasp_dz + 0.012])
            place_yaw = self.comfort[i][dst]
            if not dst_free:
                if tcp[2] < CARRY_Z - 0.03:
                    return np.array([tcp[0], tcp[1], CARRY_Z]), cur_yaw, True, "lift"
                wait = 0.5 * (self.home_tcp[i] + place)
                wait[2] = CARRY_Z + 0.06
                return wait, place_yaw, True, "wait"
            dxy = float(np.linalg.norm(tcp[:2] - place[:2]))
            yaw_err = abs(wrap_angle(cur_yaw - place_yaw))
            if tcp[2] < CARRY_Z - 0.03 and dxy > 0.06:
                return np.array([tcp[0], tcp[1], CARRY_Z]), cur_yaw, True, "lift"
            allow = min(CARRY_Z - place[2], max(0.0, 5.0 * (dxy - 0.01) + 0.4 * max(0.0, yaw_err - 0.1)))
            if dxy > 0.02 or yaw_err > 0.25 or tcp[2] > place[2] + 0.012:
                return np.array([place[0], place[1], place[2] + allow]), place_yaw, True, ("carry" if allow > 0.05 else "place")
            return place, place_yaw, False, "release"

        # Not holding anything.
        candidates = [k for k in ctx.objects if object_on_pad(ctx, k, src)]
        src_ok = bool(candidates) and not arm_near_pad(ctx, prev_arm, src) and dst_free
        if src_ok:
            c = pad_center(src)
            k = min(candidates, key=lambda m: np.linalg.norm(ctx.objects[m][0][:2] - c[:2]))
            pos, quat, _ = ctx.objects[k]
            shape = SHAPES[k]
            grasp = pos + np.array([0.0, 0.0, shape.grasp_dz])
            yaw = self.grasp_yaw(i, k, quat, src)
            dxy = float(np.linalg.norm(tcp[:2] - grasp[:2]))
            yaw_err = abs(wrap_angle(cur_yaw - yaw))
            pre = np.array([grasp[0], grasp[1], grasp[2] + PRE_GRASP])
            # Funnel: the allowed height above the grasp point shrinks with the alignment
            # error, so a nearly aligned gripper keeps descending while it corrects.
            if tcp[2] < grasp[2] + 0.07 and dxy > 0.035:
                return np.array([tcp[0], tcp[1], grasp[2] + PRE_GRASP]), cur_yaw, False, "retreat"
            allow = min(PRE_GRASP, max(0.0, 5.0 * (dxy - 0.008) + 0.4 * max(0.0, yaw_err - 0.06)))
            if dxy > 0.015 or yaw_err > 0.15 or tcp[2] > grasp[2] + 0.012:
                target = np.array([grasp[0], grasp[1], grasp[2] + allow])
                return target, yaw, False, ("approach" if allow > 0.05 else "descend")
            if width < shape.grasp_width - 0.004:
                # Fingers closed past the object (a miss) or are still re-opening: open and hold.
                return grasp, yaw, False, "descend"
            return grasp, yaw, True, "close"

        if tcp[2] < 0.15:
            return np.array([tcp[0], tcp[1], CARRY_Z]), cur_yaw, False, "retreat"
        return self.home_tcp[i].copy(), self.neutral_yaw[i], False, "home"

    def act(self, ctx: StepContext, i: int):
        target, yaw, close, phase = self.plan(ctx, i)
        ee = ee_label(ctx.tcp[i], ctx.rot[i], target, yaw, arm_theta(i) + math.pi)
        self.debug[i] = {
            "phase": phase, "tcp": np.round(ctx.tcp[i], 3).tolist(), "target": np.round(np.asarray(target), 3).tolist(),
            "yaw_err": round(float(wrap_angle(hand_yaw(ctx.rot[i]) - yaw)), 3), "width": round(ctx.width[i], 4),
        }
        return ee, bool(close), PHASE_ID[phase]
