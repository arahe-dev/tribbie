"""Privileged teacher for the mobile-manipulation course (data labels only).

Reads true poses from PhysX and returns, for the current state, the base
velocity, arm joint deltas and gripper command a competent operator would
issue. Decisions depend only on the current state, so any state the student
reaches can be relabelled (DAgger). The teacher never drives a demo run.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from course.scene import (
    ARENA_X,
    ARENA_Y,
    BIN_HEIGHT,
    BIN_INNER,
    BIN_X,
    BIN_YS,
    START_POSE,
    STOW_Q,
    TABLE_CENTER,
    TABLE_SIZE,
    TABLE_TOP,
    TCP_OFFSET,
    bin_goal_pose,
)
from relay.scene import SHAPES
from sim.common import down_hand_quat, wrap_angle, yaw_of_quat
from sim.ee import ee_label

PHASES = ["approach", "descend", "close", "lift", "stow", "drive", "extend", "release", "done", "return", "retreat", "fail"]
PHASE_ID = {p: i for i, p in enumerate(PHASES)}
DQ_MAX = 0.09
V_MAX = 0.55
W_MAX = 0.9
GRID = 0.05
INFLATE = 0.78
ROBOT_HALF = np.array([0.48, 0.40])     # Ridgeback half length / half width
PRE_GRASP = 0.11
RELEASE_Z = BIN_HEIGHT + 0.32


def hand_yaw(rot) -> float:
    return math.atan2(rot[1, 0], rot[0, 0])


class NavField:
    """Dijkstra distance-to-goal over an inflated occupancy grid."""

    def __init__(self, obstacles, goal_xy, extra_blocked=()):
        self.x0, self.y0 = ARENA_X[0], ARENA_Y[0]
        self.nx = int((ARENA_X[1] - ARENA_X[0]) / GRID) + 1
        self.ny = int((ARENA_Y[1] - ARENA_Y[0]) / GRID) + 1
        xs = self.x0 + GRID * np.arange(self.nx)
        ys = self.y0 + GRID * np.arange(self.ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        occ = np.zeros((self.nx, self.ny), dtype=bool)
        wall = 0.03 + INFLATE - 0.08
        occ |= (X < ARENA_X[0] + wall) | (X > ARENA_X[1] - wall) | (Y < ARENA_Y[0] + wall) | (Y > ARENA_Y[1] - wall)
        for o in obstacles:
            occ |= _inside_inflated(o, X, Y, INFLATE)
        # Table and bins (inflated less: the robot docks right next to them).
        occ |= (np.abs(X - TABLE_CENTER[0]) < TABLE_SIZE[0] / 2 + 0.45) & (np.abs(Y - TABLE_CENTER[1]) < TABLE_SIZE[1] / 2 + 0.38)
        for by in BIN_YS:
            occ |= (np.abs(X - BIN_X) < BIN_INNER / 2 + 0.47) & (np.abs(Y - by) < BIN_INNER / 2 + 0.36)
        for (cx, cy, r) in extra_blocked:
            occ |= (X - cx) ** 2 + (Y - cy) ** 2 < r * r
        self.occ = occ
        self.goal = np.asarray(goal_xy, dtype=np.float64)
        gi, gj = self._cell(self.goal)
        occ[max(gi - 2, 0):gi + 3, max(gj - 2, 0):gj + 3] = False
        dist = np.full((self.nx, self.ny), np.inf)
        dist[gi, gj] = 0.0
        heap = [(0.0, gi, gj)]
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
        while heap:
            d, i, j = heapq.heappop(heap)
            if d > dist[i, j]:
                continue
            for di, dj, c in steps:
                a, b = i + di, j + dj
                if 0 <= a < self.nx and 0 <= b < self.ny and not occ[a, b]:
                    nd = d + c * GRID
                    if nd < dist[a, b]:
                        dist[a, b] = nd
                        heapq.heappush(heap, (nd, a, b))
        self.dist = dist

    def _cell(self, p):
        i = int(round((p[0] - self.x0) / GRID))
        j = int(round((p[1] - self.y0) / GRID))
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def reachable(self, p) -> bool:
        i, j = self._cell(p)
        return bool(np.isfinite(self.dist[i, j]))

    def waypoint(self, p, lookahead=9):
        """Follow steepest descent for `lookahead` cells from p's cell."""
        i, j = self._cell(p)
        if not np.isfinite(self.dist[i, j]):
            # Off the free space (too close to something): head to the best nearby free cell.
            best, bi, bj = np.inf, i, j
            for di in range(-8, 9):
                for dj in range(-8, 9):
                    a, b = i + di, j + dj
                    if 0 <= a < self.nx and 0 <= b < self.ny and self.dist[a, b] + 0.3 * math.hypot(di, dj) * GRID < best:
                        best, bi, bj = self.dist[a, b] + 0.3 * math.hypot(di, dj) * GRID, a, b
            i, j = bi, bj
        for _ in range(lookahead):
            best, nb = self.dist[i, j], None
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if (di or dj) and 0 <= a < self.nx and 0 <= b < self.ny and self.dist[a, b] < best:
                        best, nb = self.dist[a, b], (a, b)
            if nb is None:
                break
            i, j = nb
        return np.array([self.x0 + i * GRID, self.y0 + j * GRID]), float(self.dist[i, j])


def _inside_inflated(o, X, Y, r):
    dx, dy = X - o.center[0], Y - o.center[1]
    if o.kind == "cyl":
        return dx * dx + dy * dy < (o.size[0] + r) ** 2
    c, s = math.cos(o.yaw), math.sin(o.yaw)
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    qx = np.maximum(np.abs(lx) - o.size[0] / 2, 0.0)
    qy = np.maximum(np.abs(ly) - o.size[1] / 2, 0.0)
    return qx * qx + qy * qy < r * r


def footprint_hits(pose, obstacles, margin=0.0) -> bool:
    """True if the base rectangle overlaps any obstacle (evaluation only)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for u in np.linspace(-1, 1, 9):
        for v in (-1, 1):
            pts.append((u * ROBOT_HALF[0], v * ROBOT_HALF[1]))
            pts.append((v * ROBOT_HALF[0], u * ROBOT_HALF[1]))
    P = np.array([[x + c * a - s * b, y + s * a + c * b] for a, b in pts])
    for o in obstacles:
        if _inside_inflated(o, P[:, 0], P[:, 1], margin).any():
            return True
        # Obstacle points inside the rectangle (thin features).
        if o.kind == "cyl":
            ang = np.linspace(0, 2 * math.pi, 16, endpoint=False)
            Q = o.center + o.size[0] * np.stack([np.cos(ang), np.sin(ang)], 1)
        else:
            oc, os_ = math.cos(o.yaw), math.sin(o.yaw)
            corners = np.array([[sx * o.size[0] / 2, sy * o.size[1] / 2] for sx in (-1, 1) for sy in (-1, 1)])
            Q = o.center + corners @ np.array([[oc, os_], [-os_, oc]])
        d = Q - np.array([x, y])
        lx, ly = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        if ((np.abs(lx) < ROBOT_HALF[0] + margin) & (np.abs(ly) < ROBOT_HALF[1] + margin)).any():
            return True
    return False


class CourseExpert:
    def __init__(self, scene):
        self.scene = scene
        self.fields = {}
        self.comfort_table = None
        self.comfort_bin = {}
        self.debug = None

    def new_episode(self):
        """Rebuild navigation fields for this episode's obstacles. Returns False if a goal is unreachable."""
        s = self.scene
        self.fields = {}
        ok = True
        for b in range(len(BIN_YS)):
            f = NavField(s.obstacles, bin_goal_pose(b)[:2])
            self.fields[("bin", b)] = f
            ok &= f.reachable(START_POSE[:2] + np.array([0.0, -0.25]))
        self.fields["table"] = NavField(s.obstacles, START_POSE[:2])
        self.comfort_table = self._comfort_yaw(TABLE_CENTER + np.array([0.0, -0.10, TABLE_TOP + 0.15]), START_POSE)
        for b in range(len(BIN_YS)):
            self.comfort_bin[b] = self._comfort_yaw(np.array([BIN_X, BIN_YS[b], RELEASE_Z]), bin_goal_pose(b))
        return ok

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

    # ------------------------------------------------------------------ predicates
    def held_slot(self, objs, tcp, rot, width):
        for slot, (pos, _, _) in objs.items():
            shape = SHAPES[self.scene.object_shape[slot]]
            d = rot.T @ (pos + np.array([0, 0, shape.grasp_dz]) - tcp)
            half_len = 0.5 * max(shape.dims) if shape.kind == "cuboid" else (0.5 * shape.dims[1] + shape.dims[0] if shape.kind == "capsule" else 0.04)
            if abs(d[2]) < 0.05 and abs(d[1]) < 0.5 * shape.grasp_width + 0.015 and abs(d[0]) < half_len + 0.01:
                if 0.004 < width < shape.grasp_width + 0.004:
                    return slot
        return None

    @staticmethod
    def on_table(pos, vel):
        return (abs(pos[0] - TABLE_CENTER[0]) < TABLE_SIZE[0] / 2 and abs(pos[1] - TABLE_CENTER[1]) < TABLE_SIZE[1] / 2
                and TABLE_TOP < pos[2] < TABLE_TOP + 0.08 and np.linalg.norm(vel) < 0.08)

    @staticmethod
    def in_bin(pos):
        for b, by in enumerate(BIN_YS):
            if abs(pos[0] - BIN_X) < BIN_INNER / 2 and abs(pos[1] - by) < BIN_INNER / 2 and pos[2] < BIN_HEIGHT:
                return b
        return None

    # ------------------------------------------------------------------ helpers
    def _nav(self, field, goal_pose, pose):
        pos, yaw = pose[:2], pose[2]
        to_goal = goal_pose[:2] - pos
        dist = float(np.linalg.norm(to_goal))
        if dist < 0.18:
            v_world = np.clip(1.6 * to_goal, -0.3, 0.3)
            desired = goal_pose[2]
        else:
            wp, _ = field.waypoint(pos)
            d = wp - pos
            n = max(float(np.linalg.norm(d)), 1e-6)
            speed = min(V_MAX, 1.0 * dist)
            v_world = d / n * speed
            desired = math.atan2(d[1], d[0]) if dist > 0.6 else goal_pose[2]
        err = wrap_angle(desired - yaw)
        wz = float(np.clip(2.2 * err, -W_MAX, W_MAX))
        v_world = v_world * max(0.25, math.cos(min(abs(err), math.pi / 2)))
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * v_world[0] + s * v_world[1], -s * v_world[0] + c * v_world[1], wz])

    @staticmethod
    def _hold(goal, pose):
        """Gentle holonomic P-correction toward a docking pose (robot frame)."""
        d = goal[:2] - pose[:2]
        v_world = np.clip(1.5 * d, -0.25, 0.25)
        wz = float(np.clip(2.0 * wrap_angle(goal[2] - pose[2]), -0.5, 0.5))
        c, s_ = math.cos(pose[2]), math.sin(pose[2])
        v = np.array([c * v_world[0] + s_ * v_world[1], -s_ * v_world[0] + c * v_world[1], wz])
        v[np.abs(v) < 0.005] = 0.0
        return v

    def _arm_to(self, tcp_target, yaw, q, pose):
        """Arm-frame gripper velocity label toward (tcp_target, yaw)."""
        return ee_label(self._tcp, self._rot, tcp_target, yaw, pose[2]), True

    def _stow_target(self, pose):
        tcp, rot = self.scene.tcp(STOW_Q, pose=pose)
        return tcp, hand_yaw(rot)

    def _stow(self, pose):
        target, yaw = self._stow_target(pose)
        return ee_label(self._tcp, self._rot, target, yaw, pose[2])

    # ------------------------------------------------------------------ policy
    def act(self, state):
        """state: dict(pose, q, width, tcp, rot, objs). Returns (base_vel, dq, close, phase_id)."""
        s = self.scene
        pose, q, width, tcp, rot, objs = state["pose"], state["q"], state["width"], state["tcp"], state["rot"], state["objs"]
        target, goal_bin = s.target_slot, s.target_bin
        zero = np.zeros(3)
        self._tcp, self._rot = tcp, rot
        stow_tcp, _ = self._stow_target(pose)
        stowed = float(np.linalg.norm(tcp - stow_tcp)) < 0.04
        cur_yaw = hand_yaw(rot)
        held = self.held_slot(objs, tcp, rot, width)
        t_pos, t_quat, t_vel = objs[target]
        bin_now = self.in_bin(t_pos)

        def result(base, dq, close, phase, extra=None):
            d_obj = rot.T @ (t_pos - tcp)
            self.debug = {"phase": phase, "pose": np.round(pose, 3).tolist(), "tcp": np.round(tcp, 3).tolist(), "held": held,
                          "obj_in_hand": np.round(d_obj, 3).tolist(), "obj": np.round(t_pos, 3).tolist(), "width": round(float(width), 4), **(extra or {})}
            return base, dq, close, PHASE_ID[phase]

        if bin_now is not None:
            return result(zero, self._stow(pose), False, "done" if bin_now == goal_bin else "fail")

        if held is not None:
            table_err = float(np.linalg.norm(pose[:2] - START_POSE[:2]))
            if table_err < 0.4 and tcp[2] < TABLE_TOP + 0.20:
                dq, _ = self._arm_to(np.array([tcp[0], tcp[1], TABLE_TOP + 0.26]), cur_yaw, q, pose)
                return result(self._hold(START_POSE, pose) if table_err < 0.3 else zero, dq, True, "lift")
            goal = bin_goal_pose(goal_bin)
            perr = float(np.linalg.norm(goal[:2] - pose[:2]))
            yerr = abs(wrap_angle(goal[2] - pose[2]))
            if perr > 0.15 or yerr > 0.25:
                if not stowed and perr > 0.3:
                    return result(zero, self._stow(pose), True, "stow")
                base = self._nav(self.fields[("bin", goal_bin)], goal, pose)
                return result(base, self._stow(pose) if perr > 0.3 else np.zeros(4, np.float32), True, "drive", {"perr": round(perr, 3)})
            # Docked (or nearly): hold/correct the base while the arm works in world coordinates.
            base = self._hold(goal, pose)
            place = np.array([BIN_X, BIN_YS[goal_bin], RELEASE_Z])
            if np.linalg.norm(tcp - place) > 0.05 or abs(wrap_angle(cur_yaw - self.comfort_bin[goal_bin])) > 0.3:
                dq, _ = self._arm_to(place, self.comfort_bin[goal_bin], q, pose)
                return result(base, dq, True, "extend")
            return result(base, np.zeros(4, np.float32), False, "release")

        if self.on_table(t_pos, t_vel):
            perr = float(np.linalg.norm(START_POSE[:2] - pose[:2]))
            yerr = abs(wrap_angle(START_POSE[2] - pose[2]))
            if perr > 0.30 or yerr > 0.35:
                if not stowed:
                    if tcp[2] < TABLE_TOP + 0.15:
                        dq, _ = self._arm_to(np.array([tcp[0], tcp[1], TABLE_TOP + 0.26]), cur_yaw, q, pose)
                        return result(zero, dq, False, "retreat")
                    return result(zero, self._stow(pose), False, "stow")
                return result(self._nav(self.fields["table"], START_POSE, pose), np.zeros(4, np.float32), False, "return")
            base = self._hold(START_POSE, pose)
            if perr > 0.10 or yerr > 0.15:
                # Close to the table but not docked: re-dock first, arm waits above the table.
                dq = self._stow(pose) if tcp[2] < TABLE_TOP + 0.15 else np.zeros(4, np.float32)
                return result(base, dq, False, "return")
            shape = SHAPES[s.object_shape[target]]
            grasp = t_pos + np.array([0.0, 0.0, shape.grasp_dz])
            yaw = self._grasp_yaw(shape, t_quat)
            dxy = float(np.linalg.norm(tcp[:2] - grasp[:2]))
            yaw_err = abs(wrap_angle(cur_yaw - yaw))
            if tcp[2] < grasp[2] + 0.07 and dxy > 0.035:
                dq, _ = self._arm_to(np.array([tcp[0], tcp[1], grasp[2] + PRE_GRASP]), cur_yaw, q, pose)
                return result(base, dq, False, "retreat")
            allow = min(PRE_GRASP, max(0.0, 5.0 * (dxy - 0.008) + 0.4 * max(0.0, yaw_err - 0.06)))
            if dxy > 0.015 or yaw_err > 0.15 or tcp[2] > grasp[2] + 0.012:
                dq, _ = self._arm_to(grasp + np.array([0, 0, allow]), yaw, q, pose)
                return result(base, dq, False, "approach" if allow > 0.05 else "descend", {"dxy": round(dxy, 3), "yaw_err": round(yaw_err, 3)})
            if width < shape.grasp_width - 0.004:
                dq, _ = self._arm_to(grasp, yaw, q, pose)
                return result(base, dq, False, "descend")
            dq, _ = self._arm_to(grasp, yaw, q, pose)
            return result(base, dq, True, "close")

        # Target dropped somewhere unreachable for this teacher: stop safely.
        return result(zero, self._stow(pose), False, "fail")

    def _grasp_yaw(self, shape, quat):
        ref = self.comfort_table
        if shape.symmetry <= 0:
            return ref
        phi = yaw_of_quat(quat)
        best, err = phi, 1e9
        for n in range(-4, 5):
            cand = phi + n * shape.symmetry
            e = abs(wrap_angle(cand - ref))
            if e < err:
                best, err = cand, e
        return ref + wrap_angle(best - ref)
