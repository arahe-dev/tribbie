"""End-effector action space shared by teacher labels and policy execution.

An arm action is (vx, vy, vz, wz): gripper-centre velocity in the arm's base
frame (m/s) and wrist yaw rate (rad/s), with the hand kept pointing down. The
policy outputs this; a fixed IK step turns it into the next joint targets.
"""

from __future__ import annotations

import math

import numpy as np

from sim.common import down_hand_quat, wrap_angle

VMAX = 0.50
WMAX = 2.0
K_POS = 6.0
K_YAW = 6.0
DQ_CAP = 0.15
TCP_OFFSET = 0.1034


def hand_yaw(rot) -> float:
    return math.atan2(rot[1, 0], rot[0, 0])


def ee_label(tcp, rot, target, yaw, base_yaw) -> np.ndarray:
    """Teacher label: proportional move toward (target, yaw), saturated, in the arm frame."""
    v = K_POS * (np.asarray(target, dtype=np.float64) - np.asarray(tcp, dtype=np.float64))
    n = float(np.linalg.norm(v))
    if n > VMAX:
        v *= VMAX / n
    wz = float(np.clip(K_YAW * wrap_angle(yaw - hand_yaw(rot)), -WMAX, WMAX))
    c, s = math.cos(base_yaw), math.sin(base_yaw)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2], wz], dtype=np.float32)


def ee_goal(tcp, rot, ee, base_yaw, dt):
    """Hand pose one control step ahead for an arm-frame action."""
    ee = np.asarray(ee, dtype=np.float64)
    c, s = math.cos(base_yaw), math.sin(base_yaw)
    v = np.array([c * ee[0] - s * ee[1], s * ee[0] + c * ee[1], ee[2]])
    tcp_next = np.asarray(tcp, dtype=np.float64) + v * dt
    yaw_next = hand_yaw(rot) + float(ee[3]) * dt
    return tcp_next + np.array([0.0, 0.0, TCP_OFFSET]), down_hand_quat(yaw_next)


def cap(dq) -> np.ndarray:
    dq = np.asarray(dq, dtype=np.float32)
    peak = float(np.max(np.abs(dq)))
    return dq * (DQ_CAP / peak) if peak > DQ_CAP else dq
