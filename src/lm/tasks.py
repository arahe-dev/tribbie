"""Per-task mapping from episode arrays to student inputs and teacher targets."""

import numpy as np

RELAY_DQ_MAX = 0.09
EE_SCALE = np.array([0.50, 0.50, 0.50, 2.0], dtype=np.float32)   # sim.ee VMAX / WMAX


def _relay_proprio(d):
    return np.concatenate([d["q"], d["width"].reshape(-1, 1)], axis=1)


COURSE_ACT_SCALE = np.concatenate([np.array([0.55, 0.55, 0.9], dtype=np.float32), EE_SCALE])


def _course_proprio(d):
    return np.concatenate([d["q"], d["width"].reshape(-1, 1), d["base_vel"]], axis=1)


TASKS = {
    "relay": {
        "scene": "station",
        "hand": "wrist",
        "proprio": _relay_proprio,
        "action": lambda d: d["ee"],
        "act_scale": EE_SCALE,
        "binary": lambda d: d["close"].astype(np.float32),
        "num_phases": 10,
    },
    "course": {
        "scene": "head",
        "hand": "wrist",
        "proprio": _course_proprio,
        "action": lambda d: np.concatenate([d["base_act"], d["ee"]], axis=1),
        "act_scale": COURSE_ACT_SCALE,
        "binary": lambda d: d["close"].astype(np.float32),
        "command": lambda d: d["cmd"],
        "num_phases": 12,
    },
    "skills": {
        "scene": "head",
        "hand": "wrist",
        "proprio": _course_proprio,
        "action": lambda d: np.concatenate([d["base_act"], d["arm_act"]], axis=1),
        "act_scale": np.array([0.55, 0.55, 0.5, 0.5, 0.5], dtype=np.float32),
        "binary": lambda d: d["close"].astype(np.float32),
        "command": lambda d: d["cmd"],
        "num_phases": 13,
    },
}
