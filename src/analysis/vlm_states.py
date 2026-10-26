"""Render scene states with known object locations for the VLM scene-verification test.

Each state places the three objects (red cube, blue ball, green puck) at random
locations: on the table, inside one of the three bins, or on the floor. The table,
bins and overview cameras are saved with the PhysX ground truth.
"""

import json
import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

from course.scene import BIN_NAMES, BIN_X, BIN_YS, TABLE_CENTER, TABLE_TOP, CourseScene
from course.skills import SkillWorld
from relay.scene import SHAPES
from sim.common import yaw_quat

OUT = ROOT / "results" / "vlm_states"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # The dedicated planner cameras rendered black in this build, so one working 1280x720
    # camera is moved between three viewpoints instead.
    scene = CourseScene(video=True, planner_cams=False, seed=5)
    from sim.common import look_at_quat
    VIEWS = {"overview": ([2.6, -5.2, 4.2], [2.6, 0.0, 0.0]),
             "table": ([0.0, 0.25, 1.7], [0.0, 0.88, 0.40]),
             "bins": ([BIN_X - 2.2, 0.0, 2.6], [BIN_X, 0.0, 0.1])}
    rng = np.random.default_rng(77)
    meta = []
    for i in range(N):
        scene.reset(seed=1000 + i, fixed=True)
        truth = {}
        for k in scene.active:
            where = rng.choice(["table", "teal", "magenta", "gold", "floor"], p=[0.4, 0.15, 0.15, 0.15, 0.15])
            shape = SHAPES[k]
            if where == "table":
                pos = np.array([TABLE_CENTER[0] + rng.uniform(-0.28, 0.28), TABLE_CENTER[1] + rng.uniform(-0.18, 0.12), TABLE_TOP + shape.rest_half_height + 0.003])
            elif where == "floor":
                pos = np.array([rng.uniform(1.0, 4.4), rng.uniform(-1.6, 1.6), shape.rest_half_height + 0.003])
            else:
                b = BIN_NAMES.index(where)
                pos = np.array([BIN_X + rng.uniform(-0.1, 0.1), BIN_YS[b] + rng.uniform(-0.1, 0.1), 0.02 + shape.rest_half_height + 0.01])
            scene.objects[k].set_world_pose(position=pos, orientation=yaw_quat(rng.uniform(-3, 3)))
        for _ in range(30):
            scene.world.step(render=False)
        for _ in range(12):
            scene.world.render()
        for k in scene.active:
            p = scene.object_state(k)[0]
            truth[scene.colors[k]] = SkillWorld.in_bin(p) or ("table" if SkillWorld.on_table(p) else ("floor" if p[2] < 0.2 else "other"))
        files = {}
        for name, (pos, tgt) in VIEWS.items():
            scene.video_cam.set_world_pose(position=np.array(pos, dtype=np.float32), orientation=look_at_quat(pos, tgt), camera_axes="world")
            img = None
            for _ in range(16):
                scene.world.render()
                img = scene.frame(scene.video_cam)
                if img is not None and img.std() > 10:
                    break
            for _ in range(4):
                scene.world.render()
            img = scene.frame(scene.video_cam)
            path = OUT / f"state{i:02d}_{name}.png"
            Image.fromarray(img).save(path)
            files[name] = path.name
        meta.append({"state": i, "truth": truth, "files": files})
        print("STATE", i, truth, flush=True)
    (OUT / "states.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
finally:
    app.close()
