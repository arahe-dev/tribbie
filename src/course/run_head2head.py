"""Head-to-head pick experiment: Claude Opus alone vs Claude Opus + the 16-D latent skill.

Same robot, same start state, same cameras, same instruction ("pick up the <color> object").
  direct: Opus sees a side camera and the wrist camera and issues every gripper move itself
          (move dx/dy/dz in cm in the robot frame, open/close), one call at a time.
  skill:  Opus decides what to pick and verifies the result; pick(color) is executed by the
          learned 16-D latent policy at 15 Hz.
Success = the commanded object is held and lifted at least 5 cm above the table (PhysX).
The simulation keeps running in real time while Opus thinks (the robot holds still), so the
recorded frames and the reported times include thinking time.

  python.bat src/course/run_head2head.py --mode direct --trials 4 --frames artifacts/h2h_direct
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("direct", "skill"), required=True)
parser.add_argument("--trials", type=int, default=4)
parser.add_argument("--seed", type=int, default=900)
parser.add_argument("--max-seconds", type=float, default=180.0)
parser.add_argument("--max-calls", type=int, default=14)
parser.add_argument("--checkpoint-pick", default="models/skills_p3.pt")
parser.add_argument("--frames", default="")
parser.add_argument("--report", default="")
args = parser.parse_args()

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw

from course.scene import CONTROL_HZ, TABLE_TOP, CourseScene
from course.skills import DONE_PHASES, HEADINGS, PHASES, SkillWorld, arm_label, command_vector
from relay.scene import SHAPES
from sim import overlay as ov
from sim.common import look_at_quat

EXE = shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude.exe")
ENV = {k: v for k, v in os.environ.items() if not (k.startswith("CLAUDE") or k.startswith("ANTHROPIC"))}
SIDE_POS, SIDE_TGT = [1.15, 0.25, 1.05], [0.0, 0.80, 0.42]
COLORS = ["blue", "red", "green"]

DIRECT_PROMPT = """You control a robot arm (Franka Panda with a parallel gripper) mounted on a mobile base that is parked at a table.
You see two images: a side camera that shows the robot, gripper and table, and the wrist camera that looks down between the gripper fingers
(fingers at the bottom of that image). You move the gripper with one command per turn: a displacement in centimetres in the ROBOT frame
(dx = forward, toward the far edge of the table, away from the robot base; dy = to the robot's left; dz = up), each between -8 and 8,
and the gripper state (open or close). The gripper points straight down. To grasp: centre the open gripper over the object, lower it until
the fingers are around the object, close, then lift. Set done=true once the object is held and lifted, or if you give up."""
DIRECT_SCHEMA = {"type": "object", "properties": {
    "observation": {"type": "string"}, "dx_cm": {"type": "number"}, "dy_cm": {"type": "number"}, "dz_cm": {"type": "number"},
    "gripper": {"type": "string", "enum": ["open", "close"]}, "done": {"type": "boolean"}},
    "required": ["observation", "dx_cm", "dy_cm", "dz_cm", "gripper", "done"], "additionalProperties": False}

SKILL_PROMPT = """You supervise a robot arm parked at a table. It has a learned skill pick(color) that grasps and lifts the object of that
colour on its own in a few seconds. You see a side camera and the wrist camera. Decide whether to call pick(color) or stop (done) because
the commanded object is already held and lifted, or cannot be picked. Verify from the images."""
SKILL_SCHEMA = {"type": "object", "properties": {
    "observation": {"type": "string"}, "action": {"type": "string", "enum": ["pick", "done"]},
    "color": {"type": "string", "enum": ["red", "blue", "green", "none"]}},
    "required": ["observation", "action", "color"], "additionalProperties": False}


def png_b64(img, side=768, nearest=False):
    im = Image.fromarray(img)
    s = side / max(im.size)
    im = im.resize((int(im.width * s), int(im.height * s)), Image.NEAREST if nearest else Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call_opus(prompt, schema, text, side_img, wrist_img):
    content = [{"type": "text", "text": "[side camera]"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64(side_img)}},
               {"type": "text", "text": "[wrist camera, 64x64 upscaled]"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64(wrist_img, 256, True)}},
               {"type": "text", "text": text}]
    msg = {"type": "user", "message": {"role": "user", "content": content}}
    cmd = [EXE, "-p", "--model", "opus", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose", "--tools", "",
           "--strict-mcp-config", "--no-session-persistence", "--append-system-prompt", prompt, "--json-schema", json.dumps(schema), "--effort", "low"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, input=json.dumps(msg) + "\n", capture_output=True, text=True, encoding="utf-8", env=ENV, timeout=240)
    out = None
    for line in p.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            out = ev.get("structured_output")
    return out, time.perf_counter() - t0


class Trial:
    def __init__(self, scene, world, frames_dir, label, color):
        self.scene, self.world, self.frames_dir, self.label, self.color = scene, world, frames_dir, label, color
        self.frame = 0
        self.status = "starting"
        self.calls = 0
        self.t0 = scene.sim_time
        self.last = ""
        self.pinfo = None

    def record(self):
        if self.frames_dir is None:
            return
        base = self.scene.frame(self.scene.video_cam)
        if base is None:
            return
        im = Image.fromarray(base).convert("RGB")
        d = ImageDraw.Draw(im, "RGBA")
        ov.panel(d, (16, 16, 760, 132))
        d.text((32, 24), self.label, font=ov.font(24, True), fill=ov.ACCENT)
        d.text((32, 60), f"task: pick up the {self.color} object", font=ov.font(19), fill=ov.TEXT)
        d.text((32, 92), self.status, font=ov.font(19, True), fill=ov.WARN if "thinking" in self.status else ov.GOOD)
        ov.panel(d, (1280 - 236, 16, 1280 - 16, 120))
        d.text((1280 - 220, 24), f"{self.scene.sim_time - self.t0:5.1f} s", font=ov.font(40, True), fill=ov.TEXT)
        d.text((1280 - 220, 80), f"Opus calls: {self.calls}", font=ov.font(17), fill=ov.DIM)
        if self.last:
            ov.panel(d, (16, 720 - 92, 1280 - 16, 720 - 16))
            for i, line in enumerate(ov.wrap('"' + self.last + '"', 120)[:2]):
                d.text((32, 720 - 84 + 26 * i), line, font=ov.font(17), fill=ov.TEXT)
        im.save(self.frames_dir / f"f_{self.frame:05d}.jpg", quality=92)
        self.frame += 1

    def hold(self, close):
        st = self.world.state()
        self.world.actuate("pick", HEADINGS["north"], np.zeros(2), np.zeros(3), close, st)
        self.record()

    def think(self, fn, close):
        """Run an Opus call in a thread while the simulation keeps running (robot holds)."""
        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(fn)
            t = time.perf_counter()
            s0 = self.scene.sim_time
            while not fut.done():
                self.status = f"Claude thinking... {time.perf_counter() - t:4.1f} s"
                self.hold(close)
                if self.scene.sim_time - s0 > time.perf_counter() - t:   # never run ahead of the real clock
                    time.sleep(0.02)
            wall = time.perf_counter() - t
            # If rendering made the simulation lag, keep holding until simulated waiting time equals the real
            # thinking time, so the on-screen clock reports Claude's true latency.
            while self.scene.sim_time - s0 < wall:
                self.status = f"Claude thinking... {min(self.scene.sim_time - s0, wall):4.1f} s"
                self.hold(close)
            return fut.result()

    def lifted(self):
        k = self.world.slot_of(self.color)
        st = self.world.state()
        pos = st["objs"][k][0]
        return self.world.held(st) == k and pos[2] > TABLE_TOP + 0.05


def run_direct(trial, scene, world):
    close = False
    trial.t_mark, trial.w_mark = scene.sim_time, time.perf_counter()
    history = []
    while trial.calls < args.max_calls and scene.sim_time - trial.t0 < args.max_seconds:
        side = scene.frame(scene.video_cam)
        _, wrist = scene.images()
        text = (f"Task: pick up the {trial.color} object. Gripper is currently {'closed' if close else 'open'}. "
                f"Previous commands: {'; '.join(history[-6:]) or 'none'}. Give the next command.")
        out, secs = trial.think(lambda: call_opus(DIRECT_PROMPT, DIRECT_SCHEMA, text, side, wrist), close)
        trial.calls += 1
        if out is None:
            history.append("(call failed)")
            continue
        trial.last = out["observation"]
        if out["done"]:
            break
        d = np.clip([out["dx_cm"], out["dy_cm"], out["dz_cm"]], -8, 8) / 100.0
        history.append(f"move ({d[0]*100:.0f},{d[1]*100:.0f},{d[2]*100:.0f}) cm, {out['gripper']}")
        st = world.state()
        yaw = st["pose"][2]
        c, s = np.cos(yaw), np.sin(yaw)
        target = st["tcp"] + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])
        target[2] = max(target[2], TABLE_TOP + 0.01)
        trial.status = f"executing Claude's move ({d[0]*100:+.0f}, {d[1]*100:+.0f}, {d[2]*100:+.0f}) cm"
        for _ in range(int(1.6 * CONTROL_HZ)):
            st = world.state()
            if np.linalg.norm(st["tcp"] - target) < 0.004:
                break
            world.actuate("pick", HEADINGS["north"], np.zeros(2), arm_label(st["tcp"], target, st["pose"][2]), close, st)
            trial.record()
        close = out["gripper"] == "close"
        trial.status = f"gripper {'closing' if close else 'opening'}"
        for _ in range(int(0.6 * CONTROL_HZ)):
            trial.hold(close)
        trial.t_mark, trial.w_mark = scene.sim_time, time.perf_counter()
    return close


def run_skill(trial, scene, world, policy):
    close = False
    trial.t_mark, trial.w_mark = scene.sim_time, time.perf_counter()
    for attempt in range(3):
        side = scene.frame(scene.video_cam)
        _, wrist = scene.images()
        text = f"Task: pick up the {trial.color} object. Attempts so far: {attempt}."
        out, secs = trial.think(lambda: call_opus(SKILL_PROMPT, SKILL_SCHEMA, text, side, wrist), close)
        trial.calls += 1
        if out is None:
            continue
        trial.last = out["observation"]
        if out["action"] == "done" or out["color"] == "none":
            break
        cmd = command_vector("pick", None, out["color"])
        policy.reset()
        done = 0
        t0 = scene.sim_time
        while scene.sim_time - t0 < 20.0:
            st = world.state()
            head, wrist = scene.images()
            prop = np.concatenate([st["q"], [st["width"]], st["base_vel"]]).astype(np.float32)
            o = policy.act([head], [wrist], prop[None], cmd[None])
            act = o["act"][0]
            close = bool(o["bin"][0, 0])
            world.actuate("pick", HEADINGS["north"], np.zeros(2), act[2:5], close, st)
            ph = PHASES[int(o["phase"][0])]
            trial.status = f"16-D latent policy: pick({out['color']})  ·  {o['ms_total']:.0f} ms/decision  ·  phase {ph}"
            trial.record()
            done = done + 1 if ph in DONE_PHASES else 0
            if done >= 8:
                break
        trial.t_mark, trial.w_mark = scene.sim_time, time.perf_counter()
    return close


def main():
    scene = CourseScene(video=True, seed=args.seed)
    world = SkillWorld(scene)
    policy = None
    if args.mode == "skill":
        from lm.runtime import StudentRuntime
        policy = StudentRuntime(str(ROOT / args.checkpoint_pick), batch=1)
    results = []
    label = "CLAUDE OPUS ALONE  -  issues every gripper move" if args.mode == "direct" else "CLAUDE OPUS + 16-D LATENT SKILL"
    for i in range(args.trials):
        color = COLORS[i % 3]
        scene.reset(seed=args.seed + i, start_bin=None, prefill=0, fixed=True)
        world.new_episode()
        scene.video_cam.set_world_pose(position=np.array(SIDE_POS, dtype=np.float32), orientation=look_at_quat(SIDE_POS, SIDE_TGT), camera_axes="world")
        for _ in range(8):
            scene.world.render()
        fdir = None
        if args.frames:
            fdir = Path(args.frames) / f"trial{i}"
            fdir.mkdir(parents=True, exist_ok=True)
        trial = Trial(scene, world, fdir, label, color)
        w0 = time.perf_counter()
        (run_direct if args.mode == "direct" else (lambda t, s, w: run_skill(t, s, w, policy)))(trial, scene, world)
        # hold a moment so the final state is visible, then judge from PhysX
        for _ in range(int(1.0 * CONTROL_HZ)):
            trial.hold(world.held(world.state()) is not None)
        ok = bool(trial.lifted())
        rec = {"mode": args.mode, "trial": i, "color": color, "success": ok, "opus_calls": trial.calls,
               "seconds": round(scene.sim_time - trial.t0, 1), "wall_seconds": round(time.perf_counter() - w0, 1), "frames": trial.frame}
        print("TRIAL_JSON " + json.dumps(rec), flush=True)
        results.append(rec)
    if args.report:
        Path(ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / args.report).write_text(json.dumps(results, indent=2), encoding="utf-8")


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    app.close()
