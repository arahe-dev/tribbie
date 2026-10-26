"""Execute primitive scripts (face / goto / pick / drop) on the mobile manipulator.

Modes: expert (teacher drives learned skills), dagger (per-step mix), student (learned
policy only; skill completion decided by the policy's own phase head).

  python.bat src/course/run_skills.py --mode expert --episodes 40 --save-data data/skills/expert
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("expert", "dagger", "student"), default="expert")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint", default="")
parser.add_argument("--checkpoint-pick", default="", help="specialist model for pick() (default: same as --checkpoint)")
parser.add_argument("--beta", type=float, default=0.0)
parser.add_argument("--noise", type=float, default=0.0)
parser.add_argument("--save-data", default="")
parser.add_argument("--frames", default="")
parser.add_argument("--report", default="")
parser.add_argument("--start-bin-prob", type=float, default=0.3)
parser.add_argument("--deliveries", type=int, default=0, help="0 = random 1..2")
parser.add_argument("--only-pick", action="store_true", help="pick-only episodes from the table dock (focused data)")
parser.add_argument("--only-goto", action="store_true", help="drive-only episodes (table <-> bins) for focused data")
parser.add_argument("--goto-planner", action="store_true", help="goto() by the classical grid path planner instead of the learned policy")
args = parser.parse_args()

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw

from course.scene import BIN_NAMES, CONTROL_HZ, CourseScene
from course.skills import DONE_PHASES, HEADINGS, PHASES, PHASE_ID, SkillWorld, command_vector
from relay.scene import SHAPES
from sim import overlay as ov

TIMEOUT = {"goto": 30.0, "pick": 20.0, "drop": 12.0, "face": 6.0}


def make_script(rng, scene, start_bin):
    colors = [scene.colors[k] for k in scene.active if SkillWorld.on_table(scene.object_state(k)[0])]
    n = args.deliveries or int(rng.integers(1, 3))
    picks = list(rng.choice(colors, min(n, len(colors)), replace=False))
    steps = []
    if args.only_pick:
        return [("pick", str(rng.choice(colors)))], []
    if args.only_goto:
        if start_bin is None:
            return [("face", "east"), ("goto", BIN_NAMES[int(rng.integers(0, 3))])], []
        return [("face", "west"), ("goto", "table")], []
    if start_bin is not None:
        steps += [("face", "west"), ("goto", "table"), ("face", "north")]
    for i, c in enumerate(picks):
        b = BIN_NAMES[int(rng.integers(0, 3))]
        steps += [("pick", c), ("face", "east"), ("goto", b), ("drop", None)]
        if i < len(picks) - 1:
            steps += [("face", "west"), ("goto", "table"), ("face", "north")]
    return steps, picks


def main():
    scene = CourseScene(video=bool(args.frames), seed=args.seed)
    world = SkillWorld(scene)
    student = picker = None
    if args.mode in ("dagger", "student"):
        from lm.runtime import StudentRuntime
        student = StudentRuntime(args.checkpoint, batch=1)
        picker = StudentRuntime(args.checkpoint_pick, batch=1) if args.checkpoint_pick else student
    rng = np.random.default_rng(args.seed + 3)
    frames_dir = Path(args.frames) if args.frames else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)
    frame_no = 0
    summary = []
    for ep in range(args.episodes):
        ep_seed = args.seed * 1000 + ep
        r = np.random.default_rng(ep_seed + 7)
        start_bin = int(r.integers(0, 3)) if (r.random() < (0.5 if args.only_goto else args.start_bin_prob) and not args.only_pick) else None
        scene.reset(seed=ep_seed, start_bin=start_bin, prefill=0, fixed=True)
        world.new_episode()
        steps, picks = make_script(r, scene, start_bin)
        heading = HEADINGS["east"] if start_bin is not None else HEADINGS["north"]
        buf = {k: [] for k in ("head", "wrist", "q", "width", "base_vel", "base_act", "arm_act", "close", "phase", "cmd", "t", "arm", "executed_teacher")}
        log = []
        lat = []
        t_global = 0
        wall0 = time.perf_counter()
        current_pick = None
        carry = False   # gripper latched closed between a successful pick and the next drop
        for skill, arg in steps:
            t0 = scene.sim_time
            if skill == "face":
                heading = HEADINGS[arg]
                while scene.sim_time - t0 < TIMEOUT["face"]:
                    st = world.state()
                    if world.face(heading, st, close=carry if args.goto_planner else None):
                        break
                log.append({"skill": f"face({arg})", "seconds": round(scene.sim_time - t0, 2), "result": "ok"})
                continue
            target = arg if skill == "goto" else None
            color = arg if skill == "pick" else None
            if skill == "pick":
                current_pick = arg
            cmd = command_vector(skill, target, color)
            policy = picker if skill == "pick" else student
            if policy is not None:
                policy.reset()
            done_run = fail_run = 0
            result = "timeout"
            while scene.sim_time - t0 < TIMEOUT[skill]:
                st = world.state()
                head, wrist = scene.images()
                b_t, a_t, c_t, ph_t = world.teacher(skill, target, color, st) if (args.mode != "student" or (args.goto_planner and skill == "goto")) else (np.zeros(2), np.zeros(3), False, "move")
                use_teacher = student is None or (args.mode == "dagger" and rng.random() < args.beta) or (args.goto_planner and skill == "goto")
                out = None
                if student is not None and not (args.goto_planner and skill == "goto"):
                    prop = np.concatenate([st["q"], [st["width"]], st["base_vel"]]).astype(np.float32)
                    out = policy.act([head], [wrist], prop[None], cmd[None])
                    lat.append(out["ms_total"])
                if use_teacher:
                    b, a, c = np.asarray(b_t, np.float32), np.asarray(a_t, np.float32), c_t
                    if args.goto_planner and skill == "goto":
                        c = carry
                    if args.noise > 0:
                        b = b + rng.normal(0, args.noise * 0.08, 2).astype(np.float32)
                        a = a + rng.normal(0, args.noise * 0.04, 3).astype(np.float32)
                else:
                    act = out["act"][0]
                    b, a, c = act[:2], act[2:5], bool(out["bin"][0, 0])
                if args.save_data:
                    buf["head"].append(head); buf["wrist"].append(wrist)
                    buf["q"].append(st["q"]); buf["width"].append(st["width"]); buf["base_vel"].append(st["base_vel"].astype(np.float32))
                    buf["base_act"].append(np.asarray(b_t, np.float32)); buf["arm_act"].append(np.asarray(a_t, np.float32))
                    buf["close"].append(bool(c_t)); buf["phase"].append(PHASE_ID[ph_t]); buf["cmd"].append(cmd)
                    buf["t"].append(t_global); buf["arm"].append(0); buf["executed_teacher"].append(use_teacher)
                    t_global += 1
                world.actuate(skill, heading, b, a, c, st)
                if frames_dir is not None:
                    base_img = scene.frame(scene.video_cam)
                    if base_img is not None:
                        im = Image.fromarray(base_img).convert("RGB")
                        d = ImageDraw.Draw(im, "RGBA")
                        ov.panel(d, (16, 16, 600, 120))
                        who = ("PATH PLANNER (not learned)" if (args.goto_planner and skill == "goto") else "LEARNED POLICY") if args.mode == "student" else ("TEACHER" if args.mode == "expert" else "DAgger mix")
                        d.text((30, 24), f"SKILL: {skill}({arg or ''})   -   {who}", font=ov.font(17, True), fill=ov.ACCENT)
                        d.text((30, 52), "script: " + "  ".join(f"{s}({a or ''})" for s, a in steps)[:90], font=ov.font(13), fill=ov.DIM)
                        d.text((30, 76), "learned (camera-only): pick + drop arm motion and gripper", font=ov.font(13), fill=ov.DIM)
                        d.text((30, 96), "not learned: goto path planner, base heading, wrist angle", font=ov.font(13), fill=ov.DIM)
                        ov.inset(im, head, (1280 - 216, 46), 200, "policy head camera")
                        ov.inset(im, wrist, (1280 - 216, 282), 200, "policy wrist camera")
                        if out is not None:
                            ov.policy_panel(im, (16, 720 - 166), PHASES[int(out["phase"][0])], out["gate"][0], out["latent"][0], out["ms_total"])
                        im.save(frames_dir / f"f_{frame_no:05d}.jpg", quality=90)
                        frame_no += 1
                ph = ph_t if (args.mode != "student" or out is None) else PHASES[int(out["phase"][0])]
                done_run = done_run + 1 if ph in DONE_PHASES else 0
                fail_run = fail_run + 1 if ph == "fail" else 0
                if done_run >= (4 if args.mode != "student" else 8):
                    result = "done"
                    break
                if fail_run >= 15:
                    result = "fail"
                    break
            log.append({"skill": f"{skill}({arg or ''})", "seconds": round(scene.sim_time - t0, 2), "result": result})
            if skill == "pick":
                carry = result == "done"
            if skill == "drop":
                carry = False
        outcome = {}
        for c in picks:
            k = world.slot_of(c)
            pos = scene.object_state(k)[0]
            outcome[c] = world.in_bin(pos) or ("table" if world.on_table(pos) else "floor")
        intended = {}
        for (s1, a1), (s2, a2) in zip(steps, steps[1:] + [("", None)]):
            pass
        bins_for = {}
        cur = None
        for s1, a1 in steps:
            if s1 == "pick":
                cur = a1
            if s1 == "goto" and a1 != "table" and cur:
                bins_for[cur] = a1
        ok = sum(1 for c in picks if outcome[c] == bins_for.get(c))
        if args.only_goto:
            tgt = steps[-1][1]
            from course.skills import dock_xy
            ok = int(np.linalg.norm(scene.base_pose()[:2] - dock_xy(tgt)) < 0.10)
            picks = [tgt]
        if args.only_pick:
            picked = [e for e in log if e["skill"].startswith("pick")]
            k = world.slot_of(steps[0][1])
            ok = int(world.held(world.state()) == k)
            picks = [steps[0][1]]
        rec = {"episode": ep, "seed": ep_seed, "mode": args.mode, "deliveries_ok": ok, "deliveries": len(picks), "outcome": outcome,
               "intended": bins_for, "skills": log, "sim_seconds": round(scene.sim_time, 1), "wall_seconds": round(time.perf_counter() - wall0, 1),
               "flat_retries": getattr(scene, "flat_retries", 0)}
        if lat:
            rec["policy_ms_mean"] = float(np.mean(lat))
        print("EPISODE_JSON " + json.dumps(rec), flush=True)
        summary.append(rec)
        if args.save_data and buf["head"]:
            path = Path(args.save_data) / f"ep_{ep_seed:06d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **{k: np.asarray(v) for k, v in buf.items()})
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    app.close()
