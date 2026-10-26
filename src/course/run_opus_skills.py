"""Demo 3 (v2): Claude Opus writes primitive programs; learned skills execute them.

  python.bat src/course/run_opus_skills.py --checkpoint models/skills_s4.pt --frames artifacts/demo3v2_frames
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--checkpoint-pick", default="", help="specialist model for pick() (default: same as --checkpoint)")
parser.add_argument("--instruction", default="Put the blue ball in the teal bin, then put the red cube in the gold bin.")
parser.add_argument("--seed", type=int, default=21)
parser.add_argument("--max-seconds", type=float, default=240.0)
parser.add_argument("--frames", default="")
parser.add_argument("--report", default="results/demo3v2_run.json")
parser.add_argument("--effort", default="low")
parser.add_argument("--goto-planner", action="store_true", help="goto() by the classical grid path planner")
args = parser.parse_args()

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw

from course.opus_programmer import OpusProgrammer
from course.scene import CourseScene
from course.skills import DONE_PHASES, HEADINGS, PHASES, SkillWorld, command_vector
from lm.runtime import StudentRuntime
from relay.scene import SHAPES
from sim import overlay as ov

TIMEOUT = {"goto": 30.0, "pick": 20.0, "drop": 12.0, "face": 6.0}


def main():
    scene = CourseScene(video=bool(args.frames), planner_cams=True, seed=args.seed)
    scene.reset(seed=args.seed, start_bin=None, prefill=0, fixed=True)
    world = SkillWorld(scene)
    world.new_episode()
    student = StudentRuntime(args.checkpoint, batch=1)
    picker = StudentRuntime(args.checkpoint_pick, batch=1) if args.checkpoint_pick else student
    opus = OpusProgrammer(effort=args.effort, log_path=str(ROOT / "logs" / "demo3v2_opus_calls.jsonl"))
    frames_dir = Path(args.frames) if args.frames else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)
    frame_no = [0]
    heading = HEADINGS["north"]
    history, exec_log = [], []
    program, idx = [], 0
    last = {"say": "Looking at the cell...", "observation": ""}
    pinfo = {"phase": "-", "gate": np.zeros(4), "latent": np.zeros(16), "ms": 0.0}
    state = "plan"
    requested = False
    finished_at = None
    carry = [False]

    def render(head, wrist, status, col):
        if frames_dir is None:
            return
        base = scene.frame(scene.video_cam)
        if base is None:
            return
        im = Image.fromarray(base).convert("RGB")
        d = ImageDraw.Draw(im, "RGBA")
        ov.panel(d, (16, 16, 600, 300))
        d.text((30, 22), "OPERATOR", font=ov.font(13, True), fill=ov.DIM)
        for i, line in enumerate(ov.wrap(args.instruction, 66)[:2]):
            d.text((30, 40 + 19 * i), line, font=ov.font(15), fill=ov.TEXT)
        d.text((30, 84), "CLAUDE OPUS 5.5  (claude -p)  -  writes the program", font=ov.font(14, True), fill=ov.ACCENT)
        d.text((30, 104), status, font=ov.font(15, True), fill=col)
        for i, line in enumerate(ov.wrap('"' + last["say"] + '"', 64)[:2]):
            d.text((30, 126 + 18 * i), line, font=ov.font(14), fill=ov.TEXT)
        y = 168
        for j, st in enumerate(program[:7]):
            mark = ">" if (state == "run" and j == idx) else ("v" if j < idx else " ")
            colr = ov.GOOD if j < idx else (ov.WARN if mark == ">" else ov.DIM)
            d.text((30, y + 18 * j), f"{mark} {st['op']}({'' if st['arg'] == 'none' else st['arg']})", font=ov.font(14, mark == ">"), fill=colr)
        ov.inset(im, head, (1280 - 216, 46), 200, "policy head camera")
        ov.inset(im, wrist, (1280 - 216, 282), 200, "policy wrist camera")
        ov.policy_panel(im, (16, 720 - 166), pinfo["phase"], pinfo["gate"], pinfo["latent"], pinfo["ms"],
                        title="LEARNED pick/drop (camera-only latent policy)")
        d = ImageDraw.Draw(im, "RGBA")
        ov.panel(d, (440, 720 - 126, 1060, 720 - 16))
        d.text((454, 720 - 120), "EXECUTION LOG", font=ov.font(13, True), fill=ov.DIM)
        for i, e in enumerate(exec_log[-4:]):
            d.text((454, 720 - 100 + 19 * i), e, font=ov.font(14), fill=ov.GOOD if "done" in e or " ok" in e else ov.WARN)
        d.text((1060 - 150, 720 - 36), f"sim t = {scene.sim_time:5.1f} s", font=ov.font(14), fill=ov.DIM)
        im.save(frames_dir / f"f_{frame_no[0]:05d}.jpg", quality=90)
        frame_no[0] += 1

    while scene.sim_time < args.max_seconds:
        if state == "plan":
            st = world.state()
            head, wrist = scene.images()
            if not requested:
                imgs = [("table camera", scene.frame(scene.table_cam)), ("bins camera", scene.frame(scene.bins_cam)),
                        ("overview camera", scene.frame(scene.video_cam)), ("robot head camera", head)]
                opus.request(args.instruction, history[-12:], [(n, i) for n, i in imgs if i is not None])
                requested = True
            elif not opus.busy():
                decision, record = opus.poll()
                requested = False
                if decision is None:
                    history.append(f"(planner call failed: {str(record.get('error'))[:60]})")
                else:
                    last = {"say": decision["say"], "observation": decision["observation"]}
                    program = [p for p in decision["program"]]
                    idx = 0
                    history.append(f"You wrote: {' '.join(p['op'] + '(' + p['arg'] + ')' for p in program)}")
                    if not program or program[0]["op"] == "done":
                        state, finished_at = "finished", scene.sim_time
                    else:
                        state = "run"
            world.face(heading, st, close=carry[0])   # hold still (heading controller only) while Opus thinks
            render(head, wrist, f"thinking... {opus.elapsed():4.1f} s (robot holds still)", ov.WARN)
            continue
        if state == "finished":
            st = world.state()
            head, wrist = scene.images()
            world.face(heading, st, close=carry[0])
            render(head, wrist, "program finished: done()", ov.GOOD)
            if scene.sim_time - finished_at > 3.0:
                break
            continue
        # run the current step
        step = program[idx]
        op, arg = step["op"], step["arg"]
        t0 = scene.sim_time
        result = "timeout"
        if op == "done":
            state, finished_at = "finished", scene.sim_time
            continue
        if op == "face":
            heading = HEADINGS.get(arg, heading)
            while scene.sim_time - t0 < TIMEOUT["face"]:
                st = world.state()
                head, wrist = scene.images()
                ok = world.face(heading, st, close=carry[0])
                render(head, wrist, f"running: face({arg})  [fixed heading controller]", ov.GOOD)
                if ok:
                    result = "ok"
                    break
        else:
            target = arg if op == "goto" else None
            color = arg if op == "pick" else None
            cmd = command_vector(op, target, color)
            policy = picker if op == "pick" else student
            policy.reset()
            done_run = fail_run = 0
            while scene.sim_time - t0 < TIMEOUT[op]:
                st = world.state()
                head, wrist = scene.images()
                if args.goto_planner and op == "goto":
                    b_t, a_t, c_t, ph = world.teacher("goto", target, None, st)
                    world.actuate(op, heading, b_t, a_t, carry[0], st)
                    render(head, wrist, f"running: goto({arg})  [path planner, not learned]", ov.GOOD)
                else:
                    prop = np.concatenate([st["q"], [st["width"]], st["base_vel"]]).astype(np.float32)
                    out = policy.act([head], [wrist], prop[None], cmd[None])
                    act = out["act"][0]
                    world.actuate(op, heading, act[:2], act[2:5], bool(out["bin"][0, 0]), st)
                    ph = PHASES[int(out["phase"][0])]
                    pinfo = {"phase": ph, "gate": out["gate"][0], "latent": out["latent"][0], "ms": out["ms_total"]}
                    render(head, wrist, f"running: {op}({'' if arg == 'none' else arg})  [learned]", ov.GOOD)
                done_run = done_run + 1 if ph in DONE_PHASES else 0
                fail_run = fail_run + 1 if ph == "fail" else 0
                if done_run >= (4 if (args.goto_planner and op == "goto") else 8):
                    result = "done"
                    break
                if fail_run >= 15:
                    result = "failed"
                    break
        secs = scene.sim_time - t0
        entry = f"{op}({'' if arg == 'none' else arg}): {result} in {secs:.1f} s"
        exec_log.append(entry)
        history.append("Execution: " + entry + (" (policy's own completion signal)" if op != "face" else ""))
        if op == "pick":
            carry[0] = result == "done"
        if op == "drop":
            carry[0] = False
        idx += 1
        if result not in ("done", "ok") or idx >= len(program):
            state = "plan"

    outcome = {}
    for k in scene.active:
        pos = scene.object_state(k)[0]
        outcome[f"{scene.colors[k]} {SHAPES[k].name}"] = world.in_bin(pos) or ("table" if world.on_table(pos) else "floor")
    report = {"instruction": args.instruction, "outcome_physx": outcome, "execution_log": exec_log,
              "opus_calls": [{"wall_s": c["wall_s"], "decision": c.get("decision"), "error": c.get("error")} for c in opus.calls],
              "sim_seconds": round(scene.sim_time, 1), "checkpoint": args.checkpoint}
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DEMO3_JSON " + json.dumps({"outcome": outcome, "log": exec_log, "calls": len(opus.calls)}), flush=True)


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    app.close()
