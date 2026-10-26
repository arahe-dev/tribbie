"""Demo 3: Claude Opus plans, the camera-only latent policy executes.

Opus (through `claude -p`) looks at the table/bins/overview cameras and the
operator's instruction, and picks the next delivery (object colour -> bin).
The learned policy drives base, arm and gripper from its own two cameras.
The privileged teacher is not loaded. Task outcomes are judged afterwards
from PhysX object poses.

  python.bat src/course/run_planner.py --checkpoint models/course_student.pt --frames artifacts/demo3_frames
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--instruction", default="Put every round object (ball, cylinder, puck, capsule) in the teal bin and everything with flat faces in the magenta bin.")
parser.add_argument("--change-after", type=int, default=0, help="after this many deliveries, the operator changes the instruction")
parser.add_argument("--new-instruction", default="")
parser.add_argument("--shapes", default="0,2,4", help="shape indices on the table (cube, ball, puck)")
parser.add_argument("--colors", default="red,blue,green")
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--obstacles", type=int, default=3)
parser.add_argument("--max-seconds", type=float, default=300.0)
parser.add_argument("--exec-timeout", type=float, default=75.0)
parser.add_argument("--frames", default="")
parser.add_argument("--report", default="results/demo3_run.json")
parser.add_argument("--effort", default="low")
parser.add_argument("--fixed", action="store_true", help="narrow demo world used to train the executor")
args = parser.parse_args()

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw

from course.scene import BIN_NAMES, CONTROL_HZ, OBJECT_COLORS, STOW_Q, CourseScene
from course.expert import CourseExpert, PHASES
from course.planner import OpusPlanner
from lm.runtime import StudentRuntime
from relay.scene import SHAPES
from sim import overlay as ov

DONE_ID, FAIL_ID = PHASES.index("done"), PHASES.index("fail")


def command_vector(color, bin_index):
    v = np.zeros(len(OBJECT_COLORS) + len(BIN_NAMES), dtype=np.float32)
    v[OBJECT_COLORS.index(color)] = 1.0
    v[len(OBJECT_COLORS) + bin_index] = 1.0
    return v


def where_is(scene, k):
    """Ground-truth location of object k from PhysX (evaluation only)."""
    pos = scene.object_state(k)[0]
    b = CourseExpert.in_bin(pos)
    if b is not None:
        return BIN_NAMES[b]
    if CourseExpert.on_table(pos, np.zeros(3)):
        return "table"
    return "floor"


def main():
    scene = CourseScene(video=bool(args.frames), planner_cams=True, seed=args.seed)
    shapes = [int(s) for s in args.shapes.split(",")]
    colors = args.colors.split(",")
    scene.reset(seed=args.seed, n_objects=len(shapes), n_obstacles=args.obstacles, shapes=shapes, colors=colors,
                target_slot=shapes[0], target_bin=0, fixed=args.fixed)
    student = StudentRuntime(args.checkpoint, batch=1)
    planner = OpusPlanner(effort=args.effort, log_path=str(ROOT / "logs" / "demo3_planner_calls.jsonl"))
    frames_dir = Path(args.frames) if args.frames else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    instruction = args.instruction
    history = []
    deliveries = []
    state = "plan"
    cmd = None
    current = None
    exec_start = 0.0
    done_count = fail_count = 0
    last = {"say": "Reading the scene...", "observation": "", "wall": None}
    student_info = {"phase": "-", "gate": np.zeros(4), "latent": np.zeros(16), "ms": 0.0}
    planner_wall = []
    step = 0
    hold_q = scene.arm_q()[0]
    finished_at = None
    requested = False
    while scene.sim_time < args.max_seconds:
        head, wrist = scene.images()
        q, width = scene.arm_q()
        if state == "plan":
            res = None
            if not requested:
                note = ""
                if args.change_after and len(deliveries) == args.change_after and args.new_instruction and instruction != args.new_instruction:
                    instruction = args.new_instruction
                    note = "NOTE: the operator just changed the instruction (see above). Re-plan from the current scene."
                    history.append(f"Operator changed the instruction to: '{instruction}'")
                imgs = [("table camera", scene.frame(scene.table_cam)), ("bins camera", scene.frame(scene.bins_cam))]
                if scene.video_cam is not None:
                    imgs.append(("overview camera", scene.frame(scene.video_cam)))
                planner.request(instruction, history, [(n, i) for n, i in imgs if i is not None], note)
                requested = True
            elif not planner.busy():
                res = planner.poll()
            if res is not None:
                requested = False
                decision, record = res
                planner_wall.append(record["wall_s"])
                if decision is None:
                    history.append(f"(planner call failed: {record.get('error', '')[:80]})")
                    last = {"say": "Planner call failed; retrying.", "observation": "", "wall": record["wall_s"]}
                else:
                    last = {"say": decision["say"], "observation": decision["observation"], "wall": record["wall_s"]}
                    if decision["done"] or decision["object_color"] == "none" or decision["bin"] == "none":
                        state = "finished"
                        finished_at = scene.sim_time
                    elif decision["object_color"] not in scene.colors.values():
                        history.append(f"You asked for a {decision['object_color']} object, but no such object exists.")
                    else:
                        slot = [k for k, c in scene.colors.items() if c == decision["object_color"]][0]
                        b = BIN_NAMES.index(decision["bin"])
                        scene.set_command(slot, b)
                        cmd = command_vector(decision["object_color"], b)
                        current = {"color": decision["object_color"], "shape": SHAPES[slot].name, "bin": decision["bin"],
                                   "slot": slot, "t_start": round(scene.sim_time, 2), "planner_s": record["wall_s"]}
                        student.reset()
                        done_count = fail_count = 0
                        exec_start = scene.sim_time
                        state = "execute"
            # While Opus thinks the robot simply holds still (no command to execute).
            scene.control_step(np.zeros(3), hold_q, False, render=True)
        elif state == "execute":
            prop = np.concatenate([q, [width], scene.base_velocity_local()]).astype(np.float32)
            out = student.act([head], [wrist], prop[None], cmd[None])
            a = out["act"][0]
            phase_id = int(out["phase"][0])
            student_info = {"phase": PHASES[phase_id], "gate": out["gate"][0], "latent": out["latent"][0], "ms": out["ms_total"]}
            tcp, rot = scene.tcp(q)
            pose = scene.base_pose()
            scene.control_step(a[:3], scene.ee_to_q(q, tcp, rot, a[3:], pose), bool(out["bin"][0, 0]), render=True)
            hold_q = scene.arm_q()[0]
            done_count = done_count + 1 if phase_id == DONE_ID else 0
            fail_count = fail_count + 1 if phase_id == FAIL_ID else 0
            elapsed = scene.sim_time - exec_start
            if done_count >= CONTROL_HZ or fail_count >= 2 * CONTROL_HZ or elapsed > args.exec_timeout:
                why = "its phase head reported 'done'" if done_count >= CONTROL_HZ else ("its phase head reported 'fail'" if fail_count else f"timed out after {elapsed:.0f} s")
                truth = where_is(scene, current["slot"])
                current.update({"t_end": round(scene.sim_time, 2), "controller_report": why, "physx_location": truth,
                                "correct_bin": truth == current["bin"]})
                deliveries.append(current)
                history.append(f"Delivery {len(deliveries)}: you asked for the {current['color']} object -> {current['bin']} bin. "
                               f"The controller finished because {why}. Check the images for the actual result.")
                state = "plan"
        else:  # finished
            scene.control_step(np.zeros(3), hold_q, False, render=True)
            if scene.sim_time - finished_at > 3.0:
                break

        if frames_dir is not None:
            base = scene.frame(scene.video_cam)
            if base is not None:
                im = Image.fromarray(base).convert("RGB")
                d = ImageDraw.Draw(im, "RGBA")
                ov.panel(d, (16, 16, 560, 210))
                d.text((30, 24), "OPERATOR INSTRUCTION", font=ov.font(14, True), fill=ov.DIM)
                for i, line in enumerate(ov.wrap(instruction, 62)[:3]):
                    d.text((30, 44 + 20 * i), line, font=ov.font(16), fill=ov.TEXT)
                d.text((30, 112), "CLAUDE OPUS 5.5  -  planner via claude -p", font=ov.font(14, True), fill=ov.ACCENT)
                if state == "plan":
                    status = f"thinking... {planner.elapsed():4.1f} s (robot holds still)"
                    col = ov.WARN
                elif state == "execute":
                    status = f"task sent: {current['color']} {current['shape']} -> {current['bin']} bin"
                    col = ov.GOOD
                else:
                    status = "instruction complete"
                    col = ov.GOOD
                d.text((30, 134), status, font=ov.font(16, True), fill=col)
                for i, line in enumerate(ov.wrap('"' + last["say"] + '"', 64)[:2]):
                    d.text((30, 158 + 20 * i), line, font=ov.font(15), fill=ov.TEXT)
                if last["wall"] is not None:
                    d.text((420, 112), f"last call {last['wall']:.1f} s", font=ov.font(13), fill=ov.DIM)
                ov.inset(im, head, (1280 - 16 - 200, 46), 200, "policy head camera")
                ov.inset(im, wrist, (1280 - 16 - 200, 46 + 236), 200, "policy wrist camera")
                ov.policy_panel(im, (16, 720 - 16 - 150), student_info["phase"], student_info["gate"], student_info["latent"], student_info["ms"])
                d = ImageDraw.Draw(im, "RGBA")
                ov.panel(d, (440, 720 - 16 - 110, 1060, 720 - 16))
                d.text((454, 720 - 120), "DELIVERIES (outcome measured from PhysX)", font=ov.font(13, True), fill=ov.DIM)
                for i, dv in enumerate(deliveries[-4:]):
                    ok = dv["correct_bin"]
                    d.text((454, 720 - 100 + 20 * i), f"{dv['color']} {dv['shape']} -> {dv['bin']}:  {'in bin' if ok else 'landed: ' + dv['physx_location']}",
                           font=ov.font(15), fill=ov.GOOD if ok else ov.WARN)
                d.text((1060 - 150, 720 - 40), f"sim t = {scene.sim_time:5.1f} s", font=ov.font(14), fill=ov.DIM)
                im.save(frames_dir / f"f_{step:05d}.jpg", quality=90)
        step += 1

    final = {k: where_is(scene, k) for k in scene.active}
    report = {
        "instruction": args.instruction, "new_instruction": args.new_instruction or None, "change_after": args.change_after,
        "objects": {f"{scene.colors[k]} {SHAPES[k].name}": final[k] for k in scene.active},
        "deliveries": deliveries, "planner_calls": len(planner.calls), "planner_wall_s": planner_wall,
        "planner_decisions": [c.get("decision") for c in planner.calls], "sim_seconds": round(scene.sim_time, 2),
        "checkpoint": args.checkpoint,
    }
    Path(ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DEMO3_JSON " + json.dumps(report), flush=True)


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    app.close()
