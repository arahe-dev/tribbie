"""Run the mobile-manipulation course with the teacher (data) or the student (demo/eval).

  run.py --mode expert --episodes 40 --save-data data/course/expert
  run.py --mode student --checkpoint models/course_student.pt --episodes 10 --report results/course_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("expert", "student", "dagger"), default="expert")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seconds", type=float, default=75.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--objects", type=int, default=0, help="0 = random 1..3")
parser.add_argument("--obstacles", type=int, default=0, help="0 = random")
parser.add_argument("--fixed", action="store_true", help="narrow demo world: fixed obstacles and object slots")
parser.add_argument("--save-data", type=str, default="")
parser.add_argument("--video", type=str, default="")
parser.add_argument("--snapshots", action="store_true")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--beta", type=float, default=0.0)
parser.add_argument("--noise", type=float, default=0.0, help="DART-style noise scale on executed teacher actions (labels stay clean)")
parser.add_argument("--report", type=str, default="")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--policy-device", default="cuda", choices=("cuda", "cpu"))
parser.add_argument("--no-graph", action="store_true")
parser.add_argument("--probe", default="", help="diagnostics: import_torch | runtime_noact")
parser.add_argument("--no-teacher", action="store_true", help="student mode without computing teacher labels at all")
parser.add_argument("--frames", type=str, default="", help="write overlay demo frames here")
parser.add_argument("--start-bin-prob", type=float, default=0.0, help="probability an episode starts parked at a bin")
parser.add_argument("--prefill-prob", type=float, default=0.0, help="probability one non-target object starts in a bin")
args = parser.parse_args()

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw

from course.scene import BIN_NAMES, CONTROL_HZ, OBJECT_COLORS, CourseScene
from course.expert import PHASES, CourseExpert, footprint_hits
from relay.scene import SHAPES
from sim import overlay as ov

BASE_SCALE = np.array([0.55, 0.55, 0.9], dtype=np.float32)


def command_vector(scene):
    v = np.zeros(len(OBJECT_COLORS) + len(BIN_NAMES), dtype=np.float32)
    v[OBJECT_COLORS.index(scene.colors[scene.target_slot])] = 1.0
    v[len(OBJECT_COLORS) + scene.target_bin] = 1.0
    return v


def gather(scene):
    q, width = scene.arm_q()
    tcp, rot = scene.tcp(q)
    objs = {slot: scene.object_state(slot) for slot in scene.active}
    return {"pose": scene.base_pose(), "q": q, "width": width, "tcp": tcp, "rot": rot, "objs": objs,
            "base_vel": scene.base_velocity_local()}


def main():
    scene = CourseScene(video=bool(args.video or args.frames), seed=args.seed)
    frames_dir = Path(args.frames) if args.frames else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)
    frame_no = 0
    expert = CourseExpert(scene)
    student = None
    if args.probe == "import_torch":
        import torch  # noqa: F401
    if args.probe == "runtime_noact":
        from lm.runtime import StudentRuntime
        _unused = StudentRuntime(args.checkpoint, batch=1, device=args.policy_device)
    if args.mode in ("student", "dagger"):
        from lm.runtime import StudentRuntime
        student = StudentRuntime(args.checkpoint, batch=1, device=args.policy_device, cuda_graph=not args.no_graph)
    rng = np.random.default_rng(args.seed + 17)
    steps = int(args.seconds * CONTROL_HZ)
    video_dir = Path(args.video) if args.video else None
    if video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    ep = 0
    attempt = 0
    while ep < args.episodes:
        ep_seed = args.seed * 1000 + attempt
        attempt += 1
        r = np.random.default_rng(ep_seed + 555)
        start_bin = int(r.integers(0, 3)) if r.random() < args.start_bin_prob else None
        prefill = 1 if r.random() < args.prefill_prob else 0
        scene.reset(seed=ep_seed, n_objects=args.objects or None, n_obstacles=args.obstacles or None, start_bin=start_bin, prefill=prefill, fixed=args.fixed)
        frame_start = frame_no
        if not expert.new_episode():
            print(f"SKIP seed={ep_seed}: a bin is unreachable for this obstacle layout", flush=True)
            continue
        if student is not None:
            student.reset()
        cmd = command_vector(scene)
        buf = {k: [] for k in ("head", "wrist", "q", "width", "base_vel", "base_act", "ee", "close", "phase", "cmd", "t", "arm", "executed_teacher")}
        collided = False
        collisions = 0
        lat = []
        head_std = []
        head_dump = []
        phase_hist = np.zeros(len(PHASES), dtype=np.int64)
        done_step = None
        wall0 = time.perf_counter()
        for step in range(steps):
            st = gather(scene)
            if footprint_hits(st["pose"], scene.obstacles):
                collisions += 1
                collided = True
                if args.debug:
                    print(f"COLLIDE t={scene.sim_time:5.1f} pose={np.round(st['pose'], 3).tolist()} phase={PHASES[expert.act(st)[3]]}", flush=True)
            head, wrist = scene.images()
            if student is None:
                head_std.append(float(head.astype(np.float32).std()))
            if args.probe == "dump":
                head_dump.append(head)
            if args.no_teacher:
                base_t, dq_t, close_t, phase_t = np.zeros(3), np.zeros(4, np.float32), False, PHASES.index("approach")
            else:
                base_t, dq_t, close_t, phase_t = expert.act(st)
            use_teacher = student is None or (args.mode == "dagger" and rng.random() < args.beta)
            if student is not None:
                prop = np.concatenate([st["q"], [st["width"]], st["base_vel"]]).astype(np.float32)
                out = student.act([head], [wrist], prop[None], cmd[None])
                head_std.append(float(head.astype(np.float32).std()))
                lat.append(out["ms_total"])
            if use_teacher:
                base, dq, close = base_t, dq_t, close_t
                if args.noise > 0:
                    base = np.asarray(base, dtype=np.float64) + rng.normal(0, args.noise * np.array([0.12, 0.12, 0.2]))
                    dq = dq + rng.normal(0, args.noise, 4).astype(np.float32) * np.array([0.04, 0.04, 0.03, 0.15], np.float32)
            else:
                a = out["act"][0]
                base, dq, close = a[:3], a[3:], bool(out["bin"][0, 0])
            phase_hist[phase_t] += 1
            if args.save_data:
                buf["head"].append(head)
                buf["wrist"].append(wrist)
                buf["q"].append(st["q"])
                buf["width"].append(st["width"])
                buf["base_vel"].append(st["base_vel"].astype(np.float32))
                buf["base_act"].append(np.asarray(base_t, dtype=np.float32))
                buf["ee"].append(dq_t)
                buf["close"].append(close_t)
                buf["phase"].append(phase_t)
                buf["cmd"].append(cmd)
                buf["t"].append(step)
                buf["arm"].append(0)
                buf["executed_teacher"].append(use_teacher)
            if args.snapshots and step in (3, 45, 150, 300):
                snap = ROOT / "tmp" / "course_snapshots"
                snap.mkdir(parents=True, exist_ok=True)
                Image.fromarray(head).resize((288, 288), Image.NEAREST).save(snap / f"head_{step:04d}.png")
                Image.fromarray(wrist).resize((256, 256), Image.NEAREST).save(snap / f"wrist_{step:04d}.png")
                if scene.video_cam is not None:
                    Image.fromarray(np.asarray(scene.video_cam.get_rgba())[..., :3]).save(snap / f"video_{step:04d}.jpg")
            scene.control_step(base, scene.ee_to_q(st["q"], st["tcp"], st["rot"], dq, st["pose"]), close, render=True)
            if video_dir is not None:
                frame = np.asarray(scene.video_cam.get_rgba())[..., :3]
                if frame.size:
                    Image.fromarray(frame).save(video_dir / f"ep{ep:02d}_{step:05d}.jpg", quality=92)
            if args.debug and (step % CONTROL_HZ == 0 or (step < 60 and step % 2 == 0)):
                extra = ""
                if student is not None:
                    extra = f" STUDENT phase={PHASES[int(out['phase'][0])]} base={np.round(out['act'][0][:3], 2).tolist()} close={bool(out['bin'][0, 0])}"                             f" teacher_base={np.round(np.asarray(base_t), 2).tolist()} teacher_close={close_t} dq_err={float(np.abs(out['act'][0][3:] - dq_t).max()):.3f}"
                print(f"DBG t={scene.sim_time:5.1f} " + json.dumps(expert.debug) + extra, flush=True)
            if frames_dir is not None and student is not None:
                base_img = np.asarray(scene.video_cam.get_rgba())
                if base_img.ndim == 3 and base_img.size:
                    im = Image.fromarray(base_img[..., :3]).convert("RGB")
                    d = ImageDraw.Draw(im, "RGBA")
                    ov.panel(d, (16, 16, 600, 150))
                    d.text((30, 24), "MOBILE MANIPULATOR  -  CAMERA-ONLY LATENT POLICY", font=ov.font(16, True), fill=ov.ACCENT)
                    target = f"{scene.colors[scene.target_slot]} {SHAPES[scene.target_slot].name}"
                    d.text((30, 50), f"command: deliver the {target} -> {BIN_NAMES[scene.target_bin]} bin", font=ov.font(16, True), fill=ov.TEXT)
                    pos = scene.object_state(scene.target_slot)[0]
                    lb = expert.in_bin(pos)
                    status = "in the commanded bin" if lb == scene.target_bin else ("in the WRONG bin" if lb is not None else "not delivered yet")
                    d.text((30, 76), f"object status (PhysX): {status}", font=ov.font(15), fill=ov.GOOD if lb == scene.target_bin else ov.DIM)
                    d.text((30, 102), "base, arm and gripper all driven by the learned policy", font=ov.font(14), fill=ov.DIM)
                    d.text((30, 122), "holonomic base model; friction grasp, no welds", font=ov.font(14), fill=ov.DIM)
                    ov.inset(im, head, (1280 - 16 - 200, 46), 200, "policy head camera")
                    ov.inset(im, wrist, (1280 - 16 - 200, 46 + 236), 200, "policy wrist camera")
                    ov.policy_panel(im, (16, 720 - 16 - 150), PHASES[int(out["phase"][0])], out["gate"][0], out["latent"][0], out["ms_total"])
                    d = ImageDraw.Draw(im, "RGBA")
                    d.text((1280 - 170, 720 - 36), f"sim t = {scene.sim_time:5.1f} s", font=ov.font(14), fill=ov.DIM)
                    im.save(frames_dir / f"f_{frame_no:05d}.jpg", quality=90)
                    frame_no += 1
            if args.no_teacher:
                # Without a teacher, end on the policy's own 'done' report (evaluation still uses PhysX poses).
                phase_t = int(out["phase"][0])
            if PHASES[phase_t] in ("done", "fail"):
                done_step = step if done_step is None else done_step
            else:
                done_step = None
            if done_step is not None and step - done_step > 3 * CONTROL_HZ:
                break
        target_pos = scene.object_state(scene.target_slot)[0]
        landed = expert.in_bin(target_pos)
        success = landed == scene.target_bin
        record = {
            "episode": ep, "seed": ep_seed, "mode": args.mode, "success": bool(success), "landed_bin": landed,
            "target_bin": BIN_NAMES[scene.target_bin], "target_object": f"{scene.colors[scene.target_slot]} {SHAPES[scene.object_shape[scene.target_slot]].name}",
            "frames": [frame_start, frame_no], "objects": len(scene.active), "start_bin": start_bin, "prefill": prefill, "obstacles": len(scene.obstacles), "collision_steps": collisions,
            "sim_seconds": round(scene.sim_time, 2), "wall_seconds": round(time.perf_counter() - wall0, 1),
            "phase_hist": {PHASES[i]: int(c) for i, c in enumerate(phase_hist)},
        }
        if args.probe == "dump":
            np.savez_compressed(ROOT / "tmp" / "head_dump.npz", head=np.asarray(head_dump))
        if head_std:
            record["head_flat_frac"] = round(float(np.mean(np.array(head_std) < 12)), 3)
        if lat:
            record["policy_ms_mean"] = float(np.mean(lat))
            record["policy_ms_p95"] = float(np.percentile(lat, 95))
        print("EPISODE_JSON " + json.dumps(record), flush=True)
        summary.append(record)
        if args.save_data:
            path = Path(args.save_data) / f"ep_{ep_seed:06d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **{k: np.asarray(v) for k, v in buf.items()})
        ep += 1
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
