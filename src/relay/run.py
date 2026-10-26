"""Run the six-arm relay with the privileged teacher or the camera-only student.

Examples (Isaac Sim's python.bat):
  run.py --mode expert --episodes 1 --seconds 40 --snapshots
  run.py --mode expert --episodes 12 --save-data data/relay/expert
  run.py --mode student --checkpoint models/relay_student.pt --video artifacts/relay_frames
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
parser.add_argument("--seconds", type=float, default=60.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--objects", type=int, default=4)
parser.add_argument("--save-data", type=str, default="")
parser.add_argument("--video", type=str, default="")
parser.add_argument("--snapshots", action="store_true")
parser.add_argument("--no-cameras", action="store_true")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--beta", type=float, default=0.0, help="DAgger: probability the teacher's action is executed")
parser.add_argument("--report", type=str, default="")
parser.add_argument("--noise", type=float, default=0.0, help="DART-style noise on executed teacher actions (labels stay clean)")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--policy-device", default="cuda", choices=("cuda", "cpu"))
parser.add_argument("--no-graph", action="store_true")
parser.add_argument("--no-teacher", action="store_true", help="student mode without even computing teacher labels")
parser.add_argument("--frames", type=str, default="", help="write overlay video frames here (student demo)")
args = parser.parse_args()

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw

from relay.scene import ARM_COUNT, CONTROL_HZ, PAD_TOP, SHAPES, RelayScene, pad_center
from relay.expert import PHASES, RelayExpert, gather, object_on_pad
from sim import overlay as ov
from sim.common import look_at_quat


class TransferTracker:
    """Counts completed tray-to-tray hand-offs and dropped objects from PhysX state."""

    def __init__(self, active):
        self.last_pad = {k: None for k in active}
        self.transfers = []
        self.drops = set()

    def update(self, ctx, sim_time):
        for k in ctx.objects:
            pos = ctx.objects[k][0]
            if pos[2] < -0.05:
                if k not in self.drops:
                    self.drops.add(k)
                continue
            for j in range(ARM_COUNT):
                if object_on_pad(ctx, k, j):
                    if self.last_pad[k] is not None and self.last_pad[k] != j:
                        self.transfers.append({"object": SHAPES[k].name, "from": self.last_pad[k], "to": j, "t": round(sim_time, 2)})
                    self.last_pad[k] = j


def save_episode(path: Path, buf: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in buf.items()})


def main():
    scene = RelayScene(object_count=args.objects, video=bool(args.video or args.frames), cameras=not args.no_cameras, seed=args.seed)
    frames_dir = Path(args.frames) if args.frames else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)
    expert = RelayExpert(scene)
    student = None
    if args.mode in ("student", "dagger"):
        from lm.runtime import StudentRuntime
        student = StudentRuntime(args.checkpoint, batch=ARM_COUNT, device=args.policy_device, cuda_graph=not args.no_graph)
    rng = np.random.default_rng(args.seed + 991)
    steps = int(args.seconds * CONTROL_HZ)
    summary = []
    video_dir = Path(args.video) if args.video else None
    if video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(args.episodes):
        ep_seed = args.seed * 1000 + ep
        scene.reset(seed=ep_seed)
        if student is not None:
            student.reset()
        tracker = TransferTracker(scene.active)
        buf = {key: [] for key in ("station", "wrist", "q", "width", "ee", "close", "phase", "arm", "t", "executed_teacher")}
        wall0 = time.perf_counter()
        phase_counts = np.zeros(len(PHASES), dtype=np.int64)
        latencies = []
        for step in range(steps):
            ctx = gather(scene)
            tracker.update(ctx, scene.sim_time)
            station, wrist = scene.images() if scene.station_cams else ([None] * ARM_COUNT, [None] * ARM_COUNT)
            if args.no_teacher:
                teacher = [(np.zeros(4, np.float32), False, 0)] * ARM_COUNT
            else:
                teacher = [expert.act(ctx, i) for i in range(ARM_COUNT)]
            if student is not None:
                t0 = time.perf_counter()
                prop = np.stack([np.concatenate([ctx.q[i], [ctx.width[i]]]) for i in range(ARM_COUNT)]).astype(np.float32)
                out = student.act(station, wrist, prop)
                latencies.append(out["ms_total"])
            if args.debug and student is not None and step < 60 and step % 3 == 0:
                for i in (0, 1):
                    print(f"CMP t={scene.sim_time:4.2f} arm={i} teacher_phase={PHASES[teacher[i][2]]} student_phase={PHASES[int(out['phase'][i])]} "
                          f"teacher_ee={np.round(teacher[i][0], 3).tolist()} student_ee={np.round(out['act'][i], 3).tolist()} "
                          f"close t/s={teacher[i][1]}/{bool(out['bin'][i, 0])} q={np.round(ctx.q[i], 2).tolist()}", flush=True)
            q_targets, closes = [], []
            for i in range(ARM_COUNT):
                ee_t, close_t, phase_t = teacher[i]
                use_teacher = student is None or (args.mode == "dagger" and rng.random() < args.beta)
                if use_teacher:
                    ee, close = ee_t, close_t
                    if args.noise > 0:
                        ee = ee + rng.normal(0, args.noise, 4).astype(np.float32) * np.array([0.04, 0.04, 0.03, 0.15], np.float32)
                else:
                    ee, close = out["act"][i].astype(np.float32), bool(out["bin"][i, 0])
                q_targets.append(scene.ee_to_q(i, ctx.q[i], ctx.tcp[i], ctx.rot[i], ee))
                closes.append(close)
                phase_counts[phase_t] += 1
                if args.save_data:
                    buf["station"].append(station[i])
                    buf["wrist"].append(wrist[i])
                    buf["q"].append(ctx.q[i])
                    buf["width"].append(ctx.width[i])
                    buf["ee"].append(ee_t)
                    buf["close"].append(close_t)
                    buf["phase"].append(phase_t)
                    buf["arm"].append(i)
                    buf["t"].append(step)
                    buf["executed_teacher"].append(use_teacher)
            if args.snapshots and step in (2, 30, 60, 120, 180):
                snap = ROOT / "tmp" / "relay_snapshots"
                snap.mkdir(parents=True, exist_ok=True)
                for arm in (0, 1):
                    Image.fromarray(station[arm]).resize((256, 256), Image.NEAREST).save(snap / f"station{arm}_{step:04d}.png")
                    Image.fromarray(wrist[arm]).resize((256, 256), Image.NEAREST).save(snap / f"wrist{arm}_{step:04d}.png")
                if scene.video_cam is not None:
                    Image.fromarray(np.asarray(scene.video_cam.get_rgba())[..., :3]).save(snap / f"video_{step:04d}.jpg")
            scene.control_step(q_targets, closes, render=True)
            if frames_dir is not None:
                ang = -0.6 + 2.0 * np.pi * scene.sim_time / 75.0
                cpos = np.array([3.1 * np.cos(ang), 3.1 * np.sin(ang), 1.9])
                scene.video_cam.set_world_pose(position=cpos, orientation=look_at_quat(cpos, [0, 0, 0.12]), camera_axes="world")
                base = np.asarray(scene.video_cam.get_rgba())
                if base.ndim == 3 and base.size and student is None:
                    im = Image.fromarray(base[..., :3]).convert("RGB")
                    d = ImageDraw.Draw(im, "RGBA")
                    ov.panel(d, (16, 16, 640, 150))
                    d.text((30, 24), "SIX FRANKA ARMS  -  SCRIPTED TEACHER (training-data generator)", font=ov.font(16, True), fill=ov.WARN)
                    d.text((30, 50), f"objects in play: {', '.join(SHAPES[k].name for k in scene.active)}", font=ov.font(15), fill=ov.TEXT)
                    d.text((30, 74), f"hand-offs completed: {len(tracker.transfers)}     dropped: {len(tracker.drops)}", font=ov.font(17, True), fill=ov.GOOD)
                    d.text((30, 102), "arms driven by a scripted controller using true object poses", font=ov.font(14), fill=ov.DIM)
                    d.text((30, 122), "real PhysX friction grasps, no welds; hand-offs counted from object poses", font=ov.font(14), fill=ov.DIM)
                    ov.inset(im, station[0], (1280 - 16 - 200, 46), 200, "arm 1: station camera")
                    ov.inset(im, wrist[0], (1280 - 16 - 200, 46 + 236), 200, "arm 1: wrist camera")
                    d = ImageDraw.Draw(im, "RGBA")
                    d.text((1280 - 170, 720 - 36), f"sim t = {scene.sim_time:5.1f} s", font=ov.font(14), fill=ov.DIM)
                    im.save(frames_dir / f"f_{step:05d}.jpg", quality=90)
                if base.ndim == 3 and base.size and student is not None:
                    im = Image.fromarray(base[..., :3]).convert("RGB")
                    d = ImageDraw.Draw(im, "RGBA")
                    ov.panel(d, (16, 16, 600, 150))
                    d.text((30, 24), "SIX FRANKA ARMS  -  ONE SHARED CAMERA-ONLY LATENT POLICY", font=ov.font(16, True), fill=ov.ACCENT)
                    d.text((30, 50), f"objects in play: {', '.join(SHAPES[k].name for k in scene.active)}", font=ov.font(15), fill=ov.TEXT)
                    d.text((30, 74), f"hand-offs completed: {len(tracker.transfers)}     dropped: {len(tracker.drops)}", font=ov.font(17, True), fill=ov.GOOD)
                    d.text((30, 102), "every arm is driven by the learned policy; friction grasps, no welds", font=ov.font(14), fill=ov.DIM)
                    d.text((30, 122), "hand-offs counted from PhysX object poses", font=ov.font(14), fill=ov.DIM)
                    ov.inset(im, station[0], (1280 - 16 - 200, 46), 200, "arm 1: station camera")
                    ov.inset(im, wrist[0], (1280 - 16 - 200, 46 + 236), 200, "arm 1: wrist camera")
                    ov.policy_panel(im, (16, 720 - 16 - 150), PHASES[int(out["phase"][0])], out["gate"][0], out["latent"][0], out["ms_total"],
                                    title="ARM 1 POLICY  (latency = all 6 arms, one batch)")
                    d = ImageDraw.Draw(im, "RGBA")
                    d.text((1280 - 170, 720 - 36), f"sim t = {scene.sim_time:5.1f} s", font=ov.font(14), fill=ov.DIM)
                    im.save(frames_dir / f"f_{step:05d}.jpg", quality=90)
            if video_dir is not None:
                frame = np.asarray(scene.video_cam.get_rgba())[..., :3]
                if frame.size:
                    Image.fromarray(frame).save(video_dir / f"ep{ep:02d}_{step:05d}.jpg", quality=92)
            if args.debug and step % CONTROL_HZ == 0:
                for i in range(ARM_COUNT):
                    if expert.debug[i] and expert.debug[i]['phase'] not in ('home',):
                        print(f'DBG t={scene.sim_time:5.1f} arm={i} ' + json.dumps(expert.debug[i]), flush=True)
            if step % (5 * CONTROL_HZ) == 0:
                phases = [PHASES[teacher[i][2]] for i in range(ARM_COUNT)]
                print(f"EP {ep} t={scene.sim_time:5.1f}s wall={time.perf_counter() - wall0:6.1f}s transfers={len(tracker.transfers)} "
                      f"drops={len(tracker.drops)} ik_fail={expert.ik_failures} phases={phases}", flush=True)
        wall = time.perf_counter() - wall0
        record = {
            "episode": ep, "seed": ep_seed, "mode": args.mode, "sim_seconds": round(scene.sim_time, 2), "wall_seconds": round(wall, 1),
            "objects": [SHAPES[k].name for k in scene.active], "colors": scene.object_colors,
            "transfers": len(tracker.transfers), "drops": len(tracker.drops), "ik_failures": expert.ik_failures,
            "phase_hist": {PHASES[p]: int(c) for p, c in enumerate(phase_counts)},
            "transfer_log": tracker.transfers,
        }
        if latencies:
            record["policy_ms_mean"] = float(np.mean(latencies))
            record["policy_ms_p95"] = float(np.percentile(latencies, 95))
        print("EPISODE_JSON " + json.dumps(record), flush=True)
        summary.append(record)
        if args.save_data:
            save_episode(Path(args.save_data) / f"ep_{ep_seed:06d}.npz", buf)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    app.close()
