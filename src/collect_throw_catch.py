"""Collect camera-only teacher labels for the six-arm visual relay in Isaac Sim."""

from pathlib import Path
import argparse
import csv
import json
import random
import sys
import traceback

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np
from pxr import Gf, Sdf, UsdLux

from throw_catch_scene import (
    ACTION_EVERY_STEPS,
    ARM_COUNT,
    BALL_Z,
    DT,
    HANDOFF_DWELL_S,
    build_relay_scene,
    camera_rgb,
    flight_position,
    set_arm_leds,
    station_xy,
    write_frame_png,
)


ROOT = Path(__file__).resolve().parents[1]
DEBUG_PATH = ROOT / "logs" / "throw_catch_collect_progress.txt"


def debug(message):
    DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(str(message) + "\n")


def main():
    DEBUG_PATH.write_text("main entered\n", encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=261005)
    parser.add_argument("--max-existing", action="store_true", help="refuse to overwrite an existing dataset")
    args = parser.parse_args()
    debug(f"args episodes={args.episodes} seed={args.seed}")
    data_root = ROOT / "data" / "throw_catch"
    image_root = data_root / "images"
    labels_path = data_root / "labels.csv"
    if args.max_existing and (labels_path.exists() or image_root.exists()):
        raise FileExistsError(f"Refusing to overwrite existing throw/catch data at {data_root}")
    image_root.mkdir(parents=True, exist_ok=True)
    debug("building scene")
    world, camera, arms, ball = build_relay_scene()
    debug("scene built")
    rows = []
    rng = random.Random(args.seed)
    sample_id = 0

    try:
        for episode in range(args.episodes):
            episode_sample_start = sample_id
            episode_id = f"ep{episode + 1:02d}"
            active = episode % ARM_COUNT
            peak = 0.36 + 0.045 * (episode % 6)
            flight_steps = 36 + 2 * (episode % 6)
            # Rotate the task through the ring so no single station defines the label.
            ball.set([*station_xy(active), BALL_Z + 0.26])

            def capture(label, transfer, phase, timestep):
                nonlocal sample_id
                rgb = camera_rgb(camera)
                if rgb is None:
                    raise RuntimeError("Relay camera returned no RGB frame")
                path = image_root / f"{episode_id}_{sample_id:05d}.png"
                write_frame_png(rgb, path)
                rows.append({
                    "image_path": path.relative_to(ROOT).as_posix(),
                    "label": label,
                    "episode_id": episode_id,
                    "split": "train" if episode < max(1, int(args.episodes * 0.67)) else ("validation" if episode < max(2, int(args.episodes * 0.84)) else "test"),
                    "transfer_index": transfer,
                    "phase": phase,
                    "sim_step": timestep,
                    "start_arm": episode % ARM_COUNT,
                    "flight_steps": flight_steps,
                    "peak_height_m": round(peak, 4),
                })
                sample_id += 1

            episode_step = 0
            transfer_count = ARM_COUNT
            set_arm_leds(arms, receiving=active, holding=True, ready=True)
            # Let the rendered displayColor/emissive beacon reach the camera before
            # attaching the THROW teacher label (the camera can return one stale frame).
            for _ in range(3):
                world.step(render=True)
                episode_step += 1
            capture("THROW", 0, "ready", episode_step)

            for transfer in range(transfer_count):
                target = (active + 1) % ARM_COUNT
                set_arm_leds(arms, receiving=target, holding=False, ready=False)
                for flight_tick in range(1, flight_steps + 1):
                    t = flight_tick / flight_steps
                    ball.set(flight_position(active, target, t, peak))
                    world.step(render=True)
                    episode_step += 1
                    if flight_tick % ACTION_EVERY_STEPS == 0 or flight_tick == flight_steps:
                        label = "CATCH" if t >= 0.74 else "WAIT"
                        capture(label, transfer, "flight", episode_step)

                # The held frame and status LED give the recurrent hand-off state
                # a visible cue for the frozen visual encoder.
                ball.set([*station_xy(target), BALL_Z + 0.26])
                set_arm_leds(arms, receiving=target, holding=True, ready=False)
                hold_steps = max(12, int(HANDOFF_DWELL_S / DT))
                for hold_tick in range(1, hold_steps + 1):
                    world.step(render=True)
                    episode_step += 1
                    if hold_tick % ACTION_EVERY_STEPS == 0 or hold_tick == hold_steps:
                        capture("HOLD", transfer, "held", episode_step)
                set_arm_leds(arms, receiving=target, holding=True, ready=True)
                for _ in range(3):
                    world.step(render=True)
                    episode_step += 1
                capture("THROW", transfer + 1, "ready", episode_step)
                active = target
            print(f"COLLECT episode={episode_id} frames={sample_id - episode_sample_start} total={sample_id}/{args.episodes}", flush=True)
            debug(f"episode={episode_id} sample_count={sample_id}")

        data_root.mkdir(parents=True, exist_ok=True)
        with labels_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        counts = {}
        for row in rows:
            counts[row["label"]] = counts.get(row["label"], 0) + 1
        summary = {
            "task": "six-arm circular throw/catch relay",
            "episodes": args.episodes,
            "frames": len(rows),
            "labels": counts,
            "image_size": [720, 720],
            "fps_sampling": 60.0 / ACTION_EVERY_STEPS,
            "seed": args.seed,
            "encoder_source": "models/autoresearch_spatial_moe_v3.pt (image_encoder + to_latent reused; new task head trained)",
            "label_source": "simulated relay phase; pixels are the policy input",
        }
        (data_root / "collection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        debug(f"complete frames={len(rows)}")
    except Exception:
        debug(traceback.format_exc())
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
