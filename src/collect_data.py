"""Collect real Isaac Sim RGB observations with balanced state-supervisor labels."""

from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import argparse
import csv
import json
import math
import random
import time

import numpy as np
from PIL import Image

from isaac_common import (
    ROBOT_START_Z,
    action_command,
    build_scene,
    camera_rgb,
    place_target_relative,
    target_relative_state,
    teacher_action,
    update_front_camera,
    yaw_quaternion,
)


ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "data" / "labels.csv"
IMAGE_DIR = ROOT / "data" / "images"


def split_for_episode(number, total):
    train_end = max(3, int(total * 0.6))
    val_end = max(train_end + 1, int(total * 0.8))
    if number <= train_end:
        return "train"
    if number <= val_end:
        return "val"
    return "test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--samples-per-class", type=int, default=13)
    parser.add_argument("--seed", type=int, default=43017)
    args = parser.parse_args()
    if args.episodes < 5:
        raise ValueError("Use at least five grouped collection episodes")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for old_image in IMAGE_DIR.glob("ep*_*.png"):
        old_image.unlink()

    rng = random.Random(args.seed)
    world, robot, target, camera, _, controller = build_scene()
    rows = []
    # Let the RTX camera pipeline and exposure settle before sampling.
    for _ in range(40):
        robot.apply_wheel_actions(controller.forward(command=[0.0, 0.0]))
        update_front_camera(camera, robot)
        world.step(render=True)

    sample_id = 0
    start_time = time.time()
    for ep_number in range(1, args.episodes + 1):
        episode_id = f"ep{ep_number:02d}"
        actions = []
        for action in ("LEFT", "FORWARD", "RIGHT", "STOP"):
            actions.extend([action] * args.samples_per_class)
        rng.shuffle(actions)
        for action in actions:
            sample_id += 1
            yaw = rng.uniform(-0.55, 0.55)
            if action == "LEFT":
                bearing = rng.uniform(0.23, 0.72)
                distance = rng.uniform(0.85, 2.35)
            elif action == "RIGHT":
                bearing = -rng.uniform(0.23, 0.72)
                distance = rng.uniform(0.85, 2.35)
            elif action == "FORWARD":
                bearing = rng.uniform(-0.12, 0.12)
                distance = rng.uniform(1.55, 2.35)
            else:
                bearing = rng.uniform(-0.12, 0.12)
                distance = rng.uniform(1.35, 1.48)

            robot.set_world_pose(
                position=np.array([0.0, 0.0, ROBOT_START_Z], dtype=np.float32),
                orientation=yaw_quaternion(yaw),
            )
            robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
            target_xy = place_target_relative(target, np.array([0.0, 0.0]), yaw, bearing, distance)
            controller.reset()
            frame = None
            for _ in range(5):
                robot.apply_wheel_actions(controller.forward(command=[0.0, 0.0]))
                update_front_camera(camera, robot)
                world.step(render=True)
                frame = camera_rgb(camera)
            if frame is None or int(frame.max()) == 0:
                raise RuntimeError(f"No rendered RGB for {episode_id} sample {sample_id}")

            relative_path = Path("data") / "images" / f"{episode_id}_{sample_id:05d}.png"
            Image.fromarray(frame).save(ROOT / relative_path)
            measured_distance, measured_bearing, measured_yaw = target_relative_state(robot, target)
            # Labels come from a transparent geometric supervisor; perception at inference uses RGB only.
            label = teacher_action(measured_distance, measured_bearing)
            if label != action:
                raise RuntimeError(f"Supervisor mismatch at {episode_id} sample {sample_id}: expected {action}, measured {label}")
            rows.append(
                {
                    "timestamp": time.time(),
                    "image_path": relative_path.as_posix(),
                    "label": label,
                    "episode_id": episode_id,
                    "split": split_for_episode(ep_number, args.episodes),
                    "x": float(robot.get_world_pose()[0][0]),
                    "y": float(robot.get_world_pose()[0][1]),
                    "yaw": measured_yaw,
                    "target_x": float(target_xy[0]),
                    "target_y": float(target_xy[1]),
                    "target_distance_m": measured_distance,
                    "target_bearing_rad": measured_bearing,
                    "label_source": "simulator_state_supervisor",
                    "seed": args.seed,
                }
            )
            if sample_id % 64 == 0:
                print(f"COLLECTED frames={sample_id}/{args.episodes * args.samples_per_class * 4}", flush=True)

    fieldnames = list(rows[0].keys())
    with LABELS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset_version": "visible-stop-v2",
        "frames": len(rows),
        "episodes": args.episodes,
        "samples_per_class_per_episode": args.samples_per_class,
        "stop_distance_m": 1.50,
        "stop_sample_distance_range_m": [1.35, 1.48],
        "class_distribution": {action: sum(row["label"] == action for row in rows) for action in ("LEFT", "FORWARD", "RIGHT", "STOP")},
        "episode_splits": {f"ep{i:02d}": split_for_episode(i, args.episodes) for i in range(1, args.episodes + 1)},
        "label_source": "simulator_state_supervisor; images alone are policy inputs",
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    (ROOT / "data" / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DATASET", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
