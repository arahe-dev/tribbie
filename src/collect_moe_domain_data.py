"""Collect a new episode-grouped, appearance-randomized MoE training set."""

from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import argparse
import csv
import json
import random
import time

import numpy as np
from PIL import Image
from pxr import UsdLux

from isaac_common import (
    ROBOT_START_Z,
    build_scene,
    camera_rgb,
    place_target_relative,
    target_relative_state,
    teacher_action,
    update_front_camera,
    yaw_quaternion,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "moe_domain_v1"
IMAGE_DIR = OUT_DIR / "images"
CLASSES = ("LEFT", "FORWARD", "RIGHT", "STOP")


def split_for_episode(number, total):
    train_end = max(3, int(total * 0.60))
    val_end = max(train_end + 1, int(total * 0.80))
    if number <= train_end:
        return "train"
    if number <= val_end:
        return "val"
    return "test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7262026)
    args = parser.parse_args()
    if args.episodes < 10:
        raise ValueError("Use at least 10 independently randomized episodes")
    if OUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing collection: {OUT_DIR}")
    IMAGE_DIR.mkdir(parents=True, exist_ok=False)

    rng = random.Random(args.seed)
    world, robot, target, camera, _, controller = build_scene()
    light = UsdLux.DomeLight(world.stage.GetPrimAtPath("/World/DomeLight"))
    target_material = target.get_applied_visual_material()
    rows = []
    started_at = time.time()
    try:
        for _ in range(40):
            robot.apply_wheel_actions(controller.forward(command=[0.0, 0.0]))
            update_front_camera(camera, robot)
            world.step(render=True)

        sample_id = 0
        for episode_number in range(1, args.episodes + 1):
            episode_id = f"ep{episode_number:02d}"
            split = split_for_episode(episode_number, args.episodes)
            intensity = rng.uniform(650.0, 1900.0)
            shade = [rng.uniform(0.84, 1.0), rng.uniform(0.025, 0.19), rng.uniform(0.005, 0.13)]
            light.GetIntensityAttr().Set(intensity)
            target_material.set_color(np.asarray(shade, dtype=np.float32))
            actions = [action for action in CLASSES for _ in range(args.samples_per_class)]
            rng.shuffle(actions)

            for desired_action in actions:
                sample_id += 1
                yaw = rng.uniform(-0.65, 0.65)
                if desired_action == "LEFT":
                    bearing, distance = rng.uniform(0.19, 0.68), rng.uniform(0.90, 2.65)
                elif desired_action == "RIGHT":
                    bearing, distance = -rng.uniform(0.19, 0.68), rng.uniform(0.90, 2.65)
                elif desired_action == "FORWARD":
                    bearing, distance = rng.uniform(-0.12, 0.12), rng.uniform(1.55, 2.65)
                else:
                    bearing, distance = rng.uniform(-0.12, 0.12), rng.uniform(1.35, 1.50)

                robot_position = np.array([0.0, 0.0, ROBOT_START_Z], dtype=np.float32)
                robot.set_world_pose(position=robot_position, orientation=yaw_quaternion(yaw))
                robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
                robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
                target_xy = place_target_relative(target, robot_position, yaw, bearing, distance)
                controller.reset()
                frame = None
                for _ in range(5):
                    robot.apply_wheel_actions(controller.forward(command=[0.0, 0.0]))
                    update_front_camera(camera, robot)
                    world.step(render=True)
                    frame = camera_rgb(camera)
                if frame is None or int(frame.max()) == 0:
                    raise RuntimeError(f"No rendered RGB for {episode_id} sample {sample_id}")

                relative_path = Path("data") / "moe_domain_v1" / "images" / f"{episode_id}_{sample_id:05d}.png"
                Image.fromarray(frame).save(ROOT / relative_path)
                measured_distance, measured_bearing, measured_yaw = target_relative_state(robot, target)
                label = teacher_action(measured_distance, measured_bearing)
                if label != desired_action:
                    raise RuntimeError(f"Supervisor mismatch at {episode_id} sample {sample_id}: expected {desired_action}, got {label}")
                rows.append({
                    "timestamp": time.time(),
                    "image_path": relative_path.as_posix(),
                    "label": label,
                    "episode_id": episode_id,
                    "split": split,
                    "x": float(robot.get_world_pose()[0][0]),
                    "y": float(robot.get_world_pose()[0][1]),
                    "yaw": measured_yaw,
                    "target_x": float(target_xy[0]),
                    "target_y": float(target_xy[1]),
                    "target_distance_m": measured_distance,
                    "target_bearing_rad": measured_bearing,
                    "dome_intensity": intensity,
                    "target_color_rgb": json.dumps(shade),
                    "label_source": "simulator_state_supervisor",
                    "seed": args.seed,
                })
            print(f"COLLECTED {episode_id}/{args.episodes} split={split} light={intensity:.0f} color={shade}", flush=True)

        with (OUT_DIR / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "dataset_version": "moe-domain-randomized-v1",
            "episodes": args.episodes,
            "frames": len(rows),
            "samples_per_class_per_episode": args.samples_per_class,
            "episode_splits": {f"ep{i:02d}": split_for_episode(i, args.episodes) for i in range(1, args.episodes + 1)},
            "class_distribution": {action: sum(row["label"] == action for row in rows) for action in CLASSES},
            "randomized_per_episode": {"dome_intensity": [650.0, 1900.0], "red_rgb": "sampled red shades", "start_yaw": [-0.65, 0.65]},
            "start_distance_ranges_m": {"LEFT_RIGHT": [0.90, 2.65], "FORWARD": [1.55, 2.65], "STOP": [1.35, 1.50]},
            "label_source": "simulator pose supervisor; image only is provided to the policy",
            "seed": args.seed,
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        (OUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("DATASET", json.dumps(summary, sort_keys=True), flush=True)
    finally:
        world.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
