"""Collect episode-grouped, domain-randomized teacher rollouts for MoE imitation."""

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
    parser.add_argument("--output-dir", type=Path, default=Path("data/moe_rollout_v2"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--rollouts-per-episode", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=120)
    parser.add_argument("--steps-per-decision", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9262026)
    args = parser.parse_args()

    if args.episodes < 10 or args.rollouts_per_episode < 2:
        raise ValueError("Use at least 10 appearance episodes and 2 rollouts per episode")
    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out_dir = out_dir.resolve()
    relative_out_dir = out_dir.relative_to(ROOT)
    image_dir = out_dir / "images"
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing collection: {out_dir}")
    image_dir.mkdir(parents=True, exist_ok=False)

    rng = random.Random(args.seed)
    world, robot, target, camera, _, controller = build_scene()
    light = UsdLux.DomeLight(world.stage.GetPrimAtPath("/World/DomeLight"))
    target_material = target.get_applied_visual_material()
    rows = []
    rollout_results = []
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

            for rollout_number in range(1, args.rollouts_per_episode + 1):
                rollout_id = f"{episode_id}_r{rollout_number:02d}"
                yaw = rng.uniform(-0.18, 0.18)
                bearing_centers = [0.0, 0.40, -0.40]
                center = bearing_centers[(rollout_number - 1) % len(bearing_centers)]
                bearing = center + (rng.uniform(-0.055, 0.055) if center else rng.uniform(-0.10, 0.10))
                distance = rng.uniform(1.85, 2.45)

                robot_position = np.array([0.0, 0.0, ROBOT_START_Z], dtype=np.float32)
                robot.set_world_pose(position=robot_position, orientation=yaw_quaternion(yaw))
                robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
                robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
                target_xy = place_target_relative(target, robot_position, yaw, bearing, distance)
                controller.reset()

                for _ in range(5):
                    robot.apply_wheel_actions(controller.forward(command=[0.0, 0.0]))
                    update_front_camera(camera, robot)
                    world.step(render=True)

                stopped = False
                for decision in range(1, args.max_decisions + 1):
                    update_front_camera(camera, robot)
                    frame = camera_rgb(camera)
                    if frame is None or int(frame.max()) == 0:
                        raise RuntimeError(f"No rendered RGB for {rollout_id} decision {decision}")
                    measured_distance, measured_bearing, measured_yaw = target_relative_state(robot, target)
                    label = teacher_action(measured_distance, measured_bearing)

                    sample_id += 1
                    relative_path = relative_out_dir / "images" / f"{sample_id:06d}.png"
                    Image.fromarray(frame).save(ROOT / relative_path)
                    position = robot.get_world_pose()[0]
                    rows.append({
                        "timestamp": time.time(),
                        "image_path": relative_path.as_posix(),
                        "label": label,
                        "episode_id": episode_id,
                        "rollout_id": rollout_id,
                        "decision": decision,
                        "split": split,
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "yaw": measured_yaw,
                        "target_x": float(target_xy[0]),
                        "target_y": float(target_xy[1]),
                        "target_distance_m": measured_distance,
                        "target_bearing_rad": measured_bearing,
                        "dome_intensity": intensity,
                        "target_color_rgb": json.dumps(shade),
                        "label_source": "closed_loop_simulator_pose_teacher",
                        "seed": args.seed,
                    })

                    if label == "STOP":
                        stopped = True
                        break

                    linear, angular = action_command(label)
                    for _ in range(args.steps_per_decision):
                        update_front_camera(camera, robot)
                        robot.apply_wheel_actions(controller.forward(command=[linear, angular]))
                        world.step(render=True)

                for _ in range(12):
                    robot.apply_wheel_actions(controller.forward(command=[0.0, 0.0]))
                    update_front_camera(camera, robot)
                    world.step(render=True)

                final_distance, final_bearing, _ = target_relative_state(robot, target)
                rollout_results.append({
                    "episode_id": episode_id,
                    "rollout_id": rollout_id,
                    "split": split,
                    "initial_yaw_rad": yaw,
                    "initial_bearing_rad": bearing,
                    "initial_distance_m": distance,
                    "decisions": min(decision, args.max_decisions),
                    "stopped": stopped,
                    "final_distance_m": final_distance,
                    "final_bearing_rad": final_bearing,
                    "success": bool(stopped and 1.20 <= final_distance <= 1.65 and abs(final_bearing) <= 0.18),
                })
            print(f"COLLECTED {episode_id}/{args.episodes} split={split} light={intensity:.0f} rows={len(rows)}", flush=True)

        metadata_path = out_dir / "metadata.csv"
        with metadata_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (out_dir / "rollouts.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rollout_results[0].keys()))
            writer.writeheader()
            writer.writerows(rollout_results)

        summary = {
            "dataset_version": "moe-domain-randomized-closed-loop-v2",
            "episodes": args.episodes,
            "rollouts": len(rollout_results),
            "frames": len(rows),
            "class_distribution": {action: sum(row["label"] == action for row in rows) for action in CLASSES},
            "split_frames": {name: sum(row["split"] == name for row in rows) for name in ("train", "val", "test")},
            "split_episodes": {name: sum(split_for_episode(i, args.episodes) == name for i in range(1, args.episodes + 1)) for name in ("train", "val", "test")},
            "rollout_successes": sum(result["success"] for result in rollout_results),
            "rollout_success_percent": 100.0 * sum(result["success"] for result in rollout_results) / len(rollout_results),
            "randomized_per_episode": {"dome_intensity": [650.0, 1900.0], "red_rgb": "sampled red shades"},
            "label_source": "pose-based expert action recorded during closed-loop rollout; pixels only are model input",
            "seed": args.seed,
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("DATASET", json.dumps(summary, sort_keys=True), flush=True)
    finally:
        world.stop()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
