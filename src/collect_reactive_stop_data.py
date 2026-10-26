"""Collect balanced, randomized image states for surprise-stop policy training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np
from PIL import Image
from pxr import UsdLux

from isaac_common import (
    ROBOT_START_Z,
    build_scene,
    camera_rgb,
    place_target_relative,
    update_front_camera,
    yaw_quaternion,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "reactive_stop_v1"
LABELS = ("LEFT", "FORWARD", "RIGHT", "STOP")
SAMPLES_PER_CLASS_PER_EPISODE = 12
DEFAULT_TARGET_COLOR = np.array([0.08, 0.72, 0.10], dtype=np.float32)


def target_condition(label, rng):
    if label == "STOP":
        # Keep the post inside this camera's visibly resolvable near-field range.
        return float(rng.uniform(1.08, 1.42)), float(rng.uniform(-0.14, 0.14))
    if label == "FORWARD":
        return float(rng.uniform(1.65, 2.8)), float(rng.uniform(-0.14, 0.14))
    distance = float(rng.uniform(1.55, 2.8))
    magnitude = float(rng.uniform(0.20, 0.50))
    return distance, magnitude if label == "LEFT" else -magnitude


def visible_cue_pixels(rgb, label):
    pixels = np.asarray(rgb, dtype=np.uint8)[:, :, :3].astype(np.int16)
    red, green, blue = [pixels[:, :, channel] for channel in range(3)]
    if label == "STOP":
        mask = (red >= 70) & (red >= green * 1.12) & (red >= blue * 1.12)
    else:
        mask = (green >= 70) & (green >= red * 1.12) & (green >= blue * 1.12)
    return int(mask.sum())


def episode_plan(episode_index, episode_count, samples_per_class, base_seed):
    """Regenerate one episode exactly, including prior captures after a resume."""
    episode_seed = base_seed + episode_index * 997
    rng = np.random.default_rng(episode_seed)
    labels = np.repeat(np.asarray(LABELS, dtype=object), samples_per_class)
    rng.shuffle(labels)
    episode_yaw = float(rng.uniform(-0.14, 0.14))
    train_end = int(episode_count * 2 / 3)
    validation_end = int(episode_count * 5 / 6)
    split = "train" if episode_index < train_end else ("val" if episode_index < validation_end else "test")
    samples = []
    for label_value in labels:
        label = str(label_value)
        distance, bearing = target_condition(label, rng)
        yaw = episode_yaw + float(rng.uniform(-0.04, 0.04))
        if label == "STOP":
            color = [rng.uniform(0.88, 1.0), rng.uniform(0.035, 0.13), rng.uniform(0.015, 0.075)]
        else:
            color = [rng.uniform(0.035, 0.13), rng.uniform(0.68, 0.95), rng.uniform(0.035, 0.13)]
        light = float(rng.uniform(800.0, 1600.0))
        target_x = float(distance * np.cos(yaw + bearing))
        target_y = float(distance * np.sin(yaw + bearing))
        samples.append({
            "label": label,
            "split": split,
            "distance": distance,
            "bearing": bearing,
            "yaw": yaw,
            "color": np.asarray(color, dtype=np.float32),
            "light": light,
            "target_x": target_x,
            "target_y": target_y,
            "seed": episode_seed,
        })
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS_PER_EPISODE)
    parser.add_argument("--seed", type=int, default=621006)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-episode", type=int, default=1,
                        help="Resume capture at this 1-based episode; prior image metadata is regenerated from the seed")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(f"COLLECT_START episodes={args.episodes} samples_per_class={args.samples_per_class} seed={args.seed}", flush=True)
    if args.episodes < 12 or args.samples_per_class < 1 or not 1 <= args.start_episode <= args.episodes:
        raise ValueError("Use at least 12 episodes and at least one frame per class per episode")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    labels_path = output_dir / "metadata.csv"
    image_root = output_dir / "images"
    if (labels_path.exists() or image_root.exists()) and not args.overwrite:
        raise FileExistsError(f"Dataset exists at {output_dir}; pass --overwrite to replace it")
    image_root.mkdir(parents=True, exist_ok=True)

    world = None
    rows = []
    train_end = int(args.episodes * 2 / 3)
    validation_end = int(args.episodes * 5 / 6)
    for episode_index in range(args.start_episode - 1):
        episode_id = f"ep{episode_index + 1:03d}"
        for sample_index, sample in enumerate(
                episode_plan(episode_index, args.episodes, args.samples_per_class, args.seed), start=1):
            relative_path = output_dir.relative_to(ROOT) / "images" / f"{episode_id}_{sample_index:04d}.png"
            if not (ROOT / relative_path).is_file():
                raise FileNotFoundError(f"Cannot resume: prior frame is missing: {relative_path}")
            rows.append({
                "image_path": relative_path.as_posix(),
                "label": sample["label"],
                "episode_id": episode_id,
                "split": sample["split"],
                "sample_index": sample_index,
                "condition": "surprise_stop_cue" if sample["label"] == "STOP" else "approach_state",
                "target_distance_m": round(sample["distance"], 5),
                "target_bearing_rad": round(sample["bearing"], 5),
                "robot_yaw_rad": round(sample["yaw"], 5),
                "target_x": round(sample["target_x"], 5),
                "target_y": round(sample["target_y"], 5),
                "target_actual_world_x": round(sample["target_x"], 5),
                "target_actual_world_y": round(sample["target_y"], 5),
                "target_actual_world_z": round(0.45, 5),
                "target_color_rgb": json.dumps([round(float(value), 5) for value in sample["color"]]),
                "dome_intensity": round(sample["light"], 2),
                "label_source": "randomized_geometry_teacher; RGB only at policy inference",
                "seed": sample["seed"],
            })
    if rows:
        print(f"RESUMED prior_frames={len(rows)} start_episode={args.start_episode}", flush=True)
        with labels_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    try:
        world, robot, target, camera, _, _ = build_scene(with_overhead=False, static_stop_pole=False)
        print("COLLECT_SCENE_READY", flush=True)
        dome_prim = world.stage.GetPrimAtPath("/World/DomeLight")
        dome = UsdLux.DomeLight(dome_prim)
        for _ in range(40):
            update_front_camera(camera, robot)
            world.step(render=True)

        for episode_index in range(args.start_episode - 1, args.episodes):
            episode_id = f"ep{episode_index + 1:03d}"
            episode_frames = 0
            for sample_index, sample in enumerate(
                    episode_plan(episode_index, args.episodes, args.samples_per_class, args.seed), start=1):
                label = sample["label"]
                split = sample["split"]
                distance = sample["distance"]
                bearing = sample["bearing"]
                yaw = sample["yaw"]
                color = sample["color"]
                light = sample["light"]
                robot_position = np.array([0.0, 0.0, ROBOT_START_Z], dtype=np.float32)
                robot.set_world_pose(position=robot_position, orientation=yaw_quaternion(yaw))
                robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
                robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
                active_target = target
                material = active_target.get_applied_visual_material()
                if material is None:
                    raise RuntimeError("Target object has no mutable visual material")
                material.set_color(color)
                dome.GetIntensityAttr().Set(light)
                target_xy = place_target_relative(active_target, robot_position, yaw, bearing, distance)
                rgb = None
                consecutive_visible_frames = 0
                for frame_index in range(60):
                    update_front_camera(camera, robot)
                    world.step(render=True)
                    rgb = camera_rgb(camera)
                    cue_pixels = visible_cue_pixels(rgb, label) if rgb is not None else 0
                    if cue_pixels >= 25:
                        consecutive_visible_frames += 1
                        # Camera output and kinematic USD transforms are
                        # asynchronous; wait through stale frames before
                        # accepting a still-visible previous cue.
                        if frame_index >= 4 and consecutive_visible_frames >= 2:
                            break
                    else:
                        consecutive_visible_frames = 0
                if rgb is None or visible_cue_pixels(rgb, label) < 25:
                    raise RuntimeError(
                        f"Cue not visible in {episode_id} sample {sample_index} label={label} "
                        f"distance={distance:.3f}m"
                    )
                actual_target_position, _ = active_target.get_world_pose()
                relative_path = output_dir.relative_to(ROOT) / "images" / f"{episode_id}_{sample_index:04d}.png"
                Image.fromarray(rgb[:, :, :3]).save(ROOT / relative_path)
                rows.append({
                    "image_path": relative_path.as_posix(),
                    "label": label,
                    "episode_id": episode_id,
                    "split": split,
                    "sample_index": sample_index,
                    "condition": "surprise_stop_cue" if label == "STOP" else "approach_state",
                    "target_distance_m": round(distance, 5),
                    "target_bearing_rad": round(bearing, 5),
                    "robot_yaw_rad": round(yaw, 5),
                    "target_x": round(float(target_xy[0]), 5),
                    "target_y": round(float(target_xy[1]), 5),
                    "target_actual_world_x": round(float(actual_target_position[0]), 5),
                    "target_actual_world_y": round(float(actual_target_position[1]), 5),
                    "target_actual_world_z": round(float(actual_target_position[2]), 5),
                    "target_color_rgb": json.dumps([round(float(value), 5) for value in color]),
                    "dome_intensity": round(light, 2),
                    "label_source": "randomized_geometry_teacher; RGB only at policy inference",
                    "seed": sample["seed"],
                })
                episode_frames += 1
            print(f"COLLECT {episode_id} split={split} frames={episode_frames} total={len(rows)}", flush=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            with labels_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        output_dir.mkdir(parents=True, exist_ok=True)
        with labels_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "dataset_version": "randomized-surprise-stop-v1",
            "task": "pixels-only visual policy; randomized red stop cue, green non-event marker, and directional approach actions",
            "episodes": args.episodes,
            "samples_per_class_per_episode": args.samples_per_class,
            "frames": len(rows),
            "class_counts": {label: sum(row["label"] == label for row in rows) for label in LABELS},
            "split_episodes": {
                split: sorted({row["episode_id"] for row in rows if row["split"] == split})
                for split in ("train", "val", "test")
            },
            "randomized_variables": ["episode yaw", "target distance", "target bearing", "red/green hue", "dome light intensity"],
            "policy_observation": "320x240 RGB only; geometry fields are labels/evaluation metadata",
            "seed": args.seed,
            "metadata": labels_path.relative_to(ROOT).as_posix(),
        }
        (output_dir / "collection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("DATASET", json.dumps(summary, sort_keys=True), flush=True)
    except Exception as error:
        print(f"COLLECT_ERROR {type(error).__name__}: {error}", flush=True)
        raise
    finally:
        if world is not None:
            world.stop()
        app.close()


if __name__ == "__main__":
    main()
