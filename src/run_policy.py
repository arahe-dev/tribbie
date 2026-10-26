"""Run the 16-D ResNet/PCA/k-NN policy in Isaac Sim via local GPU inference IPC."""

from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from collections import Counter, deque
import argparse
import csv
import json
import math
import os
import re
import socket
import struct
import subprocess
import time

import numpy as np
from pxr import UsdLux

from isaac_common import (
    DT,
    ROBOT_START_Z,
    action_command,
    build_scene,
    camera_rgb,
    place_target_relative,
    target_relative_state,
    update_front_camera,
    yaw_quaternion,
)


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = ROOT / "src" / "inference_server.py"
VIDEO_FRAME_DIR = ROOT / "artifacts" / "video_frames"


class InferenceClient:
    def __init__(self, connection):
        self.connection = connection
        self.buffer = bytearray()

    def decide(self, rgb):
        payload = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
        t0 = time.perf_counter_ns()
        self.connection.sendall(struct.pack("!I", len(payload)) + payload)
        while b"\n" not in self.buffer:
            chunk = self.connection.recv(4096)
            if not chunk:
                raise ConnectionError("Inference worker closed the connection")
            self.buffer.extend(chunk)
        line, _, tail = self.buffer.partition(b"\n")
        self.buffer = bytearray(tail)
        result = json.loads(line.decode("utf-8"))
        result["latency_ms"]["client_round_trip"] = (time.perf_counter_ns() - t0) / 1e6
        return result


def start_inference_worker(policy, moe_checkpoint=None, moe_device="auto"):
    if not VENV_PYTHON.is_file():
        raise FileNotFoundError(f"Missing project environment: {VENV_PYTHON}")
    env = os.environ.copy()
    # Keep the CUDA worker independent of Kit/Isaac's bundled CUDA DLL search paths.
    for key in ("PYTHONPATH", "PYTHONHOME", "CARB_APP_PATH", "ISAAC_PATH", "EXP_PATH"):
        env.pop(key, None)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    command = [str(VENV_PYTHON), "-u", str(SERVER), "--port", "0", "--policy", policy]
    if moe_checkpoint is not None:
        command.extend(["--moe-checkpoint", str(moe_checkpoint)])
    command.extend(["--moe-device", moe_device])
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=flags,
    )
    try:
        deadline = time.monotonic() + 120
        ready_line = ""
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if line:
                ready_line = line.strip()
                print(f"INFERENCE_WORKER {ready_line}", flush=True)
                if ready_line.startswith("READY "):
                    break
            elif process.poll() is not None:
                raise RuntimeError(f"Inference worker exited with code {process.returncode}")
        if not ready_line.startswith("READY "):
            raise TimeoutError("Inference worker did not become ready within 120 seconds")
        port = int(ready_line.split()[1])
        connection = socket.create_connection(("127.0.0.1", port), timeout=30)
        connection.settimeout(30)
        return process, InferenceClient(connection)
    except Exception:
        process.terminate()
        process.wait(timeout=10)
        raise


def majority_vote(history):
    counts = Counter(history)
    max_count = max(counts.values())
    return next(action for action in reversed(history) if counts[action] == max_count)


def episode_initial_conditions(index, rng):
    signs = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    bearing = signs[(index - 1) % len(signs)] * rng.uniform(0.22, 0.34)
    distance = rng.uniform(1.90, 2.20)
    yaw = rng.uniform(-0.08, 0.08)
    return yaw, bearing, distance


def apply_case_appearance(world, target, case):
    """Apply manifest-controlled nuisance variations before each rollout."""
    if "dome_intensity" in case:
        light_prim = world.stage.GetPrimAtPath("/World/DomeLight")
        if not light_prim.IsValid():
            raise RuntimeError("Robustness case requested lighting variation but DomeLight is missing")
        UsdLux.DomeLight(light_prim).GetIntensityAttr().Set(float(case["dome_intensity"]))
    scale_factor = float(case.get("target_scale_factor", 1.0))
    if not 0.65 <= scale_factor <= 1.4:
        raise ValueError(f"Target scale outside permitted robustness range: {scale_factor}")
    if scale_factor != 1.0:
        raise ValueError("Per-case target scaling is not supported by Isaac's active physics tensor view")
    if "target_color_rgb" in case:
        color = np.asarray(case["target_color_rgb"], dtype=np.float32)
        if color.shape != (3,) or np.any(color < 0.0) or np.any(color > 1.0):
            raise ValueError(f"Invalid target RGB color in case {case.get('case_id')}")
        material = target.get_applied_visual_material()
        if material is None:
            raise RuntimeError("Target has no mutable visual material")
        material.set_color(color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-decisions", type=int, default=120)
    parser.add_argument("--steps-per-decision", type=int, default=9)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--policy", choices=("raw", "pca16", "pca64", "moe16"), default="pca16")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--seed", type=int, default=62024)
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--case-start-index", type=int, default=0)
    parser.add_argument("--case-count", type=int)
    parser.add_argument("--run-tag")
    parser.add_argument("--model-checkpoint", type=Path)
    parser.add_argument("--moe-device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.smoothing_window not in (1, 3, 5):
        raise ValueError("smoothing-window must be one of 1, 3, or 5")

    cases = None
    case_ids = []
    if args.case_manifest:
        manifest_path = args.case_manifest if args.case_manifest.is_absolute() else ROOT / args.case_manifest
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        all_cases = manifest.get("cases", manifest) if isinstance(manifest, dict) else manifest
        if not isinstance(all_cases, list) or not all_cases:
            raise ValueError("Case manifest must be a non-empty JSON list or an object with a 'cases' list")
        if args.case_start_index < 0 or args.case_start_index >= len(all_cases):
            raise ValueError("case-start-index is outside the manifest")
        end_index = len(all_cases) if args.case_count is None else min(len(all_cases), args.case_start_index + args.case_count)
        if end_index <= args.case_start_index:
            raise ValueError("case-count must select at least one case")
        cases = all_cases[args.case_start_index:end_index]
        for index, case in enumerate(cases, start=args.case_start_index):
            required = ("case_id", "yaw_rad", "bearing_rad", "distance_m")
            missing = [key for key in required if key not in case]
            if missing:
                raise ValueError(f"Case at index {index} is missing fields: {missing}")
            case_ids.append(str(case["case_id"]))
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case_id values must be unique within the selected manifest slice")
        if args.run_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag):
            raise ValueError("run-tag may contain only letters, numbers, underscores, and hyphens")
        args.episodes = len(cases)

    (ROOT / "results").mkdir(exist_ok=True)
    policy_frame_dir = VIDEO_FRAME_DIR / args.policy
    front_dir = policy_frame_dir / "front"
    overhead_dir = policy_frame_dir / "overhead"
    if not args.no_record:
        for directory in (front_dir, overhead_dir):
            directory.mkdir(parents=True, exist_ok=True)
            for path in directory.glob("*.png"):
                path.unlink()
    checkpoint = args.model_checkpoint
    if checkpoint is not None:
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint}")
        if args.policy != "moe16":
            raise ValueError("--model-checkpoint is only valid with --policy moe16")
    worker, client = start_inference_worker(args.policy, checkpoint, args.moe_device)
    world = None
    try:
        world, robot, target, front_camera, overhead_camera, controller = build_scene(with_overhead=True)
        decision_rows = []
        episode_rows = []
        frame_rows = []
        frame_number = 0
        rng = np.random.default_rng(args.seed)

        for episode_number in range(1, args.episodes + 1):
            case = cases[episode_number - 1] if cases is not None else None
            episode_id = str(case["case_id"]) if case is not None else f"closed_{episode_number:02d}"
            if case is None:
                initial_yaw, initial_bearing, initial_distance = episode_initial_conditions(episode_number, rng)
            else:
                initial_yaw = float(case["yaw_rad"])
                initial_bearing = float(case["bearing_rad"])
                initial_distance = float(case["distance_m"])
                if abs(initial_yaw) > 0.45 or abs(initial_bearing) > 0.65 or not 1.4 <= initial_distance <= 2.8:
                    raise ValueError(f"Case {episode_id} initial condition outside supported evaluation range")
                apply_case_appearance(world, target, case)
            robot_position = np.array([0.0, 0.0, ROBOT_START_Z], dtype=np.float32)
            robot.set_world_pose(position=robot_position, orientation=yaw_quaternion(initial_yaw))
            robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
            target_xy = place_target_relative(target, robot_position, initial_yaw, initial_bearing, initial_distance)
            controller.reset()

            for _ in range(30):
                robot.apply_wheel_actions(controller.forward(command=[0.0, 0.0]))
                update_front_camera(front_camera, robot)
                world.step(render=True)
            history = deque(maxlen=args.smoothing_window)
            min_distance = math.inf
            stopped = False
            stop_streak = 0
            start_sim_seconds = 0.0
            for decision_index in range(1, args.max_decisions + 1):
                rgb = camera_rgb(front_camera)
                if rgb is None or int(rgb.max()) == 0:
                    for _ in range(2):
                        update_front_camera(front_camera, robot)
                        world.step(render=True)
                    rgb = camera_rgb(front_camera)
                if rgb is None or int(rgb.max()) == 0:
                    raise RuntimeError(f"No RGB frame available for {episode_id} decision {decision_index}")

                response = client.decide(rgb)
                raw_action = response["action"]
                history.append(raw_action)
                action = raw_action if args.smoothing_window == 1 else majority_vote(history)
                stop_streak = stop_streak + 1 if raw_action == "STOP" else 0
                if action == "STOP" and stop_streak < 2:
                    non_stop_actions = [candidate for candidate in history if candidate != "STOP"]
                    action = non_stop_actions[-1] if non_stop_actions else "FORWARD"
                confidence = float(response["confidence"])
                latency = response["latency_ms"]
                distance, bearing, yaw = target_relative_state(robot, target)
                position = robot.get_world_pose()[0]
                min_distance = min(min_distance, distance)
                decision_rows.append(
                    {
                        "episode_id": episode_id,
                        "decision": decision_index,
                        "raw_prediction": raw_action,
                        "smoothed_action": action,
                        "confidence": confidence,
                        "distance_for_evaluation_m": distance,
                        "bearing_for_evaluation_rad": bearing,
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "yaw_rad": yaw,
                        **{f"latency_{key}_ms": value for key, value in latency.items()},
                    }
                )

                front_path = ""
                overhead_path = ""
                if not args.no_record:
                    front_path = (Path("artifacts") / "video_frames" / args.policy / "front" / f"{frame_number:05d}.png").as_posix()
                    overhead_rgb = camera_rgb(overhead_camera)
                    if overhead_rgb is None:
                        overhead_rgb = np.zeros_like(rgb)
                    overhead_path = (Path("artifacts") / "video_frames" / args.policy / "overhead" / f"{frame_number:05d}.png").as_posix()
                    from PIL import Image

                    Image.fromarray(rgb).save(ROOT / front_path)
                    Image.fromarray(overhead_rgb).save(ROOT / overhead_path)
                    frame_rows.append(
                        {
                            "frame_index": frame_number,
                            "episode_id": episode_id,
                            "decision": decision_index,
                            "front_path": front_path,
                            "overhead_path": overhead_path,
                            "raw_prediction": raw_action,
                            "action": action,
                            "confidence": confidence,
                            "latency_ms": latency["client_round_trip"],
                            "policy": args.policy,
                        }
                    )
                    frame_number += 1

                if action == "STOP":
                    robot.apply_wheel_actions(controller.forward(command=[0.0, 0.0]))
                    for _ in range(12):
                        update_front_camera(front_camera, robot)
                        world.step(render=True)
                    stopped = True
                    break

                linear, angular = action_command(action)
                for _ in range(args.steps_per_decision):
                    update_front_camera(front_camera, robot)
                    robot.apply_wheel_actions(controller.forward(command=[linear, angular]))
                    world.step(render=True)

            for _ in range(12):
                robot.apply_wheel_actions(controller.forward(command=[0.0, 0.0]))
                update_front_camera(front_camera, robot)
                world.step(render=True)
            final_distance, final_bearing, _ = target_relative_state(robot, target)
            collisions = int(min_distance < 0.30)
            success = bool(stopped and 1.20 <= final_distance <= 1.65 and abs(final_bearing) <= 0.18 and collisions == 0)
            latency_values = [row["latency_client_round_trip_ms"] for row in decision_rows if row["episode_id"] == episode_id]
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "initial_yaw_rad": initial_yaw,
                    "initial_target_bearing_rad": initial_bearing,
                    "initial_distance_m": initial_distance,
                    "dome_intensity": float(case["dome_intensity"]) if case is not None and "dome_intensity" in case else 1200.0,
                    "target_scale_factor": float(case.get("target_scale_factor", 1.0)) if case is not None else 1.0,
                    "target_color_rgb": json.dumps(case["target_color_rgb"]) if case is not None and "target_color_rgb" in case else "[1.0,0.08,0.04]",
                    "target_x": float(target_xy[0]),
                    "target_y": float(target_xy[1]),
                    "success": int(success),
                    "completion_time_sim_s": round(len(latency_values) * args.steps_per_decision * DT, 3),
                    "collision_count_geometry_proxy": collisions,
                    "number_of_decisions": len(latency_values),
                    "mean_decision_latency_ms": float(np.mean(latency_values)) if latency_values else 0.0,
                    "median_decision_latency_ms": float(np.median(latency_values)) if latency_values else 0.0,
                    "final_distance_m": final_distance,
                    "final_bearing_rad": final_bearing,
                    "predicted_stop": int(stopped),
                    "smoothing_window": args.smoothing_window,
                }
            )
            print(
                f"EPISODE {episode_id} success={int(success)} stop={int(stopped)} decisions={len(latency_values)} "
                f"distance={final_distance:.3f}m bearing={final_bearing:.3f}rad collisions={collisions}",
                flush=True,
            )

        def write_csv(path, rows):
            if not rows:
                return
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        if cases is not None:
            manifest_stem = args.case_manifest.stem
            run_tag = args.run_tag or f"robust_{args.policy}_{manifest_stem}_{args.case_start_index:02d}_{args.case_start_index + len(cases):02d}"
            episode_output = ROOT / "results" / f"{run_tag}_episodes.csv"
            decision_output = ROOT / "results" / f"{run_tag}_decisions.csv"
            summary_output = ROOT / "results" / f"{run_tag}_summary.json"
        else:
            episode_output = ROOT / "results" / "episode_results.csv"
            decision_output = ROOT / "results" / "decision_log.csv"
            summary_output = ROOT / "results" / "closed_loop_summary.json"
        write_csv(episode_output, episode_rows)
        write_csv(decision_output, decision_rows)
        if frame_rows:
            write_csv(ROOT / "results" / "demo_frames.csv", frame_rows)
        success_count = sum(row["success"] for row in episode_rows)
        summary = {
            "episodes": len(episode_rows),
            "successes": success_count,
            "failures": len(episode_rows) - success_count,
            "success_percent": 100.0 * success_count / max(1, len(episode_rows)),
            "label_source": "model predictions only for control; simulator pose used only for evaluation",
            "collision_metric": "geometry proxy: any robot-target center distance below 0.30 m",
            "control_hz_nominal": 1.0 / (args.steps_per_decision * DT),
            "smoothing_window": args.smoothing_window,
            "policy": args.policy,
            "case_manifest": str(args.case_manifest.as_posix()) if args.case_manifest else None,
            "case_ids": case_ids,
            "stop_confirmation_count": 2,
            "success_distance_band_m": [1.20, 1.65],
            "success_bearing_tolerance_rad": 0.18,
            "seed": args.seed,
            "dataset_version": json.loads((ROOT / "data" / "dataset_summary.json").read_text(encoding="utf-8")).get("dataset_version", "unspecified"),
        }
        summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("CLOSED_LOOP", json.dumps(summary, sort_keys=True), flush=True)
    finally:
        try:
            client.connection.sendall(struct.pack("!I", 0))
            client.connection.close()
        except Exception:
            pass
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
        if world is not None:
            world.stop()
        app.close()


if __name__ == "__main__":
    main()
