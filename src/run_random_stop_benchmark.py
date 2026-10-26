"""Paired, long-horizon randomized emergency-stop benchmark in Isaac Sim."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import secrets
import socket
import struct
import subprocess
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np
from PIL import Image
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
    yaw_from_quaternion,
    yaw_quaternion,
)


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = ROOT / "src" / "inference_server.py"
MOE_CHECKPOINT = ROOT / "models" / "random_stop_moe_v2.pt"
PCA_MODELS_DIR = ROOT / "models" / "random_stop_pca16_v2"
DT_MS = DT * 1000.0
GO_DISTANCE_M = 2.05
COLLISION_PROXY_M = 0.30
IMMEDIATE_RESPONSE_S = 0.25
DEFAULT_TARGET_COLOR = np.array([0.08, 0.80, 0.08], dtype=np.float32)


class InferenceClient:
    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.buffer = bytearray()

    def decide(self, rgb: np.ndarray) -> dict:
        payload = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8).tobytes()
        started = time.perf_counter_ns()
        self.connection.sendall(struct.pack("!I", len(payload)) + payload)
        while b"\n" not in self.buffer:
            block = self.connection.recv(4096)
            if not block:
                raise ConnectionError("The inference worker closed its connection")
            self.buffer.extend(block)
        line, _, tail = self.buffer.partition(b"\n")
        self.buffer = bytearray(tail)
        result = json.loads(line.decode("utf-8"))
        result["latency_ms"]["client_round_trip"] = (time.perf_counter_ns() - started) / 1e6
        return result


def start_worker(policy: str, moe_device: str):
    if not VENV_PYTHON.is_file():
        raise FileNotFoundError(f"Missing project Python environment: {VENV_PYTHON}")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "CARB_APP_PATH", "ISAAC_PATH", "EXP_PATH"):
        env.pop(key, None)
    command = [
        str(VENV_PYTHON), "-u", str(SERVER), "--port", "0", "--policy", policy,
        "--moe-device", moe_device,
    ]
    if policy == "moe16":
        command.extend(["--moe-checkpoint", str(MOE_CHECKPOINT)])
    else:
        command.extend(["--models-dir", str(PCA_MODELS_DIR)])
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        deadline = time.monotonic() + 120.0
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
        connection = socket.create_connection(("127.0.0.1", int(ready_line.split()[1])), timeout=30)
        connection.settimeout(60)
        return process, InferenceClient(connection)
    except Exception:
        process.terminate()
        process.wait(timeout=10)
        raise


def make_episode(seed: int, episode_id: str, event_count: int):
    rng = np.random.default_rng(seed)
    yaw = float(rng.uniform(-0.12, 0.12))
    # Keep event hues camera-visible while varying exposure within the training range.
    dome_intensity = float(rng.uniform(1300.0, 1600.0))
    first_onset = float(rng.uniform(1.0, 2.0))
    cursor = first_onset
    events = []
    for event_index in range(1, event_count + 1):
        duration = float(rng.uniform(1.8, 2.7))
        color = [
            float(rng.uniform(0.93, 1.0)),
            float(rng.uniform(0.08, 0.13)),
            float(rng.uniform(0.04, 0.075)),
        ]
        events.append({
            "event_id": f"{episode_id}_event_{event_index:03d}",
            "episode_id": episode_id,
            "index": event_index,
            "scheduled_onset_sim_s": round(cursor, 4),
            "duration_s": round(duration, 4),
            "distance_m": round(float(rng.uniform(1.08, 1.42)), 4),
            "bearing_rad": round(float(rng.uniform(-0.14, 0.14)), 5),
            "target_color_rgb": color,
        })
        cursor += duration + float(rng.uniform(1.7, 3.8))
    return {
        "episode_id": episode_id,
        "seed": int(seed),
        "initial_yaw_rad": yaw,
        "dome_intensity": round(dome_intensity, 1),
        "events": events,
    }


def set_target_pose(target, robot, distance: float, bearing: float):
    position, orientation = robot.get_world_pose()
    yaw = yaw_from_quaternion(orientation)
    target_xy = place_target_relative(target, position, yaw, bearing, distance)
    return target_xy


def set_target_color(target, color):
    material = target.get_applied_visual_material()
    if material is None:
        raise RuntimeError("The randomized stop target has no mutable visual material")
    material.set_color(np.asarray(color, dtype=np.float32))


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else None


def cue_pixel_measure(rgb, distance_m, bearing_rad):
    pixels = np.asarray(rgb, dtype=np.uint8)[:, :, :3].astype(np.int16)
    red, green, blue = [pixels[:, :, channel] for channel in range(3)]
    image_h, image_w = pixels.shape[:2]
    focal_px_x = image_w * 0.8 / 2.0955
    focal_px_y = image_h * 0.8 / 1.571625
    distance = max(float(distance_m), 0.25)
    center_x = int(round(image_w / 2.0 - focal_px_x * math.tan(float(bearing_rad))))
    center_y = int(round(image_h / 2.0 - focal_px_y * (0.45 - 0.42) / distance))
    half_width = int(math.ceil(focal_px_x * 0.09 / distance)) + 5
    half_height = int(math.ceil(focal_px_y * 0.45 / distance)) + 5
    x0, x1 = max(0, center_x - half_width), min(image_w, center_x + half_width + 1)
    y0, y1 = max(0, center_y - half_height), min(image_h, center_y + half_height + 1)
    roi = pixels[y0:y1, x0:x1]
    if roi.size == 0:
        return 0, 0, 0, False
    roi_red, roi_green, roi_blue = [roi[:, :, channel] for channel in range(3)]
    red_mask = (roi_red >= 70) & (roi_red >= roi_green * 1.12) & (roi_red >= roi_blue * 1.12)
    roi_max = roi.max(axis=2)
    roi_min = roi.min(axis=2)
    dark_mask = (roi_max <= 80) & ((roi_max - roi_min) <= 45)
    red_pixels = int(red_mask.sum())
    dark_pixels = int(dark_mask.sum())
    expected_area = (focal_px_x * 0.18 / distance) * (focal_px_y * 0.90 / distance)
    dark_threshold = max(80, int(expected_area * 0.20))
    return max(red_pixels, dark_pixels), red_pixels, dark_pixels, red_pixels >= 25 or dark_pixels >= dark_threshold


def run_episode(
    world, robot, target, camera, controller, client, policy, episode,
    steps_per_decision, capture_dir=None, rgb_frame_dir=None,
):
    episode_id = episode["episode_id"]
    robot.set_world_pose(
        position=np.array([0.0, 0.0, ROBOT_START_Z], dtype=np.float32),
        orientation=yaw_quaternion(episode["initial_yaw_rad"]),
    )
    robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
    robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
    set_target_color(target, DEFAULT_TARGET_COLOR)
    dome_prim = world.stage.GetPrimAtPath("/World/DomeLight")
    dome = UsdLux.DomeLight(dome_prim)
    dome.GetIntensityAttr().Set(float(episode["dome_intensity"]))
    set_target_pose(target, robot, GO_DISTANCE_M, 0.0)
    controller.reset()

    for _ in range(30):
        robot.apply_wheel_actions(controller.forward(command=[0.0, 0.0]))
        update_front_camera(camera, robot)
        world.step(render=True)

    sim_steps = 0
    latency_credit_s = 0.0
    current_action = "FORWARD"
    last_render_event_id = None
    next_event_index = 0
    active_event = None
    event_by_id = {event["event_id"]: event for event in episode["events"]}
    event_rows = []
    decision_rows = []
    false_stop_predictions = 0
    false_stop_time_s = 0.0
    latency_values = []
    rounds = 0
    captured_event_ids = set()
    event_observation_counts = {}
    rendered_event_observation_counts = {}
    first_visible_cue_captured = set()

    def now_s():
        return sim_steps * DT

    def finalize_event(event):
        collision = bool(event.get("collision_proxy_hit", False))
        reaction = event.get("reaction_delay_sim_s")
        cue_reaction = event.get("reaction_delay_from_visible_s")
        event.update({
            "policy": policy,
            "actual_onset_sim_s": round(event["actual_onset_sim_s"], 5),
            "first_observed_sim_s": round(event["first_observed_sim_s"], 5) if event.get("first_observed_sim_s") is not None else None,
            "first_stop_applied_sim_s": round(event["first_stop_applied_sim_s"], 5) if event.get("first_stop_applied_sim_s") is not None else None,
            "cue_appearance_delay_sim_s": round(event["first_observed_sim_s"] - event["actual_onset_sim_s"], 5) if event.get("first_observed_sim_s") is not None else None,
            "reaction_delay_sim_s": round(reaction, 5) if reaction is not None else None,
            "reaction_delay_from_visible_cue_sim_s": round(cue_reaction, 5) if cue_reaction is not None else None,
            "minimum_center_distance_m": round(event["minimum_center_distance_m"], 5),
            "collision_proxy_hit": int(collision),
            "responded": int(reaction is not None),
            "immediate_response_250ms": int(reaction is not None and reaction <= IMMEDIATE_RESPONSE_S),
            "cue_response_250ms": int(cue_reaction is not None and cue_reaction <= IMMEDIATE_RESPONSE_S),
            "safe_stop_success": int(reaction is not None and reaction <= IMMEDIATE_RESPONSE_S and not collision),
            "safe_stop_success_from_cue": int(cue_reaction is not None and cue_reaction <= IMMEDIATE_RESPONSE_S and not collision),
        })
        event_rows.append(event)

    def ensure_event_state():
        nonlocal active_event, next_event_index
        current = now_s()
        if active_event is not None and current >= active_event["end_sim_s"]:
            finalize_event(active_event)
            active_event = None
            set_target_color(target, DEFAULT_TARGET_COLOR)
            set_target_pose(target, robot, GO_DISTANCE_M, 0.0)
        if active_event is None and next_event_index < len(episode["events"]):
            planned = episode["events"][next_event_index]
            if current >= planned["scheduled_onset_sim_s"]:
                active_event = dict(planned)
                active_event["actual_onset_sim_s"] = current
                active_event["end_sim_s"] = current + float(planned["duration_s"])
                active_event["minimum_center_distance_m"] = float("inf")
                active_event["collision_proxy_hit"] = False
                active_event["first_observed_sim_s"] = None
                active_event["first_stop_applied_sim_s"] = None
                active_event["reaction_delay_sim_s"] = None
                active_event["reaction_delay_from_visible_s"] = None
                active_event["already_stopped_before_first_visible_cue"] = 0
                event_by_id[active_event["event_id"]] = active_event
                set_target_color(target, planned["target_color_rgb"])
                set_target_pose(target, robot, float(planned["distance_m"]), float(planned["bearing_rad"]))
                next_event_index += 1
                print(
                    f"EVENT_ONSET {policy} {active_event['event_id']} t={current:.3f}s "
                    f"distance={planned['distance_m']:.2f}m bearing={planned['bearing_rad']:.3f}rad",
                    flush=True,
                )
        if active_event is None:
            # Keep the non-event image in the training distribution without giving
            # the policy a hazard schedule or a privileged state input.
            set_target_pose(target, robot, GO_DISTANCE_M, 0.0)

    def physics_step(render):
        nonlocal sim_steps, last_render_event_id, false_stop_time_s
        ensure_event_state()
        if active_event is not None:
            distance, _, _ = target_relative_state(robot, target)
            active_event["minimum_center_distance_m"] = min(active_event["minimum_center_distance_m"], distance)
            if distance <= COLLISION_PROXY_M:
                active_event["collision_proxy_hit"] = True
        linear, angular = action_command(current_action)
        if current_action == "FORWARD":
            linear = 0.35
        robot.apply_wheel_actions(controller.forward(command=[linear, angular]))
        update_front_camera(camera, robot)
        world.step(render=render)
        sim_steps += 1
        if render:
            last_render_event_id = active_event["event_id"] if active_event is not None else None

    max_sim_time = episode["events"][-1]["scheduled_onset_sim_s"] + episode["events"][-1]["duration_s"] + 2.0
    while now_s() < max_sim_time and len(event_rows) < len(episode["events"]):
        ensure_event_state()
        rgb = camera_rgb(camera)
        if rgb is None or int(rgb.max()) == 0:
            raise RuntimeError(f"No valid RGB camera frame in {episode_id} at {now_s():.3f}s")
        capture_time = now_s()
        rendered_event_id = last_render_event_id
        if rgb_frame_dir is not None:
            rgb_frame_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb[:, :, :3]).save(
                rgb_frame_dir / f"frame_{rounds:05d}.jpg", format="JPEG", quality=88
            )
        cue_distance, cue_bearing, _ = target_relative_state(robot, target)
        cue_pixels, red_pixels, dark_pixels, cue_visible = cue_pixel_measure(rgb, cue_distance, cue_bearing)
        observed_event_id = rendered_event_id if cue_visible else None
        if capture_dir is not None:
            capture_dir.mkdir(parents=True, exist_ok=True)
            if rounds == 0:
                Image.fromarray(rgb[:, :, :3]).save(capture_dir / f"{policy}_{episode_id}_green_start.png")
            if rendered_event_id and rendered_event_id not in captured_event_ids:
                Image.fromarray(rgb[:, :, :3]).save(
                    capture_dir / f"{policy}_{rendered_event_id}_first_rendered_event.png"
                )
                captured_event_ids.add(rendered_event_id)
            if rendered_event_id:
                rendered_count = rendered_event_observation_counts.get(rendered_event_id, 0) + 1
                rendered_event_observation_counts[rendered_event_id] = rendered_count
                if rendered_count in (2, 5, 10, 15, 30):
                    Image.fromarray(rgb[:, :, :3]).save(
                        capture_dir / f"{policy}_{rendered_event_id}_rendered_observation_{rendered_count:02d}.png"
                    )
            if observed_event_id:
                observation_count = event_observation_counts.get(observed_event_id, 0) + 1
                event_observation_counts[observed_event_id] = observation_count
                if observation_count in (1, 3, 5, 10, 20):
                    Image.fromarray(rgb[:, :, :3]).save(
                        capture_dir / f"{policy}_{observed_event_id}_observation_{observation_count:02d}.png"
                    )
                captured_event_ids.add(observed_event_id)
            if observed_event_id and observed_event_id not in first_visible_cue_captured:
                Image.fromarray(rgb[:, :, :3]).save(
                    capture_dir / f"{policy}_{observed_event_id}_first_visible_cue.png"
                )
                first_visible_cue_captured.add(observed_event_id)
        response = client.decide(rgb)
        action = response["action"]
        if action not in ("LEFT", "FORWARD", "RIGHT", "STOP"):
            raise RuntimeError(f"Unexpected action from {policy}: {action}")
        # This benchmark tests an emergency stop, not route following: preserve
        # the straight approach while the policy decides whether to stop.
        commanded_action = action if action == "STOP" else "FORWARD"
        latency = response["latency_ms"]
        round_trip_ms = float(latency["client_round_trip"])
        latency_values.append(round_trip_ms)

        # Let the simulated robot continue under its last command while the
        # synchronous worker computes. The accumulator preserves sub-frame time.
        latency_credit_s += round_trip_ms / 1000.0
        while latency_credit_s >= DT:
            physics_step(render=False)
            latency_credit_s -= DT

        action_applied_time = now_s()
        observed_event = event_by_id.get(observed_event_id) if observed_event_id else None
        if observed_event is not None and observed_event.get("first_observed_sim_s") is None:
            observed_event["first_observed_sim_s"] = capture_time
            observed_event["already_stopped_before_first_visible_cue"] = int(current_action == "STOP")
            print(
                f"CUE_VISIBLE {policy} {observed_event_id} t={capture_time:.3f}s "
                f"cue_pixels={cue_pixels} red={red_pixels} dark={dark_pixels}",
                flush=True,
            )
        if action == "STOP":
            if observed_event is not None:
                # Only credit a reaction when a moving command transitions to
                # STOP after the cue is visible. A repeated STOP from a prior
                # false alarm is not a new reaction.
                if current_action != "STOP" and observed_event.get("first_stop_applied_sim_s") is None:
                    observed_event["first_stop_applied_sim_s"] = action_applied_time
                    observed_event["reaction_delay_sim_s"] = action_applied_time - observed_event["actual_onset_sim_s"]
                    observed_event["reaction_delay_from_visible_s"] = action_applied_time - observed_event["first_observed_sim_s"]
            else:
                false_stop_predictions += 1
        distance, bearing, yaw = target_relative_state(robot, target)
        position = robot.get_world_pose()[0]
        decision_rows.append({
            "policy": policy,
            "episode_id": episode_id,
            "decision": rounds + 1,
            "sim_time_s": round(capture_time, 5),
            "action_applied_sim_s": round(action_applied_time, 5),
            "observed_event_id": observed_event_id or "",
            "rendered_event_id": rendered_event_id or "",
            "camera_stop_cue_pixels": cue_pixels,
            "camera_red_cue_pixels_in_roi": red_pixels,
            "camera_dark_cue_pixels_in_roi": dark_pixels,
            "active_event_id": active_event["event_id"] if active_event is not None else "",
            "raw_action": action,
            "commanded_action": commanded_action,
            "confidence": float(response["confidence"]),
            "target_distance_for_evaluation_m": distance,
            "target_bearing_for_evaluation_rad": bearing,
            "x": float(position[0]),
            "y": float(position[1]),
            "yaw_rad": yaw,
            "latency_client_round_trip_ms": round_trip_ms,
            **{f"latency_{key}_ms": value for key, value in latency.items() if key != "client_round_trip"},
        })
        rounds += 1
        current_action = commanded_action
        if rounds in (1, 25, 50, 100, 200):
            print(f"DECISIONS {policy} {episode_id} n={rounds} t={capture_time:.3f}s", flush=True)
        if commanded_action == "STOP" and observed_event_id is None:
            false_stop_time_s += (round_trip_ms / 1000.0) + steps_per_decision * DT
        # One rendered frame per nominal decision. With the default two 60 Hz
        # physics ticks, policy observations arrive at 30 Hz.
        for step_index in range(steps_per_decision):
            physics_step(render=(step_index == steps_per_decision - 1))

    if active_event is not None:
        finalize_event(active_event)
    completed_ids = {row["event_id"] for row in event_rows}
    for event in episode["events"]:
        if event["event_id"] not in completed_ids:
            unfinished = dict(event)
            unfinished.update({
                "policy": policy,
                "actual_onset_sim_s": None,
                "first_observed_sim_s": None,
                "first_stop_applied_sim_s": None,
                "reaction_delay_sim_s": None,
                "reaction_delay_from_visible_cue_sim_s": None,
                "minimum_center_distance_m": None,
                "collision_proxy_hit": 0,
                "responded": 0,
                "immediate_response_250ms": 0,
                "cue_response_250ms": 0,
                "safe_stop_success": 0,
                "safe_stop_success_from_cue": 0,
                "already_stopped_before_first_visible_cue": 0,
                "incomplete": 1,
            })
            event_rows.append(unfinished)

    episode_summary = {
        "policy": policy,
        "episode_id": episode_id,
        "seed": episode["seed"],
        "events": len(episode["events"]),
        "completed_events": sum(row.get("actual_onset_sim_s") is not None for row in event_rows),
        "responses": sum(row.get("responded", 0) for row in event_rows),
        "immediate_responses_250ms": sum(row.get("immediate_response_250ms", 0) for row in event_rows),
        "cue_responses_250ms": sum(row.get("cue_response_250ms", 0) for row in event_rows),
        "safe_stop_successes": sum(row.get("safe_stop_success", 0) for row in event_rows),
        "safe_stop_successes_from_cue": sum(row.get("safe_stop_success_from_cue", 0) for row in event_rows),
        "collision_proxy_hits": sum(row.get("collision_proxy_hit", 0) for row in event_rows),
        "false_stop_predictions": false_stop_predictions,
        "false_stop_time_s": round(false_stop_time_s, 4),
        "decisions": rounds,
        "simulated_seconds": round(now_s(), 4),
        "median_request_latency_ms": percentile(latency_values, 50),
        "p95_request_latency_ms": percentile(latency_values, 95),
    }
    return event_rows, decision_rows, episode_summary


def summarize_policy(policy, events, episodes, decisions):
    reactions = [float(row["reaction_delay_sim_s"]) for row in events if row.get("reaction_delay_sim_s") is not None]
    cue_reactions = [float(row["reaction_delay_from_visible_cue_sim_s"]) for row in events if row.get("reaction_delay_from_visible_cue_sim_s") is not None]
    cue_delays = [float(row["cue_appearance_delay_sim_s"]) for row in events if row.get("cue_appearance_delay_sim_s") is not None]
    latencies = [float(row["latency_client_round_trip_ms"]) for row in decisions]
    visible_events = [row for row in events if row.get("first_observed_sim_s") is not None]
    return {
        "policy": policy,
        "episodes": len(episodes),
        "randomized_events": len(events),
        "event_responses": sum(int(row.get("responded", 0)) for row in events),
        "immediate_responses_250ms": sum(int(row.get("immediate_response_250ms", 0)) for row in events),
        "immediate_response_rate": float(np.mean([int(row.get("immediate_response_250ms", 0)) for row in events])) if events else 0.0,
        "cue_responses_250ms": sum(int(row.get("cue_response_250ms", 0)) for row in events),
        "cue_response_rate_250ms": float(np.mean([int(row.get("cue_response_250ms", 0)) for row in events])) if events else 0.0,
        "cue_response_rate_250ms_given_visible_cue": float(np.mean([int(row.get("cue_response_250ms", 0)) for row in visible_events])) if visible_events else None,
        "camera_visible_events": len(visible_events),
        "camera_visible_event_rate": float(len(visible_events) / len(events)) if events else 0.0,
        "safe_stop_successes": sum(int(row.get("safe_stop_success", 0)) for row in events),
        "safe_stop_success_rate": float(np.mean([int(row.get("safe_stop_success", 0)) for row in events])) if events else 0.0,
        "safe_stop_successes_from_cue": sum(int(row.get("safe_stop_success_from_cue", 0)) for row in events),
        "safe_stop_success_rate_from_cue": float(np.mean([int(row.get("safe_stop_success_from_cue", 0)) for row in events])) if events else 0.0,
        "safe_stop_success_rate_from_cue_given_visible_cue": float(np.mean([int(row.get("safe_stop_success_from_cue", 0)) for row in visible_events])) if visible_events else None,
        "collision_proxy_hits": sum(int(row.get("collision_proxy_hit", 0)) for row in events),
        "median_reaction_sim_s": percentile(reactions, 50),
        "p95_reaction_sim_s": percentile(reactions, 95),
        "median_reaction_from_visible_cue_sim_s": percentile(cue_reactions, 50),
        "p95_reaction_from_visible_cue_sim_s": percentile(cue_reactions, 95),
        "median_cue_appearance_delay_sim_s": percentile(cue_delays, 50),
        "p95_cue_appearance_delay_sim_s": percentile(cue_delays, 95),
        "false_stop_predictions": sum(int(row["false_stop_predictions"]) for row in episodes),
        "false_stop_time_s": round(sum(float(row["false_stop_time_s"]) for row in episodes), 4),
        "decisions": sum(int(row["decisions"]) for row in episodes),
        "median_request_latency_ms": percentile(latencies, 50),
        "p95_request_latency_ms": percentile(latencies, 95),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--events-per-episode", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional replay seed; omitted means a fresh random seed recorded in the manifest")
    parser.add_argument("--steps-per-decision", type=int, default=2, help="60 Hz physics ticks per 30 Hz visual decision")
    parser.add_argument("--policies", nargs="+", choices=("moe16", "pca16"), default=("moe16", "pca16"))
    parser.add_argument("--moe-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--run-tag", default="random_stop_long_20261006")
    parser.add_argument("--capture-event-frames", action="store_true",
                        help="Save the first non-event and first observed event RGB frame per episode")
    parser.add_argument("--capture-rgb-frames", action="store_true",
                        help="Save every policy RGB observation as JPEG for video review")
    args = parser.parse_args()
    if args.episodes < 1 or args.events_per_episode < 1 or args.steps_per_decision < 1:
        raise ValueError("episodes, events-per-episode, and steps-per-decision must be positive")
    if len(args.policies) != len(set(args.policies)):
        raise ValueError("Do not repeat a policy in --policies")
    if not args.run_tag.replace("_", "").replace("-", "").isalnum():
        raise ValueError("run-tag may contain only letters, numbers, underscores, and hyphens")
    if (not MOE_CHECKPOINT.is_file() or not PCA_MODELS_DIR.joinpath("pca_16.joblib").is_file()
            or not PCA_MODELS_DIR.joinpath("knn_16.joblib").is_file()):
        raise FileNotFoundError("Train/check the MoE checkpoint and matched PCA-16 baseline before the benchmark")

    effective_seed = int(args.seed) if args.seed is not None else secrets.randbits(32)
    seed_source = "user_supplied_replay_seed" if args.seed is not None else "fresh_os_random_seed"
    episodes = [
        make_episode(effective_seed + index * 1009, f"ep{index + 1:02d}", args.events_per_episode)
        for index in range(args.episodes)
    ]
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    manifest_path = results_dir / f"{args.run_tag}_manifest.json"
    manifest_path.write_text(json.dumps({
        "benchmark": "randomized reactive emergency stops",
        "seed": effective_seed,
        "seed_source": seed_source,
        "episodes": episodes,
        "policy_observation": "RGB camera only; event timing/pose schedule withheld from worker",
        "simulator_randomization": "event onset, duration, inter-event gap, distance, bearing, red cue hue, and per-episode lighting/yaw; the run seed is recorded for replay but is not sent to either policy",
        "paired_comparison": "same episode seeds and event schedule for both policies",
    }, indent=2) + "\n", encoding="utf-8")

    world = None
    all_summaries = {}
    try:
        world, robot, target, camera, _, controller = build_scene(with_overhead=False, static_stop_pole=True)
        capture_dir = results_dir / f"{args.run_tag}_frames" if args.capture_event_frames else None
        rgb_root = results_dir / f"{args.run_tag}_rgb_frames" if args.capture_rgb_frames else None
        for policy in args.policies:
            print(f"POLICY_START {policy}", flush=True)
            worker, client = start_worker(policy, args.moe_device)
            try:
                policy_events = []
                policy_decisions = []
                policy_episodes = []
                for episode in episodes:
                    print(f"EPISODE_START {policy} {episode['episode_id']}", flush=True)
                    event_rows, decision_rows, episode_summary = run_episode(
                        world, robot, target, camera, controller, client, policy, episode, args.steps_per_decision,
                        capture_dir=capture_dir,
                        rgb_frame_dir=rgb_root / f"{policy}_{episode['episode_id']}" if rgb_root is not None else None,
                    )
                    policy_events.extend(event_rows)
                    policy_decisions.extend(decision_rows)
                    policy_episodes.append(episode_summary)
                    print(
                        f"EPISODE {policy} {episode_summary['episode_id']} safe="
                        f"{episode_summary['safe_stop_successes']}/{episode_summary['events']} "
                        f"collisions={episode_summary['collision_proxy_hits']} "
                        f"false_stops={episode_summary['false_stop_predictions']} "
                        f"p95={episode_summary['p95_request_latency_ms']:.2f}ms",
                        flush=True,
                    )
                all_summaries[policy] = summarize_policy(policy, policy_events, policy_episodes, policy_decisions)
                write_csv(results_dir / f"{args.run_tag}_{policy}_events.csv", policy_events)
                write_csv(results_dir / f"{args.run_tag}_{policy}_episodes.csv", policy_episodes)
                write_csv(results_dir / f"{args.run_tag}_{policy}_decisions.csv", policy_decisions)
                (results_dir / f"{args.run_tag}_{policy}_summary.json").write_text(
                    json.dumps(all_summaries[policy], indent=2) + "\n", encoding="utf-8"
                )
                print("POLICY_SUMMARY", json.dumps(all_summaries[policy], sort_keys=True), flush=True)
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

        if "moe16" in all_summaries and "pca16" in all_summaries:
            paired = {"scheduled_onset": {}, "visible_cue": {}, "cue_visible": {}}
            for policy in ("moe16", "pca16"):
                with (results_dir / f"{args.run_tag}_{policy}_events.csv").open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                    paired["scheduled_onset"][policy] = {
                        (row["episode_id"], row["event_id"]): int(row["safe_stop_success"])
                        for row in rows
                    }
                    paired["visible_cue"][policy] = {
                        (row["episode_id"], row["event_id"]): int(row["safe_stop_success_from_cue"])
                        for row in rows
                    }
                    paired["cue_visible"][policy] = {
                        (row["episode_id"], row["event_id"]): int(row.get("first_observed_sim_s", "") != "")
                        for row in rows
                    }
            keys = sorted(set(paired["scheduled_onset"]["moe16"]) & set(paired["scheduled_onset"]["pca16"]))
            common_visible_keys = [key for key in keys if paired["cue_visible"]["moe16"][key] and paired["cue_visible"]["pca16"][key]]
            def paired_outcomes(metric, selected_keys=keys):
                outcomes = paired[metric]
                return {
                    "paired_event_count": len(selected_keys),
                    "moe_only_successes": sum(outcomes["moe16"][key] == 1 and outcomes["pca16"][key] == 0 for key in selected_keys),
                    "pca16_only_successes": sum(outcomes["moe16"][key] == 0 and outcomes["pca16"][key] == 1 for key in selected_keys),
                    "both_success": sum(outcomes["moe16"][key] == outcomes["pca16"][key] == 1 for key in selected_keys),
                    "both_failure": sum(outcomes["moe16"][key] == outcomes["pca16"][key] == 0 for key in selected_keys),
                }
            comparison = {
                "benchmark": "randomized reactive emergency stops",
                "seed": effective_seed,
                "seed_source": seed_source,
                "episodes": args.episodes,
                "events_per_episode": args.events_per_episode,
                "paired_event_count": len(keys),
                "models": all_summaries,
                "paired_safe_stop_outcomes": {
                    "from_scheduled_onset": paired_outcomes("scheduled_onset"),
                    "from_first_visible_cue": paired_outcomes("visible_cue"),
                    "from_first_visible_cue_on_events_visible_to_both": paired_outcomes("visible_cue", common_visible_keys),
                },
                "interpretation_limit": "Paired simulation study. Each default run uses a fresh OS-random seed; saved events are reproducible for analysis. The sequence is hidden from the policy, not an unseeded physical-world disturbance.",
            }
            (results_dir / f"{args.run_tag}_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
            print("COMPARISON", json.dumps(comparison, sort_keys=True), flush=True)
    finally:
        if world is not None:
            world.stop()
        app.close()


if __name__ == "__main__":
    main()
