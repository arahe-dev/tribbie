"""Run a six-arm relay using the adapted 16-D visual MoE and record annotated frames."""

from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import math
import os
import socket
import struct
import subprocess
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np
from PIL import Image, ImageDraw

from throw_catch_scene import (
    ACTION_EVERY_STEPS,
    ARM_COUNT,
    BALL_Z,
    DT,
    HANDOFF_DWELL_S,
    VIDEO_EVERY_STEPS,
    build_relay_scene,
    camera_rgb,
    flight_position,
    pose_arm,
    set_arm_leds,
    station_xy,
)


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = ROOT / "src" / "throw_catch_inference_server.py"
FRAMES = ROOT / "artifacts" / "throw_catch_6arm_frames"
VIDEO = ROOT / "artifacts" / "demo_throw_catch_6arm.mp4"


class InferenceClient:
    def __init__(self, connection):
        self.connection = connection
        self.buffer = bytearray()

    def decide(self, rgb):
        if rgb is None or rgb.shape != (720, 720, 3):
            raise RuntimeError(f"Expected a 720x720 RGB frame, got {None if rgb is None else rgb.shape}")
        payload = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
        started = time.perf_counter_ns()
        self.connection.sendall(struct.pack("!I", len(payload)) + payload)
        while b"\n" not in self.buffer:
            block = self.connection.recv(4096)
            if not block:
                raise ConnectionError("Relay inference worker closed its connection")
            self.buffer.extend(block)
        line, _, tail = self.buffer.partition(b"\n")
        self.buffer = bytearray(tail)
        result = json.loads(line.decode("utf-8"))
        result["latency_ms"]["client_round_trip"] = (time.perf_counter_ns() - started) / 1e6
        return result


def start_worker(device):
    if not VENV_PYTHON.is_file():
        raise FileNotFoundError(f"Missing project environment: {VENV_PYTHON}")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "CARB_APP_PATH", "ISAAC_PATH", "EXP_PATH"):
        env.pop(key, None)
    command = [str(VENV_PYTHON), "-u", str(SERVER), "--port", "0", "--device", device]
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        deadline = time.monotonic() + 120.0
        ready = ""
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if line:
                ready = line.strip()
                print(f"INFERENCE_WORKER {ready}", flush=True)
                if ready.startswith("READY "):
                    break
            elif process.poll() is not None:
                raise RuntimeError(f"Relay inference worker exited with code {process.returncode}")
        if not ready.startswith("READY "):
            raise TimeoutError("Relay inference worker did not become ready")
        port = int(ready.split()[1])
        connection = socket.create_connection(("127.0.0.1", port), timeout=30.0)
        connection.settimeout(30.0)
        return process, InferenceClient(connection)
    except Exception:
        process.terminate()
        process.wait(timeout=10)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfers", type=int, default=18, help="Number of station-to-station throws")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=261005)
    args = parser.parse_args()
    if args.transfers < 6:
        raise ValueError("Use at least 6 transfers to show one complete circuit")
    FRAMES.mkdir(parents=True, exist_ok=True)
    for old_frame in FRAMES.glob("*.png"):
        old_frame.unlink()

    worker, client = start_worker(args.device)
    world = None
    try:
        world, camera, arms, ball = build_relay_scene()
        active = 0
        ball.set([*station_xy(active), BALL_Z + 0.26])
        set_arm_leds(arms, receiving=active, holding=True, ready=True)

        sim_step = 0
        frame_index = 0
        latest_rgb = None
        current_state = "READY"
        transfer_index = 0
        last_prediction = {"action": "WAIT", "confidence": 0.0, "latency_ms": {"client_round_trip": 0.0}}
        action_counts = Counter()
        event_rows = []
        latency_values = []
        throw_signals = 0
        catch_signals = 0
        hold_signals = 0
        throw_fallbacks = 0
        catch_fallbacks = 0

        def save_video_frame():
            nonlocal frame_index, latest_rgb
            rgb = camera_rgb(camera)
            if rgb is None:
                return
            latest_rgb = rgb
            canvas = Image.fromarray(rgb, mode="RGB")
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((12, 12, 455, 132), fill=(13, 21, 33))
            prediction = last_prediction["action"]
            confidence = float(last_prediction.get("confidence", 0.0))
            inference_ms = float(last_prediction.get("latency_ms", {}).get("client_round_trip", 0.0))
            ring = transfer_index % ARM_COUNT
            draw.text((25, 23), "SIX-ARM LATENT MoE RELAY", fill=(245, 248, 252))
            draw.text((25, 47), f"TRANSFER {transfer_index + 1:02d}/{args.transfers}  ACTIVE ARM {ring + 1}/6  STATE {current_state}", fill=(120, 220, 190))
            draw.text((25, 72), f"MODEL ACTION {prediction}  confidence {confidence:.2f}  round-trip {inference_ms:.2f} ms", fill=(255, 224, 150))
            draw.text((25, 97), "16-D latent  |  4 experts  |  actions: WAIT / CATCH / HOLD / THROW", fill=(220, 230, 244))
            canvas.save(FRAMES / f"frame_{frame_index:05d}.png")
            frame_index += 1

        def step_once():
            nonlocal sim_step
            world.step(render=True)
            sim_step += 1
            if sim_step % VIDEO_EVERY_STEPS == 0:
                save_video_frame()

        def inference_cycle(state):
            nonlocal last_prediction
            for _ in range(ACTION_EVERY_STEPS):
                step_once()
            rgb = latest_rgb if latest_rgb is not None else camera_rgb(camera)
            last_prediction = client.decide(rgb)
            action = last_prediction["action"]
            action_counts[action] += 1
            latency_values.append(float(last_prediction["latency_ms"]["client_round_trip"]))
            event_rows.append({
                "transfer": transfer_index + 1,
                "state": state,
                "model_action": action,
                "confidence": float(last_prediction["confidence"]),
                "client_round_trip_ms": float(last_prediction["latency_ms"]["client_round_trip"]),
                "gate_weights": json.dumps(last_prediction.get("gate_weights", [])),
            })
            print(f"POLICY transfer={transfer_index + 1} state={state} action={action} confidence={last_prediction['confidence']:.3f} latency_ms={last_prediction['latency_ms']['client_round_trip']:.2f}", flush=True)
            return action

        for transfer_index in range(args.transfers):
            target = (active + 1) % ARM_COUNT
            set_arm_leds(arms, receiving=active, holding=True, ready=True)
            pose_arm(arms[active], "idle", 0.0)
            current_state = "READY TO THROW"
            throw_seen = False
            for _ in range(6):
                if inference_cycle("ready") == "THROW":
                    throw_seen = True
                    throw_signals += 1
                    break
            if not throw_seen:
                throw_fallbacks += 1
                print(f"POLICY_FALLBACK transfer={transfer_index + 1} reason=no_THROW_signal", flush=True)

            # Wind up and extend the source arm before releasing the scripted ball arc.
            current_state = "WIND-UP / RELEASE"
            for tick in range(12):
                pose_arm(arms[active], "throw", tick / 11.0)
                step_once()
            source = active
            active = target
            set_arm_leds(arms, receiving=active, holding=False, ready=False)
            for index in range(ARM_COUNT):
                if index != source:
                    pose_arm(arms[index], "idle", 0.0)

            current_state = f"IN FLIGHT TO ARM {target + 1}"
            flight_steps = 36 + 2 * (transfer_index % 6)
            peak = 0.36 + 0.045 * (transfer_index % 6)
            elapsed = 0
            saw_catch_signal = False
            while elapsed < flight_steps:
                advance = min(ACTION_EVERY_STEPS, flight_steps - elapsed)
                fraction = (elapsed + advance) / flight_steps
                ball.set(flight_position(source, target, fraction, peak))
                for _ in range(advance):
                    step_once()
                elapsed += advance
                action = inference_cycle("flight") if (elapsed % ACTION_EVERY_STEPS == 0 or elapsed == flight_steps) else last_prediction["action"]
                if fraction >= 0.72 and action == "CATCH":
                    saw_catch_signal = True
                    catch_signals += 1

            # At contact the receiver closes around the ball only after the model's catch cue;
            # if it misses, the simulator completes a safe endpoint catch and logs the miss.
            for _ in range(3):
                if last_prediction["action"] == "CATCH":
                    saw_catch_signal = True
                    break
                action = inference_cycle("contact_wait")
                if action == "CATCH":
                    saw_catch_signal = True
                    catch_signals += 1
                    break
            current_state = f"CATCH ARM {target + 1}"
            for tick in range(10):
                pose_arm(arms[target], "catch", tick / 9.0)
                ball.set([*station_xy(target), BALL_Z + 0.26])
                step_once()
            ball.set([*station_xy(target), BALL_Z + 0.26])
            set_arm_leds(arms, receiving=target, holding=True, ready=False)
            if not saw_catch_signal:
                catch_fallbacks += 1
                print(f"POLICY_FALLBACK transfer={transfer_index + 1} reason=no_CATCH_signal", flush=True)

            current_state = f"HELD AT ARM {target + 1}"
            hold_cycles = max(2, math.ceil(HANDOFF_DWELL_S / (ACTION_EVERY_STEPS * DT)))
            for _ in range(hold_cycles):
                if inference_cycle("held") == "HOLD":
                    hold_signals += 1
            if transfer_index + 1 < args.transfers:
                active = target
                pose_arm(arms[active], "idle", 0.0)
                set_arm_leds(arms, receiving=active, holding=True, ready=True)
            else:
                current_state = "RELAY COMPLETE"

        # Hold on the final catch long enough for a readable closing beat.
        current_state = "RELAY COMPLETE"
        for _ in range(45):
            step_once()

        summary = {
            "task": "six-arm circular throw/catch relay",
            "requested_transfers": args.transfers,
            "completed_simulated_transfers": args.transfers,
            "complete_circuits": args.transfers / ARM_COUNT,
            "policy_throw_signals": throw_signals,
            "policy_catch_signals": catch_signals,
            "policy_hold_signals": hold_signals,
            "model_triggered_throw_transfers": args.transfers - throw_fallbacks,
            "model_triggered_catch_transfers": args.transfers - catch_fallbacks,
            "throw_fallbacks": throw_fallbacks,
            "catch_fallbacks": catch_fallbacks,
            "policy_action_counts": dict(action_counts),
            "mean_inference_round_trip_ms": float(np.mean(latency_values)) if latency_values else 0.0,
            "median_inference_round_trip_ms": float(np.median(latency_values)) if latency_values else 0.0,
            "mean_after_first_inference_ms": float(np.mean(latency_values[1:])) if len(latency_values) > 1 else 0.0,
            "median_after_first_inference_ms": float(np.median(latency_values[1:])) if len(latency_values) > 1 else 0.0,
            "p95_inference_round_trip_ms": float(np.percentile(latency_values, 95)) if latency_values else 0.0,
            "frame_count": frame_index,
            "video_fps": 30,
            "video_duration_seconds": frame_index / 30.0,
            "video_path": VIDEO.relative_to(ROOT).as_posix(),
            "model_checkpoint": "models/throw_catch_moe16.pt",
            "controller_input": "rendered 720x720 RGB resized to checkpoint-configured 224x224",
            "limitations": [
                "Arm links are stylized kinematic USD geometry, not articulated robot actuators.",
                "Ball flight is a scripted arc; the model selects relay actions but does not predict ball physics.",
                "A simulator endpoint catch completes safely if the visual policy emits no CATCH cue; see catch_fallbacks.",
            ],
        }
        RESULTS = ROOT / "results"
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "throw_catch_demo.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        with (RESULTS / "throw_catch_policy_trace.csv").open("w", newline="", encoding="utf-8") as stream:
            if event_rows:
                writer = csv.DictWriter(stream, fieldnames=event_rows[0].keys())
                writer.writeheader()
                writer.writerows(event_rows)
        print("DEMO_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
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
