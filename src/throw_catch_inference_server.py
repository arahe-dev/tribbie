"""Serve the relay MoE from the project venv, isolated from Isaac Kit's CUDA DLLs."""

from pathlib import Path
import argparse
import json
import socket
import struct
import time

import numpy as np
from PIL import Image
import torch

from moe_latent import MoELatentPolicy


ROOT = Path(__file__).resolve().parents[1]
FRAME_SHAPE = (720, 720, 3)


def recv_exact(connection, count):
    chunks = bytearray()
    while len(chunks) < count:
        block = connection.recv(count - len(chunks))
        if not block:
            return None
        chunks.extend(block)
    return bytes(chunks)


def serve(checkpoint_path: Path, port: int, requested_device: str):
    torch.set_num_threads(1)
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for relay inference, but no CUDA device is available")
    device_name = "cuda" if requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available()) else "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    classes = list(cfg["class_order"])
    model = MoELatentPolicy(
        latent_dim=int(cfg["latent_dim"]), num_experts=int(cfg["num_experts"]),
        num_classes=int(cfg["num_classes"]), dropout=float(cfg["dropout"]),
        spatial_pool=int(cfg["spatial_pool"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(1)
    listener.settimeout(120.0)
    print(f"READY {listener.getsockname()[1]} DEVICE={device} LATENT={cfg['latent_dim']}", flush=True)
    try:
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(30.0)
            while True:
                header = recv_exact(connection, 4)
                if header is None:
                    break
                length = struct.unpack("!I", header)[0]
                if length == 0:
                    break
                payload = recv_exact(connection, length)
                if payload is None:
                    break
                started = time.perf_counter_ns()
                frame = np.frombuffer(payload, dtype=np.uint8).reshape(FRAME_SHAPE)
                resized = Image.fromarray(frame, mode="RGB").resize((int(cfg["image_size"]), int(cfg["image_size"])), Image.Resampling.BILINEAR)
                array = np.asarray(resized, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_start = time.perf_counter_ns()
                with torch.inference_mode():
                    log_probs, gates, latent = model(tensor, return_aux=True)
                    probs = log_probs.exp()[0]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                probs = probs.detach().cpu().numpy()
                response = {
                    "action": classes[int(np.argmax(probs))],
                    "confidence": float(np.max(probs)),
                    "probabilities": probs.tolist(),
                    "gate_weights": gates[0].detach().cpu().numpy().tolist(),
                    "latent": latent[0].detach().cpu().numpy().tolist(),
                    "latency_ms": {
                        "inference": (time.perf_counter_ns() - inference_start) / 1e6,
                        "server_total": (time.perf_counter_ns() - started) / 1e6,
                    },
                }
                connection.sendall(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")
    finally:
        listener.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "throw_catch_moe16.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    serve(args.checkpoint, args.port, args.device)


if __name__ == "__main__":
    main()
