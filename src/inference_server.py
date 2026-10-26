"""Local CUDA ResNet/PCA/k-NN server for the Isaac Sim controller."""

from pathlib import Path
import argparse
import json
import socket
import struct
import time

import joblib
import numpy as np
import torch
from PIL import Image

from encoder import build_encoder
from moe_latent import MoELatentPolicy


ROOT = Path(__file__).resolve().parents[1]
FRAME_SHAPE = (240, 320, 3)


def recv_exact(connection, count):
    chunks = bytearray()
    while len(chunks) < count:
        block = connection.recv(count - len(chunks))
        if not block:
            return None
        chunks.extend(block)
    return bytes(chunks)


def serve(port, models_dir, policy, moe_checkpoint=None, moe_device="auto"):
    torch.set_num_threads(1)
    moe = None
    if policy == "raw":
        model, transform, device, pca = None, None, torch.device("cpu"), None
        knn = joblib.load(models_dir / "knn_raw.joblib")
        policy_name = "raw"
    elif policy == "moe16":
        if moe_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for MoE inference, but no CUDA device is available")
        device_name = "cuda" if moe_device == "cuda" or (moe_device == "auto" and torch.cuda.is_available()) else "cpu"
        device = torch.device(device_name)
        checkpoint_path = moe_checkpoint or (models_dir / "moe_latent16.pt")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        config = checkpoint["config"]
        if config.get("class_order") != ["LEFT", "FORWARD", "RIGHT", "STOP"]:
            raise RuntimeError("MoE checkpoint action order does not match the controller")
        moe = MoELatentPolicy(
            latent_dim=int(config["latent_dim"]),
            num_experts=int(config["num_experts"]),
            num_classes=int(config["num_classes"]),
            dropout=float(config["dropout"]),
            spatial_pool=int(config.get("spatial_pool", 1)),
        ).to(device)
        moe.load_state_dict(checkpoint["state_dict"])
        moe.eval()
        model, transform, pca, knn = None, None, None, None
        policy_name = "moe16"
    else:
        latent_dim = int(policy.removeprefix("pca"))
        model, transform, device, weights = build_encoder()
        pca = joblib.load(models_dir / f"pca_{latent_dim}.joblib")
        knn = joblib.load(models_dir / f"knn_{latent_dim}.joblib")
        policy_name = f"pca{latent_dim}"
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(1)
    listener.settimeout(60.0)
    ready_info = f"READY {listener.getsockname()[1]} DEVICE={device} POLICY={policy_name}"
    if policy in ("pca16", "pca64"):
        ready_info += f" WEIGHTS={weights.url}"
    print(ready_info, flush=True)
    try:
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(60.0)
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
                total_start = time.perf_counter_ns()
                image_array = np.frombuffer(payload, dtype=np.uint8).reshape(FRAME_SHAPE)
                prep_start = time.perf_counter_ns()
                moe_ms = 0.0
                if policy == "raw":
                    image = Image.fromarray(image_array, mode="RGB").resize((64, 64), Image.Resampling.BILINEAR)
                    classifier_input = np.asarray(image, dtype=np.float32).reshape(-1)[None, :] / 255.0
                    preprocess_ms = (time.perf_counter_ns() - prep_start) / 1e6
                    transfer_ms = encoder_ms = cpu_transfer_ms = pca_ms = 0.0
                elif policy == "moe16":
                    image = Image.fromarray(image_array, mode="RGB").resize((64, 64), Image.Resampling.BILINEAR)
                    pixels = np.asarray(image, dtype=np.float32) / 255.0
                    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).contiguous()
                    preprocess_ms = (time.perf_counter_ns() - prep_start) / 1e6
                    transfer_start = time.perf_counter_ns()
                    tensor = tensor.to(device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    transfer_ms = (time.perf_counter_ns() - transfer_start) / 1e6
                    encoder_ms = cpu_transfer_ms = pca_ms = 0.0
                    moe_start = time.perf_counter_ns()
                    with torch.inference_mode():
                        probabilities = moe(tensor).exp()[0]
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    moe_ms = (time.perf_counter_ns() - moe_start) / 1e6
                    probabilities = probabilities.detach().cpu().numpy()
                    prediction = ["LEFT", "FORWARD", "RIGHT", "STOP"][int(np.argmax(probabilities))]
                    confidence = float(np.max(probabilities))
                else:
                    tensor = transform(Image.fromarray(image_array, mode="RGB")).unsqueeze(0)
                    preprocess_ms = (time.perf_counter_ns() - prep_start) / 1e6

                    transfer_start = time.perf_counter_ns()
                    tensor = tensor.to(device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    transfer_ms = (time.perf_counter_ns() - transfer_start) / 1e6

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    encoder_start = time.perf_counter_ns()
                    with torch.inference_mode():
                        latent = model(tensor)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    encoder_ms = (time.perf_counter_ns() - encoder_start) / 1e6

                    cpu_start = time.perf_counter_ns()
                    z = latent.detach().cpu().numpy().astype(np.float32, copy=False)
                    cpu_transfer_ms = (time.perf_counter_ns() - cpu_start) / 1e6

                    pca_start = time.perf_counter_ns()
                    classifier_input = pca.transform(z)
                    pca_ms = (time.perf_counter_ns() - pca_start) / 1e6
                if policy == "moe16":
                    knn_ms = 0.0
                else:
                    knn_start = time.perf_counter_ns()
                    probabilities = knn.predict_proba(classifier_input)[0]
                    prediction = str(knn.classes_[int(np.argmax(probabilities))])
                    confidence = float(np.max(probabilities))
                    knn_ms = (time.perf_counter_ns() - knn_start) / 1e6
                pipeline_ms = (time.perf_counter_ns() - total_start) / 1e6
                response = {
                    "action": prediction,
                    "confidence": confidence,
                    "latency_ms": {
                        "preprocessing": preprocess_ms,
                        "cpu_to_device": transfer_ms,
                        "encoder": encoder_ms,
                        "device_to_cpu": cpu_transfer_ms,
                        "pca": pca_ms,
                        "knn": knn_ms,
                        "moe": moe_ms,
                        "server_total": pipeline_ms,
                    },
                }
                connection.sendall(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")
    finally:
        listener.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--policy", choices=("raw", "pca16", "pca64", "moe16"), default="pca16")
    parser.add_argument("--moe-checkpoint", type=Path)
    parser.add_argument("--moe-device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    serve(args.port, args.models_dir, args.policy, args.moe_checkpoint, args.moe_device)


if __name__ == "__main__":
    main()
