"""Encode the physics-driven six-arm relay's annotated frames as H.264."""

from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "artifacts" / "throw_catch_physics_6arm_frames"
DESTINATION = ROOT / "artifacts" / "demo_throw_catch_6arm.mp4"
TEMP = ROOT / "artifacts" / "demo_throw_catch_6arm_physics_tmp.mp4"
WIDTH, HEIGHT, FPS = 1280, 720, 30


def main():
    paths = sorted(FRAMES.glob("frame_*.png"))
    if not paths:
        raise RuntimeError(f"No rendered frames found under {FRAMES}")

    writer = imageio_ffmpeg.write_frames(
        str(TEMP),
        (WIDTH, HEIGHT),
        fps=FPS,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="warning",
        output_params=["-preset", "medium", "-crf", "18", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for path in paths:
            with Image.open(path) as image:
                frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if frame.shape != (HEIGHT, WIDTH, 3):
                raise RuntimeError(f"Unexpected frame size in {path.name}: {frame.shape}")
            writer.send(np.ascontiguousarray(frame).tobytes())
    finally:
        writer.close()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(TEMP), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        TEMP.unlink(missing_ok=True)
        details = probe.stderr[-1500:] or probe.stdout[-1500:]
        raise RuntimeError(f"FFmpeg could not decode the encoded relay video: {details}")

    TEMP.replace(DESTINATION)
    print(
        f"VIDEO_OK path={DESTINATION} frames={len(paths)} fps={FPS} codec=libx264 "
        f"duration_s={len(paths) / FPS:.2f} bytes={DESTINATION.stat().st_size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
