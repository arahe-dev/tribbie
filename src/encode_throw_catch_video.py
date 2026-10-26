"""Encode the annotated simulator PNG sequence as an H.264 MP4."""

from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "artifacts" / "throw_catch_6arm_frames"
DESTINATION = ROOT / "artifacts" / "demo_throw_catch_6arm.mp4"
TEMP = ROOT / "artifacts" / "demo_throw_catch_6arm_tmp.mp4"


def encode(codec):
    paths = sorted(FRAMES.glob("frame_*.png"))
    if not paths:
        raise RuntimeError(f"No frames found under {FRAMES}")
    writer = imageio_ffmpeg.write_frames(
        str(TEMP), (720, 720), fps=30, codec=codec, pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", macro_block_size=1, ffmpeg_log_level="warning",
        output_params=["-preset", "fast", "-crf", "20"] if codec == "libx264" else ["-q:v", "5"],
    )
    writer.send(None)
    try:
        for path in paths:
            with Image.open(path) as image:
                frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if frame.shape != (720, 720, 3):
                raise RuntimeError(f"Unexpected frame size for {path.name}: {frame.shape}")
            writer.send(frame.tobytes())
    finally:
        writer.close()
    TEMP.replace(DESTINATION)
    return len(paths)


def main():
    try:
        count = encode("libx264")
        codec = "libx264"
    except Exception as error:
        print(f"H264 encoding failed: {error}; using MPEG-4 Part 2", flush=True)
        TEMP.unlink(missing_ok=True)
        count = encode("mpeg4")
        codec = "mpeg4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(DESTINATION), "-f", "null", "-"], capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError("FFmpeg could not decode the newly written video: " + (probe.stderr[-1200:] or probe.stdout[-1200:]))
    print(f"VIDEO_OK path={DESTINATION} frames={count} fps=30 codec={codec} bytes={DESTINATION.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
