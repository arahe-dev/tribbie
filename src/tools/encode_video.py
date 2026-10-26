"""Encode overlay frames (f_*.jpg) into an H.264 MP4. One frame per 15 Hz control step,
so playback at 15 fps runs in simulated real time.

  .venv\\Scripts\\python.exe src\\tools\\encode_video.py artifacts\\demo1_frames artifacts\\demo1_relay.mp4
"""

import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image


def main(frames_dir, out_path, fps=15, title=None):
    frames = sorted(Path(frames_dir).glob("f_*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {frames_dir}")
    w, h = Image.open(frames[0]).size
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(out_path), (w, h), fps=fps, codec="libx264", quality=8,
                                         pix_fmt_out="yuv420p", macro_block_size=8, output_params=["-movflags", "+faststart"])
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(np.asarray(Image.open(f).convert("RGB"))))
    writer.close()
    print(f"wrote {out_path} ({len(frames)} frames, {len(frames) / fps:.1f} s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 15)
