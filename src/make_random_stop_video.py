"""Create a time-aligned side-by-side MP4 from recorded paired RGB frames."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_policy(run_tag, policy, episode_id):
    results = ROOT / "results"
    rows = [
        row for row in read_csv(results / f"{run_tag}_{policy}_decisions.csv")
        if row["episode_id"] == episode_id
    ]
    if not rows:
        raise RuntimeError(f"No recorded decisions for {policy}/{episode_id}")
    frames_dir = results / f"{run_tag}_rgb_frames" / f"{policy}_{episode_id}"
    frame_paths = [frames_dir / f"frame_{index:05d}.jpg" for index in range(len(rows))]
    missing = [str(path) for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} captured RGB frames; first is {missing[0]}")
    rows.sort(key=lambda row: float(row["sim_time_s"]))
    times = [float(row["sim_time_s"]) for row in rows]
    # The capture index is the one-based decision number. Sort rows by time, then
    # use that index to recover the corresponding JPEG from disk.
    frames = [
        frames_dir / f"frame_{int(row['decision']) - 1:05d}.jpg"
        for row in rows
    ]
    return {"rows": rows, "times": times, "frames": frames}


def frame_at(recording, sim_time):
    index = bisect.bisect_right(recording["times"], sim_time) - 1
    index = max(0, min(index, len(recording["times"]) - 1))
    frame = cv2.imread(str(recording["frames"][index]), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read RGB frame {recording['frames'][index]}")
    return frame, recording["rows"][index]


def draw_panel(frame, title, sim_time, row, width, image_height, header_height, footer_height):
    frame = cv2.resize(frame, (width, image_height), interpolation=cv2.INTER_AREA)
    panel = np.zeros((header_height + image_height + footer_height, width, 3), dtype=np.uint8)
    panel[:] = (25, 30, 38)
    panel[header_height:header_height + image_height] = frame
    cv2.putText(panel, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (238, 242, 247), 1, cv2.LINE_AA)
    cue = row["observed_event_id"] or "no visible cue"
    action = row["raw_action"]
    footer = f"t={sim_time:5.1f}s   action={action:<7}   {cue}"
    cv2.putText(panel, footer[:74], (8, header_height + image_height + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 226, 234), 1, cv2.LINE_AA)
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--episode", default="ep01")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--playback-speed", type=float, default=4.0,
                        help="Simulated seconds shown per video second")
    args = parser.parse_args()
    if args.fps < 1 or args.playback_speed <= 0:
        raise ValueError("fps and playback-speed must be positive")

    results = ROOT / "results"
    with (results / f"{args.run_tag}_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    with (results / f"{args.run_tag}_moe16_summary.json").open(encoding="utf-8") as stream:
        moe_summary = json.load(stream)
    with (results / f"{args.run_tag}_pca16_summary.json").open(encoding="utf-8") as stream:
        pca_summary = json.load(stream)

    moe = load_policy(args.run_tag, "moe16", args.episode)
    pca = load_policy(args.run_tag, "pca16", args.episode)
    max_sim_time = max(moe["times"][-1], pca["times"][-1])
    frame_count = int(max_sim_time / args.playback_speed * args.fps) + 1
    width, image_height = 320, 240
    header_height, footer_height = 28, 24
    canvas_size = (width * 2, header_height + image_height + footer_height)
    output = results / f"{args.run_tag}_{args.episode}_side_by_side.mp4"
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, canvas_size
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer with the available codec")

    try:
        for index in range(frame_count):
            sim_time = index * args.playback_speed / args.fps
            left_frame, left_row = frame_at(moe, sim_time)
            right_frame, right_row = frame_at(pca, sim_time)
            left = draw_panel(
                left_frame, f"MoE-16  |  p95 {moe_summary['p95_request_latency_ms']:.1f} ms",
                sim_time, left_row, width, image_height, header_height, footer_height,
            )
            right = draw_panel(
                right_frame, f"PCA-16  |  p95 {pca_summary['p95_request_latency_ms']:.1f} ms",
                sim_time, right_row, width, image_height, header_height, footer_height,
            )
            writer.write(np.concatenate([left, right], axis=1))
    finally:
        writer.release()

    print(json.dumps({
        "output": str(output),
        "episode": args.episode,
        "video_fps": args.fps,
        "simulated_seconds_per_video_second": args.playback_speed,
        "video_duration_s": round(frame_count / args.fps, 2),
        "seed": manifest["seed"],
        "frames": frame_count,
    }, indent=2))


if __name__ == "__main__":
    main()
