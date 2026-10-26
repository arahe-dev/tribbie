"""Create a fixed challenge suite of randomized but reproducible simulator starts."""

from pathlib import Path
import argparse
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "robustness_challenge_v1.json"
RED_SHADES = [
    [1.00, 0.08, 0.04],
    [1.00, 0.17, 0.08],
    [0.88, 0.035, 0.045],
    [0.98, 0.10, 0.015],
    [0.92, 0.06, 0.12],
]


def latin_hypercube(rng, count, low, high):
    bins = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    rng.shuffle(bins)
    return low + (high - low) * bins


def make_cases(start, count, seed):
    rng = np.random.default_rng(seed)
    distances = latin_hypercube(rng, count, 1.72, 2.52)
    bearing_magnitudes = latin_hypercube(rng, count, 0.18, 0.48)
    bearings = bearing_magnitudes * np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
    yaw = latin_hypercube(rng, count, -0.28, 0.28)
    light = latin_hypercube(rng, count, 650.0, 1900.0)
    shades = rng.permutation(np.resize(np.arange(len(RED_SHADES)), count))
    cases = []
    for local_index in range(count):
        index = start + local_index
        cases.append({
            "case_id": f"challenge_v1_{index + 1:03d}",
            "case_family": "pose_lighting_and_red_shade",
            "yaw_rad": round(float(yaw[local_index]), 5),
            "bearing_rad": round(float(bearings[local_index]), 5),
            "distance_m": round(float(distances[local_index]), 4),
            "dome_intensity": round(float(light[local_index]), 1),
            "target_scale_factor": 1.0,
            "target_color_rgb": RED_SHADES[int(shades[local_index])],
        })
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=90826)
    parser.add_argument("--append-five-after-summary", type=Path)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.append_five_after_summary:
        if not output.exists():
            raise FileNotFoundError(f"Cannot extend missing challenge manifest: {output}")
        summary_path = args.append_five_after_summary
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("episodes", 0)) != 10 or int(summary.get("successes", 0)) < 8:
            raise ValueError("The five-case extension gate requires a 10-case run with at least 8 successes")
        manifest = json.loads(output.read_text(encoding="utf-8"))
        if summary_path.name in manifest.get("extension_triggers", []):
            raise ValueError("This qualifying run already triggered a five-case extension")
        existing = manifest["cases"]
        prior_count = len(existing)
        seed = int(manifest["seed"]) + 7919 * (prior_count + 1)
        added = make_cases(prior_count, 5, seed)
        manifest["cases"].extend(added)
        manifest["extension_triggers"] = manifest.get("extension_triggers", []) + [summary_path.name]
        manifest["extensions"] = manifest.get("extensions", []) + [{
            "trigger_summary": summary_path.name,
            "prior_case_count": prior_count,
            "added_case_ids": [case["case_id"] for case in added],
            "added_seed": seed,
            "gate": "10 cases and at least 8 successes",
        }]
        output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"manifest": output.as_posix(), "added": added, "total_cases": len(manifest["cases"])}, indent=2))
        return

    if args.count < 25:
        raise ValueError("The initial challenge suite must include at least 25 cases")
    cases = make_cases(0, args.count, args.seed)
    manifest = {
        "suite": "latent-moE robustness challenge",
        "version": 1,
        "purpose": "locked simulator challenge cases; never train on these cases",
        "seed": args.seed,
        "initial_case_count": args.count,
        "case_selection_rule": "Use rotating 10-case windows; when a candidate reaches at least 8/10, append five fresh cases before judging whether it holds the threshold.",
        "cases": cases,
        "extensions": [],
        "extension_triggers": [],
    }
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": output.as_posix(), "case_count": len(cases), "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
