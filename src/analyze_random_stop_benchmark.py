"""Recompute reactive STOP transitions from a completed paired benchmark run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def percentile(values, q):
    if not values:
        return None
    values = sorted(float(value) for value in values)
    index = (len(values) - 1) * q / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(values[lower], 5)
    return round(values[lower] * (upper - index) + values[upper] * (index - lower), 5)


def analyze_policy(run_tag: str, policy: str):
    results = ROOT / "results"
    events = read_csv(results / f"{run_tag}_{policy}_events.csv")
    decisions = read_csv(results / f"{run_tag}_{policy}_decisions.csv")
    with (results / f"{run_tag}_{policy}_summary.json").open(encoding="utf-8") as stream:
        original_summary = json.load(stream)

    by_episode = {}
    for row in decisions:
        by_episode.setdefault(row["episode_id"], []).append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda row: float(row["sim_time_s"]))

    outcomes = {}
    cue_delays = []
    cue_to_stop_delays = []
    onset_to_stop_delays = []
    visible_count = 0
    already_stopped_count = 0
    transition_count = 0
    safe_from_cue_count = 0
    safe_from_onset_count = 0

    for event in events:
        episode_rows = by_episode[event["episode_id"]]
        event_id = event["event_id"]
        visible_indices = [
            index for index, row in enumerate(episode_rows)
            if row["observed_event_id"] == event_id
        ]
        outcome = {
            "cue_visible": bool(visible_indices),
            "stop_transition_after_visible_cue": False,
            "safe_stop_from_visible_cue": False,
            "safe_stop_from_event_onset": False,
            "already_stopped_before_visible_cue": False,
            "cue_to_stop_transition_sim_s": None,
            "onset_to_stop_transition_sim_s": None,
            "collision_proxy_hit": event.get("collision_proxy_hit") == "1",
        }
        outcomes[event_id] = outcome
        if not visible_indices:
            continue

        visible_count += 1
        first_index = visible_indices[0]
        first_visible_time = float(episode_rows[first_index]["sim_time_s"])
        event_onset = float(event["actual_onset_sim_s"])
        event_end = float(event["end_sim_s"])
        previous_command = episode_rows[first_index - 1]["commanded_action"] if first_index else "FORWARD"
        outcome["already_stopped_before_visible_cue"] = previous_command == "STOP"
        already_stopped_count += int(outcome["already_stopped_before_visible_cue"])
        cue_delays.append(first_visible_time - event_onset)

        # Count a command transition only on an observation where the evaluator
        # confirms the cue is visible. A STOP already active before the cue is
        # not credited as a reaction.
        for row in episode_rows[first_index:]:
            if float(row["sim_time_s"]) > event_end or row["active_event_id"] != event_id:
                break
            command = row["commanded_action"]
            action_applied_time = float(row["action_applied_sim_s"])
            if (
                row["observed_event_id"] == event_id
                and command == "STOP"
                and previous_command != "STOP"
                and action_applied_time <= event_end
            ):
                cue_delay = action_applied_time - first_visible_time
                onset_delay = action_applied_time - event_onset
                no_collision = not outcome["collision_proxy_hit"]
                outcome.update({
                    "stop_transition_after_visible_cue": True,
                    "cue_to_stop_transition_sim_s": round(cue_delay, 5),
                    "onset_to_stop_transition_sim_s": round(onset_delay, 5),
                    "safe_stop_from_visible_cue": cue_delay <= 0.25 and no_collision,
                    "safe_stop_from_event_onset": onset_delay <= 0.25 and no_collision,
                })
                transition_count += 1
                cue_to_stop_delays.append(cue_delay)
                onset_to_stop_delays.append(onset_delay)
                safe_from_cue_count += int(outcome["safe_stop_from_visible_cue"])
                safe_from_onset_count += int(outcome["safe_stop_from_event_onset"])
                break
            previous_command = command

    decisions_count = len(decisions)
    stop_outputs = sum(row["raw_action"] == "STOP" for row in decisions)
    event_count = len(events)
    return {
        "policy": policy,
        "scheduled_events": event_count,
        "camera_visible_events": visible_count,
        "cue_visibility_rate": round(visible_count / event_count, 4) if event_count else 0.0,
        "already_stopped_before_first_visible_cue": already_stopped_count,
        "new_stop_transitions_after_visible_cue": transition_count,
        "safe_reactive_stops_within_250ms_from_cue": safe_from_cue_count,
        "safe_reactive_stop_rate_given_visible_cue": round(safe_from_cue_count / visible_count, 4) if visible_count else None,
        "safe_reactive_stops_within_250ms_from_scheduled_event_onset": safe_from_onset_count,
        "safe_reactive_stop_rate_from_scheduled_onset_over_all_events": round(safe_from_onset_count / event_count, 4) if event_count else 0.0,
        "cue_to_stop_transition_median_sim_s": percentile(cue_to_stop_delays, 50),
        "cue_to_stop_transition_p95_sim_s": percentile(cue_to_stop_delays, 95),
        "onset_to_first_visible_observation_median_sim_s": percentile(cue_delays, 50),
        "onset_to_first_visible_observation_p95_sim_s": percentile(cue_delays, 95),
        "false_stop_predictions": original_summary["false_stop_predictions"],
        "false_stop_time_sim_s": original_summary["false_stop_time_s"],
        "raw_stop_output_rate": round(stop_outputs / decisions_count, 4) if decisions_count else 0.0,
        "decision_count": decisions_count,
        "median_inference_round_trip_ms": original_summary["median_request_latency_ms"],
        "p95_inference_round_trip_ms": original_summary["p95_request_latency_ms"],
        "events": outcomes,
    }


def paired_counts(left, right, keys):
    a = left["events"]
    b = right["events"]
    return {
        "paired_event_count": len(keys),
        "moe16_only_successes": sum(a[key]["safe_stop_from_visible_cue"] and not b[key]["safe_stop_from_visible_cue"] for key in keys),
        "pca16_only_successes": sum(b[key]["safe_stop_from_visible_cue"] and not a[key]["safe_stop_from_visible_cue"] for key in keys),
        "both_success": sum(a[key]["safe_stop_from_visible_cue"] and b[key]["safe_stop_from_visible_cue"] for key in keys),
        "both_failure": sum(not a[key]["safe_stop_from_visible_cue"] and not b[key]["safe_stop_from_visible_cue"] for key in keys),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    results = ROOT / "results"
    with (results / f"{args.run_tag}_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    policies = {name: analyze_policy(args.run_tag, name) for name in ("moe16", "pca16")}
    moe_events = policies["moe16"]["events"]
    pca_events = policies["pca16"]["events"]
    keys = sorted(set(moe_events) & set(pca_events))
    common_visible = [key for key in keys if moe_events[key]["cue_visible"] and pca_events[key]["cue_visible"]]
    visible_union = [key for key in keys if moe_events[key]["cue_visible"] or pca_events[key]["cue_visible"]]
    paired = paired_counts(policies["moe16"], policies["pca16"], common_visible)
    report = {
        "benchmark": "randomized reactive emergency stops",
        "run_tag": args.run_tag,
        "seed": manifest["seed"],
        "seed_source": manifest["seed_source"],
        "episodes": len(manifest["episodes"]),
        "events_per_episode": len(manifest["episodes"][0]["events"]) if manifest["episodes"] else 0,
        "scheduled_paired_event_count": len(keys),
        "events_visible_to_both_policies": len(common_visible),
        "events_visible_to_neither_policy": sum(not moe_events[key]["cue_visible"] and not pca_events[key]["cue_visible"] for key in keys),
        "events_visible_to_only_one_policy": sum(moe_events[key]["cue_visible"] != pca_events[key]["cue_visible"] for key in keys),
        "policies": policies,
        "paired_reactive_safe_stops_on_events_visible_to_both": paired,
        "interpretation_limits": [
            "This is a single-seed simulator comparison; the seed is recorded for replay and hidden from both policies.",
            "Scheduled cues were missing from both RGB streams in some events; visibility rates are reported and invisible events receive no reaction credit.",
            "PCA-16 has substantial false-stop behavior, so cue-conditional reaction alone does not imply useful task performance.",
            "The collision metric is a 0.30 m center-distance proxy in simulation, not a physical collision or safety certification.",
            "The rendered red-range cue may appear dark red or near-black in the current RTX material path.",
        ],
    }
    output = results / f"{args.run_tag}_transition_analysis.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps({
        "seed": report["seed"],
        "events_visible_to_both": report["events_visible_to_both_policies"],
        "events_visible_to_neither": report["events_visible_to_neither_policy"],
        "events_visible_to_only_one": report["events_visible_to_only_one_policy"],
        "paired_common_visible_outcomes": paired,
        "policies": {
            name: {key: value for key, value in details.items() if key != "events"}
            for name, details in policies.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
