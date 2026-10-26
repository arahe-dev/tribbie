"""Summarize measured closed-loop episodes and update metrics.csv."""

from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    results_path = ROOT / "results" / "episode_results.csv"
    episodes = pd.read_csv(results_path)
    if episodes.empty:
        raise RuntimeError("No closed-loop episode results to summarize")
    run_summary_path = ROOT / "results" / "closed_loop_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8")) if run_summary_path.exists() else {}
    policy = run_summary.get("policy", "pca16")
    method_by_policy = {"pca16": "ResNet + PCA-16 + kNN", "pca64": "ResNet + PCA-64 + kNN", "raw": "Raw RGB + kNN", "moe16": "Compact image MoE + learned latent-16"}
    summary = {
        "episodes": int(len(episodes)),
        "successes": int(episodes["success"].sum()),
        "failures": int(len(episodes) - episodes["success"].sum()),
        "success_percent": float(episodes["success"].mean() * 100.0),
        "mean_completion_time_sim_s": float(episodes["completion_time_sim_s"].mean()),
        "mean_decisions": float(episodes["number_of_decisions"].mean()),
        "mean_decision_latency_ms": float(episodes["mean_decision_latency_ms"].mean()),
        "median_final_distance_m": float(episodes["final_distance_m"].median()),
        "label_source": "model predictions only for control; simulator pose used for evaluation",
        "policy": policy,
    }
    for field in (
        "dataset_version",
        "seed",
        "smoothing_window",
        "control_hz_nominal",
        "stop_confirmation_count",
        "success_distance_band_m",
        "success_bearing_tolerance_rad",
        "collision_metric",
    ):
        if field in run_summary:
            summary[field] = run_summary[field]
    (ROOT / "results" / "closed_loop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Keep one concise trial summary per tested visual policy.
    runs_path = ROOT / "results" / "closed_loop_runs.csv"
    if runs_path.exists():
        with runs_path.open(newline="", encoding="utf-8") as stream:
            runs = list(csv.DictReader(stream))
    else:
        runs = []
    run_row = {
        "policy": policy,
        "method": method_by_policy.get(policy, policy),
        "episodes": summary["episodes"],
        "successes": summary["successes"],
        "failures": summary["failures"],
        "success_percent": f"{summary['success_percent']:.4f}",
        "mean_decision_latency_ms": f"{summary['mean_decision_latency_ms']:.6f}",
        "mean_final_distance_m": f"{float(episodes['final_distance_m'].mean()):.6f}",
        "seed": run_summary.get("seed", ""),
        "dataset_version": run_summary.get("dataset_version", "unspecified"),
    }
    runs = [row for row in runs if row.get("policy") != policy] + [run_row]
    with runs_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=run_row.keys())
        writer.writeheader()
        writer.writerows(runs)

    metrics_path = ROOT / "results" / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0].keys()) if rows else ["method", "dimension", "accuracy", "macro_f1"]
    if "closed_loop_success_pct" not in fields:
        fields.append("closed_loop_success_pct")
    success_pct = f"{summary['success_percent']:.4f}"
    method = method_by_policy.get(policy)
    if method and not any(row.get("method") == method for row in rows) and policy == "moe16":
        model_metrics_path = ROOT / "results" / "moe_latent16_metrics.json"
        if model_metrics_path.exists():
            model_metrics = json.loads(model_metrics_path.read_text(encoding="utf-8"))
            test_metrics = model_metrics["test"]
            new_row = {field: "" for field in fields}
            new_row.update({
                "method": method,
                "dimension": 16,
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "test_samples": model_metrics["split"]["samples"]["test"],
            })
            rows.append(new_row)
    for row in rows:
        if row["method"] == method:
            row["closed_loop_success_pct"] = success_pct
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("CLOSED_LOOP_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
