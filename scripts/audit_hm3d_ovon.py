#!/usr/bin/env python3
"""Audit the public HM3D-OVON episode metadata without loading scene meshes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

SPLITS = {
    "train": "train",
    "val_seen": "val_seen",
    "val_seen_synonyms": "val_seen_synonyms",
    "val_unseen": "val_unseen",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_split(root: Path, name: str, directory: str) -> dict:
    content_dir = root / directory / "content"
    shards = sorted(content_dir.glob("*.json.gz"))
    if not shards:
        raise FileNotFoundError(f"No per-scene episode shards under {content_dir}")

    category_names: set[str] = set()
    episode_count = 0
    scene_ids: set[str] = set()
    for shard in shards:
        scene_ids.add(shard.name.removesuffix(".json.gz"))
        with gzip.open(shard, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        for episode in payload.get("episodes", []):
            episode_count += 1
            category = episode.get("object_category")
            if category:
                category_names.add(str(category).strip().casefold())

    return {
        "split": name,
        "scene_count": len(scene_ids),
        "episode_shard_count": len(shards),
        "episode_count": episode_count,
        "goal_category_count": len(category_names),
        "goal_categories": sorted(category_names),
        "scene_ids": sorted(scene_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/datasets/ovon/hm3d"))
    parser.add_argument("--archive", type=Path, default=Path("data/hm3d_ovon/hm3d.tar.gz"))
    parser.add_argument("--output", type=Path, default=Path("results/hm3d_ovon_inventory.json"))
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    reports = [audit_split(root, name, directory) for name, directory in SPLITS.items()]
    all_categories = sorted({category for report in reports for category in report["goal_categories"]})
    all_scenes = sorted({scene for report in reports for scene in report["scene_ids"]})
    archive_path = args.archive.resolve()
    result = {
        "benchmark": "HM3D-OVON",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "episode_archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256(archive_path),
        },
        "all_splits_scene_count": len(all_scenes),
        "all_splits_goal_category_count": len(all_categories),
        "splits": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        "benchmark": result["benchmark"],
        "all_splits_scene_count": result["all_splits_scene_count"],
        "all_splits_goal_category_count": result["all_splits_goal_category_count"],
        "splits": [
            {key: report[key] for key in ("split", "scene_count", "episode_count", "goal_category_count")}
            for report in reports
        ],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
