from pathlib import Path
import csv
import json

import numpy as np
from PIL import Image

from src.encoder import build_encoder

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"LEFT", "FORWARD", "RIGHT", "STOP"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def test_dataset_is_complete_balanced_and_episode_grouped():
    rows = read_csv(ROOT / "data" / "labels.csv")
    assert len(rows) == 520
    assert {row["label"] for row in rows} == CLASSES
    assert all(row["label_source"] == "simulator_state_supervisor" for row in rows)
    assert all((ROOT / row["image_path"]).is_file() for row in rows)
    counts = {label: sum(row["label"] == label for row in rows) for label in CLASSES}
    assert set(counts.values()) == {130}
    episode_splits = {}
    for row in rows:
        episode_splits.setdefault(row["episode_id"], set()).add(row["split"])
    assert len(episode_splits) == 10
    assert all(len(splits) == 1 for splits in episode_splits.values())
    assert {episode for episode, splits in episode_splits.items() if "test" in splits} == {"ep09", "ep10"}


def test_camera_and_actuation_smoke_artifacts_exist():
    image = Image.open(ROOT / "artifacts" / "camera_test.png")
    assert image.size == (320, 240)
    rows = read_csv(ROOT / "results" / "camera_actuation_check.csv")
    assert {row["action"] for row in rows} == CLASSES
    forward = next(row for row in rows if row["action"] == "FORWARD")
    assert float(forward["x_after"]) > float(forward["x_before"]) + 0.05


def test_resnet_embedding_shape():
    model, transform, device, _ = build_encoder(device="cpu")
    import torch

    image = Image.open(ROOT / "artifacts" / "camera_test.png").convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.inference_mode():
        feature = model(tensor)
    assert tuple(feature.shape) == (1, 512)
    assert model.training is False


def test_saved_pca16_policy_accepts_512d_test_embeddings():
    import joblib

    pca = joblib.load(ROOT / "models" / "pca_16.joblib")
    knn = joblib.load(ROOT / "models" / "knn_16.joblib")
    embeddings = np.load(ROOT / "features" / "test_embeddings.npy")
    compact = pca.transform(embeddings[:4])
    predictions = knn.predict(compact)
    assert compact.shape == (4, 16)
    assert len(predictions) == 4
    assert set(predictions).issubset(CLASSES)


def test_metrics_use_disjoint_train_and_test_episodes():
    metrics = read_csv(ROOT / "results" / "metrics.csv")
    assert {row["method"] for row in metrics} == {
        "Raw RGB + kNN",
        "ResNet + kNN",
        "ResNet + PCA-16 + kNN",
        "Compact image MoE + learned latent-16",
    }
    assert all(int(row["test_samples"]) == 104 for row in metrics)
    metadata = read_csv(ROOT / "features" / "metadata.csv")
    train = {row["episode_id"] for row in metadata if row["split"] == "train"}
    test = {row["episode_id"] for row in metadata if row["split"] == "test"}
    assert train.isdisjoint(test)
    assert train == {"ep01", "ep02", "ep03", "ep04", "ep05", "ep06"}


def test_closed_loop_trials_and_h264_videos_are_recorded():
    runs = {row["policy"]: row for row in read_csv(ROOT / "results" / "closed_loop_runs.csv")}
    assert {"raw", "pca16"}.issubset(runs)
    for row in runs.values():
        assert int(row["episodes"]) == 5
        assert int(row["successes"]) + int(row["failures"]) == 5
        assert row["dataset_version"] == "visible-stop-v2"
    info = json.loads((ROOT / "results" / "video_info.json").read_text(encoding="utf-8"))
    assert info["h264_validation"] is True
    assert info["codec"] == "h264"
    assert (ROOT / info["path"]).is_file()
    assert (ROOT / "artifacts" / "demo_pca16_matched.mp4").is_file()
