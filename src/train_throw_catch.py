"""Adapt the pretrained 16-D MoE encoder to the six-arm relay actions."""

from pathlib import Path
import argparse
import csv
import json
import random

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
from torch.nn import functional as F

from moe_latent import MoELatentPolicy, gate_balance_loss


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["WAIT", "CATCH", "HOLD", "THROW"]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASSES)}
IMAGE_SIZE = 224


def load_rows(rows):
    images = []
    labels = []
    for row in rows:
        with Image.open(ROOT / row["image_path"]) as source:
            image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            images.append(np.asarray(image, dtype=np.uint8))
        labels.append(LABEL_TO_ID[row["label"]])
    x = np.stack(images).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
    return x, torch.tensor(labels, dtype=torch.long)


def metrics(model, x, y, device):
    model.eval()
    outputs = []
    gates = []
    with torch.inference_mode():
        for start in range(0, len(y), 64):
            log_probs, gate, _ = model(x[start:start + 64].to(device), return_aux=True)
            outputs.append(log_probs.exp().cpu().numpy())
            gates.append(gate.cpu().numpy())
    probs = np.concatenate(outputs)
    pred = probs.argmax(axis=1)
    truth = y.numpy()
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=range(len(CLASSES)), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, pred, labels=range(len(CLASSES))).tolist(),
        "class_support": {name: int((truth == i).sum()) for i, name in enumerate(CLASSES)},
        "mean_gate_usage": np.concatenate(gates).mean(axis=0).tolist(),
        "predictions": pred,
        "probabilities": probs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=261005)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fine-tune-encoder", action="store_true")
    parser.add_argument("--balance-weight", type=float, default=0.02)
    parser.add_argument("--class-weight-power", type=float, default=0.75)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")

    data_path = ROOT / "data" / "throw_catch" / "labels.csv"
    with data_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    partitions = {name: [row for row in rows if row["split"] == name] for name in ("train", "validation", "test")}
    if any(not partitions[name] for name in partitions):
        raise RuntimeError(f"Each split needs examples; counts: { {k: len(v) for k, v in partitions.items()} }")
    train_x, train_y = load_rows(partitions["train"])
    val_x, val_y = load_rows(partitions["validation"])
    test_x, test_y = load_rows(partitions["test"])

    source_path = ROOT / "models" / "autoresearch_spatial_moe_v3.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    cfg = source["config"]
    model = MoELatentPolicy(
        latent_dim=int(cfg["latent_dim"]), num_experts=int(cfg["num_experts"]),
        num_classes=len(CLASSES), dropout=float(cfg["dropout"]),
        spatial_pool=int(cfg.get("spatial_pool", 1)),
    )
    encoder_weights = {key: value for key, value in source["state_dict"].items() if key.startswith(("image_encoder.", "to_latent."))}
    result = model.load_state_dict(encoder_weights, strict=False)
    unexpected = list(result.unexpected_keys)
    missing_encoder = [key for key in result.missing_keys if key.startswith(("image_encoder.", "to_latent."))]
    if unexpected or missing_encoder:
        raise RuntimeError(f"Pretrained visual encoder mismatch: unexpected={unexpected}, missing={missing_encoder}")
    for name, parameter in model.named_parameters():
        is_visual = name.startswith(("image_encoder.", "to_latent."))
        parameter.requires_grad = args.fine_tune_encoder or not is_visual
    model.to(device)
    train_counts = torch.bincount(train_y, minlength=len(CLASSES)).float().clamp_min(1)
    weights = (train_counts.sum() / (len(CLASSES) * train_counts)).pow(args.class_weight_power).to(device)
    if args.fine_tune_encoder:
        optimizer = torch.optim.AdamW([
            {"params": model.image_encoder.parameters(), "lr": 0.00008},
            {"params": model.to_latent.parameters(), "lr": 0.0002},
            {"params": list(model.gate.parameters()) + list(model.experts.parameters()), "lr": 0.0015},
        ], weight_decay=1e-3)
    else:
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.002, weight_decay=1e-3)

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    n = len(train_y)
    for epoch in range(1, args.epochs + 1):
        model.train()
        if not args.fine_tune_encoder:
            model.image_encoder.eval()
            model.to_latent.eval()
        order = torch.randperm(n)
        loss_sum = 0.0
        for start in range(0, n, args.batch_size):
            idx = order[start:start + args.batch_size]
            batch = train_x[idx].to(device)
            targets = train_y[idx].to(device)
            flip = torch.rand(len(idx), device=device) < 0.5
            if flip.any():
                batch[flip] = batch[flip].flip(-1)
            # Small photometric jitter avoids memorizing one exposure while preserving LED colors.
            brightness = torch.empty((len(idx), 1, 1, 1), device=device).uniform_(0.9, 1.1)
            contrast = torch.empty((len(idx), 1, 1, 1), device=device).uniform_(0.9, 1.1)
            means = batch.mean(dim=(2, 3), keepdim=True)
            batch = ((batch - means) * contrast + means) * brightness
            batch = batch.clamp(0.0, 1.0)
            log_probs, gates, _ = model(batch, return_aux=True)
            loss = F.nll_loss(log_probs, targets, weight=weights) + args.balance_weight * gate_balance_loss(gates)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)

        validation = metrics(model, val_x, val_y, device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n, 1),
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "validation_gate_usage": validation["mean_gate_usage"],
        }
        history.append(row)
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"TRAIN epoch={epoch} val_acc={validation['accuracy']:.3f} val_macro_f1={validation['macro_f1']:.3f}", flush=True)
        if stale >= args.patience:
            break

    model.load_state_dict(best_state)
    validation = metrics(model, val_x, val_y, device)
    test = metrics(model, test_x, test_y, device)
    encoder_exact = all(torch.equal(model.state_dict()[key].cpu(), source["state_dict"][key].cpu()) for key in encoder_weights)
    config = {
        "input": f"RGB {IMAGE_SIZE}x{IMAGE_SIZE} scaled to [0,1]",
        "latent_dim": int(cfg["latent_dim"]),
        "num_experts": int(cfg["num_experts"]),
        "num_classes": len(CLASSES),
        "dropout": float(cfg["dropout"]),
        "spatial_pool": int(cfg.get("spatial_pool", 1)),
        "class_order": CLASSES,
        "pretrained_checkpoint": source_path.relative_to(ROOT).as_posix(),
        "pretrained_encoder_and_latent_weights_preserved_exactly": encoder_exact,
        "pretrained_encoder_used_as_initialization": True,
        "encoder_fine_tuned": args.fine_tune_encoder,
        "selected_epoch": best_epoch,
        "image_size": IMAGE_SIZE,
    }
    checkpoint = {"config": config, "state_dict": best_state, "validation": {k: v for k, v in validation.items() if k not in ("predictions", "probabilities")}, "test": {k: v for k, v in test.items() if k not in ("predictions", "probabilities")}}
    checkpoint_path = ROOT / "models" / "throw_catch_moe16.pt"
    torch.save(checkpoint, checkpoint_path)

    report = {
        "task": "six-arm circular throw/catch relay action recognition",
        "config": config,
        "split_frames": {name: len(partitions[name]) for name in partitions},
        "split_episodes": {name: sorted({row["episode_id"] for row in partitions[name]}) for name in partitions},
        "train_label_counts": {name: int((train_y.numpy() == i).sum()) for i, name in enumerate(CLASSES)},
        "gate_balance_weight": args.balance_weight,
        "class_weight_power": args.class_weight_power,
        "best_validation_epoch": best_epoch,
        "validation": checkpoint["validation"],
        "test": checkpoint["test"],
        "history": history,
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
    }
    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    report_path = result_dir / "throw_catch_training.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("config", "split_frames", "split_episodes", "train_label_counts", "validation", "test", "checkpoint")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
