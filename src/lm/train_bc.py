"""Train the camera-only latent MoE student on teacher-labelled episodes (BC/DAgger).

Runs in the project .venv (CUDA PyTorch). Episodes are .npz files written by the
Isaac runners; every sample holds the images the student would see, its
proprioception and the teacher's label for that exact state.

  .venv\\Scripts\\python.exe src\\lm\\train_bc.py --task relay --data data/relay/expert data/relay/dagger1 --out models/relay_student.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lm.policy import LatentMoEPolicy, policy_loss  # noqa: E402
from lm.tasks import TASKS  # noqa: E402


def load_episodes(dirs, task):
    spec = TASKS[task]
    episodes = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            with np.load(f) as z:
                data = {k: z[k] for k in z.files}
            if len(data[spec["scene"]]) == 0:
                continue
            episodes.append((f, data))
    if not episodes:
        raise SystemExit(f"No episodes found in {dirs}")
    return episodes


def build_arrays(episodes, task, chunk):
    spec = TASKS[task]
    scene, hand, prop, act, binv, phase, cmd, prev_idx, fut_idx, fut_valid, ep_id = ([] for _ in range(11))
    offset = 0
    for e, (_, d) in enumerate(episodes):
        n = len(d[spec["scene"]])
        agents = d["arm"].astype(np.int64) if "arm" in d else np.zeros(n, dtype=np.int64)
        steps = d["t"].astype(np.int64)
        a_count = int(agents.max()) + 1
        if not np.all(np.arange(n) == steps * a_count + agents):
            raise ValueError("Episode samples must be ordered step-major, agent-minor")
        scene.append(d[spec["scene"]])
        hand.append(d[spec["hand"]])
        prop.append(spec["proprio"](d).astype(np.float32))
        act.append(np.clip(spec["action"](d) / spec["act_scale"], -1.0, 1.0).astype(np.float32))
        binv.append(spec["binary"](d).astype(np.float32).reshape(n, -1))
        phase.append(d["phase"].astype(np.int64))
        cmd.append(spec["command"](d).astype(np.float32) if spec.get("command") else np.zeros((n, 0), np.float32))
        local = np.arange(n)
        prev = np.where(steps > 0, local - a_count, local)
        prev_idx.append(prev + offset)
        f_idx = np.stack([local + a_count * j for j in range(chunk)], axis=1)
        valid = f_idx < n
        f_idx = np.where(valid, f_idx, local[:, None])
        fut_idx.append(f_idx + offset)
        fut_valid.append(valid)
        ep_id.append(np.full(n, e))
        offset += n
    cat = lambda xs: np.concatenate(xs, axis=0)  # noqa: E731
    return dict(scene=cat(scene), hand=cat(hand), prop=cat(prop), act=cat(act), bin=cat(binv), phase=cat(phase),
                cmd=cat(cmd), prev=cat(prev_idx), fut=cat(fut_idx), valid=cat(fut_valid), ep=cat(ep_id))


def augment(img, generator=None):
    """Random +-4 px shift and photometric jitter, on GPU, batch-wise."""
    n, c, h, w = img.shape
    pad = 4
    padded = torch.nn.functional.pad(img, (pad, pad, pad, pad), mode="replicate")
    dx = torch.randint(0, 2 * pad + 1, (n,), device=img.device)
    dy = torch.randint(0, 2 * pad + 1, (n,), device=img.device)
    rows = (dy.view(n, 1) + torch.arange(h, device=img.device).view(1, h))
    cols = (dx.view(n, 1) + torch.arange(w, device=img.device).view(1, w))
    out = padded[torch.arange(n, device=img.device).view(n, 1, 1, 1), torch.arange(c, device=img.device).view(1, c, 1, 1),
                 rows.view(n, 1, h, 1), cols.view(n, 1, 1, w)]
    bright = torch.empty(n, 1, 1, 1, device=img.device).uniform_(0.85, 1.15)
    contrast = torch.empty(n, 1, 1, 1, device=img.device).uniform_(0.85, 1.15)
    mean = out.mean(dim=(1, 2, 3), keepdim=True)
    return ((out - mean) * contrast + mean).mul(bright).clamp(0, 1)


def to_input(arr_scene, idx, prev, device):
    cur = torch.from_numpy(arr_scene[idx]).to(device)
    old = torch.from_numpy(arr_scene[prev[idx]]).to(device)
    x = torch.cat([cur, old], dim=-1).permute(0, 3, 1, 2).float().div_(255.0)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--init", type=str, default="")
    ap.add_argument("--zero-scene", action="store_true", help="ablation: blank the scene/head camera")
    ap.add_argument("--zero-hand", action="store_true", help="ablation: blank the wrist camera")
    ap.add_argument("--metrics-out", type=str, default="")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = TASKS[args.task]

    episodes = load_episodes(args.data, args.task)
    arrays = build_arrays(episodes, args.task, args.chunk)
    n_ep = len(episodes)
    order = np.random.permutation(n_ep)
    n_val = max(1, int(round(n_ep * args.val_frac))) if n_ep > 1 else 0
    val_eps = set(order[:n_val].tolist())
    is_val = np.isin(arrays["ep"], list(val_eps))
    # Isaac occasionally returns an all-black frame right after a reset; never train on those.
    # ...and two concurrent Isaac renderers can corrupt frames into flat sky; drop those too.
    def usable(x, check_flat):
        flat = x.reshape(len(x), -1)
        out = np.empty(len(x), dtype=bool)
        for s in range(0, len(x), 4096):
            blk = flat[s:s + 4096].astype(np.float32)
            out[s:s + 4096] = (blk.mean(1) > 3) & ((blk.std(1) > 10) if check_flat else True)
        return out
    # Only the scene/head camera gets unfinished flat frames; a wrist camera legitimately sees flat surfaces.
    ok = usable(arrays["scene"], True) & usable(arrays["hand"], False)
    print(f"dropping {int((~ok).sum())} samples with black or flat frames", flush=True)
    train_idx = np.nonzero(~is_val & ok)[0]
    val_idx = np.nonzero(is_val & ok)[0]
    prop_mean = arrays["prop"][train_idx].mean(axis=0)
    prop_std = arrays["prop"][train_idx].std(axis=0) + 1e-3
    prop_n = (arrays["prop"] - prop_mean) / prop_std
    print(f"episodes={n_ep} samples train={len(train_idx)} val={len(val_idx)} act_dim={arrays['act'].shape[1]} "
          f"bin_dim={arrays['bin'].shape[1]} cmd_dim={arrays['cmd'].shape[1]}", flush=True)

    model = LatentMoEPolicy(act_dim=arrays["act"].shape[1], bin_dim=arrays["bin"].shape[1], proprio_dim=prop_n.shape[1],
                            command_dim=arrays["cmd"].shape[1], latent_dim=args.latent, num_experts=args.experts,
                            chunk=args.chunk, num_phases=spec["num_phases"]).to(device)
    if args.init:
        state = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        print(f"initialized from {args.init}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(train_idx) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.08)

    def batch_tensors(idx, train):
        scene = to_input(arrays["scene"], idx, arrays["prev"], device)
        hand = to_input(arrays["hand"], idx, arrays["prev"], device)
        if train:
            scene = augment(scene)
            hand = augment(hand)
        if args.zero_scene:
            scene = torch.zeros_like(scene)
        if args.zero_hand:
            hand = torch.zeros_like(hand)
        fut = arrays["fut"][idx]
        act = torch.from_numpy(arrays["act"][fut]).to(device)
        binv = torch.from_numpy(arrays["bin"][fut]).to(device)
        valid = torch.from_numpy(arrays["valid"][idx].astype(np.float32)).to(device)
        prop = torch.from_numpy(prop_n[idx]).to(device)
        cmd = torch.from_numpy(arrays["cmd"][idx]).to(device) if arrays["cmd"].shape[1] else None
        phase = torch.from_numpy(arrays["phase"][idx]).to(device)
        return scene, hand, prop, cmd, act, binv, phase, valid

    def evaluate():
        model.eval()
        tot = {"act_l1_scaled": 0.0, "bin_acc": 0.0, "phase_acc": 0.0, "n": 0}
        with torch.inference_mode():
            for s in range(0, len(val_idx), 512):
                idx = val_idx[s:s + 512]
                scene, hand, prop, cmd, act, binv, phase, valid = batch_tensors(idx, False)
                out = model(scene, hand, prop, cmd)
                tot["act_l1_scaled"] += float((out["act"][:, 0] - act[:, 0]).abs().mean(dim=1).sum())
                tot["bin_acc"] += float(((out["bin_logit"][:, 0] > 0).float() == binv[:, 0]).float().mean(dim=1).sum())
                tot["phase_acc"] += float((out["phase_logit"].argmax(-1) == phase).float().sum())
                tot["n"] += len(idx)
        n = max(tot.pop("n"), 1)
        model.train()
        return {k: v / n for k, v in tot.items()}

    best = None
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(train_idx)
        logs = []
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            scene, hand, prop, cmd, act, binv, phase, valid = batch_tensors(idx, True)
            out = model(scene, hand, prop, cmd)
            loss, parts = policy_loss(out, act, binv, phase, valid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            logs.append(parts)
        val = evaluate() if len(val_idx) else {}
        row = {"epoch": epoch, "train": {k: float(np.mean([l[k] for l in logs])) for k in logs[0]}, "val": val,
               "minutes": round((time.time() - t0) / 60, 2)}
        history.append(row)
        print("EPOCH " + json.dumps(row), flush=True)
        score = val.get("act_l1_scaled", 0.0) - 0.2 * val.get("bin_acc", 0.0) if val else -epoch
        if best is None or score < best[0]:
            best = (score, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "task": args.task,
        "config": model.config,
        "state_dict": best[2],
        "selected_epoch": best[1],
        "norm": {"proprio_mean": prop_mean.tolist(), "proprio_std": prop_std.tolist()},
        "act_scale": np.asarray(spec["act_scale"], dtype=np.float32).tolist(),
        "scene_res": int(arrays["scene"].shape[1]),
        "hand_res": int(arrays["hand"].shape[1]),
        "data": [str(Path(d)) for d in args.data],
        "episodes": n_ep,
        "samples": {"train": int(len(train_idx)), "val": int(len(val_idx))},
        "history": history,
    }
    torch.save(ckpt, out_path)
    print(f"SAVED {out_path} epoch={best[1]}", flush=True)
    if args.metrics_out:
        model.load_state_dict(best[2])
        final = evaluate()
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps({"val": final, "epoch": best[1], "latent": args.latent,
                                                      "zero_scene": args.zero_scene, "zero_hand": args.zero_hand,
                                                      "samples": ckpt["samples"]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
