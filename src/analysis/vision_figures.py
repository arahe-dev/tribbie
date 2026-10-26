"""Figures for the paper: latent map, spatial-softmax keypoints, expert usage, phase confusion."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from lm.policy import LatentMoEPolicy
SKILLS = ["goto", "pick", "drop"]
PHASES = ["move", "arrived", "approach", "descend", "close", "lift", "stow_held", "picked", "extend", "release", "stow_empty", "dropped", "fail"]

FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "figure.dpi": 200, "savefig.bbox": "tight"})
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = LatentMoEPolicy(**ck["config"]).to(DEV).eval()
    m.load_state_dict(ck["state_dict"])
    return m, ck


def episodes(dirs, limit=None):
    out = []
    for d in dirs:
        for f in sorted((ROOT / d).glob("*.npz"))[:limit]:
            with np.load(f) as z:
                out.append({k: z[k] for k in z.files})
    return out


def inputs(ep, idx, ck):
    prev = np.maximum(idx - 1, 0)
    def stack(key):
        x = np.concatenate([ep[key][idx], ep[key][prev]], axis=-1)
        return torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255).to(DEV)
    prop = np.concatenate([ep["q"][idx], ep["width"][idx].reshape(-1, 1), ep["base_vel"][idx]], axis=1).astype(np.float32)
    prop = (prop - np.array(ck["norm"]["proprio_mean"], np.float32)) / np.array(ck["norm"]["proprio_std"], np.float32)
    return stack("head"), stack("wrist"), torch.from_numpy(prop).to(DEV), torch.from_numpy(ep["cmd"][idx].astype(np.float32)).to(DEV)


def run_model(m, ck, eps):
    rows = {"latent": [], "phase": [], "pred": [], "gate": [], "skill": []}
    with torch.no_grad():
        for ep in eps:
            n = len(ep["head"])
            for s in range(0, n, 256):
                idx = np.arange(s, min(n, s + 256))
                out = m(*inputs(ep, idx, ck))
                rows["latent"].append(out["latent"].cpu().numpy())
                rows["pred"].append(out["phase_logit"].argmax(-1).cpu().numpy())
                rows["gate"].append(out["gate"].cpu().numpy())
                rows["phase"].append(ep["phase"][idx])
                rows["skill"].append(ep["cmd"][idx][:, :3].argmax(1))
    return {k: np.concatenate(v) for k, v in rows.items()}


def keypoints(stream_enc, x, top=10):
    f = stream_enc.trunk(x)
    n, c, h, w = f.shape
    p = torch.softmax(f.reshape(n, c, h * w), -1)
    peak = p.max(-1).values
    ys = torch.linspace(-1, 1, h, device=x.device).view(h, 1).expand(h, w).reshape(-1)
    xs = torch.linspace(-1, 1, w, device=x.device).view(1, w).expand(h, w).reshape(-1)
    ex, ey = (p * xs).sum(-1), (p * ys).sum(-1)
    order = peak.argsort(dim=1, descending=True)[:, :top]
    return [(ex[i, order[i]].cpu().numpy(), ey[i, order[i]].cpu().numpy(), peak[i, order[i]].cpu().numpy()) for i in range(n)]


def main():
    m, ck = load(ROOT / "models" / "skills_p3.pt")
    eps = episodes(["data/skills/pick_dagger2", "data/skills/expert"], limit=10)
    r = run_model(m, ck, eps)
    stats = {}

    # ---- latent t-SNE coloured by phase
    sel = np.random.default_rng(0).choice(len(r["latent"]), min(5000, len(r["latent"])), replace=False)
    emb = TSNE(n_components=2, perplexity=40, init="pca", random_state=0).fit_transform(r["latent"][sel])
    present = [p for p in range(len(PHASES)) if (r["phase"][sel] == p).sum() > 20]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for j, p in enumerate(present):
        mk = r["phase"][sel] == p
        ax.scatter(emb[mk, 0], emb[mk, 1], s=2.5, color=cmap(j), label=PHASES[p], alpha=0.8, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=4, fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title("16-D latent (t-SNE), coloured by teacher phase")
    fig.savefig(FIG / "latent_tsne.png"); plt.close(fig)

    # ---- phase confusion (row-normalised)
    present_all = [p for p in range(len(PHASES)) if (r["phase"] == p).sum() > 20]
    cm = np.zeros((len(present_all), len(present_all)))
    for a, p in enumerate(present_all):
        mk = r["phase"] == p
        for b, q in enumerate(present_all):
            cm[a, b] = (r["pred"][mk] == q).mean()
    stats["phase_acc"] = float((r["pred"] == r["phase"]).mean())
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(present_all))); ax.set_xticklabels([PHASES[p] for p in present_all], rotation=60, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(present_all))); ax.set_yticklabels([PHASES[p] for p in present_all], fontsize=6.5)
    for a in range(len(present_all)):
        for b in range(len(present_all)):
            if cm[a, b] > 0.05:
                ax.text(b, a, f"{cm[a, b]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if cm[a, b] > 0.6 else "black")
    ax.set_xlabel("predicted from latent"); ax.set_ylabel("teacher phase")
    ax.set_title(f"Phase decoding (acc. {stats['phase_acc']:.3f})")
    fig.savefig(FIG / "phase_confusion.png"); plt.close(fig)

    # ---- expert gate usage per skill
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    usage = np.array([r["gate"][r["skill"] == s].mean(0) if (r["skill"] == s).any() else np.zeros(4) for s in range(3)])
    bottom = np.zeros(3)
    for e in range(usage.shape[1]):
        ax.bar(SKILLS, usage[:, e], bottom=bottom, color=plt.get_cmap("Set2")(e), label=f"expert {e + 1}", width=0.6)
        bottom += usage[:, e]
    ax.set_ylabel("mean gate weight"); ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.savefig(FIG / "gate_usage.png"); plt.close(fig)
    stats["gate_usage"] = {SKILLS[s]: usage[s].round(3).tolist() for s in range(3)}

    # ---- keypoint overlays on a pick sequence
    ep = eps[0]
    ph = ep["phase"]
    wanted = []
    for name in ("approach", "descend", "close", "lift"):
        clean = ep["head"].reshape(len(ep["head"]), -1).std(1) > 28
        hits = np.nonzero((ph == PHASES.index(name)) & clean)[0]
        if len(hits):
            wanted.append((name, int(hits[len(hits) // 2])))
    idx = np.array([w[1] for w in wanted])
    head, wrist, _, _ = inputs(ep, idx, ck)
    with torch.no_grad():
        kh = keypoints(m.scene_enc, head)
        kw = keypoints(m.hand_enc, wrist)
    fig, axes = plt.subplots(2, len(wanted), figsize=(1.75 * len(wanted), 3.6))
    for j, (name, i) in enumerate(wanted):
        for row, (img, kp, res) in enumerate(((ep["head"][i], kh[j], 96), (ep["wrist"][i], kw[j], 64))):
            a = axes[row, j]
            a.imshow(img)
            xs = (kp[0] + 1) / 2 * (res - 1)
            ys = (kp[1] + 1) / 2 * (res - 1)
            a.scatter(xs, ys, s=18 * kp[2] / kp[2].max() + 4, c="#ff3b30", edgecolors="white", linewidths=0.4)
            a.set_xticks([]); a.set_yticks([])
            if row == 0:
                a.set_title(name)
        axes[0, 0].set_ylabel("head camera")
        axes[1, 0].set_ylabel("wrist camera")
    fig.suptitle("Spatial-softmax keypoints (10 most peaked channels)", y=1.0, fontsize=9)
    fig.savefig(FIG / "keypoints.png"); plt.close(fig)

    (ROOT / "paper" / "vision_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
