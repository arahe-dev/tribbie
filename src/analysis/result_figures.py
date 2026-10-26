"""Charts from logged results: DAgger curves, frame integrity, latency, scene montage."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
BLUE, ORANGE, GREY, GREEN = "#2f6db5", "#e07b28", "#8a8f98", "#3a9a5b"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


# DAgger / closed-loop success vs share of steps executed by the student
fig, ax = plt.subplots(figsize=(3.4, 2.3))
series = {
    "pick skill (specialist)": ([0.75, 0.90, 1.00], [(57, 80), (60, 80), (14, 20)], BLUE),
    "monolithic delivery policy": ([0.70, 0.88, 1.00], [(13, 20), (7, 24), (0, 8)], ORANGE),
    "goto skill": ([0.80, 1.00], [(34, 36), (1, 10)], GREEN),
}
for name, (xs, kn, col) in series.items():
    p = [wilson(k, n) for k, n in kn]
    ax.errorbar(xs, [a[0] for a in p], yerr=[[a[0] - a[1] for a in p], [a[2] - a[0] for a in p]], marker="o", ms=3.5,
                capsize=2, lw=1.4, color=col, label=name)
ax.set_xlabel("share of control steps executed by the student")
ax.set_ylabel("success rate (95% Wilson CI)")
ax.set_ylim(-0.03, 1.05); ax.set_xlim(0.66, 1.03)
ax.legend(frameon=False, fontsize=6.8, loc="lower left")
fig.savefig(FIG / "dagger_curves.png"); plt.close(fig)

# Frame integrity: share of unusable head-camera frames
fig, ax = plt.subplots(figsize=(3.4, 2.0))
labels = ["2 jobs\n(A)", "2 jobs\n(B)", "DAgger,\nno fix", "teacher,\nsequential", "re-render\nfix"]
vals = [1.00, 0.54, 0.47, 0.00, 0.00]
cols = [ORANGE, ORANGE, ORANGE, GREY, BLUE]
ax.bar(range(len(vals)), vals, color=cols, width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=7)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=6.8)
ax.set_ylabel("flat / unfinished frames"); ax.set_ylim(0, 1.12)
fig.savefig(FIG / "frame_integrity.png"); plt.close(fig)

# Latency
fig, ax = plt.subplots(figsize=(3.4, 1.9))
names = ["forward pass,\nCUDA graph", "full decision,\n6 arms, idle GPU", "same, CPU", "inside simulator\n(GPU rendering)", "VLM programmer\ncall (Opus)"]
vals = [0.6, 1.5, 7.5, 13.0, 10000.0]
ax.barh(range(len(vals)), vals, color=[BLUE, BLUE, GREY, ORANGE, "#7a4fb5"], height=0.55)
ax.set_xscale("log")
for i, v in enumerate(vals):
    ax.text(v * 1.15, i, f"{v:g} ms" if v < 1000 else "~10 s", va="center", fontsize=7)
ax.set_yticks(range(len(vals))); ax.set_yticklabels(names, fontsize=6.5); ax.invert_yaxis()
ax.set_xlabel("latency (ms, log scale)"); ax.set_xlim(0.3, 1e5)
fig.savefig(FIG / "latency.png"); plt.close(fig)

# Scene montage from a rendered evaluation state
st = ROOT / "results" / "vlm_states"
ov = Image.open(st / "state03_overview.png").convert("RGB")
tb = Image.open(st / "state03_table.png").convert("RGB")
bn = Image.open(st / "state03_bins.png").convert("RGB")
fig = plt.figure(figsize=(7.0, 2.35))
gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1], hspace=0.08, wspace=0.04)
ax = fig.add_subplot(gs[:, 0]); ax.imshow(ov); ax.set_title("overview camera", fontsize=8); ax.axis("off")
ax = fig.add_subplot(gs[0, 1]); ax.imshow(tb); ax.set_title("table camera (VLM)", fontsize=8); ax.axis("off")
ax = fig.add_subplot(gs[0, 2]); ax.imshow(bn); ax.set_title("bins camera (VLM)", fontsize=8); ax.axis("off")
data = np.load(sorted((ROOT / "data" / "skills" / "pick_dagger2").glob("*.npz"))[2])
clean = np.nonzero((data["head"].reshape(len(data["head"]), -1).std(1) > 25) & (data["wrist"].reshape(len(data["wrist"]), -1).mean(1) > 20) & (data["phase"] == 3))[0]
i = int(clean[len(clean) // 2])
ax = fig.add_subplot(gs[1, 1]); ax.imshow(data["head"][i]); ax.set_title("policy head 96x96", fontsize=8, y=-0.28); ax.axis("off")
ax = fig.add_subplot(gs[1, 2]); ax.imshow(data["wrist"][i]); ax.set_title("policy wrist 64x64", fontsize=8, y=-0.28); ax.axis("off")
fig.savefig(FIG / "scene.png"); plt.close(fig)

# Corrupted vs clean frame example
d2 = np.load(sorted((ROOT / "data" / "skills" / "pick_dagger2").glob("*.npz"))[0])
sd = d2["head"].reshape(len(d2["head"]), -1).std(1)
bad, good = int(np.argmin(sd)), int(np.argmax(sd > 30))
fig, axes = plt.subplots(1, 2, figsize=(2.6, 1.4))
for a, j, t in ((axes[0], good, "rendered"), (axes[1], bad, "unfinished (flat)")):
    a.imshow(d2["head"][j]); a.set_title(t, fontsize=7); a.axis("off")
fig.savefig(FIG / "flat_example.png"); plt.close(fig)
print("ok")
