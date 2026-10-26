# Compact Latent Visual State for Robot Actuation

This project tests compact visual policies for reactive actuation in a simple simulator task. The initial pipeline used a frozen pretrained visual representation compressed to a small vector; the current experimental controller is a task-trained MoE latent policy. The machine had no Gazebo or ROS 2 installation. Isaac Sim was already installed, so this is an Isaac Sim study, not a Gazebo study.

## Objective and tested systems

The task is to turn toward a red pillar, approach it, and stop near it using LEFT, FORWARD, RIGHT, and STOP commands. The main latent pipeline is:

```text
320 x 240 RGB -> frozen ImageNet ResNet-18 -> 512-D feature -> training-fit PCA-16
              -> distance-weighted k-NN -> smoothed action -> Jetbot wheel command
```

Comparators are bilinear-resized 64 x 64 RGB flattened to 12,288 values and the full 512-D ResNet feature. A separate sweep evaluates 2, 8, 16, 32, 64, and 512 dimensions. The image policy gets RGB only. A geometric supervisor reads target range and bearing to assign labels; robot and target pose are reserved for labels and evaluation. This is not reinforcement learning, a world model, or a newly trained encoder.

## Environment

- Windows 11 Pro, build 26200 (64-bit)
- NVIDIA RTX A2000 12GB, driver 582.16, driver-reported CUDA 13.0
- Isaac Sim 5.1.0-rc.19; bundled Warp reports CUDA Toolkit 12.8
- Project Python 3.11.13; PyTorch 2.13.0+cu130; torchvision 0.28.0+cu130
- Gazebo and ROS 2 not installed/discoverable; standalone `nvcc` unavailable
- Git 2.52.0.windows.1

Captured system output is in [artifacts/system_info.txt](artifacts/system_info.txt); Python package versions are in [environment.txt](environment.txt). PyTorch 2.13 and torchvision 0.28 are official compatible releases. The [PyTorch 2.13 release notes](https://dev-discuss.pytorch.org/t/pytorch-2-13-0-general-availability/3412) and [official CUDA 13.0 Windows wheel notes](https://dev-discuss.pytorch.org/t/pytorch-2-13-final-rc-available/3410) document the installed version route.

## Setup (PowerShell)

The scripts assume Isaac Sim is installed at `C:\isaacsim`. Setup creates a local environment from Isaac Kit Python 3.11, installs the CUDA-enabled PyTorch wheel and pinned requirements, then verifies CUDA. It does not change the system execution policy.

```powershell
Set-Location C:\Users\mirai\latent_robot_demo
.\scripts\setup.ps1
```

Isaac runs through its bundled `C:\isaacsim\python.bat` launcher. PyTorch inference runs in `.venv`, as Kit Python's DLL environment conflicted with the CUDA-enabled wheel. Isaac sends RGB frames to the local inference worker over localhost TCP.

## Camera and actuation check

```powershell
.\scripts\run_isaac.ps1 -EntryPoint 'src\camera_test.py'
```

This saves `artifacts/camera_test.png`, near-stop visibility views, and `results/camera_actuation_check.csv`. The check measured a 320 x 240 RGB camera, about 0.118 m forward translation, about +0.426 rad yaw for LEFT, return yaw for RIGHT, and stable STOP.

## Dataset and training

Collect the seeded dataset:

```powershell
.\scripts\collect.ps1
```

This creates `data/labels.csv`, `data/dataset_summary.json`, and PNG frames in `data/images/`. The current `visible-stop-v2` dataset contains 520 real simulator frames from 10 episodes, 130 per class. Episodes 1–6 train, 7–8 validate, and 9–10 test. Each split is episode-grouped: 312/104/104 frames. All STOP and FORWARD frames show the pillar in the image. Pose is not passed to the visual policy.

Extract frozen features and fit the policies:

```powershell
.\scripts\train.ps1
.\scripts\evaluate.ps1
```

Training selects k from 1, 3, 5, 7, and 11 on validation macro F1, then reports the untouched test episodes. The selected k is 3 for raw RGB, 3 for ResNet-512, and 1 for PCA-16. The PCA-16 transform is fit on training episodes only.

### Experimental learned-latent MoE

Train the compact mixture-of-experts (MoE) policy:

```powershell
.\scripts\train_moe.ps1
```

Unlike PCA-16, this model learns its 16-number bottleneck directly from 64 x 64 RGB images using the action labels. A small depthwise-separable image encoder feeds a 16-D latent code. A learned gate mixes four small experts, each of which predicts LEFT, FORWARD, RIGHT, or STOP. Training uses only train episodes; random seeds and stopping epoch are selected on validation macro F1. The held-out test episodes are reserved for final comparison against PCA-16 + k-NN and the one-expert ablation. Metrics, gate-use diagnostics, validation history, and `models/moe_latent16.pt` are saved separately.

To run the learned policy in Isaac Sim after training:

```powershell
.\scripts\demo.ps1 -Policy moe16
```

Reproduce the one-expert ablation with the same training recipe:

```powershell
.\scripts\train_moe.ps1 -Experts 1
```

This is a preliminary learned baseline, not yet a demonstrated novel method or a validated general improvement. Model-forward timing excludes camera capture, preprocessing, and IPC; closed-loop latency is logged separately.

### Current MoE autoresearch result

The current checkpoint is [models/autoresearch_spatial_moe_v3.pt](models/autoresearch_spatial_moe_v3.pt). It uses domain-randomized expert rollouts, a 2 x 2 spatial feature pool, a learned 16-D latent code, four gated experts, and inverse-square-root class weighting. On the simulator challenge it reached **25/25 on the original cases**, **20/20 across four appended five-case batches**, and **45/45 overall**. It scored 10/10 on cases 1–10; both newest five-case extensions scored 5/5.

The model-forward p95 was 1.48 ms on an A2000 with a preloaded tensor. The full warm request p95 during Isaac rendering was 5.04 ms on the GPU path. A CPU worker on the same workstation measured 1.98 ms p95 and reached 10/10 on cases 1–10 plus 5/5 on cases 36–40; that latency is **CPU inference**, not an A2000 inference measurement. Both request figures exclude simulator rendering, physics, and actuation. A matched one-expert ablation scored 23/25 on the original cases and 5/5 on the newest matched extension; its held-out image score was essentially tied and its A2000 model-forward p95 was 0.72 ms. The current evidence does not show a reliable MoE routing advantage. See [docs/autoresearch_log.md](docs/autoresearch_log.md) for training splits, per-window results, reproduction commands, and limits.

This remains a controlled simulation result. The earlier challenge cases were inspected during model iteration, the scene is simple, and no physical robot has been tested. The result does not establish that MoE routing is better than a matched single-expert model on this new dataset.

### Container transfer and latency check

The trained policies were also run in the CUDA-enabled Docker image against the 104 balanced held-out frames from episodes 9–10. The original four-expert model scored 0.875 accuracy / 0.872 macro F1; its matched one-expert model scored 0.885 / 0.883; the spatial MoE v3 checkpoint scored 0.769 / 0.760 on this separate collection. In-container model-forward p95 was 1.39–1.66 ms, and resize/transfer/inference/output p95 was 1.82–2.18 ms. These are decision-path measurements below 10 ms, excluding simulator rendering, physics, socket IPC, and actuation. They do not establish 100 Hz robot control. Details and exact outputs are in [docs/results.md](docs/results.md) and `results/moe_*_container_crossdataset.json`.

Reproduce the v3 container check from PowerShell:

```powershell
$root = 'C:\Users\mirai\latent_robot_demo'
docker run --rm --gpus all `
  --mount "type=bind,source=$root,target=/workspace/latent_robot_demo" `
  --workdir /workspace/latent_robot_demo `
  iclr-arm-a:cu128 python scripts/evaluate_moe_container.py --iterations 300
```

## Closed-loop demos

The strongest measured controller is the raw-pixel baseline. Run its five seeded approach trials and make the primary video:

```powershell
.\scripts\demo.ps1 -Policy raw
```

To run the compact latent controller with the same start range, smoothing, STOP confirmation, seed, and scoring band:

```powershell
.\scripts\demo.ps1 -Policy pca16
```

`demo.ps1` writes trial and decision logs, screenshots, an H.264 MP4, and current summaries. Run the complete raw-controller pipeline, tests, and optional data recollection with:

```powershell
.\scripts\reproduce_all.ps1
.\scripts\reproduce_all.ps1 -CollectData
```

Visual controllers use a three-decision majority window and require two consecutive raw STOP predictions. Trials start 1.9–2.2 m from the target. A success requires a predicted stop at 1.20–1.65 m, absolute target bearing at most 0.18 rad, and no geometry-proxy collision (robot-target center distance below 0.30 m). The nominal simulated decision interval is 0.15 s (6.67 Hz); synchronous inference and rendering add wall-clock time.

## Initial held-out image-label benchmark (520-frame dataset)

Episodes 9–10 provide 104 held-out images (26 per class). The reported macro F1 is averaged equally over the four labels. Latency is a 500-iteration, batch-one pipeline benchmark after 50 warmups; it covers image preprocessing, device transfers, encoder (where used), PCA, and classifier, but excludes simulator capture/rendering and localhost IPC.

| Method | Dimension | Accuracy | Macro F1 | Median latency | p95 latency | Closed-loop success |
|---|---:|---:|---:|---:|---:|---:|
| Raw RGB + k-NN | 12,288 | 0.9808 | 0.9808 | 250.41 ms | 268.33 ms | 4/5 (80%) |
| ResNet + k-NN | 512 | 0.7596 | 0.7579 | 47.92 ms | 50.71 ms | not run |
| ResNet + PCA-16 + k-NN | 16 | 0.6731 | 0.6658 | 35.38 ms | 38.14 ms | 3/5 (60%) |
| RGB + learned 16-D latent + 4-expert MoE | 16 | 0.8750 | 0.8724 | 0.74 ms* | 1.01 ms* | 4/5 (80%) |
| Same learned encoder + one expert | 16 | 0.8846 | 0.8830 | 0.61 ms* | 0.84 ms* | not run |

*For both learned models, latency is model-forward median/p95 on a preloaded 64 x 64 tensor. It excludes image resizing, transfer, IPC, and simulator work, so it is not directly comparable to the other latency rows. In the live worker log, warm client round-trip p95 was 5.03 ms; the first cold CUDA request took 155.8 ms. That round trip still excludes simulator camera rendering, physics, and actuation.

Raw RGB remains the strongest image-label baseline at 0.981 macro F1. Both learned 16-D models outperform fixed PCA-16 on these held-out frames, but the single-expert model slightly exceeds the four-expert MoE (0.883 vs 0.872 macro F1). So this pilot supports a task-trained compact encoder; it does **not** show that expert routing adds value yet. The four-expert model matches raw RGB at 4/5 closed-loop successes with much lower warm latency. Each controller has only five live trials, so these results are preliminary. PCA-64 scored 0.7692 accuracy / 0.7670 macro F1 on the dimensionality sweep; the other sweep values are in `results/dimensionality_metrics.csv`.

## Figures, video, and result files

- Latest MoE demo: `artifacts/demo.mp4` (validated H.264, 64.25 seconds)
- Raw-RGB baseline demo: `artifacts/demo_raw_final.mp4`
- Matched compact-model demo: `artifacts/demo_pca16_matched.mp4`
- Latent projection: `artifacts/figures/latent_space.png`
- Confusion matrix: `artifacts/figures/confusion_matrix.png`
- Accuracy and classifier latency by dimension: `artifacts/figures/dimension_vs_accuracy.png`
- Pipeline timing: `artifacts/figures/latency_breakdown.png`
- Main metrics and predictions: `results/metrics.csv`, `results/metrics.json`, `results/test_predictions.csv`
- Trial scores and action traces: `results/episode_results.csv`, `results/decision_log.csv`, `results/closed_loop_runs.csv`; matched PCA-16 trial details are in `results/pca16_matched/`
- Offline and closed-loop failure images: `artifacts/failure_cases/`
- Experimental MoE metrics: `results/moe_latent16_metrics.json` (created by `train_moe.ps1`)
- Experimental MoE checkpoint: `models/moe_latent16.pt` (created by `train_moe.ps1`)
- Single-expert ablation metrics: `results/moe_latent16_e1_metrics.json`
- Nearest neighbors: `artifacts/nearest_neighbors/`
- Earlier dataset and failed pilot runs: `archive/v1/`

Live client round-trip latency was 229.42/237.04 ms median/p95 for raw RGB, 18.19/18.89 ms for PCA-16, and 4.71/5.03 ms for MoE16. The first cold request was about 155 ms for both latent policies. These are separate measurements from the model-forward timing table. See [docs/methodology.md](docs/methodology.md), [docs/results.md](docs/results.md), and [docs/limitations.md](docs/limitations.md) for definitions and caveats.

## Supported conclusion

On the initial 520-frame benchmark, the four-expert 16-D model reached 0.872 macro F1 and 4/5 seeded approaches, while a one-expert ablation reached 0.883 macro F1. The later spatial MoE checkpoint reached 45/45 on the current simulator manifest, including two newest five-case checks at 5/5, and 1.48 ms p95 for A2000 model-forward timing. Its full GPU request path measured 5.04 ms p95 while sharing the GPU with Isaac rendering; CPU inference on the same workstation measured 1.98 ms p95. A matched one-expert model reached 23/25 and passed the newest five at 5/5, so the experiments do not establish a MoE-specific benefit. These results show strong feasibility in this controlled scene, not a physical-robot result or general navigation claim.
