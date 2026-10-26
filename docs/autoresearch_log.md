# MoE latent-space autoresearch log

Date: 2026-09-26. Hardware: Windows workstation with an NVIDIA RTX A2000 12 GB; Isaac Sim 5.1.0; CUDA-enabled PyTorch worker. This is a simulator result, not a physical Duckiebot result.

## Best checkpoint

`models/autoresearch_spatial_moe_v3.pt` uses a depthwise image encoder, a 2 x 2 spatial feature pool, a learned 16-D latent code, a four-expert action mixture, and class-weighted training. The 2 x 2 pool retains coarse image position for the latent encoder. The policy receives RGB only at run time; simulator pose is used to label teacher actions and score rollouts.

Training data: 4,788 frames from 90 teacher-guided rollouts across 30 appearance groups. Dome intensity and red target shade vary by group. The split is grouped by appearance: 18 train, 6 validation, 6 test groups. The natural training counts are LEFT 110, FORWARD 2,642, RIGHT 101, STOP 54. Training uses inverse-square-root class weighting (power 0.5), five seeds, and selects by validation macro F1. The selected run is seed 101, epoch 26. On the held-out rollout groups it scored 98.8% accuracy and 0.943 macro F1. This is image-label accuracy, not navigation success.

## Locked closed-loop challenge

Success requires a predicted stop at 1.20–1.65 m, absolute target bearing no more than 0.18 rad, and no geometry-proxy collision (target distance below 0.30 m). Actions are smoothed over three decisions; STOP needs two consecutive raw predictions. The manifest started with 25 cases and received five fresh cases each time a variation first crossed the 8/10 gate. The current manifest has 45 cases. None of these cases were added to the training set.

| Checkpoint / case window | Result | Notes |
|---|---:|---|
| Earlier balanced independent-frame MoE, cases 1–25 | 4/25 | 93.0% held-out frame accuracy did not transfer to navigation |
| Earlier closed-loop-trained MoE, cases 1–25 | 15/25 | Reached 8/10 on cases 21–30; 2/5 on its fresh cases 31–35 |
| Spatial MoE v3, original cases 1–25 | **25/25** | 10/10 on each window 1–10 and 11–20; 5/5 on 21–25 |
| Spatial MoE v3, additions 26–40 | **15/15** | 5/5 on each of the three five-case blocks |
| Spatial MoE v3, fresh addition 36–40 | **5/5** | Added immediately after the v3 10/10 result on cases 1–10 |
| Spatial MoE v3, matched fresh addition 41–45 | **5/5** | Added after the one-expert control reached 9/10; both models passed |
| Spatial MoE v3, all 45 cases | **45/45** | No geometry-proxy collisions |
| Matched one-expert control, original cases 1–25 | 23/25 | 9/10, 9/10, and 5/5 |
| Matched one-expert control, fresh cases 41–45 | **5/5** | Same new cases as the MoE |

The first spatial-model trigger result was 10/10, followed by 5/5 on five cases generated afterward. The model also scored 10/10 on cases 11–20 and 21–30. Cases 26–35 originated as extensions for earlier variations; v3 passed them too. Cases 41–45 were generated for the matched one-expert control and were also tested on v3. Case-wise episode CSVs and summaries are in `results/autoresearch_spatial_moe_v3_*` and `results/autoresearch_spatial_single_v3_*`.

## Matched one-expert ablation

The one-expert model uses the same rollout data, spatial pool, class weights, train/validation/test groups, and five-seed validation selection. Its held-out accuracy/macro F1 were 98.8% / 0.942, versus 98.8% / 0.943 for the MoE. It reached 23/25 on the original cases and 5/5 on fresh cases 41–45, versus 25/25 and 5/5 for the MoE. The two-case gap is not enough to establish a reliable MoE advantage; both passed the newest extension. One-expert A2000 forward p95 was 0.72 ms, about half the MoE's 1.48 ms.

## Latency

All figures below are warm-request measurements from the A2000 workstation. They do not include simulator image capture/rendering, physics, or wheel actuation.

| Measurement scope | Device running MoE | Median | p95 |
|---|---|---:|---:|
| Model forward on a preloaded 64 x 64 tensor | RTX A2000, four experts | 0.75 ms | 1.48 ms |
| Local client round trip with resize, transfer, model, and localhost IPC during Isaac Sim rendering | RTX A2000, four experts | 4.75 ms | 5.04 ms |
| Same GPU request path | RTX A2000, one expert | 4.08 ms | 4.33 ms |
| Local client round trip | CPU, on the A2000 workstation, four experts | 1.82 ms | 1.98 ms |

The A2000 model-forward result is below 3 ms; the full GPU request path is not. The CPU worker option puts the measured request path below 2 ms, but that figure must not be described as A2000 inference. CPU mode scored 10/10 on cases 1–10 and 5/5 on cases 36–40. A cold GPU request reached about 167 ms. Request p50/p95 values exclude the first request of each worker; process startup is not included.

## Reproduction

```powershell
.\scripts\collect_moe_rollout.ps1 -Episodes 30 -RolloutsPerEpisode 3 -Seed 9262026
.\scripts\train_moe.ps1 -MetadataPath data/moe_rollout_v2/metadata.csv -ModelName autoresearch_spatial_moe_v3 -ClassWeightPower 0.5 -SpatialPool 2
.\scripts\run_robustness.ps1 -Policy moe16 -StartIndex 0 -Count 10 -RunTag autoresearch_spatial_moe_v3_cases01_10 -ModelCheckpoint models/autoresearch_spatial_moe_v3.pt
```

The remaining manifest windows use the same command with their start index, count, and a unique run tag. Use `-MoeDevice cpu` to run the policy worker on CPU for the local low-latency path.

## Research limits

This demonstrates a promising result in one simple rendered scene. Cases 1–40 were inspected during earlier variations, so 45/45 is not a pristine prospective benchmark; cases 41–45 are the newest extension and were not used for training. The scene has one red pole, fixed camera geometry, simulator labels, and four coarse actions; there is no physical-robot evaluation. The matched ablation shows that one expert nearly matches the MoE and runs faster, so the current evidence does not establish a routing advantage. A publication claim also needs stronger baselines (including color segmentation and non-MoE learned controllers), multiple seeds reported without best-seed selection, broader scenes, and a larger prospectively locked test suite.
