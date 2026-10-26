# Results

## Current MoE autoresearch follow-up

The latest checkpoint, `models/autoresearch_spatial_moe_v3.pt`, was trained on 4,788 frames from 90 domain-randomized teacher rollouts, with episode-grouped train/validation/test splits. Its held-out rollout-frame score was 98.8% accuracy / 0.943 macro F1. In closed loop it reached 25/25 on the initial challenge suite and 20/20 on appended cases, for 45/45 across the current manifest. It scored 5/5 on the two newest extensions. A matched one-expert ablation scored 23/25 on the original cases and 5/5 on the newest matched cases. Detailed per-window results and latency scopes are in [autoresearch_log.md](autoresearch_log.md).

The 1.48 ms p95 figure is A2000 model-forward time on a preloaded tensor. The full GPU request path measured 5.04 ms p95 while Isaac Sim shared the GPU. CPU inference on that workstation measured 1.98 ms p95 for the same request path; it is not GPU inference. All figures exclude simulator rendering, physics, and actuation. The original benchmark below is retained for historical comparison.

### Container-side collection-transfer check

On 2026-09-27, the saved four-expert, single-expert, and spatial-MoE checkpoints were run in `iclr-arm-a:cu128` with CUDA exposed to the RTX A2000. They were scored on the same 104 balanced images from episodes 9–10 of `data/labels.csv`; these images were not in the v3 checkpoint's 4,788-frame rollout collection. This is a useful collection-transfer check in the same simple Isaac scene, not an HM3D-OVON result.

| Checkpoint | Experts | Accuracy | Macro F1 | Model p95 | Resize + transfer + inference + output p95 |
|---|---:|---:|---:|---:|---:|
| `moe_latent16.pt` | 4 | 0.8750 | 0.8724 | 1.64 ms | 2.10 ms |
| `moe_latent16_e1.pt` | 1 | 0.8846 | 0.8830 | 1.39 ms | 1.82 ms |
| `autoresearch_spatial_moe_v3.pt` | 4 | 0.7692 | 0.7599 | 1.66 ms | 2.18 ms |

Each timing used 300 batch-one iterations after warmup. The request path starts with an in-memory 320 x 240 RGB frame and excludes disk I/O, socket IPC, Isaac rendering/physics, and actuation. The classifier and model p95s are below 10 ms, but they do not mean the full robot loop runs at 100 Hz. The v3 model's lower score on this separately collected balanced split shows that its 98.8% score on rollout test frames does not transfer unchanged. The matched one-expert policy is still slightly more accurate and faster here, so this comparison provides no evidence that four-expert routing helps. Full outputs are in `results/moe_initial_container_crossdataset.json`, `results/moe_initial_e1_container_crossdataset.json`, and `results/moe_v3_container_crossdataset.json`; the reproducible container evaluator is `scripts/evaluate_moe_container.py`.

## Initial dataset v2 and split

The first dataset exposed a camera-range failure: the pillar was not visible in 87% of STOP examples. That dataset and its results remain in `archive/v1/`. The camera visibility check showed the pillar clearly at 1.35 m and clipped it at shorter ranges. The revised `visible-stop-v2` supervisor labels STOP inside 1.50 m, and STOP samples are centered at 1.35–1.48 m. This is a transparent task setup change, not a post-test split change.

Current dataset counts:

| Split | Episodes | Samples | Per class |
|---|---|---:|---:|
| Train | ep01–ep06 | 312 | 78 |
| Validation | ep07–ep08 | 104 | 26 |
| Test | ep09–ep10 | 104 | 26 |

There are 520 images total: 130 LEFT, 130 FORWARD, 130 RIGHT, and 130 STOP. Per-image supervisor geometry and the episode split are saved in `data/labels.csv`.

## Offline held-out comparison

All primary scores are on episodes 9–10 only. The classifiers and PCA k are selected on validation episodes. The complete held-out confusion matrix and per-class metrics are in `results/metrics.json` and `results/per_class_metrics.csv`.

| Method | Dim | Accuracy | Macro F1 | Median latency | p95 latency |
|---|---:|---:|---:|---:|---:|
| Raw 64 x 64 RGB + k-NN (k=3) | 12,288 | 0.9808 | 0.9808 | 250.41 ms | 268.33 ms |
| ResNet-18 + k-NN (k=3) | 512 | 0.7596 | 0.7579 | 47.92 ms | 50.71 ms |
| ResNet-18 + PCA-16 + k-NN (k=1) | 16 | 0.6731 | 0.6658 | 35.38 ms | 38.14 ms |
| RGB + learned 16-D latent + 4-expert MoE | 16 | 0.8750 | 0.8724 | 0.74 ms* | 1.01 ms* |
| Same learned encoder + one expert | 16 | 0.8846 | 0.8830 | 0.61 ms* | 0.84 ms* |

*Learned-model timing is model-forward median/p95 on a preloaded 64 x 64 tensor and excludes resize, transfer, IPC, and simulation. It is not directly comparable to the other rows. The MoE has 8,620 parameters; the one-expert model has 7,045. The raw baseline remains strongest offline. The one-expert model slightly exceeds the four-expert MoE (0.883 vs. 0.872 macro F1), so this ablation does not show an advantage from expert routing. Both learned encoders outperform fixed PCA-16 on this test, but only across two held-out episodes. PCA-64 reaches 0.7692 accuracy / 0.7670 macro F1; PCA-32 scores 0.7596 / 0.7577, PCA-8 scores 0.7500 / 0.7452, and PCA-2 scores 0.4904 / 0.4963. The dimension sweep is in `results/dimensionality_metrics.csv`.

## Latency components

The offline benchmark used 50 warmups and 500 measured, batch-one runs on CUDA. Median component times for the PCA-16 path were: preprocessing 1.89 ms, CPU-to-device 0.60 ms, encoder 20.48 ms, device-to-CPU 0.10 ms, PCA-16 0.33 ms, and k-NN 11.70 ms. Total decision median was 35.38 ms and p95 was 38.14 ms. Raw pixel preprocessing was 0.40 ms, while raw k-NN accounted for about 250.01 ms median. These times exclude Isaac camera capture/rendering and local socket IPC. Full percentiles are in `results/latency_breakdown.csv`.

The actual controller client round-trip medians/p95s from saved decisions were 229.42/237.04 ms for raw RGB, 18.19/18.89 ms for PCA-16, and 4.71/5.03 ms for MoE16. The first cold request took about 155 ms for the latent policies; steady-state p95 after that first request was 5.03 ms for MoE16. This includes image resize, transfer, model inference, and local request/response, but excludes simulator capture/rendering, physics, and actuation.

## Closed-loop trials

Five seeded episodes used the same start range (1.9–2.2 m), target bearing range, smoothing window of 3, two-consecutive-STOP confirmation, action mapping, and success rule. The rule requires a stop at 1.20–1.65 m, absolute bearing no more than 0.18 rad, and no robot-target center distance below 0.30 m.

| Policy | Successes | Failures | Success rate | Mean decision round-trip |
|---|---:|---:|---:|---:|
| Raw RGB + k-NN | 4/5 | 1 | 80% | 228.30 ms |
| ResNet-18 + PCA-16 + k-NN | 3/5 | 2 | 60% | 18.51 ms |
| Compact image MoE + learned latent-16 | 4/5 | 1 | 80% | 5.89 ms |

Raw final distances were 1.478, 1.483, 1.496, 1.704, and 1.469 m. Its only failed run (closed_04) stopped at 1.704 m, just outside the scored band. PCA-16 failed once to produce a STOP within 120 decisions and once stopped with 0.353 rad bearing. No trial triggered the 0.30 m geometry collision proxy. Per-episode results are saved in `results/raw_final/episode_results.csv` and `results/pca16_matched/episode_results.csv`; full action logs are beside those files.

The latest MoE controller demo is `artifacts/demo.mp4`. The raw-RGB baseline footage is `artifacts/demo_raw_final.mp4`; the matched PCA-16 footage is `artifacts/demo_pca16_matched.mp4`. These are actual Isaac Sim recordings rendered to H.264.

## Scope of claims

The initial 520-frame comparison, the rollout autoresearch split, and the container-side collection-transfer check are separate experiments; their offline scores answer different questions. The autoresearch result is strong within one simple Isaac Sim scene, but earlier challenge windows informed model iteration. The newest five-case extension is a more prospective check, not enough by itself to establish broad generalization or MoE-specific novelty. No physical robot or HM3D-OVON benchmark has been evaluated.
