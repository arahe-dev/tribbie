# Methodology

## Question and system

The experiment asks whether a frozen pretrained vision representation, compressed with PCA and queried by a small classifier, provides enough information for a simple target-approach controller. The tested stack is Isaac Sim 5.1.0-rc.19 on Windows, the Jetbot differential-drive USD robot, a forward RGB camera, frozen ImageNet ResNet-18 (512 outputs), training-fit PCA, and distance-weighted k-nearest neighbors (k selected from 1, 3, 5, 7, 11 on validation episodes).

Gazebo and ROS 2 were not installed or discoverable in this Windows environment. Isaac Sim was already installed and is used as the native Windows simulator fallback. This is not a Gazebo experiment.

## World and task

The world contains a ground grid, a lit scene, a red pillar target, and the Jetbot. The front camera supplies 320 x 240 RGB frames. The command set is LEFT, FORWARD, RIGHT, STOP. The robot motion check measured forward translation and left/right yaw changes before machine-learning work began. Detailed raw measurements are in `results/camera_actuation_check.csv`.

## Labels and splits

A transparent geometric supervisor labels target-relative bearing and range. Dataset `visible-stop-v2` has 10 independently reset episodes and 520 images (130 per action). Episodes 1–6 are training, 7–8 validation, 9–10 test. Adjacent images never cross a split. The initial STOP samples were visually clipped at short range; the revised task labels STOP at distances up to 1.50 m and samples STOP images at 1.35–1.48 m, where the pillar is visible. The earlier dataset and results remain in `archive/v1/`.

Pose fields are stored for supervision and evaluation only; they are not supplied to the image encoders, PCA, k-NN, or closed-loop action policy. The live five-episode evaluation begins 1.9–2.2 m from the target and uses a separate seed (62024).

## Offline comparison

Three primary policies are compared: bilinear-resized 64 x 64 RGB flattened to 12,288 values, ResNet-18's 512-D output, and the same output projected by PCA to 16 dimensions. Each is classified with distance-weighted k-NN. PCA is fitted on training episodes only. k is selected on validation episodes by macro F1 (accuracy breaks ties; lower k breaks any remaining tie). The held-out test is not used for selection.

A separate dimensionality sweep evaluates 2, 8, 16, 32, 64, and 512-D representations. Reported metrics include held-out accuracy, macro F1, per-class scores, and confusion matrices. This is supervised action-label prediction from simulator-generated views; it does not by itself establish autonomous task completion.

## Experimental learned-latent MoE

The separate `train_moe.py` model starts from 64 x 64 RGB pixels rather than the cached ResNet embedding. A compact depthwise-separable encoder learns a 16-D bottleneck under four-action supervision. A soft gate combines four experts, and a small load-balancing penalty discourages the gate from using only one expert. The model is trained with train episodes only, mild brightness/contrast/saturation augmentation, and horizontal reflection with LEFT/RIGHT labels swapped. Five fixed initialization seeds are compared on validation macro F1; the selected checkpoint is evaluated on the held-out episodes and compared to the existing PCA-16 + k-NN model. This remains a small-data exploratory baseline. The model-forward benchmark excludes camera capture, image resize, and IPC.

A one-expert ablation uses the same image encoder, latent dimension, augmentations, seed selection, and episode split, with the gate reduced to its identity mixture. On the current held-out set it slightly exceeds the four-expert variant, so expert routing has not shown a benefit yet. Its metrics are in `results/moe_latent16_e1_metrics.json`.

## Runtime measurements

GPU inference is hosted by a separate project Python process because Isaac Sim's Kit Python DLL environment conflicted with the CUDA-enabled PyTorch wheel. The simulator sends each real camera frame over localhost TCP to the worker and applies its predicted action. Offline component timings and live client round-trip timings are reported separately. The component benchmark uses 50 warmup iterations and 500 measured iterations by default. It is a single-frame inference benchmark, not a hard real-time guarantee.

## Closed loop

`src/run_policy.py` runs five seeded episodes by default. It sends only camera RGB to the selected visual policy, temporally smooths votes over three decisions, and requires two consecutive STOP predictions before braking. The action runs include `raw` (64 x 64 pixel k-NN), `pca16`, and `pca64`. Simulator state is read for scoring and safety diagnostics only. Success requires STOP at 1.20–1.65 m, absolute bearing <=0.18 rad, and no geometry-proxy collision (robot-target center distance <0.30 m). The nominal simulated command interval is 9/60 s (6.67 Hz); wall time also includes perception, socket transfer, rendering, and physics.

Trial-level results, action traces, image frames, timing, and score summaries are saved under `results/` and `artifacts/`. H.264 validation details are in `results/video_info.json`.
