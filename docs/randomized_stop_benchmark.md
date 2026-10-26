# Randomized reactive-stop benchmark

## Task

An NVIDIA JetBot approaches a green cuboid marker at a fixed speed. At irregular times, the same marker moves to a randomized near-field pose and changes to a sampled red-range stop color. In this headless RTX material path, the cue can render dark red or near-black; the evaluator checks the target's projected image region for red or dark pixels. The policy itself receives only RGB. The event schedule, pose, and simulator state are not sent to either inference worker.

Cue onset, cue duration, inter-event interval, distance, bearing, red cue hue, lighting, and episode yaw vary. Each benchmark episode has 20 stop events. By default, each run gets a fresh OS-random seed. The manifest records it so results can be replayed; neither policy receives it. Thus the policy cannot infer the event from the seed or simulator clock, although this remains a reproducible simulation experiment rather than a physical random disturbance.

## Training and comparison

`data/reactive_stop_v2/metadata.csv` holds 2,304 randomized images from 48 collection episodes. Episodes, rather than individual frames, are split 32/8/8 for train, validation, and test. The policy input is RGB only; pose and distance columns are teacher labels and evaluation metadata.

The learned policy is a compact four-expert MoE with a 16-dimensional latent representation. The comparator is a frozen ImageNet ResNet-18 feature extractor followed by train-only PCA-16 and class-reweighted k-NN. PCA dimension and classifier settings are fitted without test data. Both policies use the same episodes and paired event manifest.

## Measurements

- Time from scheduled event onset until the changed cue first registers in an RGB observation. This includes observation cadence and simulator rendering, not just sensor exposure.
- STOP within 250 ms of the first RGB-visible cue.
- Safe stop success from visible cue: a new FORWARD-to-STOP command transition within 250 ms of first visibility and no center-distance collision proxy hit. A policy already stopped before the cue receives no reaction credit.
- Scheduled-onset-to-STOP is also retained as an end-to-end diagnostic, so camera delay is not attributed to policy reaction alone.
- Missed responses, collision proxy hits, false stops, and simulated false-stop time.
- Cue visibility coverage, median and p95 onset-to-visible-observation delay, cue-to-STOP transition time, scheduled-onset-to-STOP delay, false-stop behavior, and end-to-end worker round-trip inference time.
- Per-event and per-episode outcomes in CSV, plus a seed and paired summary in JSON.

The benchmark runs at 60 Hz physics and normally requests an RGB decision at 30 Hz. Worker latency advances the simulation under the previous command, so slow inference consumes braking time. The collision proxy is a 0.30 m horizontal center-distance threshold; it is a simulator metric, not a certified robot safety boundary. The cue is a visual simulation event with a recorded pose proxy, not a rigid-body obstacle collision test.

## Limits

This is a paired simulation experiment with reproducible pseudorandom events. It does not establish performance on a physical robot, robustness to arbitrary real-world objects, or statistical generalization beyond the tested simulator assets and held-out episodes.
