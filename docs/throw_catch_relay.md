# Six-arm physics relay

## Current video

`artifacts/demo_throw_catch_6arm.mp4` is the physics-driven replacement for the earlier scripted animation at the same path. The final render completes 18/18 contact-verified handoffs (three circuits) through six articulated Franka Panda arms. It is 1,559 frames, 30 fps, 1280×720 H.264, with HERO, OVERHEAD, SIDE, and REVERSE camera shots. The earlier video is kept as `artifacts/demo_throw_catch_6arm_before_physics.mp4`.

## Simulation

Isaac Sim runs the articulated Panda assets with 120 Hz PhysX integration and 30 fps rendering. A 60 g, 3.3 cm radius dynamic rigid-body ball receives a release impulse; PhysX then integrates its gravity-driven flight and contacts. The catcher closes its parallel gripper, and a contact sensor must register a finger-to-ball collision before a fixed grasp constraint carries the ball into the next throw pose. The arm motion eases joint targets from measured poses.

The same `models/throw_catch_moe16.pt` checkpoint is loaded for visual inference. It has a 16-D latent and four experts. In this scene all 72 model cues were `CATCH`, so a scheduled controller supplied each of the 18 throw actions. Mean inference round-trip latency was 11.70 ms and p95 was 21.65 ms. The video and run report retain those model cues and count every scheduled throw; the result is not an end-to-end learned throw policy.

Robot link collisions are disabled except for the active catcher's finger meshes; the ball, turntable, and studio floor retain PhysX collision. This keeps the six arms from interpenetrating one another during this relay choreography. The video is a simulator result, not a physical robot trial or a broad generalization benchmark.

## Reproduce

```powershell
& .\scripts\run_isaac.ps1 -EntryPoint 'src\demo_throw_catch_physics_6arm.py' -ScriptArgs @('--transfers','18','--device','cuda')
.\.venv\Scripts\python.exe src\encode_physics_relay_video.py
```

The run report is `results/throw_catch_physics_6arm.json`; confirmed finger contacts are in `results/throw_catch_physics_contacts.csv`, and sampled ball/catcher trajectories are in `results/throw_catch_physics_ball_trajectory.csv`.
