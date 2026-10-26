# Learned-control demos (Isaac Sim 5.1)

Three demos of a compact, camera-only "latent" control policy, trained by imitation
learning (behaviour cloning + DAgger) from a privileged scripted teacher, plus a
Claude Opus planner on top. Everything runs in NVIDIA Isaac Sim with PhysX.

## What is real and what is not

| Component | How it works |
|---|---|
| Physics | PhysX rigid bodies; grasps are friction only (no welds, no teleporting, no velocity setting after reset). |
| Teacher | A scripted controller that reads true object/robot poses from the simulator. It is used **only to label training data**. It never drives a demo labelled "student". |
| Student (the "latent model") | Small CNN per camera -> **16-D latent** -> 4-expert mixture -> action chunk. Inputs: two RGB cameras (2-frame stack), joint angles, gripper width (+ base velocity and a task command for the mobile robot). No simulator state. |
| Action space | Student outputs gripper velocity in the arm's own frame + wrist yaw rate + gripper open/close (and base velocity for the mobile robot). A fixed inverse-kinematics step turns that into joint targets each 15 Hz step (actuation layer, not decision making). |
| Mobile base | Isaac's holonomic Ridgeback model: base moves through x/y/yaw joints, not simulated wheel friction. |
| Planner (Demo 3) | Claude Opus 5.5 via the Claude Code CLI (`claude -p`), given camera images and the instruction, returns one structured decision (which object -> which bin, or done). It never touches the motors. |
| Success metrics | Measured from PhysX object poses after the fact (in bin / on tray), not from the policy's own report. |

## Training recipe

1. Teacher episodes with small action noise injected (labels stay clean).
2. Behaviour cloning of the student on those episodes.
3. DAgger rounds: the student drives most of the time, the teacher labels every visited state, retrain on the aggregate.

Problems found and fixed along the way (useful for the report):

- Isaac cameras default to a ~1 m near clipping plane; the robot could not see anything close. Fixed by setting the clip range on every camera.
- Two Isaac instances on one GPU, and occasionally a single busy instance, return unfinished "sky-only" frames. Fixed by running jobs strictly one at a time and re-rendering when a frame comes back flat; training filters any remaining flat/black frames.
- Joint-space actions could not express the millimetre corrections needed to grasp (the student's error was as large as the correction). Switching to an arm-frame end-effector action space fixed learnability.
- A hard "align within 1 cm, then descend" teacher rule made the student hover forever; a descent funnel (allowed height shrinks with alignment error) fixed it.

## Results (measured, not cherry-picked)

All numbers come from the logged runs in `results/` and `logs/`. "Student drives X%" means DAgger
data collection where each 15 Hz step is executed by the student with probability X and by the teacher otherwise.

### Teachers (data generators)

| Task | Result |
|---|---|
| 6-arm relay, 4 mixed shapes | about 20 hand-offs per 90 s, 0 drops (end-effector teacher) |
| Mobile course, random layouts, 1-3 objects | 67/70 deliveries with injected action noise |
| Mobile course, fixed demo world | 45/45 deliveries |

### Student (camera-only 16-D latent policy)

| Setting | Result |
|---|---|
| Course, fixed world, student drives 70% | 13/20 delivered |
| Course, fixed world, student drives 88% | 7/24 delivered |
| Course, random layouts, student drives 80% | 19/30 delivered |
| Course, student alone (no teacher) | 0 of all student-only evaluation runs so far |
| Relay, student drives 70-80% | 0-9 hand-offs per 90 s (teacher: ~20) |

Decision latency of the student network: ~0.6 ms forward pass / 1.5 ms per decision for all
six arms (CUDA graph, GPU otherwise idle); 6-15 ms inside the running simulator while the GPU
is also rendering, and higher while recording 720p video.

### Honest summary

- The physics, the teachers, the Opus planner and the full imitation-learning/DAgger pipeline work.
- The student clearly learns the task structure (its own phase prediction tracks the teacher's
  98 % of the time on held-out data, and it completes deliveries when the teacher intervenes on a
  minority of steps), but it is **not yet reliable on its own**: small errors in the grasp approach
  compound and it drifts away from the object.
- Most of the debugging time went to simulator issues (camera near-clip default, corrupted frames
  from concurrent/busy RTX rendering) that silently fed the student blank images; once fixed, there
  was not enough time left for the additional DAgger rounds that pure-student control needs.
- Next steps that would most likely close the gap: more DAgger rounds at low teacher share, higher
  wrist-camera resolution for the final centimetres, and a recurrent/longer history encoder.

### Demo 3 (Opus planner + student executor), recorded run

Instruction: "Put the blue ball in the teal bin and the red cube in the magenta bin."
Opus 5.5 (via `claude -p`, 5 calls, 6.0-6.7 s each) planned blue -> teal, saw from the table/bin cameras
that the ball was still on the table, retried, switched to the red cube, retried, and finally stopped
with "Both objects failed twice and remain on the table." The planner's decisions were correct and
grounded in the images; the executor (student) did not complete a delivery. Full log:
`results/demo3_run.json`, `logs/demo3_planner_calls.jsonl`.

## Videos

| File | What it shows |
|---|---|
| `artifacts/demo1_relay_teacher.mp4` | Six Franka arms passing four different shapes simultaneously (12 hand-offs in 60 s, 0 drops). **Driven by the scripted teacher** (labelled on screen). |
| `artifacts/demo2_mobile_student.mp4` | Ridgeback+Franka driven **only by the learned camera-only policy** (teacher not loaded). Two episodes, neither delivered. |
| `artifacts/demo3_opus_planner.mp4` | Claude Opus planning through `claude -p` with the learned policy executing; re-planning after observed failures. |

## Reproduce

```bash
# teacher data, train, DAgger (see scripts/fast_course.sh, scripts/fast_course2.sh)
/c/isaacsim/python.bat src/course/run.py --mode expert --fixed --noise 0.3 --episodes 45 --save-data data/course/fixed_expert
.venv/Scripts/python.exe src/lm/train_bc.py --task course --data data/course/fixed_expert --out models/course_fixed_s1.pt
/c/isaacsim/python.bat src/course/run.py --mode dagger --fixed --beta 0.3 --checkpoint models/course_fixed_s1.pt --save-data data/course/fixed_dagger1
# demos
/c/isaacsim/python.bat src/relay/run.py --mode expert --seconds 60 --frames artifacts/demo1_teacher_frames
/c/isaacsim/python.bat src/course/run.py --mode student --no-teacher --fixed --checkpoint models/course_fixed_s3.pt --frames artifacts/demo2_frames_s3
/c/isaacsim/python.bat src/course/run_planner.py --fixed --checkpoint models/course_fixed_s3.pt --frames artifacts/demo3_frames
.venv/Scripts/python.exe src/tools/encode_video.py artifacts/demo3_frames artifacts/demo3_opus_planner.mp4
```

Run Isaac jobs one at a time: two instances on one GPU corrupt camera frames.


---

## Version 2: skills + Claude Opus as the robot programmer (current)

After v1 the policy was redesigned around short skills, Astra-style:

- **Claude Opus 5.5 writes the program.** Through `claude -p` it sees the table, bin, overview and robot
  cameras plus an execution log, and returns a short program, e.g.
  `pick(blue) face(east) goto(teal) drop()`. After each program, or after any failed step, it checks
  the images and writes the next one (retrying, reordering or stopping).
- **Learned (camera-only latent policy):** `pick(color)` and `drop()`, i.e. gripper x/y/z velocity and open/close
  from the head + wrist cameras. Two specialist checkpoints of the same 16-D latent MoE network:
  `models/skills_p3.pt` for pick, `models/skills_s2.pt` for drop.
- **Not learned, labelled on screen:** `face(heading)` (heading controller), `goto(target)` (grid path planner
  using the map and robot pose), wrist angle hold, and the inverse-kinematics actuation layer. The learned
  `goto` reached the dock 34/36 times with 20% teacher help but only 1/10 on its own, so it was replaced
  by the planner for the demos.

### v2 results (all measured from PhysX)

| Test | Result |
|---|---|
| Learned `pick`, student only, 20 attempts | **14/20 (70%)** |
| Learned `drop`, during DAgger | 28/28 |
| Full deliveries, learned pick/drop + planner goto, 4 episodes x 2 | **4/8** (episode 4: 2/2) |
| Demo 3 take 1 (blue ball -> teal, red cube -> gold) | blue ball delivered; red cube pick timed out twice; Opus then stopped, wrongly believing the cube had left the table |
| Demo 3 take 2 (blue ball -> teal, green puck -> gold) | 0/2: every learned pick timed out; Opus retried, re-docked and switched objects |

Opus calls took 8.8-12.7 s each (structured output, low effort).

### v2 videos

| File | What it shows |
|---|---|
| `artifacts/demo2_skills_best_run.mp4` | Best of 4 episodes: two complete deliveries, pick/drop by the learned policy (42 s) |
| `artifacts/demo2_skills_all_runs.mp4` | All 4 episodes unedited (4/8 deliveries) |
| `artifacts/demo3_opus_programs_robot.mp4` | Opus writing and revising programs; blue ball delivered (take 1, 130 s) |
| `artifacts/demo1_relay_teacher.mp4` | Six-arm relay, scripted teacher (unchanged from v1) |
