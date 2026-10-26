from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import csv
import numpy as np
from PIL import Image

from isaac_common import ROBOT_START_Z, action_command, build_scene, camera_rgb, place_target_relative, target_relative_state, update_front_camera, yaw_quaternion


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

try:
    world, robot, target, camera, _, controller = build_scene()
    print(f"CAMERA_LENS focal={camera.get_focal_length():.4f} aperture={camera.get_horizontal_aperture():.4f}", flush=True)
    rows = []
    first = None
    for tick in range(35):
        update_front_camera(camera, robot)
        world.step(render=True)
        if tick >= 24:
            frame = camera_rgb(camera)
            if frame is not None:
                first = frame
                break
    if first is None:
        raise RuntimeError("RGB camera did not publish a frame after 35 render steps")
    Image.fromarray(first).save(ARTIFACTS / "camera_test.png")
    print(f"CAMERA_FRAME width={first.shape[1]} height={first.shape[0]} min={first.min()} max={first.max()}", flush=True)

    start = np.asarray(robot.get_world_pose()[0], dtype=float)
    for action in ("FORWARD", "LEFT", "RIGHT", "STOP"):
        v, w = action_command(action)
        before = np.asarray(robot.get_world_pose()[0], dtype=float)
        for _ in range(36):
            update_front_camera(camera, robot)
            robot.apply_wheel_actions(controller.forward(command=[v, w]))
            world.step(render=True)
        after = np.asarray(robot.get_world_pose()[0], dtype=float)
        distance, bearing, yaw = target_relative_state(robot, target)
        rows.append({"action": action, "x_before": before[0], "y_before": before[1], "x_after": after[0], "y_after": after[1], "distance_to_target_m": distance, "target_bearing_rad": bearing, "yaw_rad": yaw})
        print(f"ACTUATION action={action} pose_delta_m={np.linalg.norm(after[:2]-before[:2]):.4f} yaw_rad={yaw:.3f}", flush=True)

    with (ROOT / "results" / "camera_actuation_check.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"ACTUATION_CHECK start_pose={start.tolist()} final_pose={robot.get_world_pose()[0].tolist()}", flush=True)

    robot.set_world_pose(position=np.array([0.0, 0.0, ROBOT_START_Z], dtype=np.float32), orientation=yaw_quaternion(0.0))
    robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
    robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
    for distance in (1.15, 1.25, 1.35, 1.45, 1.80):
        robot.set_world_pose(position=np.array([0.0, 0.0, ROBOT_START_Z], dtype=np.float32), orientation=yaw_quaternion(0.0))
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        place_target_relative(target, np.array([0.0, 0.0]), 0.0, 0.0, distance)
        for _ in range(30):
            update_front_camera(camera, robot)
            world.step(render=True)
        sample = camera_rgb(camera)
        label = f"distance_{distance:.2f}"
        Image.fromarray(sample).save(ARTIFACTS / f"{label}.png")
        red_mask = (sample[:, :, 0] > 110) & (sample[:, :, 1] < 140) & (sample[:, :, 2] < 120)
        target_position = target.get_world_pose()[0]
        print(f"VISIBILITY distance={distance:.2f} red_pixels={int(red_mask.sum())} target_x={float(target_position[0]):.3f} target_z={float(target_position[2]):.3f}", flush=True)
    stop_frame = np.asarray(Image.open(ARTIFACTS / "distance_1.35.png").convert("RGB"))
    Image.fromarray(stop_frame).save(ARTIFACTS / "stop_camera_test.png")
finally:
    app.close()
