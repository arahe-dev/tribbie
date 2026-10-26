from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import numpy as np
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import Franka, KinematicsSolver
from pxr import UsdGeom

try:
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 30.0)
    robot = world.scene.add(Franka(prim_path="/World/NativeFranka", name="native_franka", position=np.array([0.98, 0.0, 0.18]), orientation=np.array([0.0, 0.0, 0.0, 1.0])))
    world.reset()
    print(f"DOF {robot.num_dof} {robot.dof_names}", flush=True)
    print(f"EE_PATH {robot._end_effector_prim_path}", flush=True)
    print(f"HAND_VALID {world.stage.GetPrimAtPath('/World/NativeFranka/panda_hand').IsValid()}", flush=True)
    hand = world.stage.GetPrimAtPath('/World/NativeFranka/panda_hand')
    mat = UsdGeom.XformCache().GetLocalToWorldTransform(hand)
    print(f"HAND_POSE {mat.ExtractTranslation()} {mat.ExtractRotationQuat()}", flush=True)
    solver = KinematicsSolver(robot)
    rotation = mat.ExtractRotationQuat()
    home_orientation = np.array([rotation.GetReal(), *rotation.GetImaginary()], dtype=np.float32)
    print(f"HOME_Q {home_orientation}", flush=True)
    for pos in ([0.416, 0.0, 0.896], [0.416, 0.24, 0.70], [0.416, 0.0, 0.796]):
        result, ok = solver.compute_inverse_kinematics(target_position=np.array(pos, dtype=np.float32), target_orientation=home_orientation)
        print(f"IK pos={pos} orientation=true {ok} {None if result is None else result.joint_positions}", flush=True)
    print(f"LULA_FRAMES {solver._kinematics.get_all_frame_names()}", flush=True)
    for _ in range(60):
        world.step(render=True)
finally:
    app.close()
