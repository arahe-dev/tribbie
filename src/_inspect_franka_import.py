from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import os
import traceback
import omni.kit.commands
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdLux
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.sensors.camera import Camera


def look_orientation(position, target):
    forward = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    forward /= np.linalg.norm(forward)
    up0 = np.array([0.0, 0.0, 1.0])
    left = np.cross(up0, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    m = np.column_stack([forward, left, up])
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)

try:
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 30.0)
    stage = world.stage
    ext = get_extension_path_from_name("isaacsim.asset.importer.urdf")
    urdf = os.path.join(ext, "data", "urdf", "robots", "franka_description", "robots", "panda_arm_hand.urdf")
    status, config = omni.kit.commands.execute("URDFCreateImportConfig")
    config.merge_fixed_joints = False
    config.convex_decomp = False
    config.import_inertia_tensor = True
    config.fix_base = True
    config.collision_from_visuals = False
    config.create_physics_scene = True
    config.make_default_prim = False
    status, root = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=urdf, import_config=config, get_articulation_root=True)
    print(f"IMPORT status={status} root={root} urdf={urdf}", flush=True)
    rp = stage.GetPrimAtPath(root)
    xf = UsdGeom.Xformable(rp)
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Light"))
    dome.CreateIntensityAttr(900.0)
    camera_position = np.array([0.0, -2.3, 1.2], dtype=np.float32)
    camera = Camera(prim_path="/World/Camera", name="camera", position=camera_position, orientation=look_orientation(camera_position, [0.0, 0.0, 0.55]), resolution=(960, 720), frequency=30)
    camera.set_focal_length(1.8)
    world.reset()
    articulation_path = root.rsplit("/", 1)[0]
    articulation = SingleArticulation(prim_path=articulation_path, name="franka")
    articulation.initialize()
    print(f"ARTIC joints={articulation.num_dof} names={articulation.dof_names}", flush=True)
    print(f"INITIAL={articulation.get_joint_positions().tolist()}", flush=True)
    solver = KinematicsSolver(articulation, end_effector_frame_name="panda_hand")
    actions, ok = solver.compute_inverse_kinematics(target_position=np.array([0.55, 0.0, 0.8], dtype=np.float32))
    print(f"IK success={ok} action={None if actions is None else actions.joint_positions}", flush=True)
    if ok:
        articulation.apply_action(actions)
    camera.initialize()
    for i in range(240):
        world.step(render=True)
        if i == 80:
            rgb = camera.get_rgb(device="cpu")
            if rgb is not None:
                Image.fromarray(np.asarray(rgb, dtype=np.uint8)[..., :3]).save(r"C:\Users\mirai\latent_robot_demo\artifacts\_franka_import_smoke.png")
    print("DONE", flush=True)
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()
