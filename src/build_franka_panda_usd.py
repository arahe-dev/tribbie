"""Convert the bundled Franka Panda URDF into a reusable Isaac USD asset."""

from pathlib import Path
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.kit.commands
from isaacsim.core.utils.extensions import get_extension_path_from_name

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "franka_panda_imported.usd"

try:
    extension = Path(get_extension_path_from_name("isaacsim.asset.importer.urdf"))
    urdf = extension / "data" / "urdf" / "robots" / "franka_description" / "robots" / "panda_arm_hand.urdf"
    status, config = omni.kit.commands.execute("URDFCreateImportConfig")
    config.merge_fixed_joints = False
    config.convex_decomp = False
    config.import_inertia_tensor = True
    config.fix_base = True
    config.collision_from_visuals = False
    config.create_physics_scene = False
    config.make_default_prim = False
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf),
        import_config=config,
        dest_path=str(DEST).replace("\\", "/"),
        get_articulation_root=True,
    )
    print(f"ASSET_IMPORT status={status} root={prim_path} dest={DEST} exists={DEST.exists()}", flush=True)
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().Save()
finally:
    app.close()
