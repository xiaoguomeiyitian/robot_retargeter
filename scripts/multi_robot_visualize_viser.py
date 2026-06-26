"""Visualize motion trajectories of multiple robots simultaneously using Viser.

This provides a browser-based 3D visualization alternative to the MuJoCo native viewer.
Open the printed URL in a browser to view the simulation.

File naming convention: <motion_name>_<robot_name>.csv, for example Form_1_stageii_g1.csv.
So you only need to provide one motion name and a list of robot names; the script will
construct the file paths automatically.

Note: MuJoCo free-joint qpos ordering is [pos(xyz), quat(wxyz)].
After loading CSV files, quaternions must be converted from xyzw to wxyz.

Usage example:
    python scripts/multi_robot_visualize_viser.py \
        --motion body_check_001__A548_M_from_g1 \
        --robots h2 \
        --source_fps 120 \
        --render_fps 30 \
        --port 8080
"""

import argparse
import math
import os
import time
from typing import Any

import mujoco
import numpy as np
import trimesh
import viser
import yaml
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Robot config directory: each robot has a <robot_name>.yaml that contains a
# "robot_xml_path" field (relative to the project root) pointing to its MJCF.
ROBOT_CONFIG_DIR = os.path.join(PROJECT_DIR, "config", "robot")

MOTION_DIR = os.path.join(SCRIPT_DIR, "..", "output_data", "robot_motion")

# 动作中文名称映射
MOTION_LABELS_CN = {
    "Form_1_stageii": "形体检查",
    "dance1": "舞蹈 1",
    "dance2": "舞蹈 2",
    "fallAndGetUp1": "跌倒起身 1",
    "fallAndGetUp2": "跌倒起身 2",
    "fallAndGetUp3": "跌倒起身 3",
    "fight1": "格斗 1",
    "fightAndSports1": "格斗运动 1",
    "grab_walk_ff_180_001__A550": "抓握行走",
    "jumps1": "跳跃 1",
    "run1": "跑步 1",
    "run2": "跑步 2",
    "sprint1": "冲刺 1",
    "walk1": "行走 1",
    "walk2": "行走 2",
    "walk3": "行走 3",
    "walk4": "行走 4",
    "body_check_001__A548": "形体检查",
}

# 未注册机器人的 fallback 配置 (机器人名 -> 使用的已注册机器人配置)
ROBOT_FALLBACK = {
    "M": "g1",
}


def resolve_robot(robot: str) -> str:
    """如果机器人未注册，返回 fallback 配置名。"""
    if robot in available_robots():
        return robot
    if robot in ROBOT_FALLBACK:
        return ROBOT_FALLBACK[robot]
    return robot


def get_motion_label_cn(motion_name: str) -> str:
    """获取动作中文名称，没有映射则返回原名。"""
    return MOTION_LABELS_CN.get(motion_name, motion_name)


def available_robots() -> list[str]:
    """List robot names by scanning config/robot/*.yaml (filename = robot name)."""
    if not os.path.isdir(ROBOT_CONFIG_DIR):
        return []
    names = [
        os.path.splitext(f)[0]
        for f in os.listdir(ROBOT_CONFIG_DIR)
        if f.endswith(".yaml")
    ]
    return sorted(names)


def get_robot_xml(robot: str) -> str:
    """Resolve a robot's MJCF path by reading robot_xml_path from its yaml config."""
    config_path = os.path.join(ROBOT_CONFIG_DIR, f"{robot}.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Robot config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    xml_path = config.get("robot_xml_path")
    if not xml_path:
        raise KeyError(f"'robot_xml_path' missing in config: {config_path}")
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(PROJECT_DIR, xml_path)
    return xml_path


def load_motion(csv_path: str) -> np.ndarray:
    """Load a motion CSV and convert the base quaternion from xyzw to wxyz."""
    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 7:
        raise ValueError(
            f"Motion file {csv_path} must have at least 7 columns (pos3 + quat4), got {data.shape[1]}"
        )

    qpos = data.copy()
    quat_xyzw = data[:, 3:7]
    qpos[:, 3] = quat_xyzw[:, 3]  # w
    qpos[:, 4] = quat_xyzw[:, 0]  # x
    qpos[:, 5] = quat_xyzw[:, 1]  # y
    qpos[:, 6] = quat_xyzw[:, 2]  # z
    return qpos


def build_combined_spec(robots: list[str]) -> mujoco.MjSpec:
    """Merge multiple robot MJCF models into one scene."""
    spec = mujoco.MjSpec()
    spec.modelname = "multi_robot"

    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.specular = [0.3, 0.3, 0.3]

    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=512,
    )

    grid_res = 256
    line_px = 3
    grid_rgba = np.zeros((grid_res, grid_res, 4), dtype=np.uint8)
    bg_color = (100, 100, 100)
    line_color = (50, 50, 50)
    grid_rgba[:, :, :3] = bg_color
    grid_rgba[:, :, 3] = 255
    grid_rgba[:line_px, :, :3] = line_color
    grid_rgba[-line_px:, :, :3] = line_color
    grid_rgba[:, :line_px, :3] = line_color
    grid_rgba[:, -line_px:, :3] = line_color

    grid_tex = spec.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=grid_res,
        height=grid_res,
        nchannel=4,
    )
    grid_tex.data = grid_rgba.reshape(-1).tobytes()

    spec.add_material(
        name="groundplane",
        textures=["", "groundplane"],
        texrepeat=[10, 10],
        texuniform=True,
        reflectance=0.0,
    )

    spec.worldbody.add_light(
        pos=[0, 0, 20.0],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.7, 0.7, 0.7],
        specular=[0.3, 0.3, 0.3],
    )

    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        material="groundplane",
        pos=[0, 0, 0],
    )

    for robot in robots:
        resolved = resolve_robot(robot)
        robot_spec = mujoco.MjSpec.from_file(get_robot_xml(resolved))
        for j in robot_spec.joints:
            if j.type == mujoco.mjtJoint.mjJNT_FREE:
                j.name = "floating_base_joint"
                break
        frame = spec.worldbody.add_frame()
        spec.attach(robot_spec, prefix=f"{robot}_", frame=frame)

    for g in spec.geoms:
        g.contype = 0
        g.conaffinity = 0

    return spec


def get_qpos_start(model: mujoco.MjModel, freejoint_name: str) -> int:
    """Return the starting qpos index of a robot free joint in the merged model."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, freejoint_name)
    if jid < 0:
        raise ValueError(f"Joint not found in model: {freejoint_name}")
    return model.jnt_qposadr[jid]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Viser-based multi-robot motion visualization (browser-based)"
    )
    parser.add_argument(
        "--motion",
        default="body_check_001__A548_M_from_g1",
        help="Motion name (without robot suffix or extension), filename format: <motion_name>_<robot_name>.csv",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        default=["h2"],
        help="List of robot names to visualize (any name is accepted, unregistered robots use fallback config)",
    )
    parser.add_argument(
        "--motion_dir",
        default=MOTION_DIR,
        help="Directory containing motion CSV files",
    )
    parser.add_argument("--source_fps", type=float, default=30.0, help="Source data frame rate")
    parser.add_argument(
        "--render_fps",
        type=float,
        default=30.0,
        help="Target rendering FPS for playback",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port for viser server")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument(
        "--no-ground",
        action="store_true",
        help="Do not show ground plane",
    )
    args = parser.parse_args()

    robots = args.robots

    # Load motion data for each robot
    robot_qpos: dict[str, np.ndarray] = {}
    for robot in robots:
        csv_path = os.path.join(args.motion_dir, f"{args.motion}_{robot}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Motion file not found: {csv_path}")
        robot_qpos[robot] = load_motion(csv_path)

    n_frames = min(q.shape[0] for q in robot_qpos.values())
    print(
        "  ".join(f"{r} frames={robot_qpos[r].shape[0]} (qpos {robot_qpos[r].shape[1]})" for r in robots)
        + f"  ->  playing {n_frames} frames"
    )

    # Build combined MuJoCo spec and compile
    spec = build_combined_spec(robots)
    model = spec.compile()
    data = mujoco.MjData(model)

    # Get qpos start indices for each robot
    robot_start: dict[str, int] = {}
    robot_dim: dict[str, int] = {}
    for robot in robots:
        start = get_qpos_start(model, f"{robot}_floating_base_joint")
        dim = robot_qpos[robot].shape[1]
        if start + dim > model.nq:
            raise ValueError(
                f"qpos dimension mismatch: {robot}[{start}:{start + dim}] exceeds model.nq={model.nq}"
            )
        robot_start[robot] = start
        robot_dim[robot] = dim

    # Compute XY placement offsets (grid layout)
    spacing = 2.0
    n = len(robots)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    robot_offset: dict[str, tuple[float, float]] = {}
    for idx, robot in enumerate(robots):
        col = idx % cols
        row = idx // cols
        dx = (col - (cols - 1) / 2.0) * spacing
        dy = (row - (rows - 1) / 2.0) * spacing
        robot_offset[robot] = (dx, dy)

    # Extract geom info for visualization
    geom_name_to_id: dict[str, int] = {}
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and name != "ground":
            geom_name_to_id[name] = i

    # Create viser server
    server = viser.ViserServer(port=args.port, verbose=False)
    server.scene.set_up_direction("+z")
    server.scene.configure_default_lights()

    # Add ground plane
    if not args.no_ground:
        ground_mesh = trimesh.creation.box(extents=[20, 20, 0.02])
        ground_mesh.apply_translation([0, 0, -0.01])
        server.scene.add_mesh_trimesh(
            "ground",
            ground_mesh,
            position=(0, 0, -0.01),
            wxyz=(1, 0, 0, 0),
        )

    # Add geom-based meshes grouped by robot
    geom_handles: dict[str, Any] = {}

    for geom_name, geom_id in geom_name_to_id.items():
        # Determine which robot this geom belongs to
        parent_robot = None
        for robot in robots:
            if geom_name.startswith(f"{robot}_"):
                parent_robot = robot
                break
        if parent_robot is None:
            continue

        geom = model.geom(geom_id)
        geom_type = geom.type
        size = geom.size

        # Create trimesh based on geom type
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=size * 2)
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mesh = trimesh.creation.cylinder(radius=size[0], height=size[1] * 2)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            mesh = trimesh.creation.capsule(radius=size[0], height=size[1] * 2)
        elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            mesh = trimesh.creation.box(extents=[2, 2, 0.01])
        else:
            continue

        # Get parent body name for scene tree organization
        body_id = geom.bodyid
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None:
            body_name = "unknown"

        # Get initial transform
        geom_pos = data.geom_xpos[geom_id].copy()
        geom_mat = data.geom_xmat[geom_id].copy().reshape(3, 3)

        dx, dy = robot_offset[parent_robot]
        geom_pos[0] += dx
        geom_pos[1] += dy

        # Convert rotation matrix to quaternion (wxyz)
        quat_xyzw = Rotation.from_matrix(geom_mat).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        handle = server.scene.add_mesh_trimesh(
            f"robots/{parent_robot}/{body_name}/{geom_name}",
            mesh,
            position=tuple(geom_pos),
            wxyz=tuple(quat_wxyz),
        )
        geom_handles[geom_name] = handle

    # Frame skipping
    step = max(1, round(args.source_fps / args.render_fps))
    dt = step / args.source_fps
    print(f"source_fps={args.source_fps:g}, render_fps≈{args.source_fps / step:g}, render_every={step} frame(s)")
    print(f"Open browser at: http://localhost:{args.port}")
    print("Controls: [Space] Pause/Resume  [R] Replay")

    # Add GUI controls
    with server.gui.add_folder("Playback Controls"):
        gui_pause = server.gui.add_button("Pause")
        gui_replay = server.gui.add_button("Replay")
        gui_frame = server.gui.add_slider("Frame", min=0, max=n_frames - 1, step=1, initial_value=0)
        gui_speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)

    paused = False
    replay = False
    current_frame = 0

    @gui_pause.on_click
    def _(_):
        nonlocal paused
        paused = not paused
        gui_pause.name = "Resume" if paused else "Pause"

    @gui_replay.on_click
    def _(_):
        nonlocal replay
        replay = True

    # Apply initial frame
    def apply_frame(frame_idx: int) -> None:
        """Update all geom transforms for the given frame."""
        # Set MuJoCo qpos for all robots
        for robot in robots:
            start = robot_start[robot]
            dim = robot_dim[robot]
            data.qpos[start:start + dim] = robot_qpos[robot][frame_idx]
            dx, dy = robot_offset[robot]
            data.qpos[start] += dx
            data.qpos[start + 1] += dy

        # Forward kinematics to update body/geom transforms
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        # Update viser mesh transforms
        for geom_name, handle in geom_handles.items():
            geom_id = geom_name_to_id[geom_name]
            geom_pos = data.geom_xpos[geom_id].copy()
            geom_mat = data.geom_xmat[geom_id].copy().reshape(3, 3)

            # Find parent robot for offset
            parent_robot = None
            for robot in robots:
                if geom_name.startswith(f"{robot}_"):
                    parent_robot = robot
                    break
            if parent_robot is not None:
                dx, dy = robot_offset[parent_robot]
                geom_pos[0] += dx
                geom_pos[1] += dy

            quat_xyzw = Rotation.from_matrix(geom_mat).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

            handle.position = tuple(geom_pos)
            handle.wxyz = tuple(quat_wxyz)

    apply_frame(0)

    # Main loop
    while True:
        t0 = time.time()

        # Handle replay
        if replay:
            current_frame = 0
            replay = False
            paused = False
            gui_pause.name = "Pause"

        # Handle GUI frame slider
        if gui_frame.value != current_frame:
            current_frame = int(gui_frame.value)
            apply_frame(current_frame)
            gui_frame.value = current_frame

        if not paused:
            apply_frame(current_frame)
            gui_frame.value = current_frame
            current_frame += step
            if current_frame >= n_frames:
                if args.loop:
                    current_frame = 0
                else:
                    current_frame = n_frames - 1
                    paused = True
                    gui_pause.name = "Resume"

        # Frame rate control
        elapsed = time.time() - t0
        sleep_time = dt / gui_speed.value - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
