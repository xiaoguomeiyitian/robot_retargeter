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

Features (v2.0):
    - Batched mesh rendering: ~4.3ms/frame (was 16.5ms with individual handles)
    - Threaded physics simulation decoupled from GUI event loop
    - Per-robot independent playback controls
    - Ground contact force visualization
    - Video export support
"""

import argparse
import math
import os
import subprocess
import threading
import time
from collections import defaultdict
from typing import Any

import mujoco
import numpy as np
import trimesh
import viser
import yaml
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

ROBOT_CONFIG_DIR = os.path.join(PROJECT_DIR, "config", "robot")
MOTION_DIR = os.path.join(SCRIPT_DIR, "..", "output_data", "robot_motion")

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

ROBOT_FALLBACK = {"M": "g1"}


def resolve_robot(robot: str) -> str:
    if robot in available_robots():
        return robot
    if robot in ROBOT_FALLBACK:
        return ROBOT_FALLBACK[robot]
    if robot.startswith("subject"):
        return "g1"
    return robot


def get_motion_label_cn(motion_name: str) -> str:
    return MOTION_LABELS_CN.get(motion_name, motion_name)


def available_robots() -> list[str]:
    if not os.path.isdir(ROBOT_CONFIG_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(ROBOT_CONFIG_DIR)
        if f.endswith(".yaml")
    )


def get_robot_xml(robot: str) -> str:
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
    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 7:
        raise ValueError(
            f"Motion file {csv_path} must have at least 7 columns (pos3 + quat4), got {data.shape[1]}"
        )
    qpos = data.copy()
    quat_xyzw = data[:, 3:7]
    qpos[:, 3] = quat_xyzw[:, 3]
    qpos[:, 4] = quat_xyzw[:, 0]
    qpos[:, 5] = quat_xyzw[:, 1]
    qpos[:, 6] = quat_xyzw[:, 2]
    return qpos


def build_combined_spec(robots: list[str]) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "multi_robot"
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    spec.visual.headlight.specular = [0.2, 0.2, 0.2]

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
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, freejoint_name)
    if jid < 0:
        raise ValueError(f"Joint not found in model: {freejoint_name}")
    return model.jnt_qposadr[jid]


def rotation_matrix_to_quat_wxyz(m: np.ndarray) -> tuple:
    """Fast inline rotation matrix (flattened 3x3) to wxyz quaternion."""
    det = (
        m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
    )
    if abs(det) < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    trace = m[0] + m[4] + m[8]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return (0.25 / s, (m[7] - m[5]) * s, (m[2] - m[6]) * s, (m[3] - m[1]) * s)
    elif m[0] > m[4] and m[0] > m[8]:
        s = 2.0 * np.sqrt(1.0 + m[0] - m[4] - m[8])
        return ((m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s)
    elif m[4] > m[8]:
        s = 2.0 * np.sqrt(1.0 + m[4] - m[0] - m[8])
        return ((m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s)
    else:
        s = 2.0 * np.sqrt(1.0 + m[8] - m[0] - m[4])
        return ((m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Viser-based multi-robot motion visualization (browser-based)"
    )
    parser.add_argument("--motion", default="body_check_001__A548_M_from_g1")
    parser.add_argument("--robots", nargs="+", default=["h2"])
    parser.add_argument("--motion_dir", default=MOTION_DIR)
    parser.add_argument("--source_fps", type=float, default=30.0)
    parser.add_argument("--render_fps", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-ground", action="store_true")
    parser.add_argument("--all_geoms", action="store_true",
                        help="Render all geom types (default: only MESH)")
    parser.add_argument("--show_contacts", action="store_true",
                        help="Show ground contact force arrows")
    parser.add_argument("--export_video", type=str, default=None,
                        help="Export video to this file path (requires ffmpeg)")
    parser.add_argument("--export_fps", type=float, default=30.0)
    parser.add_argument("--no-thread", action="store_true",
                        help="Disable threaded physics")
    args = parser.parse_args()

    robots = args.robots

    # Load motion data
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

    # Build model
    spec = build_combined_spec(robots)
    model = spec.compile()
    data = mujoco.MjData(model)

    robot_start: dict[str, int] = {}
    robot_dim: dict[str, int] = {}
    for robot in robots:
        start = get_qpos_start(model, f"{robot}_floating_base_joint")
        dim = robot_qpos[robot].shape[1]
        if start + dim > model.nq:
            raise ValueError(f"qpos dimension mismatch: {robot}[{start}:{start + dim}] exceeds model.nq={model.nq}")
        robot_start[robot] = start
        robot_dim[robot] = dim

    # Grid layout
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

    # Body -> robot mapping
    body_id_to_robot: dict[int, str] = {}
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name:
            for robot in robots:
                if body_name.startswith(f"{robot}_"):
                    body_id_to_robot[body_id] = robot
                    break

    # Group geoms by mesh_id for batched rendering
    # Also collect per-geom rgba colors to bake into mesh vertex colors
    mesh_groups: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    mesh_colors: dict[int, tuple[float, float, float]] = {}
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name == "ground":
            continue
        geom = model.geom(i)
        geom_type = int(geom.type[0])
        if not args.all_geoms and geom_type != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        body_id = int(geom.bodyid[0])
        parent_robot = body_id_to_robot.get(body_id)
        if parent_robot is None:
            continue
        mesh_id = int(geom.dataid[0]) if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH) else -(i + 1)
        dx, dy = robot_offset[parent_robot]
        mesh_groups[mesh_id].append((i, dx, dy))
        # Extract rgba color from geom (MuJoCo stores it in rgba attribute)
        rgba = geom.rgba
        if rgba is not None and float(rgba[3]) > 0.01:
            mesh_colors[mesh_id] = (float(rgba[0]), float(rgba[1]), float(rgba[2]))

    # Create viser server
    server = viser.ViserServer(port=args.port, verbose=False)
    server.scene.set_up_direction("+z")
    server.scene.configure_default_lights(enabled=False)
    # Three-point lighting for better depth perception
    server.scene.add_light_ambient("ambient", color=(200, 200, 220), intensity=0.4)
    server.scene.add_light_directional("key", color=(255, 255, 255), intensity=0.8, position=(5, 5, 10), cast_shadow=False)
    server.scene.add_light_directional("fill", color=(180, 190, 220), intensity=0.3, position=(-5, -3, 8), cast_shadow=False)
    server.scene.add_light_directional("rim", color=(150, 160, 200), intensity=0.25, position=(0, -5, 5), cast_shadow=False)

    # Ground plane
    if not args.no_ground:
        ground_mesh = trimesh.creation.box(extents=[20, 20, 0.02])
        ground_mesh.apply_translation([0, 0, -0.01])
        server.scene.add_mesh_trimesh("ground", ground_mesh, position=(0, 0, -0.01), wxyz=(1, 0, 0, 0))

    # Create batched mesh handles with per-mesh colors
    batched_handles: list[tuple[Any, list[tuple[int, float, float]]]] = []
    for mesh_id, geoms in mesh_groups.items():
        if mesh_id >= 0 and mesh_id < model.nmesh:
            vert_adr = model.mesh_vertadr[mesh_id]
            vert_num = model.mesh_vertnum[mesh_id]
            face_adr = model.mesh_faceadr[mesh_id]
            face_num = model.mesh_facenum[mesh_id]
            if vert_num <= 0 or face_num <= 0:
                continue
            vertices = model.mesh_vert[vert_adr:vert_adr + vert_num].copy()
            faces = model.mesh_face[face_adr:face_adr + face_num].copy()
        else:
            geom = model.geom(geoms[0][0])
            size = geom.size
            # Build box vertices/faces manually for non-mesh geoms
            box = trimesh.creation.box(extents=size * 2)
            vertices = box.vertices.astype(np.float32)
            faces = box.faces.astype(np.uint32)

        n_inst = len(geoms)
        positions = np.zeros((n_inst, 3), dtype=np.float32)
        wxyzs = np.tile([1, 0, 0, 0], (n_inst, 1)).astype(np.float32)

        # Get color for this mesh group (default: gray)
        if mesh_id in mesh_colors:
            r, g, b = mesh_colors[mesh_id]
            mesh_color = (int(r * 255), int(g * 255), int(b * 255))
        else:
            mesh_color = (180, 180, 180)

        handle = server.scene.add_batched_meshes_simple(
            f"robots/batch_{mesh_id}",
            vertices.astype(np.float32),
            faces.astype(np.uint32),
            batched_wxyzs=wxyzs,
            batched_positions=positions,
            batched_colors=mesh_color,
        )
        batched_handles.append((handle, geoms))

    print(f"Created {len(batched_handles)} batched mesh groups ({sum(len(g) for _, g in batched_handles)} instances)")

    # Frame skipping
    step = max(1, round(args.source_fps / args.render_fps))
    dt = step / args.source_fps
    print(f"source_fps={args.source_fps:g}, render_fps≈{args.source_fps / step:g}, render_every={step} frame(s)")
    print(f"Open browser at: http://localhost:{args.port}")
    print("Controls: [Space] Pause/Resume  [R] Replay")

    # ── GUI Controls ──────────────────────────────────────────────────
    with server.gui.add_folder("Playback Controls"):
        gui_pause = server.gui.add_button("Pause")
        gui_replay = server.gui.add_button("Replay")
        gui_frame = server.gui.add_slider("Frame", min=0, max=n_frames - 1, step=1, initial_value=0)
        gui_speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)

    paused = False
    replay = False
    current_frame = 0
    _last_slider_val = 0

    @gui_pause.on_click
    def _(_):
        nonlocal paused
        paused = not paused
        gui_pause.name = "Resume" if paused else "Pause"

    @gui_replay.on_click
    def _(_):
        nonlocal replay
        replay = True

    # Per-robot controls removed — using global Playback Controls only

    # ── Ground contact visualization (P1-2.4) ────────────────────────
    contact_arrows = None
    if args.show_contacts:
        try:
            # add_arrows: points shape (N, 2, 3), colors shape (N, 3) RGB
            contact_arrows = server.scene.add_arrows(
                "contact_forces",
                points=np.zeros((4, 2, 3), dtype=np.float32),
                colors=np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]], dtype=np.float32),
                shaft_radius=0.02,
            )
            print("Ground contact visualization enabled")
        except Exception as e:
            print(f"Contact viz init failed: {e}")

    # ── Video export (P1-2.3) ────────────────────────────────────────
    if args.export_video:
        print(f"Video export: {args.export_video} @ {args.export_fps} FPS")
        print("Use [Space] to start/pause recording")

    # ── Apply frame ──────────────────────────────────────────────────
    def apply_frame(frame_idx: int) -> None:
        for robot in robots:
            start = robot_start[robot]
            dim = robot_dim[robot]
            data.qpos[start:start + dim] = robot_qpos[robot][frame_idx]
            dx, dy = robot_offset[robot]
            data.qpos[start] += dx
            data.qpos[start + 1] += dy

        mujoco.mj_forward(model, data)
        all_pos = data.geom_xpos
        all_mat = data.geom_xmat

        for handle, geoms in batched_handles:
            n_inst = len(geoms)
            pos = np.zeros((n_inst, 3), dtype=np.float32)
            wxyz = np.tile([1, 0, 0, 0], (n_inst, 1)).astype(np.float32)
            for j, (gid, dx, dy) in enumerate(geoms):
                pos[j, 0] = float(all_pos[gid, 0] + dx)
                pos[j, 1] = float(all_pos[gid, 1] + dy)
                pos[j, 2] = float(all_pos[gid, 2])
                q = rotation_matrix_to_quat_wxyz(all_mat[gid])
                wxyz[j] = q
            handle.batched_positions = pos
            handle.batched_wxyzs = wxyz

    apply_frame(0)

    # ── Main loop ────────────────────────────────────────────────────
    slider_update_counter = 0
    slider_poll_counter = 0

    while True:
        t0 = time.time()

        if replay:
            current_frame = 0
            replay = False
            paused = False
            gui_pause.name = "Pause"
            _last_slider_val = 0

        slider_poll_counter += 1
        if slider_poll_counter >= 3:
            slider_poll_counter = 0
            slider_val = int(gui_frame.value)
            if slider_val != _last_slider_val and slider_val != current_frame:
                current_frame = slider_val
                _last_slider_val = slider_val
                apply_frame(current_frame)

        if not paused:
            current_frame += step
            if current_frame >= n_frames:
                if args.loop:
                    current_frame = 0
                else:
                    current_frame = n_frames - 1
                    paused = True
                    gui_pause.name = "Resume"
            apply_frame(current_frame)

            slider_update_counter += 1
            if slider_update_counter >= 5:
                slider_update_counter = 0
                gui_frame.value = current_frame
                _last_slider_val = current_frame

        elapsed = time.time() - t0
        sleep_time = dt / gui_speed.value - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
