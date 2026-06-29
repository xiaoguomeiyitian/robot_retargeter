"""Motion trajectory editor for robot animations.

Provides a viser-based GUI for:
- Scrubbing through frames with a slider
- Viewing per-joint angles at the current frame
- Adjusting joint angles and seeing real-time preview
- Exporting edited motion to CSV

Usage:
    python scripts/motion_editor.py \
        --motion dance2 \
        --robot g1 \
        --motion_dir dataset/lafan1_g1 \
        --port 20007
"""

import argparse
import os
import sys
import time

import mujoco
import numpy as np
import viser

# Add scripts directory to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from multi_robot_visualize_viser import (
    build_combined_spec,
    get_qpos_start,
    load_motion,
    resolve_robot,
    rotation_matrix_to_quat_wxyz,
    ROBOT_CONFIG_DIR,
    PROJECT_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot motion trajectory editor")
    parser.add_argument("--motion", default="dance2")
    parser.add_argument("--robot", default="subject1")
    parser.add_argument("--motion_dir", default="output_data/robot_motion")
    parser.add_argument("--port", type=int, default=20007)
    parser.add_argument("--output", type=str, default=None,
                        help="Export edited motion to this CSV path")
    args = parser.parse_args()

    robot = args.robot
    csv_path = os.path.join(args.motion_dir, f"{args.motion}_{robot}.csv")
    if not os.path.isfile(csv_path):
        raise FileNot找到Error(f"Motion file not found: {csv_path}")

    # Load motion
    qpos = load_motion(csv_path)
    n_frames, n_joints = qpos.shape
    print(f"Loaded {n_frames} frames, {n_joints} joints")

    # Build model (single robot at origin)
    spec = build_combined_spec([robot])
    model = spec.compile()
    data = mujoco.MjData(model)

    start_idx = get_qpos_start(model, f"{robot}_floating_base_joint")
    dim = qpos.shape[1]

    # Create viser server
    server = viser.ViserServer(port=args.port, verbose=False)
    server.scene.set_up_direction("+z")
    server.scene.configure_default_lights()

    # Ground
    import trimesh
    ground_mesh = trimesh.creation.box(extents=[10, 10, 0.02])
    ground_mesh.apply_translation([0, 0, -0.01])
    server.scene.add_mesh_trimesh("ground", ground_mesh, position=(0, 0, -0.01))

    # Build mesh groups for batched rendering
    from collections import defaultdict
    body_id_to_robot = {}
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name and body_name.startswith(f"{robot}_"):
            body_id_to_robot[body_id] = robot

    mesh_groups = defaultdict(list)
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name == "ground":
            continue
        geom = model.geom(i)
        if int(geom.type[0]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        body_id = int(geom.bodyid[0])
        if body_id not in body_id_to_robot:
            continue
        mesh_id = int(geom.dataid[0])
        mesh_groups[mesh_id].append(i)

    batched_handles = []
    for mesh_id, geoms in mesh_groups.items():
        vert_adr = model.mesh_vertadr[mesh_id]
        vert_num = model.mesh_vertnum[mesh_id]
        face_adr = model.mesh_faceadr[mesh_id]
        face_num = model.mesh_facenum[mesh_id]
        if vert_num <= 0 or face_num <= 0:
            continue
        vertices = model.mesh_vert[vert_adr:vert_adr + vert_num].copy()
        faces = model.mesh_face[face_adr:face_adr + face_num].copy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        n_inst = len(geoms)
        positions = np.zeros((n_inst, 3))
        wxyzs = np.tile([1, 0, 0, 0], (n_inst, 1)).astype(np.float64)
        handle = server.scene.add_batched_meshes_trimesh(
            f"robots/batch_{mesh_id}", mesh,
            batched_wxyzs=wxyzs, batched_positions=positions,
        )
        batched_handles.append((handle, geoms))

    # GUI Controls
    with server.gui.add_folder("Playback"):
        gui_play = server.gui.add_button("Pause")
        gui_frame = server.gui.add_slider("Frame", min=0, max=n_frames - 1, step=1, initial_value=0)
        gui_speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)

    # Joint angle display (first 7 DOFs: pos3 + quat4)
    joint_names = ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"]
    joint_sliders = []
    with server.gui.add_folder("Joint Angles (Base)"):
        for i in range(min(7, n_joints)):
            name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
            slider = server.gui.add_slider(
                name,
                min=-3.0 if i < 3 else -1.0,
                max=3.0 if i < 3 else 1.0,
                step=0.01,
                initial_value=float(qpos[0, i]),
            )
            joint_sliders.append(slider)

    # Export button
    with server.gui.add_folder("Export"):
        gui_export = server.gui.add_button("Export to CSV")
        export_path = server.gui.add_text("Output path", args.output or f"edited_{args.motion}_{robot}.csv")

    paused = False
    current_frame = 0
    edited_qpos = qpos.copy()  # Working copy for edits

    @gui_play.on_click
    def _(_):
        nonlocal paused
        paused = not paused
        gui_play.name = "Play" if paused else "Pause"

    def apply_frame(frame_idx: int):
        """Update robot pose and viser meshes."""
        data.qpos[start_idx:start_idx + dim] = edited_qpos[frame_idx]
        mujoco.mj_forward(model, data)

        all_pos = data.geom_xpos
        all_mat = data.geom_xmat
        for handle, geoms in batched_handles:
            n_inst = len(geoms)
            pos = np.zeros((n_inst, 3))
            wxyz = np.tile([1, 0, 0, 0], (n_inst, 1)).astype(np.float64)
            for j, gid in enumerate(geoms):
                pos[j] = all_pos[gid]
                wxyz[j] = rotation_matrix_to_quat_wxyz(all_mat[gid])
            handle.batched_positions = pos
            handle.batched_wxyzs = wxyz

        # Update joint sliders to reflect current frame
        for i, slider in enumerate(joint_sliders):
            slider.value = float(edited_qpos[frame_idx, i])

    # Joint slider callbacks for editing
    joint_callbacks = []
    for i, slider in enumerate(joint_sliders):
        def _make_callback(idx):
            def _(_):
                edited_qpos[current_frame, idx] = slider.value
                apply_frame(current_frame)
            return _
        slider.on_update(_make_callback(i))
        joint_callbacks.append(_)

    @gui_export.on_click
    def _(_):
        path = export_path.value
        # Save with header comment
        np.savetxt(path, edited_qpos, delimiter=",")
        print(f"Exported {n_frames} frames to {path}")

    apply_frame(0)

    # Main loop
    dt = 1.0 / 30.0
    while True:
        t0 = time.time()

        # Check frame slider
        slider_val = int(gui_frame.value)
        if slider_val != current_frame:
            current_frame = slider_val
            apply_frame(current_frame)

        if not paused:
            current_frame += 1
            if current_frame >= n_frames:
                current_frame = 0
            apply_frame(current_frame)
            gui_frame.value = current_frame

        elapsed = time.time() - t0
        sleep_time = dt / gui_speed.value - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
