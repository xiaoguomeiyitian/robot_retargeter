#!/usr/bin/env python3
"""Export retargeted robot motion CSV to NPZ format for RL training.

Produces NPZ files compatible with unitree_rl_mjlab's MotionLoader:
  - joint_pos:    (T, num_joints)  joint positions in radians
  - joint_vel:    (T, num_joints)  joint velocities
  - body_pos_w:   (T, num_bodies, 3)  body world positions
  - body_quat_w:  (T, num_bodies, 4)  body world quaternions (wxyz)
  - body_lin_vel_w: (T, num_bodies, 3)  body linear velocities
  - body_ang_vel_w: (T, num_bodies, 3)  body angular velocities
  - fps: [output_fps]  frame rate

Usage:
    # Export with default settings (30fps input → 50fps output)
    python scripts/export_npz.py \
        --csv output_data/robot_motion/Form_1_stageii_g1.csv \
        --robot g1 \
        --output output_data/npz/Form_1_stageii_g1.npz

    # Custom frame rates
    python scripts/export_npz.py \
        --csv output_data/robot_motion/dance1_subject2_from_g1_h2.csv \
        --robot h2 \
        --input-fps 30 \
        --output-fps 50 \
        --output output_data/npz/dance1_h2.npz

    # Batch export all CSVs for a robot
    python scripts/export_npz.py \
        --csv-dir output_data/robot_motion \
        --robot g1 \
        --output-dir output_data/npz/g1 \
        --pattern "Form_1_stageii_g1"
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config" / "robot"

# ── Robot-specific configurations ──────────────────────────────────────────
# Joint names in the order expected by unitree_rl_mjlab's MotionLoader.
# These match the order in the robot's MJCF free-joint + joints.

ROBOT_CONFIGS = {
    "g1": {
        "joint_names": [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        "num_joints": 29,
        "body_names": [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
    },
    "g1_23dof": {
        "joint_names": [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ],
        "num_joints": 23,
        "body_names": [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export retargeted motion CSV to NPZ for RL training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Input CSV file path (qpos format: xyz + quat_xyzw + joints).",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=None,
        help="Directory of CSV files (batch mode).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for batch mode (default: *.csv).",
    )
    parser.add_argument(
        "--robot",
        type=str,
        required=True,
        choices=list(ROBOT_CONFIGS.keys()),
        help="Robot name (determines joint/body names).",
    )
    parser.add_argument(
        "--input-fps",
        type=float,
        default=30.0,
        help="Input CSV frame rate (default: 30).",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=50.0,
        help="Output NPZ frame rate (default: 50).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output NPZ path (single-file mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (batch mode).",
    )
    return parser.parse_args()


def load_robot_xml(robot_name: str) -> str:
    """Resolve robot MJCF path from YAML config."""
    config_path = CONFIG_DIR / f"{robot_name}.yaml"
    if not config_path.exists():
        # Try g1 for g1_23dof
        if robot_name == "g1_23dof":
            config_path = CONFIG_DIR / "g1.yaml"
        else:
            raise FileNotFoundError(f"Robot config not found: {config_path}")
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    xml_path = config.get("robot_xml_path", "")
    if not xml_path:
        raise KeyError(f"robot_xml_path missing in {config_path}")
    if not os.path.isabs(xml_path):
        xml_path = str(PROJECT_ROOT / xml_path)
    return xml_path


def resample_motion(
    base_pos: np.ndarray,
    base_rot: np.ndarray,
    joint_pos: np.ndarray,
    input_fps: float,
    output_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample motion from input_fps to output_fps.

    Uses linear interpolation for positions and SLERP for rotations.
    """
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    input_frames = base_pos.shape[0]
    duration = (input_frames - 1) * input_dt

    output_times = np.arange(0, duration, output_dt)
    output_frames = len(output_times)

    # Compute blend factors
    phase = output_times / duration
    idx0 = np.minimum((phase * (input_frames - 1)).astype(int), input_frames - 2)
    idx1 = idx0 + 1
    blend = phase * (input_frames - 1) - idx0

    # Lerp positions
    resampled_pos = base_pos[idx0] * (1 - blend[:, None]) + base_pos[idx1] * blend[:, None]
    resampled_joints = joint_pos[idx0] * (1 - blend[:, None]) + joint_pos[idx1] * blend[:, None]

    # SLERP rotations using scipy Slerp
    key_times = np.arange(input_frames, dtype=np.float64)
    # Convert wxyz → xyzw for scipy
    rot_xyzw = base_rot[:, [1, 2, 3, 0]]
    slerp = Slerp(key_times, Rotation.from_quat(rot_xyzw))
    output_frame_indices = phase * (input_frames - 1)
    interp_rot = slerp(output_frame_indices)
    # Convert back: scipy xyzw → wxyz
    rot_scipy = interp_rot.as_quat()  # (T, 4) xyzw
    resampled_rot = rot_scipy[:, [3, 0, 1, 2]]

    return resampled_pos, resampled_rot, resampled_joints


def compute_velocities(
    base_pos: np.ndarray,
    base_rot: np.ndarray,
    joint_pos: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute velocities via finite differences."""
    base_lin_vel = np.gradient(base_pos, dt, axis=0)
    joint_vel = np.gradient(joint_pos, dt, axis=0)

    # Angular velocity from quaternion derivative
    base_ang_vel = np.zeros((base_rot.shape[0], 3), dtype=np.float64)
    for i in range(1, base_rot.shape[0] - 1):
        q_prev = base_rot[i - 1]
        q_next = base_rot[i + 1]
        # Relative rotation: q_next * conj(q_prev)
        w1, x1, y1, z1 = q_prev
        w2, x2, y2, z2 = q_next
        # conj(q_prev)
        q_conj = np.array([w1, -x1, -y1, -z1])
        # q_rel = q_next * q_conj (quaternion multiplication)
        w_rel = w2*q_conj[0] - x2*q_conj[1] - y2*q_conj[2] - z2*q_conj[3]
        x_rel = w2*q_conj[1] + x2*q_conj[0] + y2*q_conj[3] - z2*q_conj[2]
        y_rel = w2*q_conj[2] - x2*q_conj[3] + y2*q_conj[0] + z2*q_conj[1]
        z_rel = w2*q_conj[3] + x2*q_conj[2] - y2*q_conj[1] + z2*q_conj[0]
        # axis-angle from quaternion
        angle = 2 * np.arctan2(np.sqrt(x_rel**2 + y_rel**2 + z_rel**2), w_rel)
        if abs(angle) < 1e-8:
            base_ang_vel[i] = 0.0
        else:
            axis = np.array([x_rel, y_rel, z_rel])
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-8:
                axis /= axis_norm
            base_ang_vel[i] = axis * (angle / (2 * dt))

    # Boundary: copy neighbor
    base_ang_vel[0] = base_ang_vel[1]
    base_ang_vel[-1] = base_ang_vel[-2]

    return base_lin_vel, base_ang_vel, joint_vel


def export_csv_to_npz(
    csv_path: str,
    output_path: str,
    robot_name: str,
    input_fps: float = 30.0,
    output_fps: float = 50.0,
) -> None:
    """Convert a single CSV to NPZ."""
    print(f"Loading CSV: {csv_path}")
    motion = np.loadtxt(csv_path, delimiter=",")
    if motion.ndim == 1:
        motion = motion[None, :]

    input_frames = motion.shape[0]
    print(f"  Input: {input_frames} frames @ {input_fps} fps")

    # Parse CSV: xyz(3) + quat_xyzw(4) + joints(N)
    base_pos = motion[:, :3].copy()
    base_rot_xyzw = motion[:, 3:7].copy()
    base_rot_wxyz = base_rot_xyzw[:, [3, 0, 1, 2]]  # xyzw → wxyz
    joint_pos_raw = motion[:, 7:]

    # Resample to output fps
    print(f"  Resampling: {input_fps} fps → {output_fps} fps")
    resampled_pos, resampled_rot, resampled_joints = resample_motion(
        base_pos, base_rot_xyzw, joint_pos_raw, input_fps, output_fps
    )
    output_frames = resampled_pos.shape[0]
    output_dt = 1.0 / output_fps
    print(f"  Output: {output_frames} frames @ {output_fps} fps")

    # Load robot model for forward kinematics
    xml_path = load_robot_xml(robot_name)
    print(f"  Robot model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    robot_cfg = ROBOT_CONFIGS[robot_name]
    num_joints = robot_cfg["num_joints"]
    body_names = robot_cfg["body_names"]

    # Get body indices
    body_indices = []
    for name in body_names:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx < 0:
            print(f"  [WARN] Body '{name}' not found in model, using root")
            idx = 0
        body_indices.append(idx)
    body_indices = np.array(body_indices)

    # Get joint indices (map from qpos)
    joint_qpos_start = []
    for name in robot_cfg["joint_names"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found in model {robot_name}")
        # Get qpos address
        qpos_adr = model.jnt_qposadr[jid]
        joint_qpos_start.append(qpos_adr)

    # Allocate output arrays
    # NOTE: MotionLoader 使用 body_indexes 对 body_pos_w 做 fancy indexing:
    #   self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    # 因此 body_pos_w 的维度 1 必须是模型 body 总数（而非 tracking 任务的
    # body_names 数量），否则 body_indexes 最大值超过维度 1 会越界。
    num_model_bodies = model.nbody - 1  # exclude world
    joint_pos_all = np.zeros((output_frames, num_joints), dtype=np.float32)
    body_pos_all = np.zeros((output_frames, num_model_bodies, 3), dtype=np.float32)
    body_quat_all = np.zeros((output_frames, num_model_bodies, 4), dtype=np.float32)

    print(f"  Running forward kinematics...")
    for t in range(output_frames):
        # Set root state
        data.qpos[:3] = resampled_pos[t]
        data.qpos[3:7] = resampled_rot[t]  # wxyz
        # Set joint positions
        for j, qpos_adr in enumerate(joint_qpos_start):
            if j < resampled_joints.shape[1]:
                data.qpos[qpos_adr] = resampled_joints[t, j]

        # Zero velocities for pure kinematic forward
        data.qvel[:] = 0.0

        # Forward kinematics
        mujoco.mj_forward(model, data)

        # Extract joint positions
        for j, qpos_adr in enumerate(joint_qpos_start):
            joint_pos_all[t, j] = data.qpos[qpos_adr]

        # Extract body states (store all model bodies; MotionLoader will
        # select the subset it needs via body_indexes)
        for b in range(num_model_bodies):
            body_pos_all[t, b] = data.xpos[b + 1]  # skip world (index 0)
            body_quat_all[t, b] = data.xquat[b + 1]  # wxyz

    # Compute velocities
    print(f"  Computing velocities...")
    base_lin_vel, base_ang_vel, joint_vel = compute_velocities(
        resampled_pos, resampled_rot, resampled_joints, output_dt
    )

    # Compute body velocities via finite differences
    body_lin_vel = np.zeros((output_frames, num_model_bodies, 3), dtype=np.float32)
    body_ang_vel = np.zeros((output_frames, num_model_bodies, 3), dtype=np.float32)

    for b in range(num_model_bodies):
        body_lin_vel[:, b] = np.gradient(body_pos_all[:, b], output_dt, axis=0)

    for b in range(num_model_bodies):
        for t in range(1, output_frames - 1):
            q_prev = body_quat_all[t - 1, b]
            q_next = body_quat_all[t + 1, b]
            w1, x1, y1, z1 = q_prev
            w2, x2, y2, z2 = q_next
            q_conj = np.array([w1, -x1, -y1, -z1])
            w_rel = w2*q_conj[0] - x2*q_conj[1] - y2*q_conj[2] - z2*q_conj[3]
            x_rel = w2*q_conj[1] + x2*q_conj[0] + y2*q_conj[3] - z2*q_conj[2]
            y_rel = w2*q_conj[2] - x2*q_conj[3] + y2*q_conj[0] + z2*q_conj[1]
            z_rel = w2*q_conj[3] + x2*q_conj[2] - y2*q_conj[1] + z2*q_conj[0]
            angle = 2 * np.arctan2(np.sqrt(x_rel**2 + y_rel**2 + z_rel**2), abs(w_rel))
            if abs(angle) < 1e-8:
                body_ang_vel[t, b] = 0.0
            else:
                axis = np.array([x_rel, y_rel, z_rel])
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-8:
                    axis /= axis_norm
                body_ang_vel[t, b] = axis * (angle / (2 * output_dt))
        # Boundary
        if output_frames > 1:
            body_ang_vel[0, b] = body_ang_vel[1, b]
            body_ang_vel[-1, b] = body_ang_vel[-2, b]

    # Save NPZ
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(
        output_path,
        fps=np.array([output_fps], dtype=np.float32),
        joint_pos=joint_pos_all.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos_all.astype(np.float32),
        body_quat_w=body_quat_all.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
    )
    print(f"  Saved: {output_path}")
    print(f"  Shapes: joint_pos={joint_pos_all.shape}, body_pos_w={body_pos_all.shape}")


def main() -> None:
    args = parse_args()

    if args.csv:
        # Single file mode
        output_path = args.output
        if output_path is None:
            csv_stem = Path(args.csv).stem
            output_path = str(PROJECT_ROOT / "output_data" / "npz" / f"{csv_stem}.npz")
        export_csv_to_npz(
            csv_path=args.csv,
            output_path=output_path,
            robot_name=args.robot,
            input_fps=args.input_fps,
            output_fps=args.output_fps,
        )
    elif args.csv_dir:
        # Batch mode
        output_dir = args.output_dir or str(PROJECT_ROOT / "output_data" / "npz" / args.robot)
        csv_files = sorted(Path(args.csv_dir).glob(args.pattern))
        if not csv_files:
            print(f"No CSV files matching '{args.pattern}' in {args.csv_dir}")
            sys.exit(1)
        print(f"Found {len(csv_files)} CSV files to convert")
        for csv_file in csv_files:
            output_name = csv_file.stem + ".npz"
            output_path = os.path.join(output_dir, output_name)
            export_csv_to_npz(
                csv_path=str(csv_file),
                output_path=output_path,
                robot_name=args.robot,
                input_fps=args.input_fps,
                output_fps=args.output_fps,
            )
    else:
        print("Error: specify --csv or --csv-dir")
        sys.exit(1)

    print("\n✅ Export complete!")


if __name__ == "__main__":
    main()
