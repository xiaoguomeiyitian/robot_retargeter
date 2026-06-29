"""SMPL-X fitting module: fit SMPL-X body model to 3D keypoints.

This module provides:
  - FitSMPLX: Fit SMPL-X body model to 3D keypoints using optimization
  - Output SMPL-X parameters compatible with robot_retargeter's smpl_replay.py

Usage:
    from scripts.video_to_robot import FitSMPLX

    fitter = FitSMPLX(smplx_model_dir="asset/smplx")
    smpl_params = fitter.fit(keypoints_3d, fps=30)
    fitter.save_npz(smpl_params, "output_data/motion.npz")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import smplx


# ── SMPL-X joint names (22 body joints) ────────────────────────────────────
SMPLX_BODY_JOINTS = [
    "hips",           # 0
    "left_up_leg",    # 1
    "right_up_leg",   # 2
    "spine1",         # 3
    "left_leg",       # 4
    "right_leg",      # 5
    "spine2",         # 6
    "left_foot",      # 7
    "right_foot",     # 8
    "chest",          # 9
    "left_toe",       # 10
    "right_toe",      # 11
    "neck",           # 12
    "head",           # 13
    "left_shoulder",  # 14
    "right_shoulder", # 15
    "left_arm",       # 16
    "right_arm",      # 17
    "left_fore_arm",  # 18
    "right_fore_arm", # 19
    "left_hand",      # 20
    "right_hand",     # 21
]

# Mapping from COCO 17 keypoints to SMPL-X body joints
COCO_TO_SMPLX = {
    0: 13,   # nose → head
    5: 14,   # left_shoulder → left_shoulder
    6: 15,   # right_shoulder → right_shoulder
    7: 16,   # left_elbow → left_arm
    8: 17,   # right_elbow → right_arm
    9: 18,   # left_wrist → left_fore_arm
    10: 19,  # right_wrist → right_fore_arm
    11: 1,   # left_hip → left_up_leg
    12: 2,   # right_hip → right_up_leg
    13: 4,   # left_knee → left_leg
    14: 5,   # right_knee → right_leg
    15: 7,   # left_ankle → left_foot
    16: 8,   # right_ankle → right_foot
}

# Derived body centers (matching robot_retargeter's smpl_replay.py)
DERIVED_BODY_CENTERS = {
    "hips_mean": ("left_up_leg", "right_up_leg"),
    "shoulder_mean": ("left_arm", "right_arm"),
}


class FitSMPLX:
    """Fit SMPL-X body model to 3D keypoints.

    This class provides optimization-based fitting of SMPL-X parameters
    to match observed 3D keypoints. The output is compatible with
    robot_retargeter's smpl_replay.py pipeline.

    Attributes:
        smplx_model_dir: Directory containing SMPL-X model files.
        gender: SMPL-X model gender ('neutral', 'male', 'female').
        use_pca: Whether to use PCA for body pose.
        num_pca_comps: Number of PCA components.
    """

    def __init__(
        self,
        smplx_model_dir: str = "asset/smplx",
        gender: str = "neutral",
        use_pca: bool = True,
        num_pca_comps: int = 12,
    ):
        """Initialize the FitSMPLX.

        Args:
            smplx_model_dir: Directory containing SMPL-X model files.
            gender: SMPL-X model gender.
            use_pca: Whether to use PCA for body pose.
            num_pca_comps: Number of PCA components.
        """
        self.smplx_model_dir = smplx_model_dir
        self.gender = gender
        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps

        self._init_model()

    def _init_model(self) -> None:
        """Initialize the SMPL-X model."""
        model_path = Path(self.smplx_model_dir)
        if not model_path.exists():
            print(f"[警告] SMPL-X 模型目录未找到: {model_path}")
            print("[警告] SMPL-X 拟合将不可用")
            self.model = None
            return

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # smplx.create() appends model_type internally, so we need
            # the parent of the directory containing model files.
            expected_file = model_path / "SMPLX_{}.npz".format(self.gender.upper())
            if expected_file.exists():
                smplx_parent = model_path.parent
            else:
                # Try model_path / model_type subdirectory
                alt = model_path / "smplx"
                if (alt / "SMPLX_{}.npz".format(self.gender.upper())).exists():
                    smplx_parent = model_path
                else:
                    smplx_parent = model_path
            self.model = smplx.create(
                model_path=str(smplx_parent),
                model_type="smplx",
                gender=self.gender,
                use_pca=self.use_pca,
                num_pca_comps=self.num_pca_comps,
                device=device,
            )
            self.device = device
            print(f"[信息] SMPL-X 模型已加载 (性别={self.gender}, 设备={device})")
        except Exception as e:
            print(f"[警告] SMPL-X 模型加载失败: {e}")
            self.model = None

    def fit(
        self,
        keypoints_3d: np.ndarray,
        fps: float = 30.0,
        num_iterations: int = 100,
        learning_rate: float = 1e-3,
    ) -> dict:
        """Fit SMPL-X model to 3D keypoints.

        Args:
            keypoints_3d: (N, 17, 3) array of 3D COCO keypoints.
            fps: Frame rate of the motion.
            num_iterations: Number of optimization iterations per frame.
            learning_rate: Learning rate for optimization.

        Returns:
            Dictionary with SMPL-X parameters:
                - body_pose: (N, 21, 3) body joint rotations (axis-angle)
                - global_orient: (N, 3) global orientation
                - transl: (N, 3) global translation
                - betas: (N, 10) body shape parameters
                - fps: frame rate
        """
        if self.model is None:
            print("[警告] SMPL-X 模型不可用，使用简单 IK 拟合")
            return self._fit_simple_ik(keypoints_3d, fps)

        import torch

        n_frames = keypoints_3d.shape[0]
        print(f"[信息] 正在拟合 SMPL-X，共 {n_frames} 帧...")

        # Initialize parameters
        body_pose = torch.zeros(n_frames, 21, 3, device=self.device)
        global_orient = torch.zeros(n_frames, 3, device=self.device)
        transl = torch.zeros(n_frames, 3, device=self.device)
        betas = torch.zeros(n_frames, 10, device=self.device)

        # Fit each frame
        for frame_idx in range(n_frames):
            kp = keypoints_3d[frame_idx]
            if np.isnan(kp).all():
                continue

            # Simple optimization: minimize distance between SMPL-X joints and target
            bp = body_pose[frame_idx].clone().requires_grad_(True)
            go = global_orient[frame_idx].clone().requires_grad_(True)
            tr = transl[frame_idx].clone().requires_grad_(True)
            be = betas[frame_idx].clone().requires_grad_(True)

            optimizer = torch.optim.Adam([bp, go, tr, be], lr=learning_rate)

            target_joints = torch.tensor(kp, dtype=torch.float32, device=self.device)

            for _ in range(num_iterations):
                optimizer.zero_grad()
                output = self.model(
                    body_pose=bp.unsqueeze(0),
                    global_orient=go.unsqueeze(0),
                    transl=tr.unsqueeze(0),
                    betas=be.unsqueeze(0),
                )
                joints = output.joints[0]  # (J, 3)

                # Compute loss on matched joints
                loss = torch.tensor(0.0, device=self.device)
                count = 0
                for coco_idx, smpl_idx in COCO_TO_SMPLX.items():
                    if not np.isnan(kp[coco_idx]).any():
                        loss += ((joints[smpl_idx] - target_joints[coco_idx]) ** 2).sum()
                        count += 1

                if count > 0:
                    loss = loss / count
                    loss.backward()
                    optimizer.step()

            body_pose[frame_idx] = bp.detach()
            global_orient[frame_idx] = go.detach()
            transl[frame_idx] = tr.detach()
            betas[frame_idx] = be.detach()

            if (frame_idx + 1) % 100 == 0:
                print(f"[信息] 已拟合 {frame_idx + 1}/{n_frames} 帧")

        return {
            "body_pose": body_pose.cpu().numpy(),
            "global_orient": global_orient.cpu().numpy(),
            "transl": transl.cpu().numpy(),
            "betas": betas.cpu().numpy(),
            "fps": fps,
        }

    def _fit_simple_ik(
        self,
        keypoints_3d: np.ndarray,
        fps: float = 30.0,
    ) -> dict:
        """Simple IK-based fitting (fallback when SMPL-X model is unavailable).

        This method directly uses the 3D keypoints to compute joint angles
        using analytical IK, without the SMPL-X body model.

        Args:
            keypoints_3d: (N, 17, 3) array of 3D COCO keypoints.
            fps: Frame rate.

        Returns:
            Dictionary with simplified motion parameters.
        """
        n_frames = keypoints_3d.shape[0]
        print(f"[信息] 简单 IK 拟合，共 {n_frames} 帧")

        # Compute joint angles from keypoint positions
        # This is a simplified approach that directly uses keypoint positions
        # as the basis for retargeting

        # Compute root position (hip center)
        root_pos = np.zeros((n_frames, 3), dtype=np.float32)
        root_rot = np.zeros((n_frames, 3), dtype=np.float32)  # axis-angle

        for i in range(n_frames):
            kp = keypoints_3d[i]
            # Root position = average of hips
            if not np.isnan(kp[11]).any() and not np.isnan(kp[12]).any():
                root_pos[i] = (kp[11] + kp[12]) / 2

            # Root orientation from shoulder-hip line
            if (not np.isnan(kp[5]).any() and not np.isnan(kp[6]).any()
                    and not np.isnan(kp[11]).any() and not np.isnan(kp[12]).any()):
                shoulder_center = (kp[5] + kp[6]) / 2
                hip_center = (kp[11] + kp[12]) / 2
                forward = shoulder_center - hip_center
                forward_norm = np.linalg.norm(forward)
                if forward_norm > 1e-6:
                    forward = forward / forward_norm
                    # Convert to axis-angle (simplified: yaw only)
                    yaw = np.arctan2(forward[0], forward[2])
                    root_rot[i] = np.array([0, yaw, 0])

        # Compute relative joint positions (relative to root)
        joint_pos_relative = np.zeros((n_frames, 17, 3), dtype=np.float32)
        for i in range(n_frames):
            joint_pos_relative[i] = keypoints_3d[i] - root_pos[i, np.newaxis, :]

        return {
            "root_pos": root_pos,
            "root_rot": root_rot,
            "joint_pos_relative": joint_pos_relative,
            "keypoints_3d": keypoints_3d,
            "fps": fps,
        }

    def save_npz(
        self,
        params: dict,
        output_path: Union[str, Path],
    ) -> Path:
        """Save SMPL-X parameters to NPZ file.

        The output format is compatible with robot_retargeter's smpl_replay.py.

        Args:
            params: Dictionary of SMPL-X parameters.
            output_path: Path to save the NPZ file.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to AMASS-compatible format
        save_dict = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                save_dict[key] = value
            else:
                save_dict[key] = np.array(value)

        np.savez(output_path, **save_dict)
        print(f"[信息] 已保存 SMPL-X 参数: {output_path}")
        return output_path
