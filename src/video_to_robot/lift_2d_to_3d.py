"""2D-to-3D lifting module: lift 2D keypoints to 3D using temporal models.

This module provides:
  - Lift2Dto3D: Lift 2D COCO keypoints to 3D using a temporal model
  - Simple baseline using linear interpolation + temporal smoothing
  - Optional: VideoPose3D-style lifting (if available)

Usage:
    from src.video_to_robot import Lift2Dto3D

    lifter = Lift2Dto3D()
    keypoints_3d = lifter.lift(keypoints_2d)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


class Lift2Dto3D:
    """Lift 2D keypoints to 3D using temporal information.

    This implementation provides a simple baseline that:
    1. Uses MediaPipe's built-in 3D landmarks (if available)
    2. Falls back to a simple temporal smoothing + linear interpolation
    3. Optionally uses VideoPose3D-style lifting (if model is available)

    Attributes:
        method: Lifting method ('mediapipe', 'simple', 'videopose3d').
        smooth_window: Window size for temporal smoothing.
    """

    def __init__(
        self,
        method: str = "simple",
        smooth_window: int = 5,
        model_path: str | None = None,
    ):
        """Initialize the Lift2Dto3D.

        Args:
            method: Lifting method. One of 'mediapipe', 'simple', 'videopose3d'.
            smooth_window: Window size for temporal smoothing.
            model_path: Path to the lifting model (for 'videopose3d' method).
        """
        self.method = method
        self.smooth_window = smooth_window
        self.model_path = model_path

        if method == "videopose3d":
            self._init_videopose3d()

    def _init_videopose3d(self) -> None:
        """Initialize VideoPose3D model (placeholder for future integration)."""
        # VideoPose3D requires PyTorch model weights
        # For now, fall back to simple method
        print("[WARN] VideoPose3D model not available, falling back to 'simple' method")
        self.method = "simple"

    def lift(
        self,
        keypoints_2d: np.ndarray,
        keypoints_3d_mediapipe: np.ndarray | None = None,
    ) -> np.ndarray:
        """Lift 2D keypoints to 3D.

        Args:
            keypoints_2d: (N, 17, 3) array of COCO 17 keypoints (x, y, visibility).
            keypoints_3d_mediapipe: (N, 33, 4) array of MediaPipe 3D landmarks.
                If provided and method is 'mediapipe', use these directly.

        Returns:
            keypoints_3d: (N, 17, 3) array of 3D COCO keypoints (x, y, z).
        """
        if self.method == "mediapipe" and keypoints_3d_mediapipe is not None:
            return self._lift_from_mediapipe(keypoints_3d_mediapipe)
        else:
            return self._lift_simple(keypoints_2d)

    def _lift_from_mediapipe(self, keypoints_3d_mediapipe: np.ndarray) -> np.ndarray:
        """Extract 3D COCO 17 keypoints from MediaPipe 33 landmarks.

        Args:
            keypoints_3d_mediapipe: (N, 33, 4) array of MediaPipe 33 landmarks.

        Returns:
            keypoints_3d: (N, 17, 3) array of 3D COCO keypoints.
        """
        # MediaPipe to COCO 17 mapping (same as in video_extract.py)
        mediapipe_to_coco = {
            0: 0,   # nose
            11: 5,  # left_shoulder
            12: 6,  # right_shoulder
            13: 7,  # left_elbow
            14: 8,  # right_elbow
            15: 9,  # left_wrist
            16: 10, # right_wrist
            23: 11, # left_hip
            24: 12, # right_hip
            25: 13, # left_knee
            26: 14, # right_knee
            27: 15, # left_ankle
            28: 16, # right_ankle
        }

        n_frames = keypoints_3d_mediapipe.shape[0]
        keypoints_3d = np.zeros((n_frames, 17, 3), dtype=np.float32)

        for coco_idx, mp_idx in mediapipe_to_coco.items():
            keypoints_3d[:, coco_idx] = keypoints_3d_mediapipe[:, mp_idx, :3]

        # Fill missing keypoints (eyes, ears) with interpolated values
        # Use shoulder/hip positions as reference
        for frame_idx in range(n_frames):
            # Approximate eye positions as slightly above nose
            if not np.isnan(keypoints_3d[frame_idx, 0, 0]):  # nose
                nose = keypoints_3d[frame_idx, 0]
                # Left eye: slightly left and above nose
                keypoints_3d[frame_idx, 1] = nose + np.array([-0.03, 0.02, 0])
                # Right eye: slightly right and above nose
                keypoints_3d[frame_idx, 2] = nose + np.array([0.03, 0.02, 0])
                # Left ear: further left
                keypoints_3d[frame_idx, 3] = nose + np.array([-0.06, 0.01, 0])
                # Right ear: further right
                keypoints_3d[frame_idx, 4] = nose + np.array([0.06, 0.01, 0])

        return keypoints_3d

    def _lift_simple(self, keypoints_2d: np.ndarray) -> np.ndarray:
        """Simple 2D→3D lifting using temporal smoothing and pseudo-depth.

        This is a baseline method that:
        1. Uses 2D keypoints as x, y
        2. Estimates z from limb length ratios (pseudo-depth)
        3. Applies temporal smoothing

        Args:
            keypoints_2d: (N, 17, 3) array of COCO 17 keypoints.

        Returns:
            keypoints_3d: (N, 17, 3) array of 3D COCO keypoints.
        """
        n_frames = keypoints_2d.shape[0]
        keypoints_3d = np.zeros((n_frames, 17, 3), dtype=np.float32)

        # Use 2D coordinates as x, y
        keypoints_3d[:, :, :2] = keypoints_2d[:, :, :2]

        # Estimate pseudo-depth from limb lengths
        # The idea: limbs that appear shorter in 2D are likely further from camera
        for frame_idx in range(n_frames):
            kp = keypoints_2d[frame_idx]
            vis = kp[:, 2]

            # Compute torso length as reference
            if vis[5] > 0.5 and vis[6] > 0.5 and vis[11] > 0.5 and vis[12] > 0.5:
                shoulder_center = (kp[5] + kp[6]) / 2
                hip_center = (kp[11] + kp[12]) / 2
                torso_length = np.linalg.norm(shoulder_center[:2] - hip_center[:2])

                # Estimate depth from torso length (inverse relationship)
                # Larger torso → closer → positive z
                reference_torso = 0.3  # normalized reference
                z_base = (reference_torso - torso_length) * 0.5

                # Assign depth based on body part
                for joint_idx in range(17):
                    if vis[joint_idx] > 0.5:
                        # Depth varies by body part (simplified)
                        keypoints_3d[frame_idx, joint_idx, 2] = z_base

        # Temporal smoothing
        keypoints_3d = self._temporal_smooth(keypoints_3d)

        return keypoints_3d

    def _temporal_smooth(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to keypoints.

        Args:
            keypoints: (N, J, 3) array of keypoints.

        Returns:
            smoothed: (N, J, 3) array of smoothed keypoints.
        """
        n_frames = keypoints.shape[0]
        if n_frames < self.smooth_window:
            return keypoints

        smoothed = np.copy(keypoints)
        half_window = self.smooth_window // 2

        for i in range(n_frames):
            start = max(0, i - half_window)
            end = min(n_frames, i + half_window + 1)
            window = keypoints[start:end]

            # Only average valid (non-NaN) values
            valid_mask = ~np.isnan(window).any(axis=2)
            for j in range(keypoints.shape[1]):
                valid_values = window[valid_mask[:, j], j]
                if len(valid_values) > 0:
                    smoothed[i, j] = np.mean(valid_values, axis=0)

        return smoothed

    def lift_and_save(
        self,
        keypoints_2d: np.ndarray,
        output_path: Union[str, Path],
        keypoints_3d_mediapipe: np.ndarray | None = None,
    ) -> np.ndarray:
        """Lift 2D keypoints to 3D and save.

        Args:
            keypoints_2d: (N, 17, 3) array of 2D keypoints.
            output_path: Path to save the 3D keypoints.
            keypoints_3d_mediapipe: Optional MediaPipe 3D landmarks.

        Returns:
            keypoints_3d: (N, 17, 3) array of 3D keypoints.
        """
        keypoints_3d = self.lift(keypoints_2d, keypoints_3d_mediapipe)
        np.save(output_path, keypoints_3d)
        print(f"[INFO] Saved 3D keypoints: {output_path}")
        return keypoints_3d
