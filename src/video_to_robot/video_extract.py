"""Video extraction module: extract 2D/3D human keypoints from video using MediaPipe.

This module provides:
  - VideoExtractor: Extract 2D and 3D human keypoints from video files
  - Support for single-person videos
  - Outputs COCO-format 2D keypoints and MediaPipe 3D landmarks

Usage:
    from src.video_to_robot import VideoExtractor

    extractor = VideoExtractor()
    keypoints_2d, keypoints_3d, fps = extractor.extract("path/to/video.mp4")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision.core import vision_task_running_mode

# ── Constants ──────────────────────────────────────────────────────────────

# MediaPipe PoseLandmarker landmark indices mapping to COCO 17-keypoint format
# MediaPipe has 33 landmarks; we map the most important ones to COCO format
MEDIAPIPE_TO_COCO_17 = {
    0: 0,   # nose → nose
    2: 1,   # left_eye_inner → left_eye
    5: 2,   # right_eye_inner → right_eye
    7: 3,   # left_ear → left_ear
    8: 4,   # right_ear → right_ear
    11: 5,  # left_shoulder → left_shoulder
    12: 6,  # right_shoulder → right_shoulder
    13: 7,  # left_elbow → left_elbow
    14: 8,  # right_elbow → right_elbow
    15: 9,  # left_wrist → left_wrist
    16: 10, # right_wrist → right_wrist
    23: 11, # left_hip → left_hip
    24: 12, # right_hip → right_hip
    25: 13, # left_knee → left_knee
    26: 14, # right_knee → right_knee
    27: 15, # left_ankle → left_ankle
    28: 16, # right_ankle → right_ankle
}

# COCO 17 keypoint names
COCO_17_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# MediaPipe 33 landmark names
MEDIAPIPE_33_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


class VideoExtractor:
    """Extract human keypoints from video using MediaPipe PoseLandmarker.

    Attributes:
        model_path: Path to the MediaPipe PoseLandmarker model file.
        num_poses: Maximum number of poses to detect.
        min_detection_confidence: Minimum confidence for pose detection.
        min_tracking_confidence: Minimum confidence for pose tracking.
        min_presence_confidence: Minimum confidence for pose presence.
    """

    def __init__(
        self,
        model_path: str = "asset/models/pose_landmarker_heavy.task",
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ):
        """Initialize the VideoExtractor.

        Args:
            model_path: Path to the MediaPipe PoseLandmarker model.
            num_poses: Maximum number of poses to detect.
            min_detection_confidence: Minimum confidence for detection.
            min_tracking_confidence: Minimum confidence for tracking.
            min_presence_confidence: Minimum confidence for presence.
        """
        self.model_path = model_path
        self.num_poses = num_poses
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_presence_confidence = min_presence_confidence

        # Initialize the PoseLandmarker
        self._init_landmarker()

    def _init_landmarker(self) -> None:
        """Initialize the MediaPipe PoseLandmarker."""
        base_options = BaseOptions(model_asset_path=self.model_path)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
            num_poses=self.num_poses,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def extract(
        self,
        video_path: Union[str, Path],
        max_frames: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Extract keypoints from a video file.

        Args:
            video_path: Path to the input video file.
            max_frames: Maximum number of frames to process. None for all frames.

        Returns:
            Tuple of (keypoints_2d, keypoints_3d, fps):
                - keypoints_2d: (N, 17, 3) array of COCO 17 keypoints (x, y, visibility)
                - keypoints_3d: (N, 33, 4) array of MediaPipe 33 landmarks (x, y, z, visibility)
                - fps: Video frame rate
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[INFO] Video: {video_path.name}")
        print(f"[INFO] Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")

        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        keypoints_2d_list = []
        keypoints_3d_list = []
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames is not None and frame_idx >= max_frames:
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                # Detect pose
                result = self.landmarker.detect_for_video(mp_image, frame_idx)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]  # First person

                    # Extract 33 MediaPipe landmarks (3D)
                    kp_3d = np.array(
                        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
                        dtype=np.float32,
                    )

                    # Extract COCO 17 keypoints (2D)
                    kp_2d = np.zeros((17, 3), dtype=np.float32)
                    for coco_idx, mp_idx in MEDIAPIPE_TO_COCO_17.items():
                        lm = landmarks[mp_idx]
                        kp_2d[coco_idx] = [lm.x, lm.y, lm.visibility]

                    keypoints_2d_list.append(kp_2d)
                    keypoints_3d_list.append(kp_3d)
                else:
                    # No person detected → append NaN
                    kp_2d = np.full((17, 3), np.nan, dtype=np.float32)
                    kp_3d = np.full((33, 4), np.nan, dtype=np.float32)
                    keypoints_2d_list.append(kp_2d)
                    keypoints_3d_list.append(kp_3d)

                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"[INFO] Processed {frame_idx}/{total_frames} frames")

        finally:
            cap.release()

        if not keypoints_2d_list:
            raise RuntimeError(f"No person detected in video: {video_path}")

        keypoints_2d = np.stack(keypoints_2d_list, axis=0)  # (N, 17, 3)
        keypoints_3d = np.stack(keypoints_3d_list, axis=0)  # (N, 33, 4)

        print(f"[INFO] Extracted {len(keypoints_2d_list)} frames")
        print(f"[INFO] 2D keypoints shape: {keypoints_2d.shape}")
        print(f"[INFO] 3D keypoints shape: {keypoints_3d.shape}")

        return keypoints_2d, keypoints_3d, fps

    def extract_and_save(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        max_frames: int | None = None,
    ) -> tuple[Path, Path, float]:
        """Extract keypoints and save to disk.

        Args:
            video_path: Path to the input video.
            output_dir: Directory to save the extracted keypoints.
            max_frames: Maximum number of frames to process.

        Returns:
            Tuple of (kp2d_path, kp3d_path, fps).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        keypoints_2d, keypoints_3d, fps = self.extract(video_path, max_frames)

        video_stem = Path(video_path).stem
        kp2d_path = output_dir / f"{video_stem}_keypoints_2d.npy"
        kp3d_path = output_dir / f"{video_stem}_keypoints_3d.npy"

        np.save(kp2d_path, keypoints_2d)
        np.save(kp3d_path, keypoints_3d)

        # Save metadata
        meta = {
            "video_path": str(video_path),
            "fps": float(fps),
            "num_frames": int(keypoints_2d.shape[0]),
            "coco_17_names": COCO_17_NAMES,
            "mediapipe_33_names": MEDIAPIPE_33_NAMES,
        }
        import json
        meta_path = output_dir / f"{video_stem}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[INFO] Saved 2D keypoints: {kp2d_path}")
        print(f"[INFO] Saved 3D keypoints: {kp3d_path}")
        print(f"[INFO] Saved metadata: {meta_path}")

        return kp2d_path, kp3d_path, fps
