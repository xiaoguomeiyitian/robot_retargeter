#!/usr/bin/env python3
"""End-to-end pipeline: video → robot motion.

This script chains the full pipeline:
  1. Extract 2D/3D keypoints from video (MediaPipe)
  2. Lift 2D keypoints to 3D (temporal model)
  3. Fit SMPL-X body model → save AMASS-format NPZ
  4. Run smpl_replay.py (SMPL-X → skeleton keypoints pkl)
  5. Run robot_retarget.py (keypoints pkl → robot motion)

Usage:
    python scripts/video_to_robot.py \
        --video path/to/video.mp4 \
        --robots g1 h2 \
        --output-dir output_data/video_to_robot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.video_to_robot.video_extract import VideoExtractor
from src.video_to_robot.lift_2d_to_3d import Lift2Dto3D
from src.video_to_robot.fit_smplx import FitSMPLX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract human motion from video and retarget to robot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--robots",
        type=str,
        nargs="+",
        default=["g1"],
        help="Target robot names (default: g1).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_data/video_to_robot",
        help="Output directory (default: output_data/video_to_robot).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output frame rate. None to use video FPS.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="asset/models/pose_landmarker_heavy.task",
        help="Path to MediaPipe PoseLandmarker model.",
    )
    parser.add_argument(
        "--smplx-model-dir",
        type=str,
        default="asset/smplx",
        help="Path to SMPL-X model directory.",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["neutral", "male", "female"],
        help="SMPL-X model gender.",
    )
    parser.add_argument(
        "--lift-method",
        type=str,
        default="simple",
        choices=["mediapipe", "simple"],
        help="2D→3D lifting method.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Disable visualization in smpl_replay.py and robot_retarget.py.",
    )
    parser.add_argument(
        "--skip-retarget",
        action="store_true",
        help="Skip robot retargeting (only generate SMPL-X motion).",
    )
    return parser.parse_args()


def save_amass_npz(
    output_path: Path,
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    joint_rotvec: np.ndarray,
    fps: float,
    gender: str = "neutral",
    surface_model_type: str = "smplx",
) -> None:
    """Save motion in AMASS-compatible NPZ format.

    The output is compatible with smpl_replay.py's load_motion_arrays():
      - trans: (T, 3) root translation
      - root_orient: (T, 3) root orientation (axis-angle)
      - pose_body: (T, 63) body pose (21 joints × 3 axis-angle)
      - betas: (10,) body shape parameters
      - gender: string
      - surface_model_type: "smplx"
      - mocap_frame_rate: float
    """
    n_frames = root_pos.shape[0]

    # joint_rotvec: (T, 21, 3) → pose_body: (T, 63)
    if joint_rotvec.shape == (n_frames, 21, 3):
        pose_body = joint_rotvec.reshape(n_frames, 63)
    elif joint_rotvec.shape == (n_frames, 63):
        pose_body = joint_rotvec
    else:
        pose_body = np.zeros((n_frames, 63), dtype=np.float32)
        n_joints = min(joint_rotvec.shape[1] if joint_rotvec.ndim >= 2 else 0, 21)
        if n_joints > 0:
            if joint_rotvec.ndim == 2 and joint_rotvec.shape[1] == 3:
                pose_body[:, :3] = joint_rotvec
            elif joint_rotvec.ndim == 3:
                pose_body[:, :n_joints * 3] = joint_rotvec[:, :n_joints, :].reshape(n_frames, -1)

    betas = np.zeros(10, dtype=np.float32)

    np.savez(
        output_path,
        trans=root_pos.astype(np.float32),
        root_orient=root_rot.astype(np.float32),
        pose_body=pose_body.astype(np.float32),
        betas=betas,
        gender=np.array(gender),
        surface_model_type=np.array(surface_model_type),
        mocap_frame_rate=np.array(fps, dtype=np.float32),
    )
    print(f"[INFO] Saved AMASS-format motion: {output_path}")


def run_smpl_replay(
    motion_file: Path,
    robot_config: Path,
    smpl_model_path: Path,
    gender: str,
    no_viewer: bool = True,
) -> subprocess.CompletedProcess:
    """Run smpl_replay.py to generate skeleton keypoints pkl."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "smpl_replay.py"),
        "--motion_file", str(motion_file),
        "--robot-config", str(robot_config),
        "--smpl-model-path", str(smpl_model_path),
        "--gender", gender,
        "--no-viewer",
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def run_robot_retarget(
    robot_config: Path,
    keypoints_name: str,
    no_viewer: bool = True,
) -> subprocess.CompletedProcess:
    """Run robot_retarget.py to generate robot motion."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "robot_retarget.py"),
        "--config", str(robot_config),
        "--keypoints-name", keypoints_name,
    ]
    if no_viewer:
        cmd.append("--no-viewer")
    print(f"[INFO] Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT))


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_stem = video_path.stem

    print("=" * 60)
    print("  Video-to-Robot Pipeline")
    print("=" * 60)
    print(f"  Video: {video_path}")
    print(f"  Robots: {args.robots}")
    print(f"  Output: {output_dir}")
    print()

    # ── Step 1: Extract keypoints from video ──────────────────────────────
    print("[Step 1/5] Extracting keypoints from video...")
    t0 = time.time()

    extractor = VideoExtractor(model_path=args.model)
    kp2d_path = output_dir / f"{video_stem}_keypoints_2d.npy"
    kp3d_mediapipe_path = output_dir / f"{video_stem}_keypoints_3d_mediapipe.npy"

    if kp2d_path.exists() and kp3d_mediapipe_path.exists():
        print(f"[INFO] Loading cached keypoints: {kp2d_path}")
        keypoints_2d = np.load(kp2d_path)
        keypoints_3d_mediapipe = np.load(kp3d_mediapipe_path)
        meta_path = output_dir / f"{video_stem}_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                video_fps = meta["fps"]
        else:
            video_fps = 30.0
    else:
        keypoints_2d, keypoints_3d_mediapipe, video_fps = extractor.extract(
            video_path, max_frames=args.max_frames
        )
        np.save(kp2d_path, keypoints_2d)
        np.save(kp3d_mediapipe_path, keypoints_3d_mediapipe)
        meta = {
            "video_path": str(video_path),
            "fps": float(video_fps),
            "num_frames": int(keypoints_2d.shape[0]),
        }
        with open(output_dir / f"{video_stem}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    if args.fps is not None:
        video_fps = args.fps

    print(f"[INFO] 2D keypoints: {keypoints_2d.shape}")
    print(f"[INFO] 3D MediaPipe: {keypoints_3d_mediapipe.shape}")
    print(f"[INFO] FPS: {video_fps}")
    print(f"[INFO] Time: {time.time() - t0:.2f}s")
    print()

    # ── Step 2: Lift 2D to 3D ────────────────────────────────────────────
    print("[Step 2/5] Lifting 2D keypoints to 3D...")
    t0 = time.time()

    lifter = Lift2Dto3D(method=args.lift_method)
    kp3d_path = output_dir / f"{video_stem}_keypoints_3d.npy"

    if kp3d_path.exists():
        print(f"[INFO] Loading cached 3D keypoints: {kp3d_path}")
        keypoints_3d = np.load(kp3d_path)
    else:
        keypoints_3d = lifter.lift(keypoints_2d, keypoints_3d_mediapipe)
        np.save(kp3d_path, keypoints_3d)

    print(f"[INFO] 3D keypoints: {keypoints_3d.shape}")
    print(f"[INFO] Time: {time.time() - t0:.2f}s")
    print()

    # ── Step 3: Fit SMPL-X → AMASS NPZ ──────────────────────────────────
    print("[Step 3/5] Fitting SMPL-X body model...")
    t0 = time.time()

    fitter = FitSMPLX(
        smplx_model_dir=args.smplx_model_dir,
        gender=args.gender,
    )
    smplx_npz_path = output_dir / f"{video_stem}_smplx.npz"

    if smplx_npz_path.exists():
        print(f"[INFO] Loading cached SMPL-X: {smplx_npz_path}")
        cached = np.load(smplx_npz_path)
        root_pos = cached["trans"]
        root_rot = cached["root_orient"]
    else:
        smpl_params = fitter.fit(keypoints_3d, fps=video_fps)
        # Extract AMASS-format parameters
        root_pos = smpl_params.get("root_pos", smpl_params.get("transl", np.zeros((keypoints_3d.shape[0], 3))))
        root_rot = smpl_params.get("root_rot", smpl_params.get("global_orient", np.zeros((keypoints_3d.shape[0], 3))))
        joint_rotvec = smpl_params.get("body_pose", np.zeros((keypoints_3d.shape[0], 21, 3)))
        if joint_rotvec.ndim == 2 and joint_rotvec.shape[1] == 63:
            joint_rotvec = joint_rotvec.reshape(-1, 21, 3)
        save_amass_npz(
            smplx_npz_path,
            root_pos=root_pos,
            root_rot=root_rot,
            joint_rotvec=joint_rotvec,
            fps=video_fps,
            gender=args.gender,
        )

    print(f"[INFO] SMPL-X motion: {smplx_npz_path}")
    print(f"[INFO] Time: {time.time() - t0:.2f}s")
    print()

    # ── Step 4: Run smpl_replay.py ───────────────────────────────────────
    first_robot = args.robots[0]
    robot_config_path = PROJECT_ROOT / "config" / "robot" / f"{first_robot}.yaml"
    if not robot_config_path.exists():
        print(f"[WARN] Robot config not found: {robot_config_path}")
        print("[WARN] Skipping smpl_replay.py and robot_retarget.py")
    else:
        print("[Step 4/5] Running smpl_replay.py...")
        t0 = time.time()

        smpl_model_path = Path(args.smplx_model_dir)
        result = run_smpl_replay(
            motion_file=smplx_npz_path,
            robot_config=robot_config_path,
            smpl_model_path=smpl_model_path,
            gender=args.gender,
            no_viewer=args.no_viewer,
        )
        if result.returncode != 0:
            print(f"[WARN] smpl_replay.py returned code {result.returncode}")
        print(f"[INFO] Time: {time.time() - t0:.2f}s")
        print()

        # ── Step 5: Run robot_retarget.py ─────────────────────────────────
        if not args.skip_retarget:
            print("[Step 5/5] Running robot_retarget.py...")
            t0 = time.time()

            keypoints_name = smplx_npz_path.stem

            for robot in args.robots:
                robot_cfg = PROJECT_ROOT / "config" / "robot" / f"{robot}.yaml"
                if not robot_cfg.exists():
                    print(f"[WARN] Robot config not found: {robot_cfg}")
                    continue

                result = run_robot_retarget(
                    robot_config=robot_cfg,
                    keypoints_name=keypoints_name,
                    no_viewer=args.no_viewer,
                )
                if result.returncode != 0:
                    print(f"[WARN] robot_retarget.py returned code {result.returncode} for {robot}")

            print(f"[INFO] Time: {time.time() - t0:.2f}s")
        else:
            print("[Step 5/5] Skipped (--skip-retarget)")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  SMPL-X motion: {smplx_npz_path.name}")
    if not args.skip_retarget:
        print(f"  Keypoints name: {smplx_npz_path.stem}")
    print(f"  Robots: {args.robots}")
    print()


if __name__ == "__main__":
    main()
