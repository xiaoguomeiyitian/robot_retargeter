"""Export robot motion animation to MP4 video using MuJoCo offscreen rendering.

This script renders each frame using MuJoCo's Renderer (GPU-accelerated OpenGL)
and encodes the result to MP4 via ffmpeg.

Usage:
    python scripts/export_video.py \
        --motion dance2 \
        --robots subject1 subject3 subject5 \
        --output output.mp4 \
        --fps 30 \
        --width 1920 \
        --height 1080
"""

import argparse
import os
import subprocess
import sys
import tempfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# Add scripts directory to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from multi_robot_visualize_viser import (
    build_combined_spec,
    get_qpos_start,
    load_motion,
    resolve_robot,
    ROBOT_CONFIG_DIR,
    PROJECT_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export robot motion to MP4 video")
    parser.add_argument("--motion", required=True, help="Motion name")
    parser.add_argument("--robots", nargs="+", required=True, help="Robot names")
    parser.add_argument("--motion_dir", default="output_data/robot_motion")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--source_fps", type=float, default=30.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--all_geoms", action="store_true")
    args = parser.parse_args()

    # Check ffmpeg
    if not subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
        raise RuntimeError("ffmpeg not found. Install with: sudo apt install ffmpeg")

    # Load motion data
    robot_qpos: dict[str, np.ndarray] = {}
    for robot in args.robots:
        csv_path = os.path.join(args.motion_dir, f"{args.motion}_{robot}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Motion file not found: {csv_path}")
        robot_qpos[robot] = load_motion(csv_path)

    n_frames = min(q.shape[0] for q in robot_qpos.values())
    print(f"Total frames: {n_frames}")

    # Build model
    spec = build_combined_spec(args.robots)
    model = spec.compile()
    data = mujoco.MjData(model)

    robot_start: dict[str, int] = {}
    robot_dim: dict[str, int] = {}
    for robot in args.robots:
        start = get_qpos_start(model, f"{robot}_floating_base_joint")
        robot_start[robot] = start
        robot_dim[robot] = robot_qpos[robot].shape[1]

    # Grid layout
    import math
    spacing = 2.0
    n = len(args.robots)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    robot_offset: dict[str, tuple[float, float]] = {}
    for idx, robot in enumerate(args.robots):
        col = idx % cols
        row = idx // cols
        dx = (col - (cols - 1) / 2.0) * spacing
        dy = (row - (rows - 1) / 2.0) * spacing
        robot_offset[robot] = (dx, dy)

    # Create renderer
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    print(f"Renderer: {args.width}x{args.height}")

    # Frame skipping
    step = max(1, round(args.source_fps / args.fps))

    # Use ffmpeg pipe for efficient encoding
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{args.width}x{args.height}",
        "-pix_fmt", "rgb24",
        "-r", str(args.fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        args.output,
    ]
    print(f"Encoding to: {args.output}")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        for frame_idx in range(0, n_frames, step):
            # Set qpos
            for robot in args.robots:
                start = robot_start[robot]
                dim = robot_dim[robot]
                data.qpos[start:start + dim] = robot_qpos[robot][frame_idx]
                dx, dy = robot_offset[robot]
                data.qpos[start] += dx
                data.qpos[start + 1] += dy

            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            img = renderer.render()

            # Write frame to ffmpeg
            proc.stdin.write(img.tobytes())

            if frame_idx % 300 == 0:
                print(f"  Frame {frame_idx}/{n_frames} ({frame_idx/n_frames*100:.1f}%)")

        proc.stdin.close()
        proc.wait()
        print(f"Video saved: {args.output}")
    except Exception as e:
        proc.kill()
        raise e


if __name__ == "__main__":
    main()
