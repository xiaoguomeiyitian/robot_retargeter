#!/usr/bin/env python3
"""One-click training pipeline: retarget → NPZ → RL training.

Chains the full workflow:
  1. (Optional) Run retarget from keypoints CSV
  2. Export retargeted CSV to NPZ format
  3. Launch RL training in unitree_rl_mjlab

Usage:
    # Full pipeline: retarget + export + train
    python scripts/train_pipeline.py \
        --robot g1 \
        --motion-name dance1 \
        --retarget-config config/robot/g1.yaml \
        --keypoints output_data/keypoints/dance1.pkl \
        --rl-task unitree_g1_flat_tracking

    # Skip retarget, use existing CSV
    python scripts/train_pipeline.py \
        --robot g1 \
        --motion-name dance1 \
        --csv output_data/robot_motion/Form_1_stageii_g1.csv \
        --rl-task unitree_g1_flat_training

    # Export only (no training)
    python scripts/train_pipeline.py \
        --robot g1 \
        --motion-name dance1 \
        --csv output_data/robot_motion/Form_1_stageii_g1.csv \
        --export-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click RL training pipeline from retargeted motion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Source options
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Existing retargeted CSV (skip retarget step).",
    )
    source.add_argument(
        "--npz",
        type=str,
        default=None,
        help="Existing NPZ file (skip retarget + export steps).",
    )

    parser.add_argument(
        "--robot",
        type=str,
        required=True,
        help="Robot name (g1, g1_23dof, h1, h1_2, etc.).",
    )
    parser.add_argument(
        "--motion-name",
        type=str,
        required=True,
        help="Motion name (used for output file naming).",
    )
    parser.add_argument(
        "--retarget-config",
        type=str,
        default=None,
        help="Robot YAML config for retargeting (if running retarget).",
    )
    parser.add_argument(
        "--keypoints",
        type=str,
        default=None,
        help="Keypoints PKL file for retargeting.",
    )

    # Export options
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
        "--export-dir",
        type=str,
        default=None,
        help="Output directory for NPZ (default: output_data/npz/).",
    )

    # Training options
    parser.add_argument(
        "--rl-task",
        type=str,
        default=None,
        help="RL task ID in unitree_rl_mjlab (e.g., unitree_g1_flat_tracking).",
    )
    parser.add_argument(
        "--rl-root",
        type=str,
        default=None,
        help="Path to unitree_rl_mjlab (default: ../unitree_rl_mjlab).",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export NPZ, skip training.",
    )
    parser.add_argument(
        "--train-args",
        type=str,
        default="",
        help="Extra arguments to pass to train.py (quoted string).",
    )
    return parser.parse_args()


def run_retarget(
    config_path: str,
    keypoints_path: str,
    output_csv: str,
) -> None:
    """Run robot retargeting to produce CSV."""
    print("=" * 60)
    print("步骤 1: 正在执行重定向...")
    print("=" * 60)

    cmd = [
        sys.executable, str(SCRIPT_DIR / "robot_retarget.py"),
        "--config", config_path,
        "--keypoints", keypoints_path,
        "--output", output_csv,
    ]
    print(f"  命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Retarget failed with code {result.returncode}")
    print("  ✅ 重定向完成!\n")


def run_export(
    csv_path: str,
    robot_name: str,
    output_npz: str,
    input_fps: float,
    output_fps: float,
) -> None:
    """Export CSV to NPZ format."""
    print("=" * 60)
    print("步骤 2: 正在导出为 NPZ...")
    print("=" * 60)

    cmd = [
        sys.executable, str(SCRIPT_DIR / "export_npz.py"),
        "--csv", csv_path,
        "--robot", robot_name,
        "--input-fps", str(input_fps),
        "--output-fps", str(output_fps),
        "--output", output_npz,
    ]
    print(f"  命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Export failed with code {result.returncode}")
    print("  ✅ 导出完成!\n")


def run_training(
    npz_path: str,
    rl_task: str,
    rl_root: str,
    extra_args: str = "",
) -> None:
    """Launch RL training in unitree_rl_mjlab."""
    print("=" * 60)
    print("步骤 3: 正在启动 RL 训练...")
    print("=" * 60)

    train_script = Path(rl_root) / "scripts" / "train.py"
    if not train_script.exists():
        raise FileNot找到Error(f"Training script not found: {train_script}")

    cmd = [
        sys.executable, str(train_script),
        "--task", rl_task,
        "--motion-file", npz_path,
    ]
    if extra_args:
        cmd.extend(extra_args.split())

    print(f"  命令: {' '.join(cmd)}")
    print(f"  RL 根目录: {rl_root}")
    result = subprocess.run(cmd, cwd=rl_root)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")
    print("  ✅ 训练完成!\n")


def main() -> None:
    args = parse_args()

    # Resolve paths
    export_dir = Path(args.export_dir) if args.export_dir else (PROJECT_ROOT / "output_data" / "npz")
    export_dir.mkdir(parents=True, exist_ok=True)

    rl_root = args.rl_root or str(PROJECT_ROOT.parent / "unitree_rl_mjlab")

    # Determine NPZ path
    if args.npz:
        npz_path = str(Path(args.npz).resolve())
        print(f"使用已有 NPZ: {npz_path}")
    else:
        # Avoid double robot suffix (e.g. Form_1_stageii_g1 → Form_1_stageii_g1.npz, not Form_1_stageii_g1_g1.npz)
        motion_stem = args.motion_name
        suffix = f"_{args.robot}"
        if motion_stem.endswith(suffix):
            npz_stem = motion_stem
        else:
            npz_stem = f"{motion_stem}_{args.robot}"
        npz_path = str(export_dir / f"{npz_stem}.npz")

        # Step 1: Retarget (if CSV not provided)
        if args.csv:
            csv_path = str(Path(args.csv).resolve())
            print(f"使用已有 CSV: {csv_path}")
        else:
            if not args.retarget_config or not args.keypoints:
                print("错误: 请提供 --csv、--npz 或同时提供 --retarget-config + --keypoints")
                sys.exit(1)
            csv_path = str(PROJECT_ROOT / "output_data" / "robot_motion" / f"{npz_stem}.csv")
            run_retarget(args.retarget_config, args.keypoints, csv_path)

        # Step 2: Export to NPZ
        run_export(csv_path, args.robot, npz_path, args.input_fps, args.output_fps)

    # Step 3: Train (unless --export-only)
    if args.export_only:
        print("✅ 仅导出模式，跳过训练。")
        print(f"NPZ 文件: {npz_path}")
        print(f"\n手动训练:")
        print(f"  cd {rl_root}")
        print(f"  python scripts/train.py --task <TASK_ID> --motion-file {npz_path}")
    else:
        if not args.rl_task:
            print("错误: 训练需要 --rl-task（或使用 --export-only）")
            sys.exit(1)
        run_training(npz_path, args.rl_task, rl_root, args.train_args)

    print("\n🎉 流水线完成!")


if __name__ == "__main__":
    main()
