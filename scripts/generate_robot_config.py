#!/usr/bin/env python3
"""Generate a robot YAML config template from a MJCF/URDF model.

This script analyzes a robot model file and generates a YAML config
with the standard structure, reducing manual effort when adding new robots.

Usage:
    # Generate from MJCF
    python scripts/generate_robot_config.py \
        --model asset/robot/g1_description/mjcf/g1.xml \
        --name g1 \
        --output config/robot/g1.yaml

    # Generate with base config inheritance
    python scripts/generate_robot_config.py \
        --model asset/robot/h2_description/H2.xml \
        --name h2 \
        --extends g1 \
        --output config/robot/h2.yaml

    # Print to stdout
    python scripts/generate_robot_config.py \
        --model asset/robot/r1_description/R1.xml \
        --name r1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config" / "robot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate robot YAML config from MJCF/URDF model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to robot MJCF/URDF file (relative to project root or absolute)",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Robot name (used for output filename and keypoints_path)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output YAML path (default: config/robot/<name>.yaml)",
    )
    parser.add_argument(
        "--extends",
        default=None,
        help="Base robot name to inherit common settings from",
    )
    parser.add_argument(
        "--xml-path",
        default=None,
        help="Override robot_xml_path in output (relative to project root)",
    )
    return parser.parse_args()


def load_model_body_names(model_path: str) -> list[str]:
    """Load a model and return all body names."""
    spec = mujoco.MjSpec.from_file(model_path)
    model = spec.compile()
    names = []
    for i in range(model.nbody):
        name = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name >= 0:
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name:
                names.append(body_name)
    return names


def find_body_by_substring(names: list[str], substring: str) -> str | None:
    """Find a body name containing a substring (case-insensitive)."""
    substring_lower = substring.lower()
    for name in names:
        if substring_lower in name.lower():
            return name
    return None


def find_bodies_by_pattern(names: list[str], patterns: list[str]) -> list[str]:
    """Find bodies matching any of the given patterns."""
    results = []
    for pattern in patterns:
        match = find_body_by_substring(names, pattern)
        if match:
            results.append(match)
    return results


def generate_standard_config(
    robot_name: str,
    model_path: str,
    xml_path: str,
    extends: str | None = None,
) -> dict:
    """Generate a standard robot config with common defaults."""
    config = {}

    # Inheritance marker
    if extends:
        config["extends"] = extends

    # Standard fields
    config["robot_xml_path"] = xml_path
    config["verbose"] = True
    config["render_debug"] = False
    config["keypoints_path"] = (
        f"output_data/keypoints/{robot_name}/Form_1_stageii_keypoints.pkl"
    )

    # Joint limits (standard defaults)
    config["joints_limit_offset_degrees"] = {
        "knee_joint": [10.0, 0.0],
        "elbow_joint": [20.0, 0.0],
    }

    # Knee angle offset
    config["knee_angle_offset_degrees"] = 15.0

    # Placeholder robot_links (to be filled manually)
    config["robot_links"] = {
        "# TODO: Fill in robot-specific link definitions": "",
        "left_hip": ["hips_sphere", "left_hip_roll_link"],
        "left_thigh": ["left_hip_roll_link", "left_knee_link"],
        "left_calf": ["left_knee_link", "left_ankle_roll_link"],
        "right_hip": ["hips_sphere", "right_hip_roll_link"],
        "right_thigh": ["right_hip_roll_link", "right_knee_link"],
        "right_calf": ["right_knee_link", "right_ankle_roll_link"],
        "neck": ["hips_sphere", "neck_sphere"],
        "head": ["neck_sphere", "head_sphere"],
        "left_shoulder": ["neck_sphere", "left_shoulder_roll_link"],
        "left_arm": ["left_shoulder_roll_link", "left_elbow_link"],
        "left_fore_arm": ["left_elbow_link", "left_wrist_yaw_link"],
        "right_shoulder": ["neck_sphere", "right_shoulder_roll_link"],
        "right_arm": ["right_shoulder_roll_link", "right_elbow_link"],
        "right_fore_arm": ["right_elbow_link", "right_wrist_yaw_link"],
    }

    # Contact links (standard defaults)
    config["contact_links"] = [
        "left_foot_end_link",
        "left_toe_link",
        "right_foot_end_link",
        "right_toe_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]

    # Contact parameters (standard defaults)
    config["contact_vel_calculate_window"] = 6
    config["contact_vel_threshold"] = 0.5
    config["contact_height_threshold"] = 0.05
    config["contact_height_lpf_alpha"] = 0.15
    config["contact_pos_fixed_factor"] = 15.0

    # Placeholder ik_match_table (to be filled manually)
    config["ik_match_table"] = {
        "# TODO: Fill in robot-specific IK match table": "",
        "hips_mean": ["hips_sphere", 100, 0],
        "left_hip": ["left_hip_roll_link", 30, 3],
        "left_thigh": ["left_knee_link", 0.0, 3.0],
        "left_calf": ["left_ankle_roll_link", 30, 3],
        "right_hip": ["right_hip_roll_link", 30, 3],
        "right_thigh": ["right_knee_link", 0.0, 3.0],
        "right_calf": ["right_ankle_roll_link", 30, 3],
        "head": ["head_sphere", 0, 3.0],
        "left_shoulder": ["left_shoulder_roll_link", 30, 3],
        "left_arm": ["left_elbow_link", 10.0, 1],
        "left_fore_arm": ["left_wrist_yaw_link", 10, 1],
        "right_shoulder": ["right_shoulder_roll_link", 30, 3],
        "right_arm": ["right_elbow_link", 10.0, 1],
        "right_fore_arm": ["right_wrist_yaw_link", 10, 1],
    }

    # Standard key_frame_config
    config["key_frame_config"] = {
        "hips_mean": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [1.0, 0.0, 0.0],
                "z": [0.0, 1.0, 0.0],
            },
        },
        "left_up_leg": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [1.0, 0.0, 0.0],
                "z": [0.0, 1.0, 0.0],
            },
        },
        "left_leg": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [1.0, 0.0, 0.0],
                "z": [0.0, 1.0, 0.0],
            },
        },
        "left_foot": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [1.0, 0.0, 0.0],
                "z": [0.0, 1.0, 0.0],
            },
        },
        "right_up_leg": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [1.0, 0.0, 0.0],
                "z": [0.0, 1.0, 0.0],
            },
        },
        "right_leg": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [1.0, 0.0, 0.0],
                "z": [0.0, 1.0, 0.0],
            },
        },
        "right_foot": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [1.0, 0.0, 0.0],
                "z": [0.0, 1.0, 0.0],
            },
        },
        "shoulder_mean": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [1.0, 0.0, 0.0],
                "z": [0.0, 1.0, 0.0],
            },
        },
        "left_arm": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [0.0, 1.0, 0.0],
                "z": [-1.0, 0.0, 0.0],
            },
        },
        "left_fore_arm": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [0.0, 1.0, 0.0],
                "z": [-1.0, 0.0, 0.0],
            },
        },
        "left_hand": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [0.0, 1.0, 0.0],
                "z": [-1.0, 0.0, 0.0],
            },
        },
        "right_arm": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [0.0, -1.0, 0.0],
                "z": [1.0, 0.0, 0.0],
            },
        },
        "right_fore_arm": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [0.0, -1.0, 0.0],
                "z": [1.0, 0.0, 0.0],
            },
        },
        "right_hand": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [0.0, -1.0, 0.0],
                "z": [1.0, 0.0, 0.0],
            },
        },
        "head": {
            "offset_deg_xyz": [0.0, 0.0, 0.0],
            "axis_map_cols": {
                "x": [0.0, 0.0, 1.0],
                "y": [1.0, 0.0, 0.0],
                "z": [0.0, 1.0, 0.0],
            },
        },
    }

    return config


def load_base_config(base_name: str) -> dict:
    """Load a base robot config for inheritance."""
    base_path = CONFIG_DIR / f"{base_name}.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    with open(base_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()

    # Resolve model path
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = str(PROJECT_ROOT / model_path)

    if not os.path.isfile(model_path):
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Determine XML path in output
    xml_path = args.xml_path or args.model
    if os.path.isabs(xml_path):
        try:
            xml_path = os.path.relpath(xml_path, PROJECT_ROOT)
        except ValueError:
            pass  # keep absolute if on different drive

    # Generate config
    config = generate_standard_config(
        robot_name=args.name,
        model_path=model_path,
        xml_path=xml_path,
        extends=args.extends,
    )

    # If extends specified, only output overridden fields
    if args.extends:
        base_config = load_base_config(args.extends)
        # Mark as inherited and only keep differences
        config = {
            "extends": args.extends,
            "robot_xml_path": xml_path,
            "keypoints_path": config["keypoints_path"],
        }

    # Output
    output_str = yaml.dump(
        config,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Auto-generated config for {args.name}\n")
            if args.extends:
                f.write(f"# Inherits from: {args.extends}\n")
            f.write(f"# Model: {xml_path}\n")
            f.write("#\n")
            f.write("# TODO: Review and customize robot_links and ik_match_table\n")
            f.write("#       for your specific robot's body names.\n\n")
            f.write(output_str)
        print(f"Config written to: {output_path}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
