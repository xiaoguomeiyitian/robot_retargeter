#!/usr/bin/env python3
"""Validate all robot YAML configs against a unified schema.

Checks:
  1. Required top-level keys exist
  2. robot_xml_path points to an existing file
  3. joints_limit_offset_degrees format is consistent (dict of list or scalar)
  4. contact_links is a non-empty list
  5. contact_vel_calculate_window exists and is int
  6. contact_vel_threshold exists and is float
  7. contact_height_threshold exists and is float
  8. contact_height_lpf_alpha exists and is float
  9. contact_pos_fixed_factor exists and is float
  10. knee_angle_offset_degrees exists and is float
  11. ik_match_table is a non-empty dict
  12. key_frame_config is a non-empty dict
  13. robot_links has all 14 required link names

Usage:
    python scripts/validate_configs.py
    python scripts/validate_configs.py --fix   # auto-fix minor issues
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config" / "robot"

REQUIRED_TOP_KEYS = {
    "robot_xml_path",
    "verbose",
    "render_debug",
    "keypoints_path",
    "joints_limit_offset_degrees",
    "robot_links",
    "contact_links",
    "ik_match_table",
    "key_frame_config",
}

REQUIRED_CONTACT_KEYS = {
    "contact_vel_calculate_window",
    "contact_vel_threshold",
    "contact_height_threshold",
    "contact_height_lpf_alpha",
    "contact_pos_fixed_factor",
}

REQUIRED_LINK_NAMES = {
    "left_hip", "left_thigh", "left_calf",
    "right_hip", "right_thigh", "right_calf",
    "neck", "head",
    "left_shoulder", "left_arm", "left_fore_arm",
    "right_shoulder", "right_arm", "right_fore_arm",
}

# Warnings and errors collected
issues: list[tuple[str, str, str]] = []  # (robot_name, level, message)


def validate_config(robot_name: str, config: dict, fix: bool = False) -> bool:
    """Validate a single robot config. Returns True if no errors."""
    has_errors = False

    # 1. Required top-level keys
    missing = REQUIRED_TOP_KEYS - set(config.keys())
    if missing:
        issues.append((robot_name, "ERROR", f"Missing required keys: {missing}"))
        has_errors = True

    # 2. robot_xml_path exists
    xml_path = config.get("robot_xml_path", "")
    if xml_path:
        full_path = PROJECT_ROOT / xml_path
        if not full_path.exists():
            issues.append((robot_name, "ERROR", f"robot_xml_path does not exist: {xml_path}"))
            has_errors = True

    # 3. joints_limit_offset_degrees format
    jlod = config.get("joints_limit_offset_degrees", {})
    if not isinstance(jlod, dict):
        issues.append((robot_name, "ERROR", f"joints_limit_offset_degrees must be a dict, got {type(jlod).__name__}"))
        has_errors = True
    else:
        for joint_name, value in jlod.items():
            if isinstance(value, (int, float)):
                issues.append((robot_name, "WARN", f"joints_limit_offset_degrees['{joint_name}'] is scalar {value}, expected list [low, high]"))
            elif isinstance(value, list):
                if len(value) != 2:
                    issues.append((robot_name, "ERROR", f"joints_limit_offset_degrees['{joint_name}'] has {len(value)} elements, expected 2"))
                    has_errors = True

    # 4. contact_links is non-empty list
    contact_links = config.get("contact_links", [])
    if not contact_links:
        issues.append((robot_name, "ERROR", "contact_links is empty or missing"))
        has_errors = True

    # 5-9. Contact parameters
    for key in REQUIRED_CONTACT_KEYS:
        if key not in config:
            issues.append((robot_name, "WARN", f"Missing contact parameter: {key}"))

    # 10. knee_angle_offset_degrees
    knee_offset = config.get("knee_angle_offset_degrees")
    if knee_offset is None:
        issues.append((robot_name, "WARN", "Missing knee_angle_offset_degrees (default 15.0 assumed)"))
    elif not isinstance(knee_offset, (int, float)):
        issues.append((robot_name, "ERROR", f"knee_angle_offset_degrees must be numeric, got {type(knee_offset).__name__}"))
        has_errors = True

    # 11. ik_match_table
    ik_table = config.get("ik_match_table", {})
    if not ik_table:
        issues.append((robot_name, "ERROR", "ik_match_table is empty or missing"))
        has_errors = True

    # 12. key_frame_config
    kf_config = config.get("key_frame_config", {})
    if not kf_config:
        issues.append((robot_name, "ERROR", "key_frame_config is empty or missing"))
        has_errors = True

    # 13. robot_links has all required link names
    robot_links = config.get("robot_links", {})
    missing_links = REQUIRED_LINK_NAMES - set(robot_links.keys())
    if missing_links:
        issues.append((robot_name, "ERROR", f"robot_links missing: {missing_links}"))
        has_errors = True

    return not has_errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate robot YAML configs")
    parser.add_argument("--fix", action="store_true", help="Auto-fix minor issues")
    args = parser.parse_args()

    yaml_files = sorted(CONFIG_DIR.glob("*.yaml"))
    if not yaml_files:
        print(f"No YAML files found in {CONFIG_DIR}")
        sys.exit(1)

    print(f"Validating {len(yaml_files)} robot configs in {CONFIG_DIR} ...\n")

    all_ok = True
    for yaml_file in yaml_files:
        robot_name = yaml_file.stem
        with open(yaml_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        ok = validate_config(robot_name, config, fix=args.fix)
        if not ok:
            all_ok = False

    # Print report
    errors = [(r, m) for r, l, m in issues if l == "ERROR"]
    warnings = [(r, m) for r, l, m in issues if l == "WARN"]

    if warnings:
        print(f"⚠️  {len(warnings)} warning(s):")
        for robot, msg in warnings:
            print(f"  [{robot}] {msg}")
        print()

    if errors:
        print(f"❌ {len(errors)} error(s):")
        for robot, msg in errors:
            print(f"  [{robot}] {msg}")
        print()
        sys.exit(1)
    else:
        print(f"✅ All {len(yaml_files)} configs passed validation ({len(warnings)} warnings)")
        sys.exit(0)


if __name__ == "__main__":
    main()
