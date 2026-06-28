#!/usr/bin/env python3
"""Tests for the YAML config validation script."""

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_configs import validate_config


class TestValidateConfig:
    def test_valid_config(self):
        config = {
            "robot_xml_path": "asset/robot/g1_description/mjcf/g1.xml",
            "verbose": True,
            "render_debug": False,
            "keypoints_path": "output_data/keypoints/g1/test_keypoints.pkl",
            "joints_limit_offset_degrees": {
                "knee_joint": [10.0, 0.0],
                "elbow_joint": [20.0, 0.0],
            },
            "robot_links": {
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
            },
            "contact_links": ["left_foot_end_link", "left_toe_link"],
            "ik_match_table": {
                "hips_mean": ["hips_sphere", 100, 0],
            },
            "key_frame_config": {
                "hips_mean": {
                    "offset_deg_xyz": [0.0, 0.0, 0.0],
                    "axis_map_cols": {
                        "x": [0.0, 0.0, 1.0],
                        "y": [1.0, 0.0, 0.0],
                        "z": [0.0, 1.0, 0.0],
                    },
                },
            },
            "contact_vel_calculate_window": 6,
            "contact_vel_threshold": 0.5,
            "contact_height_threshold": 0.05,
            "contact_height_lpf_alpha": 0.15,
            "contact_pos_fixed_factor": 15.0,
            "knee_angle_offset_degrees": 15.0,
        }
        ok = validate_config("test_robot", config)
        assert ok

    def test_missing_required_keys(self):
        config = {
            "robot_xml_path": "asset/robot/g1_description/mjcf/g1.xml",
        }
        ok = validate_config("test_robot", config)
        assert not ok

    def test_missing_robot_links(self):
        config = {
            "robot_xml_path": "asset/robot/g1_description/mjcf/g1.xml",
            "verbose": True,
            "render_debug": False,
            "keypoints_path": "output_data/keypoints/g1/test.pkl",
            "joints_limit_offset_degrees": {},
            "robot_links": {
                "left_hip": ["hips", "left_hip_roll"],
            },
            "contact_links": ["left_foot"],
            "ik_match_table": {"hips_mean": ["hips", 100, 0]},
            "key_frame_config": {"hips_mean": {}},
            "knee_angle_offset_degrees": 15.0,
        }
        ok = validate_config("test_robot", config)
        assert not ok

    def test_scalar_joints_limit_warning(self):
        config = {
            "robot_xml_path": "asset/robot/g1_description/mjcf/g1.xml",
            "verbose": True,
            "render_debug": False,
            "keypoints_path": "output_data/keypoints/g1/test.pkl",
            "joints_limit_offset_degrees": {
                "knee_joint": 10.0,  # scalar instead of list
            },
            "robot_links": {
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
            },
            "contact_links": ["left_foot_end_link"],
            "ik_match_table": {"hips_mean": ["hips_sphere", 100, 0]},
            "key_frame_config": {"hips_mean": {}},
            "knee_angle_offset_degrees": 15.0,
        }
        # Should pass with warnings (not errors)
        ok = validate_config("test_robot", config)
        assert ok

    def test_empty_contact_links(self):
        config = {
            "robot_xml_path": "asset/robot/g1_description/mjcf/g1.xml",
            "verbose": True,
            "render_debug": False,
            "keypoints_path": "output_data/keypoints/g1/test.pkl",
            "joints_limit_offset_degrees": {},
            "robot_links": {
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
            },
            "contact_links": [],
            "ik_match_table": {"hips_mean": ["hips_sphere", 100, 0]},
            "key_frame_config": {"hips_mean": {}},
            "knee_angle_offset_degrees": 15.0,
        }
        ok = validate_config("test_robot", config)
        assert not ok


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
