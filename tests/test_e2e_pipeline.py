#!/usr/bin/env python3
"""End-to-end pipeline tests for robot_retargeter.

These tests verify the full pipeline works correctly on sample data.
They are marked with pytest.mark.e2e and can be skipped in quick test runs.

Usage:
    pytest tests/ -v                    # run all tests
    pytest tests/ -v -m "not e2e"      # skip e2e tests
    pytest tests/ -v -m e2e            # run only e2e tests
"""

import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from retarget_utils import (
    compute_contact_sequence,
    compute_windowed_point_speeds,
    apply_link_scales_to_positions,
    scale_keypoint_frame_displacements,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_keypoints():
    """Generate synthetic keypoint data for testing."""
    T, K = 100, 15  # 100 frames, 15 keypoints
    np.random.seed(42)
    positions = np.random.randn(T, K, 3).astype(np.float32)
    # Make root keypoint follow a smooth trajectory
    positions[:, 0, 0] = np.sin(np.linspace(0, 4*np.pi, T)) * 0.5
    positions[:, 0, 1] = np.cos(np.linspace(0, 4*np.pi, T)) * 0.3
    positions[:, 0, 2] = 0.9  # standing height
    return positions


@pytest.fixture
def sample_contact_positions():
    """Generate synthetic contact point data."""
    T = 100
    positions = np.zeros((T, 4, 3), dtype=np.float32)
    # Feet at ground level
    positions[:, 0, 2] = 0.02  # left foot end
    positions[:, 1, 2] = 0.03  # left toe
    positions[:, 2, 2] = 0.02  # right foot end
    positions[:, 3, 2] = 0.03  # right toe
    return positions


@pytest.fixture
def sample_skeleton_links():
    """Standard skeleton link definitions."""
    return {
        "left_hip": (0, 1),
        "left_thigh": (1, 4),
        "left_calf": (4, 7),
        "left_foot": (7, 10),
        "right_hip": (0, 2),
        "right_thigh": (2, 5),
        "right_calf": (5, 8),
        "right_foot": (8, 11),
        "neck": (0, 12),
        "head": (12, 15) if 15 < 15 else (12, 13),
        "left_shoulder": (12, 13),
        "left_arm": (13, 16) if 16 < 15 else (13, 14),
        "left_fore_arm": (14, 17) if 17 < 15 else (14, 13),
        "right_shoulder": (12, 14),
        "right_arm": (14, 18) if 18 < 15 else (14, 13),
        "right_fore_arm": (13, 19) if 19 < 15 else (13, 14),
    }


@pytest.fixture
def sample_robot_links():
    """Standard robot link definitions (body name pairs)."""
    return {
        "left_hip": ("hips_sphere", "left_hip_roll_link"),
        "left_thigh": ("left_hip_roll_link", "left_knee_link"),
        "left_calf": ("left_knee_link", "left_ankle_roll_link"),
        "right_hip": ("hips_sphere", "right_hip_roll_link"),
        "right_thigh": ("right_hip_roll_link", "right_knee_link"),
        "right_calf": ("right_knee_link", "right_ankle_roll_link"),
        "neck": ("hips_sphere", "neck_sphere"),
        "head": ("neck_sphere", "head_sphere"),
        "left_shoulder": ("neck_sphere", "left_shoulder_roll_link"),
        "left_arm": ("left_shoulder_roll_link", "left_elbow_link"),
        "left_fore_arm": ("left_elbow_link", "left_wrist_yaw_link"),
        "right_shoulder": ("neck_sphere", "right_shoulder_roll_link"),
        "right_arm": ("right_shoulder_roll_link", "right_elbow_link"),
        "right_fore_arm": ("right_elbow_link", "right_wrist_yaw_link"),
    }


# ── E2E Pipeline Tests ────────────────────────────────────────────────────

class TestPipelineStages:
    """Test each pipeline stage with synthetic data."""

    @pytest.mark.e2e
    def test_contact_detection_pipeline(self, sample_contact_positions):
        """Test contact detection on synthetic data."""
        speeds, states = compute_contact_sequence(
            sample_contact_positions,
            fps=30.0,
            vel_window=6,
            vel_threshold=0.5,
            height_threshold=0.075,
        )
        assert speeds.shape == (100, 4)
        assert states.shape == (100, 4)
        # All points are low and stationary → should be in contact
        assert np.all(states)

    @pytest.mark.e2e
    def test_link_scaling_pipeline(self, sample_keypoints, sample_skeleton_links):
        """Test link scaling on synthetic data."""
        # Filter to only valid indices
        valid_links = {}
        for name, (parent, child) in sample_skeleton_links.items():
            if parent < sample_keypoints.shape[1] and child < sample_keypoints.shape[1]:
                valid_links[name] = (parent, child)

        T = sample_keypoints.shape[0]
        scales = {name: np.full(T, 1.2) for name in valid_links}

        scaled = apply_link_scales_to_positions(
            sample_keypoints, valid_links, scales, root_keypoint=0
        )
        assert scaled.shape == sample_keypoints.shape
        # Root should be unchanged
        np.testing.assert_allclose(scaled[:, 0], sample_keypoints[:, 0])

    @pytest.mark.e2e
    def test_displacement_scaling_pipeline(self, sample_keypoints):
        """Test displacement scaling on synthetic data."""
        scaled = scale_keypoint_frame_displacements(sample_keypoints, 0.5)
        assert scaled.shape == sample_keypoints.shape
        # First frame should be unchanged
        np.testing.assert_allclose(scaled[0], sample_keypoints[0])

    @pytest.mark.e2e
    def test_full_keypoint_processing(self, sample_keypoints, sample_contact_positions):
        """Test the full keypoint processing pipeline."""
        # 1. Compute contact
        speeds, states = compute_contact_sequence(
            sample_contact_positions, fps=30.0
        )

        # 2. Scale displacements
        scaled = scale_keypoint_frame_displacements(sample_keypoints, 0.8)

        # 3. Verify output is valid
        assert not np.any(np.isnan(scaled))
        assert not np.any(np.isinf(scaled))
        assert scaled.shape == sample_keypoints.shape


# ── Data Format Tests ─────────────────────────────────────────────────────

class TestDataFormats:
    """Test data loading and saving formats."""

    @pytest.mark.e2e
    def test_keypoints_pkl_roundtrip(self, sample_keypoints, tmp_path):
        """Test that keypoints can be saved and loaded from pkl."""
        data = {
            "positions": sample_keypoints,
            "quaternions": np.random.randn(*sample_keypoints.shape[:2], 4).astype(np.float32),
            "names": [f"kp{i}" for i in range(sample_keypoints.shape[1])],
            "fps": 30.0,
        }
        pkl_path = tmp_path / "test_keypoints.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)

        np.testing.assert_allclose(loaded["positions"], sample_keypoints)
        assert loaded["fps"] == 30.0

    @pytest.mark.e2e
    def test_motion_csv_format(self, tmp_path):
        """Test motion CSV format (qpos: pos3 + quat4 + joints)."""
        T = 50
        num_joints = 20
        # qpos: xyz(3) + quat(4) + joints
        qpos = np.random.randn(T, 7 + num_joints).astype(np.float32)
        # Normalize quaternion
        qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)

        csv_path = tmp_path / "test_motion.csv"
        np.savetxt(csv_path, qpos, delimiter=",")

        loaded = np.loadtxt(csv_path, delimiter=",")
        np.testing.assert_allclose(loaded, qpos, atol=1e-5)


# ── Config Validation E2E ────────────────────────────────────────────────

class TestConfigValidation:
    """Test that all robot configs pass validation."""

    @pytest.mark.e2e
    def test_all_robot_configs_valid(self):
        """Verify all robot YAML configs pass validation."""
        config_dir = PROJECT_DIR / "config" / "robot"
        if not config_dir.exists():
            pytest.skip("Config directory not found")

        yaml_files = list(config_dir.glob("*.yaml"))
        assert len(yaml_files) > 0, "No robot configs found"

        from validate_configs import validate_config
        import yaml

        failures = []
        for yaml_file in yaml_files:
            with open(yaml_file, "r") as f:
                config = yaml.safe_load(f)
            ok = validate_config(yaml_file.stem, config)
            if not ok:
                failures.append(yaml_file.stem)

        assert not failures, f"Config validation failed for: {failures}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
