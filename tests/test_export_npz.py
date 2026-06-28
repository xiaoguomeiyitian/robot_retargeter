"""Tests for export_npz.py — NPZ export for RL training."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure scripts/ is importable
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import export_npz  # noqa: E402


class TestResampleMotion:
    """Tests for the resample_motion function."""

    def test_same_fps_no_change(self):
        """Resampling to same fps should preserve frame count (within 1 frame)."""
        pos = np.random.randn(100, 3)
        rot = np.tile([1, 0, 0, 0], (100, 1))  # identity quaternions (wxyz)
        joints = np.random.randn(100, 29)

        p, r, j = export_npz.resample_motion(pos, rot, joints, 30.0, 30.0)
        # Allow ±1 frame due to floating-point in arange
        assert abs(p.shape[0] - 100) <= 1
        assert abs(r.shape[0] - 100) <= 1
        assert abs(j.shape[0] - 100) <= 1

    def test_upsample(self):
        """Upsampling should increase frame count."""
        pos = np.random.randn(100, 3)
        rot = np.tile([1, 0, 0, 0], (100, 1))
        joints = np.random.randn(100, 29)

        p, r, j = export_npz.resample_motion(pos, rot, joints, 30.0, 50.0)
        # 100 frames @ 30fps = 3.33s → ~167 frames @ 50fps
        assert p.shape[0] > 100
        assert abs(p.shape[0] - 167) <= 2

    def test_downsample(self):
        """Downsampling should decrease frame count."""
        pos = np.random.randn(200, 3)
        rot = np.tile([1, 0, 0, 0], (200, 1))
        joints = np.random.randn(200, 29)

        p, r, j = export_npz.resample_motion(pos, rot, joints, 50.0, 30.0)
        assert p.shape[0] < 200

    def test_output_shapes(self):
        """Output shapes should match expected dimensions."""
        pos = np.random.randn(50, 3)
        rot = np.tile([1, 0, 0, 0], (50, 1))
        joints = np.random.randn(50, 29)

        p, r, j = export_npz.resample_motion(pos, rot, joints, 30.0, 50.0)
        assert p.shape[1] == 3
        assert r.shape[1] == 4
        assert j.shape[1] == 29

    def test_quaternion_normalization(self):
        """Output quaternions should be unit quaternions."""
        pos = np.random.randn(50, 3)
        rot = np.tile([1, 0, 0, 0], (50, 1))
        joints = np.random.randn(50, 29)

        _, r, _ = export_npz.resample_motion(pos, rot, joints, 30.0, 50.0)
        norms = np.linalg.norm(r, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


class TestComputeVelocities:
    """Tests for the compute_velocities function."""

    def test_constant_position_zero_velocity(self):
        """Constant position should yield zero velocity."""
        pos = np.tile([1.0, 2.0, 3.0], (100, 1))
        rot = np.tile([1, 0, 0, 0], (100, 1))
        joints = np.tile([0.5] * 29, (100, 1))

        lin_vel, ang_vel, joint_vel = export_npz.compute_velocities(
            pos, rot, joints, 1.0 / 50.0
        )
        np.testing.assert_allclose(lin_vel, 0.0, atol=1e-5)
        np.testing.assert_allclose(ang_vel, 0.0, atol=1e-5)
        np.testing.assert_allclose(joint_vel, 0.0, atol=1e-5)

    def test_linear_position_constant_velocity(self):
        """Linearly increasing position should yield constant velocity."""
        t = np.arange(100) * 0.02
        pos = np.column_stack([t, 2 * t, 3 * t])
        rot = np.tile([1, 0, 0, 0], (100, 1))
        joints = np.zeros((100, 29))

        lin_vel, _, _ = export_npz.compute_velocities(pos, rot, joints, 0.02)
        # np.gradient of t with dt=0.02 → 0.02/0.02 = 1.0
        # So velocity = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(lin_vel[50], [1.0, 2.0, 3.0], atol=0.1)

    def test_output_shapes(self):
        """Velocity shapes should match position shapes."""
        pos = np.random.randn(100, 3)
        rot = np.tile([1, 0, 0, 0], (100, 1))
        joints = np.random.randn(100, 29)

        lin_vel, ang_vel, joint_vel = export_npz.compute_velocities(
            pos, rot, joints, 0.02
        )
        assert lin_vel.shape == (100, 3)
        assert ang_vel.shape == (100, 3)
        assert joint_vel.shape == (100, 29)


class TestExportCsvToNpz:
    """Integration tests for the full CSV→NPZ export."""

    @pytest.fixture()
    def sample_csv(self, tmp_path: Path) -> str:
        """Create a sample CSV file mimicking retarget output."""
        num_frames = 300
        # xyz(3) + quat_xyzw(4) + joints(29) = 36 columns
        data = np.zeros((num_frames, 36))

        # Base position: slow sinusoidal motion
        t = np.linspace(0, 4 * np.pi, num_frames)
        data[:, 0] = np.sin(t) * 0.1  # x
        data[:, 1] = np.cos(t) * 0.05  # y
        data[:, 2] = 0.8 + np.sin(2 * t) * 0.02  # z (height)

        # Quaternion: identity (wxyz stored as xyzw in CSV)
        data[:, 3] = 1.0  # w (stored in column 3 for xyzw format)
        data[:, 4] = 0.0  # x
        data[:, 5] = 0.0  # y
        data[:, 6] = 0.0  # z

        # Joints: small sinusoidal motions
        for j in range(29):
            data[:, 7 + j] = np.sin(t * (j + 1) * 0.1) * 0.3

        csv_path = str(tmp_path / "test_motion.csv")
        np.savetxt(csv_path, data, delimiter=",")
        return csv_path

    def test_export_creates_npz(self, sample_csv: str, tmp_path: Path):
        """Export should create an NPZ file."""
        output_path = str(tmp_path / "output.npz")
        export_npz.export_csv_to_npz(
            csv_path=sample_csv,
            output_path=output_path,
            robot_name="g1",
            input_fps=30.0,
            output_fps=50.0,
        )
        assert os.path.exists(output_path)

    def test_npz_has_required_keys(self, sample_csv: str, tmp_path: Path):
        """NPZ should contain all 7 required keys."""
        required_keys = {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }
        output_path = str(tmp_path / "output.npz")
        export_npz.export_csv_to_npz(
            csv_path=sample_csv,
            output_path=output_path,
            robot_name="g1",
            input_fps=30.0,
            output_fps=50.0,
        )
        data = np.load(output_path)
        assert set(data.keys()) == required_keys

    def test_npz_shapes(self, sample_csv: str, tmp_path: Path):
        """NPZ arrays should have correct shapes."""
        output_path = str(tmp_path / "output.npz")
        export_npz.export_csv_to_npz(
            csv_path=sample_csv,
            output_path=output_path,
            robot_name="g1",
            input_fps=30.0,
            output_fps=50.0,
        )
        data = np.load(output_path)
        T = data["joint_pos"].shape[0]
        assert T > 0
        assert data["joint_pos"].shape == (T, 29)
        assert data["joint_vel"].shape == (T, 29)
        assert data["body_pos_w"].shape[0] == T
        assert data["body_pos_w"].shape[2] == 3
        assert data["body_quat_w"].shape[0] == T
        assert data["body_quat_w"].shape[2] == 4
        assert data["body_lin_vel_w"].shape[0] == T
        assert data["body_lin_vel_w"].shape[2] == 3
        assert data["body_ang_vel_w"].shape[0] == T
        assert data["body_ang_vel_w"].shape[2] == 3

    def test_npz_fps(self, sample_csv: str, tmp_path: Path):
        """NPZ should store the correct fps value."""
        output_path = str(tmp_path / "output.npz")
        export_npz.export_csv_to_npz(
            csv_path=sample_csv,
            output_path=output_path,
            robot_name="g1",
            input_fps=30.0,
            output_fps=50.0,
        )
        data = np.load(output_path)
        assert data["fps"][0] == pytest.approx(50.0)

    def test_npz_quaternions_normalized(self, sample_csv: str, tmp_path: Path):
        """Body quaternions should be unit quaternions."""
        output_path = str(tmp_path / "output.npz")
        export_npz.export_csv_to_npz(
            csv_path=sample_csv,
            output_path=output_path,
            robot_name="g1",
            input_fps=30.0,
            output_fps=50.0,
        )
        data = np.load(output_path)
        quats = data["body_quat_w"]
        norms = np.linalg.norm(quats, axis=2)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_npz_float32(self, sample_csv: str, tmp_path: Path):
        """All arrays should be float32."""
        output_path = str(tmp_path / "output.npz")
        export_npz.export_csv_to_npz(
            csv_path=sample_csv,
            output_path=output_path,
            robot_name="g1",
            input_fps=30.0,
            output_fps=50.0,
        )
        data = np.load(output_path)
        for key in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                     "body_lin_vel_w", "body_ang_vel_w", "fps"]:
            assert data[key].dtype == np.float32, f"{key} is {data[key].dtype}, expected float32"


class TestRobotConfigs:
    """Tests for ROBOT_CONFIGS dictionary."""

    def test_g1_config(self):
        """G1 config should have 29 joints."""
        cfg = export_npz.ROBOT_CONFIGS["g1"]
        assert cfg["num_joints"] == 29
        assert len(cfg["joint_names"]) == 29
        assert len(cfg["body_names"]) > 0

    def test_g1_23dof_config(self):
        """G1 23dof config should have 23 joints."""
        cfg = export_npz.ROBOT_CONFIGS["g1_23dof"]
        assert cfg["num_joints"] == 23
        assert len(cfg["joint_names"]) == 23

    def test_joint_names_unique(self):
        """Joint names should be unique within each config."""
        for name, cfg in export_npz.ROBOT_CONFIGS.items():
            assert len(cfg["joint_names"]) == len(set(cfg["joint_names"])), \
                f"{name} has duplicate joint names"

    def test_body_names_unique(self):
        """Body names should be unique within each config."""
        for name, cfg in export_npz.ROBOT_CONFIGS.items():
            assert len(cfg["body_names"]) == len(set(cfg["body_names"])), \
                f"{name} has duplicate body names"
