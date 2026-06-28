#!/usr/bin/env python3
"""Unit tests for retarget_utils.py shared utility functions."""

import sys
import os
from pathlib import Path

import numpy as np
import pytest

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from retarget_utils import (
    quat_xyzw_to_wxyz,
    quat_wxyz_to_xyzw,
    average_quaternions_wxyz,
    quat_rotate_vectors_wxyz,
    normalize_vectors,
    quat_from_two_vectors,
    quat_multiply_wxyz,
    quat_conjugate_wxyz,
    batch_rotation_between_vectors_wxyz,
    compute_windowed_point_speeds,
    compute_contact_sequence,
    apply_low_pass_filter,
    compute_contact_height_offsets,
    offset_keypoints_by_contact_height,
    compute_link_scale_factors,
    apply_link_scales_to_positions,
    scale_keypoint_frame_displacements,
    advance_frame_cursor,
    canonicalize_contact_name,
    derive_leg_body_chain_from_links,
)


# ── Quaternion Math Tests ─────────────────────────────────────────────────

class TestQuaternionConversions:
    def test_xyzw_to_wxyz_single(self):
        q_xyzw = np.array([0.0, 0.0, 0.0, 1.0])  # identity in xyzw
        q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
        np.testing.assert_allclose(q_wxyz, [1.0, 0.0, 0.0, 0.0])

    def test_wxyz_to_xyzw_single(self):
        q_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # identity in wxyz
        q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
        np.testing.assert_allclose(q_xyzw, [0.0, 0.0, 0.0, 1.0])

    def test_roundtrip(self):
        q_wxyz_orig = np.array([0.5, 0.5, 0.5, 0.5])
        q_xyzw = quat_wxyz_to_xyzw(q_wxyz_orig)
        q_wxyz_back = quat_xyzw_to_wxyz(q_xyzw)
        np.testing.assert_allclose(q_wxyz_orig, q_wxyz_back)

    def test_batch_conversion(self):
        q_xyzw = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ])
        q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
        assert q_wxyz.shape == (2, 4)
        np.testing.assert_allclose(q_wxyz[0], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(q_wxyz[1], [0.0, 1.0, 0.0, 0.0])


class TestAverageQuaternions:
    def test_same_quaternion(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        avg = average_quaternions_wxyz(q, q)
        np.testing.assert_allclose(avg, q, atol=1e-6)

    def test_opposite_signs(self):
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([-1.0, 0.0, 0.0, 0.0])
        avg = average_quaternions_wxyz(q1, q2)
        # Should handle sign ambiguity
        assert np.abs(np.linalg.norm(avg) - 1.0) < 1e-6


class TestQuatRotateVectors:
    def test_identity_rotation(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        v = np.array([1.0, 2.0, 3.0])
        result = quat_rotate_vectors_wxyz(q, v)
        np.testing.assert_allclose(result, v, atol=1e-6)

    def test_90deg_z_rotation(self):
        # 90 deg rotation about Z axis
        angle = np.pi / 2
        q = np.array([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)])
        v = np.array([1.0, 0.0, 0.0])
        result = quat_rotate_vectors_wxyz(q, v)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-6)


class TestNormalizeVectors:
    def test_already_normalized(self):
        v = np.array([1.0, 0.0, 0.0])
        result = normalize_vectors(v)
        np.testing.assert_allclose(result, v)

    def test_arbitrary_vector(self):
        v = np.array([3.0, 4.0, 0.0])
        result = normalize_vectors(v)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0)

    def test_zero_vector(self):
        v = np.array([0.0, 0.0, 0.0])
        result = normalize_vectors(v)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])


class TestQuatFromTwoVectors:
    def test_same_vector(self):
        v = np.array([1.0, 0.0, 0.0])
        q = quat_from_two_vectors(v, v)
        # Should be identity (or close)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_opposite_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        q = quat_from_two_vectors(v1, v2)
        # Should be a 180-degree rotation: w ≈ 0, |q| = 1
        assert abs(np.linalg.norm(q) - 1.0) < 1e-6
        assert abs(q[0]) < 1e-6  # w component should be ~0 for 180° rotation


class TestQuatMultiply:
    def test_identity(self):
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.5, 0.5, 0.5, 0.5])
        result = quat_multiply_wxyz(q_id, q)
        np.testing.assert_allclose(result, q, atol=1e-6)

    def test_double_rotation(self):
        # Two 90-degree Z rotations = 180-degree Z rotation
        angle = np.pi / 2
        q = np.array([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)])
        result = quat_multiply_wxyz(q, q)
        expected = np.array([np.cos(np.pi/2), 0.0, 0.0, np.sin(np.pi/2)])
        np.testing.assert_allclose(result, expected, atol=1e-6)


class TestQuatConjugate:
    def test_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        result = quat_conjugate_wxyz(q)
        np.testing.assert_allclose(result, q)

    def test_negates_vector_part(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        result = quat_conjugate_wxyz(q)
        np.testing.assert_allclose(result, [0.5, -0.5, -0.5, -0.5])


# ── Contact Detection Tests ───────────────────────────────────────────────

class TestWindowedPointSpeeds:
    def test_stationary_points(self):
        positions = np.zeros((10, 2, 3))
        speeds = compute_windowed_point_speeds(positions, fps=30.0, window=6)
        np.testing.assert_allclose(speeds, 0.0, atol=1e-6)

    def test_constant_velocity(self):
        T = 20
        t = np.arange(T)
        # Point moving at 1 m/s in x direction
        positions = np.zeros((T, 1, 3))
        positions[:, 0, 0] = t / 30.0  # 1 m/s at 30 fps
        speeds = compute_windowed_point_speeds(positions, fps=30.0, window=6)
        # Middle frames should have speed close to 1.0
        assert speeds[10, 0] > 0.5  # rough check

    def test_single_frame(self):
        positions = np.zeros((1, 2, 3))
        speeds = compute_windowed_point_speeds(positions, fps=30.0, window=6)
        np.testing.assert_allclose(speeds, 0.0, atol=1e-6)


class TestContactSequence:
    def test_no_contact_high_position(self):
        positions = np.zeros((10, 2, 3))
        positions[:, :, 2] = 1.0  # high above ground
        _, states = compute_contact_sequence(positions, fps=30.0)
        assert not np.any(states)

    def test_contact_low_and_slow(self):
        positions = np.zeros((10, 2, 3))
        positions[:, :, 2] = 0.01  # very low
        _, states = compute_contact_sequence(positions, fps=30.0)
        assert np.all(states)


class TestLowPassFilter:
    def test_no_filter(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_low_pass_filter(values, alpha=1.0)
        np.testing.assert_allclose(result, values)

    def test_heavy_filter(self):
        values = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        result = apply_low_pass_filter(values, alpha=0.1)
        # Should decay slowly
        assert result[-1] > 0.0
        assert result[-1] < values[0]


class TestContactHeightOffsets:
    def test_no_contacts(self):
        keypoints = np.zeros((10, 5, 3))
        contact_positions = np.zeros((10, 2, 3))
        contact_positions[:, :, 2] = 1.0  # high
        contact_states = np.zeros((10, 2), dtype=bool)
        names = [f"kp{i}" for i in range(5)]
        offsets = compute_contact_height_offsets(
            keypoints, names, ("kp0", "kp1"),
            contact_positions, contact_states, height_lpf_alpha=1.0
        )
        np.testing.assert_allclose(offsets, 0.0, atol=1e-6)

    def test_constant_ground(self):
        keypoints = np.zeros((10, 5, 3))
        keypoints[:, :, 2] = 0.5  # all at height 0.5
        contact_positions = np.zeros((10, 2, 3))
        contact_positions[:, :, 2] = 0.0  # ground at 0
        contact_states = np.ones((10, 2), dtype=bool)
        names = [f"kp{i}" for i in range(5)]
        offsets = compute_contact_height_offsets(
            keypoints, names, ("kp0", "kp1"),
            contact_positions, contact_states, height_lpf_alpha=1.0
        )
        np.testing.assert_allclose(offsets, 0.0, atol=1e-6)


class TestOffsetKeypointsByContactHeight:
    def test_no_change_when_no_contacts(self):
        keypoints = np.random.randn(10, 5, 3).astype(np.float32)
        contact_positions = np.zeros((10, 2, 3))
        contact_states = np.zeros((10, 2), dtype=bool)
        names = [f"kp{i}" for i in range(5)]
        adjusted, offsets = offset_keypoints_by_contact_height(
            keypoints, names, ("kp0", "kp1"),
            contact_positions, contact_states
        )
        np.testing.assert_allclose(adjusted, keypoints, atol=1e-5)


# ── Link Scaling Tests ────────────────────────────────────────────────────

class TestComputeLinkScaleFactors:
    def test_uniform_scale(self):
        robot_lengths = {"thigh": 0.5, "calf": 0.4}
        T = 10
        skeleton_lengths = {
            "thigh": np.full(T, 0.4),
            "calf": np.full(T, 0.3),
        }
        skeleton_links = {
            "thigh": ("hip", "knee"),
            "calf": ("knee", "ankle"),
        }
        scales = compute_link_scale_factors(robot_lengths, skeleton_lengths, skeleton_links)
        np.testing.assert_allclose(scales["thigh"], 0.5 / 0.4)
        np.testing.assert_allclose(scales["calf"], 0.4 / 0.3)


class TestApplyLinkScalesToPositions:
    def test_identity_scale(self):
        T, K = 5, 4
        positions = np.random.randn(T, K, 3).astype(np.float32)
        links = {
            "a": (0, 1),
            "b": (1, 2),
            "c": (2, 3),
        }
        scales = {
            "a": np.ones(T),
            "b": np.ones(T),
            "c": np.ones(T),
        }
        result = apply_link_scales_to_positions(positions, links, scales, root_keypoint=0)
        np.testing.assert_allclose(result, positions, atol=1e-5)

    def test_uniform_scale(self):
        T = 5
        positions = np.zeros((T, 3, 3))
        positions[:, 1, 0] = 1.0  # link a: length 1
        positions[:, 2, 0] = 2.0  # link b: length 1 (from 1 to 2)
        links = {"a": (0, 1), "b": (1, 2)}
        scales = {"a": np.full(T, 2.0), "b": np.full(T, 2.0)}
        result = apply_link_scales_to_positions(positions, links, scales, root_keypoint=0)
        # After scaling: link a should be length 2, link b should be length 2
        np.testing.assert_allclose(result[:, 1, 0], 2.0, atol=1e-5)
        np.testing.assert_allclose(result[:, 2, 0], 4.0, atol=1e-5)


class TestScaleKeypointFrameDisplacements:
    def test_no_scale(self):
        keypoints = np.random.randn(10, 5, 3).astype(np.float32)
        result = scale_keypoint_frame_displacements(keypoints, 1.0)
        np.testing.assert_allclose(result, keypoints, atol=1e-5)

    def test_zero_scale(self):
        keypoints = np.random.randn(10, 5, 3).astype(np.float32)
        result = scale_keypoint_frame_displacements(keypoints, 0.0)
        # All frames should collapse to first frame
        for t in range(1, 10):
            np.testing.assert_allclose(result[t], keypoints[0], atol=1e-5)


# ── Playback Helper Tests ─────────────────────────────────────────────────

class TestAdvanceFrameCursor:
    def test_forward(self):
        assert advance_frame_cursor(0, 1, 10, loop=False) == 1

    def test_loop_wrap(self):
        assert advance_frame_cursor(9, 1, 10, loop=True) == 0

    def test_clip_forward(self):
        assert advance_frame_cursor(9, 1, 10, loop=False) == 9

    def test_clip_backward(self):
        assert advance_frame_cursor(0, -1, 10, loop=False) == 0

    def test_loop_backward_wrap(self):
        assert advance_frame_cursor(0, -1, 10, loop=True) == 9


# ── Contact Name Canonicalization Tests ───────────────────────────────────

class TestCanonicalizeContactName:
    def test_left_toe(self):
        assert canonicalize_contact_name("left_toe_link") == "left_toe"
        assert canonicalize_contact_name("Left_Toe") == "left_toe"

    def test_right_foot_end(self):
        assert canonicalize_contact_name("left_foot_end_link") == "left_foot_end"
        assert canonicalize_contact_name("right_foot_end_link") == "right_foot_end"

    def test_hand(self):
        assert canonicalize_contact_name("left_wrist_yaw_link") == "left_hand"
        assert canonicalize_contact_name("right_hand") == "right_hand"

    def test_unknown(self):
        assert canonicalize_contact_name("some_random_body") == "some_random_body"


# ── Leg Chain Derivation Tests ────────────────────────────────────────────

class TestDeriveLegBodyChain:
    def test_standard_links(self):
        links = {
            "left_hip": ("hips", "left_hip_roll"),
            "left_thigh": ("left_hip_roll", "left_knee"),
            "left_calf": ("left_knee", "left_ankle"),
        }
        chain = derive_leg_body_chain_from_links(links)
        assert chain == ("hips", "left_hip_roll", "left_knee", "left_ankle")

    def test_missing_links(self):
        links = {
            "left_hip": ("hips", "left_hip_roll"),
        }
        chain = derive_leg_body_chain_from_links(links)
        assert chain == ("hips", "left_hip_roll")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
