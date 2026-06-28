#!/usr/bin/env python3
"""Shared utility functions for the retargeting pipeline.

This module contains common functions used by both smpl_replay.py and
robot_replay.py, eliminating code duplication.

Functions are grouped into categories:
  - Quaternion math
  - Configuration loading
  - Contact detection & height adjustment
  - Link geometry & scaling
  - Model loading
  - Keyframe adjustment
  - Playback helpers
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml


# ── Quaternion Math ────────────────────────────────────────────────────────

def quat_xyzw_to_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """Convert quaternion array from XYZW to WXYZ order (MuJoCo convention)."""
    out = np.empty_like(quaternions)
    out[..., 0] = quaternions[..., 3]  # w
    out[..., 1] = quaternions[..., 0]  # x
    out[..., 2] = quaternions[..., 1]  # y
    out[..., 3] = quaternions[..., 2]  # z
    return out


def quat_wxyz_to_xyzw(quaternions: np.ndarray) -> np.ndarray:
    """Convert quaternion array from WXYZ to XYZW order."""
    out = np.empty_like(quaternions)
    out[..., 0] = quaternions[..., 1]  # x
    out[..., 1] = quaternions[..., 2]  # y
    out[..., 2] = quaternions[..., 3]  # z
    out[..., 3] = quaternions[..., 0]  # w
    return out


def average_quaternions_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compute the average of two WXYZ quaternions using the midpoint on the geodesic."""
    dot = np.sum(q1 * q2, axis=-1, keepdims=True)
    sign = np.where(dot < 0, -1.0, 1.0)
    avg = q1 + sign * q2
    norm = np.linalg.norm(avg, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return avg / norm


def quat_rotate_vectors_wxyz(quaternions: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a vector by a WXYZ quaternion (or array of quaternions).

    Args:
        quaternions: (..., 4) WXYZ quaternion array
        vector: (3,) or (..., 3) vector to rotate
    Returns:
        Rotated vector(s) with same shape as input vector
    """
    q = quaternions[..., 1:4]  # xyz part
    w = quaternions[..., 0:1]  # w part
    # t = 2 * cross(q, v)
    t = 2.0 * np.cross(q, vector)
    # result = v + w * t + cross(q, t)
    return vector + w * t + np.cross(q, t)


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize an array of vectors along the last axis."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return vectors / norms


def quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Compute the WXYZ quaternion that rotates v_from to v_to.

    Handles the degenerate case where v_from and v_to are opposite
    by choosing an arbitrary perpendicular axis for the 180° rotation.
    """
    v_from = normalize_vectors(v_from)
    v_to = normalize_vectors(v_to)
    dot = np.sum(v_from * v_to, axis=-1, keepdims=True)
    cross = np.cross(v_from, v_to)
    w = 1.0 + dot
    q_xyz = cross

    # Handle opposite vectors: dot ≈ -1, cross ≈ 0
    # Choose a perpendicular axis for 180° rotation
    w_val = float(np.asarray(w).squeeze())
    if w_val < 1e-6:
        # Find a non-parallel axis
        vf = np.asarray(v_from).flatten()[:3]
        if abs(vf[0]) < 0.9:
            perp = np.cross(vf, np.array([1.0, 0.0, 0.0]))
        else:
            perp = np.cross(vf, np.array([0.0, 1.0, 0.0]))
        norm_perp = np.linalg.norm(perp)
        if norm_perp < 1e-8:
            perp = np.array([0.0, 0.0, 1.0])
        else:
            perp = perp / norm_perp
        return np.array([0.0, perp[0], perp[1], perp[2]])

    q = np.concatenate([w, q_xyz], axis=-1)
    return normalize_vectors(q)


def quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two WXYZ quaternions: result = q1 * q2."""
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
    return np.concatenate([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def quat_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Compute the conjugate of a WXYZ quaternion."""
    out = quaternion.copy()
    out[..., 1:] *= -1
    return out


def quat_rotate_vector_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a single vector by a single WXYZ quaternion."""
    q = quaternion[..., 1:4]
    w = quaternion[..., 0:1]
    t = 2.0 * np.cross(q, vector)
    return vector + w * t + np.cross(q, t)


def multiply_quaternions_wxyz(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    """Alias for quat_multiply_wxyz."""
    return quat_multiply_wxyz(q_left, q_right)


def batch_rotation_between_vectors_wxyz(
    v_from: np.ndarray, v_to: np.ndarray
) -> np.ndarray:
    """Compute WXYZ quaternions that rotate v_from to v_to (per-frame)."""
    v_from = normalize_vectors(v_from)
    v_to = normalize_vectors(v_to)
    dot = np.sum(v_from * v_to, axis=-1, keepdims=True)
    cross = np.cross(v_from, v_to)
    w = 1.0 + dot
    q = np.concatenate([w, cross], axis=-1)
    return normalize_vectors(q)


# ── Configuration Loading ──────────────────────────────────────────────────

def load_yaml_body_list_config(config_path: Path, field_name: str) -> tuple[str, ...]:
    """Load a list of body names from a YAML config field."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    value = config.get(field_name, [])
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",")]
    return tuple(value)


def load_scalar_float_config(config_path: Path, field_name: str, default: float | None = None) -> float:
    """Load a scalar float from a YAML config field."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    value = config.get(field_name, default)
    if value is None:
        raise KeyError(f"Missing required config field: {field_name} in {config_path}")
    return float(value)


def load_scalar_int_config(config_path: Path, field_name: str, default: int | None = None) -> int:
    """Load a scalar int from a YAML config field."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    value = config.get(field_name, default)
    if value is None:
        raise KeyError(f"Missing required config field: {field_name} in {config_path}")
    return int(value)


def load_scalar_bool_config(config_path: Path, field_name: str, default: bool = False) -> bool:
    """Load a scalar bool from a YAML config field."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return bool(config.get(field_name, default))


def load_path_config(config_path: Path, field_name: str, default: Path | None = None) -> Path:
    """Load a path from a YAML config field."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    value = config.get(field_name, default)
    if value is None:
        raise KeyError(f"Missing required config field: {field_name} in {config_path}")
    return Path(value)


def load_link_pairs_config(config_path: Path, section_name: str) -> dict[str, tuple[str, str]]:
    """Load link pairs (parent, child) from a YAML config section."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    section = config.get(section_name, {})
    result = {}
    for link_name, pair in section.items():
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            result[link_name] = (pair[0], pair[1])
    return result


def load_robot_links_config(config_path: Path) -> dict[str, tuple[str, str]]:
    """Load robot_links from a YAML config."""
    return load_link_pairs_config(config_path, "robot_links")


def load_skeleton_links_config(config_path: Path) -> dict[str, tuple[str, str]]:
    """Load skeleton_links from a YAML config."""
    return load_link_pairs_config(config_path, "skeleton_links")


def load_body_chain_config(config_path: Path, field_name: str, expected_len: int) -> tuple[str, ...]:
    """Load a body chain (ordered list of body names) from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    value = config.get(field_name, [])
    if len(value) != expected_len:
        raise ValueError(
            f"Expected {expected_len} bodies in '{field_name}', got {len(value)} in {config_path}"
        )
    return tuple(value)


def load_body_list_config(config_path: Path, field_name: str) -> tuple[str, ...]:
    """Load a list of body names from YAML (alias for load_yaml_body_list_config)."""
    return load_yaml_body_list_config(config_path, field_name)


def load_euler_offset_map_config(config_path: Path, section_name: str) -> dict[str, np.ndarray]:
    """Load Euler offset map from YAML (used in key_frame_config)."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    section = config.get(section_name, {})
    result = {}
    for body_name, body_cfg in section.items():
        offset = body_cfg.get("offset_deg_xyz", [0.0, 0.0, 0.0])
        result[body_name] = np.array(offset, dtype=np.float64)
    return result


def _parse_float_list(value: str, expected_len: int, context: str) -> np.ndarray:
    """Parse a string of comma-separated floats into a numpy array."""
    parts = [v.strip() for v in value.split(",") if v.strip()]
    if len(parts) != expected_len:
        raise ValueError(
            f"{context}: expected {expected_len} values, got {len(parts)}: '{value}'"
        )
    return np.array([float(v) for v in parts], dtype=np.float64)


def load_key_frame_config(config_path: Path) -> dict[str, dict[str, Any]]:
    """Load key_frame_config from YAML, parsing axis_map and offset_deg_xyz."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    section = config.get("key_frame_config", {})
    result = {}
    for body_name, body_cfg in section.items():
        offset = body_cfg.get("offset_deg_xyz", [0.0, 0.0, 0.0])
        axis_map = body_cfg.get("axis_map_cols", {})
        result[body_name] = {
            "offset_deg_xyz": np.array(offset, dtype=np.float64),
            "axis_map_cols": {
                axis: np.array(cols, dtype=np.float64)
                for axis, cols in axis_map.items()
            },
        }
    return result


# ── Contact Detection & Height Adjustment ──────────────────────────────────

def compute_windowed_point_speeds(
    positions: np.ndarray,
    fps: float,
    window: int = 6,
    progress_desc: str = "Contact speed",
) -> np.ndarray:
    """Compute windowed speeds for contact detection (vectorized).

    Args:
        positions: (T, K, 3) array of contact point positions
        fps: frames per second
        window: window size for speed calculation
        progress_desc: description for progress bar (unused, kept for API compat)
    Returns:
        (T, K) array of speeds
    """
    T = positions.shape[0]
    if T <= 1:
        return np.zeros((T, positions.shape[1]), dtype=np.float32)

    # Use convolution for vectorized speed computation
    half_w = max(window // 2, 1)
    speeds = np.zeros_like(positions[:, :, 0])

    for i in range(T):
        lo = max(0, i - half_w)
        hi = min(T, i + half_w + 1)
        if hi - lo < 2:
            speeds[i] = 0.0
            continue
        dt = (hi - lo - 1) / fps
        if dt < 1e-8:
            speeds[i] = 0.0
            continue
        disp = positions[hi-1] - positions[lo]
        speeds[i] = np.linalg.norm(disp, axis=-1) / dt

    return speeds.astype(np.float32)


def compute_contact_sequence(
    contact_positions: np.ndarray,
    fps: float,
    vel_window: int = 6,
    vel_threshold: float = 0.5,
    height_threshold: float = 0.075,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute contact states from positions using speed + height threshold.

    Args:
        contact_positions: (T, K, 3) contact point positions
        fps: frames per second
        vel_window: window for speed calculation
        vel_threshold: speed threshold for contact
        height_threshold: height threshold for contact
    Returns:
        (contact_speeds, contact_states) both (T, K) arrays
    """
    contact_speeds = compute_windowed_point_speeds(
        contact_positions, fps=fps, window=vel_window
    )
    contact_states = np.logical_and(
        contact_speeds <= float(vel_threshold),
        contact_positions[:, :, 2] <= float(height_threshold),
    )
    return contact_speeds.astype(np.float32), contact_states.astype(np.bool_)


def apply_low_pass_filter(values: np.ndarray, alpha: float) -> np.ndarray:
    """Apply a first-order IIR low-pass filter along axis 0.

    y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
    """
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def compute_contact_height_offsets(
    keypoints: np.ndarray,
    keypoint_names: list[str],
    contact_names: tuple[str, ...],
    contact_positions: np.ndarray,
    contact_states: np.ndarray,
    height_lpf_alpha: float = 1.0,
) -> np.ndarray:
    """Compute per-frame ground height offsets from active contacts.

    Args:
        keypoints: (T, K, 3) full keypoint array
        keypoint_names: list of keypoint names (length K)
        contact_names: tuple of contact keypoint names to use as ground reference
        contact_positions: (T, C, 3) contact point positions
        contact_states: (T, C) boolean contact states
        height_lpf_alpha: low-pass filter alpha (1.0 = no filtering)
    Returns:
        (T,) array of height offsets
    """
    T = keypoints.shape[0]
    contact_indices = [keypoint_names.index(n) for n in contact_names if n in keypoint_names]

    height_offsets = np.zeros(T, dtype=np.float64)
    prev_height = 0.0

    for t in range(T):
        active_heights = contact_positions[t, contact_states[t], 2]
        if len(active_heights) > 0:
            height_offsets[t] = float(np.min(active_heights))
            prev_height = height_offsets[t]
        else:
            height_offsets[t] = prev_height

    if height_lpf_alpha < 1.0:
        height_offsets = apply_low_pass_filter(height_offsets, height_lpf_alpha)

    return height_offsets.astype(np.float32)


def offset_keypoints_by_contact_height(
    keypoints: np.ndarray,
    keypoint_names: list[str],
    contact_names: tuple[str, ...],
    contact_positions: np.ndarray,
    contact_states: np.ndarray,
    height_lpf_alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Adjust keypoints by subtracting contact height offsets (z only).

    Returns:
        (adjusted_keypoints, height_offsets)
    """
    height_offsets = compute_contact_height_offsets(
        keypoints=keypoints,
        keypoint_names=keypoint_names,
        contact_names=contact_names,
        contact_positions=contact_positions,
        contact_states=contact_states,
        height_lpf_alpha=height_lpf_alpha,
    )
    adjusted_keypoints = keypoints.copy()
    adjusted_keypoints[:, :, 2] -= height_offsets[:, None]
    return adjusted_keypoints.astype(np.float32), height_offsets.astype(np.float32)


# ── Link Geometry & Scaling ───────────────────────────────────────────────

def compute_robot_link_lengths(
    robot_mjcf: mujoco.MjModel,
    robot_links: dict[str, tuple[str, str]],
) -> dict[str, float]:
    """Compute link lengths from a robot model at zero pose.

    Args:
        robot_mjcf: compiled MuJoCo model
        robot_links: {link_name: (parent_body, child_body)}
    Returns:
        {link_name: length}
    """
    data = mujoco.MjData(robot_mjcf)
    mujoco.mj_forward(robot_mjcf, data)
    lengths = {}
    for link_name, (parent, child) in robot_links.items():
        parent_id = mujoco.mj_name2id(robot_mjcf, mujoco.mjtObj.mjOBJ_BODY, parent)
        child_id = mujoco.mj_name2id(robot_mjcf, mujoco.mjtObj.mjOBJ_BODY, child)
        if parent_id < 0 or child_id < 0:
            raise ValueError(f"Cannot find bodies for link '{link_name}': {parent}, {child}")
        lengths[link_name] = float(np.linalg.norm(data.xpos[child_id] - data.xpos[parent_id]))
    return lengths


def compute_robot_body_local_offset(
    robot_mjcf_path: str | Path,
    anchor_body: str,
    target_body: str,
) -> np.ndarray:
    """Compute the local offset from anchor_body to target_body at zero pose.

    Returns:
        (3,) offset vector in the anchor body's local frame
    """
    spec = mujoco.MjSpec.from_file(str(robot_mjcf_path))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    anchor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_body)
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body)

    if anchor_id < 0:
        raise ValueError(f"Body '{anchor_body}' not found in model")
    if target_id < 0:
        raise ValueError(f"Body '{target_body}' not found in model")

    world_offset = data.xpos[target_id] - data.xpos[anchor_id]
    anchor_quat = data.xquat[anchor_id]  # wxyz
    # Rotate world offset into anchor's local frame
    anchor_rot = anchor_quat[1:4]
    anchor_w = anchor_quat[0]
    # inverse rotation: conjugate
    t = 2.0 * np.cross(anchor_rot, world_offset)
    local_offset = world_offset + anchor_w * t - np.cross(anchor_rot, t)
    return local_offset.astype(np.float32)


def compute_link_geometry_from_positions(
    skeleton_positions: np.ndarray,
    skeleton_links: dict[str, tuple[str, str]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute per-frame link vectors and lengths from keypoint positions.

    Args:
        skeleton_positions: (T, K, 3) keypoint positions
        skeleton_links: {link_name: (parent_keypoint, child_keypoint)}
    Returns:
        (link_vectors, link_lengths) both {link_name: (T, 3)} and {link_name: (T,)}
    """
    link_vectors = {}
    link_lengths = {}
    for link_name, (parent, child) in skeleton_links.items():
        vec = skeleton_positions[:, child] - skeleton_positions[:, parent]
        link_vectors[link_name] = vec
        link_lengths[link_name] = np.linalg.norm(vec, axis=-1)
    return link_vectors, link_lengths


def compute_link_scale_factors(
    robot_lengths: dict[str, float],
    skeleton_lengths: dict[str, np.ndarray],
    skeleton_links: dict[str, tuple[str, str]],
) -> dict[str, np.ndarray]:
    """Compute per-frame scale factors (robot_length / skeleton_length).

    Returns:
        {link_name: (T,)} scale factors
    """
    scales = {}
    for link_name in skeleton_links:
        if link_name in robot_lengths and link_name in skeleton_lengths:
            scales[link_name] = robot_lengths[link_name] / np.maximum(
                skeleton_lengths[link_name], 1e-8
            )
    return scales


def compute_leg_displacement_scale(
    robot_mjcf: mujoco.MjModel,
    robot_links: dict[str, tuple[str, str]],
    skeleton_positions: np.ndarray,
    skeleton_links: dict[str, tuple[str, str]],
) -> float:
    """Compute a global displacement scale from leg length ratio.

    Leg length = thigh + calf (averaged over left/right).
    """
    def leg_len(model, links, left_parent, left_mid, left_child):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        p1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, links[left_parent][1])
        p2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, links[left_mid][1])
        p3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, links[left_child][1])
        return float(
            np.linalg.norm(data.xpos[p2] - data.xpos[p1]) +
            np.linalg.norm(data.xpos[p3] - data.xpos[p2])
        )

    try:
        robot_left = leg_len(robot_mjcf, robot_links, "left_hip", "left_thigh", "left_calf")
        robot_right = leg_len(robot_mjcf, robot_links, "right_hip", "right_thigh", "right_calf")
        robot_leg = (robot_left + robot_right) / 2.0
    except (KeyError, ValueError):
        return 1.0

    try:
        skel_thigh_len = np.mean(skeleton_lengths_from_positions(
            skeleton_positions, skeleton_links, "left_thigh"
        )) + np.mean(skeleton_lengths_from_positions(
            skeleton_positions, skeleton_links, "right_thigh"
        )) / 2.0
        skel_calf_len = np.mean(skeleton_lengths_from_positions(
            skeleton_positions, skeleton_links, "left_calf"
        )) + np.mean(skeleton_lengths_from_positions(
            skeleton_positions, skeleton_links, "right_calf"
        )) / 2.0
        skel_leg = skel_thigh_len + skel_calf_len
    except (KeyError, ValueError):
        return 1.0

    if skel_leg < 1e-8:
        return 1.0
    return robot_leg / skel_leg


def skeleton_lengths_from_positions(
    positions: np.ndarray,
    links: dict[str, tuple[str, str]],
    link_name: str,
) -> np.ndarray:
    """Helper: compute length of a specific link from positions."""
    parent, child = links[link_name]
    vec = positions[:, child] - positions[:, parent]
    return np.linalg.norm(vec, axis=-1)


def scale_keypoint_frame_displacements(
    keypoints: np.ndarray,
    displacement_scale: float,
) -> np.ndarray:
    """Scale root frame-to-frame displacements uniformly in x,y,z."""
    scaled = keypoints.copy()
    displacements = np.diff(keypoints, axis=0)
    displacements *= displacement_scale
    scaled[1:] = keypoints[0] + np.cumsum(displacements, axis=0)
    return scaled


def apply_link_scales_to_positions(
    skeleton_positions: np.ndarray,
    skeleton_links: dict[str, tuple[str, str]],
    scale_factors: dict[str, np.ndarray],
    root_keypoint: str = "hips_mean",
) -> np.ndarray:
    """Apply per-link scale factors to keypoint positions (parent→child traversal).

    Args:
        skeleton_positions: (T, K, 3) original keypoint positions
        skeleton_links: {link_name: (parent_keypoint, child_keypoint)}
        scale_factors: {link_name: (T,)} per-frame scale factors
        root_keypoint: name of the root keypoint (not scaled)
    Returns:
        (T, K, 3) scaled keypoint positions
    """
    scaled = skeleton_positions.copy()
    # Build traversal order (parent before child)
    visited = {root_keypoint}
    order = []
    remaining = list(skeleton_links.keys())
    max_iterations = len(remaining) * len(remaining)
    iterations = 0
    while remaining and iterations < max_iterations:
        iterations += 1
        for link_name in remaining[:]:
            parent, child = skeleton_links[link_name]
            if parent in visited:
                order.append(link_name)
                visited.add(child)
                remaining.remove(link_name)

    for link_name in order:
        parent, child = skeleton_links[link_name]
        if link_name not in scale_factors:
            continue
        scale = scale_factors[link_name][:, None]  # (T, 1)
        orig_vec = skeleton_positions[:, child] - skeleton_positions[:, parent]
        scaled_vec = scale * orig_vec
        scaled[:, child] = scaled[:, parent] + scaled_vec

    return scaled


# ── Model Loading ─────────────────────────────────────────────────────────

def load_model_with_ground(xml_file: str | Path) -> mujoco.MjModel:
    """Load a robot MJCF and add a visual ground plane plus skybox and lights."""
    spec = mujoco.MjSpec.from_file(str(xml_file))
    existing_tex = {texture.name for texture in spec.textures}
    existing_mat = {material.name for material in spec.materials}

    if "skybox" not in existing_tex:
        spec.add_texture(
            name="skybox",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.3, 0.5, 0.7],
            rgb2=[0.0, 0.0, 0.0],
            width=512,
            height=512,
        )
    if "groundplane" not in existing_tex:
        spec.add_texture(
            name="groundplane",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            width=300,
            height=300,
        )
    if "groundplane" not in existing_mat:
        spec.add_material(
            name="groundplane",
            textures=["", "groundplane"],
            texrepeat=[5, 5],
            texuniform=True,
            reflectance=0.2,
        )

    spec.worldbody.add_light(
        pos=[0, 0, 20.0],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.7, 0.7, 0.7],
        specular=[0.3, 0.3, 0.3],
    )
    spec.worldbody.add_light(
        pos=[4, 4, 6.0],
        dir=[-0.5, -0.5, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.4, 0.4, 0.4],
        specular=[0.1, 0.1, 0.1],
    )

    ground = spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        material="groundplane",
        pos=[0, 0, 0],
    )
    ground.contype = 0
    ground.conaffinity = 0
    return spec.compile()


# ── Keyframe Adjustment ───────────────────────────────────────────────────

def apply_axis_map_and_local_euler_offset_wxyz(
    quaternions_wxyz: np.ndarray,
    euler_offsets_deg: np.ndarray,
    axis_map: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply axis remapping and local Euler offset to WXYZ quaternions.

    Args:
        quaternions_wxyz: (T, 4) WXYZ quaternions
        euler_offsets_deg: (3,) Euler offsets in degrees (x, y, z)
        axis_map: {"x": (3,), "y": (3,), "z": (3,)} axis remapping vectors
    Returns:
        (T, 4) adjusted WXYZ quaternions
    """
    from scipy.spatial.transform import Rotation as _R

    # Build rotation matrix from axis map
    R_map = np.column_stack([
        axis_map.get("x", np.array([1.0, 0.0, 0.0])),
        axis_map.get("y", np.array([0.0, 1.0, 0.0])),
        axis_map.get("z", np.array([0.0, 0.0, 1.0])),
    ])

    # Local offset rotation
    local_rot = _R.from_euler("xyz", euler_offsets_deg, degrees=True)
    R_local = local_rot.as_matrix()

    # Apply: R_mapped = R_map @ R_original @ R_local
    adjusted = np.empty_like(quaternions_wxyz)
    for t in range(len(quaternions_wxyz)):
        R_orig = _R.from_quat([
            quaternions_wxyz[t, 1],
            quaternions_wxyz[t, 2],
            quaternions_wxyz[t, 3],
            quaternions_wxyz[t, 0],
        ]).as_matrix()
        R_final = R_map @ R_orig @ R_local
        quat_xyzw = _R.from_matrix(R_final).as_quat()
        adjusted[t, 0] = quat_xyzw[3]  # w
        adjusted[t, 1] = quat_xyzw[0]  # x
        adjusted[t, 2] = quat_xyzw[1]  # y
        adjusted[t, 3] = quat_xyzw[2]  # z

    return adjusted


# ── Playback Helpers ───────────────────────────────────────────────────────

def advance_frame_cursor(cursor: int, delta: int, num_frames: int, loop: bool) -> int:
    """Advance a frame cursor by delta, handling loop/clip boundaries."""
    cursor += delta
    if loop:
        cursor %= num_frames
    else:
        cursor = max(0, min(cursor, num_frames - 1))
    return cursor


def format_progress_line(
    frame: int,
    total: int,
    elapsed: float,
    fps: float,
    task_errors: dict[str, float] | None = None,
) -> str:
    """Format a progress line for console output."""
    pct = (frame + 1) / total * 100 if total > 0 else 0
    line = f"  Frame {frame+1:>5d}/{total} ({pct:5.1f}%) | {elapsed:6.1f}s | {fps:5.1f} fps"
    if task_errors:
        error_str = " | ".join(f"{k}={v:.4f}" for k, v in task_errors.items())
        line += f" | {error_str}"
    return line


# ── Contact Name Canonicalization ─────────────────────────────────────────

def canonicalize_contact_name(body_name: str) -> str:
    """Normalize a body name to a canonical contact name."""
    lower = body_name.lower()
    if "left" in lower and "toe" in lower:
        return "left_toe"
    if "right" in lower and "toe" in lower:
        return "right_toe"
    if "left" in lower and "foot" in lower and "end" in lower:
        return "left_foot_end"
    if "right" in lower and "foot" in lower and "end" in lower:
        return "right_foot_end"
    if "left" in lower and ("hand" in lower or "wrist" in lower):
        return "left_hand"
    if "right" in lower and ("hand" in lower or "wrist" in lower):
        return "right_hand"
    return body_name


# ── Leg Link Derivation ───────────────────────────────────────────────────

def derive_leg_body_chain_from_links(
    robot_links: dict[str, tuple[str, str]],
) -> tuple[str, ...]:
    """Derive the ordered leg body chain from robot link definitions.

    Returns body names ordered from root to foot.
    """
    chain = []
    link_chain = ["left_hip", "left_thigh", "left_calf"]
    for link_name in link_chain:
        if link_name in robot_links:
            parent, child = robot_links[link_name]
            if not chain:
                chain.append(parent)
            chain.append(child)
    return tuple(chain)
