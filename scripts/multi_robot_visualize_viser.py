"""Visualize motion trajectories of multiple robots simultaneously using Viser.

Browser-based 3D visualization with dynamic robot add/remove support.
Can start with an empty scene and add robots via the GUI panel.

File naming convention: <motion_name>_<robot_name>.csv, for example Form_1_stageii_g1.csv.
MuJoCo free-joint qpos ordering is [pos(xyz), quat(wxyz)].
After loading CSV files, quaternions are converted from xyzw to wxyz.

Usage examples:
    # Empty scene (add robots via browser GUI)
    python scripts/multi_robot_visualize_viser.py --port 8080

    # Pre-load robots
    python scripts/multi_robot_visualize_viser.py \
        --robots g1 h2 \
        --data_dirs output_data/robot_motion dataset/lafan1_g1 \
        --port 8080

Features:
    - Dynamic robot add/remove via browser GUI (thread-safe)
    - Batched mesh rendering for efficient multi-robot visualization
    - Chinese names for robot types and motions
    - Grid layout with automatic spacing
    - Playback controls (pause, replay, speed, frame slider)
"""

import argparse
import glob as glob_mod
import math
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import mujoco
import numpy as np
import trimesh
import viser
import yaml
from scipy.spatial.transform import Rotation  # noqa: F401

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

ROBOT_CONFIG_DIR = os.path.join(PROJECT_DIR, "config", "robot")
MOTION_DIR = os.path.join(SCRIPT_DIR, "..", "output_data", "robot_motion")

MOTION_LABELS_CN = {
    "Form_1_stageii": "形体检查",
    "dance1": "舞蹈 1",
    "dance2": "舞蹈 2",
    "fallAndGetUp1": "跌倒起身 1",
    "fallAndGetUp2": "跌倒起身 2",
    "fallAndGetUp3": "跌倒起身 3",
    "fight1": "格斗 1",
    "fightAndSports1": "格斗运动 1",
    "grab_walk_ff_180_001__A550": "抓握行走",
    "jumps1": "跳跃 1",
    "run1": "跑步 1",
    "run2": "跑步 2",
    "sprint1": "冲刺 1",
    "walk1": "行走 1",
    "walk2": "行走 2",
    "walk3": "行走 3",
    "walk4": "行走 4",
    "body_check_001__A548": "形体检查",
}

# Robot type Chinese name mapping
ROBOT_LABELS_CN = {
    "agibot_x2": "艾博特 X2",
    "booster_t1": "Booster T1",
    "DR02": "DR02",
    "g1": "G1 人形",
    "g1_d": "G1D",
    "h1": "H1 人形",
    "h1_2": "H1-2 人形",
    "h2": "H2 人形",
    "hightorque_hi": "高力矩 HI",
    "hightorque_pi": "高力矩 PI",
    "jaka_pi": "Jaka PI",
    "limx_oli": "LIMX OLI",
    "noetix_e1": "Noetix E1",
    "noetix_n2": "Noetix N2",
    "pm01": "PM01",
    "pnd_adam": "PND Adam",
    "r1": "R1 人形",
    "t800": "T800",
    "tienkung": "天坤",
    "unitree_a2": "宇树 A2",
    "unitree_a2w": "宇树 A2W",
    "xbot": "XBot",
}

ROBOT_FALLBACK = {"M": "g1"}


def get_robot_label_cn(robot_type: str) -> str:
    """Get Chinese label for a robot type, formatted as 'English (中文)'."""
    cn = ROBOT_LABELS_CN.get(robot_type, "")
    if cn and cn != robot_type:
        return f"{robot_type} ({cn})"
    return robot_type


def get_motion_label_cn(motion_name: str) -> str:
    """Get Chinese label for a motion, formatted as '中文 (English)' to ensure uniqueness."""
    cn = MOTION_LABELS_CN.get(motion_name, "")
    if cn:
        return f"{cn} ({motion_name})"
    return motion_name


def resolve_robot(robot: str) -> str:
    if robot in available_robots():
        return robot
    if robot in ROBOT_FALLBACK:
        return ROBOT_FALLBACK[robot]
    if robot.startswith("subject"):
        return "g1"
    # Fallback: strip __N suffix (e.g. g1__1 -> g1) for duplicate robot instances
    base_robot = robot.split("__")[0]
    if base_robot in available_robots():
        return base_robot
    return robot



def available_robots() -> list[str]:
    if not os.path.isdir(ROBOT_CONFIG_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(ROBOT_CONFIG_DIR)
        if f.endswith(".yaml")
    )


def get_robot_xml(robot: str) -> str:
    config_path = os.path.join(ROBOT_CONFIG_DIR, f"{robot}.yaml")
    if not os.path.isfile(config_path):
        # Fallback: strip __N suffix (e.g. g1__1 -> g1) for duplicate robot instances
        base_robot = robot.split("__")[0]
        fallback_path = os.path.join(ROBOT_CONFIG_DIR, f"{base_robot}.yaml")
        if os.path.isfile(fallback_path):
            print(f"[WARN] No config for '{robot}', falling back to '{base_robot}'")
            config_path = fallback_path
        else:
            raise FileNotFoundError(f"Robot config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    xml_path = config.get("robot_xml_path")
    if not xml_path:
        raise KeyError(f"'robot_xml_path' missing in config: {config_path}")
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(PROJECT_DIR, xml_path)
    return xml_path


def load_motion(csv_path: str) -> np.ndarray:
    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 7:
        raise ValueError(
            f"Motion file {csv_path} must have at least 7 columns (pos3 + quat4), got {data.shape[1]}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"Motion file {csv_path} contains NaN or Inf values")
    qpos = data.copy()
    quat_xyzw = data[:, 3:7]
    qpos[:, 3] = quat_xyzw[:, 3]
    qpos[:, 4] = quat_xyzw[:, 0]
    qpos[:, 5] = quat_xyzw[:, 1]
    qpos[:, 6] = quat_xyzw[:, 2]
    return qpos


def build_combined_spec(robots: list[str]) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "multi_robot"
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.specular = [0.3, 0.3, 0.3]

    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=512,
    )

    grid_res = 256
    line_px = 3
    grid_rgba = np.zeros((grid_res, grid_res, 4), dtype=np.uint8)
    bg_color = (100, 100, 100)
    line_color = (50, 50, 50)
    grid_rgba[:, :, :3] = bg_color
    grid_rgba[:, :, 3] = 255
    grid_rgba[:line_px, :, :3] = line_color
    grid_rgba[-line_px:, :, :3] = line_color
    grid_rgba[:, :line_px, :3] = line_color
    grid_rgba[:, -line_px:, :3] = line_color

    grid_tex = spec.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=grid_res,
        height=grid_res,
        nchannel=4,
    )
    grid_tex.data = grid_rgba.reshape(-1).tobytes()

    spec.add_material(
        name="groundplane",
        textures=["", "groundplane"],
        texrepeat=[10, 10],
        texuniform=True,
        reflectance=0.0,
    )
    spec.worldbody.add_light(
        pos=[0, 0, 20.0],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.7, 0.7, 0.7],
        specular=[0.3, 0.3, 0.3],
    )
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        material="groundplane",
        pos=[0, 0, 0],
    )

    # Track how many times each robot name has appeared (for dedup)
    _robot_count: dict[str, int] = {}
    for robot in robots:
        resolved = resolve_robot(robot)
        robot_spec = mujoco.MjSpec.from_file(get_robot_xml(resolved))
        for j in robot_spec.joints:
            if j.type == mujoco.mjtJoint.mjJNT_FREE:
                j.name = "floating_base_joint"
                break
        frame = spec.worldbody.add_frame()
        # Use unique prefix: robot_1, robot_2, etc. for duplicate robot names
        _robot_count[robot] = _robot_count.get(robot, 0) + 1
        if _robot_count[robot] > 1:
            prefix = f"{robot}_{_robot_count[robot]}_"
        else:
            prefix = f"{robot}_"
        spec.attach(robot_spec, prefix=prefix, frame=frame)

    for g in spec.geoms:
        g.contype = 0
        g.conaffinity = 0
    return spec


# ── RobotInstance ──────────────────────────────────────────────────
@dataclass
class RobotInstance:
    """One robot instance in the scene."""
    robot_type: str              # e.g. "g1", "h2"
    motion_name: str             # resolved motion CSV name
    key: str                     # unique key (e.g. "g1__1", "g1__2")
    start: int = 0               # qpos start index (filled during rebuild)
    dim: int = 0                 # qpos dimension
    offset: tuple = (0.0, 0.0)  # grid offset (dx, dy)
    qpos_data: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))


# ── CSV / Motion discovery ────────────────────────────────────────
def _find_csv(motion_dir: str, motion_name: str, robot_type: str) -> str:
    """Find a motion CSV file, trying multiple naming conventions.

    Supports:
      {motion}_{robot}.csv, {motion}_{robot}__N.csv, {motion}_M.csv,
      {motion}_subject*.csv (lafan1 format)
    """
    # Strip _from_{robot} suffix if present
    clean_motion = motion_name
    if motion_name.endswith(f"_from_{robot_type}"):
        clean_motion = motion_name[: -len(robot_type) - 6]

    candidates: list[str] = []
    # Direct match
    candidates.append(os.path.join(motion_dir, f"{motion_name}.csv"))
    candidates.append(os.path.join(motion_dir, f"{clean_motion}.csv"))
    # Standard: {motion}_{robot}.csv (recursive for subdirs like output_data)
    candidates.append(os.path.join(motion_dir, "**", f"{clean_motion}_{robot_type}.csv"))
    # Indexed: {motion}_{robot}__N.csv (for temp dirs with multiple same-type robots)
    candidates.append(os.path.join(motion_dir, "**", f"{clean_motion}_{robot_type}__*.csv"))
    # Bones format: {motion}_M.csv
    candidates.append(os.path.join(motion_dir, "**", f"{clean_motion}_M.csv"))
    # Lafan1 format: {motion}_subject*.csv (recursive)
    candidates.append(os.path.join(motion_dir, "**", f"{clean_motion}_subject*.csv"))

    for c in candidates:
        if "*" in c:
            matches = sorted(glob_mod.glob(c, recursive=True))
            if matches:
                return matches[0]
        elif os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"Motion file not found for {robot_type}/{motion_name} in {motion_dir}"
    )


def _strip_robot_suffix(base: str, robot_type: str) -> str:
    """Strip robot suffix from CSV base name, handling __N indexed formats.

    Examples:
        'fight1_g1__2' with robot_type='g1' → 'fight1'
        'fight1_g1' with robot_type='g1' → 'fight1'
        'dance1_M' with robot_type='g1' → 'dance1'
        'dance1_subject1' with robot_type='g1' → 'dance1_subject1' (no match)
    """
    # Pattern: {motion}_{robot}__N  (e.g. fight1_g1__2)
    m = re.match(r'^(.+)_' + re.escape(robot_type) + r'__\d+$', base)
    if m:
        return m.group(1)
    # Pattern: {motion}_{robot}  (e.g. fight1_g1)
    m = re.match(r'^(.+)_' + re.escape(robot_type) + r'$', base)
    if m:
        return m.group(1)
    # Pattern: {motion}_M  (bones_g1 format, implicit g1)
    if base.endswith("_M"):
        return base[:-2]
    return base


def _scan_motions_single_dir(motion_dir: str, robot_type: str) -> list[str]:
    """Return sorted list of unique motion names available for a robot type."""
    motions: set[str] = set()
    for f in glob_mod.glob(os.path.join(motion_dir, "**", "*.csv"), recursive=True):
        base = os.path.splitext(os.path.basename(f))[0]
        extracted = _strip_robot_suffix(base, robot_type)
        if extracted != base:
            # Successfully stripped robot suffix
            motions.add(extracted)
        else:
            # Try lafan1 format: {motion}_subject{N}
            m = re.match(r'^(.+)_subject\d+$', base)
            if m:
                motions.add(m.group(1))
    return sorted(motions)


def _scan_known_robots(motion_dir: str) -> list[str]:
    """Scan motion directory to discover available robot types from CSV filenames."""
    robots: set[str] = set()
    # Known robot name patterns
    known = ("g1", "h2", "g2", "h1", "b2", "agibot_x2", "r1", "h1_2")
    # Scan recursively (CSV files may be in subdirectories)
    for f in glob_mod.glob(os.path.join(motion_dir, "**", "*.csv"), recursive=True):
        base = os.path.splitext(os.path.basename(f))[0]
        for r in known:
            # Check: {motion}_{r}.csv, {motion}_{r}__N.csv, {motion}_{r}_M.csv
            if (base.endswith(f"_{r}") and not base.endswith(f"_{r}_M")) or \
               base.endswith(f"_{r}__") or \
               re.match(r'.+_' + re.escape(r) + r'__\d+$', base):
                robots.add(r)
                break
        else:
            # Also check _M suffix pattern: {motion}_M.csv → robot = g1
            if base.endswith("_M"):
                robots.add("g1")
    return sorted(robots)


def get_qpos_start(model: mujoco.MjModel, freejoint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, freejoint_name)
    if jid < 0:
        raise ValueError(f"Joint not found in model: {freejoint_name}")
    return model.jnt_qposadr[jid]


def rotation_matrix_to_quat_wxyz(m: np.ndarray) -> tuple:
    """Fast inline rotation matrix (flattened 3x3) to wxyz quaternion."""
    det = (
        m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
    )
    if abs(det) < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    trace = m[0] + m[4] + m[8]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return (0.25 / s, (m[7] - m[5]) * s, (m[2] - m[6]) * s, (m[3] - m[1]) * s)
    elif m[0] > m[4] and m[0] > m[8]:
        s = 2.0 * np.sqrt(1.0 + m[0] - m[4] - m[8])
        return ((m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s)
    elif m[4] > m[8]:
        s = 2.0 * np.sqrt(1.0 + m[4] - m[0] - m[8])
        return ((m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s)
    else:
        s = 2.0 * np.sqrt(1.0 + m[8] - m[0] - m[4])
        return ((m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s)


class MultiRobotViserApp:
    """Dynamic multi-robot visualization with add/remove support."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.motion_dir = args.motion_dir

        # Data directories for dynamic add/remove (original data dirs, not temp dir)
        self.data_dirs: list[str] = args.data_dirs or [args.motion_dir]

        # Parse per-robot motion mapping
        self.robot_motion_map: dict[str, str] = {}
        if args.robot_motion:
            for rm in args.robot_motion:
                parts = rm.split(":", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid --robot-motion format: {rm}")
                self.robot_motion_map[parts[0]] = parts[1]

        # Active robots and pending list
        self.active_robots: list[RobotInstance] = []
        self.pending_robots: list[RobotInstance] = []
        self.type_counts: dict[str, int] = defaultdict(int)

        # Viser server (created early for GUI)
        self.server = viser.ViserServer(port=args.port, verbose=False)
        self.server.scene.set_up_direction("+z")
        # 使用 viser 默认光照（与 gr00t_mjlab_autodl 一致）
        self.server.scene.configure_default_lights(enabled=True)
        # 灰色地面网格（细线条）
        self.server.scene.add_grid(
            "ground",
            width=20,
            height=20,
            cell_size=1.0,
            cell_thickness=0.5,
            cell_color=(80, 80, 80),
            section_thickness=0.8,
            section_color=(50, 50, 50),
            position=(0, 0, -0.01),
            wxyz=(1, 0, 0, 0),
        )

        # MuJoCo state
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.batched_handles: list[tuple[Any, list[tuple[int, float, float]]]] = []
        self.mesh_colors: dict[int, tuple[float, float, float]] = {}

        # Thread safety lock for rebuild operations
        self._rebuild_lock = threading.Lock()

        # Playback state
        self.paused = False
        self.replay = False
        self.current_frame = 0
        self.n_frames = 0
        self.step = max(1, round(args.source_fps / args.render_fps))
        self.dt = self.step / args.source_fps

        # Video recording state
        self.recording = False
        self.record_dir: Optional[str] = None
        self.record_frame_count = 0
        self._record_thread: Optional[threading.Thread] = None
        self._record_stop_event = threading.Event()

        # Initialize robots from CLI (may be empty for browser-only workflow)
        for robot_name in args.robots:
            motion_name = self.robot_motion_map.get(robot_name, args.motion)
            key = self._generate_key(robot_name)
            csv_path = _find_csv(self.motion_dir, motion_name, robot_name)
            qpos = load_motion(csv_path)
            inst = RobotInstance(
                robot_type=robot_name,
                motion_name=motion_name,
                key=key,
                qpos_data=qpos,
            )
            self.active_robots.append(inst)

        # Build initial scene (handles empty robot list)
        self._rebuild()

        # GUI
        self._build_gui()

    # ── Key generation ───────────────────────────────────────────────
    def _reset_type_counts(self) -> None:
        """Reset type_counts based on current active_robots (call after _rebuild)."""
        self.type_counts.clear()
        for inst in self.active_robots:
            self.type_counts[inst.robot_type] += 1


    def _generate_key(self, robot_type: str) -> str:
        self.type_counts[robot_type] += 1
        count = self.type_counts[robot_type]
        if count == 1:
            return robot_type
        return f"{robot_type}__{count}"

    # ── Rebuild scene ────────────────────────────────────────────────
    def _rebuild(self) -> None:
        """Rebuild MuJoCo model and recreate all scene objects."""
        # Dispose old handles
        for handle, _ in self.batched_handles:
            handle.remove()
        self.batched_handles.clear()

        if not self.active_robots:
            self.model = None
            self.data = None
            self.n_frames = 0
            self.current_frame = 0
            self._reset_type_counts()
            return

        # Build combined spec
        robot_types = [r.robot_type for r in self.active_robots]
        spec = build_combined_spec(robot_types)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        # Position robots in grid
        self._compute_layout()

        # Map qpos — need to find the correct joint for each robot instance
        # build_combined_spec uses prefix "robot_" for first, "robot_N_" for duplicates
        # So joint names are "g1_floating_base_joint", "g1_2_floating_base_joint", etc.
        # Also build prefix map for body->robot mapping (done in single pass)
        # Reset type_counts here to ensure correct joint naming
        self.type_counts.clear()
        inst_prefixes: dict[str, str] = {}  # key -> prefix
        for inst in self.active_robots:
            self.type_counts[inst.robot_type] = self.type_counts.get(inst.robot_type, 0) + 1
            count = self.type_counts[inst.robot_type]
            if count == 1:
                joint_name = f"{inst.robot_type}_floating_base_joint"
                inst_prefixes[inst.key] = f"{inst.robot_type}_"
            else:
                joint_name = f"{inst.robot_type}_{count}_floating_base_joint"
                inst_prefixes[inst.key] = f"{inst.robot_type}_{count}_"
            inst.start = get_qpos_start(self.model, joint_name)
            if inst.start < 0:
                raise ValueError(f"Joint not found in model: {joint_name}")

        # Compute qpos dim for each robot: from its start to the next robot's start (or nq)
        # This gives the total qpos slots belonging to this robot in the model.
        for idx, inst in enumerate(self.active_robots):
            if idx + 1 < len(self.active_robots):
                next_start = self.active_robots[idx + 1].start
            else:
                next_start = self.model.nq
            model_dim = next_start - inst.start
            csv_dim = inst.qpos_data.shape[1]
            inst.dim = min(model_dim, csv_dim)

        body_id_to_robot: dict[int, str] = {}
        for body_id in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name:
                for inst in self.active_robots:
                    if body_name.startswith(inst_prefixes[inst.key]):
                        body_id_to_robot[body_id] = inst.key
                        break

        # Group geoms
        mesh_groups: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
        self.mesh_colors.clear()
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name == "ground":
                continue
            geom = self.model.geom(i)
            geom_type = int(geom.type[0])
            if not self.args.all_geoms and geom_type != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            body_id = int(geom.bodyid[0])
            parent_key = body_id_to_robot.get(body_id)
            if parent_key is None:
                continue
            mesh_id = int(geom.dataid[0]) if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH) else -(i + 1)
            # Find offset
            dx, dy = 0.0, 0.0
            for inst in self.active_robots:
                if inst.key == parent_key:
                    dx, dy = inst.offset
                    break
            mesh_groups[mesh_id].append((i, dx, dy))
            rgba = geom.rgba
            if rgba is not None and float(rgba[3]) > 0.01:
                self.mesh_colors[mesh_id] = (float(rgba[0]), float(rgba[1]), float(rgba[2]))

        # Create batched handles
        for mesh_id, geoms in mesh_groups.items():
            if mesh_id >= 0 and mesh_id < self.model.nmesh:
                vert_adr = self.model.mesh_vertadr[mesh_id]
                vert_num = self.model.mesh_vertnum[mesh_id]
                face_adr = self.model.mesh_faceadr[mesh_id]
                face_num = self.model.mesh_facenum[mesh_id]
                if vert_num <= 0 or face_num <= 0:
                    continue
                vertices = self.model.mesh_vert[vert_adr:vert_adr + vert_num].copy()
                faces = self.model.mesh_face[face_adr:face_adr + face_num].copy()
            else:
                geom = self.model.geom(geoms[0][0])
                size = geom.size
                box = trimesh.creation.box(extents=size * 2)
                vertices = box.vertices.astype(np.float32)
                faces = box.faces.astype(np.uint32)

            n_inst = len(geoms)
            positions = np.zeros((n_inst, 3), dtype=np.float32)
            wxyzs = np.tile([1, 0, 0, 0], (n_inst, 1)).astype(np.float32)

            if mesh_id in self.mesh_colors:
                r, g, b = self.mesh_colors[mesh_id]
                mesh_color = (int(r * 255), int(g * 255), int(b * 255))
            else:
                mesh_color = (180, 180, 180)

            handle = self.server.scene.add_batched_meshes_simple(
                f"robots/batch_{mesh_id}",
                vertices.astype(np.float32),
                faces.astype(np.uint32),
                batched_wxyzs=wxyzs,
                batched_positions=positions,
                batched_colors=mesh_color,
            )
            self.batched_handles.append((handle, geoms))

        # Compute frame count
        self.n_frames = min(inst.qpos_data.shape[0] for inst in self.active_robots) if self.active_robots else 0
        self.current_frame = min(self.current_frame, max(0, self.n_frames - 1))

        # Update GUI slider max if GUI is already built
        if hasattr(self, 'gui_frame'):
            self.gui_frame.max = max(1, self.n_frames - 1)

        print(f"[rebuild] {len(self.active_robots)} robots, {self.n_frames} frames, "
              f"{len(self.batched_handles)} mesh groups")

        # Update frame slider range (unified location)
        if hasattr(self, 'gui_frame'):
            self.gui_frame.max = max(1, self.n_frames - 1)
        self._reset_type_counts()

    # ── Layout ───────────────────────────────────────────────────────
    def _compute_layout(self) -> None:
        spacing = 2.0
        n = len(self.active_robots)
        if n == 0:
            return
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)
        for idx, inst in enumerate(self.active_robots):
            col = idx % cols
            row = idx // cols
            dx = (col - (cols - 1) / 2.0) * spacing
            dy = (row - (rows - 1) / 2.0) * spacing
            inst.offset = (dx, dy)

    # ── GUI ──────────────────────────────────────────────────────────
    def _build_gui(self) -> None:
        s = self.server
        # Remove old GUI if any (viser doesn't have clear, so we just add new)
        # Playback controls
        with s.gui.add_folder("播放控制"):
            self.gui_pause = s.gui.add_button("暂停")
            self.gui_replay = s.gui.add_button("重新播放")
            self.gui_frame = s.gui.add_slider("帧", min=0, max=max(1, self.n_frames - 1), step=1.0, initial_value=0.0)
            self.gui_speed = s.gui.add_slider("速度", min=0.1, max=3.0, step=0.1, initial_value=1.0)

        @self.gui_pause.on_click
        def _(_):
            self.paused = not self.paused
            self.gui_pause.label = "继续" if self.paused else "暂停"

        @self.gui_replay.on_click
        def _(_):
            self.replay = True

        # Video recording controls
        with s.gui.add_folder("录制与导出"):
            self.gui_record = s.gui.add_button("开始录制")
            self.gui_export_frame = s.gui.add_button("导出当前帧")
            self.gui_record_status = s.gui.add_markdown("未录制")

        @self.gui_record.on_click
        def _(_):
            self._toggle_recording()

        @self.gui_export_frame.on_click
        def _(_):
            self._export_current_frame()

        # ── Dynamic Robot Management ──────────────────────────────────
        with s.gui.add_folder("机器人管理"):
            # Robot type selector — all registered robots (from config) + data-derived
            # Display Chinese names in dropdown, store raw names as values
            available_types = self._scan_all_robots()
            robot_display_options = [get_robot_label_cn(r) for r in available_types]
            self._robot_display_to_raw = {get_robot_label_cn(r): r for r in available_types}
            initial_robot_display = get_robot_label_cn(available_types[0]) if available_types else "g1"
            self.gui_robot_type = s.gui.add_dropdown("机器人类型", options=robot_display_options, initial_value=initial_robot_display)
            # Motion selector — display Chinese names
            self.gui_motion = s.gui.add_dropdown("动作", options=["default"], initial_value="default")
            # Add button + Clear button (grouped together, both relate to pending list)
            self.gui_add = s.gui.add_button("添加到待添加")
            self.gui_clear = s.gui.add_button("清空待添加")
            # Pending list display
            self.gui_pending_md = s.gui.add_markdown("暂无待添加机器人")
            # Apply button
            self.gui_apply = s.gui.add_button("应用更改")
            # Active robot list with remove buttons
            self.gui_active_md = s.gui.add_markdown("暂无活动机器人")
            self._remove_buttons: list[Any] = []
            # Remove buttons for active robots (created dynamically)
            self._rebuild_remove_buttons()

        @self.gui_add.on_click
        def _(_):
            self._add_to_pending()

        @self.gui_apply.on_click
        def _(_):
            self._apply_changes()

        @self.gui_clear.on_click
        def _(_):
            self._clear_pending()

        # Update motion list when robot type changes
        @self.gui_robot_type.on_update
        def _(_):
            raw_type = self._robot_display_to_raw.get(self.gui_robot_type.value, self.gui_robot_type.value)
            motions = self._scan_all_motions(raw_type)
            motion_display = [get_motion_label_cn(m) for m in motions]
            self._motion_display_to_raw = {get_motion_label_cn(m): m for m in motions}
            self.gui_motion.options = motion_display if motion_display else ["default"]
            self.gui_motion.value = motion_display[0] if motion_display else "default"

        # Initial scan
        raw_type = self._robot_display_to_raw.get(self.gui_robot_type.value, self.gui_robot_type.value)
        motions = self._scan_all_motions(raw_type)
        motion_display = [get_motion_label_cn(m) for m in motions]
        self._motion_display_to_raw = {get_motion_label_cn(m): m for m in motions}
        self.gui_motion.options = motion_display if motion_display else ["default"]
        self.gui_motion.value = motion_display[0] if motion_display else "default"

        self._update_pending_gui()
        self._update_active_gui()

    # ── Video Recording ─────────────────────────────────────────────
    def _toggle_recording(self) -> None:
        """Toggle video recording on/off."""
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _export_current_frame(self) -> None:
        """Export the current frame as a PNG image."""
        client = self._get_first_client()
        if client is None:
            self.gui_record_status.value = "错误：无客户端连接\n请先打开浏览器页面"
            return
        try:
            img = client.get_render(height=1080, width=1920, transport_format="png")
            import cv2
            import time as _time
            ts = _time.strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(PROJECT_DIR, "output_data", "record_frames")
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, f"viser_frame_{ts}.png")
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
            self.gui_record_status.value = f"帧已保存: {output_path}"
            print(f"[Export] Frame saved to {output_path}")
        except Exception as e:
            self.gui_record_status.value = f"错误: {e}"
            print(f"[Export] Error: {e}")

    def _start_recording(self) -> None:
        """Start recording video by capturing browser frames via viser's get_render API."""
        import tempfile

        self.record_dir = tempfile.mkdtemp(prefix="viser_record_")
        self.record_frame_count = 0
        self.recording = True
        self._record_stop_event.clear()
        self.gui_record.label = "停止录制"
        self.gui_record_status.value = f"录制中: 0 帧\n临时目录: {self.record_dir}"

        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()
        print(f"[Recording] Started, saving frames to {self.record_dir}")

    def _stop_recording(self) -> None:
        """Stop recording and encode video with OpenCV."""
        self.recording = False
        self._record_stop_event.set()
        self.gui_record.label = "开始录制"

        if self._record_thread is not None:
            self._record_thread.join(timeout=5.0)
            self._record_thread = None

        n = self.record_frame_count
        self.gui_record_status.value = f"已录制 {n} 帧\n正在编码视频..."

        # Encode video with ffmpeg in a background thread
        thread = threading.Thread(target=self._encode_video, daemon=True)
        thread.start()
        print(f"[Recording] 已停止。已捕获 {n} 帧，正在编码视频...")

    def _record_loop(self) -> None:
        """Background loop: capture frames via viser's get_render API at ~30fps."""
        # Wait for a client to connect
        time.sleep(1.0)
        client = self._get_first_client()
        if client is None:
            print("[Recording] 无客户端连接，请先打开浏览器页面。")
            self.recording = False
            self.gui_record.label = "开始录制"
            self.gui_record_status.value = "错误：无客户端连接\n请先打开浏览器页面"
            return

        while not self._record_stop_event.is_set():
            try:
                # Request render from client (JPEG, 720p for reasonable file size)
                img = client.get_render(height=720, width=1280, transport_format="jpeg")
                frame_path = os.path.join(self.record_dir, f"frame_{self.record_frame_count:06d}.png")
                # Save with OpenCV (convert RGB->BGR for cv2.imwrite)
                import cv2
                cv2.imwrite(frame_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                self.record_frame_count += 1
                if self.record_frame_count % 30 == 0:
                    self.gui_record_status.value = f"录制中: {self.record_frame_count} 帧"
            except Exception as e:
                print(f"[Recording] 截图错误: {e}")
                break

            # ~30fps capture
            self._record_stop_event.wait(1.0 / 30.0)

        print(f"[Recording] 截图循环结束。已保存 {self.record_frame_count} 帧。")

    def _get_first_client(self) -> Any:
        """Get the first connected client handle, or None."""
        for _ in range(30):  # Wait up to 30 seconds for a client
            clients = self.server.get_clients()
            if clients:
                return next(iter(clients.values()))
            time.sleep(1.0)
        return None

    def _encode_video(self) -> None:
        """Encode recorded frames into MP4 video using OpenCV."""
        import cv2

        if self.record_frame_count == 0:
            self.gui_record_status.value = "未录制任何帧"
            return

        out_dir = os.path.join(PROJECT_DIR, "output_data", "record_video")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "viser_recording.mp4")

        # Read first frame to get dimensions
        first_frame_path = os.path.join(self.record_dir, "frame_000000.png")
        first_img = cv2.imread(first_frame_path)
        if first_img is None:
            self.gui_record_status.value = "错误：无法读取首帧"
            return

        h, w = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

        if not writer.isOpened():
            self.gui_record_status.value = "错误：无法创建视频写入器"
            return

        try:
            for i in range(self.record_frame_count):
                frame_path = os.path.join(self.record_dir, f"frame_{i:06d}.png")
                img = cv2.imread(frame_path)
                if img is not None:
                    writer.write(img)
                if i % 100 == 0:
                    self.gui_record_status.value = f"编码中: {i}/{self.record_frame_count}"

            writer.release()
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            msg = f"视频已保存: {output_path}\n({size_mb:.1f} MB, {self.record_frame_count} 帧)"
            self.gui_record_status.value = msg
            print(f"[Recording] {msg}")
        except Exception as e:
            writer.release()
            self.gui_record_status.value = f"错误: {e}"
            print(f"[Recording] 编码错误: {e}")

    # ── Multi-dir scanning for GUI ──────────────────────────────────
    def _scan_all_robots(self) -> list[str]:
        """Return all registered robot types (from config + data dirs).

        Sorted with priority robots first, then alphabetical.
        """
        robots: set[str] = set()
        # 1. Scan config/robot/ directory for all registered robots
        if os.path.isdir(ROBOT_CONFIG_DIR):
            for f in os.listdir(ROBOT_CONFIG_DIR):
                if f.endswith(".yaml"):
                    robots.add(f[:-5])  # strip .yaml
        # 2. Also include robots found in data dirs
        for d in self.data_dirs:
            robots.update(_scan_known_robots(d))

        # Priority order: g1, g1_d, h1, h1_2, h2, unitree_a2, unitree_a2w, then alphabetical
        _priority = ["g1", "g1_d", "h1", "h1_2", "h2", "unitree_a2", "unitree_a2w"]

        def _sort_key(name: str) -> tuple:
            if name in _priority:
                return (0, _priority.index(name))
            return (1, name)

        return sorted(robots, key=_sort_key)

    def _scan_all_motions(self, robot_type: str) -> list[str]:
        """Scan all data_dirs for motions of given robot type."""
        motions: set[str] = set()
        for d in self.data_dirs:
            motions.update(_scan_motions_single_dir(d, robot_type))
        return sorted(motions)

    def _find_csv_in_data_dirs(self, motion_name: str, robot_type: str) -> str:
        """Find CSV across all data_dirs, trying each in order."""
        for d in self.data_dirs:
            try:
                return _find_csv(d, motion_name, robot_type)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(
            f"Motion file not found for {robot_type}/{motion_name} in any of {self.data_dirs}"
        )

    # ── Pending / Active management ──────────────────────────────────
    def _add_to_pending(self) -> None:
        # Map display names back to raw names
        rtype = self._robot_display_to_raw.get(self.gui_robot_type.value, self.gui_robot_type.value)
        motion = self._motion_display_to_raw.get(self.gui_motion.value, self.gui_motion.value)
        # Don't increment type_counts here — _generate_key will do it
        try:
            csv_path = self._find_csv_in_data_dirs(motion, rtype)
            qpos = load_motion(csv_path)
        except FileNotFoundError as e:
            print(f"[warn] {e}")
            return
        key = self._generate_key(rtype)
        inst = RobotInstance(robot_type=rtype, motion_name=motion, key=key, qpos_data=qpos)
        self.pending_robots.append(inst)
        self._update_pending_gui()
        print(f"[pending] Added {key} ({motion}), total pending: {len(self.pending_robots)}")

    def _apply_changes(self) -> None:
        if not self.pending_robots:
            print("[apply] No pending changes")
            return
        with self._rebuild_lock:
            self.active_robots.extend(self.pending_robots)
            self.pending_robots.clear()
            self._rebuild()
            # Update frame slider
            self._update_pending_gui()
            self._update_active_gui()
            self._rebuild_remove_buttons()
            print(f"[apply] Scene updated: {len(self.active_robots)} robots active")

    def _clear_pending(self) -> None:
        self.pending_robots.clear()
        self._update_pending_gui()
        print("[clear] Pending list cleared")

    def _remove_active(self, key: str) -> None:
        with self._rebuild_lock:
            self.active_robots = [r for r in self.active_robots if r.key != key]
            self._rebuild()
            self._update_active_gui()
            self._rebuild_remove_buttons()
            print(f"[remove] Removed {key}, {len(self.active_robots)} robots remaining")

    # ── GUI update helpers ───────────────────────────────────────────
    def _update_pending_gui(self) -> None:
        if not self.pending_robots:
            self.gui_pending_md.content = "暂无待添加机器人"
        else:
            lines = ["**待添加机器人：**"]
            for inst in self.pending_robots:
                robot_label = get_robot_label_cn(inst.robot_type)
                motion_label = get_motion_label_cn(inst.motion_name)
                lines.append(f"- `{inst.key}` — {robot_label} / {motion_label}")
            self.gui_pending_md.content = "\n".join(lines)

    def _rebuild_remove_buttons(self) -> None:
        """Remove old remove buttons and create new ones for each active robot."""
        for btn in self._remove_buttons:
            btn.remove()
        self._remove_buttons.clear()
        for inst in self.active_robots:
            btn = self.server.gui.add_button(f"移除 {inst.key}")
            @btn.on_click
            def _(evt, key=inst.key):
                self._remove_active(key)
            self._remove_buttons.append(btn)

    def _update_active_gui(self) -> None:
        if not self.active_robots:
            self.gui_active_md.content = "暂无活动机器人"
        else:
            lines = ["**活动机器人：**"]
            for inst in self.active_robots:
                frames = inst.qpos_data.shape[0]
                robot_label = get_robot_label_cn(inst.robot_type)
                motion_label = get_motion_label_cn(inst.motion_name)
                lines.append(f"- `{inst.key}` — {robot_label} / {motion_label} ({frames} 帧)")
            self.gui_active_md.content = "\n".join(lines)

    # ── Apply frame ─────────────────────────────────────────────────
    def apply_frame(self, frame_idx: int) -> None:
        if self.model is None or self.data is None:
            return
        # Clamp frame index to valid range (may shrink after rebuild)
        if frame_idx >= self.n_frames:
            frame_idx = max(0, self.n_frames - 1)
            self.current_frame = frame_idx
        with self._rebuild_lock:
            self._apply_frame_inner(frame_idx)

    def _apply_frame_inner(self, frame_idx: int) -> None:
        for inst in self.active_robots:
            start = inst.start
            dim = inst.dim
            # Use min of model dim and CSV dim to handle mismatch
            csv_dim = inst.qpos_data.shape[1]
            copy_dim = min(dim, csv_dim)
            # Clamp frame_idx to this robot's available frames
            safe_frame = min(frame_idx, inst.qpos_data.shape[0] - 1)
            self.data.qpos[start:start + copy_dim] = inst.qpos_data[safe_frame, :copy_dim]
            # Zero-fill remaining qpos if model has more dims than CSV
            if dim > csv_dim:
                self.data.qpos[start + csv_dim:start + dim] = 0.0
            dx, dy = inst.offset
            self.data.qpos[start] += dx
            self.data.qpos[start + 1] += dy

        mujoco.mj_forward(self.model, self.data)
        all_pos = self.data.geom_xpos
        all_mat = self.data.geom_xmat

        for handle, geoms in self.batched_handles:
            n_inst = len(geoms)
            pos = np.zeros((n_inst, 3), dtype=np.float32)
            wxyz = np.tile([1, 0, 0, 0], (n_inst, 1)).astype(np.float32)
            for j, (gid, dx, dy) in enumerate(geoms):
                pos[j, 0] = float(all_pos[gid, 0] + dx)
                pos[j, 1] = float(all_pos[gid, 1] + dy)
                pos[j, 2] = float(all_pos[gid, 2])
                wxyz[j] = rotation_matrix_to_quat_wxyz(all_mat[gid])
            handle.batched_positions = pos
            handle.batched_wxyzs = wxyz

    # ── Main loop ───────────────────────────────────────────────────
    def run(self) -> None:
        n_active = len(self.active_robots)
        print(f"source_fps={self.args.source_fps:g}, render_fps≈{self.args.source_fps / self.step:g}, "
              f"render_every={self.step} frame(s)")
        print(f"Open browser at: http://localhost:{self.args.port}")
        if n_active == 0:
            print("No robots loaded — use the browser GUI to add robots dynamically.")

        # Ground plane — already added in __init__ (skip if --no-ground)

        self.apply_frame(0)

        slider_update_counter = 0
        slider_poll_counter = 0
        _last_slider_val = 0

        while True:
            t0 = time.time()

            if self.replay:
                self.current_frame = 0
                self.replay = False
                self.paused = False
                self.gui_pause.label = "暂停"
                _last_slider_val = 0
                self.apply_frame(0)
                self.gui_frame.value = 0
                slider_update_counter = 0
                # Skip playback advance this frame so user sees frame 0
                elapsed = time.time() - t0
                sleep_time = self.dt / self.gui_speed.value - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            slider_poll_counter += 1
            if slider_poll_counter >= 3:
                slider_poll_counter = 0
                slider_val = self.gui_frame.value
                if not (slider_val == slider_val) or slider_val is None:  # NaN / None guard
                    slider_val = 0
                slider_val = int(slider_val)
                if slider_val != _last_slider_val and slider_val != self.current_frame:
                    self.current_frame = slider_val
                    _last_slider_val = slider_val
                    self.apply_frame(self.current_frame)

            # Only advance playback if there are robots and frames
            if not self.paused and self.n_frames > 0:
                self.current_frame += self.step
                if self.current_frame >= self.n_frames:
                    if self.args.loop:
                        self.current_frame = 0
                    else:
                        self.current_frame = self.n_frames - 1
                        self.paused = True
                        self.gui_pause.label = "继续"
                # Clamp in case n_frames shrank from a concurrent rebuild
                self.current_frame = min(self.current_frame, max(0, self.n_frames - 1))
                self.apply_frame(self.current_frame)

                slider_update_counter += 1
                if slider_update_counter >= 5:
                    slider_update_counter = 0
                    self.gui_frame.value = self.current_frame
                    _last_slider_val = self.current_frame

            elapsed = time.time() - t0
            sleep_time = self.dt / self.gui_speed.value - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif self.n_frames == 0:
                # No robots loaded — reduce CPU usage
                time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Viser-based multi-robot motion visualization (browser-based)"
    )
    parser.add_argument("--motion", default="body_check_001__A548_M_from_g1")
    parser.add_argument("--robots", nargs="*", default=[])
    parser.add_argument("--motion_dir", default=MOTION_DIR,
                        help="Primary motion directory (temp dir with symlinks from start.sh)")
    parser.add_argument("--data_dirs", nargs="+", default=None,
                        help="Original data directories for dynamic robot add/remove (e.g. dataset output_data/robot_motion)")
    parser.add_argument("--robot-motion", action="append", default=None,
                        help="Per-robot motion name: <robot>:<motion> (can be used multiple times)")
    parser.add_argument("--source_fps", type=float, default=30.0)
    parser.add_argument("--render_fps", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-ground", action="store_true")
    parser.add_argument("--all_geoms", action="store_true",
                        help="Render all geom types (default: only MESH)")
    parser.add_argument("--show_contacts", action="store_true",
                        help="Show ground contact force arrows (reserved)")
    parser.add_argument("--export_video", type=str, default=None,
                        help="Export video to this file path (requires ffmpeg, reserved)")
    parser.add_argument("--export_fps", type=float, default=30.0,
                        help="Export FPS (reserved)")
    parser.add_argument("--no-thread", action="store_true",
                        help="Disable threaded physics")
    args = parser.parse_args()

    app = MultiRobotViserApp(args)
    app.run()


if __name__ == "__main__":
    main()
