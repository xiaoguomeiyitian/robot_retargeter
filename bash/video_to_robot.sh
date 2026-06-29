#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# video_to_robot.sh — 端到端视频到机器人动作流水线
#
# 用法:
#   ./bash/video_to_robot.sh path/to/video.mp4
#   ./bash/video_to_robot.sh path/to/video.mp4 --robots g1 h2 t800
#   ./bash/video_to_robot.sh path/to/video.mp4 --max-frames 500
#
# 环境变量:
#   VIDEO_PATH      输入视频路径 (必需)
#   VIS_ROBOTS      目标机器人列表 (默认: g1)
#   OUTPUT_DIR      输出目录 (默认: output_data/video_to_robot)
#   PYTHON_BIN      Python 解释器 (默认: 自动检测)
#   MAX_FRAMES     最大处理帧数 (默认: 全部)
#   RENDER_FPS      可视化帧率 (默认: 30)
# ============================================================================

VIDEO_PATH="${1:-}"
if [[ -z "${VIDEO_PATH}" ]]; then
  echo "[ERROR] Usage: $0 <video_path> [--robots g1 h2] [--max-frames N]"
  exit 1
fi

# Parse optional arguments
VIS_ROBOTS="${VIS_ROBOTS:-g1}"
OUTPUT_DIR="${OUTPUT_DIR:-output_data/video_to_robot}"
MAX_FRAMES="${MAX_FRAMES:-}"
RENDER_FPS="${RENDER_FPS:-30}"

# Shift to parse remaining args
shift
while [[ $# -gt 0 ]]; do
  case $1 in
    --robots)
      VIS_ROBOTS="${2:-g1}"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-output_data/video_to_robot}"
      shift 2
      ;;
    *)
      echo "[WARN] Unknown argument: $1"
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Resolve Python interpreter
if [[ -n "${PYTHON_BIN:-}" ]]; then
  : # use user-provided
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "[ERROR] No python interpreter found"
  exit 127
fi

echo "========================================"
echo "  Video-to-Robot Pipeline"
echo "========================================"
echo "  Video: ${VIDEO_PATH}"
echo "  Robots: ${VIS_ROBOTS}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Python: ${PYTHON_BIN}"
echo ""

# Check video file exists
if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "[ERROR] Video file not found: ${VIDEO_PATH}"
  exit 1
fi

# Check MediaPipe model exists
MODEL_PATH="asset/models/pose_landmarker_heavy.task"
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[INFO] Downloading MediaPipe PoseLandmarker model..."
  mkdir -p asset/models
  wget -q "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task" \
    -O "${MODEL_PATH}"
  echo "[INFO] Model downloaded: ${MODEL_PATH}"
fi

# Build command
CMD_ARGS=(
  "${PYTHON_BIN}" scripts/video_to_robot.py
  --video "${VIDEO_PATH}"
  --robots ${VIS_ROBOTS}
  --output-dir "${OUTPUT_DIR}"
  --model "${MODEL_PATH}"
)

if [[ -n "${MAX_FRAMES}" ]]; then
  CMD_ARGS+=(--max-frames "${MAX_FRAMES}")
fi

echo "[INFO] Running pipeline..."
echo "[INFO] Command: ${CMD_ARGS[*]}"
echo ""

"${CMD_ARGS[@]}"

echo ""
echo "[DONE] Pipeline complete!"
echo ""
echo "To visualize:"
echo "  python scripts/multi_robot_visualize.py --motion $(basename "${VIDEO_PATH}" .mp4)_smplx --robots ${VIS_ROBOTS}"
