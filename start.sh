#!/usr/bin/env bash
# ============================================================================
# robot_retargeter 启动入口脚本
#
# 交互式:   ./start.sh
# 非交互:   ./start.sh <mode> [args...]
#
# 模式:
#   smpl        从 SMPL-X 动作重定向到机器人 (SMPL-X → 机器人)
#   robot       从源机器人动作重定向到目标机器人 (机器人 → 机器人)
#   viser       使用 Viser 浏览器可视化已有动作
#   list        列出所有可用机器人
#   doctor      环境健康检查
#
# 用法示例:
#   ./start.sh                                       # 交互式
#   ./start.sh list                                  # 列出可用机器人
#   ./start.sh smpl                                  # 交互式 SMPL-X 重定向
#   ./start.sh smpl --motion dataset/ACCAD/Form_1_stageii.npz --robots g1 h2
#   ./start.sh robot --motion dataset/lafan1_g1/dance1_subject2.csv --origin g1 --robots h2 r1
#   ./start.sh viser --motion Form_1_stageii --robots g1 h2 t800 --port 8080
#   ./start.sh doctor                                # 环境健康检查
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
SRC_DIR="$PROJECT_DIR/scripts"

# ── 全局变量 ───────────────────────────────────────────────────────────────
MODE=""
PYTHON_BIN=""

# smpl / robot 通用
MOTION_FILE=""
VIS_ROBOTS=""
SOURCE_FPS="30"
RENDER_FPS="30"
ORIGIN_ROBOT="g1"
RENDER_DEBUG="false"

# viser 专用
VISER_PORT="20006"
LOOP="false"
NO_GROUND="false"

# viser 动作-机器人映射 (由 scan_motions 填充)
declare -A MOTION_ROBOT_MAP

# 动作中文名称映射
declare -A MOTION_LABELS_CN=(
    ["Form_1_stageii"]="形体检查"
    ["dance1"]="舞蹈 1"
    ["dance2"]="舞蹈 2"
    ["fallAndGetUp1"]="跌倒起身 1"
    ["fallAndGetUp2"]="跌倒起身 2"
    ["fallAndGetUp3"]="跌倒起身 3"
    ["fight1"]="格斗 1"
    ["fightAndSports1"]="格斗运动 1"
    ["grab_walk_ff_180_001__A550"]="抓握行走"
    ["jumps1"]="跳跃 1"
    ["run1"]="跑步 1"
    ["run2"]="跑步 2"
    ["sprint1"]="冲刺 1"
    ["walk1"]="行走 1"
    ["walk2"]="行走 2"
    ["walk3"]="行走 3"
    ["walk4"]="行走 4"
    ["body_check_001__A548"]="形体检查"
)

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[1;36m'; BOLD='\033[1m'; NC='\033[0m'

log_banner() { echo -e "${BOLD}${BLUE}$1${NC}"; }
log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_cmd()   { echo -e "${CYAN}[CMD]${NC}   $1"; }

# ── Python 检测 ────────────────────────────────────────────────────────────
detect_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        : # 用户已指定
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        log_error "未找到 Python 解释器，请手动设置 PYTHON_BIN"
        exit 127
    fi

    if ! "${PYTHON_BIN}" --version >/dev/null 2>&1; then
        log_error "PYTHON_BIN 不是有效的 Python 解释器: ${PYTHON_BIN}"
        exit 127
    fi
}

# ── 检查关键依赖 ──────────────────────────────────────────────────────────
check_core_deps() {
    local missing=()
    "$PYTHON_BIN" -c "import mujoco" 2>/dev/null   || missing+=("mujoco")
    "$PYTHON_BIN" -c "import mink" 2>/dev/null     || missing+=("mink")
    "$PYTHON_BIN" -c "import numpy" 2>/dev/null    || missing+=("numpy")
    "$PYTHON_BIN" -c "import yaml" 2>/dev/null     || missing+=("PyYAML")
    if [ ${#missing[@]} -gt 0 ]; then
        log_warn "缺少核心依赖: ${missing[*]}"
        return 1
    fi
    return 0
}

# ── 列出可用机器人 ─────────────────────────────────────────────────────────
list_robots() {
    local config_dir="$PROJECT_DIR/config/robot"
    if [ ! -d "$config_dir" ]; then
        log_error "机器人配置目录不存在: $config_dir"
        return 1
    fi
    local robots=()
    for f in "$config_dir"/*.yaml; do
        [ -f "$f" ] || continue
        robots+=("$(basename "$f" .yaml)")
    done
    if [ ${#robots[@]} -eq 0 ]; then
        log_warn "未找到任何机器人配置"
        return 0
    fi
    echo ""
    log_banner "═══ 可用机器人 (共 ${#robots[@]} 个) ═══"
    echo ""
    for r in $(printf '%s\n' "${robots[@]}" | sort); do
        echo -e "  ${CYAN}●${NC} $r"
    done
    echo ""
}

# ── 列出可用动作文件 ──────────────────────────────────────────────────────
list_motions() {
    local motion_dir="$PROJECT_DIR/output_data/robot_motion"
    if [ ! -d "$motion_dir" ]; then
        log_warn "动作数据目录不存在: $motion_dir"
        log_info "请先运行 smpl/robot 模式生成动作数据"
        return 0
    fi
    local motions=()
    for f in "$motion_dir"/*.csv; do
        [ -f "$f" ] || continue
        motions+=("$(basename "$f" .csv)")
    done
    if [ ${#motions[@]} -eq 0 ]; then
        log_warn "未找到任何动作文件"
        return 0
    fi
    echo ""
    log_banner "═══ 可用动作 (共 ${#motions[@]} 个) ═══"
    echo ""
    for m in $(printf '%s\n' "${motions[@]}" | sort); do
        echo -e "  ${CYAN}●${NC} $m"
    done
    echo ""
}

# ── 交互式选择 (从列表中选择) ─────────────────────────────────────────────
prompt_select() {
    local prompt="$1"; shift
    local options=("$@")
    local depth="${PROMPT_DEPTH:-0}"
    if [ "$depth" -ge 3 ]; then
        log_warn "递归过深，使用默认 (1)" >&2
        echo "0"; return
    fi
    PROMPT_DEPTH=$((depth+1))
    echo -e "${BOLD}${prompt}${NC}" >&2
    for i in "${!options[@]}"; do
        echo -e "  ${CYAN}$((i+1))${NC}) ${options[$i]}" >&2
    done
    local choice
    if ! read -p "请选择 [1-${#options[@]}] (默认 1): " choice; then
        echo "0"; PROMPT_DEPTH=0; return
    fi
    if [ -z "$choice" ]; then echo "0"; PROMPT_DEPTH=0; return; fi
    if [[ ! "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#options[@]}" ]; then
        log_error "无效选择，请重新输入" >&2
        prompt_select "$prompt" "${options[@]}"
    else
        PROMPT_DEPTH=0
        echo "$((choice-1))"
    fi
}

# ── 交互式输入 ─────────────────────────────────────────────────────────────
prompt_input() {
    local p="$1" d="${2:-}" req="${3:-false}"
    while true; do
        if [ -n "$d" ]; then
            read -p "$p [$d]: " v
            v="${v:-$d}"
        else
            read -p "$p: " v
        fi
        [ "$req" = "true" ] && [ -z "$v" ] && { log_warn "此项必填"; continue; }
        echo "$v"; return
    done
}

# ── 是/否确认 ──────────────────────────────────────────────────────────────
prompt_yn() {
    local p="$1" d="${2:-y}"
    local v
    if [ "$d" = "y" ]; then
        read -p "$p [Y/n] (默认 Y): " v
        v="${v:-y}"
    else
        read -p "$p [y/N] (默认 N): " v
        v="${v:-n}"
    fi
    case "${v,,}" in
        y|yes) echo "true" ;;
        *)     echo "false" ;;
    esac
}

# ── 用法说明 ──────────────────────────────────────────────────────────────
show_usage() {
    cat <<EOF
用法:
  $0                              # 交互式启动
  $0 <mode> [args...]             # 非交互式启动

模式:
  smpl        从 SMPL-X 动作重定向到目标机器人
  robot       从源机器人动作重定向到目标机器人
  viser       使用 Viser 浏览器可视化已有动作
  list        列出可用机器人和动作
  doctor      环境健康检查

通用参数:
  --motion <path>          动作文件路径 (.npz 或 .csv)
  --robots <r1> [r2 ...]   目标机器人列表 (空格分隔)
  --source-fps <F>         源数据帧率 (默认: 30)
  --render-fps <F>         渲染帧率 (默认: 30)

smpl 模式:
  --motion <path>          SMPL-X 动作文件 (.npz，必需)
  --robots <r1> [r2 ...]    目标机器人列表

robot 模式:
  --motion <path>          源机器人动作文件 (.csv，必需)
  --origin <name>          源机器人名称 (默认: g1)
  --robots <r1> [r2 ...]    目标机器人列表

viser 模式:
  --motion <name>          动作名称 (不含机器人后缀，必需)
  --robots <r1> [r2 ...]    要显示的机器人列表
  --port <N>               Viser 端口 (默认: 20006)
  --loop                   循环播放
  --no-ground              不显示地面

示例:
  $0 list
  $0 doctor
  $0 smpl
  $0 smpl --motion dataset/ACCAD/Form_1_stageii.npz --robots g1 h2
 motion dataset/lafan1_g1/dance1_subject2.csv --origin g1 --robots h2
  $0 viser --motion Form_1_stageii --robots g1 h2 t800 --port 8080
EOF
}

# ══════════════════════════════════════════════════════════════════════════════
# 交互式模式配置
# ══════════════════════════════════════════════════════════════════════════════
select_mode() {
    local idx=$(prompt_select "请选择启动模式:" \
        "smpl   — SMPL-X → 机器人 重定向" \
        "robot  — 机器人 → 机器人 重定向" \
        "viser  — Viser 浏览器可视化" \
        "list   — 列出可用机器人和动作" \
        "doctor — 环境健康检查")
    case $idx in
        0) MODE="smpl" ;;
        1) MODE="robot" ;;
        2) MODE="viser" ;;
        3) MODE="list" ;;
        4) MODE="doctor" ;;
    esac
    log_info "已选择: $MODE"
}

# ── 扫描可用机器人 ─────────────────────────────────────────────────────────
scan_robots() {
    local config_dir="$PROJECT_DIR/config/robot"
    SCAN_ROBOT_ARRAY=()
    [ -d "$config_dir" ] || return 0
    for f in "$config_dir"/*.yaml; do
        [ -f "$f" ] || continue
        SCAN_ROBOT_ARRAY+=("$(basename "$f" .yaml)")
    done
}

# ── 交互式选择机器人 (多选) ──────────────────────────────────────────────
select_robots_multi() {
    scan_robots
    if [ ${#SCAN_ROBOT_ARRAY[@]} -eq 0 ]; then
        log_error "未找到任何机器人配置"
        exit 1
    fi
    SCAN_ROBOT_SORTED=("${SCAN_ROBOT_ARRAY[@]}")

    echo ""
    log_info "可用机器人:"
    local sorted=($(printf '%s\n' "${SCAN_ROBOT_ARRAY[@]}" | sort))
    for i in "${!sorted[@]}"; do
        echo -e "  ${CYAN}$((i+1))${NC}) ${sorted[$i]}"
    done

    echo ""
    echo -e "${BOLD}选择目标机器人 (输入编号，多个用空格分隔，回车=全部)${NC}"
    local choice
    read -p "选择: " choice

    SELECTED_ROBOTS=()
    if [ -z "$choice" ]; then
        SELECTED_ROBOTS=("${sorted[@]}")
    else
        for num in $choice; do
            if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "${#sorted[@]}" ]; then
                SELECTED_ROBOTS+=("${sorted[$((num-1))]}")
            else
                log_warn "忽略无效编号: $num"
            fi
        done
    fi

    if [ ${#SELECTED_ROBOTS[@]} -eq 0 ]; then
        log_error "未选择任何机器人"
        exit 1
    fi
    log_info "已选择: ${SELECTED_ROBOTS[*]}"
}

# ── 扫描可用 SMPL 动作文件 (.npz) ────────────────────────────────────────
scan_smpl_motions() {
    local motion_dir="$PROJECT_DIR/dataset/ACCAD"
    SCAN_SMPL_MOTION_ARRAY=()
    [ -d "$motion_dir" ] || return 0
    for f in "$motion_dir"/*.npz; do
        [ -f "$f" ] || continue
        SCAN_SMPL_MOTION_ARRAY+=("$(basename "$f")")
    done
    # 也检查其他 npz 目录
    for d in "$PROJECT_DIR"/dataset/*/; do
        [ -d "$d" ] || continue
        for f in "$d"/*.npz; do
            [ -f "$f" ] || continue
            local base; base="$(basename "$f")"
            local exists=false
            for m in "${SCAN_SMPL_MOTION_ARRAY[@]:-}"; do
                [ "$m" = "$base" ] && exists=true && break
            done
            [ "$exists" = false ] && SCAN_SMPL_MOTION_ARRAY+=("$base")
        done
    done
}

# ── smpl 模式交互配置 ─────────────────────────────────────────────────────
config_smpl() {
    echo ""
    log_banner "── SMPL-X → 机器人 重定向配置 ──"
    echo ""

    # 动作文件选择
    scan_smpl_motions
    if [ ${#SCAN_SMPL_MOTION_ARRAY[@]} -gt 0 ]; then
        echo -e "${BOLD}选择 SMPL-X 动作文件 (.npz):${NC}"
        local sorted_motions=($(printf '%s\n' "${SCAN_SMPL_MOTION_ARRAY[@]}" | sort))
        for i in "${!sorted_motions[@]}"; do
            echo -e "  ${CYAN}$((i+1))${NC}) ${sorted_motions[$i]}"
        done
        echo ""
        local idx
        idx=$(prompt_select "动作:" "${sorted_motions[@]}")
        MOTION_FILE="$PROJECT_DIR/dataset/ACCAD/${sorted_motions[$idx]}"
        # 如果 ACCAD 目录没有，搜索其他目录
        [ ! -f "$MOTION_FILE" ] && MOTION_FILE="$PROJECT_DIR/dataset/${sorted_motions[$idx]}"
        [ ! -f "$MOTION_FILE" ] && MOTION_FILE="${sorted_motions[$idx]}"
        log_info "已选择动作: $MOTION_FILE"
    else
        MOTION_FILE=$(prompt_input "SMPL-X 动作文件 (.npz 完整路径)" "" true)
    fi

    # 目标机器人
    echo ""
    select_robots_multi

    # 帧率
    echo ""
    SOURCE_FPS=$(prompt_input "源数据帧率" "30")
    RENDER_FPS=$(prompt_input "渲染帧率" "30")

    # 调试渲染
    if [ "$(prompt_yn "启用调试渲染 (显示关键点匹配)?" "n")" = "true" ]; then
        RENDER_DEBUG="true"
    fi
}

# ── 扫描可用机器人动作文件 (.csv) ────────────────────────────────────────
scan_robot_motions() {
    local motion_dir="$PROJECT_DIR/dataset"
    SCAN_ROBOT_MOTION_ARRAY=()
    [ -d "$motion_dir" ] || return 0
    for f in "$motion_dir"/**/*.csv; do
        [ -f "$f" ] || continue
        local rel="${f#$PROJECT_DIR/}"
        SCAN_ROBOT_MOTION_ARRAY+=("$rel")
    done
}

# ── robot 模式交互配置 ─────────────────────────────────────────────────────
config_robot() {
    echo ""
    log_banner "── 机器人 → 机器人 重定向配置 ──"
    echo ""

    # 动作文件选择
    scan_robot_motions
    if [ ${#SCAN_ROBOT_MOTION_ARRAY[@]} -gt 0 ]; then
        echo -e "${BOLD}选择源机器人动作文件 (.csv):${NC}"
        local sorted_motions=($(printf '%s\n' "${SCAN_ROBOT_MOTION_ARRAY[@]}" | sort))
        for i in "${!sorted_motions[@]}"; do
            echo -e "  ${CYAN}$((i+1))${NC}) ${sorted_motions[$i]}"
        done
        echo ""
        local idx
        idx=$(prompt_select "动作:" "${sorted_motions[@]}")
        MOTION_FILE="$PROJECT_DIR/${sorted_motions[$idx]}"
        log_info "已选择动作: $MOTION_FILE"
    else
        MOTION_FILE=$(prompt_input "源机器人动作文件 (.csv 完整路径)" "" true)
    fi

    # 源机器人
    echo ""
    local default_origin="g1"
    ORIGIN_ROBOT=$(prompt_input "源机器人名称" "$default_origin")

    # 目标机器人
    echo ""
    select_robots_multi

    # 帧率
    echo ""
    SOURCE_FPS=$(prompt_input "源数据帧率" "30")
    RENDER_FPS=$(prompt_input "渲染帧率" "30")

    # 调试渲染
    if [ "$(prompt_yn "启用调试渲染 (显示关键点匹配)?" "n")" = "true" ]; then
        RENDER_DEBUG="true"
    fi
}

# ── 扫描已有动作文件 ─────────────────────────────────────────────────────
# 扫描以下目录:
#   1. output_data/robot_motion/  (重定向后的动作)
#   2. dataset/lafan1_g1/         (预生成的机器人动作，已含机器人后缀)
# 注意: MOTION_ROBOT_MAP 必须在调用 scan_motions 之前用 declare -A 声明
scan_motions() {
    SCAN_MOTION_ARRAY=()

    # 辅助函数: 将 "xxx_robotname" 解析为 "动作名" + "机器人名"
    _add_motion_and_robot() {
        local base="$1"  # e.g. "Form_1_stageii_g1"
        # 从末尾提取已知的机器人名后缀
        local motion_name="" robot_name=""
        for r in "${SCAN_ROBOT_ARRAY[@]:-}"; do
            local suffix="_${r}"
            if [[ "$base" == *"${suffix}" ]]; then
                motion_name="${base%${suffix}}"
                robot_name="$r"
                break
            fi
        done
        # 如果没匹配到已知机器人，尝试去掉最后一个 _xxx
        if [ -z "$motion_name" ]; then
            motion_name="${base%_*}"
            robot_name="${base##*_}"
        fi
        # 去重添加动作
        local exists=false
        for m in "${SCAN_MOTION_ARRAY[@]:-}"; do
            [ "$m" = "$motion_name" ] && exists=true && break
        done
        [ "$exists" = false ] && SCAN_MOTION_ARRAY+=("$motion_name")
        # 记录动作对应的机器人
        MOTION_ROBOT_MAP["$motion_name"]="$robot_name"
    }

    # 扫描 output_data/robot_motion/
    local motion_dir="$PROJECT_DIR/output_data/robot_motion"
    if [ -d "$motion_dir" ]; then
        for f in "$motion_dir"/*.csv; do
            [ -f "$f" ] || continue
            local base; base="$(basename "$f" .csv)"
            _add_motion_and_robot "$base"
        done
    fi

    # 扫描 dataset/lafan1_g1/ (预生成数据)
    local lafan_dir="$PROJECT_DIR/dataset/lafan1_g1"
    if [ -d "$lafan_dir" ]; then
        for f in "$lafan_dir"/*.csv; do
            [ -f "$f" ] || continue
            local base; base="$(basename "$f" .csv)"
            _add_motion_and_robot "$base"
        done
    fi

    # 扫描 dataset/bones_g1/
    local bones_dir="$PROJECT_DIR/dataset/bones_g1"
    if [ -d "$bones_dir" ]; then
        for f in "$bones_dir"/*.csv; do
            [ -f "$f" ] || continue
            local base; base="$(basename "$f" .csv)"
            _add_motion_and_robot "$base"
        done
    fi
}

# ── viser 模式交互配置 ─────────────────────────────────────────────────────
config_viser() {
    echo ""
    log_banner "── Viser 浏览器可视化配置 ──"
    echo ""

    # 动作选择
    scan_motions
    if [ ${#SCAN_MOTION_ARRAY[@]} -eq 0 ]; then
        log_warn "未找到任何动作数据，请先运行 smpl/robot 模式生成"
        echo ""
        MOTION_FILE=$(prompt_input "或手动输入动作名称" "" true)
        echo ""
        select_robots_multi
    else
        echo -e "${BOLD}选择要可视化的动作:${NC}"
        local sorted_motions=($(printf '%s\n' "${SCAN_MOTION_ARRAY[@]}" | sort))
        for i in "${!sorted_motions[@]}"; do
            local m="${sorted_motions[$i]}"
            local cn="${MOTION_LABELS_CN[$m]:-}"
            local default_robot="${MOTION_ROBOT_MAP[$m]:-}"
            local display="${m}"
            [ -n "$cn" ] && display="${m}  ${CYAN}(${cn})${NC}"
            if [ -n "$default_robot" ]; then
                echo -e "  ${CYAN}$((i+1))${NC}) ${display}  ${GREEN}(默认机器人: ${default_robot})${NC}"
            else
                echo -e "  ${CYAN}$((i+1))${NC}) ${display}"
            fi
        done
        echo ""
        local idx
        idx=$(prompt_select "动作:" "${sorted_motions[@]}")
        MOTION_FILE="${sorted_motions[$idx]}"
        log_info "已选择动作: $MOTION_FILE"

        # 目标机器人 - 自动预选该动作对应的机器人
        local preselected_robot="${MOTION_ROBOT_MAP[$MOTION_FILE]:-}"
        echo ""
        if [ -n "$preselected_robot" ]; then
            log_info "该动作已有预生成数据，默认机器人: $preselected_robot"
            if [ "$(prompt_yn "是否只显示该机器人? (否=选择其他)" "y")" = "true" ]; then
                SELECTED_ROBOTS=("$preselected_robot")
            else
                select_robots_multi
            fi
        else
            select_robots_multi
        fi
    fi

    # 端口
    echo ""
    VISER_PORT=$(prompt_input "Viser 端口" "20006")

    # 循环
    if [ "$(prompt_yn "循环播放?" "n")" = "true" ]; then
        LOOP="true"
    fi

    # 地面
    if [ "$(prompt_yn "显示地面?" "y")" = "false" ]; then
        NO_GROUND="true"
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
# 命令构建与执行
# ══════════════════════════════════════════════════════════════════════════════
build_and_run() {
    case "$MODE" in
        smpl)
            # 检查动作文件
            if [ ! -f "$MOTION_FILE" ]; then
                log_error "动作文件不存在: $MOTION_FILE"
                exit 1
            fi

            local robot_list="${SELECTED_ROBOTS[*]:-$VIS_ROBOTS}"
            local num_robots
            read -r -a robot_arr <<< "$robot_list"
            num_robots=${#robot_arr[@]}

            echo ""
            log_banner "══════════════════ SMPL-X → 机器人 ══════════════════"
            log_info "动作文件: $MOTION_FILE"
            log_info "目标机器人: ${robot_arr[*]}"
            log_info "源帧率: $SOURCE_FPS, 渲染帧率: $RENDER_FPS"
            echo ""

            for idx in "${!robot_arr[@]}"; do
                local robot="${robot_arr[$idx]}"
                local robot_config="config/robot/${robot}.yaml"

                if [ ! -f "$PROJECT_DIR/$robot_config" ]; then
                    log_error "机器人配置不存在: $robot_config"
                    exit 1
                fi

                echo ""
                log_info "═══ [机器人 $((idx+1))/${num_robots}] ${robot} ═══"

                # Step 1: smpl_replay
                echo "[1/3] smpl_replay (提取关键点)"
                "$PYTHON_BIN" scripts/smpl_replay.py \
                    --no-viewer \
                    --motion_file "$MOTION_FILE" \
                    --robot-config "$robot_config" \
                    --skeleton-config config/skeleton/skeleton.yaml \
                    --fps "$SOURCE_FPS"

                # Step 2: robot_retarget
                echo "[2/3] robot_retarget (逆运动学)"
                local retarget_args=(--config "$robot_config")
                local keypoints_name
                keypoints_name="$(basename "${MOTION_FILE}" .npz)"
                retarget_args+=(--keypoints-name "$keypoints_name")

                if [ "$RENDER_DEBUG" = "true" ]; then
                    retarget_args+=(--render-debug)
                else
                    retarget_args+=(--no-render-debug)
                fi

                "$PYTHON_BIN" scripts/robot_retarget.py "${retarget_args[@]}"
            done

            # Step 3: 可视化
            echo ""
            log_info "[3/3] 可视化"
            local motion_name
            motion_name="$(basename "${MOTION_FILE}" .npz)"
            local viser_args=(
                --motion "$motion_name"
                --robots "${robot_arr[@]}"
                --source-fps "$SOURCE_FPS"
                --render-fps "$RENDER_FPS"
                --port "$VISER_PORT"
            )
            [ "$LOOP" = "true" ] && viser_args+=(--loop)
            [ "$NO_GROUND" = "true" ] && viser_args+=(--no-ground)

            log_info "启动 Viser 可视化..."
            echo ""
            exec "$PYTHON_BIN" scripts/multi_robot_visualize_viser.py "${viser_args[@]}"
            ;;

        robot)
            # 检查动作文件
            if [ ! -f "$MOTION_FILE" ]; then
                log_error "动作文件不存在: $MOTION_FILE"
                exit 1
            fi

            local robot_list="${SELECTED_ROBOTS[*]:-$VIS_ROBOTS}"
            local num_robots
            read -r -a robot_arr <<< "$robot_list"
            num_robots=${#robot_arr[@]}

            local source_config="config/robot/${ORIGIN_ROBOT}.yaml"
            if [ ! -f "$PROJECT_DIR/$source_config" ]; then
                log_error "源机器人配置不存在: $source_config"
                exit 1
            fi

            echo ""
            log_banner "══════════════════ 机器人 → 机器人 ══════════════════"
            log_info "动作文件: $MOTION_FILE"
            log_info "源机器人: $ORIGIN_ROBOT"
            log_info "目标机器人: ${robot_arr[*]}"
            log_info "源帧率: $SOURCE_FPS, 渲染帧率: $RENDER_FPS"
            echo ""

            for idx in "${!robot_arr[@]}"; do
                local robot="${robot_arr[$idx]}"
                local robot_config="config/robot/${robot}.yaml"

                if [ ! -f "$PROJECT_DIR/$robot_config" ]; then
                    log_error "机器人配置不存在: $robot_config"
                    exit 1
                fi

                echo ""
                log_info "═══ [机器人 $((idx+1))/${num_robots}] ${robot} ═══"

                # Step 1: robot_replay
                echo "[1/3] robot_replay (提取关键点)"
                "$PYTHON_BIN" scripts/robot_replay.py \
                    --no-viewer \
                    --motion-file "$MOTION_FILE" \
                    --source-robot-config "$source_config" \
                    --target-robot-config "$robot_config" \
                    --fps "$SOURCE_FPS"

                # Step 2: robot_retarget
                echo "[2/3] robot_retarget (逆运动学)"
                local retarget_args=(--config "$robot_config")
                local keypoints_name
                keypoints_name="$(basename "${MOTION_FILE}" .csv)"
                retarget_args+=(--keypoints-name "${keypoints_name}_from_${ORIGIN_ROBOT}")

                if [ "$RENDER_DEBUG" = "true" ]; then
                    retarget_args+=(--render-debug)
                else
                    retarget_args+=(--no-render-debug)
                fi

                "$PYTHON_BIN" scripts/robot_retarget.py "${retarget_args[@]}"
            done

            # Step 3: 可视化
            echo ""
            log_info "[3/3] 可视化"
            local motion_name
            motion_name="$(basename "${MOTION_FILE}" .csv)_from_${ORIGIN_ROBOT}"
            local viser_args=(
                --motion "$motion_name"
                --robots "${robot_arr[@]}"
                --source-fps "$SOURCE_FPS"
                --render-fps "$RENDER_FPS"
                --port "$VISER_PORT"
            )
            [ "$LOOP" = "true" ] && viser_args+=(--loop)
            [ "$NO_GROUND" = "true" ] && viser_args+=(--no-ground)

            log_info "启动 Viser 可视化..."
            echo ""
            exec "$PYTHON_BIN" scripts/multi_robot_visualize_viser.py "${viser_args[@]}"
            ;;

        viser)
            local robot_list="${SELECTED_ROBOTS[*]:-$VIS_ROBOTS}"
            read -r -a robot_arr <<< "$robot_list"

            # 查找每个机器人的动作 CSV 文件
            local motion_dir_arg=""
            local found_all=true
            for robot in "${robot_arr[@]}"; do
                local csv_found=""
                # 1. output_data/robot_motion/
                if [ -f "$PROJECT_DIR/output_data/robot_motion/${MOTION_FILE}_${robot}.csv" ]; then
                    csv_found="$PROJECT_DIR/output_data/robot_motion"
                # 2. dataset/lafan1_g1/
                elif [ -f "$PROJECT_DIR/dataset/lafan1_g1/${MOTION_FILE}_${robot}.csv" ]; then
                    csv_found="$PROJECT_DIR/dataset/lafan1_g1"
                # 3. dataset/bones_g1/
                elif [ -f "$PROJECT_DIR/dataset/bones_g1/${MOTION_FILE}_${robot}.csv" ]; then
                    csv_found="$PROJECT_DIR/dataset/bones_g1"
                fi
                if [ -z "$csv_found" ]; then
                    log_warn "未找到 ${MOTION_FILE}_${robot}.csv"
                    found_all=false
                else
                    log_info "找到 ${robot} 动作: ${csv_found}/${MOTION_FILE}_${robot}.csv"
                    # 使用第一个找到的目录作为 motion_dir
                    [ -z "$motion_dir_arg" ] && motion_dir_arg="$csv_found"
                fi
            done

            if [ "$found_all" = false ]; then
                log_error "部分机器人动作文件缺失，请先运行 smpl/robot 模式生成"
                exit 1
            fi

            echo ""
            log_banner "══════════════════ Viser 可视化 ══════════════════"
            log_info "动作名称: $MOTION_FILE"
            log_info "机器人: ${robot_arr[*]}"
            log_info "端口: $VISER_PORT"
            log_info "数据目录: $motion_dir_arg"
            echo ""

            local viser_args=(
                --motion "$MOTION_FILE"
                --robots "${robot_arr[@]}"
                --motion_dir "$motion_dir_arg"
                --port "$VISER_PORT"
            )
            [ "$LOOP" = "true" ] && viser_args+=(--loop)
            [ "$NO_GROUND" = "true" ] && viser_args+=(--no-ground)

            exec "$PYTHON_BIN" scripts/multi_robot_visualize_viser.py "${viser_args[@]}"
            ;;

        list)
            echo ""
            log_banner "══════════════════ 资源列表 ══════════════════"
            list_robots
            list_motions
            ;;

        doctor)
            echo ""
            log_banner "══════════════════ 环境健康检查 ══════════════════"
            echo ""
            echo -e "  Python:  $($PYTHON_BIN --version 2>&1)"
            echo -e "  路径:    $PYTHON_BIN"
            echo -e "  项目:    $PROJECT_DIR"
            echo ""

            log_banner "═══ 核心依赖 ═══"
            for pkg in mujoco mink numpy yaml torch smplx trimesh scipy tqdm viser; do
                local import_pkg="$pkg"
                [ "$pkg" = "yaml" ] && import_pkg="yaml"

                local ver
                ver=$("$PYTHON_BIN" -c "
try:
    m = __import__('${import_pkg}')
    print(getattr(m, '__version__', 'OK'))
except Exception as e:
    print(f'✗ {e}')
" 2>/dev/null)
                local status="✓"
                [[ "$ver" == *"✗"* ]] && status="�"
                printf "  %-12s %s %s\n" "$pkg:" "$status" "$ver"
            done
            echo ""

            log_banner "═══ MuJoCo 后端 ═══"
            "$PYTHON_BIN" -c "
import mujoco
print(f'  MuJoCo 版本: {mujoco.__version__}')
spec = mujoco.MjSpec()
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0,0,0.01])
model = spec.compile()
print(f'  渲染后端: 可用')
" 2>/dev/null || log_warn "MuJoCo 后端检查失败"
            echo ""

            log_banner "═══ 机器人配置 ═══"
            local config_dir="$PROJECT_DIR/config/robot"
            if [ -d "$config_dir" ]; then
                local count=$(find "$config_dir" -name "*.yaml" | wc -l)
                echo -e "  配置文件: ${count} 个"
                for f in $(ls "$config_dir"/*.yaml 2>/dev/null | sort); do
                    local name=$(basename "$f" .yaml)
                    local xml_path
                    xml_path=$(grep -E '^robot_xml_path:' "$f" 2>/dev/null | head -1 | sed 's/.*robot_xml_path: *//' | tr -d '"' | tr -d "'")
                    local xml_status="✓"
                    [ -n "$xml_path" ] && [ ! -f "$PROJECT_DIR/$xml_path" ] && xml_status="✗ 模型文件缺失"
                    [ -z "$xml_path" ] && xml_status="✗ xml路径未配置"
                    printf "  %-20s  %s\n" "$name" "$xml_status"
                done
            fi
            echo ""

            log_banner "═══ 示例数据 ═══"
            for d in dataset/ACCAD dataset/lafan1_g1 dataset/bones_g1; do
                if [ -d "$PROJECT_DIR/$d" ]; then
                    local count=$(find "$PROJECT_DIR/$d" -type f | wc -l)
                    echo -e "  $d: ${count} 个文件"
                else
                    echo -e "  $d: 不存在"
                fi
            done
            echo ""

            log_info "环境健康检查完成"
            ;;
    esac
}

# ══════════════════════════════════════════════════════════════════════════════
# 配置确认
# ══════════════════════════════════════════════════════════════════════════════
confirm_config() {
    echo ""
    log_banner "══════════════════ 配置确认 ══════════════════"
    echo ""

    case "$MODE" in
        smpl)
            echo -e "  模式:       ${BOLD}smpl${NC} (SMPL-X → 机器人)"
            echo -e "  动作文件:   ${BOLD}${MOTION_FILE}${NC}"
            echo -e "  目标机器人: ${BOLD}${OTS[*]}${NC}"
            echo -e "  帧率:       ${BOLD}源${SOURCE_FPS} / 渲染${RENDER_FPS}${NC}"
            echo -e "  调试渲染:   ${BOLD}${RENDER_DEBUG}${NC}"
            ;;
        robot)
            echo -e "  模式:       ${BOLD}robot${NC} (机器人 → 机器人)"
            echo -e "  动作文件:   ${BOLD}${MOTION_FILE}${NC}"
            echo -e "  源机器人:   ${BOLD}${ORIGIN_ROBOT}${NC}"
            echo -e "  目标机器人: ${BOLD}${SELECTED_ROBOTS[*]}${NC}"
            echo -e "  帧率:       ${BOLD}源${SOURCE_FPS} / 渲染${RENDER_FPS}${NC}"
            echo -e "  调试渲染:   ${BOLD}${RENDER_DEBUG}${NC}"
            ;;
        viser)
            echo -e "  模式:       ${BOLD}viser${NC} (浏览器可视化)"
            echo -e "  动作名称:   ${BOLD}${MOTION_FILE}${NC}"
            echo -e "  机器人:     ${BOLD}${SELECTED_ROBOTS[*]}${NC}"
            echo -e "  端口:       ${BOLD}${VISER_PORT}${NC}"
            echo -e "  循环:       ${BOLD}${LOOP}${NC}"
            echo -e "  地面:       ${BOLD}$([ "$NO_GROUND" = "true" ] && echo "隐藏" || echo "显示")${NC}"
            ;;
        list)
            echo -e "  模式:       ${BOLD}list${NC}"
            ;;
        doctor)
            echo -e "  模式:       ${BOLD}doctor${NC}"
            ;;
    esac
    echo -e "  Python:     ${BOLD}${PYTHON_BIN}${NC}"
    echo ""
}

# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════
detect_python

if [ $# -eq 0 ]; then
    # ── 交互式 ──
    log_banner "═══════════════════════════════════════════════════════════════"
    log_banner "          robot_retargeter — 启动入口"
    log_banner "═══════════════════════════════════════════════════════════════"
    echo ""
    log_info "Python: $PYTHON_BIN"
    echo ""
    check_core_deps || log_warn "依赖不全，可用: $0 doctor"
    echo ""
    select_mode
    case "$MODE" in
        smpl)    config_smpl ;;
        robot)   config_robot ;;
        viser)   config_viser ;;
        list)    ;;
        doctor)  ;;
    esac
    confirm_config

    # 确认执行
    if [ "$MODE" != "list" ] && [ "$MODE" != "doctor" ]; then
        if [ "$(prompt_yn "确认执行?" "y")" != "true" ]; then
            log_info "已取消"
            exit 0
        fi
    fi

    build_and_run
else
    # ── 非交互式 ──
    case "${1:-}" in
        smpl|robot|viser|list|doctor) MODE="$1"; shift ;;
        -h|--help|help) show_usage; exit 0 ;;
        *) log_error "未知模式: $1"; show_usage; exit 1 ;;
    esac

    # 解析参数
    VIS_ROBOTS=""
    SELECTED_ROBOTS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            # 通用
            --motion)       MOTION_FILE="$2"; shift 2 ;;
            --robots)
                # 支持 --robots g1 h2 t800 或 --robots "g1 h2 t800"
                shift
                while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                    SELECTED_ROBOTS+=("$1")
                    shift
                done
                ;;
            --source-fps)   SOURCE_FPS="$2"; shift 2 ;;
            --render-fps)   RENDER_FPS="$2"; shift 2 ;;
            # smpl/robot
            --origin)       ORIGIN_ROBOT="$2"; shift 2 ;;
            --debug)        RENDER_DEBUG="true"; shift 1 ;;
            # viser
            --port)         VISER_PORT="$2"; shift 2 ;;
            --loop)         LOOP="true"; shift 1 ;;
            --no-ground)     NO_GROUND="true"; shift 1 ;;
            # help
            -h|--help|help) show_usage; exit 0 ;;
            *)
                log_warn "忽略未知参数: $1"; shift 1 ;;
        esac
    done

    # 校验必需参数
    case "$MODE" in
        smpl|robot)
            if [ -z "$MOTION_FILE" ]; then
                log_error "$MODE 模式需要 --motion 参数"
                show_usage; exit 1
            fi
            if [ ${#SELECTED_ROBOTS[@]} -eq 0 ]; then
                log_error "$MODE 模式需要 --robots 参数"
                show_usage; exit 1
            fi
            ;;
        viser)
            if [ -z "$MOTION_FILE" ]; then
                log_error "viser 模式需要 --motion 参数 (动作名称)"
                show_usage; exit 1
            fi
            if [ ${#SELECTED_ROBOTS[@]} -eq 0 ]; then
                log_error "viser 模式需要 --robots 参数"
                show_usage; exit 1
            fi
            ;;
    esac

    confirm_config
    build_and_run
fi
