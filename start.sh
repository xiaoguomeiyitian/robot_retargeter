#!/usr/bin/env bash
# ============================================================================
# robot_retargeter 启动入口脚本
#
# 交互式:   ./start.sh
# 非交互:   ./start.sh <mode> [args...]
#
# 模式:
#   viser       使用 Viser 浏览器可视化 (空场景启动，浏览器中动态添加机器人)
#   mujoco      使用 MuJoCo 原生可视化
#   smpl        从 SMPL-X 动作重定向到机器人 (SMPL-X → 机器人)
#   robot       从源机器人动作重定向到目标机器人 (机器人 → 机器人)
#   rl          NPZ 导出与 RL 训练流水线 (CSV → NPZ → train)
#   list        列出所有可用机器人和动作
#   doctor      环境健康检查
#
# 用法示例:
#   ./start.sh                                       # 交互式
#   ./start.sh list                                  # 列出可用机器人
#   ./start.sh smpl                                  # 交互式 SMPL-X 重定向
#   ./start.sh smpl --motion dataset/ACCAD/Form_1_stageii.npz --robots g1 h2
#   ./start.sh smpl --robots g1 h2 --robot-motion g1:dataset/ACCAD/Form_1.npz --robot-motion h2:dataset/ACCAD/Form_2.npz
#   ./start.sh robot --motion dataset/lafan1_g1/dance1_subject2.csv --origin g1 --robots h2 r1
#   ./start.sh robot --origin g1 --robots g1 h2 --robot-motion g1:dataset/lafan1_g1/dance1.csv --robot-motion h2:dataset/lafan1_g1/dance2.csv
#   ./start.sh viser --port 8080                          # 空场景启动，浏览器中动态添加
#   ./start.sh mujoco --motion Form_1_stageii --robots g1 h2 t800
#   ./start.sh rl --robot g1 --motion dataset/lafan1_g1/dance1_subject2.csv --rl-task unitree_g1_flat_tracking
#   ./start.sh rl --robot g1 --motion output_data/robot_motion/Form_1_stageii_g1.csv --export-only
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

# 每个机器人对应的动作文件 (robot:motion 模式)
# 在交互模式下由 select_robots_with_motions 填充
declare -A ROBOT_MOTIONS
# 每个机器人 key 对应的真实机器人名 (去掉 __N 后缀)
declare -A ROBOT_REAL_NAMES

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

# 机器人中文名称映射 (与 viser 界面 ROBOT_LABELS_CN 保持一致)
declare -A ROBOT_LABELS_CN=(
    ["agibot_x2"]="艾博特 X2"
    ["booster_t1"]="Booster T1"
    ["DR02"]="DR02"
    ["g1"]="G1 人形"
    ["g1_d"]="G1D"
    ["h1"]="H1 人形"
    ["h1_2"]="H1-2 人形"
    ["h2"]="H2 人形"
    ["hightorque_hi"]="高力矩 HI"
    ["hightorque_pi"]="高力矩 PI"
    ["jaka_pi"]="Jaka PI"
    ["limx_oli"]="LIMX OLI"
    ["noetix_e1"]="Noetix E1"
    ["noetix_n2"]="Noetix N2"
    ["pm01"]="PM01"
    ["pnd_adam"]="PND Adam"
    ["r1"]="R1 人形"
    ["t800"]="T800"
    ["tienkung"]="天坤"
    ["unitree_a2"]="宇树 A2"
    ["unitree_a2w"]="宇树 A2W"
    ["xbot"]="XBot"
)

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[1;36m'; BOLD='\033[1m'; NC='\033[0m'

log_banner() { echo -e "${BOLD}${BLUE}$1${NC}"; }
log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_cmd()   { echo -e "${CYAN}[CMD]${NC}   $1"; }

# ── 中文名辅助函数 ────────────────────────────────────────────────────────
# 获取机器人显示标签 (英文名 + 中文名)
get_robot_label_cn() {
    local name="$1"
    local cn="${ROBOT_LABELS_CN[$name]:-}"
    if [ -n "$cn" ] && [ "$cn" != "$name" ]; then
        echo "${name} (${cn})"
    else
        echo "$name"
    fi
}

# 获取动作显示标签 (中文名 + 英文名)
# 支持完整路径输入 (如 dataset/lafan1_g1/dance1_subject2.csv)，
# 自动提取短名称匹配中文名映射
get_motion_label_cn() {
    local name="$1"
    local cn="${MOTION_LABELS_CN[$name]:-}"
    if [ -n "$cn" ]; then
        echo "${cn} (${name})"
        return
    fi
    # 尝试从完整路径提取短名称: 去掉路径和后缀，再去掉 _subjectN / _M 等后缀
    local base; base="$(basename "$name" .csv)"
    # 去掉 _subjectN 后缀
    local stem="${base%%_subject[0-9]*}"
    # 去掉 _M 后缀 (bones_g1 格式)
    [ "$stem" = "$base" ] && stem="${base%%_M}"
    # 去掉 _from_xxx 后缀
    stem="${stem%%_from_*}"
    # 去掉已知的机器人名后缀 (如 _g1, _h2 等)
    for r in "${!ROBOT_LABELS_CN[@]}"; do
        [ "$stem" != "${stem%_${r}}" ] && stem="${stem%_${r}}" && break
    done
    cn="${MOTION_LABELS_CN[$stem]:-}"
    if [ -n "$cn" ]; then
        echo "${cn} (${name})"
    else
        echo "$name"
    fi
}

# ── Python 检测 ────────────────────────────────────────────────────────────
detect_python() {
    # 优先使用项目 .venv 中的 Python
    local venv_python="$PROJECT_DIR/.venv/bin/python"
    # 备选: 同级 trainBot 项目的 venv
    local trainbot_venv="$(cd "$PROJECT_DIR/../trainBot" 2>/dev/null && pwd)/.venv/bin/python"
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        : # 用户已显式指定
    elif [[ -x "$venv_python" ]]; then
        PYTHON_BIN="$venv_python"
        log_info "使用项目虚拟环境: $PYTHON_BIN"
    elif [[ -x "$trainbot_venv" ]]; then
        PYTHON_BIN="$trainbot_venv"
        log_info "使用 trainBot 虚拟环境: $PYTHON_BIN"
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
        local label; label="$(get_robot_label_cn "$r")"
        echo -e "  ${CYAN}●${NC} ${label}"
    done
    echo ""
}

# ── 列出可用动作文件 ──────────────────────────────────────────────────────
# 扫描以下目录 (与 viser 模式的 scan_motions 保持一致):
#   1. output_data/robot_motion/  (重定向后的动作)
#   2. dataset/lafan1_g1/         (预生成的机器人动作)
#   3. dataset/bones_g1/          (预生成的机器人动作)
list_motions() {
    local -A MOTION_SET=()
    local -A MOTION_ROBOT_MAP_LOCAL=()
    local scan_dirs=(
        "$PROJECT_DIR/output_data/robot_motion"
        "$PROJECT_DIR/dataset/lafan1_g1"
        "$PROJECT_DIR/dataset/bones_g1"
    )

    # 辅助函数: 解析 "动作名_机器人名" 并去重
    _add_motion() {
        local base="$1"
        local motion_name="" robot_name=""
        for r in "${SCAN_ROBOT_ARRAY[@]:-}"; do
            local suffix="_${r}"
            if [[ "$base" == *"${suffix}" ]]; then
                motion_name="${base%${suffix}}"
                robot_name="$r"
                break
            fi
        done
        if [ -z "$motion_name" ]; then
            motion_name="${base%_*}"
            robot_name="${base##*_}"
        fi
        if [ -z "${MOTION_SET[$motion_name]:-}" ]; then
            MOTION_SET["$motion_name"]="$robot_name"
        fi
    }

    for dir in "${scan_dirs[@]}"; do
        [ -d "$dir" ] || continue
        for f in "$dir"/*.csv; do
            [ -f "$f" ] || continue
            local base; base="$(basename "$f" .csv)"
            _add_motion "$base"
        done
    done

    local motions=("${!MOTION_SET[@]}")
    if [ ${#motions[@]} -eq 0 ]; then
        log_warn "未找到任何动作文件"
        log_info "请先运行 smpl/robot 模式生成动作数据"
        return 0
    fi
    echo ""
    log_banner "═══ 可用动作 (共 ${#motions[@]} 个) ═══"
    echo ""
    for m in $(printf '%s\n' "${motions[@]}" | sort); do
        local robot="${MOTION_SET[$m]:-}"
        local mlabel; mlabel="$(get_motion_label_cn "$m")"
        if [ -n "$robot" ]; then
            local rlabel; rlabel="$(get_robot_label_cn "$robot")"
            echo -e "  ${CYAN}●${NC} ${mlabel}  ${GREEN}(默认机器人: ${rlabel})${NC}"
        else
            echo -e "  ${CYAN}●${NC} ${mlabel}"
        fi
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
        read -p "$p [y/n] (默认 y): " v
        v="${v:-y}"
    else
        read -p "$p [y/n] (默认 n): " v
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
  mujoco      使用 MuJoCo 原生可视化已有动作
  rl          NPZ 导出与 RL 训练流水线 (CSV → NPZ → train)
  list        列出可用机器人和动作
  doctor      环境健康检查

通用参数:
  --motion <path>              动作文件路径 (.npz 或 .csv)
  --robots <r1> [r2 ...]       目标机器人列表 (空格分隔)
  --robot-motion <r>:<motion>  为指定机器人设置动作文件 (可多次使用)
                                示例: --robot-motion g1:dataset/ACCAD/Form_1.npz --robot-motion h2:dataset/ACCAD/Form_2.npz
  --source-fps <F>             源数据帧率 (默认: 30)
  --render-fps <F>             渲染帧率 (默认: 30)

smpl 模式:
  --motion <path>              SMPL-X 动作文件 (.npz，必需)
  --robots <r1> [r2 ...]        目标机器人列表

robot 模式:
  --motion <path>              源机器人动作文件 (.csv，必需)
  --origin <name>              源机器人名称 (默认: g1)
  --robots <r1> [r2 ...]        目标机器人列表

viser 模式:
  --motion <name>              动作名称 (不含机器人后缀，必需)
  --robots <r1> [r2 ...]        要显示的机器人列表
  --port <N>                   Viser 端口 (默认: 20006)
  --loop                       循环播放
  --no-ground                   不显示地面

mujoco 模式:
  --motion <name>              动作名称 (不含机器人后缀，必需)
  --robots <r1> [r2 ...]        要显示的机器人列表
  --source-fps <F>             源数据帧率 (默认: 30)
  --render-fps <F>             渲染帧率 (默认: 60)
  --loop                       循环播放

rl 模式:
  --robot <name>               机器人名称 (必需)
  --motion <path>              输入动作文件 (.csv)
  --rl-task <task>              RL 任务 ID (如 unitree_g1_flat_tracking)
  --rl-root <path>              unitree_rl_mjlab 路径 (默认: ../unitree_rl_mjlab)
  --input-fps <F>              输入帧率 (默认: 30)
  --output-fps <F>             输出帧率 (默认: 50)
  --export-only                 仅导出 NPZ，不启动训练
  --train-args <str>            传递给 train.py 的额外参数

示例:
  $0 list
  $0 doctor
  $0 smpl
  $0 smpl --motion dataset/ACCAD/Form_1_stageii.npz --robots g1 h2
  $0 smpl --robots g1 h2 --robot-motion g1:dataset/ACCAD/Form_1.npz --robot-motion h2:dataset/ACCAD/Form_2.npz
  $0 robot --motion dataset/lafan1_g1/dance1_subject2.csv --origin g1 --robots h2 r1
  $0 robot --origin g1 --robots g1 h2 --robot-motion g1:dataset/lafan1_g1/dance1.csv --robot-motion h2:dataset/lafan1_g1/dance2.csv
  $0 viser --motion Form_1_stageii --robots g1 h2 t800 --port 8080
  $0 viser --robots g1 h2 --robot-motion g1:Form_1 --robot-motion h2:Form_2 --port 8080
  $0 rl --robot g1 --motion output_data/robot_motion/Form_1_stageii_g1.csv --rl-task unitree_g1_flat_tracking
  $0 rl --robot g1 --motion output_data/robot_motion/Form_1_stageii_g1.csv --export-only
EOF
}

# ══════════════════════════════════════════════════════════════════════════════
# 交互式模式配置
# ══════════════════════════════════════════════════════════════════════════════
select_mode() {
    local idx=$(prompt_select "请选择启动模式:" \
        "viser  — Viser 浏览器可视化" \
        "mujoco — MuJoCo 原生可视化" \
        "smpl   — SMPL-X → 机器人 重定向" \
        "robot  — 机器人 → 机器人 重定向" \
        "rl     — NPZ 导出与 RL 训练流水线" \
        "list   — 列出可用机器人和动作" \
        "doctor — 环境健康检查")
    case $idx in
        0) MODE="viser" ;;
        1) MODE="mujoco" ;;
        2) MODE="smpl" ;;
        3) MODE="robot" ;;
        4) MODE="rl" ;;
        5) MODE="list" ;;
        6) MODE="doctor" ;;
    esac
    log_info "已选择: $MODE"
}

# ── 扫描可用机器人 ─────────────────────────────────────────────────────────
# 排序规则与 viser 界面 _scan_all_robots 一致:
# 优先级机器人 (g1, g1_d, h1, h1_2, h2, unitree_a2, unitree_a2w) 在前，其余按字母序
scan_robots() {
    local config_dir="$PROJECT_DIR/config/robot"
    local all_robots=()
    [ -d "$config_dir" ] || return 0
    for f in "$config_dir"/*.yaml; do
        [ -f "$f" ] || continue
        all_robots+=("$(basename "$f" .yaml)")
    done
    # 优先级排序 (与 viser 下拉框一致)
    local priority=("g1" "g1_d" "h1" "h1_2" "h2" "unitree_a2" "unitree_a2w")
    local priority_found=()
    local rest=()
    for r in "${all_robots[@]}"; do
        local is_priority=false
        for p in "${priority[@]}"; do
            [ "$r" = "$p" ] && is_priority=true && break
        done
        if $is_priority; then
            priority_found+=("$r")
        else
            rest+=("$r")
        fi
    done
    # 按优先级顺序排列 + 其余字母排序
    SCAN_ROBOT_ARRAY=()
    for p in "${priority[@]}"; do
        for r in "${priority_found[@]}"; do
            [ "$r" = "$p" ] && SCAN_ROBOT_ARRAY+=("$r") && break
        done
    done
    IFS=$'\n' sorted_rest=($(printf '%s\n' "${rest[@]}" | sort)); unset IFS
    SCAN_ROBOT_ARRAY+=("${sorted_rest[@]}")
}

# ── 交互式选择机器人 (多选) ──────────────────────────────────────────────
select_robots_multi() {
    scan_robots
    if [ ${#SCAN_ROBOT_ARRAY[@]} -eq 0 ]; then
        log_error "未找到任何机器人配置"
        exit 1
    fi
    echo ""
    log_info "可用机器人:"
    for i in "${!SCAN_ROBOT_ARRAY[@]}"; do
        local label; label="$(get_robot_label_cn "${SCAN_ROBOT_ARRAY[$i]}")"
        echo -e "  ${CYAN}$((i+1))${NC}) ${label}"
    done

    echo ""
    echo -e "${BOLD}选择目标机器人 (输入编号，多个用空格分隔，回车=全部)${NC}"
    local choice
    read -p "选择: " choice

    SELECTED_ROBOTS=()
    if [ -z "$choice" ]; then
        SELECTED_ROBOTS=("${SCAN_ROBOT_ARRAY[@]}")
    else
        for num in $choice; do
            if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le "${#SCAN_ROBOT_ARRAY[@]}" ]; then
                SELECTED_ROBOTS+=("${SCAN_ROBOT_ARRAY[$((num-1))]}")
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

# ── 交互式选择机器人 + 为每个机器人指定动作 ──────────────────────────────
# 类似 trainBot 的 config_sim_robots: 先选机器人类型和数量,
# 再为每个机器人选择动作文件.
# 填充 SELECTED_ROBOTS 数组和 ROBOT_MOTIONS 关联数组.
# 参数: $1 = 模式 (smpl|robot|viser|mujoco), 决定动作文件类型
select_robots_with_motions() {
    local mode="${1:-smpl}"
    scan_robots
    if [ ${#SCAN_ROBOT_ARRAY[@]} -eq 0 ]; then
        log_error "未找到任何机器人配置"
        exit 1
    fi

    SELECTED_ROBOTS=()
    ROBOT_MOTIONS=()
    # 用于 viser/mujoco 查找 CSV 的真实机器人名 (去掉 __N 后缀)
    ROBOT_REAL_NAMES=()

    echo ""
    log_banner "── 添加机器人 ──"

    while true; do
        # 显示已添加的机器人
        if [ ${#SELECTED_ROBOTS[@]} -gt 0 ]; then
            echo ""
            log_info "已添加的机器人:"
            for i in "${!SELECTED_ROBOTS[@]}"; do
                local r="${SELECTED_ROBOTS[$i]}"
                local rlabel; rlabel="$(get_robot_label_cn "${ROBOT_REAL_NAMES[$r]:-$r}")"
                local m="${ROBOT_MOTIONS[$r]:-<未指定>}"
                echo -e "  ${GREEN}$((i+1)))${NC} ${CYAN}${rlabel}${NC}  动作: ${m}"
            done
            echo ""
        fi

        # 选择: 添加机器人 或 完成添加
        # 构建带中文名的选项列表 (使用 SCAN_ROBOT_ARRAY 保持与 viser 一致的排序)
        local opts=()
        for s in "${SCAN_ROBOT_ARRAY[@]}"; do
            opts+=("$(get_robot_label_cn "$s")")
        done
        opts+=("── ✅ 完成添加 ──")
        local ri
        ri=$(prompt_select "添加机器人类型 (或完成添加):" "${opts[@]}")
        if [ "$ri" -ge "${#SCAN_ROBOT_ARRAY[@]}" ]; then
            if [ ${#SELECTED_ROBOTS[@]} -eq 0 ]; then
                # viser/mujoco mode: allow empty start (add via browser)
                if [ "$mode" = "viser" ]; then
                    log_info "跳过选择 — 将通过浏览器添加机器人"
                    break
                fi
                log_warn "至少添加一个机器人"; continue
            fi
            break
        fi

        local rtype="${SCAN_ROBOT_ARRAY[$ri]}"

        # 数量
        local count
        count=$(prompt_input "  ${rtype} 数量" "1" true)
        [[ ! "$count" =~ ^[1-9][0-9]*$ ]] && { log_warn "无效, 默认为 1"; count=1; } || true

        # 为每个实例选择动作
        for n in $(seq 1 "$count"); do
            local instance_label="${rtype}"
            [ "$count" -gt 1 ] && instance_label="${rtype} #$n"

            echo ""
            log_info "为 ${instance_label} 选择动作文件:"

            local motion_path=""
            case "$mode" in
                smpl)
                    # SMPL-X 模式: 选择 .npz 文件
                    scan_smpl_motions
                    if [ ${#SCAN_SMPL_MOTION_ARRAY[@]} -gt 0 ]; then
                        local sorted_motions=($(printf '%s\n' "${SCAN_SMPL_MOTION_ARRAY[@]}" | sort))
                        local mi
                        mi=$(prompt_select "  选择 SMPL-X 动作:" "${sorted_motions[@]}")
                        motion_path="$PROJECT_DIR/dataset/ACCAD/${sorted_motions[$mi]}"
                        [ ! -f "$motion_path" ] && motion_path="$PROJECT_DIR/dataset/${sorted_motions[$mi]}"
                        [ ! -f "$motion_path" ] && motion_path="${sorted_motions[$mi]}"
                    else
                        motion_path=$(prompt_input "  SMPL-X 动作文件 (.npz 路径)" "" true)
                    fi
                    ;;
                robot)
                    # 机器人→机器人模式: 选择 .csv 文件
                    scan_robot_motions
                    if [ ${#SCAN_ROBOT_MOTION_ARRAY[@]} -gt 0 ]; then
                        local sorted_motions=($(printf '%s\n' "${SCAN_ROBOT_MOTION_ARRAY[@]}" | sort))
                        local mi
                        mi=$(prompt_select "  选择源机器人动作:" "${sorted_motions[@]}")
                        motion_path="$PROJECT_DIR/${sorted_motions[$mi]}"
                    else
                        motion_path=$(prompt_input "  源机器人动作文件 (.csv 路径)" "" true)
                    fi
                    ;;
                viser|mujoco)
                    # 可视化模式: 选择已有动作名称
                    scan_motions
                    if [ ${#SCAN_MOTION_ARRAY[@]} -gt 0 ]; then
                        local sorted_motions=($(printf '%s\n' "${SCAN_MOTION_ARRAY[@]}" | sort))
                        # 显示中文名 (与 viser 界面一致: 中文名 (英文名))
                        local display_opts=()
                        for m in "${sorted_motions[@]}"; do
                            display_opts+=("$(get_motion_label_cn "$m")")
                        done
                        local mi
                        mi=$(prompt_select "  选择动作:" "${display_opts[@]}")
                        motion_path="${sorted_motions[$mi]}"
                    else
                        motion_path=$(prompt_input "  动作名称" "" true)
                    fi
                    ;;
            esac

            # 添加到列表: 始终使用唯一 key (带 __N 后缀避免覆盖)
            local idx=${#SELECTED_ROBOTS[@]}
            local robot_key="${rtype}__$((idx+1))"
            SELECTED_ROBOTS+=("$robot_key")
            ROBOT_MOTIONS["$robot_key"]="$motion_path"
            ROBOT_REAL_NAMES["$robot_key"]="$rtype"
            log_info "已添加: ${instance_label} → ${motion_path}"
        done
    done

    log_info "机器人配置完成: ${SELECTED_ROBOTS[*]}"
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

    # 选择机器人 + 为每个机器人指定动作
    select_robots_with_motions smpl

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

    # 源机器人
    echo ""
    ORIGIN_ROBOT=$(prompt_input "源机器人名称" "g1")

    # 选择目标机器人 + 为每个机器人指定动作
    select_robots_with_motions robot

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

    # 机器人和动作通过浏览器 GUI 动态添加，跳过交互式选择

    # 端口
    VISER_PORT=$(prompt_input "Viser 端口" "20006")

    # 循环
    if [ "$(prompt_yn "循环播放?" "y")" = "true" ]; then
        LOOP="true"
    fi

    # 地面
    if [ "$(prompt_yn "显示地面?" "y")" = "false" ]; then
        NO_GROUND="true"
    fi
}

# ── mujoco 模式交互配置 ───────────────────────────────────────────────────
config_mujoco() {
    echo ""
    log_banner "── MuJoCo 原生可视化配置 ──"
    echo ""

    # 选择机器人 + 为每个机器人指定动作
    select_robots_with_motions mujoco

    # 帧率
    echo ""
    SOURCE_FPS=$(prompt_input "源数据帧率" "30")
    RENDER_FPS=$(prompt_input "渲染帧率" "60")

    # 循环
    if [ "$(prompt_yn "循环播放?" "y")" = "true" ]; then
        LOOP="true"
    fi
}

# ── rl 模式交互配置 ───────────────────────────────────────────────────────
config_rl() {
    echo ""
    log_banner "── RL 训练流水线配置 ──"
    echo ""

    # 机器人 (使用 SCAN_ROBOT_ARRAY 保持与 viser 一致的排序)
    scan_robots
    local robot_opts=()
    for s in "${SCAN_ROBOT_ARRAY[@]}"; do
        robot_opts+=("$(get_robot_label_cn "$s")")
    done
    local ri
    ri=$(prompt_select "选择机器人:" "${robot_opts[@]}")
    RL_ROBOT="${SCAN_ROBOT_ARRAY[$ri]}"

    # 输入 CSV
    echo ""
    log_info "选择输入动作文件:"
    scan_robot_motions
    if [ ${#SCAN_ROBOT_MOTION_ARRAY[@]} -gt 0 ]; then
        local sorted_motions=($(printf '%s\n' "${SCAN_ROBOT_MOTION_ARRAY[@]}" | sort))
        local mi
        mi=$(prompt_select "  选择动作 CSV:" "${sorted_motions[@]}")
        RL_CSV="$PROJECT_DIR/${sorted_motions[$mi]}"
    else
        RL_CSV=$(prompt_input "  动作 CSV 路径" "" true)
    fi

    # 帧率
    echo ""
    RL_INPUT_FPS=$(prompt_input "输入帧率" "30")
    RL_OUTPUT_FPS=$(prompt_input "输出帧率 (NPZ)" "50")

    # 是否仅导出
    if [ "$(prompt_yn "仅导出 NPZ (不启动训练)?" "n")" = "true" ]; then
        RL_EXPORT_ONLY="true"
    else
        RL_EXPORT_ONLY="false"
        # RL 任务
        local default_task="unitree_${RL_ROBOT}_flat_tracking"
        RL_TASK=$(prompt_input "RL 任务 ID" "$default_task")
        # RL 根目录
        RL_ROOT=$(prompt_input "unitree_rl_mjlab 路径" "$PROJECT_DIR/../unitree_rl_mjlab")
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
# 命令构建与执行
# ══════════════════════════════════════════════════════════════════════════════
build_and_run() {
    case "$MODE" in
        smpl)
            local robot_list="${SELECTED_ROBOTS[*]:-$VIS_ROBOTS}"
            local num_robots
            read -r -a robot_arr <<< "$robot_list"
            num_robots=${#robot_arr[@]}

            echo ""
            log_banner "══════════════════ SMPL-X → 机器人 ══════════════════"
            log_info "目标机器人: ${robot_arr[*]}"
            log_info "源帧率: $SOURCE_FPS, 渲染帧率: $RENDER_FPS"
            echo ""

            for idx in "${!robot_arr[@]}"; do
                local robot="${robot_arr[$idx]}"
                local real_name="${ROBOT_REAL_NAMES[$robot]:-$robot}"
                local robot_config="config/robot/${real_name}.yaml"
                local this_motion="${ROBOT_MOTIONS[$robot]:-$MOTION_FILE}"

                if [ ! -f "$PROJECT_DIR/$robot_config" ]; then
                    log_error "机器人配置不存在: $robot_config"
                    exit 1
                fi
                if [ ! -f "$this_motion" ]; then
                    log_error "动作文件不存在: $this_motion"
                    exit 1
                fi

                echo ""
                log_info "═══ [机器人 $((idx+1))/${num_robots}] ${real_name} ═══"
                log_info "  动作: $this_motion"

                # Step 1: smpl_replay
                echo "[1/3] smpl_replay (提取关键点)"
                "$PYTHON_BIN" scripts/smpl_replay.py \
                    --no-viewer \
                    --motion_file "$this_motion" \
                    --robot-config "$robot_config" \
                    --skeleton-config config/skeleton/skeleton.yaml \
                    --fps "$SOURCE_FPS"

                # Step 2: robot_retarget
                echo "[2/3] robot_retarget (逆运动学)"
                local retarget_args=(--config "$robot_config")
                local keypoints_name
                keypoints_name="$(basename "${this_motion}" .npz)"
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
            # 使用第一个机器人的动作名作为默认
            local first_robot="${robot_arr[0]}"
            local first_motion="${ROBOT_MOTIONS[$first_robot]:-$MOTION_FILE}"
            local motion_name
            motion_name="$(basename "${first_motion}" .npz)"

            # 构建真实机器人名数组
            local real_robot_arr=()
            for robot in "${robot_arr[@]}"; do
                real_robot_arr+=("${ROBOT_REAL_NAMES[$robot]:-$robot}")
            done

            local viser_args=(
                --motion "$motion_name"
                --robots "${real_robot_arr[@]}"
                --source_fps "$SOURCE_FPS"
                --render_fps "$RENDER_FPS"
                --port "$VISER_PORT"
            )
            [ "$LOOP" = "true" ] && viser_args+=(--loop)
            [ "$NO_GROUND" = "true" ] && viser_args+=(--no-ground)

            log_info "启动 Viser 可视化..."
            echo ""
            exec "$PYTHON_BIN" scripts/multi_robot_visualize_viser.py "${viser_args[@]}"
            ;;

        robot)
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
            log_info "源机器人: $ORIGIN_ROBOT"
            log_info "目标机器人: ${robot_arr[*]}"
            log_info "源帧率: $SOURCE_FPS, 渲染帧率: $RENDER_FPS"
            echo ""

            for idx in "${!robot_arr[@]}"; do
                local robot="${robot_arr[$idx]}"
                local real_name="${ROBOT_REAL_NAMES[$robot]:-$robot}"
                local robot_config="config/robot/${real_name}.yaml"
                local this_motion="${ROBOT_MOTIONS[$robot]:-$MOTION_FILE}"

                if [ ! -f "$PROJECT_DIR/$robot_config" ]; then
                    log_error "机器人配置不存在: $robot_config"
                    exit 1
                fi
                if [ ! -f "$this_motion" ]; then
                    log_error "动作文件不存在: $this_motion"
                    exit 1
                fi

                echo ""
                log_info "═══ [机器人 $((idx+1))/${num_robots}] ${real_name} ═══"
                log_info "  动作: $this_motion"

                # Step 1: robot_replay
                echo "[1/3] robot_replay (提取关键点)"
                "$PYTHON_BIN" scripts/robot_replay.py \
                    --no-viewer \
                    --motion-file "$this_motion" \
                    --source-robot-config "$source_config" \
                    --target-robot-config "$robot_config" \
                    --fps "$SOURCE_FPS"

                # Step 2: robot_retarget
                echo "[2/3] robot_retarget (逆运动学)"
                local retarget_args=(--config "$robot_config")
                local keypoints_name
                keypoints_name="$(basename "${this_motion}" .csv)"
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
            local first_robot="${robot_arr[0]}"
            local first_motion="${ROBOT_MOTIONS[$first_robot]:-$MOTION_FILE}"
            local motion_name
            motion_name="$(basename "${first_motion}" .csv)_from_${ORIGIN_ROBOT}"

            # 构建真实机器人名数组
            local real_robot_arr=()
            for robot in "${robot_arr[@]}"; do
                real_robot_arr+=("${ROBOT_REAL_NAMES[$robot]:-$robot}")
            done

            local viser_args=(
                --motion "$motion_name"
                --robots "${real_robot_arr[@]}"
                --source_fps "$SOURCE_FPS"
                --render_fps "$RENDER_FPS"
                --port "$VISER_PORT"
            )
            [ "$LOOP" = "true" ] && viser_args+=(--loop)
            [ "$NO_GROUND" = "true" ] && viser_args+=(--no-ground)

            log_info "启动 Viser 可视化..."
            echo ""
            exec "$PYTHON_BIN" scripts/multi_robot_visualize_viser.py "${viser_args[@]}"
            ;;

        viser)
            # Viser 模式: 启动空场景，通过浏览器 GUI 动态添加机器人和动作
            echo ""
            log_banner "══════════════════ Viser 可视化 ══════════════════"
            log_info "端口: $VISER_PORT"
            log_info "启动空场景后，请在浏览器中添加机器人和动作"
            echo ""

            local viser_args=(
                --data_dirs "$PROJECT_DIR/output_data/robot_motion" "$PROJECT_DIR/dataset/lafan1_g1" "$PROJECT_DIR/dataset/bones_g1" "$PROJECT_DIR/dataset/bones_g1_origin"
                --port "$VISER_PORT"
            )
            [ "$LOOP" = "true" ] && viser_args+=(--loop)
            [ "$NO_GROUND" = "true" ] && viser_args+=(--no-ground)

            exec "$PYTHON_BIN" scripts/multi_robot_visualize_viser.py "${viser_args[@]}"
            ;;

        mujoco)
            local robot_list="${SELECTED_ROBOTS[*]:-$VIS_ROBOTS}"
            read -r -a robot_arr <<< "$robot_list"

            # 查找每个机器人的动作 CSV 文件 (支持多种命名格式)
            # 并创建临时目录，将 CSV 链接为 Python 脚本期望的 {motion}_{robot}.csv 格式
            mkdir -p "${PROJECT_DIR}/.tmp"
            local tmp_motion_dir
            tmp_motion_dir=$(mktemp -d "${PROJECT_DIR}/.tmp/.tmp_motion_XXXXXX")
            local found_all=true
            for robot in "${robot_arr[@]}"; do
                local this_motion="${ROBOT_MOTIONS[$robot]:-$MOTION_FILE}"
                local real_name="${ROBOT_REAL_NAMES[$robot]:-$robot}"
                local csv_full_path=""
                _find_csv() {
                    local dir="$1" motion="$2" rname="$3"
                    # 格式1: {motion}_{robot}.csv (精确匹配)
                    [ -f "${dir}/${motion}_${rname}.csv" ] && { echo "${dir}/${motion}_${rname}.csv"; return 0; }
                    # 格式2: {motion}_subject*.csv (lafan1_g1 格式，要求不含 _from_ 后缀)
                    local f
                    for f in "${dir}/${motion}_subject"*.csv; do
                        [ -f "$f" ] || continue
                        local base
                        base="$(basename "$f" .csv)"
                        [[ "$base" == *"_from_"* ]] && continue
                        echo "$f"
                        return 0
                    done
                    # 格式3: {motion}_M.csv (bones_g1 格式，隐含 g1)
                    [ -f "${dir}/${motion}_M.csv" ] && { echo "${dir}/${motion}_M.csv"; return 0; }
                    # 格式4: {motion}_{robot}_*.csv (如 dance1_subject2_from_g1_h2)
                    for f in "${dir}/${motion}_${rname}_"*.csv; do
                        [ -f "$f" ] && { echo "$f"; return 0; }
                    done
                    # 格式5: {motion}_M*.csv (bones_g1 通用)
                    for f in "${dir}/${motion}_M"*.csv; do
                        [ -f "$f" ] && { echo "$f"; return 0; }
                    done
                    return 1
                }
                for _dir in "$PROJECT_DIR/output_data/robot_motion" \
                            "$PROJECT_DIR/dataset/lafan1_g1" \
                            "$PROJECT_DIR/dataset/bones_g1" \
                            "$PROJECT_DIR/dataset/bones_g1_origin"; do
                    [ -d "$_dir" ] || continue
                    if _result=$(_find_csv "$_dir" "$this_motion" "$real_name"); then
                        csv_full_path="$_result"
                        break
                    fi
                done
                if [ -z "$csv_full_path" ]; then
                    log_warn "未找到 ${this_motion} 对应的 CSV 文件"
                    found_all=false
                else
                    log_info "找到 ${real_name} 动作: $(basename "$csv_full_path")"
                    # Use robot key (with __N suffix) in filename to ensure uniqueness
                    ln -sf "$csv_full_path" "${tmp_motion_dir}/${this_motion}_${robot}.csv"
                fi
            done

            if [ "$found_all" = false ]; then
                rm -rf "$tmp_motion_dir"
                log_error "部分机器人动作文件缺失，请先运行 smpl/robot 模式生成"
                exit 1
            fi

            echo ""
            log_banner "══════════════════ MuJoCo 原生可视化 ══════════════════"
            log_info "动作名称: $MOTION_FILE"
            log_info "机器人: ${robot_arr[*]}"
            log_info "源帧率: $SOURCE_FPS, 渲染帧率: $RENDER_FPS"
            echo ""

            # 传机器人 key (带 __N 后缀) 给 Python 脚本，确保同名机器人有唯一标识
            local robot_key_arr=()
            for robot in "${robot_arr[@]}"; do
                robot_key_arr+=("$robot")
            done

            # 构建 per-robot motion 参数
            local robot_motion_args=()
            local has_custom_motion=false
            for robot in "${robot_arr[@]}"; do
                local this_motion="${ROBOT_MOTIONS[$robot]:-}"
                if [ -n "$this_motion" ] && [ "$this_motion" != "$MOTION_FILE" ]; then
                    robot_motion_args+=(--robot-motion "${robot}:${this_motion}")
                    has_custom_motion=true
                fi
            done

            local mujoco_args=(
                --motion "${MOTION_FILE:-${ROBOT_MOTIONS[${robot_arr[0]}]:-}}"
                --robots "${robot_key_arr[@]}"
                --motion_dir "$tmp_motion_dir"
                --source_fps "$SOURCE_FPS"
                --render_fps "$RENDER_FPS"
            )
            [ "$has_custom_motion" = "true" ] && mujoco_args+=("${robot_motion_args[@]}")
            [ "$LOOP" = "true" ] && mujoco_args+=(--loop)

            exec "$PYTHON_BIN" scripts/multi_robot_visualize.py "${mujoco_args[@]}"
            ;;

        rl)
            # RL 训练流水线: CSV → NPZ → train
            local rl_robot="${RL_ROBOT:-}"
            local rl_csv="${RL_CSV:-$MOTION_FILE}"
            local rl_input_fps="${RL_INPUT_FPS:-30}"
            local rl_output_fps="${RL_OUTPUT_FPS:-50}"
            local rl_task="${RL_TASK:-}"
            local rl_root="${RL_ROOT:-$PROJECT_DIR/../unitree_rl_mjlab}"
            local rl_export_only="${RL_EXPORT_ONLY:-false}"

            if [ -z "$rl_robot" ]; then
                log_error "rl 模式需要 --robot 参数"
                exit 1
            fi
            if [ -z "$rl_csv" ]; then
                log_error "rl 模式需要 --motion 参数 (CSV 文件路径)"
                exit 1
            fi

            # 动作名称 (用于 NPZ 文件命名)
            local motion_stem
            motion_stem="$(basename "$rl_csv" .csv)"

            echo ""
            log_banner "══════════════════ RL 训练流水线 ══════════════════"
            log_info "机器人: $rl_robot"
            log_info "输入 CSV: $rl_csv"
            log_info "帧率: ${rl_input_fps} fps → ${rl_output_fps} fps (NPZ)"
            if [ "$rl_export_only" = "true" ]; then
                log_info "模式: 仅导出 NPZ"
            else
                log_info "RL 任务: $rl_task"
                log_info "RL 根目录: $rl_root"
            fi
            echo ""

            # 构建 train_pipeline.py 参数
            local pipeline_args=(
                --robot "$rl_robot"
                --motion-name "$motion_stem"
                --csv "$rl_csv"
                --input-fps "$rl_input_fps"
                --output-fps "$rl_output_fps"
            )
            if [ "$rl_export_only" = "true" ]; then
                pipeline_args+=(--export-only)
            else
                pipeline_args+=(--rl-task "$rl_task" --rl-root "$rl_root")
                [ -n "${RL_TRAIN_ARGS:-}" ] && pipeline_args+=(--train-args "$RL_TRAIN_ARGS")
            fi

            exec "$PYTHON_BIN" scripts/train_pipeline.py "${pipeline_args[@]}"
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
            echo -e "  目标机器人: ${BOLD}${SELECTED_ROBOTS[*]}${NC}"
            echo -e "  机器人动作映射:"
            for robot in "${SELECTED_ROBOTS[@]:-}"; do
                local m="${ROBOT_MOTIONS[$robot]:-<未指定>}"
                echo -e "    ${GREEN}${robot}${NC} → ${m}"
            done
            echo -e "  帧率:       ${BOLD}源${SOURCE_FPS} / 渲染${RENDER_FPS}${NC}"
            echo -e "  调试渲染:   ${BOLD}${RENDER_DEBUG}${NC}"
            ;;
        robot)
            echo -e "  模式:       ${BOLD}robot${NC} (机器人 → 机器人)"
            echo -e "  源机器人:   ${BOLD}${ORIGIN_ROBOT}${NC}"
            echo -e "  目标机器人: ${BOLD}${SELECTED_ROBOTS[*]}${NC}"
            echo -e "  机器人动作映射:"
            for robot in "${SELECTED_ROBOTS[@]:-}"; do
                local m="${ROBOT_MOTIONS[$robot]:-<未指定>}"
                echo -e "    ${GREEN}${robot}${NC} → ${m}"
            done
            echo -e "  帧率:       ${BOLD}源${SOURCE_FPS} / 渲染${RENDER_FPS}${NC}"
            echo -e "  调试渲染:   ${BOLD}${RENDER_DEBUG}${NC}"
            ;;
        viser)
            echo -e "  模式:       ${BOLD}viser${NC} (浏览器可视化)"
            echo -e "  机器人:     ${BOLD}通过浏览器动态添加${NC}"
            echo -e "  端口:       ${BOLD}${VISER_PORT}${NC}"
            echo -e "  循环:       ${BOLD}${LOOP}${NC}"
            echo -e "  地面:       ${BOLD}$([ "$NO_GROUND" = "true" ] && echo "隐藏" || echo "显示")${NC}"
            ;;
        mujoco)
            echo -e "  模式:       ${BOLD}mujoco${NC} (MuJoCo 原生可视化)"
            echo -e "  机器人:     ${BOLD}${SELECTED_ROBOTS[*]}${NC}"
            echo -e "  机器人动作映射:"
            for robot in "${SELECTED_ROBOTS[@]:-}"; do
                local m="${ROBOT_MOTIONS[$robot]:-<未指定>}"
                echo -e "    ${GREEN}${robot}${NC} → ${m}"
            done
            echo -e "  帧率:       ${BOLD}源${SOURCE_FPS} / 渲染${RENDER_FPS}${NC}"
            echo -e "  循环:       ${BOLD}${LOOP}${NC}"
            ;;
        rl)
            echo -e "  模式:       ${BOLD}rl${NC} (RL 训练流水线)"
            echo -e "  机器人:     ${BOLD}${RL_ROBOT:-}${NC}"
            echo -e "  输入 CSV:   ${BOLD}${RL_CSV:-$MOTION_FILE}${NC}"
            echo -e "  帧率:       ${BOLD}${RL_INPUT_FPS:-30} fps → ${RL_OUTPUT_FPS:-50} fps${NC}"
            if [ "${RL_EXPORT_ONLY:-false}" = "true" ]; then
                echo -e "  操作:       ${BOLD}仅导出 NPZ${NC}"
            else
                echo -e "  RL 任务:    ${BOLD}${RL_TASK:-}${NC}"
                echo -e "  RL 根目录:  ${BOLD}${RL_ROOT:-}${NC}"
            fi
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
        mujoco)  config_mujoco ;;
        rl)      config_rl ;;
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
        smpl|robot|viser|mujoco|rl|list|doctor) MODE="$1"; shift ;;
        -h|--help|help) show_usage; exit 0 ;;
        *) log_error "未知模式: $1"; show_usage; exit 1 ;;
    esac

    # 解析参数
    VIS_ROBOTS=""
    SELECTED_ROBOTS=()
    ROBOT_MOTIONS=()
    MOTION_FILE=""
    while [ $# -gt 0 ]; do
        case "$1" in
            # 通用
            --motion)       MOTION_FILE="$2"; shift 2 ;;
            --robots)
                # 支持 --robots g1 h2 t800 或 --robots "g1 h2 t800"
                shift
                while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                    # 去重: 避免与 --robot-motion 自动添加的机器人重复
                    _r_dup=false
                    for _r_existing in "${SELECTED_ROBOTS[@]:-}"; do
                        [ "$_r_existing" = "$1" ] && _r_dup=true && break
                    done
                    [ "$_r_dup" = false ] && SELECTED_ROBOTS+=("$1")
                    shift
                done
                ;;
            --robot-motion)
                # 格式: --robot-motion <robot>:<motion_file>
                # 可多次使用, 为不同机器人指定不同动作
                # 例如: --robot-motion g1:dataset/ACCAD/Form_1.npz --robot-motion h2:dataset/ACCAD/Form_2.npz
                rm_spec="$2"; shift 2
                rm_robot="${rm_spec%%:*}"
                rm_motion="${rm_spec#*:}"
                ROBOT_MOTIONS["$rm_robot"]="$rm_motion"
                # 如果机器人不在 SELECTED_ROBOTS 中, 自动添加
                found=false
                for r in "${SELECTED_ROBOTS[@]:-}"; do
                    [ "$r" = "$rm_robot" ] && found=true && break
                done
                [ "$found" = false ] && SELECTED_ROBOTS+=("$rm_robot")
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
            # rl
            --robot)        RL_ROBOT="$2"; shift 2 ;;
            --rl-task)       RL_TASK="$2"; shift 2 ;;
            --rl-root)       RL_ROOT="$2"; shift 2 ;;
            --input-fps)     RL_INPUT_FPS="$2"; shift 2 ;;
            --output-fps)    RL_OUTPUT_FPS="$2"; shift 2 ;;
            --export-only)   RL_EXPORT_ONLY="true"; shift 1 ;;
            --train-args)    RL_TRAIN_ARGS="$2"; shift 2 ;;
            # help
            -h|--help|help) show_usage; exit 0 ;;
            *)
                log_warn "忽略未知参数: $1"; shift 1 ;;
        esac
    done

    # 校验必需参数
    case "$MODE" in
        smpl|robot)
            # 如果没有 --motion, 检查是否所有机器人都有 --robot-motion
            if [ -z "$MOTION_FILE" ]; then
                if [ ${#ROBOT_MOTIONS[@]} -eq 0 ]; then
                    log_error "$MODE 模式需要 --motion 参数 或 --robot-motion 参数"
                    show_usage; exit 1
                fi
                # 使用第一个 --robot-motion 作为默认 MOTION_FILE (用于可视化)
                for robot in "${!ROBOT_MOTIONS[@]}"; do
                    MOTION_FILE="${ROBOT_MOTIONS[$robot]}"
                    break
                done
            fi
            if [ ${#SELECTED_ROBOTS[@]} -eq 0 ]; then
                log_error "$MODE 模式需要 --robots 参数"
                show_usage; exit 1
            fi
            # 为没有 --robot-motion 的机器人填充默认 MOTION_FILE
            for robot in "${SELECTED_ROBOTS[@]}"; do
                if [ -z "${ROBOT_MOTIONS[$robot]:-}" ]; then
                    if [ -n "$MOTION_FILE" ]; then
                        ROBOT_MOTIONS[$robot]="$MOTION_FILE"
                    else
                        log_error "机器人 $robot 没有指定动作 (使用 --robot-motion $robot:<motion>)"
                        exit 1
                    fi
                fi
            done
            ;;
        viser|mujoco)
            # 如果没有 --motion, 检查是否所有机器人都有 --robot-motion
            if [ -z "$MOTION_FILE" ]; then
                if [ ${#ROBOT_MOTIONS[@]} -eq 0 ]; then
                    log_error "$MODE 模式需要 --motion 参数 (动作名称) 或 --robot-motion 参数"
                    show_usage; exit 1
                fi
                # 使用第一个 --robot-motion 作为默认 MOTION_FILE
                for robot in "${!ROBOT_MOTIONS[@]}"; do
                    MOTION_FILE="${ROBOT_MOTIONS[$robot]}"
                    break
                done
            fi
            if [ ${#SELECTED_ROBOTS[@]} -eq 0 ]; then
                log_error "$MODE 模式需要 --robots 参数"
                show_usage; exit 1
            fi
            # 为没有 --robot-motion 的机器人填充默认 MOTION_FILE
            for robot in "${SELECTED_ROBOTS[@]}"; do
                if [ -z "${ROBOT_MOTIONS[$robot]:-}" ]; then
                    if [ -n "$MOTION_FILE" ]; then
                        ROBOT_MOTIONS[$robot]="$MOTION_FILE"
                    else
                        log_error "机器人 $robot 没有指定动作 (使用 --robot-motion $robot:<motion>)"
                        exit 1
                    fi
                fi
            done
            ;;
        rl)
            if [ -z "${RL_ROBOT:-}" ]; then
                log_error "rl 模式需要 --robot 参数"
                show_usage; exit 1
            fi
            if [ -z "${MOTION_FILE:-}" ]; then
                log_error "rl 模式需要 --motion 参数 (CSV 文件路径)"
                show_usage; exit 1
            fi
            ;;
    esac

    confirm_config
    build_and_run
fi
