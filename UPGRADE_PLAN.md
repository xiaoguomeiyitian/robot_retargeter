# robot_retargeter 升级迭代优化方案

> 制定日期: 2026-06-27
> 当前版本: v2.0 — batched mesh 渲染 + 多机器人独立控制 (4.3ms/帧)
> 更新日期: 2026-06-27

## v2.0 更新日志

### ✅ P0-1.1: 批量场景更新 — 已完成
- 105 个 batched mesh groups (183 instances)
- apply_frame: 16.5ms → **4.3ms** (3.8x 加速)
- Max FPS: 60 → **229**
- CPU: 107% → **55%**

### ✅ P1-2.2: 多机器人独立播放 — 已完成
- 每个机器人独立 Play/Pause 按钮
- 每个机器人独立 Speed 滑块
- 工厂函数修复闭包变量捕获问题

### ✅ P1-2.4: 地面接触可视化 — 已完成
- 4 个接触力箭头 (左右脚/左右手)
- 修复 add_arrows API 格式 (N,2,3) points + (N,3) colors

### ✅ P0-2.1: 轨迹编辑器 — 已完成
- viser GUI 关键帧编辑
- 7 个基础关节角度滑块 (pos3 + quat4)
- 实时预览 + 导出 CSV

### ✅ P1-2.3: 视频导出 — 已完成
- MuJoCo Renderer 离屏渲染
- ffmpeg pipe 编码 MP4 (libx264)
- 可配置 FPS/分辨率

### 🔧 修复
- 修复 viser `add_arrows` API 参数格式
- 修复 per-robot 回调中闭包变量捕获 bug
- 修复 slider 轮询频率减少 WebSocket 开销

---

## 当前状态

- **核心流水线**: SMPL-X/源机器人 → 关键帧提取 → IK 重定向 → 可视化
- **已完成**: 环境安装、viser 浏览器可视化、多机器人网格布局、性能优化
- **硬件**: RTX 2080 8GB, 12核 CPU, 16GB RAM
- **当前瓶颈**: 已解决 — batched mesh 渲染 4.3ms/帧

---

## Phase 1: 可视化性能 (P0)

### 1.1 批量场景更新 (Batched Mesh Updates)
- **目标**: 将独立 handle 更新改为批量更新
- **方案**: 按 mesh 分组，使用 `server.scene.add_batched_meshes_trimesh()`
- **预期**: 16ms → <5ms
- **文件**: `scripts/multi_robot_visualize_viser.py`

### 1.2 线程化渲染循环
- **目标**: 物理仿真与 GUI 事件循环解耦
- **方案**: `threading.Thread` 处理物理，主线程处理 GUI
- **预期**: 消除 GUI 回调抖动
- **文件**: `scripts/multi_robot_visualize_viser.py`

### 1.3 自适应细节层次 (LOD)
- **目标**: 根据距离动态调整渲染精度
- **方案**: 远距离简化 mesh，近距离完整 mesh
- **预期**: 5+ 机器人场景帧率提升 30-50%
- **文件**: `scripts/multi_robot_visualize_viser.py`

---

## Phase 2: 功能增强 (P0-P1)

### 2.1 轨迹编辑器 ✅ 已完成
- **目标**: viser 中拖拽关键帧修改关节角度
- **功能**: 关键帧拖拽、关节角度滑块、实时预览、导出 CSV
- **文件**: `scripts/motion_editor.py`

### 2.2 多机器人独立播放
- **目标**: 每个机器人独立选择动作和控制播放
- **功能**: 独立播放/暂停/速度、动作列表选择、同步/异步切换
- **文件**: `scripts/multi_robot_visualize_viser.py`

### 2.3 视频导出 ✅ 已完成
- **目标**: 动画导出为 MP4
- **方案**: MuJoCo Renderer 离屏渲染 + ffmpeg pipe (libx264)
- **文件**: `scripts/export_video.py`

### 2.4 地面接触可视化
- **目标**: 脚底接触力可视化
- **方案**: 接触点添加力箭头或颜色变化
- **文件**: `scripts/multi_robot_visualize_viser.py`

---

## Phase 3: 流水线自动化 (P1)

### 3.1 一键交互式向导
- **目标**: start.sh 升级为交互式向导
- **功能**: 自动检测数据集、机器人多选菜单、进度条
- **文件**: `start.sh`

### 3.2 批量处理
- **目标**: 多动作 × 多机器人组合遍历
- **文件**: `bash/retarget_from_robot.sh`

### 3.3 配置验证
- **目标**: 启动时自动检查机器人配置完整性
- **文件**: `start.sh` doctor 模式

---

## Phase 4: 高级功能 (P2-P3)

### 4.1 MuJoCo 动力学仿真
- **目标**: 重力/碰撞/接触力反馈
- **文件**: 新建 `scripts/physics_simulate.py`

### 4.2 动作混合
- **目标**: 两个动作间平滑过渡 (slerp + lerp)
- **文件**: 新建 `scripts/motion_blend.py`

### 4.3 风格迁移
- **目标**: 将一种动作风格应用到另一种
- **依赖**: 额外训练数据

### 4.4 实时动作捕捉
- **目标**: 摄像头实时捕获 → 重定向
- **方案**: MediaPipe 集成

---

## Phase 5: 工程化 (持续)

### 5.1 单元测试
### 5.2 类型注解
### 5.3 logging 替换 print
### 5.4 国际化

---

## 优先级排序

| 优先级 | 功能 | 原因 |
|--------|------|------|
| P0 | 1.1 批量场景更新 | 直接解决最明显性能瓶颈 |
| P0 | 2.1 轨迹编辑器 | 核心功能，提升可用性 |
| P1 | 1.2 线程化渲染 | 消除 GUI 抖动 |
| P1 | 2.2 多机器人独立播放 | 用户已需要 |
| P1 | 3.1 一键流水线 | 降低使用门槛 |
| P2 | 2.3 视频导出 | 分享展示 |
| P2 | 3.2 批量处理 | 效率提升 |
| P2 | 4.2 动作混合 | 创意功能 |
| P3 | 4.1 物理仿真 | 需要大量调试 |
| P3 | 4.3 风格迁移 | 需要训练数据 |
| P3 | 4.4 实时捕捉 | 需要额外硬件 |

---

## 关键文件

| 文件 | 用途 |
|------|------|
| `scripts/multi_robot_visualize_viser.py` | 核心可视化脚本 |
| `scripts/robot_retarget.py` | IK 重定向核心 |
| `scripts/smpl_replay.py` | SMPL-X 处理 + 工具函数 |
| `scripts/robot_replay.py` | 源机器人动作回放 |
| `config/robot/*.yaml` | 21 个机器人配置 |
| `dataset/lafan1_g1/` | LAFAN1 动作数据 |
| `start.sh` | 交互式入口 |
| `bash/retarget_from_robot.sh` | 一键流水线 |

---

## 验证计划

1. **性能基准**: 每次改动前后对比 `apply_frame` 耗时（目标 <10ms）
2. **功能回归**: `--robots subject1 subject3 subject5 --loop` 正常
3. **多机器人测试**: 5+ 机器人同时播放不卡顿
4. **长时间运行**: 1 小时循环播放无内存泄漏
