# PythonProject1: 自主探索与路径规划仿真系统

本项目集成了移动机器人自主探索、SLAM地图构建、前沿检测、A*+DWA路径规划、仿真与可视化、数据导出等功能，适合算法学习、仿真验证和数据分析。

---

## 项目简介

本项目主要功能包括：
- **SLAM地图构建与仿真**（simulate_exploration.py）
- **前沿检测自主探索**（exploration/frontier_detect.py）
- **A*全局路径规划与DWA动力学局部避障**（planner/）
- **仿真与可视化**（matplotlib动画、地图图片导出等）
- **数据导出**（轨迹、地图、激光数据等多种格式）
- **全局起点终点统一配置**

---

## 目录结构

```
PythonProject1/
├── main.py                        # 主流程入口（自动先仿真再路径规划）
├── main_pipeline.py               # 路径规划与可视化流程
├── simulate_exploration.py        # 地图探索仿真与数据导出
├── exploration_trajectory.txt     # 导出的轨迹点（文本）
├── exploration_final_map.txt      # 导出的最终已知地图（文本）
├── exploration_lidar.json         # 导出的激光数据（含时间戳、位姿）
├── exploration_final_map.png      # 最终地图图片
├── exploration_trajectory.npy     # 轨迹点（npy二进制）
├── exploration_final_map.npy      # 地图（npy二进制）
├── requirements.txt               # 依赖包列表
├── config/
│   ├── settings.py                # 全局参数、起点终点配置
│   └── map.py                     # 地图障碍物线段配置
├── exploration/
│   └── frontier_detect.py         # 前沿检测与探索算法
├── planner/
│   ├── path_planner.py            # A*路径规划
│   ├── dwa_planner.py             # DWA动力学局部规划
│   ├── visualize_path.py          # 路径可视化
│   └── visualize_dwa_path.py      # DWA仿真可视化
├── logs/                          # 日志模块
├── viz/                           # 可视化模块
└── PythonRobotics/                # 第三方路径规划算法库
```

---

## 快速上手

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 一键运行主流程
```bash
python main.py
```
- 会自动先运行地图探索仿真（simulate_exploration.py），再运行路径规划与可视化（main_pipeline.py）。
- 生成的轨迹、地图、激光数据等会自动保存在项目根目录。

### 3. 数据导出与可视化
- **轨迹**：exploration_trajectory.txt（文本）、exploration_trajectory.npy（二进制）
- **地图**：exploration_final_map.txt（文本）、exploration_final_map.npy（二进制）、exploration_final_map.png（图片）
- **激光数据**：exploration_lidar.json（含每步时间戳、位姿、scan）
- **轨迹线段json**：trajectory_segments.json（与SEGMENTS格式一致）

#### 可视化脚本示例
可用matplotlib等工具直接可视化txt/map/png/json等数据，或自定义脚本叠加轨迹、障碍物等。

---

## 全局起点和终点设置

- **起点**：config/settings.py 中的 START_POSITION
- **终点**：config/settings.py 中的 EXIT_POSITION
- 所有仿真、路径规划、可视化均自动使用全局设置。

---

## 主要功能模块说明

- **地图探索仿真**：simulate_exploration.py，自动探索未知地图，导出轨迹、地图、激光数据。
- **前沿检测**：exploration/frontier_detect.py，自动检测未知区域边界。
- **路径规划与可视化**：main_pipeline.py、planner/，支持A*、DWA等多种算法。
- **数据导出**：支持txt、npy、json、png等多种格式，便于分析和二次开发。

---

## 常见问题
- **.npy文件不可直接用文本编辑器打开**，请用numpy加载。
- **.txt/.json/.png文件可直接查看或用Python/Excel等工具处理。**
- **如需自定义导出格式或可视化脚本，请参考simulate_exploration.py或联系维护者。**

---

## 贡献与许可
- 欢迎提交Issue和PR改进项目。
- 本项目采用MIT许可证，详见LICENSE文件。

---

**注意：本项目仅供学习和研究使用，实际部署前请充分测试。**
