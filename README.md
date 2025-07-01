# 自主导航机器人系统

这是一个基于Python的自主导航机器人系统，实现了SLAM地图构建、前沿检测探索、路径规划（A*+DWA）和实时导航等功能，支持动力学仿真与可视化。

---

## 项目简介

本项目集成了移动机器人自主导航的核心算法与工具，包括：
- **SLAM地图构建**
- **前沿检测自主探索**
- **A*全局路径规划**
- **DWA动力学局部避障**
- **可视化与数据记录**
- **蓝牙通信与硬件接口**

适用于机器人算法学习、仿真验证和软硬件结合开发。

---

## 如何获取和配置本项目

### 1. 克隆项目到本地

如果你已有GitHub账号，推荐直接克隆：

```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
```

或者下载ZIP包并解压到本地。

### 2. 配置Python环境和依赖库

建议使用 Python 3.7+，推荐先安装 [Anaconda](https://www.anaconda.com/) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。

#### 使用 requirements.txt 一键安装依赖：

```bash
pip install -r requirements.txt
```

#### 或手动安装主要依赖：

```bash
pip install numpy scipy matplotlib breezyslam roboviz pyserial
```

如需使用 `PythonRobotics` 路径规划算法，请确保 `PythonRobotics` 目录已包含在本项目中。

---

## 如何提交本地修改到 GitHub

1. **在本地修改代码后，依次执行：**

```bash
git status           # 查看修改了哪些文件
git add .            # 添加所有更改（或指定文件）
git commit -m "你的修改说明"
git push             # 推送到GitHub远程仓库
```

2. **如果是第一次推送，需先关联远程仓库：**

```bash
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin master
```

3. **如遇 remote 已存在，可先移除再添加：**

```bash
git remote remove origin
git remote add origin https://github.com/你的用户名/你的仓库名.git
```

---

## 目录结构

```
PythonProject1/
├── main.py                      # 主控制程序
├── README.md                    # 项目说明文档
├── requirements.txt             # 依赖包列表
├── config/
│   └── settings.py              # 系统配置参数
├── slam/
│   └── mapper.py                # SLAM地图构建模块
├── exploration/
│   └── frontier_detect.py       # 前沿检测算法
├── planner/
│   ├── path_planner.py          # A*全局路径规划
│   ├── dwa_planner.py           # DWA动力学局部规划
│   ├── visualize_path.py        # 路径规划可视化
│   └── visualize_dwa_path.py    # DWA动力学仿真与可视化
├── test_dwa_visualization.py    # DWA可视化测试脚本
├── viz/
│   └── map_viz.py               # 地图与轨迹可视化
├── comm/
│   └── bluetooth.py             # 蓝牙通信模块
├── logs/
│   └── data_logger.py           # 数据日志记录
└── PythonRobotics/              # 第三方路径规划算法库
```

---

## 快速上手

### 1. 运行主程序
```bash
python main.py
```

### 2. 路径规划与可视化
- **A* 路径规划可视化**
  ```bash
  python planner/visualize_path.py
  ```
- **DWA 动力学仿真与可视化**
  ```bash
  python planner/visualize_dwa_path.py
  ```
  - 支持地图、障碍物、全局路径、DWA轨迹、机器人朝向等可视化
  - 可在脚本内自定义起点、终点和障碍物分布

### 3. DWA可视化测试脚本
```bash
python test_dwa_visualization.py
```
- 自动构建简单地图并运行DWA仿真，适合快速验证环境

### 4. 配置参数
编辑 `config/settings.py` 文件，调整地图、机器人、DWA等参数。

---

## 主要功能模块说明

### 1. SLAM地图构建（`slam/mapper.py`）
- 基于BreezySLAM的实时地图构建与位姿估计
- 支持RPLIDAR等激光雷达数据

### 2. 前沿检测探索（`exploration/frontier_detect.py`）
- 自动检测未知区域边界，实现自主探索

### 3. 路径规划（`planner/`）
- **A*全局路径规划**：最优路径搜索
- **DWA动力学局部避障**：动态窗口法，考虑机器人动力学约束
- 支持路径平滑、障碍物膨胀

### 4. 可视化（`viz/`、`planner/visualize_*.py`）
- 实时地图、轨迹、路径、机器人状态可视化
- 支持静态图和动画
- 支持网页交互及其可视化(终端运行)
```bash
streamlit run planner/web_viz.py
``` 

### 5. 通信与数据记录
- 蓝牙串口通信（`comm/bluetooth.py`）
- 传感器与轨迹数据日志（`logs/data_logger.py`）

---

## 常见问题与调试

- **SLAM不收敛**：检查激光数据质量和里程计精度
- **路径规划失败**：确认地图正确性和起点终点可达性
- **DWA仿真卡住**：调整起点/终点、障碍物分布或DWA参数，确保A*能返回多点路径
- **通信中断**：检查蓝牙连接和串口配置
- **可视化卡顿**：降低地图分辨率或更新频率

---

## 贡献与许可

- 欢迎提交 Issue 和 Pull Request 改进项目。
- 本项目采用 MIT 许可证，详见 LICENSE 文件。

---

## 联系方式

- 提交 GitHub Issue
- 或发送邮件至项目维护者

---

**注意：本项目仅供学习和研究使用，实际部署前请充分测试。** 
