import numpy as np
import sys
import os

# 添加PythonRobotics库路径
current_dir = os.path.dirname(os.path.abspath(__file__))
pythonrobotics_path = os.path.join(current_dir, '..', 'PythonRobotics', 'PathPlanning', 'DynamicWindowApproach')
sys.path.insert(0, pythonrobotics_path)

try:
    # 动态导入PythonRobotics模块
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "dynamic_window_approach",
        os.path.join(pythonrobotics_path, "dynamic_window_approach.py")
    )
    if spec is not None and spec.loader is not None:
        dynamic_window_approach = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dynamic_window_approach)

        dwa_control = dynamic_window_approach.dwa_control
        Config = dynamic_window_approach.Config
        RobotType = dynamic_window_approach.RobotType
        PYTHONROBOTICS_AVAILABLE = True
    else:
        raise ImportError("无法加载PythonRobotics模块")
except Exception as e:
    print(f"警告: 无法导入PythonRobotics DWA算法: {e}")
    PYTHONROBOTICS_AVAILABLE = False


class DWAPlanner:
    def __init__(self):
        if not PYTHONROBOTICS_AVAILABLE:
            raise ImportError("PythonRobotics DWA算法不可用")
        self.config = Config()
        self.config.robot_type = RobotType.circle  # 默认圆形机器人

    def grid_to_obstacles(self, occupancy_grid):
        """
        将栅格地图转换为障碍物点列表

        参数:
        - occupancy_grid: 2D numpy数组，0为自由，1为障碍

        返回:
        - obstacles: numpy数组，形状为(n, 2)，每行为(x, y)坐标
        """
        h, w = occupancy_grid.shape
        map_size_m = 5.0  # 地图实际大小（米）
        resolution = map_size_m / max(h, w)

        # 提取障碍物坐标
        obstacle_indices = np.where(occupancy_grid == 1)
        if len(obstacle_indices[0]) == 0:
            return np.array([]).reshape(0, 2)

        # 转换为实际坐标
        ox = obstacle_indices[1] * resolution
        oy = obstacle_indices[0] * resolution

        # 组合为障碍物数组
        obstacles = np.column_stack((ox, oy))
        return obstacles

    def plan(self, current_state, current_velocity, goal, occupancy_grid):
        """
        DWA路径规划主函数

        参数:
        - current_state: [x, y, theta] 当前状态
        - current_velocity: [v, omega] 当前速度
        - goal: [x, y] 目标位置
        - occupancy_grid: 占用栅格地图

        返回:
        - v: 线速度
        - omega: 角速度
        """
        x = current_state + current_velocity  # [x, y, yaw, v, omega]
        goal_array = np.array(goal)
        obstacles = self.grid_to_obstacles(occupancy_grid)
        if len(obstacles) == 0:
            obstacles = np.array([[current_state[0] + 10, current_state[1] + 10]])
        u, _ = dwa_control(x, self.config, goal_array, obstacles)
        return u[0], u[1]
