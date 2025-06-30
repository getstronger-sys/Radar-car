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
    def __init__(self, config):
        """
        初始化DWA规划器
        
        参数:
        - config: 配置字典，包含机器人参数
        """
        if PYTHONROBOTICS_AVAILABLE:
            # 创建PythonRobotics的Config对象
            self.config = Config()
            
            # 更新配置参数
            self.config.max_speed = config.get('max_speed', 1.0)
            self.config.min_speed = config.get('min_speed', 0.0)
            self.config.max_yaw_rate = config.get('max_yawrate', 1.0)
            self.config.max_accel = config.get('max_accel', 0.5)
            self.config.max_delta_yaw_rate = config.get('max_dyawrate', 1.0)
            self.config.v_resolution = config.get('v_reso', 0.05)
            self.config.yaw_rate_resolution = config.get('yawrate_reso', 0.1)
            self.config.dt = config.get('dt', 0.1)
            self.config.predict_time = config.get('predict_time', 2.0)
            self.config.to_goal_cost_gain = config.get('to_goal_cost_gain', 1.0)
            self.config.speed_cost_gain = config.get('speed_cost_gain', 1.0)
            self.config.obstacle_cost_gain = config.get('obstacle_cost_gain', 1.0)
            self.config.robot_radius = config.get('robot_radius', 0.05)
            self.config.robot_type = RobotType.circle
        else:
            # 使用简单配置
            self.robot_radius = config.get('robot_radius', 0.05)
            self.map_resolution = config.get('map_resolution', 0.1)
            self.max_speed = config.get('max_speed', 1.0)
            self.min_speed = config.get('min_speed', 0.0)
            self.max_yaw_rate = config.get('max_yawrate', 1.0)
            self.max_accel = config.get('max_accel', 0.5)
            self.max_delta_yaw_rate = config.get('max_dyawrate', 1.0)
            self.dt = config.get('dt', 0.1)
            self.predict_time = config.get('predict_time', 2.0)
            self.v_resolution = config.get('v_reso', 0.05)
            self.yaw_rate_resolution = config.get('yawrate_reso', 0.1)
            self.to_goal_cost_gain = config.get('to_goal_cost_gain', 1.0)
            self.speed_cost_gain = config.get('speed_cost_gain', 1.0)
            self.obstacle_cost_gain = config.get('obstacle_cost_gain', 1.0)

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
        if PYTHONROBOTICS_AVAILABLE:
            return self.plan_pythonrobotics(current_state, current_velocity, goal, occupancy_grid)
        else:
            return self.plan_simple(current_state, current_velocity, goal, occupancy_grid)

    def plan_pythonrobotics(self, current_state, current_velocity, goal, occupancy_grid):
        """使用PythonRobotics库的DWA算法"""
        try:
            # 转换状态格式为PythonRobotics格式 [x, y, theta, v, omega]
            x = current_state + current_velocity
            
            # 转换目标格式
            goal_array = np.array(goal)
            
            # 转换障碍物格式
            obstacles = self.grid_to_obstacles(occupancy_grid)
            
            # 如果没有障碍物，创建一个远离机器人的虚拟障碍物
            if len(obstacles) == 0:
                obstacles = np.array([[current_state[0] + 10, current_state[1] + 10]])
            
            # 调用PythonRobotics的DWA算法
            u, trajectory = dwa_control(x, self.config, goal_array, obstacles)
            
            return u[0], u[1]  # 返回线速度和角速度
            
        except Exception as e:
            print(f"PythonRobotics DWA规划错误: {e}")
            return 0.0, 0.0

    def plan_simple(self, current_state, current_velocity, goal, occupancy_grid):
        """简单的DWA实现（备用方案）"""
        from scipy.ndimage import binary_dilation
        
        def inflate_obstacles(occupancy_grid):
            """膨胀障碍物，考虑机器人半径"""
            dilation_radius = int(np.ceil(self.robot_radius / self.map_resolution))
            structure = np.ones((2*dilation_radius+1, 2*dilation_radius+1))
            inflated = binary_dilation(occupancy_grid, structure=structure).astype(np.uint8)
            return inflated

        def motion(state, control, dt):
            """机器人运动模型"""
            x, y, theta = state
            v, omega = control
            
            x_new = x + v * np.cos(theta) * dt
            y_new = y + v * np.sin(theta) * dt
            theta_new = theta + omega * dt
            
            return [x_new, y_new, theta_new]

        def calc_trajectory(init_state, v, omega):
            """计算预测轨迹"""
            traj = [init_state.copy()]
            state = init_state.copy()
            time = 0.0
            
            while time <= self.predict_time:
                state = motion(state, [v, omega], self.dt)
                traj.append(state.copy())
                time += self.dt
                
            return traj

        def calc_obstacle_cost(traj, inflated_map):
            """计算障碍物代价"""
            h, w = inflated_map.shape
            
            for x, y, _ in traj:
                gx = int(x / self.map_resolution)
                gy = int(y / self.map_resolution)
                
                if gx < 0 or gx >= w or gy < 0 or gy >= h:
                    return float('inf')
                
                if inflated_map[gy, gx] == 1:
                    return float('inf')
            
            return 0.0

        def calc_goal_cost(traj, goal):
            """计算目标代价"""
            x, y, _ = traj[-1]
            return np.hypot(goal[0] - x, goal[1] - y)

        def calc_speed_cost(v):
            """计算速度代价"""
            return self.max_speed - v

        def calc_dynamic_window(current_velocity):
            """计算动态窗口"""
            v_min = max(self.min_speed, current_velocity[0] - self.max_accel * self.dt)
            v_max = min(self.max_speed, current_velocity[0] + self.max_accel * self.dt)
            
            omega_min = max(-self.max_yaw_rate, current_velocity[1] - self.max_delta_yaw_rate * self.dt)
            omega_max = min(self.max_yaw_rate, current_velocity[1] + self.max_delta_yaw_rate * self.dt)
            
            return v_min, v_max, omega_min, omega_max

        # 主规划逻辑
        best_score = float('inf')
        best_control = [0.0, 0.0]

        # 计算动态窗口
        v_min, v_max, omega_min, omega_max = calc_dynamic_window(current_velocity)
        
        print(f"动态窗口: v=[{v_min:.3f}, {v_max:.3f}], ω=[{omega_min:.3f}, {omega_max:.3f}]")
        
        # 确保速度范围有效
        if v_min > v_max:
            v_min, v_max = v_max, v_min
        if omega_min > omega_max:
            omega_min, omega_max = omega_max, omega_min

        # 生成速度采样
        v_samples = np.arange(v_min, v_max + self.v_resolution, self.v_resolution)
        omega_samples = np.arange(omega_min, omega_max + self.yaw_rate_resolution, self.yaw_rate_resolution)
        
        print(f"速度采样: {v_samples}")
        print(f"角速度采样: {omega_samples}")
        print(f"总采样数: {len(v_samples) * len(omega_samples)}")

        # 膨胀障碍物
        inflated_map = inflate_obstacles(occupancy_grid)

        valid_trajectories = 0
        for v in v_samples:
            for omega in omega_samples:
                # 计算预测轨迹
                traj = calc_trajectory(current_state, v, omega)
                
                # 计算各项代价
                obs_cost = calc_obstacle_cost(traj, inflated_map)
                if obs_cost == float('inf'):
                    continue
                
                valid_trajectories += 1
                goal_cost = calc_goal_cost(traj, goal)
                speed_cost = calc_speed_cost(v)
                
                # 计算总代价
                total_cost = (self.to_goal_cost_gain * goal_cost +
                             self.obstacle_cost_gain * obs_cost +
                             self.speed_cost_gain * speed_cost)
                
                # 更新最优控制
                if total_cost < best_score:
                    best_score = total_cost
                    best_control = [v, omega]
        
        print(f"有效轨迹数: {valid_trajectories}")
        if valid_trajectories == 0:
            print("警告：所有轨迹都被拒绝")

        return best_control
