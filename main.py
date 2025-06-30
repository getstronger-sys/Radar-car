#!/usr/bin/env python3
"""
自主导航机器人主程序
实现：SLAM建图、边界探索、出口检测、返回起点、实时可视化
"""

import time
import numpy as np
import sys
import os
from scipy.ndimage import binary_dilation

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import *
from planner.path_planner import plan_path
from planner.dwa_planner import DWAPlanner
from slam.mapper import SLAMProcessor
from exploration.frontier_detect import detect_frontiers
from viz.roboviz_slam_viz import RoboVizSLAMViewer
from comm.bluetooth import BluetoothCommMock
from logs.data_logger import DataLogger

class AutonomousRobot:
    """自主导航机器人主控制器"""
    
    def __init__(self):
        """初始化机器人系统"""
        # 初始化核心模块
        self.slam = SLAMProcessor()
        self.dwa = DWAPlanner(DWA_CONFIG)
        self.viz = RoboVizSLAMViewer(title='自主导航机器人 - SLAM建图与导航')
        self.comm = BluetoothCommMock()
        self.logger = DataLogger()
        
        # 系统状态
        self.current_pose = [START_POSITION['x'], START_POSITION['y'], START_POSITION['theta']]
        self.start_pose = self.current_pose.copy()
        self.trajectory = [self.current_pose.copy()]
        self.global_path = []
        self.path_index = 0
        self.start_time = time.time()  # 添加start_time属性
        
        # 任务状态
        self.exploration_mode = True  # 探索模式
        self.exit_found = False       # 是否找到出口
        self.returning_home = False   # 是否正在返回起点
        self.mission_complete = False # 任务是否完成
        
        # 出口检测
        self.exit_candidates = []     # 出口候选点
        self.confirmed_exit = None    # 确认的出口点
        
        print("🤖 自主导航机器人系统初始化完成")
        print(f"   起始位置: ({self.start_pose[0]:.2f}, {self.start_pose[1]:.2f}, {np.degrees(self.start_pose[2]):.1f}°)")
        print(f"   目标出口: ({EXIT_POSITION['x']:.2f}, {EXIT_POSITION['y']:.2f})")
        print(f"   地图尺寸: {MAP_SIZE}x{MAP_SIZE} 米")
        print(f"   地图分辨率: {MAP_RESOLUTION} 米/格子")
    
    def inflate_grid(self, occupancy_grid):
        """对障碍物做膨胀处理"""
        dilation_radius = int(np.ceil(ROBOT_RADIUS / MAP_RESOLUTION))
        structure = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1))
        inflated = binary_dilation(occupancy_grid, structure=structure).astype(np.uint8)
        return inflated
    
    def detect_exit(self, occupancy_grid):
        """检测出口点"""
        # 简单的出口检测：寻找地图边缘的自由空间
        height, width = occupancy_grid.shape
        
        # 检查地图边缘
        edge_points = []
        for i in range(height):
            for j in range(width):
                if (i < 2 or i >= height-2 or j < 2 or j >= width-2) and occupancy_grid[i, j] == 0:
                    # 转换为世界坐标
                    world_x = (j - width//2) * MAP_RESOLUTION
                    world_y = (height//2 - i) * MAP_RESOLUTION
                    edge_points.append((world_x, world_y))
        
        # 选择最接近目标出口的点
        if edge_points:
            closest_exit = min(edge_points, 
                             key=lambda p: np.hypot(p[0] - EXIT_POSITION['x'], p[1] - EXIT_POSITION['y']))
            
            # 检查是否足够接近目标出口
            dist_to_target = np.hypot(closest_exit[0] - EXIT_POSITION['x'], 
                                     closest_exit[1] - EXIT_POSITION['y'])
            
            if dist_to_target < 1.0:  # 1米范围内认为是出口
                if not self.exit_found:
                    print(f"🎯 发现出口点: ({closest_exit[0]:.2f}, {closest_exit[1]:.2f})")
                    self.exit_found = True
                    self.confirmed_exit = closest_exit
                    # 使用正确的日志方法
                    self.logger.log_performance_metric("EXIT_DETECTED", {
                        'exit_position': closest_exit,
                        'distance_to_target': dist_to_target
                    })
        
        return self.confirmed_exit
    
    def plan_exploration_path(self, occupancy_grid, frontiers):
        """规划探索路径"""
        if not frontiers:
            return None
        
        # 选择最近的前沿点
        closest_frontier = min(frontiers, 
                             key=lambda f: np.hypot(f[0] - self.current_pose[0], f[1] - self.current_pose[1]))
        
        # 规划到前沿的路径
        start = {'x': self.current_pose[0], 'y': self.current_pose[1]}
        goal = {'x': closest_frontier[0], 'y': closest_frontier[1]}
        
        inflated_grid = self.inflate_grid(occupancy_grid)
        path = plan_path(inflated_grid, start, goal)
        
        if path:
            print(f"🎯 规划到前沿点 ({closest_frontier[0]:.2f}, {closest_frontier[1]:.2f})")
            # 使用正确的日志方法
            self.logger.log_path_planning(
                [self.current_pose[0], self.current_pose[1]], 
                [closest_frontier[0], closest_frontier[1]], 
                path, 
                "A*"
            )
        
        return path
    
    def plan_return_path(self, occupancy_grid):
        """规划返回起点的路径"""
        if not self.confirmed_exit:
            print("❌ 未找到出口，无法规划返回路径")
            return None
        
        # 从当前位置到出口，再到起点
        start = {'x': self.current_pose[0], 'y': self.current_pose[1]}
        exit_point = {'x': self.confirmed_exit[0], 'y': self.confirmed_exit[1]}
        home_point = {'x': self.start_pose[0], 'y': self.start_pose[1]}
        
        inflated_grid = self.inflate_grid(occupancy_grid)
        
        # 先到出口
        path_to_exit = plan_path(inflated_grid, start, exit_point)
        if not path_to_exit:
            print("❌ 无法规划到出口的路径")
            return None
        
        # 再从出口到起点
        path_to_home = plan_path(inflated_grid, exit_point, home_point)
        if not path_to_home:
            print("❌ 无法规划从出口到起点的路径")
            return None
        
        # 合并路径（去掉重复的出口点）
        full_path = path_to_exit + path_to_home[1:]
        
        print(f"🔄 规划返回路径: 当前位置 -> 出口 -> 起点")
        print(f"   路径长度: {len(full_path)}")
        # 使用正确的日志方法
        self.logger.log_path_planning(
            [self.current_pose[0], self.current_pose[1]], 
            [self.start_pose[0], self.start_pose[1]], 
            full_path, 
            "A*"
        )
        
        return full_path
    
    def execute_navigation(self, target_point):
        """执行导航到目标点"""
        # 使用DWA进行局部路径规划
        state = [self.current_pose[0], self.current_pose[1], self.current_pose[2]]
        velocity = [0.0, 0.0]  # 当前速度
        
        v, omega = self.dwa.plan(state, velocity, target_point, self.inflate_grid(self.slam.get_occupancy_grid()))
        
        if v is not None and omega is not None:
            # 发送控制命令给机器人
            target_pose = {'x': target_point[0], 'y': target_point[1], 'theta': 0.0}
            self.comm.send_target(target_pose)
            
            # 模拟机器人运动
            self.comm.move_along_path()
            
            # 记录控制命令
            self.logger.log_control_command(v, omega, target_point)
            
            return True
        else:
            print("⚠️  DWA规划失败，无法到达目标点")
            return False
    
    def update_visualization(self, map_bytes, lidar_scan):
        """更新可视化"""
        # 准备导航路径
        nav_path = None
        if self.returning_home and self.global_path:
            nav_path = self.global_path
        elif self.global_path:
            nav_path = self.global_path
        
        # 准备前沿点
        frontiers = None
        if self.exploration_mode and not self.exit_found:
            try:
                frontiers = detect_frontiers(self.slam.get_occupancy_grid(), map_resolution=MAP_RESOLUTION)
            except Exception as e:
                print(f"前沿检测错误: {e}")
                frontiers = []
        
        # 准备当前目标点
        current_goal = None
        if self.global_path and self.path_index < len(self.global_path):
            current_goal = self.global_path[self.path_index]
        
        # 准备状态信息
        status_info = {
            'Exploration': 'Active' if self.exploration_mode else 'Complete',
            'Exit Found': 'Yes' if self.exit_found else 'No',
            'Returning': 'Yes' if self.returning_home else 'No',
            'Path Progress': f"{self.path_index}/{len(self.global_path)}" if self.global_path else "0/0",
            'Frontiers': len(frontiers) if frontiers else 0,
            'Trajectory Points': len(self.trajectory),
            'Time': f"{time.time() - self.start_time:.1f}s"
        }
        
        # 更新PyRoboViz显示
        self.viz.update(
            map_bytes=map_bytes,
            pose=self.current_pose,
            lidar_scan=lidar_scan,
            trajectory=self.trajectory,
            nav_path=nav_path,
            frontiers=frontiers,
            current_goal=current_goal,
            status_info=status_info
        )
    
    def run(self):
        """主控循环：前沿检测+A*+DWA集成"""
        print("🚀 启动主控循环...")
        dwa_planner = self.dwa
        robot_state = self.current_pose  # [x, y, theta]
        robot_velocity = [0.0, 0.0]
        occupancy_grid = np.zeros((100, 100), dtype=np.uint8)  # 示例地图，可替换为SLAM地图
        MAP_RESOLUTION = 0.1
        max_steps = 1000
        trajectory = [robot_state[:]]

        for step in range(max_steps):
            # 1. 前沿检测
            frontiers = detect_frontiers(occupancy_grid, map_resolution=MAP_RESOLUTION)
            if not frontiers:
                print("探索完成，无前沿点！")
                break
            # 2. 选择最近前沿点
            target_frontier = min(frontiers, key=lambda f: np.hypot(f[0]-robot_state[0], f[1]-robot_state[1]))
            # 3. 全局路径规划
            path = plan_path(occupancy_grid, {'x': robot_state[0], 'y': robot_state[1]}, {'x': target_frontier[0], 'y': target_frontier[1]})
            if not path:
                print("A*未找到路径，跳过本轮")
                continue
            # 4. 取全局路径上的下一个点作为DWA的局部目标
            local_goal = path[min(5, len(path)-1)]
            # 5. DWA局部避障
            v, omega = dwa_planner.plan(robot_state, robot_velocity, local_goal, occupancy_grid)
            # 6. 机器人运动学更新
            dt = 0.1
            robot_state[0] += v * np.cos(robot_state[2]) * dt
            robot_state[1] += v * np.sin(robot_state[2]) * dt
            robot_state[2] += omega * dt
            robot_velocity = [v, omega]
            trajectory.append(robot_state[:])
            # 7. 可视化
            if step % 5 == 0:
                import matplotlib.pyplot as plt
                plt.clf()
                plt.imshow(occupancy_grid, cmap='gray_r', origin='lower')
                traj_np = np.array(trajectory)
                plt.plot(traj_np[:,0]/MAP_RESOLUTION, traj_np[:,1]/MAP_RESOLUTION, 'b-')
                plt.plot(target_frontier[0]/MAP_RESOLUTION, target_frontier[1]/MAP_RESOLUTION, 'ro')
                plt.pause(0.01)
            # 8. 判断是否到达目标前沿
            if np.hypot(robot_state[0]-target_frontier[0], robot_state[1]-target_frontier[1]) < 0.2:
                print(f"到达前沿点 {target_frontier}")
                # 可在此处将该前沿点标记为已探索
        print("主控循环结束")

def main():
    """主函数"""
    robot = AutonomousRobot()
    robot.run()

if __name__ == "__main__":
    main()